import os
import argparse
import faiss
import tqdm
import logging
import numpy as np
from personabench.utils.shared_args import add_shared_arguments
from personabench.utils.utils import get_api_key, load_json, convert_json_to_plain_text, save_json, set_random_seed, setup_logger
from sentence_transformers import SentenceTransformer
from personabench.models.openai import OpenAIGPT


def create_vector_base(documents_dict, retriever_model=None):
    chunks = []
    segment_ids = []
    for document_type, document_data in documents_dict.items():
        if document_type != "conversation_data":
            for data_entry in document_data:
                segment_id = data_entry["segment_id"]
                chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                chunks.append(chunk)
                segment_ids.append(segment_id)
        else:
            # deal conversation data
            for sub_conversation_data in document_data:
                for data_entry in sub_conversation_data["Conversations"]:
                    segment_id = data_entry["segment_id"]
                    chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                    chunks.append(chunk)
                    segment_ids.append(segment_id)
    if retriever_model:
        embeddings = retriever_model.encode(chunks, convert_to_numpy=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    else:
        index = None
    return chunks, index, np.array(segment_ids)


def extract_gt_segment_ids(QA, qa_gt_context_all_lookup, num_chunks):
    gt_segment_ids_retrieval = []
    for qa in QA:
        q_id = qa["q_id"]
        question_gt_ids = []
        for individual_ids in qa_gt_context_all_lookup[q_id]["segment_id"].values():
            question_gt_ids.append(individual_ids[0])
        if len(question_gt_ids) < num_chunks:
            additional_count = num_chunks - len(question_gt_ids)
            # Repeat the list to fill up the required length
            repeat_part = question_gt_ids * (additional_count // len(question_gt_ids))
            remaining_part = question_gt_ids[:additional_count % len(question_gt_ids)]
            question_gt_ids += repeat_part + remaining_part
        elif len(question_gt_ids) > num_chunks:
            raise ValueError(f"find {len(question_gt_ids)} ground truth, which is larger than {num_chunks}")
        gt_segment_ids_retrieval.append(question_gt_ids)
    return np.array(gt_segment_ids_retrieval)


def retrieval_and_generation(args):
    base_models = [r.strip() for r in args.base_models.split(',')]
    retrievers = [r.strip() for r in args.retrievers.split(',')]
    community_ids = [r.strip() for r in args.test_community_ids.split(',')]
    noises = [r.strip() for r in args.test_noises.split(',')]
    test_result_dir = os.path.join(args.log_dir, args.data_dir.split("/")[-1])

    # loop over noise level, test all Q&A
    for noise in tqdm.tqdm(noises):
        logging.info(f"Evaluating with supported documents at noise {noise}.")
        results_all_community = []
        for retriever in retrievers:
            for model_name in base_models:
                if "gpt" in model_name.lower():
                    api_key = get_api_key(args)
                    model = OpenAIGPT(api_key, model=model_name)
                else:
                    raise ValueError(f"Model type {model_name} not implemented")
                # Initialize embedding model
                if retriever == "gt-context":  # use ground-truth retriving
                    retriever_model = None
                else:
                    retriever_model = SentenceTransformer(retriever, trust_remote_code=True)  # Example embedding model
                results_all_community.append({"Model": f"{model_name}+{retriever}", "Results": []})
                logging.info(f"evaluating {model_name}+{retriever}")
                save_path = os.path.join(test_result_dir, f"results_all_community_chunk_{args.num_chunks}")
                os.makedirs(save_path, exist_ok=True)
                # evaluate all communities
                for community_id in community_ids:
                    community_path = os.path.join(args.data_dir, "synthetic_data", community_id)
                    eval_info_all = load_json(os.path.join(community_path, 'eval_info', 'eval_info_all.json'))
                    qa_gt_context_all = load_json(os.path.join(community_path, 'eval_info', f'qa_gt_context_all_noise_{noise}.json'))
                    qa_gt_context_all_lookup = {entry["q_id"]: entry for entry in qa_gt_context_all}  # for retrieve by q_id
                    # load all documents
                    conversation_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "conversation_data_all.json"))
                    user_ai_interaction_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "user_ai_interaction_data_all.json"))
                    purchase_history_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "purchase_history_data_all.json"))
                    conversation_data_all_lookup = {entry["Name"]: entry["Data"] for entry in conversation_data_all}
                    user_ai_interaction_data_all_lookup = {entry["Name"]: entry["Data"] for entry in user_ai_interaction_data_all}
                    purchase_history_data_all_lookup = {entry["Name"]: entry["Data"] for entry in purchase_history_data_all}
                    # evaluate for each person
                    for eval_info_dict in eval_info_all[:args.num_people]:
                        name = eval_info_dict["Name"]
                        eval_info = eval_info_dict["Eval_Info"]
                        conversation_data = conversation_data_all_lookup[name]
                        user_ai_interaction_data = user_ai_interaction_data_all_lookup[name]
                        purchase_history_data = purchase_history_data_all_lookup[name]
                        documents_dict = {"conversation_data": conversation_data, "user_ai_interaction_data": user_ai_interaction_data, "purchase_history_data": purchase_history_data}
                        chunks, index, segment_ids = create_vector_base(documents_dict, retriever_model)
                        QA = eval_info["qa"]
                        questions = [entry['question'] for entry in QA]
                        gt_segment_ids_retrieval = extract_gt_segment_ids(QA, qa_gt_context_all_lookup, args.num_chunks)
                        if retriever == "gt-context":  # use ground-truth context
                            id_to_index = {id_: idx for idx, id_ in enumerate(segment_ids)}
                            I = np.vectorize(id_to_index.get)(gt_segment_ids_retrieval)
                        else:
                            q_embeddings = retriever_model.encode(questions, convert_to_numpy=True)
                            D, I = index.search(q_embeddings, args.num_chunks)
                        retrival_res = np.array(chunks)[I]  # (# questions, # retrieve)
                        logging.info("Retrieve finished")
                        prompt_template = (
                            "You are provided with the following relevant information: {doc}.\n"
                            "Answer the question below as directly and concisely as possible, using only the name(s) "
                            "of the relevant entity or entities. Avoid adding any extra words or explanations.\n\n"
                            "Question: {query}")
                        for i, q_dict in enumerate(QA):
                            # generate prediction for individual question
                            instruction = prompt_template.format(query=q_dict["question"], doc=retrival_res[i])
                            prompt = {"instruction": instruction}
                            prediction = model.generate(prompt)
                            results_all_community[-1]["Results"].append({"q_id": q_dict["q_id"],
                                                                         "question": q_dict["question"],
                                                                         "answer": q_dict["answer"],
                                                                         "prediction":prediction,
                                                                         "type": q_dict["type"],
                                                                         "difficulty": q_dict["difficulty"],
                                                                         "retrieved_segment_ids": list(segment_ids[I[i]]),
                                                                         "ground_truth_segment_ids": qa_gt_context_all_lookup[q_dict["q_id"]]["segment_id"]}
                                                                         )
        save_json(results_all_community, os.path.join(save_path, f"result_noise_{noise}.json"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_shared_arguments(parser)
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to save eval results.")
    parser.add_argument("--test_community_ids", type=str, default="community_0,community_1", 
                        help="Comma-separated list of community ids to test")
    parser.add_argument('--num_people', type=int, default=3, 
                        help="Number of individuals to test in each community")
    parser.add_argument("--base_models", type=str, default="gpt-3.5-turbo,gpt-4o", 
                        help="Comma-separated list of base models to evaluate")
    parser.add_argument("--retrievers", type=str, default="gt-context,all-MiniLM-L6-v2,BAAI/bge-m3,all-mpnet-base-v2", 
                        help="Comma-separated list of retriever models to evaluate, gt-context use ground truth source as context during evaluation")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks for retrieval tasks")
    parser.add_argument("--test_noises", type=str, default="0.0,0.3,0.5,0.7", 
                        help="Comma-separated list of noise levels for testing")
    args = parser.parse_args()
    setup_logger(args.verbose)
    set_random_seed(args.seed)
    retrieval_and_generation(args)