"""
Calulate statistic analysis for retrieval and e2e answers + visualization
"""
import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from itertools import product
from personabench.utils.shared_args import add_shared_arguments
from personabench.utils.utils import load_json, save_json


def clean_text(text):
    # Remove all non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip().lower()


def split_into_words(text):
    # Split cleaned text into individual words
    return set(clean_text(text).lower().split())


def compute_recall_and_ndcg_for_single_combination(retrieved, ground_truth_combination, k=5):
    """
    Compute Recall@K and NDCG@K for a single ground truth combination.
    """
    retrieved = retrieved[:k]
    ground_truth_set = set(ground_truth_combination)
    
    # Compute Recall@K
    num_relevant_retrieved = len(set(retrieved).intersection(ground_truth_set))
    recall = num_relevant_retrieved / len(ground_truth_set) if ground_truth_set else 0.0

    # Compute NDCG
    dcg = 0.0
    for i, segment_id in enumerate(retrieved):
        if segment_id in ground_truth_set:
            dcg += 1 / np.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(len(ground_truth_set), k)):
        idcg += 1 / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg


def expand_retrieval_ground_truth(retrieval_ground_truth):
    combinations = list(product(*retrieval_ground_truth.values()))
    return [list(combination) for combination in combinations]


def calculate_precision_recall_f1(gt_list, prediction_list):
    # Convert lists to sets to handle unique items and easy set operations
    gt_set = set(gt_list)
    prediction_set = set(prediction_list)
    
    true_positives = len(gt_set & prediction_set)
    false_positives = len(prediction_set - gt_set)
    false_negatives = len(gt_set - prediction_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def calculate_retrieval_scores(results):
    retrieval_scores = {"Overall": {"total_recall": 0.0, "total_ndcg": 0.0, "count": 0}}
    for result_entry in results:
        retrieved_segment_ids = result_entry["retrieved_segment_ids"]
        ground_truth_segment_ids = result_entry["ground_truth_segment_ids"]
        # ground_truth_segment_ids = result_entry["ground_truth_segment_ids"]["segment_id"]
        ground_truth_combinations = expand_retrieval_ground_truth(ground_truth_segment_ids)
        # Evaluate all combinations and take the max Recall and NDCG
        best_recall, best_ndcg = 0.0, 0.0
        for combination in ground_truth_combinations:
            recall, ndcg = compute_recall_and_ndcg_for_single_combination(retrieved_segment_ids, combination, k=len(retrieved_segment_ids))
            best_recall = max(best_recall, recall)
            best_ndcg = max(best_ndcg, ndcg)
        question_type = result_entry["type"]
        if question_type == 'Subjective':
            continue  # Subjective questions are not used for now
        if question_type == "Preference":
            difficulty = result_entry["difficulty"]
        else:
            difficulty = "easy"
        q_category = "_".join(question_type.split(" ") + difficulty.split(" "))
        if q_category not in retrieval_scores:
            retrieval_scores[q_category] = {"total_recall": 0.0, "total_ndcg": 0.0, "count": 0}
        retrieval_scores[q_category]["total_recall"] += best_recall
        retrieval_scores[q_category]["total_ndcg"] += best_ndcg
        retrieval_scores[q_category]["count"] += 1
        retrieval_scores["Overall"]["total_recall"] += best_recall
        retrieval_scores["Overall"]["total_ndcg"] += best_ndcg
        retrieval_scores["Overall"]["count"] += 1
    for _, scores in retrieval_scores.items():
        scores["ave_recall"] = round(float(scores["total_recall"] / scores["count"]), 3)
        scores["ave_ndcg"] = round(float(scores["total_ndcg"] / scores["count"]), 3)
        del scores["total_recall"]
        del scores["total_ndcg"]
        del scores["count"]
    return retrieval_scores


def calculate_qa_scores(results, outdated_values_dict):
    end2end_scores = {"Overall": {"total_precision": 0.0, "total_recall": 0.0, "total_f1": 0.0, "count": 0},
                      "outdated_information": {"total_recall": 0.0, "count": 0}}
    for result_entry in results:
        prediction = result_entry["prediction"]
        answer = result_entry["answer"]
        question_type = result_entry["type"]
        if question_type == 'Subjective':
            continue  # Subjective questions are not used for now
        if question_type == "Preference":
            difficulty = result_entry["difficulty"]
        else:
            difficulty = "easy"
        q_category = "_".join(question_type.split(" ") + difficulty.split(" "))
        # Convert ground truth to a set of individual words
        if isinstance(answer, str):
            gt_set = split_into_words(answer)
        elif isinstance(answer, list):
            gt_words = [split_into_words(str(word)) for word in answer]
            gt_set = set().union(*gt_words)
        elif isinstance(answer, int):
            pass
        else:
            raise ValueError(f"undefined type: ", answer)

        # Convert prediction to a set of individual words
        if isinstance(prediction, str):
            prediction_set = split_into_words(prediction)
        else:
            prediction_words = [split_into_words(str(word)) for word in prediction]
            prediction_set = set().union(*prediction_words)  # In case prediction is already in list form
        
        precision, recall, f1 = calculate_precision_recall_f1(gt_set, prediction_set)
        if q_category not in end2end_scores:
            end2end_scores[q_category] = {"total_precision": 0.0, "total_recall": 0.0, "total_f1": 0.0, "count": 0}
        end2end_scores[q_category]["total_precision"] += precision
        end2end_scores[q_category]["total_recall"] += recall
        end2end_scores[q_category]["total_f1"] += f1
        end2end_scores[q_category]["count"] += 1
        end2end_scores["Overall"]["total_precision"] += precision
        end2end_scores["Overall"]["total_recall"] += recall
        end2end_scores["Overall"]["total_f1"] += f1
        end2end_scores["Overall"]["count"] += 1
        # if outdated info exist
        
        outdated_values = outdated_values_dict[result_entry["q_id"]]
        if outdated_values is not None:
            if isinstance(outdated_values, str):
                    outdated_set = split_into_words(outdated_values)
            elif isinstance(outdated_values, list):
                outdated_words = [split_into_words(str(word)) for word in outdated_values]
                outdated_set = set().union(*outdated_words)
            elif isinstance(outdated_values, int):
                pass
            else:
                raise ValueError(f"undefined type: ", outdated_set)
            _, recall, _ = calculate_precision_recall_f1(outdated_set, prediction_set)
            end2end_scores["outdated_information"]["total_recall"] += recall
            end2end_scores["outdated_information"]["count"] += 1

    for category_name, scores in end2end_scores.items():
        if "outdated_information" in category_name.lower():
            scores["ave_recall"] = round(float(scores["total_recall"] / scores["count"]), 3)
            del scores["total_recall"]
            del scores["count"]
        else:
            scores["ave_precision"] = round(float(scores["total_precision"] / scores["count"]), 3)
            scores["ave_recall"] = round(float(scores["total_recall"] / scores["count"]), 3)
            scores["ave_f1"] = round(float(scores["total_f1"] / scores["count"]), 3)
            del scores["total_precision"]
            del scores["total_recall"]
            del scores["total_f1"]
            del scores["count"]
    return end2end_scores


def print_table_to_file(table_all_noise, result_dir):
    output_file_path = os.path.join(result_dir, "table.txt")
    
    with open(output_file_path, "w") as file:
        for results_under_noise in table_all_noise:
            noise = results_under_noise["Noise"]
            Statistics = results_under_noise["Statistics"]
            end2end_eval = Statistics["end2end_eval"]
            retrieval_eval = Statistics["retrieval_eval"]
            
            # Write retriever evaluation results
            for retriever_name, results_ in retrieval_eval.items():
                file.write("\n\n")
                file.write(f"==============  EVAL RETRIEVERS AT NOISE {noise} ===========================\n")
                file.write(f"----------- retriever model: {retriever_name} -----------------\n")
                file.write(f"Basic_information: {results_['Basic_information_easy']}\n")
                file.write(f"Preference_easy: {results_['Preference_easy']}\n")
                file.write(f"Preference_hard: {results_['Preference_hard']}\n")
                file.write(f"Social: {results_['Social_easy']}\n")
                file.write(f"Overall: {results_['Overall']}\n")
                file.write(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
            
            # Write RAG model evaluation results
            for model_name, results_ in end2end_eval.items():
                file.write("\n\n")
                file.write(f"==============  EVAL RAG AT NOISE {noise} ===========================\n")
                file.write(f"----------- RAG model: {model_name} -----------------\n")
                file.write(f"Basic_information: {results_['Basic_information_easy']}\n")
                file.write(f"Preference_easy: {results_['Preference_easy']}\n")
                file.write(f"Preference_hard: {results_['Preference_hard']}\n")
                file.write(f"Social: {results_['Social_easy']}\n")
                file.write(f"Overall: {results_['Overall']}\n")
                file.write(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    print(f"Numbers saved to {output_file_path}")
    return True


def plot_noise_level(table_all_noise, result_dir):
    noise_levels = []
    rag_f1_scores = {}
    retrieval_recalls = {}
    for results_under_noise in table_all_noise:
        noise = results_under_noise["Noise"]
        Statistics = results_under_noise["Statistics"]
        end2end_eval = Statistics["end2end_eval"]
        retrieval_eval = Statistics["retrieval_eval"]
        # For e2e
        for model_name, results_ in end2end_eval.items():
            if "gt" in model_name.lower():
                continue
            if model_name not in rag_f1_scores:
                rag_f1_scores[model_name] = []
            rag_f1_scores[model_name].append(results_["Overall"]["ave_recall"])
        # For retrieval
        for retriever_name, results_ in retrieval_eval.items():
            if "gt" in retriever_name.lower():
                continue
            if retriever_name not in retrieval_recalls:
                retrieval_recalls[retriever_name] = []
            retrieval_recalls[retriever_name].append(results_["Overall"]["ave_recall"])
        noise_levels.append(noise)
    
    # Plot Recall for RAG evaluations
    plt.figure(figsize=(17, 6))
    for model_name, f1_scores in rag_f1_scores.items():
        plt.plot(noise_levels, f1_scores, marker='o', label=f'{model_name}')
    plt.xlabel('Noise Level',fontsize=18, fontweight='bold')
    plt.ylabel('Recall',fontsize=18, fontweight='bold')
    # plt.title('E2E Evaluation: F1 Scores Across Noise Levels')
    plt.legend(fontsize=12)
    plt.legend(loc="upper right", bbox_to_anchor=(1.7,1.0))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "e2e.jpg"), dpi=300)

    # Plot Recall for Retrieval evaluations
    plt.figure(figsize=(10, 6))
    for retriever_name, recalls in retrieval_recalls.items():
        plt.plot(noise_levels, recalls, marker='x', linestyle='--', label=f'Retrieval Recall: {retriever_name}')
    plt.xlabel('Noise Level',fontsize=18, fontweight='bold')
    plt.ylabel('Recall',fontsize=18, fontweight='bold')
    # plt.title('Retrieval Evaluation: Recall Across Noise Levels')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "retrival.jpg"), dpi=300)
    return True


def calculate_radar_values(model_data, noise_min_f1, noise_max_f1):
        basic_info_understanding = model_data["Basic_information_easy"]["ave_recall"]
        preference_precision = np.mean([
            model_data["Preference_easy"]["ave_precision"],
            model_data["Preference_hard"]["ave_precision"]
        ])
        preference_completeness = np.mean([
            model_data["Preference_easy"]["ave_recall"],
            model_data["Preference_hard"]["ave_recall"]
        ])
        social_relation_understanding = model_data["Social_easy"]["ave_recall"]
        information_updation = 1 - model_data["outdated_information"]["ave_recall"]
        noise_robustness = 1 - (noise_min_f1 - noise_max_f1)
        
        return [
            noise_robustness,
            social_relation_understanding,
            basic_info_understanding,
            information_updation,
            preference_completeness,
            preference_precision
        ]


def extract_radar_data(data, min_noise, max_noise):
    raw_data = {}
    for entry in data:
        noise = entry["Noise"]
        models = entry["Statistics"]["end2end_eval"]
        if noise == min_noise:
            noise_min_f1 = {model: results["Overall"]["ave_f1"] for model, results in models.items() if "gt" not in model.lower()}
        elif noise == max_noise:
            noise_max_f1 = {model: results["Overall"]["ave_f1"] for model, results in models.items() if "gt" not in model.lower()}
    
    for model_name in noise_min_f1.keys():
        if "gt" in model_name.lower():
            continue
        raw_data[model_name] = calculate_radar_values(
            models[model_name], noise_min_f1[model_name], noise_max_f1[model_name]
        )
    return raw_data


def normalize_radar_data(raw_data):
    """z-score normalization"""
    all_values = np.array(list(raw_data.values()))
    all_values_normalized = zscore(all_values, axis=0)
    
    normalized_data = {}
    for idx, model_name in enumerate(raw_data.keys()):
        normalized_data[model_name] = all_values_normalized[idx].tolist()
    
    return normalized_data


def plot_radar_chart(table_all_noise, noises, result_dir, args):
    categories = [
        "Noise Robustness",
        "Social Relations",
        "Basic Info",
        "Info Updating",
        "Preference Completeness",
        "Preference Precision"
    ]
    num_vars = len(categories)
    raw_radar_data = extract_radar_data(table_all_noise, min(noises), max(noises))
    if args.no_radar_normalization:
        normalized_radar_data = raw_radar_data
    else:
        normalized_radar_data = normalize_radar_data(raw_radar_data)
    
    # Create radar chart for each model
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    for model_name, values in normalized_radar_data.items():
        if "gt" in model_name:
            continue
        if args.radar_model:
            if args.radar_model.lower() not in model_name.split("+"):
                continue
        values += values[:1]
        ax.plot(angles, values, label=model_name, marker='o')
        ax.fill(angles, values, alpha=0.2)
    ax.tick_params(axis='both', which='major', pad=40)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=15, fontweight='bold')
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.8, 1.2), fontsize=10)
    # Show plot
    plt.tight_layout()
    figure_name = f"radar_{args.radar_model}" if args.radar_model else "radar_all" 
    plt.savefig(os.path.join(result_dir, f"{figure_name}.jpg"), dpi=300)
    return True


def eval(args):
    # Set global font properties
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold'
    })
    dataset_name = args.data_dir.split('/')[-1]
    # get outdated info
    outdated_values_dict = {}  # {"qid": outdated_values}
    synthetic_data_dir = os.path.join(args.data_dir, "synthetic_data")
    community_paths = sorted([os.path.join(synthetic_data_dir, name) for name in os.listdir(synthetic_data_dir) 
        if os.path.isdir(os.path.join(synthetic_data_dir, name))])
    for community_path in community_paths:
        eval_info_all = load_json(os.path.join(community_path, "eval_info/eval_info_all.json"))
        for eval_info_dict in eval_info_all:
            for qa_entry in eval_info_dict["Eval_Info"]["qa"]:
                if qa_entry["q_id"] in outdated_values_dict:
                    raise ValueError(f"repeat question id: {qa_entry['q_id']}")
                outdated_values_dict[qa_entry["q_id"]] = qa_entry["outdated_value"]

    result_dirs = sorted(os.path.join(args.log_dir, dataset_name, name) for name in os.listdir(os.path.join(args.log_dir, dataset_name)))
    # evaluate over all RAG settings (# chunks), save seperately
    for result_dir in result_dirs:
        result_paths = sorted([os.path.join(result_dir, name) for name in os.listdir(result_dir) if "result" in name.lower()])
        # loop over results under all noise levels
        table_all_noise = []
        noises = []
        for result_path in result_paths:
            match = re.search(r'noise_(\d+\.\d+)', result_path.split("/")[-1])
            noise = float(match.group(1))
            noises.append(noise)
            results_under_noise = load_json(result_path)
            statistic_table = {"end2end_eval": {}, "retrieval_eval": {}}
            for result_dict in results_under_noise:
                model_name = result_dict["Model"]
                base_model_name, retriever_name = model_name.split("+")
                results = result_dict["Results"]
                if retriever_name != "gt-context" and retriever_name not in statistic_table["retrieval_eval"]:
                    retrieval_scores = calculate_retrieval_scores(results)
                    statistic_table["retrieval_eval"][retriever_name] = retrieval_scores
                end2end_eval = calculate_qa_scores(results, outdated_values_dict)
                assert model_name not in statistic_table["end2end_eval"]
                statistic_table["end2end_eval"][model_name] = end2end_eval
            table_all_noise.append({"Noise": noise, "Statistics": statistic_table})
        save_json(table_all_noise, os.path.join(result_dir, "statistics_all_noises.json"))
    print_table_to_file(table_all_noise, result_dir)
    plot_noise_level(table_all_noise, result_dir)
    plot_radar_chart(table_all_noise, noises, result_dir, args)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_shared_arguments(parser)
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to the saved eval results.")
    parser.add_argument("--no_radar_normalization", action="store_true", help="Whether normalize results before drawing radar chart.")
    parser.add_argument("--radar_model", type=str)
    args = parser.parse_args()
    eval(args)