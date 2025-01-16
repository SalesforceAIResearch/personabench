python scripts/evaluation/retrieval_and_generation.py --seed 2024 \
    --log_dir logs \
    --data_dir eval_data/eval_data_v1 \
    --test_community_ids community_0,community_1 \
    --num_chunks 5 \
    --base_models gpt-4o-mini,gpt-4-0613,gpt-3.5-turbo,gpt-4o \
    --retrievers all-mpnet-base-v2,gt-context,all-MiniLM-L6-v2,BAAI/bge-m3 \
    --test_noises 0.0,0.3,0.5,0.7 \
    --verbose