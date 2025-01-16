#!/bin/bash

SEED=2024
LOG_DIR="logs"
DATA_DIR="eval_data/eval_data_v1"
TEST_COMMUNITY_IDS="community_0,community_1"
NUM_CHUNKS=5
BASE_MODELS="gpt-4o-mini,gpt-4-0613,gpt-3.5-turbo,gpt-4o"
RETRIEVERS="all-mpnet-base-v2,gt-context,all-MiniLM-L6-v2,BAAI/bge-m3"
TEST_NOISES="0.0,0.3,0.5,0.7"
VERBOSE="--verbose"

# Run retrieval and generation script
python scripts/evaluation/retrieval_and_generation.py \
    --seed $SEED \
    --log_dir $LOG_DIR \
    --data_dir $DATA_DIR \
    --test_community_ids $TEST_COMMUNITY_IDS \
    --num_chunks $NUM_CHUNKS \
    --base_models $BASE_MODELS \
    --retrievers $RETRIEVERS \
    --test_noises $TEST_NOISES \
    $VERBOSE

# Run evaluation script
python scripts/evaluation/eval.py \
    --log_dir $LOG_DIR \
    --data_dir $DATA_DIR
