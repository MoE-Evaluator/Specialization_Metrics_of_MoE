#!/bin/bash
# ============================================================================
# Example Configuration File for run_all_metrics.sh
# ============================================================================
#
# Usage:
#   1. Copy this file: cp config_example.sh config.sh
#   2. Edit config.sh with your actual paths
#   3. Run: bash run_all_metrics.sh --config config.sh
#
# ============================================================================

# ============================================================================
# Base Directory Configuration
# ============================================================================
BASE_DIR="/path/to/your/data"
TOKENIZER_PATH="/path/to/tokenizer"  # e.g., "./llm_model/DeepSeek-R1-0528"

# Model and output configuration
MODEL_NAME="qwen_235B"
RESULT_SAVE_DIR="./results/full_analysis"

# ============================================================================
# Processing Parameters
# ============================================================================
THRESHOLD=0.85
MODE="threshold"  # "threshold" or "topk"
# TOPK=8          # Uncomment if using topk mode

N_VALUES=(2 5 10 20)
RSS_SAMPLE_VALUES=(500 1000 2000)

# ============================================================================
# Input Directories
# ============================================================================
# These are directories containing results_all.jsonl files
INPUT_DIRS=(
    "${BASE_DIR}/aime_2025_messages/expert_statistics"
    "${BASE_DIR}/allenai_sciq_data_val_set_messages/expert_statistics"
    "${BASE_DIR}/bigbio_medqa_dev_messages/expert_statistics"
    "${BASE_DIR}/bigbio_medqa_test_messages/expert_statistics"
    "${BASE_DIR}/cais_hle_messages/expert_statistics"
    "${BASE_DIR}/livecodebench_code_generation_test_messages/expert_statistics"
    "${BASE_DIR}/nguha-legalbench_legalbench_messages/expert_statistics"
    "${BASE_DIR}/princeton_SWE_bench_Verified_data_test_set_messages/expert_statistics"
    "${BASE_DIR}/yale-financemath_validation_messages/expert_statistics"
)

# ============================================================================
# Postprocess Directories
# ============================================================================
# First-level directories containing merged_test_val.jsonl and subdirectories
POSTPROCESS_DIRS=(
    "${BASE_DIR}/livecodebench_code_generation_test_messages"
    "${BASE_DIR}/nguha--legalbench_legalbench_messages"
)

# ============================================================================
# Domain Configuration for Count Matrix
# ============================================================================
# JSON format mapping domain names to expert_selection_counts.csv paths
DOMAIN_CONFIG='{
    "Math": "'"${BASE_DIR}"'/aime_2025_messages/expert_statistics/expert_selection_counts.csv",
    "Science": "'"${BASE_DIR}"'/allenai_sciq_data_val_set_messages/expert_statistics/expert_selection_counts.csv",
    "Medical": "'"${BASE_DIR}"'/bigbio_medqa_dev_messages/expert_statistics/expert_selection_counts.csv",
    "Medical2": "'"${BASE_DIR}"'/bigbio_medqa_test_messages/expert_statistics/expert_selection_counts.csv",
    "Knowledge": "'"${BASE_DIR}"'/cais_hle_messages/expert_statistics/expert_selection_counts.csv",
    "Code": "'"${BASE_DIR}"'/livecodebench_code_generation_test_messages/expert_statistics/expert_selection_counts.csv",
    "Legal": "'"${BASE_DIR}"'/nguha-legalbench_legalbench_messages/expert_statistics/expert_selection_counts.csv",
    "Code2": "'"${BASE_DIR}"'/princeton_SWE_bench_Verified_data_test_set_messages/expert_statistics/expert_selection_counts.csv",
    "Finance": "'"${BASE_DIR}"'/yale-financemath_validation_messages/expert_statistics/expert_selection_counts.csv"
}'

# ============================================================================
# Output Configuration (Optional)
# ============================================================================
# OUTPUT_BASE_DIR="/path/to/output"
# SUMMARY_OUTPUT_FILE="/path/to/summary.csv"
