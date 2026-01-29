#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# Unified Metric Pipeline Script
# ============================================================================
#
# Description:
#   This script provides a unified entry point for all metric calculations.
#   It consolidates the following functionalities:
#     1. Postprocessing (image and count generation)
#     2. Expert group routing and N-gram statistics
#     3. Count matrix processing
#     4. N-gram statistics (detailed)
#     5. Rademacher complexity calculation
#     6. Results aggregation
#
# Usage:
#   bash run_all_metrics.sh [OPTIONS]
#
# Options:
#   --step <step_name>       Run specific step(s). Options:
#                              postprocess, expert_group, count_matrix,
#                              ngram_rademacher, aggregate, all (default: all)
#   --config <config_file>   Use external config file for INPUT_DIRS
#   --threshold <value>      Threshold for expert group routing (default: 0.85)
#   --n_values <values>      N values for n-gram (default: "2 5 10 20")
#   --model_name <name>      Model name for count matrix (default: "model")
#   --output_base_dir <dir>  Base directory for outputs
#   --help                   Show this help message
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRIC_DIR="$(dirname "$SCRIPT_DIR")"
UTILS_DIR="${METRIC_DIR}/utils"
POSTPROCESS_DIR="${METRIC_DIR}/postprocess"

# Python scripts
POSTPROCESS_SCRIPT="${POSTPROCESS_DIR}/postprocessing_for_image_and_count.py"
COMPUTE_METRICS_SCRIPT="${UTILS_DIR}/compute_expert_metrics.py"
AGGREGATE_NGRAM_SCRIPT="${UTILS_DIR}/aggregate_expert_group_ngram.py"
PROCESS_COUNT_MATRIX_SCRIPT="${UTILS_DIR}/process_count_matrix.py"
N_GRAM_SCRIPT="${UTILS_DIR}/n_gram_statistics.py"
RADEMACHER_SCRIPT="${UTILS_DIR}/rademacher_complexity.py"
AGGREGATE_RESULTS_SCRIPT="${UTILS_DIR}/aggregate_results.py"

# ============================================================================
# Default Configuration
# ============================================================================
BASE_DIR=""  # Set your base directory here
TOKENIZER_PATH=""  # Set your tokenizer path here (for postprocess step)
MODEL_NAME="model"
RESULT_SAVE_DIR="./results/full_analysis"

# Default parameters
THRESHOLD=0.85
MODE="threshold"
N_VALUES=(2 5 10 20)
RADEMACHER_VALUES=(500 1000 2000)

# Step control
RUN_STEP="all"
OUTPUT_BASE_DIR=""
SUMMARY_OUTPUT_FILE=""

# ============================================================================
# Input Directories Configuration
# ============================================================================
# Define your input directories here or use --config to load from external file
INPUT_DIRS=(
    # Example:
    # "${BASE_DIR}/aime_2025_messages/expert_statistics"
    # "${BASE_DIR}/allenai_sciq_data_val_set_messages/expert_statistics"
    # "${BASE_DIR}/bigbio_medqa_dev_messages/expert_statistics"
    # "${BASE_DIR}/bigbio_medqa_test_messages/expert_statistics"
    # "${BASE_DIR}/cais_hle_messages/expert_statistics"
    # "${BASE_DIR}/livecodebench_code_generation_test_messages/expert_statistics"
    # "${BASE_DIR}/nguha-legalbench_legalbench_messages/expert_statistics"
    # "${BASE_DIR}/princeton_SWE_bench_Verified_data_test_set_messages/expert_statistics"
    # "${BASE_DIR}/yale-financemath_validation_messages/expert_statistics"
)

# Domain configuration for count matrix (JSON format)
DOMAIN_CONFIG='{}'
# Example:
# DOMAIN_CONFIG='{
#     "Math": "${BASE_DIR}/aime_2025_messages/expert_statistics/expert_selection_counts.csv",
#     "Science": "${BASE_DIR}/allenai_sciq_data_val_set_messages/expert_statistics/expert_selection_counts.csv",
#     "Medical": "${BASE_DIR}/bigbio_medqa_dev_messages/expert_statistics/expert_selection_counts.csv",
#     "Code": "${BASE_DIR}/livecodebench_code_generation_test_messages/expert_statistics/expert_selection_counts.csv",
#     "Legal": "${BASE_DIR}/nguha-legalbench_legalbench_messages/expert_statistics/expert_selection_counts.csv",
#     "Finance": "${BASE_DIR}/yale-financemath_validation_messages/expert_statistics/expert_selection_counts.csv"
# }'

# Postprocess directories (first level dirs with merged_test_val.jsonl)
POSTPROCESS_DIRS=(
    # Example:
    # "${BASE_DIR}/livecodebench_code_generation_test_messages"
    # "${BASE_DIR}/nguha--legalbench_legalbench_messages"
)

# ============================================================================
# Helper Functions
# ============================================================================
show_help() {
    head -50 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

check_script_exists() {
    if [ ! -f "$1" ]; then
        log_error "Script not found: $1"
        return 1
    fi
    return 0
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            RUN_STEP="$2"
            shift 2
            ;;
        --config)
            if [ -f "$2" ]; then
                source "$2"
            else
                log_error "Config file not found: $2"
                exit 1
            fi
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --topk)
            TOPK="$2"
            MODE="topk"
            shift 2
            ;;
        --n_values)
            IFS=' ' read -ra N_VALUES <<< "$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output_base_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --tokenizer_path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --result_save_dir)
            RESULT_SAVE_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ============================================================================
# Step 1: Postprocessing
# ============================================================================
run_postprocess() {
    log_section "Step 1: Postprocessing (Image and Count Generation)"

    if ! check_script_exists "$POSTPROCESS_SCRIPT"; then
        log_error "Skipping postprocess step - script not found"
        return 1
    fi

    if [ ${#POSTPROCESS_DIRS[@]} -eq 0 ]; then
        log_info "No POSTPROCESS_DIRS defined, skipping postprocess step"
        return 0
    fi

    local total=0
    local success=0
    local failed=0

    for first_level_dir in "${POSTPROCESS_DIRS[@]}"; do
        if [ ! -d "$first_level_dir" ]; then
            log_info "Directory not found, skipping: $first_level_dir"
            continue
        fi

        local inference_file="${first_level_dir}/merged_test_val.jsonl"
        if [ ! -f "$inference_file" ]; then
            log_info "merged_test_val.jsonl not found, skipping: $first_level_dir"
            continue
        fi

        for second_level_dir in "${first_level_dir}"/*; do
            if [ ! -d "$second_level_dir" ]; then
                continue
            fi

            total=$((total + 1))
            log_info "Processing: $(basename "$first_level_dir")/$(basename "$second_level_dir")"

            if python "$POSTPROCESS_SCRIPT" \
                --inference_file "$inference_file" \
                --expert_data_file "$second_level_dir" \
                --tokenizer_path "$TOKENIZER_PATH" \
                --output_base_dir "$first_level_dir"; then
                success=$((success + 1))
                log_success "Completed: $(basename "$second_level_dir")"
            else
                failed=$((failed + 1))
                log_error "Failed: $(basename "$second_level_dir")"
            fi
        done
    done

    log_info "Postprocess Summary: Total=$total, Success=$success, Failed=$failed"
    [ $failed -eq 0 ] && return 0 || return 1
}

# ============================================================================
# Step 2: Expert Group Routing and N-gram Statistics
# ============================================================================
run_expert_group() {
    log_section "Step 2: Expert Group Routing and N-gram Statistics"

    if ! check_script_exists "$COMPUTE_METRICS_SCRIPT"; then
        log_error "Skipping expert_group step - script not found"
        return 1
    fi

    if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
        log_info "No INPUT_DIRS defined, skipping expert_group step"
        return 0
    fi

    for input_dir in "${INPUT_DIRS[@]}"; do
        if [ ! -d "$input_dir" ]; then
            log_info "Directory not found, skipping: $input_dir"
            continue
        fi

        local input_file="${input_dir}/results_all.jsonl"
        if [ ! -f "$input_file" ]; then
            log_info "results_all.jsonl not found, skipping: $input_dir"
            continue
        fi

        # Determine output directory
        local base_output_dir
        if [ -n "$OUTPUT_BASE_DIR" ]; then
            local dir_name=$(basename "$input_dir")
            base_output_dir="${OUTPUT_BASE_DIR}/${dir_name}"
        else
            base_output_dir="$input_dir"
        fi

        local routing_output_dir="${base_output_dir}/expert_group_routing"
        mkdir -p "$routing_output_dir"

        log_info "Processing: $input_dir"
        log_info "Output: $routing_output_dir"

        # Build command arguments
        local cmd_args=(
            --input_file "$input_file"
            --mode "$MODE"
            --n_values "${N_VALUES[@]}"
            --output_dir "$routing_output_dir"
        )

        if [ "$MODE" = "threshold" ]; then
            cmd_args+=(--threshold "$THRESHOLD")
        else
            cmd_args+=(--topk "$TOPK")
        fi

        if python "$COMPUTE_METRICS_SCRIPT" "${cmd_args[@]}"; then
            log_success "Expert group metrics completed: $(basename "$input_dir")"
        else
            log_error "Expert group metrics failed: $(basename "$input_dir")"
            return 1
        fi
    done

    # Run aggregation if aggregate script exists
    if check_script_exists "$AGGREGATE_NGRAM_SCRIPT" && [ ${#INPUT_DIRS[@]} -gt 0 ]; then
        log_info "Aggregating expert group n-gram results..."

        local summary_file
        if [ -n "$SUMMARY_OUTPUT_FILE" ]; then
            summary_file="$SUMMARY_OUTPUT_FILE"
        else
            local first_dir="${INPUT_DIRS[0]}"
            if [ -n "$OUTPUT_BASE_DIR" ]; then
                local dir_name=$(basename "$first_dir")
                summary_file="${OUTPUT_BASE_DIR}/${dir_name}/expert_group_routing/group_n_gram_summary.csv"
            else
                summary_file="${first_dir}/expert_group_routing/group_n_gram_summary.csv"
            fi
        fi

        python "$AGGREGATE_NGRAM_SCRIPT" \
            --input_dirs "${INPUT_DIRS[@]}" \
            --output_file "$summary_file"

        log_success "Aggregation completed: $summary_file"
    fi

    return 0
}

# ============================================================================
# Step 3: Count Matrix Processing
# ============================================================================
run_count_matrix() {
    log_section "Step 3: Count Matrix Processing"

    if ! check_script_exists "$PROCESS_COUNT_MATRIX_SCRIPT"; then
        log_error "Skipping count_matrix step - script not found"
        return 1
    fi

    if [ "$DOMAIN_CONFIG" = "{}" ]; then
        log_info "No DOMAIN_CONFIG defined, skipping count_matrix step"
        return 0
    fi

    log_info "Model: $MODEL_NAME"
    log_info "Output: $RESULT_SAVE_DIR"

    if python "$PROCESS_COUNT_MATRIX_SCRIPT" \
        --model_name "$MODEL_NAME" \
        --result_save_dir "$RESULT_SAVE_DIR" \
        --domain_config "$DOMAIN_CONFIG"; then
        log_success "Count matrix processing completed"
        return 0
    else
        log_error "Count matrix processing failed"
        return 1
    fi
}

# ============================================================================
# Step 4: N-gram Statistics and Rademacher Complexity
# ============================================================================
run_ngram_rademacher() {
    log_section "Step 4: N-gram Statistics and Rademacher Complexity"

    if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
        log_info "No INPUT_DIRS defined, skipping ngram_rademacher step"
        return 0
    fi

    for input_dir in "${INPUT_DIRS[@]}"; do
        if [ ! -d "$input_dir" ]; then
            log_info "Directory not found, skipping: $input_dir"
            continue
        fi

        local input_file="${input_dir}/results_all.jsonl"
        if [ ! -f "$input_file" ]; then
            log_info "results_all.jsonl not found, skipping: $input_dir"
            continue
        fi

        log_info "Processing: $input_dir"

        # N-gram statistics
        if check_script_exists "$N_GRAM_SCRIPT"; then
            log_info "Running N-gram statistics..."
            for n in "${N_VALUES[@]}"; do
                local output_dir="${input_dir}/n_gram/n_gram_${n}"
                mkdir -p "$output_dir"

                if python "$N_GRAM_SCRIPT" \
                    --input_file "$input_file" \
                    --n "$n" \
                    --output_dir "$output_dir"; then
                    log_success "N-gram (n=$n) completed"
                else
                    log_error "N-gram (n=$n) failed"
                fi
            done
        fi

        # Rademacher complexity
        if check_script_exists "$RADEMACHER_SCRIPT"; then
            log_info "Running Rademacher complexity..."
            for value in "${RADEMACHER_VALUES[@]}"; do
                local output_dir="${input_dir}/rademacher/rademacher_complexity_${value}"
                local output_file="${output_dir}/rademacher_complexity.json"
                mkdir -p "$output_dir"

                if python "$RADEMACHER_SCRIPT" \
                    --input_file "$input_file" \
                    --num_samples "$value" \
                    --num_simulations "$value" \
                    --output_file "$output_file"; then
                    log_success "Rademacher (samples=$value) completed"
                else
                    log_error "Rademacher (samples=$value) failed"
                fi
            done
        fi
    done

    return 0
}

# ============================================================================
# Step 5: Results Aggregation
# ============================================================================
run_aggregate() {
    log_section "Step 5: Results Aggregation"

    if ! check_script_exists "$AGGREGATE_RESULTS_SCRIPT"; then
        log_info "Aggregate results script not found, skipping"
        return 0
    fi

    if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
        log_info "No INPUT_DIRS defined, skipping aggregate step"
        return 0
    fi

    local output_file="${OUTPUT_BASE_DIR:-${INPUT_DIRS[0]}}/aggregated_results.csv"

    log_info "Aggregating results to: $output_file"

    if python "$AGGREGATE_RESULTS_SCRIPT" \
        --input_dirs "${INPUT_DIRS[@]}" \
        --output_file "$output_file"; then
        log_success "Results aggregation completed"
        return 0
    else
        log_error "Results aggregation failed"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log_section "Unified Metric Pipeline"
    log_info "Script directory: $SCRIPT_DIR"
    log_info "Metric directory: $METRIC_DIR"
    log_info "Run step: $RUN_STEP"
    log_info "Mode: $MODE"
    if [ "$MODE" = "threshold" ]; then
        log_info "Threshold: $THRESHOLD"
    else
        log_info "TopK: $TOPK"
    fi
    log_info "N values: ${N_VALUES[*]}"
    log_info "Input directories: ${#INPUT_DIRS[@]}"

    case $RUN_STEP in
        postprocess)
            run_postprocess
            ;;
        expert_group)
            run_expert_group
            ;;
        count_matrix)
            run_count_matrix
            ;;
        ngram_rademacher)
            run_ngram_rademacher
            ;;
        aggregate)
            run_aggregate
            ;;
        all)
            run_postprocess || true
            run_expert_group || true
            run_count_matrix || true
            run_ngram_rademacher || true
            run_aggregate || true
            ;;
        *)
            log_error "Unknown step: $RUN_STEP"
            log_info "Available steps: postprocess, expert_group, count_matrix, ngram_rademacher, aggregate, all"
            exit 1
            ;;
    esac

    log_section "Pipeline Completed"
}

# Run main function
main
