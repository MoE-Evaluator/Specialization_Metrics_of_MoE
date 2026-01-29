# DBES: A Systematic Benchmark and Metric Suite for Evaluating Expert Specialization in Large-Scale MoEs

This repository contains the official implementation of **DBES (Domain Bench for Expert Specialization)**, a comprehensive benchmark and metric suite for evaluating expert specialization patterns in Mixture-of-Experts (MoE) language models.

## Abstract

Mixture-of-Experts (MoE) architectures have emerged as a promising approach for scaling large language models efficiently. However, understanding whether and how experts specialize across different domains remains an open question. We present DBES, a systematic benchmark comprising diverse domain-specific datasets and a comprehensive metric suite including Rademacher complexity, N-gram transition analysis, and expert routing statistics. Our framework enables fine-grained analysis of expert activation patterns, providing insights into the specialization behavior of large-scale MoE models.

## Repository Structure

```
moe_specialization/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── sglang/                   # Modified SGLang with expert distribution recording
├── preprocess/               # Inference and data collection scripts
│   ├── server_sglang.sh      # Server launch script
│   ├── infer_and_save.py     # Batch inference with expert logging
│   └── infer_backend.py      # MoE client and log analyzer
├── metric/                   # Metric computation modules
│   ├── script/               # Unified batch processing scripts
│   │   ├── run_all_metrics.sh    # Main entry point for all metrics
│   │   └── config_example.sh     # Configuration template
│   ├── utils/                # Core metric implementations
│   │   ├── compute_expert_metrics.py    # Expert group routing & N-gram
│   │   ├── rademacher_complexity.py     # Rademacher complexity
│   │   ├── n_gram_statistics.py         # N-gram transition analysis
│   │   ├── process_count_matrix.py      # Count matrix processing
│   │   └── aggregate_results.py         # Results aggregation
│   └── postprocess/          # Data postprocessing scripts
└── databench/                # Domain-specific benchmark datasets
```

## Installation

### Step 1: Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/.git
cd 

# Create a virtual environment (recommended)
conda create -n dbes python=3.10 -y
conda activate dbes

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Modified SGLang

Our framework requires a modified version of SGLang with expert distribution recording capabilities. **You must replace the official SGLang with our modified version.**

```bash
# If you have official SGLang installed, uninstall it first
pip uninstall sglang -y

git clone https://www.github.com/sgl-project/sglang.git sglang_install
# Overwrite the original source
cp -r sglang/* sglang_install/python/sglang
# Navigate to the modified SGLang directory
cd sglang_install

# Install the modified SGLang in development mode
pip install -e .

# Return to project root
cd ..
```

The modified SGLang includes the following key features:
- `--expert-distribution-recorder-mode per_token`: Records expert selection for each token
- `--expert-distribution-recorder-buffer-size`: Configurable buffer size for recording
- API endpoints for controlling expert distribution recording (`/start_expert_distribution_record`, `/stop_expert_distribution_record`, `/dump_expert_distribution_record`)

### Step 3: Verify Installation

```bash
# Verify SGLang installation
python -c "import sglang; print(sglang.__version__)"

# Verify metric modules
python -c "from metric.utils import compute_expert_metrics, rademacher_complexity; print('Metrics OK')"
```

## Usage

### Phase 1: Start the Inference Server

Launch the SGLang server with expert distribution recording enabled:

```bash
cd preprocess

# Configure environment variables
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR='/path/to/save/expert_logs'

# Start the server (modify paths as needed)
bash server_sglang.sh
```

**Server Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-path` | Path to the MoE model | Required |
| `--tp` | Tensor parallelism degree | 16 |
| `--dp` | Data parallelism degree | 1 |
| `--expert-distribution-recorder-mode` | Recording mode (`per_token`) | per_token |
| `--expert-distribution-recorder-buffer-size` | Buffer size | 1000000 |
| `--port` | Server port | 8000 |

Example for DeepSeek-R1:
```bash
python -m sglang.launch_server \
    --model-path ./llm_model/DeepSeek-R1-0528 \
    --tp 16 \
    --expert-distribution-recorder-mode per_token \
    --expert-distribution-recorder-buffer-size 1000000 \
    --port 8000
```

### Phase 2: Run Inference and Collect Expert Data

Once the server is running, execute batch inference to collect expert activation data:

```bash
cd preprocess

# Run inference on benchmark datasets
python infer_and_save.py \
    --input_file /path/to/your/benchmark_data.jsonl \
    --api_url http://localhost:8000 \
    --root_dir ./outputs \
    --sources "aime_2025_messages.jsonl" \
              "yale-financemath/validation_messages.jsonl" \
              "livecodebench_code_generation/test_messages.jsonl" \
              "allenai_sciq/data/val_set_messages.jsonl" \
              "cais_hle_messages.jsonl" \
              "bigbio_medqa_dev_messages.jsonl" \
              "nguha--legalbench/legalbench_messages.jsonl"
```

**Inference Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_file` | Path to input JSONL file | Required |
| `--api_url` | SGLang server URL | http://localhost:8000 |
| `--root_dir` | Output directory for results | ./outputs |
| `--sources` | List of dataset sources to process | All supported |
| `--disable-expert-recording` | Disable expert logging | False |

### Phase 3: Compute Metrics

After collecting expert activation data, run the unified metric pipeline:

```bash
cd metric/script

# Option 1: Use configuration file (recommended)
cp config_example.sh config.sh
# Edit config.sh with your paths
bash run_all_metrics.sh --config config.sh

# Option 2: Run all metrics with command line arguments
bash run_all_metrics.sh \
    --base_dir /path/to/expert_statistics \
    --model_name "deepseek_r1" \
    --threshold 0.85 \
    --step all

# Option 3: Run specific metric steps
bash run_all_metrics.sh --step expert_group --config config.sh
bash run_all_metrics.sh --step ngram_rademacher --config config.sh
bash run_all_metrics.sh --step count_matrix --config config.sh
```

**Available Metric Steps:**

| Step | Description | Output |
|------|-------------|--------|
| `postprocess` | Image and count generation | Processed data files |
| `expert_group` | Expert group routing & N-gram statistics | `expert_group_routing.json/csv`, `expert_group_ngram_n{2,5,10,20}.json/csv` |
| `count_matrix` | Count matrix processing | Domain comparison matrices |
| `ngram_rademacher` | Detailed N-gram & Rademacher complexity | `n_gram_X/`, `rademacher_complexity_X/` |
| `aggregate` | Results aggregation | Summary CSV files |
| `all` | Run all steps sequentially | All outputs |

## Metrics Overview

### 1. Expert Group Routing Statistics

Analyzes the distribution of expert activations across layers:

```bash
python metric/utils/compute_expert_metrics.py \
    --input_file results_all.jsonl \
    --mode threshold \
    --threshold 0.85 \
    --output_dir output/
```

### 2. Rademacher Complexity

Measures the complexity of expert selection patterns using Monte Carlo simulation:

```bash
python metric/utils/rademacher_complexity.py \
    --input_file results_all.jsonl \
    --num_samples 1000 \
    --num_simulations 1000 \
    --output_file rademacher_results.json
```

### 3. N-gram Transition Analysis

Analyzes sequential patterns in expert activation:

```bash
python metric/utils/n_gram_statistics.py \
    --input_file results_all.jsonl \
    --n 5 \
    --output_dir n_gram_output/
```

## Supported Domains

DBES includes benchmarks across multiple domains:

| Domain | Dataset | Description |
|--------|---------|-------------|
| Mathematics | AIME 2025 | Competition-level math problems |
| Finance | Yale FinanceMath | Financial reasoning tasks |
| Code | LiveCodeBench, SWE-bench | Code generation and software engineering |
| Science | SciQ | Science question answering |
| Medical | MedQA | Medical domain QA |
| Legal | LegalBench | Legal reasoning tasks |
| Knowledge | HLE | General knowledge evaluation |

## Output Format

### Expert Statistics (JSONL)

```json
{
  "token_id": 12345,
  "layers": {
    "0": {"expert_ids": [1, 3, 7], "weights": [0.4, 0.35, 0.25]},
    "1": {"expert_ids": [2, 5, 8], "weights": [0.5, 0.3, 0.2]}
  }
}
```

### Routing Statistics (JSON)

```json
{
  "layer_0": {
    "top_experts": [1, 3, 7, 12],
    "cumulative_weight": 0.856,
    "expert_weights": {"1": 0.25, "3": 0.22, "7": 0.20, "12": 0.18}
  }
}
```

## Citation

If you use DBES in your research, please cite our paper:

```bibtex
@article{dbes2025,
  title={DBES: A Systematic Benchmark and Metric Suite for Evaluating Expert Specialization in Large-Scale MoEs},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SGLang team for the base inference framework
- Benchmark dataset creators for their valuable contributions

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
