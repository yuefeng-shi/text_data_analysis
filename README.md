# Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media
 
This repository contains the inference and evaluation pipeline accompanying the paper:
 
> **Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media**
 

 
## Data
 
All data assets live under the `data/` directory.
 
### `data/questions.csv`
 
Contains the evaluation questions together with their gold labels. The expected columns are:
 
| Column | Description |
|--------|-------------|
| `Dataset` | Dataset name (must match the directory name under `data_source/`) |
| `Question-Type` | Question type, e.g. `existence`, `count`, `calculation`, `comparison` |
| `Answer-type` | Expected answer format: `yes/no`, `number`, or `percentage` |
| `Question` | The question text passed to the model |
| `gold_{size}_{sample}` | Gold label for a given dataset size and sample index (e.g. `gold_100_1`) |

### `data/data_source/`
 
Contains the raw text files used as input to the inference pipeline. The directory layout is:
 
```
data/data_source/
└── {dataset_name}/
    └── {dataset_size}/          # e.g. 10, 50, 100, 250, 500, 750, 1000
        ├── {dataset_name}_1.txt
        ├── {dataset_name}_2.txt
        └── {dataset_name}_3.txt
```
 
Each `.txt` file contains one text entry per line. For **targeted datasets** (e.g. `ste_sentiment_analysis`, `te_stance_detection`), the filenames follow the same naming convention, but each line is tab-separated in the format `{target}\t{entry}`:
 
```
microsoft	Windows daily updates are completely broken.
microsoft	I am visiting their engineers tomorrow. Feeling excited.
```
 
---

 
---
 
## Requirements
 
**Python 3.10 or higher** is required.
 
Install all dependencies with:
 
```bash
pip install -r requirements.txt
```
 
---
 
## Part 1 — Running Inference (`run_pipeline.py`)
 
`run_pipeline.py` runs one or more LLMs over a collection of text dataset files and writes predictions to a consolidated CSV.
 

### Usage
 
```bash
python run_pipeline_a.py \
  --models gpt-4.1 gemini-2.5-flash qwen3-8b \
  --data_dir ./data \
  --question_csv ./questions.csv \
  --output_dir ./results \
  --openrouter_api_key YOUR_KEY \
  --use_openrouter \
  --openrouter_api_key YOUR_KEY \
  --use_task_description \
```
 
### All Arguments
 
| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | *(required)* | One or more model short names (see supported models below) |
| `--data_dir` | *(required)* | Root directory containing the dataset files |
| `--question_csv` | *(required)* | Path to the question CSV file |
| `--output_dir` | *(required)* | Directory where results will be written |
| `--openai_api_key` | `None` | OpenAI API key (for `--use_openrouter` off) |
| `--openrouter_api_key` | `None` | OpenRouter API key |
| `--use_openrouter` | `False` | Route all models through OpenRouter |
| `--save_raw_output` | `False` | Save raw (unparsed) model outputs to a text file alongside results |
| `--use_task_description` | `False` | Prepend a dataset-specific task description to the prompt |

 

### Output Files
 
| File | Description |
|------|-------------|
| `consolidated_results.csv` | Wide-format results: one row per question, columns `res_{size}_{run}_{model}` |
| `raw_output_{model}.txt` | Raw model responses per question (only with `--save_raw_output`) |
 
---
 
## Part 2 — Running Evaluation (`evaluation.py`)
 
`evaluation.py` takes the consolidated results CSV and produces hierarchical evaluation metrics and per-dataset aggregations.
 

### Usage
 
**Full pipeline (evaluation + aggregation):**
```bash
python evaluation.py \
  --input ./results/consolidated_results.csv \
  --eval_dir ./eval_out \
  --aggr_dir ./aggr_out
```
 
### All Arguments
 
| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Path to the wide-format input CSV (any filename accepted) |
| `--eval_dir` | `<input_stem>_eval/` | Output directory for Part 1 evaluation results |
| `--aggr_dir` | `<input_stem>_aggr/` | Output directory for Part 2 aggregation results |



### Output Files
 
**Part 1 — Evaluation (`--eval_dir`)**
 
| File | Description |
|------|-------------|
| `eval_hierarchical_datasets.xlsx` | Per-metric sheets at four hierarchy levels (Overall / By Size / By Instance), in both raw-value and formatted `score (valid/total)` form |
| `metric_F1-Macro_datasets.csv` | F1-Macro pivot table as CSV |
| `metric_RNRMSE_by_Range_datasets.csv` | RNRMSE pivot table as CSV |
| `metric_NRMSE_by_Size_datasets.csv` | NRMSE pivot table as CSV |

 
**Part 2 — Aggregation (`--aggr_dir`)**
 
| File | Description |
|------|-------------|
| `aggr_per_dataset.xlsx` | Task Average metrics per model × dataset. Contains two sheets: **Normal** (models as rows, datasets as columns) and **Transposed** (datasets as rows, models as columns) |
 
---
 

 
## Citation
 
If you use this framework in your research, please cite:
 
```bibtex
@article{teaf2026,
  title   = {Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media},
  author  = {Shi, Yuefeng and Ousidhoum, Nedjma and Camacho-Collados, Jose},
  year    = {2026}
}
```
 
---

