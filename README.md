# CeRA: Capacity-enhanced Rank Adaptation

CeRA is a parameter-efficient fine-tuning (PEFT) method that overcomes LoRA's
"linear ceiling" by replacing the low-rank linear delta with a non-linear
parallel branch:

```
output = W_0(x)  +  down_proj( Dropout( SiLU( up_proj(x) ) ) )
```

Because of the non-linearity, CeRA can express functions outside the column span
of the frozen weight matrix W_0, giving the adapter access to the full output
space rather than only a low-dimensional subspace.

## Installation

```bash
git clone https://github.com/<your-repo>/cera.git
cd cera
pip install -r requirements.txt
```

Set your HuggingFace token in a `.env` file:

```
HF_TOKEN=hf_...
```

## Quick Start

### Training

```bash
# CeRA on MathInstruct (Llama-3.1-8B, rank=128, lr=5e-4)
python train.py \
    --model_type CeRA --rank 128 --lr 5e-4 --dropout 0.1 \
    --dataset math --epochs 3

# LoRA baseline
python train.py \
    --model_type LoRA --rank 128 --lr 5e-4 --dropout 0.0 \
    --dataset math --epochs 3

# DoRA baseline
python train.py \
    --model_type DoRA --rank 128 --lr 5e-4 \
    --dataset math --epochs 3

# Different base model (e.g. Llama-3.2-3B)
python train.py \
    --model_type CeRA --rank 128 --lr 5e-4 --dropout 0.1 \
    --dataset math --epochs 3 \
    --base_model meta-llama/Llama-3.2-3B
```

Results are saved to `results/Exp_<MODEL_TYPE>_<dataset>_R<rank>_.../<MODEL_TYPE>/`.

### Evaluation

```bash
# MATH pass@1 (greedy, 500 problems)
python evaluate.py \
    --adapter_type cera --rank 128 --dropout 0.1 \
    --checkpoint results/Exp_CeRA_math_.../CeRA/cera_ckpt_best_75000.pt \
    --dataset math --num_samples 500 \
    --output_jsonl results/eval_cera_r128_math.jsonl

# GSM8K pass@1
python evaluate.py \
    --adapter_type cera --rank 128 --dropout 0.1 \
    --checkpoint results/Exp_CeRA_math_.../CeRA/cera_ckpt_best_75000.pt \
    --dataset gsm8k \
    --output_jsonl results/eval_cera_r128_gsm8k.jsonl

# MATH pass@10 (sampling, unbiased estimator)
python evaluate.py \
    --adapter_type cera --rank 128 --dropout 0.1 \
    --checkpoint results/Exp_CeRA_math_.../CeRA/cera_ckpt_best_75000.pt \
    --dataset math --num_samples 500 \
    --num_samples_per_problem 10 --temperature 0.8 --top_p 0.95 \
    --output_jsonl results/eval_cera_r128_math_pass10.jsonl
```

## CLI Reference

### train.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | `meta-llama/Llama-3.1-8B` | HuggingFace model ID |
| `--model_type` | `CeRA` | Adapter: `CeRA`, `LoRA`, `DoRA` |
| `--rank` | `64` | Adapter rank (CeRA: expansion = rank / hidden_size) |
| `--alpha` | `32` | LoRA/DoRA scaling factor |
| `--dropout` | `0.1` | CeRA structural dropout |
| `--act_fn` | `silu` | CeRA activation: `silu`, `relu`, `identity` |
| `--target_modules` | `q_proj,v_proj` | Comma-separated projection names |
| `--dataset` | `math` | `math`, `code`, `orca` |
| `--epochs` | `3` | Training epochs |
| `--lr` | `5e-4` | AdamW learning rate |

### evaluate.py

| Argument | Required | Description |
|----------|----------|-------------|
| `--adapter_type` | Yes | `cera`, `lora`, `dora` |
| `--rank` | Yes | Must match training config |
| `--checkpoint` | Yes | Path to `.pt` checkpoint file |
| `--dataset` | No | `gsm8k`, `mathinstruct`, `math` (default: `gsm8k`) |
| `--num_samples_per_problem` | No | 1 = greedy Pass@1; N > 1 = unbiased Pass@k |
| `--num_samples` | No | Cap number of problems |
| `--temperature` | No | Sampling temperature (default: 0.8) |

## Analysis

### SVD Spectrum

```bash
python analysis/analyze_svd.py \
    --base_dir results --dataset math \
    --output_json results/svd_spectra.json
```

### Effective Rank

```bash
# Manifold expansion: CeRA vs LoRA
python analysis/analyze_er.py \
    --mode manifold --rank 64 \
    --cera_path results/Exp_CeRA_math_R64_.../CeRA \
    --lora_path results/Exp_LoRA_math_R64_.../LoRA \
    | tee results/er_manifold.csv

# Ablation ER
python analysis/analyze_er.py \
    --mode ablation --rank 128 \
    --full_path      results/.../CeRA_Full/CeRA \
    --nodropout_path results/.../CeRA_NoDrop/CeRA \
    --relu_path      results/.../CeRA_ReLU/CeRA \
    | tee results/er_ablation.csv
```

### Throughput Benchmark

```bash
python analysis/benchmark.py \
    --base_model meta-llama/Llama-3.1-8B \
    --output_csv results/efficiency_results.csv
```

### Qualitative Comparison

```bash
python analysis/compare_generations.py \
    --cera_ckpt results/.../CeRA/cera_ckpt_best_75000.pt \
    --lora_ckpt results/.../LoRA/lora_ckpt_best_75000.pt \
    --cera_rank 128 --lora_rank 128 \
    --output results/case_study.txt
```

## Reproducing Paper Results

See [`paper_experiments/README.md`](paper_experiments/README.md) for a complete
guide to reproducing all tables and figures from the paper using Slurm.

```bash
bash paper_experiments/submit_main_comparison.sh --dry-run  # preview
bash paper_experiments/submit_main_comparison.sh            # submit
```

## Slurm Setup

Generic Slurm job wrappers are in `slurm/`. Before submitting, edit each script to:

1. Add your cluster account: `#SBATCH -A YOUR_ACCOUNT`
2. Activate your environment: `conda activate cera_env`

```bash
sbatch slurm/run_train.sh CeRA 128 5e-4 0.1 math 3
sbatch slurm/run_eval.sh  my_cell CeRA 128 5e-4 0.1
```

## Citation

```bibtex
@inproceedings{cera2025,
  title   = {CeRA: Capacity-enhanced Rank Adaptation},
  author  = {},
  year    = {2025},
  note    = {Paper under review}
}
```
