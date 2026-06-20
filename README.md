# CeRA: Capacity-enhanced Rank Adaptation

CeRA is a parameter-efficient fine-tuning (PEFT) method that overcomes LoRA's "linear ceiling" by replacing the low-rank linear delta with a non-linear parallel branch:

```
output = W_0(x)  +  down_proj( Dropout( SiLU( up_proj(x) ) ) )
```

Because of the non-linearity, CeRA can express functions outside the column span of the frozen weight matrix W_0, giving the adapter access to the full output space rather than only a low-dimensional subspace.

## Installation

```bash
git clone git@github.com:hhchen1105/cera-refactored.git
cd cera-refactored
pip install -r requirements.txt
```

Set your HuggingFace token in a `.env` file:

```
HF_TOKEN=hf_...
```

## Quick Start

### Training

```bash
# CeRA on MathInstruct (Llama-3.1-8B, rank=128, best LR=1e-3)
python train.py \
    --model_type CeRA --rank 128 --lr 1e-3 --dropout 0.1 \
    --dataset math --epochs 3

# LoRA baseline (best LR=3e-4)
python train.py \
    --model_type LoRA --rank 128 --lr 3e-4 --dropout 0.0 \
    --dataset math --epochs 3

# DoRA baseline (best LR=3e-4)
python train.py \
    --model_type DoRA --rank 128 --lr 3e-4 \
    --dataset math --epochs 3

# Different base model (e.g. Llama-3.2-3B)
python train.py \
    --model_type CeRA --rank 128 --lr 1e-3 --dropout 0.1 \
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
| `--lr` | `5e-4` | AdamW learning rate (use sweep-selected best: see Best LR table below, or pass `best` to `run_train.sh`) |

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

### Spectrum-derived figures (rank scaling, ER, manifold dim, signatures)

All computed from the SVD spectrum JSON produced by `analyze_svd.py`:

```bash
# PPL vs rank (Fig. 1 left)
python analysis/plot_rank_scaling.py \
    --metric ppl --results_dir results --dataset orca \
    --output results/rank_scaling_ppl_orca.pdf

# Manifold dimensionality vs rank (Fig. 1 right)
python analysis/plot_rank_scaling.py \
    --metric manifold --dataset orca \
    --svd_json results/svd_spectra_orca.json \
    --output results/rank_scaling_manifold_orca.pdf

# Effective Rank vs rank
python analysis/plot_rank_scaling.py \
    --metric er --svd_json results/svd_spectra_orca.json \
    --output results/rank_scaling_er_orca.pdf

# Singular value spectra (spectral signature; Fig. 3 shows rank 512 only,
# truncated at index 600 with the rank limit annotated)
python analysis/plot_svd_spectra.py \
    --svd_json results/svd_spectra_orca.json --ranks 512 \
    --xmax 600 --rank_limit 512 \
    --output results/svd_signature_orca.pdf
```

### Effective Rank (per-module, ablation variants)

```bash
# Ablation ER -- per-module ER at the best checkpoint of each variant
python analysis/analyze_er.py \
    --mode ablation --rank 128 \
    --full_path      results/.../CeRA_Full/CeRA \
    --nodropout_path results/.../CeRA_NoDrop/CeRA \
    --relu_path      results/.../CeRA_ReLU/CeRA \
    | tee results/er_ablation.csv
```

The ER *trajectory* modes of `analyze_er.py` (`manifold`, `lr_sensitivity`,
`dropout`) are deprecated: they need legacy per-data-count checkpoints that the
trainer no longer saves. Use `plot_rank_scaling.py --metric er` instead.

### Training curves (validation PPL over training)

```bash
python analysis/plot_training_curves.py \
    --results_dir results --dataset orca --rank 512 \
    --lrs 1e-4 5e-4 --output results/training_curves_dropout.pdf
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

See [`paper_experiments/README.md`](paper_experiments/README.md) for a complete guide to reproducing all tables and figures from the paper using Slurm.

```bash
bash paper_experiments/submit_main_comparison.sh --dry-run  # preview
bash paper_experiments/submit_main_comparison.sh            # submit
```

## Slurm Setup

Generic Slurm job wrappers are in `slurm/`. Before submitting, edit each script to:

1. Add your cluster account: `#SBATCH -A YOUR_ACCOUNT`
2. Activate your environment: `conda activate cera_env`

```bash
sbatch slurm/run_train.sh CeRA 128 best 0.1 math 3
sbatch slurm/run_eval.sh  my_cell CeRA 128 best 0.1
```

Pass `best` as the LR argument to automatically use the sweep-selected optimum for each method/rank combination (Llama-3.1-8B; Llama-3.2-1B; Llama-3.2-3B):

| Method | Rank | Best LR (8B) | Best LR (1B) | Best LR (3B) |
|--------|------|--------------|--------------|--------------|
| CeRA   | 64   | 3e-4         | 3e-4         | 3e-4         |
| CeRA   | 128  | 1e-3         | 3e-4         | 5e-4         |
| LoRA   | 64   | 3e-4         | 3e-4         | 3e-4         |
| LoRA   | 128  | 3e-4         | 5e-4         | 5e-4         |
| DoRA   | 64   | 3e-4         | 1e-3         | 1e-3         |
| DoRA   | 128  | 5e-4         | 1e-3         | 5e-4         |
| CeRA   | 512  | 3e-4         | 3e-4         | 5e-4         |
| LoRA   | 512  | 3e-4         | 5e-4         | 5e-4         |
| DoRA   | 512  | 1e-3         | 5e-4         | 5e-4         |

R512 best LRs are from the 1B/3B `math` sweep; R512 on Llama-3.1-8B (`math`)
was not swept (`--`). Full per-LR R512 numbers are in the "R512 LR Sweep
Results" section below.

### Large sweeps with job arrays

For multi-cell sweeps (e.g. LR sweep across model sizes), use the array scripts so all tasks are submitted as two jobs and SLURM auto-fills slots as they open:

```bash
# Submit all tasks in slurm/configs/sweep_1b3b.txt (32 cells, max 5 concurrent)
TRAIN_JOB=$(sbatch --parsable --array=1-32%5 slurm/run_train_array.sh slurm/configs/sweep_1b3b.txt)
sbatch --array=1-32%5 --dependency=aftercorr:$TRAIN_JOB slurm/run_eval_array.sh slurm/configs/sweep_1b3b.txt
```

Config file format (`slurm/configs/*.txt`); the trailing DATASET column is
optional and defaults to `math`:
```
# CELL_ID  METHOD  RANK  LR  DROPOUT  BASE_MODEL  [DATASET]
cera_r64_lr3e-4_1b  CeRA  64  3e-4  0.1  meta-llama/Llama-3.2-1B
cera_orca_r64       CeRA  64  3e-4  0.1  meta-llama/Llama-3.1-8B  orca
```

## Running without Slurm

The `python` commands above (training, evaluation, analysis) run directly on
any machine with a CUDA GPU -- Slurm is only an orchestration layer. Notes for
non-Slurm users:

- **A CUDA GPU is required for training.** `train.py` places the model on
  `cuda` unconditionally; there is no CPU fallback.
- **Do not run `slurm/*.sh` or `paper_experiments/*.sh` directly.** The
  `slurm/` wrappers abort outside of `sbatch` (they depend on
  `SLURM_SUBMIT_DIR`), and the `paper_experiments/` scripts submit jobs via
  `sbatch`. Use the equivalent `python` commands instead.
- **`best` LR resolution lives in the Slurm wrappers.** Pass the explicit
  value from the Best LR table above to `train.py` / `evaluate.py`.

The sweep config files under `slurm/configs/` are plain text and not tied to
Slurm. To run a sweep sequentially without a scheduler:

```bash
grep -v '^\s*#' slurm/configs/sweep_orca_rank_scaling.txt | grep -v '^\s*$' | \
while read -r CELL METHOD RANK LR DROPOUT BASE_MODEL DATASET; do
    python train.py \
        --model_type "$METHOD" --rank "$RANK" --lr "$LR" \
        --dropout "$DROPOUT" --base_model "$BASE_MODEL" \
        --dataset "${DATASET:-math}" --epochs 3
done
```

To reproduce the three-part evaluation that `slurm/run_eval.sh` bundles
(MATH pass@1, MATH pass@10, GSM8K pass@1), run the three `evaluate.py`
commands from the Evaluation section against the same checkpoint.

## Citation

```bibtex
@article{chen26cera,
  title   = {CeRA: Breaking the Linear Ceiling of Low-Rank Adaptation with Inference-Time Non-linearity},
  author  = {Hung-Hsuan Chen},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.22911}
}
```
