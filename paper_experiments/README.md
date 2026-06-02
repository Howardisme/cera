# Paper Experiments

Each script in this directory reproduces one set of experiments from the CeRA paper.
All scripts support `--dry-run` for inspection without submitting jobs.

## Quick Start

```bash
# From the refactored/ root:
bash paper_experiments/submit_main_comparison.sh --dry-run   # preview
bash paper_experiments/submit_main_comparison.sh             # submit
```

## Experiment Map

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `submit_main_comparison.sh` | Table 2 (LR sweep) | CeRA / LoRA / DoRA on Llama-3.1-8B, MATH + GSM8K |
| `submit_model_scale.sh` | Table 3 (scale) | 1B / 3B / 8B model size comparison |
| `submit_dataset_ablation.sh` | Appendix | MathInstruct vs SlimOrca |
| `submit_ablation_study.sh` | Table 4 (ablation) | Activation fn / target modules / dropout, R=128 + R=512 |
| `submit_lora_dropout.sh` | Reviewer response | LoRA with dropout=0.1 baseline |
| `submit_spectral.sh` | Fig 2-3 (spectra) | SVD spectrum + Effective Rank trajectory |
| `submit_efficiency.sh` | Table 5 (efficiency) | Throughput / latency benchmark |

## Recommended Order

1. `submit_main_comparison.sh` -- trains and evaluates all main table cells
2. `submit_model_scale.sh` -- trains 1B and 3B; 8B reuses main comparison checkpoints
3. `submit_ablation_study.sh` -- ablation variants + auto-chained ER analysis
4. `submit_spectral.sh` -- SVD + ER (requires checkpoints from step 1)
5. `submit_efficiency.sh` -- latency benchmark (no checkpoint required)
6. `submit_lora_dropout.sh` -- reviewer response ablation (submit if requested)
7. `submit_dataset_ablation.sh` -- dataset comparison (submit if needed)

## Output Locations

| Script | Output |
|--------|--------|
| All training | `results/Exp_<METHOD>_<dataset>_R<rank>_.../<METHOD>/` |
| Eval (main, scale) | `results/eval_outputs/<cell_id>/` |
| SVD spectrum | `results/svd_spectra.json` |
| ER analysis | stdout (redirect with `| tee results/er_*.csv`) |
| Efficiency | `results/efficiency_results.csv` |

## Prerequisites

- Set `HF_TOKEN` in your `.env` file (required to download gated Llama models)
- Fill in your cluster account in `slurm/run_train.sh`, `slurm/run_eval.sh`, `slurm/run_analysis.sh`
- Activate your conda/venv environment in the slurm scripts
