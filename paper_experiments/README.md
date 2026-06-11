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
| `submit_dropout_curves.sh` | Fig 2 | CeRA dropout grid {0.0-0.3} + LoRA on SlimOrca at LR 1e-4/5e-4; validation PPL curves |
| `submit_spectral.sh` | Fig 1, 3, 4 | SlimOrca rank scaling training + SVD spectra + all spectrum-derived figure plots (PPL/manifold/ER vs rank, spectral signatures) |
| `submit_efficiency.sh` | Table 5 (efficiency) | Throughput / latency benchmark |

## Recommended Order

1. `submit_main_comparison.sh` -- trains and evaluates all main table cells
2. `submit_model_scale.sh` -- trains 1B and 3B; 8B reuses main comparison checkpoints
3. `submit_ablation_study.sh` -- ablation variants + auto-chained ER analysis
4. `submit_spectral.sh` -- trains its own SlimOrca rank scaling sweep (Fig. 1/3
   are self-contained); the math SVD / Fig. 4 plots additionally require the
   math checkpoints from step 1
5. `submit_dropout_curves.sh` -- Fig. 2 dropout robustness curves (self-contained)
6. `submit_efficiency.sh` -- latency benchmark (no checkpoint required)
7. `submit_lora_dropout.sh` -- reviewer response ablation (submit if requested)
8. `submit_dataset_ablation.sh` -- dataset comparison (submit if needed)

## Output Locations

| Script | Output |
|--------|--------|
| All training | `results/Exp_<METHOD>_<dataset>_R<rank>_.../<METHOD>/` |
| Eval (main, scale) | `results/eval_outputs/<cell_id>/` |
| SVD spectra | `results/svd_spectra.json` (math), `results/svd_spectra_orca.json` |
| Fig. 1 | `results/rank_scaling.pdf` |
| Fig. 2 | `results/training_curves_dropout.pdf` |
| Fig. 3 | `results/svd_signature_orca.pdf`, `results/rank_scaling_er_orca.pdf` |
| Fig. 4 | `results/svd_signature_math.pdf`, `results/rank_scaling_er_math.pdf`, `results/rank_scaling_manifold_math.pdf` |
| ER ablation (analyze_er.py, ablation mode) | stdout (redirect with `| tee results/er_*.csv`) |
| Efficiency | `results/efficiency_results.csv` |

Note: the ER *trajectory* modes of `analyze_er.py` are deprecated -- they need
legacy per-data-count checkpoints the trainer no longer saves. ER figures are
now computed from the SVD spectrum JSONs via `plot_rank_scaling.py --metric er`.

## Prerequisites

- Set `HF_TOKEN` in your `.env` file (required to download gated Llama models)
- Fill in your cluster account in `slurm/run_train.sh`, `slurm/run_eval.sh`, `slurm/run_analysis.sh`
- Activate your conda/venv environment in the slurm scripts
- `accelerate` must be installed (`pip install accelerate`); it is listed in `requirements.txt` but not always present in base environments
- `matplotlib` must be installed for the figure plot jobs (`pip install --user matplotlib`); also in `requirements.txt` but easy to miss on cluster environments

## 1B / 3B LR Sweep

Best LRs for Llama-3.2-1B and Llama-3.2-3B are determined via a separate sweep
before running `submit_model_scale.sh`. Use the job array scripts:

```bash
TRAIN_JOB=$(sbatch --parsable --array=1-32%5 --partition=8gpus \
    slurm/run_train_array.sh slurm/configs/sweep_1b3b.txt)
sbatch --array=1-32%5 --partition=8gpus --dependency=aftercorr:$TRAIN_JOB \
    slurm/run_eval_array.sh slurm/configs/sweep_1b3b.txt
```

After the sweep completes, identify the best LR per method/rank from
`results/eval_outputs/` and update the `LR = "best"` lookup tables in
`slurm/run_train.sh` and `slurm/run_eval.sh` (see the Best LR table in the
top-level README) before running `submit_model_scale.sh`.
