# DocXEval: A Systematic Evaluation of Explainability in Multimodal Document Understanding

**Authors:**  
Saifullah Saifullah  
Stefan Agne  
Andreas Dengel  
Sheraz Ahmed  

---

## Overview

DocXEval is a framework for systematically evaluating explainability methods in multimodal document understanding. It provides standardized pipelines for training, perturbation-based robustness evaluation, explanation generation, and downstream analysis across:

- Sequence Classification  
- Token Classification  
- Question Answering  

The framework is designed for reproducibility, comparability, and extensibility of explanation methods.

---

## Framework Architecture

This repository implements the experimental interface and evaluation layer of the system.

Core training, data processing, and execution pipelines are provided via a companion module:

```toml
atriaml = { git = "https://github.com/atriaml/atria", rev = "v0.0.0-alpha" }
```

---

## Environment Setup

```bash
uv sync
source .venv/bin/activate
```

---

## Training

```bash
# Sequence Classification
./scripts/run.sh ./scripts/experiment_00/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/experiment_00/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/experiment_00/02_cfgs_qa.sh
```

---

## Experiment 1: Perturbation Robustness

Evaluate model robustness under controlled modality perturbations.

```bash
./scripts/run.sh ./scripts/experiment_01/00_cfgs_cls.sh
./scripts/run.sh ./scripts/experiment_01/01_cfgs_ser.sh
./scripts/run.sh ./scripts/experiment_01/02_cfgs_qa.sh
```

Post-processing:

```bash
python src/docxeval/perturbation_robustness/modality_wise_mean_perf.py
python src/docxeval/perturbation_robustness/task_wise_modality_rankings.py
```

---

## Experiment 2: Explanation Generation

Generate explanations using perturbation-based attribution strategies.

```bash
./scripts/run.sh ./scripts/experiment_02/00_cfgs_cls.sh
./scripts/run.sh ./scripts/experiment_02/01_cfgs_ser.sh
./scripts/run.sh ./scripts/experiment_02/02_cfgs_qa.sh
```

---

## Experiment 3: Attention-Based Explanations

Compute explanation maps using transformer attention signals.

```bash
./scripts/run.sh ./scripts/experiment_03/00_cfgs_cls.sh
./scripts/run.sh ./scripts/experiment_03/01_cfgs_ser.sh
./scripts/run.sh ./scripts/experiment_03/02_cfgs_qa.sh
```

---

## Explanation Analysis

### Metric Aggregation

```bash
./scripts/run.sh ./scripts/explanation_analysis/metrics/00_cfgs_cls.sh
./scripts/run.sh ./scripts/explanation_analysis/metrics/01_cfgs_ser.sh
./scripts/run.sh ./scripts/explanation_analysis/metrics/02_cfgs_qa.sh
```

### Visualization

```bash
./scripts/run.sh ./scripts/explanation_analysis/viz_main/00_cfgs_cls.sh
./scripts/run.sh ./scripts/explanation_analysis/viz_main/01_cfgs_ser.sh
./scripts/run.sh ./scripts/explanation_analysis/viz_main/02_cfgs_qa.sh
```

### Plot Generation

```bash
./scripts/explanation_analysis/generate_plots.sh
```