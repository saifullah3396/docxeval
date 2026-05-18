# Project README

## Environment Setup

Prepare the environment using uv:

```bash
uv sync 
```

## Training and Evaluation

To train models use the following command:

```bash
# Sequence Classification
./scripts/run.sh ./scripts/experiment_00/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/experiment_00/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/experiment_00/02_cfgs_qa.sh
```

## Experiment 1: Perturbation Robustness Evaluation

To run Experiment 1 for perturbation robustness evaluation, use the following command. This will use the checkpoints from the training runs with varying
degrees of perturbations applied to different input modalities and
evaluate the performance.

```bash
# Sequence Classification
./scripts/run.sh ./scripts/experiment_01/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/experiment_01/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/experiment_01/02_cfgs_qa.sh
```

After evaluation is done, run the analysis script to generate the modality-wise mean performance plots and modality-wise ranking of each perturbation strategy.:

```bash
python src/docxeval/perturbation_robustness/modality_wise_mean_perf.py 
```

Compute task/model wise ranking:
```bash 
python src/docxeval/perturbation_robustness/task_wise_modality_rankings.py 
```

## Experiment 2: Explanations

Using the analysis results and ranking from Experiment 1, now we can use the strategies for computing the explanations. Run the following commands to generate explanations for different tasks:

```bash
# Sequence Classification
./scripts/run.sh ./scripts/experiment_02/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/experiment_02/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/experiment_02/02_cfgs_qa.sh
```

## Experiment 3: Attention-Based Explanations 
Compute attention-based explanations for different tasks using the following commands. This is generally the same as experiment 2, but with a different set of strategies for computing explanations based on attention scores.

```bash
# Sequence Classification
./scripts/run.sh ./scripts/experiment_03/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/experiment_03/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/experiment_03/02_cfgs_qa.sh
```

## Explanation Analysis:
Once all explanations are generated, we can run the analysis scripts to summarize the metrics or visualize the data:

### Generate summarized metrics for each metric over each explanation result:
```bash
# Sequence Classification
./scripts/run.sh ./scripts/explanation_analysis/metrics/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/explanation_analysis/metrics/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/explanation_analysis/metrics/02_cfgs_qa.sh
```

### Generate aggregated summarized metrics results and plots for each task:
```bash
# Sequence Classification
./scripts/run.sh ./scripts/explanation_analysis/metrics/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/explanation_analysis/metrics/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/explanation_analysis/metrics/02_cfgs_qa.sh
```

### Generate qualitative visualizations for each task:
```bash
# Sequence Classification
./scripts/run.sh ./scripts/explanation_analysis/viz_main/00_cfgs_cls.sh

# Token Classification
./scripts/run.sh ./scripts/explanation_analysis/viz_main/01_cfgs_ser.sh

# Question Answering
./scripts/run.sh ./scripts/explanation_analysis/viz_main/02_cfgs_qa.sh
```

### Generate the quantitative plots for each task:
```bash
./scripts/explanation_analysis/generate_plots.sh
```