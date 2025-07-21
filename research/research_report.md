# Autoregressive Transformer Language Model Ablation Study

## Experiment Overview

**Date:** 2025-07-20T20:49:06.424040
**Platform:** Google Colab
**Hardware:** Tesla T4 GPU
**Total Configurations:** 6

## Experimental Setup

### Dataset
- **Source:** HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- **Size:** 500 documents
- **Tokenizer:** HuggingFaceTB/SmolLM-135M

### Model Architecture
- **Type:** Autoregressive Transformer
- **Attention:** Multi-head attention with RoPE
- **Activation:** SiLU (Swish)
- **Normalization:** RMSNorm

### Training Configuration
- **Optimizer:** AdamW with weight decay 0.1
- **Scheduler:** Cosine Annealing with warmup
- **Mixed Precision:** Automatic Mixed Precision (AMP)

## Key Findings

- **Best Performing Configuration:** large_batch
- **Most Efficient Configuration:** small_model

## Configuration Details

### Baseline
**Description:** Standard configuration with balanced parameters

- **d_model:** 384
- **n_heads:** 6
- **n_layers:** 12
- **batch_size:** 8
- **learning_rate:** 0.0003
- **seq_len:** 512

### Small_Model
**Description:** Reduced model size for efficiency comparison

- **d_model:** 256
- **n_heads:** 4
- **n_layers:** 8
- **batch_size:** 12

### High_Lr
**Description:** Higher learning rate experiment

- **learning_rate:** 0.001

### Low_Lr
**Description:** Lower learning rate experiment

- **learning_rate:** 0.0001

### Large_Batch
**Description:** Larger batch size with scaled learning rate

- **batch_size:** 16
- **learning_rate:** 0.0005

### Short_Seq
**Description:** Shorter sequence length for memory efficiency

- **batch_size:** 12
- **seq_len:** 256


## Results Summary

### Performance Metrics
Configuration performance ranked by final training loss:

1. **large_batch:** 1.3342
2. **high_lr:** 2.2270
3. **baseline:** 3.6422
4. **short_seq:** 3.8749
5. **small_model:** 4.0526
6. **low_lr:** 4.8933

## Visualizations

The following visualizations are included in this research package:

1. **training_loss_comparison.png** - Training loss progression across all configurations
2. **final_performance_comparison.png** - Final performance metrics comparison
3. **efficiency_analysis.png** - Training efficiency analysis
4. **convergence_analysis.png** - Convergence behavior analysis

## Data Files

- **research_data.json** - Complete experimental data in JSON format
- **performance_metrics.csv** - Performance metrics in CSV format for statistical analysis
- **text_samples/** - Generated text samples for each configuration

## Usage for Research Paper

This data package contains all necessary information for writing a comprehensive research paper on autoregressive transformer ablation studies. The structured data format allows for easy integration with academic writing tools and statistical analysis software.

### Recommended Sections for Paper:

1. **Introduction** - Use experimental_setup data
2. **Methodology** - Reference model architecture and training configuration
3. **Results** - Use performance_metrics and statistical_analysis
4. **Discussion** - Reference key_findings and text_generation_quality
5. **Conclusion** - Summarize from results_summary

### Citation Data

Please ensure proper citation of the SmolLM corpus dataset and any other referenced materials in your research paper.
