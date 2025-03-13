# Vision Transformer (ViT) & Masked Autoencoder (MAE) Framework

A comprehensive framework for self-supervised learning and medical image analysis with advanced transformer architectures.

## Key Features
- **ViT/MAE Implementations**: Pre-trained vision transformers and masked autoencoders
- **Multi-Modal Analysis**: Supports both ECG signals and retinal images
- **Fairness Evaluation**: Integrated unfairness metrics and visualization
- **Distributed Training**: AMP support and multi-GPU workflows
- **Adaptive Learning**: Layer-wise LR decay, cosine schedules with warmup

## Project Structure

### Core Components
| File | Description |
|------|-------------|
| **`models_vit.py`** | Enhanced ViT with global average pooling option |
| **`models_mae.py`** | MAE implementation with ViT backbone |
| **`main_features_MLP.py`** | Retinal MLP classifier trainer |
| **`yaml_config_hook.py`** | Hierarchical YAML config loader |
| **`support_args.py`** | CLI config for training/finetuning (200+ params) |
| **`support_based.py`** | Metrics (AUC/Acc/Sen/Spe) calculation core |

### Training Framework
| File | Purpose |
|------|---------|
| **`engine_finetune.py`** | AMP-optimized training/eval pipeline |
| **`main_finetune.py`** | Distributed ViT training entry point |

### Utilities
| File | Function |
|------|----------|
| **`misc.py`** | Distributed training helpers |
| **`datasets.py`** | Unified dataset builder with transforms |
| **`pos_embed.py`** | 2D positional encoding + interpolation |
| **`lr_sched.py`** | Warmup + cosine LR scheduler |
| **`lr_decay.py`** | Layer-wise learning rate decay |

### Analysis Pipeline
| File | Analysis Type |
|------|---------------|
| **`AA10_01-04_*.py`** | Unfairness metric calculation & visualization |
| **`AA02_01-04_*.py`** | Results processing → Excel reporting |
| **`AA03_01_*.py`** | Training completeness checker |
| **`AA01_01_preprocess_dataset.py`** | Retinal image organizer (train/val/test split) |

