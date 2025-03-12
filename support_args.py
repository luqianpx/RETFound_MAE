import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    # Training parameters
    parser.add_argument('--batch_size', default=40, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (to increase effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer-wise learning rate decay')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='Lower bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='Epochs to warm up LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy ("v0", "original", or "rand-m9-mstd0.5-inc1")')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # Random Erase parameters
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase probability (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first split')

    # Mixup and Cutmix parameters
    parser.add_argument('--mixup', type=float, default=0, help='Mixup alpha, enabled if > 0')
    parser.add_argument('--cutmix', type=float, default=0, help='Cutmix alpha, enabled if > 0')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='Cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix ("batch", "pair", "elem")')

    # Fine-tuning parameters
    parser.add_argument('--finetune', default='../../../RETFound_Weights/RETFound_cfp_weights.pth', type=str,
                        help='Path to finetune from checkpoint')
    parser.add_argument('--task', default='../results/', type=str, help='Task output directory')
    parser.add_argument('--global_pool', action='store_true', help='Global pooling for classification (default: True)')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool')

    # Dataset parameters
    parser.add_argument('--dataset', default='right_race_dis4_balanclassattr_kiIy', type=str, help='Dataset path')
    parser.add_argument('--data_path', default='../../../RETFound_Data/', type=str, help='Dataset path')
    parser.add_argument('--nb_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--dataset_num', default=0, type=int)

    parser.add_argument('--output_dir', default='../output_dir/', help='Path to save the output')
    parser.add_argument('--log_dir', default='../log_dir/', help='Path for TensorBoard logs')
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing (cuda or cpu)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')

    # Training resume parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enable distributed evaluation')

    # DataLoader parameters
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for efficient transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true', help='Enable distributed training')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training setup')

    return parser.parse_args()
