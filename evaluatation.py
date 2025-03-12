from util.datasets import build_transform
from PIL import Image
import os
import support_args as spa
import numpy as np
import models_vit
import torch

args = spa.get_args_parser()

# load model
model = models_vit.__dict__[args.model](
    img_size=args.input_size,
    num_classes=args.nb_classes,
    drop_path_rate=args.drop_path,
    global_pool=args.global_pool,
)
model.load_state_dict(torch.load(args.task + args.dataset + '/'+'checkpoint-best.pth'))

# dataset
transform = build_transform('test', args)
pa = '../../../RETFound_Data/' + args.dataset + '/test/'
img_li, na_li, lab_li, pred_li, fea_li = [], [], [], [], []
for i, fo in enumerate(os.listdir(pa)):
    for fi in os.listdir(pa + fo):
        fu_fi = pa + fo + '/' + fi
        img = Image.open(fu_fi)
        img_li.append(transform(img).numpy())
        na_li.append(fo + '/' + fi)
        lab_li.append(i)
img_li = np.stack(img_li)
lab_li = np.array(lab_li)