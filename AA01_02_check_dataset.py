import os
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from util.datasets import build_dataset, build_transform
import support_args as spa
from PIL import Image


def main():
    args = spa.get_args_parser()
    dataset_test = build_dataset(is_train='test', args=args)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # image 1
    all_img1 = []
    for bat in data_loader_test:
        all_img1.append(bat[0].numpy())
    all_img1 = np.concatenate(all_img1, 0)

    # image 2
    all_img2 = []
    transform = build_transform('test', args)
    pa = 'E:/Project19_transfer_unfair/RETFound_Data/GDPH/test/'
    for fo in os.listdir(pa):
        for fi in os.listdir(pa + fo):
            fu_fi = pa + fo + '/' + fi
            img = Image.open(fu_fi)
            all_img2.append(transform(img))
    all_img2 = np.stack([x.numpy() for x in all_img2])

    # compare image 1 and image 2
    aa = 1

if __name__ == '__main__':
    main()
