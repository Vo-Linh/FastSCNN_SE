import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.dataset import PandaSet
from utils.training import train_one_epoch
from utils.meanIoU import meanIOU
from src.losses.dice_loss import DiceLoss
from src.losses.focal_loss import FocalLoss
from src.models.fastscnn import FastSCNN
from src.models.squeesegv2 import SqueezeSegV2
from src.models.salanext import SalsaNext
from src.models.pointsegnet import PointSegNet
from src.models.fastscnn_se import SE_FastSCNN

# ============= HYPERPARAMETER =======================
BATCH_SIZE = 16
NUM_EPOCH = 100
LR = 2e-3
NUM_CLASSES = 13
RAMDOM_SEED = 32
device = 'cuda:0'
PATH = 'checkpoints/squeesegv2_focal.pt'
# ====================================================
list_images = os.listdir('./Pandaset/images')
train_list, valid_list = train_test_split(
    list_images, train_size=0.75, random_state=RAMDOM_SEED)

input_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(p=0.5),
])
train_dataset = PandaSet(images_dir='./Pandaset/images',
                         label_dir='./Pandaset/semanticLabels',
                         images_list=train_list,
                         transforms=input_transform)

valid_dataset = PandaSet(images_dir='./Pandaset/images',
                         label_dir='./Pandaset/semanticLabels',
                         images_list=valid_list,
                         transforms=input_transform)

training_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


loss_fn = FocalLoss()
model =  SqueezeSegV2(input_size = (BATCH_SIZE, 5, 64, 1856), num_classes = 13)
# model = SE_FastSCNN(NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=2e-4)


def lambda1(epoch): return 0.95 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

val_iou_max = 0.0
for i in range(NUM_EPOCH):
    val_iou = train_one_epoch(i, training_loader, validation_loader, model,
                    optimizer, scheduler, loss_fn, meanIOU, device)
    if val_iou >= val_iou_max:
        val_iou_max = val_iou
        torch.save(model.state_dict(), PATH)
        
print(f"Max mean IoU {val_iou_max}")