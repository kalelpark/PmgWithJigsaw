from __future__ import print_function
import logging
import random
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse

from PIL import Image
from utils import *
from models import *
from dataset import *
from trainer import *
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, required = True)         
    parser.add_argument("--dataset", type = str, required = True)
    parser.add_argument("--imgsize", type = int, default = 512, required = False)
    parser.add_argument("--crop", type = int, default = 512, required = False)
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--epochs", type = int, default = 300, required = False)
    parser.add_argument("--batchsize", type = int, default = 16, required = False)
    parser.add_argument("--lr", type = float, default = 1e-2, required = False)
    parser.add_argument("--momentum", type = float, default = 0.9, required = False)
    parser.add_argument("--weight_decay", type = float, default = 5e-4, required = False)
    parser.add_argument("--gpu_ids", type = str, required = True)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(args.seed)

    model = load_model(args, pretrain=True, require_grad=True)
    model = nn.DataParallel(model).to(args.device)

    train_loader, valid_loader = get_loader(args)

    loss_fn = nn.CrossEntropyLoss().to(args.device)
    
    optimizer = optim.SGD([
        {'params' : model.module.classifier_concat.parameters(), 'lr' : 0.002},
        {'params' : model.module.conv_block1.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier1.parameters(), 'lr' : 0.002},
        {'params' : model.module.conv_block2.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier2.parameters(), 'lr' : 0.002},
        {'params' : model.module.conv_block3.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier3.parameters(), 'lr' : 0.002},
        {'params' : model.module.features.parameters(), 'lr' : 0.0002},
    ],  momentum = args.momentum, weight_decay = args.weight_decay)
    
    wandb.init( name = args.dataset + "_PMG_Model",
                project = "Progressive-Multi-Granularity", reinit = True)

    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(args.epochs):
        train_loss_list = [0, 0, 0, 0, 0]
        correct, total = 0, 0
        train_acc, train_loss = train(args, train_loader, model, optimizer, loss_fn, lr, train_loss_list, correct, total, epoch)
        valid_acc, valid_acc_en, valid_loss = valid(args, valid_loader, model, loss_fn)

        wandb.log({
            "train_acc" : train_acc,
            "train_loss" : train_loss,
            "valid_acc" : valid_acc,
            "valid_acc_en" : valid_acc_en,
            "valid_loss" : valid_loss,
        })