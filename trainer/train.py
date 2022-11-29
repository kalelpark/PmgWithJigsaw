from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *

def train(args, train_loader, model, optimizer, loss_fn, lr, train_loss_list, correct, total, epoch):

    model.train()
    idx = 0
    for input, target in train_loader:
        idx += 1
        input, target = input.float().to(args.device), target.type(torch.LongTensor).to(args.device)

        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])

        # Step1
        optimizer.zero_grad()
        inputs1 = jigsaw_generator(input, 8)
        output_1, _, _, _ = model(inputs1)
        loss1 = loss_fn(output_1, target) * 1
        loss1.backward()
        optimizer.step()

        # Step2
        optimizer.zero_grad()
        inputs2 = jigsaw_generator(input, 4)
        _, output_2, _, _ = model(inputs2)
        loss2 = loss_fn(output_2, target) * 1
        loss2.backward()
        optimizer.step()

        # Step 3
        optimizer.zero_grad()
        inputs3 = jigsaw_generator(input, 2)
        _, _, output_3, _ = model(inputs3)
        loss3 = loss_fn(output_3, target) * 1
        loss3.backward()
        optimizer.step()

        # Step 4
        optimizer.zero_grad()
        _, _, _, output_concat = model(input)
        concat_loss = loss_fn(output_concat, target) * 2
        concat_loss.backward()
        optimizer.step()

        _, predicted = torch.max(output_concat.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        train_loss_list[0] += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
        train_loss_list[1] += loss1.item()
        train_loss_list[2] += loss2.item()
        train_loss_list[3] += loss3.item()
        train_loss_list[4] += concat_loss.item()

    train_acc = 100. * float(correct) / total
    train_loss_list[0] = train_loss_list[0] / (idx + 1)

    return train_acc, train_loss_list[0]








         


