import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from .augmentation import *
from torchvision import datasets
from torch.utils.data import DataLoader

def car_loader(args):
    train_ds = datasets.StanfordCars(root = "dataset/data/stanfordcar/",
                                     split = "train",
                                     transform = get_transform(args),
                                     download= True)
                                     
    test_ds = datasets.StanfordCars(root = "dataset/data/stanfordcar/",
                                    split = "test",
                                    transform = get_transform(args, train = False),
                                    download= True)

    train_loader = DataLoader(train_ds, batch_size = args.batchsize, shuffle = True, pin_memory = True)
    test_loader = DataLoader(test_ds, batch_size = args.batchsize, shuffle = False, pin_memory = True)

    return train_loader, test_loader