import os
import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch)) 
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])

    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()

    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0 : block_size, 0 : block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size : (x+1) * block_size,
                                                    y*block_size : (y +1) * block_size].clone()
        jigsaws[..., x * block_size: (x + 1) * block_size, y * block_size : (y + 1) * block_size] = temp

    return jigsaws

