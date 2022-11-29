import os
import torch.utils.data
import torchvision.transforms as transforms
from .cub import *
from .aircraft import *
from .stanfordcar import *

def get_loader(args):
    if args.dataset in ['cub', 'cub-200-2011', 'cub_200_2011', 'bird']:
        return cub_loader(args)
    elif args.dataset in ['stanford-cars', 'car', 'cars']:
        return car_loader(args)
    elif args.dataset in ['aircraft', 'fgvc-aircraft']:
        return aircraft_loader(args)

