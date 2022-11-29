from .resnet import *
from .pmg import *

def load_model(args, pretrain=True, require_grad=True):

    if args.dataset in ['cub', 'cub-200-2011', 'cub_200_2011', 'bird']:
        num_classes = 200
    elif args.dataset in ['stanford-cars', 'car', 'cars']:
        num_classes = 196
    elif args.dataset in ['aircraft', 'fgvc-aircraft']:
        num_classes = 100

    if args.model in ["resnet50", "resnet", "resNet50", "ResNet50"] :
        model = resnet50(pretrained=pretrain)
        for param in model.parameters():
            param.requires_grad = require_grad
        model = PMG(model, 512, num_classes)

    return model
