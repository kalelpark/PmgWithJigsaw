import os
import imp
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

def get_transform(args, train = True):
    if train:
        return transforms.Compose([
            transforms.Resize(args.imgsize),
            transforms.RandomCrop(args.crop, padding = 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(args.imgsize),
            transforms.CenterCrop(args.crop),
            transforms.ToTensor(),
            normalize
        ])


# train_transform_v2 = transforms.Compose([
#                       transforms.Resize(550),
#                       transforms.RandomCrop(448, padding = 8),
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       normalize,
#                       ])

# test_transform = transforms.Compose([
#                      transforms.Resize(550),
#                      transforms.CenterCrop(448),
#                      transforms.ToTensor(),
#                      normalize
#                      ])