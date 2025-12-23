import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms
from construct_dataset import *

#__all__ = ['CIFAR10DataLoader', 'ImageNetDataLoader', 'CIFAR100DataLoader']


class CIFAR10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)
        

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)

'''
class CIFAR10DataLoader(DataLoader):
    def __init__(self, split='train', image_size=224, batch_size=16, num_workers=0, user_num=0):
        
        transform = transforms.Compose([
                #transforms.Resize(image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = SensorDataSet(op=split, transform = transform, user_num=user_num)
        
        train = False
        if split == "train":
            train = True
        

        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            #shuffle=False if not train else True,
            shuffle=True,
            num_workers=num_workers)
 '''       
class CIFAR10DataLoader_pro(DataLoader):
    def __init__(self, x_data, y_data, split='train', image_size=224, batch_size=16, num_workers=4, user_num=0):
        
        transform = transforms.Compose([
                #transforms.Resize(image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = SensorDataSet_pro(x_data, y_data, op=split, transform = transform, user_num=user_num)
        
        train = False
        if split == "train":
            train = True
        

        super(CIFAR10DataLoader_pro, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            #shuffle=False if not train else True,
            shuffle=True,
            num_workers=num_workers)        


class CIFAR100DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)

        super(CIFAR100DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        print("the folder is:", data_dir)
        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


if __name__ == '__main__':
    
    data_loader = CIFAR10DataLoader(
        data_dir='/home/tmac/vision-transformer-pytorch-main/src/data/ImageNet/train',
        split='val',
        image_size=384,
        batch_size=16,
        num_workers=0)
'''
    for images, targets in data_loader:
        print(targets)
'''
