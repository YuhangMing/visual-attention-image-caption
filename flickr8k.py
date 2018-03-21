
# coding: utf-8

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL

class Flickr8k(Dataset):
    '''
    Args:
        img_dir (string): Location of Flickr8k Dataset images
        cap_dir (string): Location of Flickr8k Dataset text
        transform (torch.transform): transformation to perform on input image
        purpose (string): whether this is for train/validation/test
    Example Usage:
        example_transform = transforms.Compose(
            [
                transforms.Scale((224,224)),
                transforms.ToTensor(),
            ]
        )
        flickr8k_train = Caltech256('./Flickr8k_Dataset', './Flickr8k_text', example_transform, 'train')
        
        train_data = DataLoader(
            dataset = flickr8k_train,
            batch_size = 32,
            shuffle = True,
            num_workers = 4
        )
    '''
    
    def __init__(self, img_dir, cap_dir, transform=None, purpose='train'):
        self.img_dir = img_dir
        self.cap_dir = cap_dir
        self.transform = transform
        self.purpose = purpose
        self.files = {}
        # stores the names of all image files
        name_dir = os.path.join(cap_dir, 'Flickr_8k.'+purpose+'Images.txt')
        with open(name_dir, 'r') as f:
            name_list = f.read().splitlines()
        # stores the captions
        cap_dir = os.path.join(cap_dir, 'Flickr8k.lemma.token.txt')
        with open(cap_dir, 'r') as f:
            cap_list = f.read().splitlines()
        # store the image - captions pairs in files
        for cap in cap_list:
            tmp_name = cap.split('#')[0]
            tmp_cap = cap.split('\t')[1]
            if tmp_name in name_list:
                if tmp_name in self.files.keys():
                    # double check to prevent replicas
                    if tmp_cap not in self.files[tmp_name]:
                        self.files[tmp_name].append(tmp_cap)
                else:
                    self.files[tmp_name] = [tmp_cap]
                 
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name, caption = list(self.files.items())[idx]
        currdir = os.path.join(self.img_dir, img_name)
        image = PIL.Image.open(currdir).convert("RGB") # double check to prevent grayscale imgs
        if self.transform:
            image = self.transform(image)
        #### return a tuple of imagename, image and a list of 5 captions
        sample = (img_name, image, caption)
        return sample


if __name__ == '__main__':
    img_dir = './Flickr8k_Dataset'
    cap_path = './Flickr8k_text'

    mytransform = transforms.Compose(
                [
                    transforms.Scale((224,224)),
                    # transforms.RandomHorizontalFlip(),
                    # (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0].
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]
            )

    flicker8k_train = Flickr8k(img_dir, cap_path, mytransform, 'train')

    flicker8k_val = Flickr8k(img_dir, cap_path, mytransform, 'dev')

    flicker8k_test = Flickr8k(img_dir, cap_path, mytransform, 'test')



