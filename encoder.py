
# coding: utf-8

import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import pickle

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from flickr8k import Flickr8k

class mVGG19(object):
    '''
    Args:
        bn (bool): load model with batch normalization or not
        pretrain (bool): whether you want the model is pretrained or not
        layer (string): desired feature out of which convolution layer
        img_dir (string): Location of Dataset images
        cap_dir (string): Location of Dataset text
        bsz (int): batch size in extracting features
    Example Usage:
        extractor = mVGG19(True, True)
        extractor.build('conv5_1')
        features, captions = extractor.forward('./Flickr8k_Dataset', './Flickr8k_text', 32)
    '''
    def __init__(self, bn=True, pretrain=True):
        # load vgg19 with/without batch_normalization
        print('building feature extractor on vgg...')
        self.bn_flag = bn
        if bn:
            self.model = models.vgg19_bn(pretrained=pretrain)
            self.layer_dict = {
                'conv1_1': 2, 'conv1_2': 5, 'conv2_1': 9, 'conv2_2': 12,
                'conv3_1': 16, 'conv3_2': 19, 'conv3_3': 22, 'conv3_4': 25,
                'conv4_1': 29, 'conv4_2': 32, 'conv4_3': 35, 'conv4_4': 38,
                'conv5_1': 42, 'conv5_2': 45, 'conv5_3': 48, 'conv5_4': 51
            }
        else:
            self.model = models.vgg19(pretrained=pretrain)
            self.layer_dict = {
                'conv1_1': 1, 'conv1_2': 3, 'conv2_1': 6, 'conv2_2': 8,
                'conv3_1': 11, 'conv3_2': 13, 'conv3_3': 15, 'conv3_4': 17,
                'conv4_1': 20, 'conv4_2': 22, 'conv4_3': 24, 'conv4_4': 26,
                'conv5_1': 29, 'conv5_2': 31, 'conv5_3': 33, 'conv5_4': 35
            }
        
    def build(self, layer='conv5_1'):
        # build the desire feature extractor
        idx = self.layer_dict[layer]
        self.extractor = nn.Sequential(*list(self.model.features.children())[:idx+1])
        if torch.cuda.is_available():
            self.extractor.cuda()
        
    def forward(self, img_dir='./Flickr8k_Dataset', cap_dir='./Flickr8k_text', bsz=32):
        print('extracting features...')
        # load & preprocess the data
        normalization = transforms.Compose(
                    [
                        transforms.Scale((224,224)),
                        # (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0].
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ]
                )
        datasets = {
            'train': Flickr8k(img_dir, cap_dir, normalization, 'train'),
            'hold': Flickr8k(img_dir, cap_dir, normalization, 'dev'),
            'test': Flickr8k(img_dir, cap_dir, normalization, 'test')
        }
        # get dataloader
        dataloaders = {
            'train': DataLoader(
                    dataset = datasets['train'],
                    batch_size = bsz,
                    num_workers = 4
                ),
            'hold': DataLoader(
                    dataset = datasets['hold'],
                    batch_size = bsz,
                    num_workers = 4
                ),
            'test': DataLoader(
                    dataset = datasets['test'],
                    batch_size = bsz,
                    num_workers = 4
                )
        }
        # forward
        all_names = {'train': [], 'hold': [], 'test': []}
        all_features = {'train': [], 'hold': [], 'test': []}
        all_captions = {'train': [], 'hold': [], 'test': []}
        since = time.time()
        for phase in ['train', 'hold', 'test']:
            print('   '+phase+' features...')
            self.extractor.eval()
            tmp_features = []
            for sample in dataloaders[phase]:
                # get the inputs
                img_names, images, captions = sample
                images = Variable(images, volatile=True)
                if torch.cuda.is_available():
                    images = images.cuda()
                # (# of sample, # of filter, 14, 14)
                features = self.extractor(images)
                # (# of sample, # of filter, 196)
                features = features.view(features.size()[0], features.size()[1], -1)
                tmp_features.append(features.data.cpu().numpy())
                for i in range(len(captions[0])):
                    all_names[phase].append(img_names[i])
                    all_captions[phase].append([cap[i] for cap in captions])
            all_features[phase] = np.vstack(tmp_features)
        print('extracting feature took {0:.1f}s'.format(time.time()-since))
        #### return featuers as ndarray, captions as list
        return all_names, all_features, all_captions

if __name__ == '__main__':
    # get feature extractor
    extractor = mVGG19()
    extractor.build('conv5_1')
    # feature: (sample num, D, L)
    names, features, captions = extractor.forward()
    
    # save extracted features
    print('saving extracted features...')
    with open('img_names.pickle', 'wb') as handle:
        pickle.dump(names, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('features.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('captions.pickle', 'wb') as handle:
        pickle.dump(captions, handle, protocol=pickle.HIGHEST_PROTOCOL)
