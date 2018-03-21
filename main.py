
# coding: utf-8

import numpy as np
import pickle
import random
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from encoder import mVGG19
from decoder import mRNN

def extract_feature(layer, img_dir='./Flickr8k_Dataset', cap_dir='./Flickr8k_text', bsz=32, bn=True, save_feat=True, return_feat=False):
    # get feature extractor
    extractor = mVGG19(bn=bn)
    extractor.build(layer)
    # feature: (sample num, D, L)
    names, features, captions = extractor.forward(img_dir, cap_dir, bsz)
    # save extracted features
    if save_feat:
        print('saving extracted features...')
        with open('img_names.pickle', 'wb') as handle:
            pickle.dump(names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('features.pickle', 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('captions.pickle', 'wb') as handle:
            pickle.dump(captions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if return_feat:
        return names, features, captions

def train(features, captions, word2idx, att='hard', num_epochs=20, model_dir='./rnn_model.pth'):
    # initialize model
    model = mRNN(nvocab=len(word2idx), att=att).cuda()
    # check model initialized
    print('nvocab = {}, M = {}, D = {}, L = {}, N = {}, attention model is {}'.format(model.nvocab, model.M, model.D, model.L, model.N, model.att))
    print(model)
    # set loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # set other learning parameters
    since = time.time()
    pre_loss = np.inf
    stop_flag = 0
    idx_tn = np.arange(len(captions['train']))
    idx_ho = np.arange(len(captions['hold']))
    idx_ts = np.arange(len(captions['test']))
    # start learning
    for epoch in range(num_epochs):
        print('-'*30)
        print('epoch {}:'.format(epoch))
        ## TRAIN ##
        model.train()
        scheduler.step()
        counter = 0
        np.random.shuffle(idx_tn)
        for idx in idx_tn:
            counter += 1
            # clear gradients
            model.zero_grad()
            # get input and target.
            visual_info = Variable(torch.FloatTensor(features['train'][idx,:,:]).cuda())
            # reset memory
            model.init_hidden(visual_info)
            # randomly pick one caption for that image
            cap = random.choice(captions['train'][idx])
            words = cap.split()
            # start training
            last_word = Variable(torch.LongTensor([word2idx['<start>']]).cuda())
            output = model.forward(visual_info, last_word)
            if words[0].lower() in word2idx.keys():
                targ = Variable(torch.LongTensor([word2idx[words[0].lower()]]).cuda())
            else:
                targ = Variable(torch.LongTensor([word2idx['<NULL>']]).cuda())
            all_out = output
            all_targ = targ
            for iw in range(1, len(words)):
                output = model.forward(visual_info, targ)
                if words[iw].lower() in word2idx.keys():
                    targ = Variable(torch.LongTensor([word2idx[words[iw].lower()]]).cuda())
                else:
                    targ = Variable(torch.LongTensor([word2idx['<NULL>']]).cuda())
                all_out = torch.cat((all_out, output), 0)
                all_targ = torch.cat((all_targ, targ), 0)
                # all_out: seq_length*2981, all_targ: seq_length
            # calculate loss and backprop
            loss = loss_function(all_out, all_targ)
            loss.backward()
            optimizer.step()
            # print loss after every 1000 samples
            if counter % 1000 == 0:
                print(loss.data.cpu().numpy()[0])
            # DEBUG: Stop at 1000 samples
            # if counter % 1000 == 0:
            #     break
        print('-------------------')
        print('training time consuming: {0:.1f}s'.format(time.time()-since))
        since = time.time()

        ## VALIDATION ##
        model.eval()
        running_loss = 0
        for idx in idx_ho:
            visual_info = Variable(torch.FloatTensor(features['hold'][idx,:,:]).cuda())
            cap = random.choice(captions['hold'][idx])
            model.init_hidden(visual_info)
            words = cap.split()
            last_word = Variable(torch.LongTensor([word2idx['<start>']]).cuda())
            output = model.forward(visual_info, last_word)
            if words[0].lower() in word2idx.keys():
                targ = Variable(torch.LongTensor([word2idx[words[0].lower()]]).cuda())
            else:
                targ = Variable(torch.LongTensor([word2idx['<NULL>']]).cuda())
            all_out = output
            all_targ = targ
            for iw in range(1, len(words)):
                output = model.forward(visual_info, targ)
                if words[iw].lower() in word2idx.keys():
                    targ = Variable(torch.LongTensor([word2idx[words[iw].lower()]]).cuda())
                else:
                    targ = Variable(torch.LongTensor([word2idx['<NULL>']]).cuda())
                all_out = torch.cat((all_out, output), 0)
                all_targ = torch.cat((all_targ, targ), 0)
                # calculate loss and backprop
                # all_out: seq_length*2981, all_targ: seq_length
            loss = loss_function(all_out, all_targ)
            running_loss += loss.data.cpu().numpy()[0]
        # print losses
        print('cur_loss - pre_loss: {0:.6f} - {1:.6f}'.format(running_loss, pre_loss))
        # check convergence
        if running_loss >= pre_loss and stop_flag >= 2:
            break
        else:
            torch.save(model.state_dict(), model_dir)
            pre_loss = copy.deepcopy(running_loss)
            stop_flag = 0
        print('validate time consuming: {0:.1f}s'.format(time.time()-since))
        since = time.time()

def test(test_feat, word2idx, idx2word, att, model_dir='./rnn_model.pth', save_cap=True, save_dir='hard'):
    # load trained model 
    model = mRNN(nvocab=len(word2idx), att=att).cuda()
    print('loading model states from path...')
    model.load_state_dict(torch.load(model_dir))
    print('nvocab = {}, M = {}, D = {}, L = {}, N = {}, attention model is {}'.format(model.nvocab, model.M, model.D, model.L, model.N, model.att))
#     print(model.att)
    # predicting words
    caption_all = []
    feat_index_all = []
    for feat in test_feat:
        model.init_hidden(Variable(torch.FloatTensor(feat).cuda()))
        word_all = []
        feat_index = []
        word = '<start>'
        while True:
            word = model.predict(Variable(torch.FloatTensor(feat)).cuda(),
                                 Variable(torch.LongTensor([word2idx[word]])).cuda(), idx2word)
            word_all.append(word)
#             print(model.feat_idx)
            feat_index.append(model.feat_idx)
            if word == '.':
                break
        caption_all.append(word_all)
        feat_index_all.append(feat_index)
    # save generated captions
    if save_cap:
        print('Saving generated captions...')
        with open(save_dir+'_caption_generated.pickle', 'wb') as handle:
                pickle.dump(caption_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         print(len(caption_all))
        with open(save_dir+'_feature_index.pickle', 'wb') as handle:
                pickle.dump(feat_index_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         print(len(feat_index_all))
    return caption_all, feat_index_all

def visualize_attention(img_dir, feat_index, caption, att):
    plt.figure(figsize=(17,17))
    #original img
    ori_img = plt.imread(img_dir)
    img = Image.fromarray(ori_img).convert('LA')
    # img.thumbnail([224, 224], Image.ANTIALIAS)
    # print(img.size)
    img = img.resize([224, 224], Image.ANTIALIAS)
    # print(img.size)
    plt.subplot(4,4,1)
    plt.imshow(ori_img)

    # plot hard attention
    if att == 'hard':
        for i in range(len(caption)):
            if i > 14:
                break
            feat_vect = torch.zeros(1, 196)
            feat_vect[0, feat_index[i]] = 255
            feat_vect = feat_vect.view(14,14)
            feat_vect = feat_vect.numpy()
#             feat_vect = np.rot90(feat_vect, 2)
            # get the image
            pilimg = Image.fromarray(feat_vect)
            pilimg = pilimg.resize([224, 224], Image.ANTIALIAS)
            combined = np.asarray(img)[:,:,0] + np.asarray(pilimg)
        #     pilimg.thumbnail([224, 224], Image.ANTIALIAS)
            plt.subplot(4,4,i+2)
            plt.title(caption[i])
            plt.imshow(combined, cmap='gray')
        plt.show()
    # plot soft attention
    else:
        for i in range(len(caption)):
            if i > 14:
                break
            feat_vect = torch.from_numpy(feat_index[i])
            feat_vect = feat_vect.view(14,14)
            feat_vect = feat_vect.numpy()
            # get image
            pilimg = Image.fromarray(feat_vect)
            pilimg = pilimg.resize([224, 224], Image.ANTIALIAS)
            combined = np.asarray(img)[:,:,0] * np.asarray(pilimg)
            plt.subplot(4,4,i+2)
            plt.title(caption[i])
            plt.imshow(combined, cmap='gray')
        plt.show()

def calculate_score(caption_test, caption_gen, att):
    bleu = np.zeros((len(caption_test), 4))
    for i in range(0, len(caption_test)):
        sentence = caption_test[i]
        sentence_gen = caption_gen[i]
        sentence_split = []
        for j in range(0, 4):
            sentence_split.append(sentence[j].split())
        cc = SmoothingFunction()
        s1 = sentence_bleu(sentence_split, sentence_gen, weights=(1,0,0,0), smoothing_function=cc.method4)
        bleu[i][0] = s1
        s2 = sentence_bleu(sentence_split, sentence_gen, weights=(0.5,0.5,0,0), smoothing_function=cc.method4)
        bleu[i][1] = s2  
        s3 = sentence_bleu(sentence_split, sentence_gen, weights=(0.33, 0.33, 0.33, 0), smoothing_function=cc.method4)
        bleu[i][2] = s3
        s4 = sentence_bleu(sentence_split, sentence_gen, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method4)
        bleu[i][3] = s4
    # print
    for i in range(0,4):
        score = bleu[:,i]
        print('BLEU-'+str(i+1))
        print('min:'+ str(score.min()))
        print('max:' +  str(score.max()))   
        print('average:'+ str(np.mean(score)))
    # box plot
    bleu_to_plot = [bleu[:,0], bleu[:,1], bleu[:,2], bleu[:,3]]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(bleu_to_plot)
    ax.set_xticklabels(['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'])
    plt.title('box plot for '+att+' attention')
    plt.show()

def main():
    # load vocabulary
    word2idx = np.load('vocab.npy').item()
    print(len(word2idx))
    word2idx['<start>'] = 0
    word2idx['.'] = len(word2idx)
    word2idx['<NULL>'] = len(word2idx)
    idx2word = {y:x for x,y in word2idx.items()}
    print(len(word2idx), len(idx2word))

    # extract features
    extract_feature('conv5_4')

    # load features
    with open('img_names.pickle', 'rb') as fp:
        names = pickle.load(fp)
    print(len(names['train']), len(names['hold']), len(names['test']))
    with open('features.pickle', 'rb') as fp:
        features = pickle.load(fp)
    print(features['train'].shape, features['hold'].shape, features['test'].shape)
    with open('captions.pickle', 'rb') as fp:
        captions = pickle.load(fp)
        ##### CHECK LATER, SOME CAPTIONS ARE MISSING
    print(len(captions['train']), len(captions['hold']), len(captions['test']))

    #### HARD ATTENTION ####
    # train
    train(features, captions, word2idx, 'hard', model_dir='./rnn_model_hard.pth')
    # test
    caption_all, feat_index_all= test(features['test'], word2idx, idx2word, 'hard', model_dir='./rnn_model_hard.pth', save_dir='hard')
    # calculate score
    calculate_score(captions['test'], caption_all, 'HARD')
    # visualize
    i = 74
    visualize_attention('./Flickr8k_Dataset/'+names['test'][i], feat_index_all[i], caption_all[i])

    #### SOFT ATTENTION ####
    # train
    train(features, captions, word2idx, 'soft', model_dir='./rnn_model_soft.pth')
    # test
    caption_all, feat_index_all= test(features['test'], word2idx, idx2word, 'soft', model_dir='./rnn_model_soft.pth', save_dir='soft')
    # calculate score
    calculate_score(captions['test'], caption_all, 'SOFT')
    # visualize
    visualize_attention('./Flickr8k_Dataset/'+names['test'][i], feat_index_all[i], caption_all[i], 'soft')



if __name__ == '__main__':
    main()



