#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree


# ==================== Get OFFSET ====================
# 아래 evaluate 함수 내 confidence score 출력 부에서 선택
# ======== Distance 기반 점수 ==========
def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

# ======== Consine 유사도 기반 점수 ========
def calc_cosine_sim(feat1, feat2, vshift=10):
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    sims = []

    for i in range(0, len(feat1)):
        # Cosine similarity 계산 (pairwise)
        f1 = feat1[[i], :].repeat(win_size, 1)
        f2 = feat2p[i:i+win_size, :]
        cos = torch.nn.functional.cosine_similarity(f1, f2, dim=-1)
        sims.append(cos)

    return sims

# ======== hybrid(거리&코사인유사도) 기반 ==========
def calc_lse_hybrid(feat1, feat2, vshift=10, alpha=0.5, beta=0.5):
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    hybrid_scores = []

    for i in range(len(feat1)):
        f1 = feat1[[i], :].repeat(win_size, 1)
        f2 = feat2p[i:i+win_size, :]

        cos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=-1)  # [-1, 1]
        l2_dist = torch.nn.functional.pairwise_distance(f1, f2)  # [0, ∞)

        # 1. normalize cosine: -1~1 → 0~1
        cos_sim_norm = (cos_sim + 1.0) / 2.0

        # 2. normalize L2 distance: 전체 영상 기준으로 정규화
        l2_min = l2_dist.min()
        l2_max = l2_dist.max()
        l2_dist_norm = (l2_dist - l2_min) / (l2_max - l2_min + 1e-8)  # prevent divide by zero

        # 최종 hybrid score (낮을수록 좋음)
        hybrid = alpha * (1 - cos_sim_norm) + beta * l2_dist_norm
        hybrid_scores.append(hybrid)

    return hybrid_scores



# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def evaluate(self, opt, videofile):

        self.__S__.eval();

        # ========== ==========
        # Convert files
        # ========== ==========

        if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
          rmtree(os.path.join(opt.tmp_dir,opt.reference))

        os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

        command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        
        flist = glob.glob(os.path.join(opt.tmp_dir,opt.reference,'*.jpg'))
        flist.sort()

        for fname in flist:
            images.append(cv2.imread(fname))

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
        print('Compute time %.3f sec.' % (time.time()-tS))

        # ======== Distance 기반 점수 ==========
        # dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)

        # mdist = torch.mean(torch.stack(dists,1),1)

        # minval, minidx = torch.min(mdist,0)

        # offset = opt.vshift-minidx
        # conf   = torch.median(mdist) - minval

        # fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        # fconf   = torch.median(mdist).numpy() - fdist
        # fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        # numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print('Framewise conf: ')
        # print(fconfm)
        # print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        # dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        # return offset.numpy(), conf.numpy(), dists_npy
  

        # ======== Consine 유사도 기반 점수 ========
        sims = calc_cosine_sim(im_feat, cc_feat, vshift=opt.vshift)

        # similarity는 높을수록 좋은 것
        # 평균 similarity로 confidence 뽑기
        sims = calc_cosine_sim(im_feat, cc_feat, vshift=opt.vshift)

        msim = torch.mean(torch.stack(sims, 1), 1)

        maxval, maxidx = torch.max(msim, 0)

        offset = opt.vshift - maxidx
        confidence = maxval  # 바로 max similarity 사용

        fsim = numpy.stack([sim[maxidx].numpy() for sim in sims])

        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise sim: ')
        print(fsim)
        print('AV offset: \t%d \nMax sim: \t%.3f\nConfidence: \t%.3f' % (offset, maxval, confidence))

        sims_npy = numpy.array([sim.numpy() for sim in sims])
        return offset.numpy(), confidence.numpy(), sims_npy

        # ======== hybrid(거리&코사인유사도) 기반 ==========
        # scores = calc_lse_hybrid(im_feat, cc_feat, vshift=opt.vshift, alpha=0.5, beta=0.5)
        # mscore = torch.mean(torch.stack(scores, 1), 1)
        # minval, minidx = torch.min(mscore, 0)

        # offset = opt.vshift - minidx
        # confidence = -minval  # 점수가 작을수록 좋은 것이므로 반대로 출력해도 좋음

        # fscores = numpy.stack([score[minidx].numpy() for score in scores])
        # print('Framewise hybrid score: ')
        # print(fscores)
        # print('AV offset: \t%d \nLSE-H (min): \t%.3f\nHybrid Confidence: \t%.3f' % (
        #     offset, minval, -minval))  # -minval로 신뢰도 표현






    def extract_feature(self, opt, videofile):

        self.__S__.eval();
        
        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
