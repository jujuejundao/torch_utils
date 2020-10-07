import numpy as np 
import math
import copy
import random
import time
import os
from itertools import permutations
import tqdm

import cv2
import scipy.io.wavfile as wavfile

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as Func

from apex import amp
from apex.parallel import DistributedDataParallel

EPS = 1e-8

class dataset(data.Dataset):
    def __init__(self,
                list_IDs,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size,
                partition='test',
                audio_only=False,
                sampling_rate=16000):

        self.minibatch =[]
        self.audio_only = audio_only
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        # if not audio_only:
        #     batch_size = int(batch_size/2)
        mix_lst=open(list_IDs).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))

        self.C = int((len(mix_lst[0].split(','))-3)/4)
        mix_lst = sorted(mix_lst, key=lambda data: (float(data.split(',')[-1])- float(data.split(',')[-2])), reverse=True)

        start = 0
        while True:
            end = min(len(mix_lst), start + batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]
        min_length = int((float(batch_lst[-1].split(',')[-1])*self.sampling_rate -float(batch_lst[-1].split(',')[-2])*self.sampling_rate))
        mixtures=[]
        audios=[]
        for line in batch_lst:
            mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_')+'.wav'
            _, mixture = wavfile.read(mixture_path)
            mixture = self._audio_norm(mixture[:min_length])
            mixtures.append(mixture)

            target_audio=[]
            line=line.split(',')
            start = int(float(line[-2])*self.sampling_rate)
            end = start + min_length
            for c in range(self.C):
                audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                _, audio = wavfile.read(audio_path)
                target_audio.append(self._audio_norm(audio[start:end]))
            audios.append(np.asarray(target_audio))
        return np.asarray(mixtures), np.asarray(audios)

    def __len__(self):
        return len(self.minibatch)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))

class avDataset(data.Dataset):
    def __init__(self, list_IDs, path_a, path_v, mix_s = 2,
                    mix_db_ratio = 10, peak_norm = True, dynamic_mix = True,
                    visual_pretrain = 0):
        self.list_IDs = list_IDs # List of audio files 
        self.path_a = path_a
        self.path_v = path_v
        self.mix_s = mix_s # No. of audios to mix in a mixture
        self.mix_db_ratio = mix_db_ratio # Mix the audio mixtures at (0 - mix_db_ratio) db
        self.peak_norm = peak_norm # Perfrom peak normalization of loaded audio if set True
        self.dynamic_mix = dynamic_mix # Dynamically create different mixtures if set True
        self.visual_pretrain = visual_pretrain
        _ = self.__len__()

    def __len__(self):
        # Dynamic creat the mixture list every epoch
        self.mix_lst = []
        if self.dynamic_mix:
        	random.shuffle(self.list_IDs)
        for i in range(0, int(np.floor(len(self.list_IDs)/self.mix_s))):
        	IDs =[]
        	for j in range(self.mix_s):
        		IDs.append(self.list_IDs[self.mix_s*i +j])
        	self.mix_lst.append(IDs)
        return len(self.mix_lst)

    def __getitem__(self, index):
        IDs = self.mix_lst[index]
        ID = IDs.pop()
        if self.visual_pretrain:
            v_tgt = np.load('%s%s.npy' % (self.path_v,ID))
        else:
            faces = []
            for i in range(76):
                face_path = '%s%s/%s_%02d.jpg' % (self.path_v,ID,ID,i) # for grid: 2nd ID[-6:]
                if os.path.exists(face_path):
                    image = cv2.imread(face_path)
                    # image = image[80:-80, 150:-150]  # for grid only
                    crop_img = cv2.resize(image,(160,160))
                    crop_img = Func.to_tensor(crop_img)
                    faces.append(crop_img)
                    # print(crop_img.shape)
            if(len(faces) < 75):
                face = faces[-1]
                for x in range(75-len(faces)):
                    faces.append(face)
            v_tgt = torch.stack(faces)
            # print(v_tgt.size())

        _, a_tgt = wavfile.read(self.path_a+str(ID)+'.wav')
        a_tgt = self._audio_norm(a_tgt)

        a_mix = a_tgt

        for ID in IDs:
        	_, a_noise = wavfile.read(self.path_a+str(ID)+'.wav')
        	a_mix = a_mix + self._audio_norm(a_noise)

        return a_mix, v_tgt, a_tgt

    def _audio_norm(self,audio):
    	if self.peak_norm:
    		audio = np.divide(audio, np.max(np.abs(audio)))
    	return audio * (10**(random.uniform(-self.mix_db_ratio/2,self.mix_db_ratio/2)/10))


def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr

def cal_si_snr_with_pit(source, estimate_source, reorder_source = False):
    """Calculate SI-SNR with PIT training.
    Args:
        All in torch tensors
        source: [B, C, T], B: batch size, C: no. of speakers, T: sequence length
        estimate_source: [B, C, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()

    # Step 1. Zero-mean norm
    zero_mean_target = source - torch.mean(source, dim=-1, keepdim=True)
    zero_mean_estimate = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C

    # Step 3: Reorder the estimated source
    if reorder_source:
    	reorder_estimate_source = _reorder_source(estimate_source, perms, max_snr_idx)
    	return max_snr, reorder_estimate_source
    else:
    	return max_snr

def _reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """

    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert x.size(0) <= self.pe.size(0), "max_len of positional encoding shorter than audio input length"
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    # import os
    # import tqdm
    print("Start testing")

    # #Test dataloader
    # path_visual = '/home/panzexu/workspace/av_speechseparation/datasets/orig/visual_emb/'
    # path_audio = '/home/panzexu/workspace/av_speechseparation/datasets/orig/audio/'
    # list_IDs = []
    # for path, dirs, files in os.walk(path_audio):
    # 	for filename in files:
    # 		list_IDs.append(os.path.splitext(filename)[0])
    # print(list_IDs)

    # # Grid dataloader
    # # path_visual = '/home/panzexu/datasets/Grid/Train/Face/'
    # # path_audio = '/home/panzexu/datasets/Grid/Train/Audio/'
    # # list_IDs = []
    # # for path, dirs, files in os.walk(path_audio):
    # # 	for filename in files:
    # # 		list_IDs.append(path[40:]+'/'+os.path.splitext(filename)[0])
    # # print(len(list_IDs))

    # params = {'batch_size': 2,
    #           'shuffle': False,
    #           'num_workers': 2}

    # training_set = avDataset(list_IDs,path_audio, path_visual, mix_s = 2, visual_pretrain = True)
    # training_generator = data.DataLoader(training_set, **params)

    # for epoch in range(2):
    #     for i, (a_mix, v_tgt, a_tgt) in enumerate(training_generator):
    #         print(v_tgt.size())
    #         print(a_mix.size())
    #         print(a_tgt.size())
    #         print('-'*15)
    #         # print(torch.eq(a_mix, a_tgt))
    #         pass



