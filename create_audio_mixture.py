import os
import numpy as np 
import argparse
import tqdm
import scipy.io.wavfile as wavfile

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
	"""
	Write wav files in int16, support single/multi-channel
	"""
	# for multi-channel, accept ndarray [Nsamples, Nchannels]
	if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
		samps = np.transpose(samps)
		samps = np.squeeze(samps)
	# same as MATLAB and kaldi
	if normalize:
		samps = samps * MAX_INT16
		samps = samps.astype(np.int16)
	fdir = os.path.dirname(fname)
	if fdir and not os.path.exists(fdir):
		os.makedirs(fdir)
	# NOTE: librosa 0.6.0 seems could not write non-float narray
	#       so use scipy.io.wavfile instead
	wavfile.write(fname, sampling_rate, samps)


def read_wav(fname, normalize=True):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    sampling_rate, samps_int16 = wavfile.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    return sampling_rate, samps


_, audio_tgt=read_wav('/home/panzexu/Download/audio_mixture/female.wav')
target_power = np.linalg.norm(audio_tgt, 2)**2 / audio_tgt.size

_, audio = read_wav('/home/panzexu/Download/audio_mixture/male.wav')
intef_power = np.linalg.norm(audio, 2)**2 / audio.size

for snr in [-40, -35, -30, -25, -20, -15, -10, 0]:

	scalar = (10**(float(0 - snr)/20)) * np.sqrt(target_power/intef_power)
	audio_int = audio * scalar

	audio_mix = audio_tgt + audio_int

	audio_mix = np.divide(audio_mix, np.max(np.abs(audio_mix)))
	write_wav('/home/panzexu/Download/audio_mixture/mix_'+ str(int(snr)) +'.wav', audio_mix)