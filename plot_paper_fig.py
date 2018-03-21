import os
import argparse
import logging
import yaml
import numpy as np
import time
from sklearn import metrics
import cPickle
import sys
import matplotlib.pyplot as plt
import mir_eval
from scipy import signal
import librosa
import glob

sys.path.append("/user/HS229/qk00006/my_code2015.5-/python/Hat")
import config as cfg
import prepare_data as pp_data
from data_generator import DataGenerator, DataGenerator2
import vad
# from tmp01 import _global_rank_pooling

def _global_rank_pooling(input, **kwargs):
    [n_songs, n_fmaps, n_time, n_freq] = input.shape
    input2d = input.reshape((n_songs*n_fmaps, n_time*n_freq))
    
    weight1d = kwargs['weight1d']
    
    out2d = T.sort(input2d, axis=-1)[:, ::-1] * weight1d / T.sum(weight1d)
    out4d = out2d.reshape((n_songs, n_fmaps, n_time, n_freq))
    return T.sum(out4d, axis=(2,3))

# import theano
# import theano.tensor as T
# from hat.models import Model
# from hat.layers.core import *
# from hat.layers.cnn import Conv2D
# from hat.layers.normalization import BN
# from hat.layers.pooling import MaxPooling2D
# from hat.callbacks import SaveModel, Validation
# from hat.preprocessing import sparse_to_categorical
# from hat.optimizers import SGD, Adam
# from hat import serializations
# import hat.backend as K
# from hat.metrics import tp_fn_fp_tn, prec_recall_fvalue

r = 0.9995
is_scale = True
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)

def _calc_feat(audio):
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    ham_win = np.hamming(n_window)
    
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode='magnitude') 
    x = x.T
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel(sr=fs, 
                                n_fft=n_window, 
                                n_mels=64, 
                                fmin=50, 
                                fmax=8000)
    x = np.dot(x, melW.T)
    x = np.log(x + 1e-8)
    x = x.astype(np.float32)
    return x
    
def _calc_spectrogram(audio):
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    ham_win = np.hamming(n_window)
    
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode='magnitude')
    return x.T

def get_inverse_W(W):
    return W / (np.sum(W, axis=0) + 1e-8)

# Plot waveform and spectrogram. 
def plot_fig1():
    workspace = cfg.workspace
    fs = cfg.sample_rate
    
    # Read audio. 
    audio_path = os.path.join(workspace, "mixed_audio/n_events=3/00292.mixed_20db.wav")
    (audio, _) = pp_data.read_audio(audio_path, fs)
    audio = audio / np.max(np.abs(audio))
    
    # Calculate log Mel. 
    x = _calc_feat(audio)
    print(x.shape)
    
    audio_path = os.path.join(workspace, "mixed_audio/n_events=3/00292.mixed_100db.wav")
    (audio_clean, _) = pp_data.read_audio(audio_path, fs)
    x_clean = _calc_feat(audio_clean)

    # Plot. 
    fig, axs = plt.subplots(3,1, sharex=False)
    axs[0].plot(audio)
    axs[0].axis([0, len(audio), -1, 1])
    axs[0].xaxis.set_ticks([])
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Waveform")
    
    axs[1].matshow(x.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].xaxis.set_ticks([])
    axs[1].set_ylabel('Mel freq. bin')
    axs[1].set_title("Log Mel spectrogram")
    
    tmp = (np.sign(x_clean - (-7.)) + 1) / 2.
    axs[2].matshow(tmp.T, origin='lower', aspect='auto', cmap='jet')
    axs[2].xaxis.set_ticks([0, 60, 120, 180, 239])
    axs[2].xaxis.tick_bottom()
    axs[2].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[2].set_xlabel("second")
    axs[2].xaxis.set_label_coords(1.1, -0.05)
    
    axs[2].yaxis.set_ticks([0, 16, 32, 48, 63])
    axs[2].yaxis.set_ticklabels([0, 16, 32, 48, 63])
    axs[2].set_ylabel('Mel freq. bin')
    
    axs[2].set_title("T-F segmentation mask")
    
    plt.tight_layout()
    plt.show()

# Not used. 
def plot_fig2():
    workspace = cfg.workspace
    events = cfg.events
    te_fold = cfg.te_fold

    # Load data. 
    snr = 20
    n_events = 3
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
        
    x = te_x
    at_y = te_at_y
    na_list = te_na_list
    
    # Load model. 
    md_path = os.path.join(workspace, "models/tmp01/n_events=3/fold=0/snr=20/md2000_iters.p")
    md = serializations.load(md_path)
    
    # 
    observe_nodes = [md.find_layer('seg_masks').output_]
    f_forward = md.get_observe_forward_func(observe_nodes)
    [seg_masks] = md.run_function(f_forward, x, batch_size=500, tr_phase=0.)
    print(seg_masks.shape)
    
    # for (i1, na) in enumerate(na_list):
    #     if '00292' in na:
    #         idx = i1
    # print(idx)
    
    for i1 in xrange(len(seg_masks)):
        print(na_list[i1])
        print(at_y[i1])
        fig, axs = plt.subplots(5,4, sharex=True)
        axs[0, 0].matshow(x[i1].T, origin='lower', aspect='auto', cmap='jet')
        for i2 in xrange(16):
            axs[i2/4 + 1, i2%4].matshow(seg_masks[i1,i2].T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
            axs[i2/4 + 1, i2%4].set_title(events[i2])
        plt.show()

# Plot fig 3. 
def plot_fig3(data_type, audio_idx):
    workspace = cfg.workspace
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    events = cfg.events
    te_fold = cfg.te_fold
    
    # Read audio. 
    audio_path = os.path.join(workspace, "mixed_audio/n_events=3/%s.mixed_20db.wav" % audio_idx)
    (audio, _) = pp_data.read_audio(audio_path, fs)
    
    # Calculate log Mel. 
    x = _calc_feat(audio)
    sp = _calc_spectrogram(audio)
    print(x.shape)
    
    # Plot. 
    fig, axs = plt.subplots(4, 4, sharex=False)
    
    # Plot log Mel spectrogram. 
    for i2 in xrange(16):
        axs[i2/4, i2%4].set_visible(False)
    
    axs[0,0].matshow(x.T, origin='lower', aspect='auto', cmap='jet')
    axs[0,0].xaxis.set_ticks([0, 60, 120, 180, 239])
    axs[0,0].xaxis.tick_bottom()
    axs[0,0].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[0,0].set_xlabel("time (s)")
    # axs[0,0].xaxis.set_label_coords(1.12, -0.05)
    
    axs[0,0].yaxis.set_ticks([0, 16, 32, 48, 63])
    axs[0,0].yaxis.set_ticklabels([0, 16, 32, 48, 63])
    axs[0,0].set_ylabel('Mel freq. bin')
    
    axs[0,0].set_title("Log Mel spectrogram")
    axs[0,0].set_visible(True)
    
    # Plot spectrogram. 
    axs[0,2].matshow(np.log(sp.T), origin='lower', aspect='auto', cmap='jet')
    axs[0,2].xaxis.set_ticks([0, 60, 120, 180, 239])
    axs[0,2].xaxis.tick_bottom()
    axs[0,2].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[0,2].set_xlabel("time (s)")
    # axs[0,2].xaxis.set_label_coords(1.12, -0.05)
    
    axs[0,2].yaxis.set_ticks([0, 128, 256, 384, 512])
    axs[0,2].yaxis.set_ticklabels([0, 128, 256, 384, 512])
    axs[0,2].set_ylabel('FFT freq. bin')
    
    axs[0,2].set_title("Spectrogram (Plot in log)")
    axs[0,2].set_visible(True)
    
    # plt.tight_layout()
    plt.show()
    
    # Load data. 
    snr = 20
    n_events = 3
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
        
    if data_type == "train":
        x = tr_x
        at_y = tr_at_y
        sed_y = tr_sed_y
        na_list = tr_na_list
    elif data_type == "test":
        x = te_x
        at_y = te_at_y
        sed_y = te_sed_y
        na_list = te_na_list

    for (i1, na) in enumerate(na_list):
        if audio_idx in na:
            idx = i1
    print(idx)
    
    # Plot seg masks. 
    preds_dir = os.path.join(workspace, "preds", "tmp01", 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)

    at_probs_list, seg_masks_list = [], []
    bgn_iter, fin_iter, interval = 2000, 3001, 200
    for iter in xrange(bgn_iter, fin_iter, interval):
        seg_masks_path = os.path.join(preds_dir, "md%d_iters" % iter, "seg_masks.p")
        seg_masks = cPickle.load(open(seg_masks_path, 'rb'))
        seg_masks_list.append(seg_masks)
    seg_masks = np.mean(seg_masks_list, axis=0) # (n_clips, n_classes, n_time, n_freq)
    
    print(at_y[idx])
    fig, axs = plt.subplots(4,4, sharex=True)
    for i2 in xrange(16):
        axs[i2/4, i2%4].matshow(seg_masks[idx,i2].T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
        # axs[i2/4, i2%4].set_title(events[i2])
        axs[i2/4, i2%4].xaxis.set_ticks([])
        axs[i2/4, i2%4].yaxis.set_ticks([])
        axs[i2/4, i2%4].set_xlabel('time')
        axs[i2/4, i2%4].set_ylabel('Mel freq. bin')
        
    plt.show()
    
    # Plot SED probs. 
    sed_probs = np.mean(seg_masks[idx], axis=-1)    # (n_classes, n_time)
    fig, axs = plt.subplots(4,4, sharex=False)
    for i2 in xrange(16):
        axs[i2/4, i2%4].set_visible(False)
    axs[0, 0].matshow(sed_probs, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
    # axs[0, 0].xaxis.set_ticks([0, 60, 120, 180, 239])
    # axs[0, 0].xaxis.tick_bottom()
    # axs[0, 0].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[0, 0].xaxis.set_ticks([])
    # axs[0, 0].set_xlabel('time (s)')
    axs[0, 0].yaxis.set_ticks(xrange(len(events)))
    axs[0, 0].yaxis.set_ticklabels(events)
    for tick in axs[0, 0].yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    axs[0, 0].set_visible(True)
    
    axs[1, 0].matshow(sed_y[idx].T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
    # axs[1, 0].xaxis.set_ticks([])
    axs[1, 0].xaxis.set_ticks([0, 60, 120, 180, 239])
    axs[1, 0].xaxis.tick_bottom()
    axs[1, 0].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 0].yaxis.set_ticks(xrange(len(events)))
    axs[1, 0].yaxis.set_ticklabels(events)
    for tick in axs[1, 0].yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    axs[1, 0].set_visible(True)
    plt.show()

    # Masks of spectragram. 
    melW = librosa.filters.mel(sr=fs, 
                                n_fft=cfg.n_window, 
                                n_mels=64, 
                                fmin=0., 
                                fmax=fs / 2)
    inverse_melW = get_inverse_W(melW)
    
    spec_masks = np.dot(seg_masks[idx], inverse_melW)  # (n_classes, n_time, 513)
    
    fig, axs = plt.subplots(4,4, sharex=True)
    for i2 in xrange(16):
        axs[i2/4, i2%4].matshow(spec_masks[i2].T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
        # axs[i2/4, i2%4].set_title(events[i2])
        axs[i2/4, i2%4].xaxis.set_ticks([])
        axs[i2/4, i2%4].yaxis.set_ticks([])
        axs[i2/4, i2%4].set_xlabel('time')
        axs[i2/4, i2%4].set_ylabel('FFT freq. bin')
    plt.show()
    
    # Masked spectrogram. 
    masked_sps = spec_masks * sp[None, :, :]
    fig, axs = plt.subplots(4,4, sharex=True)
    for i2 in xrange(16):
        axs[i2/4, i2%4].matshow(np.log(masked_sps[i2].T), origin='lower', aspect='auto', cmap='jet')
        # axs[i2/4, i2%4].set_title(events[i2])
        axs[i2/4, i2%4].xaxis.set_ticks([])
        axs[i2/4, i2%4].yaxis.set_ticks([])
        axs[i2/4, i2%4].set_xlabel('time')
        axs[i2/4, i2%4].set_ylabel('FFT freq. bin')
    plt.show()
    
    # GT mask
    (stereo_audio, _) = pp_data.read_stereo_audio(audio_path, target_fs=fs)
    event_audio = stereo_audio[:, 0]
    noise_audio = stereo_audio[:, 1]
    mixed_audio = event_audio + noise_audio
    
    ham_win = np.hamming(n_window)
    mixed_cmplx_sp = pp_data.calc_sp(mixed_audio, fs, ham_win, n_window, n_overlap)
    mixed_sp = np.abs(mixed_cmplx_sp)
    event_sp = np.abs(pp_data.calc_sp(event_audio, fs, ham_win, n_window, n_overlap))
    noise_sp = np.abs(pp_data.calc_sp(noise_audio, fs, ham_win, n_window, n_overlap))
    
    db = -5.
    gt_mask = (np.sign(20 * np.log10(event_sp / noise_sp) - db) + 1.) / 2.  # (n_time, n_freq)
    fig, axs = plt.subplots(4,4, sharex=True)
    for i2 in xrange(16):
        ind_gt_mask = gt_mask * sed_y[idx, :, i2][:, None]
        axs[i2/4, i2%4].matshow(ind_gt_mask.T, origin='lower', aspect='auto', cmap='jet')
        # axs[i2/4, i2%4].set_title(events[i2])
        axs[i2/4, i2%4].xaxis.set_ticks([])
        axs[i2/4, i2%4].yaxis.set_ticks([])
        axs[i2/4, i2%4].set_xlabel('time')
        axs[i2/4, i2%4].set_ylabel('FFT freq. bin')
    plt.show()
    
# Plot fig 4. 
def plot_fig4(data_type, audio_idx):
    workspace = cfg.workspace
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    events = cfg.events
    te_fold = cfg.te_fold
    
    # Read audio. 
    audio_path = os.path.join(workspace, "mixed_audio/n_events=3/%s.mixed_20db.wav" % audio_idx)
    (audio, _) = pp_data.read_audio(audio_path, fs)
    
    # Calculate log Mel. 
    x = _calc_feat(audio)
    sp = _calc_spectrogram(audio)
    print(x.shape)
    
    # Plot. 
    fig, axs = plt.subplots(4, 4, sharex=False)
    
    # Plot log Mel spectrogram. 
    for i2 in xrange(16):
        axs[i2/4, i2%4].set_visible(False)
    
    axs[0,0].matshow(x.T, origin='lower', aspect='auto', cmap='jet')
    axs[0,0].xaxis.set_ticks([0, 60, 120, 180, 239])
    axs[0,0].xaxis.tick_bottom()
    axs[0,0].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[0,0].set_xlabel("time (s)")
    # axs[0,0].xaxis.set_label_coords(1.12, -0.05)
    
    axs[0,0].yaxis.set_ticks([0, 16, 32, 48, 63])
    axs[0,0].yaxis.set_ticklabels([0, 16, 32, 48, 63])
    axs[0,0].set_ylabel('Mel freq. bin')
    
    axs[0,0].set_title("Log Mel spectrogram")
    axs[0,0].set_visible(True)
    
    # Plot spectrogram. 
    axs[0,2].matshow(np.log(sp.T + 1.), origin='lower', aspect='auto', cmap='jet')
    axs[0,2].xaxis.set_ticks([0, 60, 120, 180, 239])
    axs[0,2].xaxis.tick_bottom()
    axs[0,2].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
    axs[0,2].set_xlabel("time (s)")
    # axs[0,2].xaxis.set_label_coords(1.12, -0.05)
    
    axs[0,2].yaxis.set_ticks([0, 128, 256, 384, 512])
    axs[0,2].yaxis.set_ticklabels([0, 128, 256, 384, 512])
    axs[0,2].set_ylabel('FFT freq. bin')
    
    axs[0,2].set_title("Spectrogram")
    axs[0,2].set_visible(True)
    
    # plt.tight_layout()
    plt.show()
    
    # Load data. 
    snr = 20
    n_events = 3
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
        
    if data_type == "train":
        x = tr_x
        at_y = tr_at_y
        sed_y = tr_sed_y
        na_list = tr_na_list
    elif data_type == "test":
        x = te_x
        at_y = te_at_y
        sed_y = te_sed_y
        na_list = te_na_list

    for (i1, na) in enumerate(na_list):
        if audio_idx in na:
            idx = i1
    print(idx)
    
    # GT mask
    (stereo_audio, _) = pp_data.read_stereo_audio(audio_path, target_fs=fs)
    event_audio = stereo_audio[:, 0]
    noise_audio = stereo_audio[:, 1]
    mixed_audio = event_audio + noise_audio
    
    ham_win = np.hamming(n_window)
    mixed_cmplx_sp = pp_data.calc_sp(mixed_audio, fs, ham_win, n_window, n_overlap)
    mixed_sp = np.abs(mixed_cmplx_sp)
    event_sp = np.abs(pp_data.calc_sp(event_audio, fs, ham_win, n_window, n_overlap))
    noise_sp = np.abs(pp_data.calc_sp(noise_audio, fs, ham_win, n_window, n_overlap))
    
    db = -5.
    gt_mask = (np.sign(20 * np.log10(event_sp / noise_sp) - db) + 1.) / 2.  # (n_time, n_freq)
    fig, axs = plt.subplots(4,4, sharex=True)
    for i2 in xrange(16):
        ind_gt_mask = gt_mask * sed_y[idx, :, i2][:, None]
        axs[i2/4, i2%4].matshow(ind_gt_mask.T, origin='lower', aspect='auto', cmap='jet')
        # axs[i2/4, i2%4].set_title(events[i2])
        axs[i2/4, i2%4].xaxis.set_ticks([])
        axs[i2/4, i2%4].yaxis.set_ticks([])
        axs[i2/4, i2%4].set_xlabel('time')
        axs[i2/4, i2%4].set_ylabel('FFT freq. bin')
    plt.show()
    
    for filename in ["tmp01", "tmp02", "tmp03"]:
        # Plot up sampled seg masks. 
        preds_dir = os.path.join(workspace, "preds", filename, 
                            "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)
    
        at_probs_list, seg_masks_list = [], []
        bgn_iter, fin_iter, interval = 2000, 3001, 200
        for iter in xrange(bgn_iter, fin_iter, interval):
            seg_masks_path = os.path.join(preds_dir, "md%d_iters" % iter, "seg_masks.p")
            seg_masks = cPickle.load(open(seg_masks_path, 'rb'))
            seg_masks_list.append(seg_masks)
        seg_masks = np.mean(seg_masks_list, axis=0) # (n_clips, n_classes, n_time, n_freq)
        
        print(at_y[idx])
        
        melW = librosa.filters.mel(sr=fs, 
                                    n_fft=cfg.n_window, 
                                    n_mels=64, 
                                    fmin=0., 
                                    fmax=fs / 2)
        inverse_melW = get_inverse_W(melW)
        
        spec_masks = np.dot(seg_masks[idx], inverse_melW)  # (n_classes, n_time, 513)
        
        fig, axs = plt.subplots(4,4, sharex=True)
        for i2 in xrange(16):
            axs[i2/4, i2%4].matshow(spec_masks[i2].T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
            # axs[i2/4, i2%4].set_title(events[i2])
            axs[i2/4, i2%4].xaxis.set_ticks([])
            axs[i2/4, i2%4].yaxis.set_ticks([])
            axs[i2/4, i2%4].set_xlabel('time')
            axs[i2/4, i2%4].set_ylabel('FFT freq. bin')
        fig.suptitle(filename)
        plt.show()
        
        # Plot SED probs. 
        sed_probs = np.mean(seg_masks[idx], axis=-1)    # (n_classes, n_time)
        fig, axs = plt.subplots(4,4, sharex=False)
        for i2 in xrange(16):
            axs[i2/4, i2%4].set_visible(False)
        axs[0, 0].matshow(sed_probs, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
        # axs[0, 0].xaxis.set_ticks([0, 60, 120, 180, 239])
        # axs[0, 0].xaxis.tick_bottom()
        # axs[0, 0].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
        axs[0, 0].xaxis.set_ticks([])
        # axs[0, 0].set_xlabel('time (s)')
        axs[0, 0].yaxis.set_ticks(xrange(len(events)))
        axs[0, 0].yaxis.set_ticklabels(events)
        for tick in axs[0, 0].yaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        axs[0, 0].set_visible(True)
        
        axs[1, 0].matshow(sed_y[idx].T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='jet')
        # axs[1, 0].xaxis.set_ticks([])
        axs[1, 0].xaxis.set_ticks([0, 60, 120, 180, 239])
        axs[1, 0].xaxis.tick_bottom()
        axs[1, 0].xaxis.set_ticklabels(np.arange(0, 10.1, 2.5))
        axs[1, 0].set_xlabel('time (s)')
        axs[1, 0].yaxis.set_ticks(xrange(len(events)))
        axs[1, 0].yaxis.set_ticklabels(events)
        for tick in axs[1, 0].yaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        axs[1, 0].set_visible(True)
        fig.suptitle(filename)
        plt.show()

def plot_fig5(data_type, audio_idx):
    workspace = cfg.workspace
    fs = cfg.sample_rate
    
    # Read audio. 
    audio_path = os.path.join(workspace, "mixed_audio/n_events=3/%s.mixed_20db.wav" % audio_idx)
    (audio, _) = pp_data.read_stereo_audio(audio_path, fs)
    event_audio = audio[:, 0]
    noise_audio = audio[:, 1]
    mixed_audio = (event_audio + noise_audio) / 2
    
    event_audio_1 = np.zeros_like(event_audio)
    event_audio_1[0 : int(fs * 2.5)] = event_audio[0 : int(fs * 2.5)]
    event_audio_2 = np.zeros_like(event_audio)
    event_audio_2[int(fs * 2.5) : int(fs * 5.)] = event_audio[int(fs * 2.5) : int(fs * 5.)]
    event_audio_3 = np.zeros_like(event_audio)
    event_audio_3[int(fs * 5.) : int(fs * 7.5)] = event_audio[int(fs * 5.) : int(fs * 7.5)]
    
    sep_dir = "/vol/vssp/msos/qk/workspaces/weak_source_separation/dcase2013_task2/sep_audio/tmp01/n_events=3/fold=0/snr=20"
    sep_paths = glob.glob(os.path.join(sep_dir, "%s*" % audio_idx))
    
    print([os.path.basename(e) for e in sep_paths])
    (sep_event_audio_1, _) = pp_data.read_audio(sep_paths[3])
    (sep_event_audio_2, _) = pp_data.read_audio(sep_paths[0])
    (sep_event_audio_3, _) = pp_data.read_audio(sep_paths[1])
    (sep_noise_audio, _) = pp_data.read_audio(sep_paths[2])
    
    fig, axs = plt.subplots(5,2, sharex=True)
    axs[0, 0].plot(mixed_audio)
    axs[1, 0].plot(event_audio_1)
    axs[2, 0].plot(event_audio_2)
    axs[3, 0].plot(event_audio_3)
    axs[4, 0].plot(noise_audio)
    
    axs[1, 1].plot(sep_event_audio_1)
    axs[2, 1].plot(sep_event_audio_2)
    axs[3, 1].plot(sep_event_audio_3)
    axs[4, 1].plot(sep_noise_audio)
    
    T = len(noise_audio)
    for i1 in xrange(5):
        for i2 in xrange(2):
            axs[i1, i2].axis([0, T, -1, 1])
            axs[i1, i2].xaxis.set_ticks([])
            axs[i1, i2].yaxis.set_ticks([])
            axs[i1, i2].set_ylabel("Amplitude")
    plt.show()

   
def find():
    workspace = cfg.workspace
    audio_dir = os.path.join(workspace, "mixed_audio/n_events=3")
    names = [na for na in os.listdir(audio_dir) if na.endswith('.yaml')]
    print(names)
    
    for na in names:
        yaml_path = os.path.join(audio_dir, na)
        with open(yaml_path, 'r') as f:
            data = yaml.load(f)
            curr_events = [e['event'] for e in data['events']]
            if 'alert' in curr_events and 'speech' in curr_events:
                print(data)
                raw_input()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('fig1')
    parser_b = subparsers.add_parser('fig2')
    parser_b = subparsers.add_parser('fig3')
    parser_b = subparsers.add_parser('fig4')
    parser_b = subparsers.add_parser('fig5')
    parser_b = subparsers.add_parser('find')
    
    args = parser.parse_args()
    if args.mode == 'fig1':
        plot_fig1()
    elif args.mode == 'fig2':
        plot_fig2()
    elif args.mode == 'fig3':
        # data_type = "test" # test
        # audio_idx = "00026" # 00292
        data_type = "train"
        audio_idx = "00292"
        plot_fig3(data_type, audio_idx)
    elif args.mode == 'fig4':
        data_type = "test" # test
        audio_idx = "00026" # 00292
        plot_fig4(data_type, audio_idx)
    elif args.mode == 'fig5':
        data_type = "test" # test
        audio_idx = "00026" # 00292
        plot_fig5(data_type, audio_idx)
    elif args.mode == 'find':
        find()