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
import librosa
from mir_eval.separation import bss_eval_sources

sys.path.append("/user/HS229/qk00006/my_code2015.5-/python/Hat")
import config as cfg
import prepare_data as pp_data
from data_generator import DataGenerator, DataGenerator2
import vad
from spectrogram_to_wave import spectrogram_to_wave

import theano
import theano.tensor as T
from hat.models import Model
from hat.layers.core import *
from hat.layers.cnn import Conv2D
from hat.layers.normalization import BN
from hat.layers.pooling import MaxPooling2D
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat import serializations
import hat.backend as K
from hat.metrics import *

r = 0.9995
is_scale = True
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)

at_thres = 0.2
sed_thres = 0.5
seg_thres = 0.7

def _max_along_time(input, **kwargs):
    return T.max(input, axis=1)

def train(args):
    workspace = cfg.workspace
    te_fold = cfg.te_fold
    n_events = args.n_events
    snr = args.snr

    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
        
    print(tr_x.shape, tr_at_y.shape)
    print(te_x.shape, te_at_y.shape)
    (_, n_time, n_freq) = tr_x.shape
    n_out = len(cfg.events)
    
    if False:
        for e in tr_x:
            plt.matshow(e.T, origin='lower', aspect='auto')
            plt.show()
    
    # Build model. 
    lay_in = InputLayer(in_shape=(n_time, n_freq,))
    
    a = Dense(500, act='relu')(lay_in)
    a = Dropout(p_drop=0.2)(a)
    a = Dense(500, act='relu')(a)
    a = Dropout(p_drop=0.2)(a)
    a = Dense(500, act='relu')(a)
    a = Dropout(p_drop=0.2)(a)
    a = Dense(n_out, act='sigmoid', name='detect')(a)
    
    a8 = Lambda(_max_along_time)(a)
    
    md = Model([lay_in], [a8])
    md.compile()
    md.summary(is_logging=True)
    
    # Callbacks. 
    md_dir = os.path.join(workspace, "models", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)
    pp_data.create_folder(md_dir)
    save_model = SaveModel(md_dir, call_freq=50, type='iter', is_logging=True)
    validation = Validation(te_x=te_x, te_y=te_at_y, batch_size=50, call_freq=50, metrics=['binary_crossentropy'], dump_path=None, is_logging=True)
    
    callbacks = [save_model, validation]
    
    # Generator. 
    tr_gen = DataGenerator(batch_size=32, type='train')
    eva_gen = DataGenerator2(batch_size=32, type='test')
    
    # Train. 
    loss_ary = []
    t1 = time.time()
    optimizer = Adam(1e-4)
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_at_y]):
        if md.iter_ % 50 == 0:
            logging.info("iter: %d tr_loss: %f time: %s" % (md.iter_, np.mean(loss_ary), time.time() - t1, ))
            t1 = time.time()
            loss_ary = []
        # if md.iter_ % 200 == 0:
            # write_out_at_sed(md, eva_gen, f_forward, te_x, te_at_y, te_sed_y, n_events, snr, te_fold)
        if md.iter_ == 5001:
            break
        loss = md.train_on_batch(batch_x, batch_y, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks)
        loss_ary.append(loss)

        
def recognize(args):
    workspace = cfg.workspace
    events = cfg.events
    n_events = args.n_events
    snr = args.snr
    md_na = args.model_name
    lb_to_ix = cfg.lb_to_ix
    n_out = len(cfg.events)
    te_fold = cfg.te_fold
    
    md_path = os.path.join(workspace, "models", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr, md_na)
    md = serializations.load(md_path)
    
    # Load data. 
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
    at_gts = te_at_y
    sed_gts = te_sed_y
    na_list = te_na_list
        
    # Recognize. 
    [at_pds] = md.predict(x) # (N, 16)
    
    observe_nodes = [md.find_layer('detect').output_]
    f_forward = md.get_observe_forward_func(observe_nodes)
    [seg_masks] = md.run_function(f_forward, x, batch_size=500, tr_phase=0.)    # (n_clips, n_time, n_out)
    seg_masks = np.transpose(seg_masks, (0, 2, 1))[:, :, :, np.newaxis]
    
    # Dump to pickle. 
    out_dir = os.path.join(workspace, "preds", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr, 
                          os.path.splitext(md_na)[0])
    pp_data.create_folder(out_dir)
    out_at_path = os.path.join(out_dir, "at_probs.p")
    out_seg_masks_path = os.path.join(out_dir, "seg_masks.p")
    
    cPickle.dump(at_pds, open(out_at_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(seg_masks, open(out_seg_masks_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Print stats. 
    sed_pds = np.mean(seg_masks, axis=-1)   # (N, n_out, n_time)
    sed_pds = np.transpose(sed_pds, (0, 2, 1))  # (N, n_time, n_out)
    print_stats(at_pds, at_gts, sed_pds, sed_gts)
    
def get_stats(args, bgn_iter, fin_iter, interval):
    workspace = cfg.workspace
    events = cfg.events
    te_fold = cfg.te_fold
    n_events = args.n_events
    snr = args.snr
    
    # Load ground truth data. 
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
        
    at_gts = te_at_y
    sed_gts = te_sed_y
    
    # Load and sum 
    preds_dir = os.path.join(workspace, "preds", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)

    at_probs_list, seg_masks_list = [], []
    for iter in xrange(bgn_iter, fin_iter, interval):
        at_probs_path = os.path.join(preds_dir, "md%d_iters" % iter, "at_probs.p")
        at_probs = cPickle.load(open(at_probs_path, 'rb'))
        at_probs_list.append(at_probs)
        seg_masks_path = os.path.join(preds_dir, "md%d_iters" % iter, "seg_masks.p")
        seg_masks = cPickle.load(open(seg_masks_path, 'rb'))
        seg_masks_list.append(seg_masks)
    at_probs = np.mean(at_probs_list, axis=0)   # (n_clips, n_classes)
    seg_masks = np.mean(seg_masks_list, axis=0) # (n_clips, n_classes, n_time, n_freq)
    sed_probs = np.mean(seg_masks, axis=-1).transpose(0, 2, 1)  # (n_clips, n_time, n_classes)
    
    print_stats(at_probs, at_gts, sed_probs, sed_gts)
    
def get_inverse_W(W):
    return W / (np.sum(W, axis=0) + 1e-8)
    
def hit_fa(seg_mask, event_sp, noise_sp, sed_y, thres, inside_only):
    """
    Args:
      seg_mask: (n_time, n_freq)
      event_sp: (n_time, n_freq)
      noise_sp: (n_time, n_freq)
      sed_y: (n_time,)
    """
    db = 0.
    gt_mask = (np.sign(20 * np.log10(event_sp / noise_sp) - db) + 1.) / 2.
    
    if inside_only:    
        active_locts = np.where(sed_y==1)[0]
        onset = active_locts[0]
        offset = active_locts[-1] + 1
        in_seg_mask = seg_mask[onset : offset]
        in_gt_mask = gt_mask[onset : offset]
    else:
        in_seg_mask = seg_mask
        in_gt_mask = gt_mask        
      
    if False:
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].matshow(in_gt_mask.T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(in_seg_mask.T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
    
    (tp, fn, fp, tn) = tp_fn_fp_tn(in_seg_mask.flatten(), in_gt_mask.flatten(), thres=thres, average=None)
    hit = float(tp) / (tp + fn + 1e-8)
    fa = float(fp) / (fp + tn + 1e-8)
    return hit, fa
    
def fvalue_iou(seg_mask, event_sp, noise_sp, sed_y, thres, inside_only):
    """
    Args:
      seg_mask: (n_time, n_freq)
      event_sp: (n_time, n_freq)
      noise_sp: (n_time, n_freq)
      sed_y: (n_time,)
    """
    db = -5.
    gt_mask = (np.sign(20 * np.log10(event_sp / noise_sp) - db) + 1.) / 2.
    
    if inside_only:
        active_locts = np.where(sed_y==1)[0]
        onset = active_locts[0]
        offset = active_locts[-1] + 1
        in_seg_mask = seg_mask[onset : offset]
        in_gt_mask = gt_mask[onset : offset]
    else:
        in_seg_mask = seg_mask
        in_gt_mask = gt_mask
    
    # print(in_gt_mask.flatten())
    auc = metrics.roc_auc_score(in_gt_mask.flatten(), in_seg_mask.flatten(), average=None)
    (tp, fn, fp, tn) = tp_fn_fp_tn(in_seg_mask.flatten(), in_gt_mask.flatten(), thres=thres, average=None)
    eps = 1e-8
    fvalue = float(2. * tp) / float(2. * tp + fp + fn + eps)
    iou = float(tp) / float(tp + fn + fp + eps)
    return fvalue, auc, iou, tp, fn, fp
    
def separate(args, bgn_iter, fin_iter, interval):
    workspace = cfg.workspace
    events = cfg.events
    te_fold = cfg.te_fold
    n_events = args.n_events
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    clip_duration = cfg.clip_duration
    snr = args.snr
    
    # Load ground truth data. 
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
    
    at_y = te_at_y
    sed_y = te_sed_y
    na_list = te_na_list
    
    
    # Load and sum 
    preds_dir = os.path.join(workspace, "preds", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)

    at_probs_list, seg_masks_list = [], []
    for iter in xrange(bgn_iter, fin_iter, interval):
        seg_masks_path = os.path.join(preds_dir, "md%d_iters" % iter, "seg_masks.p")
        seg_masks = cPickle.load(open(seg_masks_path, 'rb'))
        seg_masks_list.append(seg_masks)
    seg_masks = np.mean(seg_masks_list, axis=0) # (n_clips, n_classes, n_time, n_freq)
    
    print(seg_masks.shape)
    
    # 
    audio_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    
    sep_dir = os.path.join(workspace, "sep_audio", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)
    pp_data.create_folder(sep_dir)
    
    ham_win = np.hamming(n_window)
    recover_scaler = np.sqrt((ham_win**2).sum())
    melW = librosa.filters.mel(sr=fs, 
                                n_fft=n_window, 
                                n_mels=64, 
                                fmin=0., 
                                fmax=fs / 2)
    inverse_melW = get_inverse_W(melW)  # (64, 513)
    
    seg_stats = {}
    for e in events:
        seg_stats[e] = {'fvalue': [], 'auc': [], 'iou': [], 'hit': [], 'fa': [], 'tp': [], 'fn': [], 'fp': []}
    
    cnt = 0
    for (i1, na) in enumerate(na_list):
        bare_na = os.path.splitext(na)[0]
        audio_path = os.path.join(audio_dir, "%s.wav" % bare_na)
        (stereo_audio, _) = pp_data.read_stereo_audio(audio_path, target_fs=fs)
        event_audio = stereo_audio[:, 0]
        noise_audio = stereo_audio[:, 1]
        mixed_audio = event_audio + noise_audio
        
        mixed_cmplx_sp = pp_data.calc_sp(mixed_audio, fs, ham_win, n_window, n_overlap)
        mixed_sp = np.abs(mixed_cmplx_sp)
        event_sp = np.abs(pp_data.calc_sp(event_audio, fs, ham_win, n_window, n_overlap))
        noise_sp = np.abs(pp_data.calc_sp(noise_audio, fs, ham_win, n_window, n_overlap))

        sm = seg_masks[i1]  # (n_classes, n_time, n_freq)
        sm_upsampled = np.dot(sm, inverse_melW)  # (n_classes, n_time, 513)
        
        print(na)
        
        # Write out separated events. 
        for j1 in xrange(len(events)):
            if at_y[i1][j1] == 1:
                (fvalue, auc, iou, tp, fn, fp) = fvalue_iou(sm_upsampled[j1], event_sp, noise_sp, sed_y[i1, :, j1], seg_thres, inside_only=True)
                (hit, fa) = hit_fa(sm_upsampled[j1], event_sp, noise_sp, sed_y[i1, :, j1], seg_thres, inside_only=True)
                seg_stats[events[j1]]['fvalue'].append(fvalue)
                seg_stats[events[j1]]['auc'].append(auc)
                seg_stats[events[j1]]['iou'].append(iou)
                seg_stats[events[j1]]['hit'].append(hit)
                seg_stats[events[j1]]['fa'].append(fa)
                seg_stats[events[j1]]['tp'].append(tp)
                seg_stats[events[j1]]['fn'].append(fn)
                seg_stats[events[j1]]['fp'].append(fp)
      
                sep_event_sp = sm_upsampled[j1] * mixed_sp
                sep_event_s = spectrogram_to_wave.recover_wav(sep_event_sp, mixed_cmplx_sp, n_overlap=n_overlap, winfunc=np.hamming, wav_len=int(fs * clip_duration))
                sep_event_s *= recover_scaler
                                
                out_event_audio_path = os.path.join(sep_dir, "%s.%s.wav" % (bare_na, events[j1]))
                pp_data.write_audio(out_event_audio_path, sep_event_s, fs)
       
        # Write out separated noise. 
        sm_noise_upsampled = np.clip(1. - np.sum(sm_upsampled, axis=0), 0., 1.)
        sep_noise_sp = sm_noise_upsampled * mixed_sp
        sep_noise_s = spectrogram_to_wave.recover_wav(sep_noise_sp, mixed_cmplx_sp, n_overlap=n_overlap, winfunc=np.hamming, wav_len=int(fs * clip_duration))
        sep_noise_s *= recover_scaler
        out_noise_audio_path = os.path.join(sep_dir, "%s.noise.wav" % bare_na)
        pp_data.write_audio(out_noise_audio_path, sep_noise_s, fs)
        
       
        cnt += 1
        # if cnt == 2: break
        
    
    fvalues, aucs, ious, hits, fas, tps, fns, fps = [], [], [], [], [], [], [], []
    for e in events:
        fvalues.append(np.mean(seg_stats[e]['fvalue']))
        ious.append(np.mean(seg_stats[e]['iou']))
        aucs.append(np.mean(seg_stats[e]['auc']))
        hits.append(np.mean(seg_stats[e]['hit']))
        fas.append(np.mean(seg_stats[e]['fa']))
        tps.append(np.mean(seg_stats[e]['tp']))
        fns.append(np.mean(seg_stats[e]['fn']))
        fps.append(np.mean(seg_stats[e]['fp']))
    
    logging.info("%sfvalue\tauc\tiou\tHit\tFa\tHit-Fa\tTP\tFN\tFP" % ("".ljust(16)))
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Avg. of each".ljust(16), np.mean(fvalues), np.mean(aucs), np.mean(ious), np.mean(hits), np.mean(fas), np.mean(hits) - np.mean(fas), np.mean(tps), np.mean(fns), np.mean(fps)))    
    for i1 in xrange(len(events)):
        logging.info("%s%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (events[i1].ljust(16), fvalues[i1], aucs[i1], ious[i1], hits[i1], fas[i1], hits[i1] - fas[i1], tps[i1], fns[i1], fps[i1]))
    
        
def sdr_sir_sar(gt_audio, sep_audio, sed_y, inside_only):
    """gt_audio, sep_audio.shape: (n_channels, n_samples)
    """
    if inside_only:
        n_step = cfg.n_step
        active_locts = np.where(sed_y==1)[0]
        onset = int(round(active_locts[0] * n_step))
        offset = int(round((active_locts[-1] + 1) * n_step))
        in_gt_audio = gt_audio[:, onset : offset]
        in_sep_audio = sep_audio[:, onset : offset]
        (sdr, sir, sar, perm) = bss_eval_sources(in_gt_audio, in_sep_audio, compute_permutation=False)
        return sdr, sir, sar
    else:
        (sdr, sir, sar, perm) = bss_eval_sources(gt_audio, sep_audio, compute_permutation=False)
        return sdr, sir, sar
        
def evaluate_separation(args):
    workspace = cfg.workspace
    events = cfg.events
    te_fold = cfg.te_fold
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    clip_duration = cfg.clip_duration
    n_events = args.n_events
    snr = args.snr
    
    # Load ground truth data. 
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        snr=snr, 
        is_scale=is_scale)
    
    at_y = te_at_y
    sed_y = te_sed_y
    na_list = te_na_list
    
    audio_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    
    sep_dir = os.path.join(workspace, "sep_audio", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr)
                        
    sep_stats = {}
    for e in events:
        sep_stats[e] = {'sdr': [], 'sir': [], 'sar': []}
                          
    cnt = 0
    for (i1, na) in enumerate(na_list):
        bare_na = os.path.splitext(na)[0]
        gt_audio_path = os.path.join(audio_dir, "%s.wav" % bare_na)
        (stereo_audio, _) = pp_data.read_stereo_audio(gt_audio_path, target_fs=fs)
        gt_event_audio = stereo_audio[:, 0]
        gt_noise_audio = stereo_audio[:, 1]
        
        print(na)
        for j1 in xrange(len(events)):
            if at_y[i1][j1] == 1:
                sep_event_audio_path = os.path.join(sep_dir, "%s.%s.wav" % (bare_na, events[j1]))
                (sep_event_audio, _) = pp_data.read_audio(sep_event_audio_path, target_fs=fs)
                sep_noise_audio_path = os.path.join(sep_dir, "%s.noise.wav" % bare_na)
                (sep_noise_audio, _) = pp_data.read_audio(sep_noise_audio_path, target_fs=fs)
                ref_array = np.array((gt_event_audio, gt_noise_audio))
                est_array = np.array((sep_event_audio, sep_noise_audio))
                (sdr, sir, sar) = sdr_sir_sar(ref_array, est_array, sed_y[i1, :, j1], inside_only=True)
                print(sdr, sir, sar)
                sep_stats[events[j1]]['sdr'].append(sdr)
                sep_stats[events[j1]]['sir'].append(sir)
                sep_stats[events[j1]]['sar'].append(sar)
        
        cnt += 1
        # if cnt == 5: break
        
    print(sep_stats)
    sep_stat_path = os.path.join(workspace, "sep_stats", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr, "sep_stat.p")
    pp_data.create_folder(os.path.dirname(sep_stat_path))
    cPickle.dump(sep_stats, open(sep_stat_path, 'wb'))
   
def get_sep_stats(args):
    workspace = cfg.workspace
    te_fold = cfg.te_fold
    events = cfg.events
    n_events = args.n_events
    snr = args.snr
    sep_stat_path = os.path.join(workspace, "sep_stats", pp_data.get_filename(__file__), 
                          "n_events=%d" % n_events, "fold=%d" % te_fold, "snr=%d" % snr, "sep_stat.p")
    sep_stats = cPickle.load(open(sep_stat_path, 'rb'))
    print(sep_stats)
    
    sdrs, sirs, sars = [], [], []
    for e in events:
        sdr = np.mean(sep_stats[e]['sdr'][0])
        sir = np.mean(sep_stats[e]['sir'][0])
        sar = np.mean(sep_stats[e]['sar'][0])
        sdrs.append(sdr)
        sirs.append(sir)
        sars.append(sar)
        
    logging.info("%sSDR\tSIR\tSAR" % ("".ljust(16)))
    logging.info("*%s*%.3f\t*%.3f\t*%.3f" % ("Avg. of each".ljust(16), np.mean(sdrs), np.mean(sirs), np.mean(sars)))
    for i1 in xrange(len(events)):
        logging.info("%s%.3f\t%.3f\t%.3f" % (events[i1].ljust(16), sdrs[i1], sirs[i1], sars[i1]))

def sed_event_wise_tps_fns_fps(sed_pds, sed_gts, at_detected2d, sed_thres):
    """shape: (n_clips, n_time, n_classes)
    """
    (n_clips, n_time, n_classes) = sed_pds.shape
    
    detected3d = (np.sign(sed_pds - sed_thres) + 1.) / 2
    detected3d *= at_detected2d[:, None, :]

    # Event based SED. 
    tps, fns, fps = [], [], []
    for j1 in xrange(n_classes):
        total_tp = 0
        total_fn = 0
        total_fp = 0
        for i1 in xrange(n_clips):
            pd_pairs = vad.activity_detection(detected3d[i1, :, j1], sed_thres, n_smooth=24, n_salt=4)
            gt_pairs = vad.activity_detection(sed_gts[i1, :, j1], sed_thres, n_smooth=24, n_salt=4)
            pd_pairs = np.array(pd_pairs)
            gt_pairs = np.array(gt_pairs)
            if len(gt_pairs) > 0 and len(pd_pairs) > 0:
                matching = mir_eval.transcription.match_note_onsets(gt_pairs, pd_pairs, onset_tolerance=12) # onset collar=500 ms
                tp = len(matching)
            else:
                tp = 0
            fn = len(gt_pairs) - tp
            fp = len(pd_pairs) - tp
            total_tp += tp
            total_fn += fn
            total_fp += fp
        tps.append(total_tp)
        fns.append(total_fn)
        fps.append(total_fp)
        
    return tps, fns, fps

def print_stats(at_pds, at_gts, sed_pds, sed_gts):
    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)
    events = cfg.events
    
    # AT evaluate. 
    # TP, FN, FP, TN
    logging.info("====== Audio tagging (AT) ======")
    logging.info("%stp\tfn\tfp\ttn" % "".ljust(16))
    (tp, fn, fp, tn) = tp_fn_fp_tn(at_pds, at_gts, at_thres, average='micro')
    logging.info("%s%d\t%d\t%d\t%d\t" % ("Global".ljust(16), tp, fn, fp, tn))
    
    (tps, fns, fps, tns) = tp_fn_fp_tn(at_pds, at_gts, at_thres, average=None)
    for i1 in xrange(len(tps)):
        logging.info("%s%d\t%d\t%d\t%d\t" % (events[i1].ljust(16), tps[i1], fns[i1], fps[i1], tns[i1]))
        
    # Prec, recall, fvalue, AUC, eer
    logging.info("%sPrec\tRecall\tFvalue\tAUC\tEER" % "".ljust(16))
    (prec, recall, fvalue) = prec_recall_fvalue(at_pds, at_gts, at_thres, average='micro')
    auc = metrics.roc_auc_score(at_gts, at_pds, average='micro')
    eer = equal_error_rate(at_pds, at_gts, average='micro')
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Global".ljust(16), prec, recall, fvalue, auc, eer))
    (prec, recall, fvalue) = prec_recall_fvalue(at_pds, at_gts, at_thres, average='macro')
    auc = metrics.roc_auc_score(at_gts, at_pds, average='macro')
    eer = equal_error_rate(at_pds, at_gts, average='macro')
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Avg. of each".ljust(16), prec, recall, fvalue, auc, eer))
    
    (precs, recalls, fvalues) = prec_recall_fvalue(at_pds, at_gts, at_thres, average=None)
    aucs = metrics.roc_auc_score(at_gts, at_pds, average=None)
    eers = equal_error_rate(at_pds, at_gts, average=None)
    for i1 in xrange(len(tps)):
        logging.info("%s%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (events[i1].ljust(16), precs[i1], recalls[i1], fvalues[i1], aucs[i1], eers[i1]))
    
    # SED evaluate
    logging.info("====== Frame based SED ======")
    
    (n_clips, n_time, n_classes) = sed_pds.shape
    
    
    logging.info("%stp\tfn\tfp\ttn" % "".ljust(16))
    sed_pds_2d = sed_pds.reshape((n_clips * n_time, n_classes))
    sed_gts_2d = sed_gts.reshape((n_clips * n_time, n_classes))
    (tp, fn, fp, tn) = tp_fn_fp_tn(sed_pds_2d, sed_gts_2d, sed_thres, average='micro')
    logging.info("%s*%d\t*%d\t*%d\t*%d\t" % ("*Global".ljust(16), tp, fn, fp, tn))
    
    (tps, fns, fps, tns) = tp_fn_fp_tn(sed_pds_2d, sed_gts_2d, sed_thres, average=None)
    for i1 in xrange(len(tps)):
        logging.info("%s%d\t%d\t%d\t%d\t" % (events[i1].ljust(16), tps[i1], fns[i1], fps[i1], tns[i1]))
    
    # Prec, recall, fvalue
    logging.info("%sPrec\tRecall\tFvalue\tAUC\tER\tn_sub\tn_del\tn_ins" % "".ljust(16))
    (prec, recall, fvalue) = prec_recall_fvalue(sed_pds_2d, sed_gts_2d, sed_thres, average='micro')
    auc = metrics.roc_auc_score(sed_gts_2d, sed_pds_2d, average='micro')
    (er, n_sub, n_del, n_ins) = error_rate(sed_pds_2d, sed_gts_2d, sed_thres, average='micro')
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Global".ljust(16), prec, recall, fvalue, auc, er, n_sub, n_del, n_ins))
    (prec, recall, fvalue) = prec_recall_fvalue(sed_pds_2d, sed_gts_2d, sed_thres, average='macro')
    auc = metrics.roc_auc_score(sed_gts_2d, sed_pds_2d, average='macro')
    (er, n_sub, n_del, n_ins) = error_rate(sed_pds_2d, sed_gts_2d, sed_thres, average='macro')
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Avg. of each".ljust(16), prec, recall, fvalue, auc, er, n_sub, n_del, n_ins))
    
    (precs, recalls, fvalues) = prec_recall_fvalue(sed_pds_2d, sed_gts_2d, sed_thres, average=None)
    aucs = metrics.roc_auc_score(sed_gts_2d, sed_pds_2d, average=None)
    (ers, n_subs, n_dels, n_inss) = error_rate(sed_pds_2d, sed_gts_2d, sed_thres, average=None)
    for i1 in xrange(len(tps)):
        logging.info("%s%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (events[i1].ljust(16), precs[i1], recalls[i1], fvalues[i1], aucs[i1], ers[i1], n_subs[i1], n_dels[i1], n_inss[i1]))
    
    logging.info("====== Event based SED ======")
    
    at_detected2d = (np.sign(at_pds - at_thres) + 1.) / 2
    (tps, fns, fps) = sed_event_wise_tps_fns_fps(sed_pds, sed_gts, at_detected2d, sed_thres)
        
    logging.info("%stp\tfn\tfp" % "".ljust(16))
    logging.info("%s*%d\t*%d\t*%d" % ("*Total:".ljust(16), np.sum(tps), np.sum(fns), np.sum(fps)))
    for i1 in xrange(n_classes):
        logging.info("%s%d\t%d\t%d" % (events[i1].ljust(16), tps[i1], fns[i1], fps[i1]))
    
    
    logging.info("------ Prec, recall, fvalue ------")
    logging.info("%sPrec\tRecall\tFvalue\tAUC\tER\tn_sub\tn_del\tn_ins" % "".ljust(16))
    (prec, recall, fvalue) = prec_recall_fvalue_from_tp_fn_fp(tps, fns, fps, average='micro')
    (er, n_sub, n_del, n_ins) = error_rate_from_tp_fn_fp(tps, fns, fps, average='micro')
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Global".ljust(16), prec, recall, fvalue, auc, er, n_sub, n_del, n_ins))
    (prec, recall, fvalue) = prec_recall_fvalue_from_tp_fn_fp(tps, fns, fps, average='macro')
    (er, n_sub, n_del, n_ins) = error_rate_from_tp_fn_fp(tps, fns, fps, average='macro')
    logging.info("%s*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f\t*%.3f" % ("*Avg. of each".ljust(16), prec, recall, fvalue, auc, er, n_sub, n_del, n_ins))
    
    (precs, recalls, fvalues) = prec_recall_fvalue_from_tp_fn_fp(tps, fns, fps, average=None)
    (ers, n_subs, n_dels, n_inss) = error_rate_from_tp_fn_fp(tps, fns, fps, average=None)
    for i1 in xrange(n_classes):
        logging.info("%s%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (events[i1].ljust(16), precs[i1], recalls[i1], fvalues[i1], aucs[i1], ers[i1], n_subs[i1], n_dels[i1], n_inss[i1]))
        
    
    if False:
        post_det3d = np.zeros_like(detected3d)
        for i1 in xrange(n_clips):
            for j1 in xrange(n_classes):
                lists = vad.activity_detection(detected3d[i1, :, j1], sed_thres, n_smooth=24, n_salt=18)
                for (bgn, fin) in lists:
                    post_det3d[i1, bgn:fin, j1] = 1
        
        for i1 in xrange(len(sed_pds)):
            print("gt:", at_gts[i1])
            print("pd:", at_pds[i1])
            fig, axs = plt.subplots(4, 1, sharex=True)
            axs[0].matshow(sed_gts[i1].T, origin='lower', aspect='auto')
            axs[1].matshow(sed_pds[i1].T, origin='lower', aspect='auto')
            axs[2].matshow(detected3d[i1].T, origin='lower', aspect='auto')
            axs[3].matshow(post_det3d[i1].T, origin='lower', aspect='auto')
            plt.show()
        

def plot_hotmap(args):
    workspace = cfg.workspace
    events = cfg.events
    md_na = args.model_name
    n_events = args.n_events
    te_fold = cfg.te_fold
    
    feature_dir = os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events)
    yaml_dir = os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events)
    (tr_x, tr_at_y, tr_sed_y, tr_na_list, 
     te_x, te_at_y, te_sed_y, te_na_list) = pp_data.load_data(
        feature_dir=feature_dir, 
        yaml_dir=yaml_dir, 
        te_fold=te_fold, 
        is_scale=is_scale)
    
    md_path = os.path.join(workspace, "models", pp_data.get_filename(__file__), "n_events=%d" % n_events, md_na)
    md = serializations.load(md_path)
    
    x = te_x
    y = te_at_y
    
    observe_nodes = [md.find_layer('hotmap').output_]
    f_forward = md.get_observe_forward_func(observe_nodes)
    [a4] = md.run_function(f_forward, x, batch_size=500, tr_phase=0.)
    print a4.shape
    
    for i1 in xrange(len(a4)):
        # if te_na_list[i1] == 'CR_lounge_220110_0731.s2700_chunk48':
        print(y[i1])
        
        # print np.mean(a4[i1], axis=(1,2))
        
        fig, axs = plt.subplots(5,4, sharex=True)
        axs[0, 0].matshow(x[i1].T, origin='lower', aspect='auto')
        for i2 in xrange(16):
            axs[i2/4 + 1, i2%4].matshow(a4[i1,i2].T, origin='lower', aspect='auto', vmin=0, vmax=1)
            axs[i2/4 + 1, i2%4].set_title(events[i2])
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('train')
    parser_a.add_argument('--n_events', type=int)
    parser_a.add_argument('--snr', type=int)
    
    parser_b = subparsers.add_parser('recognize')
    parser_b.add_argument('--n_events', type=int)
    parser_b.add_argument('--snr', type=int)
    parser_b.add_argument('--model_name', type=str)
    
    parser_get_stats = subparsers.add_parser('get_stats')
    parser_get_stats.add_argument('--n_events', type=int)
    parser_get_stats.add_argument('--snr', type=int)
    
    parser_separate = subparsers.add_parser('separate')
    parser_separate.add_argument('--n_events', type=int)
    parser_separate.add_argument('--snr', type=int)
    
    parser_evaluate_separation = subparsers.add_parser('evaluate_separation')
    parser_evaluate_separation.add_argument('--n_events', type=int)
    parser_evaluate_separation.add_argument('--snr', type=int)
    
    parser_get_sep_stats = subparsers.add_parser('get_sep_stats')
    parser_get_sep_stats.add_argument('--n_events', type=int)
    parser_get_sep_stats.add_argument('--snr', type=int)
    
    parser_b2 = subparsers.add_parser('avg_recognize')
    parser_b2.add_argument('--n_events', type=int)
    parser_b2.add_argument('--snr', type=int)
    
    parser_c = subparsers.add_parser('plot_hotmap')
    parser_c.add_argument('--model_name', type=str)
    parser_c.add_argument('--n_events', type=int)
    
    args = parser.parse_args()
    
    logs_dir = os.path.join(cfg.workspace, "logs", pp_data.get_filename(__file__))
    pp_data.create_folder(logs_dir)
    logging = pp_data.create_logging(logs_dir, filemode='w')
    logging.info(os.path.abspath(__file__))
    logging.info(sys.argv)
    
    if args.mode == "train":
        train(args)
    elif args.mode == "recognize":
        recognize(args)
    elif args.mode == "get_stats":
        bgn_iter, fin_iter, interval = 1000, 2001, 200
        get_stats(args, bgn_iter, fin_iter, interval)
    elif args.mode == "separate":
        bgn_iter, fin_iter, interval = 2000, 3001, 200
        separate(args, bgn_iter, fin_iter, interval)
    elif args.mode == "evaluate_separation":
        evaluate_separation(args)
    elif args.mode == "get_sep_stats":
        get_sep_stats(args)
    elif args.mode == "avg_recognize":
        bgn_iter, fin_iter, interval = 2000, 3001, 200
        avg_recognize(args, bgn_iter, fin_iter, interval)
    
    elif args.mode == "plot_hotmap":
        plot_hotmap(args)