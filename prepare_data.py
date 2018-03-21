import numpy 
import os
import argparse
import numpy as np
import librosa
import csv
import soundfile
import matplotlib.pyplot as plt
import yaml
import time
from sklearn import preprocessing
from scipy import signal
import logging
import pickle
import cPickle
from scipy.signal import lfilter

import config as cfg

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def create_logging(log_dir, filemode):
    # Write out to file
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, "%05d.log" % i1)):
        i1 += 1
    log_path = os.path.join(log_dir, "%05d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def read_stereo_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def pink(N, random_state=None):
    """Pink noise. 
    Code from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py
    """
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    if random_state:
        white_noise = random_state.randn(N)
    else:
        white_noise = np.random.randn(N)
    pink_noise = lfilter(b, a, white_noise)
    pink_noise /= np.max(np.abs(pink_noise))
    return pink_noise

def avg_eng(x):
    return np.mean(np.abs(x))

def signal_scaling_factor(s_eng, n_eng, snr):
    return 10. ** (snr / 20.) * (n_eng / s_eng)

###
def mix(yaml_file, out_dir, snr_list):
    dataset_dir = cfg.dataset_dir
    workspace = cfg.workspace
    fs = cfg.sample_rate
    clip_samples = int(fs * cfg.clip_duration)
    event_max_len = cfg.event_max_len
    onset_list = cfg.onset_list
    clip_samples = int(cfg.clip_duration * fs)
    
    
    create_folder(out_dir)
    rs = np.random.RandomState(0)
    t1 = time.time()
    
    # Read mixture yaml file. 
    with open(yaml_file, 'r') as f:
        data = yaml.load(f)
        
    # Write out separate audio & yaml file. 
    for (i1, d) in enumerate(data):
        d_new = d
        
        # Background. 
        clean_audio = np.zeros(clip_samples)
        
        events_audio_list = []  # Used for compute avg. energy. 
        
        # Add events to one clip. 
        for (j1, event_data) in enumerate(d['events']):
            audio_name = event_data['file_name']
            audio_path = os.path.join(dataset_dir, audio_name)
            (audio, _) = read_audio(audio_path, target_fs=fs)
            audio /= np.max(np.abs(audio))
            audio = audio[0 : int(fs * event_max_len)]
            bgn_sample = int(onset_list[j1] * fs)
            fin_sample = bgn_sample + len(audio)
            clean_audio[bgn_sample : fin_sample] += audio
            d_new['events'][j1]['offset'] = float(fin_sample) / fs
            events_audio_list.append(audio)
            
        bare_na = os.path.splitext(d['name'])[0]
        
        # Write out clean audio. 
        out_clean_audio_path = os.path.join(out_dir, "%s.clean.wav" % bare_na)
        write_audio(out_clean_audio_path, clean_audio, fs)
            
        # Write out mixed audio. 
        event_eng = avg_eng(np.concatenate(events_audio_list, axis=0))
        white_noise = rs.randn(clip_samples)
        noise_eng = avg_eng(white_noise)
        for snr in snr_list:
            ssf = signal_scaling_factor(event_eng, noise_eng, snr)  # Signal scaling factor. 
            clean_audio_tmp = ssf * clean_audio
            mixed_audio_tmp = clean_audio_tmp + white_noise
            alpha = 1. / np.max(np.abs(mixed_audio_tmp))
            clean_audio_tmp *= alpha
            white_noise_tmp = alpha * white_noise
            stereo_audio = np.array([clean_audio_tmp, white_noise_tmp]).T
            out_stereo_audio_path = os.path.join(out_dir, "%s.mixed_%ddb.wav" % (bare_na, snr))
            write_audio(out_stereo_audio_path, stereo_audio, fs)

        # Write out yaml file. 
        out_yaml_path = os.path.join(out_dir, os.path.splitext(d['name'])[0] + ".yaml")
        with open(out_yaml_path, 'w') as f:
            f.write(yaml.dump(d_new, default_flow_style=False))
            
        print(i1, bare_na)
        
    print("Finished! %s" % (time.time() - t1,))

def calc_sp(audio, fs, ham_win, n_window, n_overlap):
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode='complex') 
    x = x.T
    return x

def calc_feat(audio, fs, ham_win, n_window, n_overlap):
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
            
def calculate_logmel(audio_dir, out_dir):
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    ham_win = np.hamming(n_window)
    
    create_folder(out_dir)
    t1 = time.time()
    cnt = 0
    
    names = os.listdir(audio_dir)
    for na in names:
        if os.path.splitext(na)[-1] == ".wav":
            # Read audio
            audio_path = os.path.join(audio_dir, na)
            (stereo_audio, _) = read_stereo_audio(audio_path, target_fs=fs)
            if stereo_audio.ndim == 1:
                audio = stereo_audio
            elif stereo_audio.ndim == 2:
                audio = np.sum(stereo_audio, axis=-1)
            
            # Extract feature. 
            x = calc_feat(audio, fs, ham_win, n_window, n_overlap)
            print(cnt, na, x.shape)
            cnt += 1

            # plt.matshow(x, origin='lower', aspect='auto', cmap='jet')
            # plt.show()
            # pause
            
            # Write out feature. 
            bare_na = os.path.splitext(na)[0]
            out_path = os.path.join(out_dir, "%s.p" % bare_na)
            cPickle.dump(x, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print("Calculate log Mel finished! %s" % (time.time() - t1,))

def compute_scaler(x):
    (N, n_time, n_freq) = x.shape
    x = x.reshape((N * n_time, n_freq))
    
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x)
    return scaler

def do_scaler_on_x3d(x, scaler):
    """Use scaler to scale input. 
    
    Args:
      x: (n_clips, n_time, n_freq). 
      scaler: object. 
      
    Returns:
      x3d: (n_clips, n_time, n_freq), scaled input. 
    """
    (N, n_time, n_freq) = x.shape
    x2d = x.reshape((N * n_time, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((N, n_time, n_freq))
    return x3d


def load_data(feature_dir, yaml_dir, te_fold, snr, is_scale):
    names = [na for na in os.listdir(feature_dir) if na.endswith("mixed_%ddb.p" % snr)]
    lb_to_ix = cfg.lb_to_ix
    n_out = len(cfg.events)
    step_in_sec = cfg.step_in_sec
    
    tr_x, tr_at_y, tr_sed_y, tr_na_list = [], [], [], []
    te_x, te_at_y, te_sed_y, te_na_list = [], [], [], []
    
    for na in names:
        bare_na = os.path.splitext(os.path.splitext(na)[0])[0]
        
        # Load yaml file. 
        yaml_path = os.path.join(yaml_dir, bare_na + ".yaml")
        with open(yaml_path, 'r') as f:
            d = yaml.load(f)
        
        # Load feature. 
        feature_path = os.path.join(feature_dir, na)
        x = cPickle.load(open(feature_path, 'rb'))
            
        # Obtain AT ground truth label. 
        at_y = np.zeros(n_out)
        for event_data in d['events']:
            event = event_data['event']
            ix = lb_to_ix[event]
            at_y[ix] = 1
        
        # Obtain SED ground truth. 
        (n_time, n_freq) = x.shape
        sed_y = np.zeros((n_time, n_out))
        for event_data in d['events']:
            event = event_data['event']
            onset_fr = int(round(event_data['onset'] / step_in_sec))
            offset_fr = int(round(event_data['offset'] / step_in_sec))
            ix = lb_to_ix[event]
            sed_y[onset_fr : offset_fr, ix] = 1
        
        # Append data. 
        if d['fold'] == te_fold:
            te_x.append(x)
            te_at_y.append(at_y)
            te_sed_y.append(sed_y)
            te_na_list.append(na)
        else:
            tr_x.append(x)
            tr_at_y.append(at_y)
            tr_sed_y.append(sed_y)
            tr_na_list.append(na)
            
    tr_x = np.array(tr_x).astype(np.float32)
    tr_at_y = np.array(tr_at_y).astype(np.float32)
    tr_sed_y = np.array(tr_sed_y).astype(np.float32)
    te_x = np.array(te_x).astype(np.float32)
    te_at_y = np.array(te_at_y).astype(np.float32)
    te_sed_y = np.array(te_sed_y).astype(np.float32)
    
    if is_scale:
        scaler = compute_scaler(tr_x)
        tr_x = do_scaler_on_x3d(tr_x, scaler)
        te_x = do_scaler_on_x3d(te_x, scaler)
        out_path = os.path.join(cfg.workspace, "scalers", "fold=%d" % te_fold, "snr=%d.scaler" % snr)
        create_folder(os.path.dirname(out_path))
        pickle.dump(scaler, open(out_path, 'wb'))

    return tr_x, tr_at_y, tr_sed_y, tr_na_list, \
           te_x, te_at_y, te_sed_y, te_na_list

if __name__ == '__main__':
    workspace = cfg.workspace
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('mix')
    parser_b = subparsers.add_parser('calculate_logmel')
    
    args = parser.parse_args()
    
    if args.mode == 'mix':
        for n_events in [1, 2, 3, 4]:
            mix(yaml_file=os.path.join(workspace, "mixture_yamls", "n_events=%d.yaml" % n_events), 
                out_dir=os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events), 
                snr_list=[-5, 0, 5, 10, 15, 20, 100])
    elif args.mode == 'calculate_logmel':
        for n_events in [1, 2, 3, 4]:
            calculate_logmel(audio_dir=os.path.join(workspace, "mixed_audio", "n_events=%d" % n_events), 
                             out_dir=os.path.join(workspace, "features", "logmel", "n_events=%d" % n_events))