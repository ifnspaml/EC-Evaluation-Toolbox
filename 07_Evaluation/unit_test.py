from audio_processing import get_BB_components
from evaluation_metrics import *

import librosa
import numpy as np
from pesq import pesq

import os
from aecmos_local import AECMOSEstimator

data_dir    = '../02_Speech_Data/Example_files/test/'
out_dir     = '../08_Results/Kalman_Enhanced/'

sampling_rate = 16000

stfe_end = 10*sampling_rate
dt_start = 20*sampling_rate

y, _ = librosa.load(data_dir + 'nearend_mic/sp1/file_1_1.wav', sr = sampling_rate)
x, _ = librosa.load(data_dir + 'farend_speech/sp1/file_1_1.wav', sr = sampling_rate)
s, _ = librosa.load(data_dir + 'nearend_speech/sp1/file_1_1.wav', sr = sampling_rate)
d, _ = librosa.load(data_dir + 'echo/sp1/file_1_1.wav', sr = sampling_rate)
n, _ = librosa.load(data_dir + 'nearend_noise/sp1/file_1_1.wav', sr = sampling_rate)
e, _ = librosa.load(out_dir  + 'Test_Set/Kalman/sp1/file_1_1_e.wav', sr = sampling_rate)

window = signal.windows.blackman(512, sym=False)
params = {'window': window, 'K_fft': 512, 'K_mask': 257, 'frame_length':512, 'frame_shift': 64}

stop_token = False

try:
    result = pesq(sampling_rate, s, e)
    result = np.around(result,decimals=2)
    if result == 1.69:
        print('PESQ: \t passed')
    else:
        print('PESQ: \t expected 1.69, scored ' + str(result))
except:
    print('PESQ: \t failed')
    stop_token = True

try:
    mean_erle, erle_over_time = compute_ERLE(y[:stfe_end], e[:stfe_end], d[:stfe_end], s_f=0.99)
    mean_erle = np.around(mean_erle,decimals=0)
    if mean_erle == 4:
        print('ERLE: \t passed')
    else:
        print('ERLE: \t expected 4 dB, scored ' + str(mean_erle))
except:
    print('ERLE: \t failed')
    stop_token = True

try:
    LSD_scores, LSD_mean = compute_LSD(s[dt_start:], e[dt_start:])
    LSD_mean = np.around(LSD_mean,decimals=0)
    if LSD_mean == 13:
        print('LSD: \t passed')
    else:
        print('LSD: \t expected 13 dB, scored ' + str(LSD_mean))
except:
    print('LSD: \t failed')
    stop_token = True

try:
    # talk_type in ['nst', 'fst', 'dt']; refering to single-talk far-end, single-talk near-end, and double-talk
    aecmos      = AECMOSEstimator(os.getcwd() +'/models/Run_1663915512_Stage_0.onnx')
    echo, other = aecmos.run(talk_type='dt', lpb_sig=x,mic_sig=y,enh_sig=e)
    echo, other = np.around(echo,decimals=2), np.around(other,decimals=2)
    if (echo == 4.23) and (other == 2.76):
        print('AECMOS: \t passed')
    else:
        print('AECMOS: \t expected 4.23 and 2.76, scored ' + str(echo) + ' and ' + str(other))
except:
    print('AECMOS: \t failed')
    stop_token = True

# components: NE speech [s], echo [d], noise [n]
[s_tilde, d_tilde, n_tilde] = get_BB_components(e, y, components=[s, d, n], framing_params=params)

print('Black-box evaluation...')

try:
    result = pesq(sampling_rate, s, s_tilde)
    results = np.around(result,decimals=2)
    if results == 4.47:
        print('PESQ: \t passed')
    else:
        print('PESQ: \t expected 4.47, scored ' + str(result))
except:
    print('PESQ: \t failed')
    stop_token = True

try:
    mean_erle, erle_over_time   = compute_ERLE(y[:stfe_end], s_tilde[:stfe_end], d[:stfe_end], d_tilde[:stfe_end], s_f=0.99)
    mean_erle = np.around(mean_erle,decimals=0)
    if mean_erle == 7.0:
        print('ERLE: \t passed')
    else:
        print('ERLE: \t expected 7 dB, scored ' + str(mean_erle))
except:
    print('ERLE: \t failed')
    stop_token = True

try:    
    LSD_scores, LSD_mean = compute_LSD(s[dt_start:], s_tilde[dt_start:])
    LSD_mean = np.around(LSD_mean,decimals=0)
    if LSD_mean == 5.0:
        print('LSD: \t passed')
    else:
        print('LSD: \t expected 5 dB, scored ' + str(LSD_mean))
except:
    print('LSD: \t failed')
    stop_token = True

if not stop_token:
    os.system("python macro_script.py --noaudio -OP evaluation -MOS -d Test_Set -m Kalman")