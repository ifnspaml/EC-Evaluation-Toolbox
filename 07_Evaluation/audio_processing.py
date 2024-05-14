import numpy as np
import soundfile as sf
import math
from scipy import signal
from evaluation_metrics import *
from librosa.core import resample

from pesq import pesq

def metric_eval(s_post_in, framing_params, eval_config, signal_components=None, BB_components=None, f_in=None):
    Fs_Hz = framing_params['f_s']

    noNEAR = eval_config['noNear']

    signal_length = s_post_in.size

    # unpack components if present
    if signal_components is not None:
        sig_eval = True
        y = signal_components['y']
        d = signal_components['d']
        n = signal_components['n']
        s = signal_components['s']
        if f_in is not None and f_in != Fs_Hz:
            y = resample(y, f_in, Fs_Hz)
            d = resample(d, f_in, Fs_Hz)
            n = resample(n, f_in, Fs_Hz)
            s = resample(s, f_in, Fs_Hz)
            y = y[:signal_length]
            d = d[:signal_length]
            n = n[:signal_length]
            s = s[:signal_length]
    else:
        sig_eval = False
        
    if BB_components is not None:
        BB_eval = True
        d_tilde_bb = BB_components['d']
        n_tilde_bb = BB_components['n']
        s_tilde_bb = BB_components['s']

        s_post = s_post_in

        y = y[:signal_length]
        d = d[:signal_length]
        n = n[:signal_length]
        s = s[:signal_length]
            
    elif signal_components is not None:
        BB_eval = True

        [s_tilde_bb,n_tilde_bb,d_tilde_bb] = get_BB_components(s_post_in=s_post_in, mic=y, components=[s,n,d], framing_params=framing_params)

        s_tilde_bb = np.append(s_tilde_bb, np.zeros(signal_length - s_tilde_bb.size))
        n_tilde_bb = np.append(n_tilde_bb, np.zeros(signal_length - n_tilde_bb.size))
        d_tilde_bb = np.append(d_tilde_bb, np.zeros(signal_length - d_tilde_bb.size))

        s_post = np.append(s_post_in, np.zeros(signal_length - s_post_in.size))

        y = y[:signal_length]
        s = s[:signal_length]
        n = n[:signal_length]
        d = d[:signal_length]

    else:

        BB_eval = False

    results = dict()

    if eval_config['sectioned'] and not eval_config['cold_start']:
        if eval_config['sectioned'] == 1:
            l_FEST  = eval_config['sections'][0][0] # length of FEST
            l_NEST  = eval_config['sections'][1][0] # length of NEST
            l_DT    = eval_config['sections'][2][0] # length of DT
            
            s_DT            = s[-l_DT:]
            n_DT            = n[-l_DT:]
            y_DT            = y[-l_DT:]
            s_bb_DT         = s_tilde_bb[-l_DT:]
            n_bb_DT         = n_tilde_bb[-l_DT:]
            s_post_DT       = s_post[-l_DT:]
            d_DT            = d[-l_DT:]
            d_bb_DT         = d_tilde_bb[-l_DT:]

            if l_NEST > 0:
                s_NEST          = s[l_FEST:-l_DT]
                s_bb_NEST       = s_tilde_bb[l_FEST:-l_DT]
                s_post_NEST     = s_post[l_FEST:-l_DT]
                n_bb_NEST       = n_tilde_bb[l_FEST:-l_DT]

            d_FEST          = d[:l_FEST]
            n_FEST          = n[:l_FEST]
            y_FEST          = y[:l_FEST]
            s_post_FEST     = s_post[:l_FEST]
            s_bb_FEST       = s_tilde_bb[:l_FEST]
            n_bb_FEST       = n_tilde_bb[:l_FEST]
            d_bb_FEST       = d_tilde_bb[:l_FEST]
            s_FEST          = s[:l_FEST]

            if eval_config['evalAECMOS']:
                x       = signal_components['x'][:signal_length]
                x_DT    = x[-l_DT:]
                x_NEST  = x[l_FEST:-l_DT]
                x_FEST  = x[:l_FEST]
                y_NEST  = y[l_FEST:-l_DT]
                d_NEST  = d[l_FEST:-l_DT]

        if eval_config['sectioned'] == 2:
            l_FEST = eval_config['sections'][0][0] # length of FEST

            d_FEST          = d[l_FEST:]
            n_FEST          = n[l_FEST:]
            y_FEST          = y[l_FEST:]
            s_post_FEST     = s_post[l_FEST:]
            s_bb_FEST       = s_tilde_bb[l_FEST:]
            n_bb_FEST       = n_tilde_bb[l_FEST:]
            d_bb_FEST       = d_tilde_bb[l_FEST:]
            s_FEST          = s[l_FEST:]

            if eval_config['evalAECMOS']:
                x       = signal_components['x'][:signal_length]
                x_FEST  = x[l_FEST:]

        if eval_config['sectioned'] == 3:
            l_NEST = eval_config['sections'][0][0] # length of FEST 
            s_NEST          = s[l_NEST:]
            s_bb_NEST       = s_tilde_bb[l_NEST:]
            s_post_NEST     = s_post[l_NEST:]
            n_bb_NEST       = n_tilde_bb[l_NEST:]
            if eval_config['evalAECMOS']:
                x       = signal_components['x'][:signal_length]
                x_NEST  = x[l_NEST:]
                y_NEST  = y[l_NEST:]
                d_NEST  = d[l_NEST:]
            
        if eval_config['sectioned'] == 1:
            results['pesq_s_bb_DT']     = pesq(16000, s_DT, s_bb_DT)
            results['pesq_s_post_DT']   = pesq(16000, s_DT, s_post_DT)
            [mean, debug_erle_DT]       = compute_ERLE(y_DT, s_bb_DT, d_DT, e=d_bb_DT)
            results['ERLE_DT_bb']       = np.maximum(0, mean)
            results['EoT_DT_bb']        = np.maximum(0, debug_erle_DT)
            [mean, debug_erle_DT]       = compute_ERLE(y_DT, s_post_DT, d_DT)
            results['ERLE_DT']          = np.maximum(0, mean)
            results['EoT_DT']           = np.maximum(0, debug_erle_DT)

            results['LSD_DT'] = compute_LSD(s_DT, s_post_DT)
            results['LSD_DT_bb'] = compute_LSD(s_DT, s_bb_DT)

            if eval_config['evalAECMOS']:
                [results['AECMOS_FDT'], results['AECMOS_NDT']]  = eval_config['AECMOS'].run(talk_type='dt', lpb_sig=x_DT,mic_sig=y_DT,enh_sig=s_post_DT)

            VAD_DT = get_VAD(d_DT, 10, Fs_Hz)

            y_VAD_DT        = get_VAD_frames(y_DT, VAD_DT, 10, Fs_Hz)
            s_VAD_bb_DT     = get_VAD_frames(s_bb_DT, VAD_DT, 10, Fs_Hz)
            d_VAD_DT        = get_VAD_frames(d_DT, VAD_DT, 10, Fs_Hz)
            d_VAD_bb_DT     = get_VAD_frames(d_bb_DT, VAD_DT, 10, Fs_Hz)
            s_post_VAD_DT   = get_VAD_frames(s_post_DT, VAD_DT, 10, Fs_Hz)

            [mean, debug_erle_DT]   = compute_ERLE(y_VAD_DT, s_VAD_bb_DT, d_VAD_DT, e=d_VAD_bb_DT)
            results['vERLE_DT_bb']  = np.maximum(0, mean)
            [mean, debug_erle_DT]   = compute_ERLE(y_VAD_DT, s_post_VAD_DT, d_VAD_DT)
            results['vERLE_DT']     = np.maximum(0, mean)

        if not eval_config['sectioned'] == 2: 
            results['pesq_s_bb_NEST']   = pesq(16000, s_NEST, s_bb_NEST)
            results['pesq_s_post_NEST'] = pesq(16000, s_NEST, s_post_NEST)

            if eval_config['evalAECMOS']:
                [snt ,results['AECMOS_NEST']]   = eval_config['AECMOS'].run(talk_type='nst', lpb_sig=x_NEST,mic_sig=y_NEST,enh_sig=s_post_NEST)
                assert snt > 4.0

        if not eval_config['sectioned'] == 3:
            [mean, debug_erle_FEST] = compute_ERLE(y_FEST, s_bb_FEST, d_FEST, e=d_bb_FEST)
            results['EoT_FEST_bb']  = np.maximum(0, debug_erle_FEST)
            results['ERLE_FEST_bb'] = np.maximum(0, mean)
            [mean, debug_erle_FEST] = compute_ERLE(y_FEST, s_post_FEST, d_FEST)
            results['ERLE_FEST']    = np.maximum(0, mean)
            results['EoT_FEST']     = np.maximum(0, debug_erle_FEST)
            if eval_config['evalAECMOS']:
                [results['AECMOS_FEST'], snt]   = eval_config['AECMOS'].run(talk_type='st', lpb_sig=x_FEST,mic_sig=y_FEST,enh_sig=s_post_FEST)
                assert snt > 4.0

            VAD_FEST = get_VAD(d_FEST, 10, Fs_Hz)

            y_VAD_FEST      = get_VAD_frames(y_FEST, VAD_FEST, 10, Fs_Hz)
            s_VAD_bb_FEST   = get_VAD_frames(s_bb_FEST, VAD_FEST, 10, Fs_Hz)
            d_VAD_FEST      = get_VAD_frames(d_FEST, VAD_FEST, 10, Fs_Hz)
            d_VAD_bb_FEST   = get_VAD_frames(d_bb_FEST, VAD_FEST, 10, Fs_Hz)
            s_post_VAD_FEST = get_VAD_frames(s_post_FEST, VAD_FEST, 10, Fs_Hz)

            [mean, debug_erle_FEST]     = compute_ERLE(y_VAD_FEST, s_VAD_bb_FEST, d_VAD_FEST, e=d_VAD_bb_FEST)
            results['vERLE_FEST_bb']    = np.maximum(0, mean) 
            [mean, debug_erle_FEST]     = compute_ERLE(y_VAD_FEST, s_post_VAD_FEST, d_VAD_FEST)
            results['vERLE_FEST']       = np.maximum(0, mean)
        
        if eval_config['EOT']:
            [_, debug_erle] = compute_ERLE(y, s_tilde_bb, d, e=d_tilde_bb, s_f=0.99925)
            results['EoT_full_bb'] = np.maximum(0, debug_erle)
            [_, debug_erle] = compute_ERLE(y, s_post, d, s_f=0.99925)
            results['EoT_full'] = np.maximum(0, debug_erle)


    elif eval_config['sectioned']:
        if eval_config['evalAECMOS']:
            x = signal_components['x'][:signal_length]
        
        if eval_config['sectioned'] == 1:
            results['pesq_s_bb_DT']     = pesq(16000, s, s_tilde_bb)
            results['pesq_s_post_DT']   = pesq(16000, s, s_post)
            [mean, debug_erle_DT]       = compute_ERLE(y, s_tilde_bb, d, e=d_tilde_bb)
            results['ERLE_DT_bb']       = np.maximum(0, mean)
            results['EoT_DT_bb']        = np.maximum(0, debug_erle_DT)
            [mean, debug_erle_DT]       = compute_ERLE(y, s_post, d)
            results['ERLE_DT']          = np.maximum(0, mean)
            results['EoT_DT']           = np.maximum(0, debug_erle_DT)

            results['LSD_DT']       = compute_LSD(s_DT, s_post)
            results['LSD_DT_bb']    = compute_LSD(s_DT, s_tilde_bb)

            if eval_config['evalAECMOS']:
                [results['AECMOS_FDT'], results['AECMOS_NDT']]  = eval_config['AECMOS'].run(talk_type='dt', lpb_sig=x,mic_sig=y,enh_sig=s_post)

            VAD_DT = get_VAD(d, 10, Fs_Hz)
        

            y_VAD_DT        = get_VAD_frames(y, VAD_DT, 10, Fs_Hz)
            s_VAD_bb_DT     = get_VAD_frames(s_tilde_bb, VAD_DT, 10, Fs_Hz)
            d_VAD_DT        = get_VAD_frames(d, VAD_DT, 10, Fs_Hz)
            d_VAD_bb_DT     = get_VAD_frames(d_tilde_bb, VAD_DT, 10, Fs_Hz)
            s_post_VAD_DT   = get_VAD_frames(s_post, VAD_DT, 10, Fs_Hz)

            [mean, debug_erle_DT]   = compute_ERLE(y_VAD_DT, s_VAD_bb_DT, d_VAD_DT, e=d_VAD_bb_DT)
            results['vERLE_DT_bb']  = np.maximum(0, mean)
            [mean, debug_erle_DT]   = compute_ERLE(y_VAD_DT, s_post_VAD_DT, d_VAD_DT)
            results['vERLE_DT']     = np.maximum(0, mean)

        if not eval_config['sectioned'] == 2: 
            results['pesq_s_bb_NEST']   = pesq(16000, s, s_tilde_bb)
            results['pesq_s_post_NEST'] = pesq(16000, s, s_post)

            if eval_config['evalAECMOS']:
                [snt ,results['AECMOS_NEST']] = eval_config['AECMOS'].run(talk_type='nst', lpb_sig=x,mic_sig=y,enh_sig=s_post)
                assert snt > 4.0

        if not eval_config['sectioned'] == 3:
            [mean, debug_erle_FEST] = compute_ERLE(y, s_tilde_bb, d, e=d_tilde_bb)
            results['EoT_FEST_bb']  = np.maximum(0, debug_erle_FEST)
            results['ERLE_FEST_bb'] = np.maximum(0, mean)
            [mean, debug_erle_FEST] = compute_ERLE(y, s_post, d)
            results['ERLE_FEST']    = np.maximum(0, mean)
            results['EoT_FEST']     = np.maximum(0, debug_erle_FEST)
            if eval_config['evalAECMOS']:
                [results['AECMOS_FEST'], snt] = eval_config['AECMOS'].run(talk_type='st', lpb_sig=x,mic_sig=y,enh_sig=s_post)
                assert snt > 4.0

            VAD_FEST = get_VAD(d, 10, Fs_Hz)

            y_VAD_FEST      = get_VAD_frames(y, VAD_FEST, 10, Fs_Hz)
            s_VAD_bb_FEST   = get_VAD_frames(s_tilde_bb, VAD_FEST, 10, Fs_Hz)
            d_VAD_FEST      = get_VAD_frames(d, VAD_FEST, 10, Fs_Hz)
            d_VAD_bb_FEST   = get_VAD_frames(d_tilde_bb, VAD_FEST, 10, Fs_Hz)
            s_post_VAD_FEST = get_VAD_frames(s_post, VAD_FEST, 10, Fs_Hz)

            [mean, debug_erle_FEST]     = compute_ERLE(y_VAD_FEST, s_VAD_bb_FEST, d_VAD_FEST, e=d_VAD_bb_FEST)
            results['vERLE_FEST_bb']    = np.maximum(0, mean) 
            [mean, debug_erle_FEST]     = compute_ERLE(y_VAD_FEST, s_post_VAD_FEST, d_VAD_FEST)
            results['vERLE_FEST']       = np.maximum(0, mean)

    else:
        if eval_config['evalAECMOS']:
            x = signal_components['x'][:signal_length]
            [results['AECMOS_FDT'], results['AECMOS_NDT']] = eval_config['AECMOS'].run('dt',lpb_sig=x,mic_sig=y,enh_sig=s_post)

        if BB_eval:
            if not noNEAR:
                if Fs_Hz==16000:
                    results['pesq_s_post_bb'] = pesq(16000, s, s_tilde_bb)
            [ERLE, debug_erle_BB]   = compute_ERLE(y, s_post, d, e=d_tilde_bb)
            results['ERLE_bb']      = np.maximum(0, ERLE)
            results['LSD_DT_bb']    = compute_LSD(s, s_tilde_bb)
            
        if sig_eval:
            if not noNEAR:
                if Fs_Hz==16000:
                    results['pesq_s_post'] = pesq(16000, s, s_post)
            [ERLE, debug_erle]  = compute_ERLE(y, s_post, d)
            results['ERLE']     = np.maximum(0, ERLE)
            results['LSD_DT']   = compute_LSD(s, s_post)

    return results


def get_framing_params_BB(f_s=16000):

    frame_scale = f_s/16000

    freq_ri_flag = True
    context_frames = 1      # provide total number of frames in context window for input features (default: 1 --> no feature context)
    fram_length = 512
    fram_shift = 64
    K_fft = 512            # only for ceps/spec_features

    if frame_scale != 1:
        fram_length = int(fram_length*frame_scale)
        fram_shift = int(fram_shift*frame_scale)
        K_fft = int(K_fft*frame_scale)

    # framing structure for LSTM speech enhancement
    framing_params = {'frame_length': fram_length,
                        'frame_shift': fram_shift,
                        'K_fft': K_fft,
                        'K_fft_scale': frame_scale,
                        'K_mask': K_fft//2+1,
                        'output_dim': K_fft//2+1,
                        'window': 'blackman',
                        'sqrt_window_flag': True,
                        'freq_ri_flag': freq_ri_flag,
                        'context_frames': context_frames}
    framing_params['f_s'] = f_s

    # Construct window
    window = signal.windows.blackman(fram_length, sym=True)
    # compute window normalization factor
    win_temp = np.hstack((window, np.zeros(window.shape)))
    end_ind = int(fram_length / fram_shift) * fram_shift
    start_ind = int(fram_shift)
    w_norm = np.sum(win_temp[start_ind:end_ind:fram_shift])
    #window = np.sqrt(window)

    framing_params['w_norm'] = w_norm
    framing_params['window'] = window

    return framing_params

def read_file(file_path, import_type):

    s = sf.read(file_path, dtype=import_type)[0]

    return s.astype('float64')


def write_file(file_path, signal, sampling_frequency):

    sf.write(file_path, signal.astype('int16'), samplerate=sampling_frequency)


def read_raw_file(file_path, sampling_frequency, import_type):

    s = sf.read(file_path, channels=1, samplerate=sampling_frequency, dtype=import_type)[0]

    return s.astype('float64')


def write_raw_file(file_path, signal, sampling_frequency):

    sf.write(file_path, signal.astype('int16'), samplerate=sampling_frequency, subtype='PCM_16')

def get_fft_frame(signal, offset, window, framing_params):

    start_idx = offset
    end_idx = offset + framing_params['frame_length']

    signal_windowed = signal[start_idx:end_idx] * window
    signal_fft_full = np.fft.rfft(signal_windowed, n=framing_params['K_fft'], axis=0)
    signal_fft = signal_fft_full[0: framing_params['K_mask']]

    return signal_fft


def get_ifft_frame(signal_fft_reduced, synthesis_window, w_norm, framing_params):

    signal_ifft_temp = np.fft.irfft(signal_fft_reduced, axis=0)
    if framing_params['K_fft'] > framing_params['frame_length']:
        signal_ifft_temp = signal_ifft_temp[0:framing_params['frame_length']]
    elif framing_params['K_fft'] < framing_params['frame_length']:
        raise ValueError('Length of FFT can not be smaller than frame length')

    if synthesis_window is None:
        signal_ifft = signal_ifft_temp
    else:
        signal_ifft = signal_ifft_temp * synthesis_window / w_norm

    return signal_ifft

def write_wav(filename, signal, fs, int_range_flag=True):

    if int_range_flag:
        # renormalize audio data from int16 value domain to [-1, 1)
        max_value_int16 = 32768
        signal = signal / max_value_int16

    folder = filename[::-1].split(os.sep, maxsplit=1)[1][::-1]

    if not os.path.exists(folder):
        os.makedirs(folder)

    sf.write((filename + '.wav'), signal, samplerate=fs)


def get_BB_components(s_post_in, mic, components: list, framing_params):

    frame_length = framing_params['frame_length']
    frame_shift  = framing_params['frame_shift']
    window       = framing_params['window']

    try:
        window_norm  = framing_params['window_norm'] 
    except:
        win_temp = np.hstack((window, np.zeros(window.shape)))
        end_ind = int(frame_length / frame_shift) * frame_shift
        start_ind = int(frame_shift)
        window_norm = np.sum(win_temp[start_ind:end_ind:frame_shift])

    signal_length = s_post_in.size

    num_frames = math.floor((signal_length - frame_length) / frame_shift)

    num_nonredundant = int(frame_length/2 + 1)

    BB_components = list()
    for comp in components:
        BB_components.append(np.zeros(signal_length))

    offset = 0

    for frame_idx in range(num_frames):
        # compute non-redundant fft bins for current frame
        s_post = get_fft_frame(s_post_in, offset, window, framing_params)

        S_post_frame = (np.hstack((np.real(s_post), np.imag(s_post[1:-1])))).astype('float32')

        masked_signal_real = S_post_frame[0:num_nonredundant]
        masked_signal_imag = np.hstack((np.zeros((1,)), S_post_frame[num_nonredundant:], np.zeros((1,))))
        s_hat_fft_complex = masked_signal_real + 1j * masked_signal_imag

        #get BB components
        y_fft = get_fft_frame(mic, offset, window, framing_params)
        gain_bb = np.clip(np.abs(s_hat_fft_complex) / np.abs(y_fft), None, 1) * (np.exp(1j * np.angle(s_hat_fft_complex)) / np.exp(1j * np.angle(y_fft)))

        for i, comp in enumerate(components):

            # compute non-redundant fft bins for current frame
            c_fft = get_fft_frame(comp, offset, window, framing_params)

            c_fft_tilde_bb = gain_bb * c_fft

            c_ifft_tilde_temp_bb = np.fft.irfft(c_fft_tilde_bb)
            
            if window is not None:
                c_ifft_tilde_bb = c_ifft_tilde_temp_bb[:frame_length] * window / window_norm
            else:
                c_ifft_tilde_bb = c_ifft_tilde_temp_bb[:frame_length]

            BB_components[i][offset:(offset + frame_length)] = BB_components[i][offset:(offset + frame_length)] + c_ifft_tilde_bb

        # increment offset by value of frame_shift
        offset = offset + frame_shift

    return BB_components