from __future__ import division
import os
import subprocess
import platform
import soundfile as sf

import numpy as np

from scipy import signal
import webrtcvad


def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return  float2pcm(sig, dtype='int16').tobytes()

def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def get_VAD_TUBS(y, d, frame_size, Fs_Hz, mode=None):
    vad = webrtcvad.Vad(mode)
    num_samples = frame_size*Fs_Hz/1000
    scores = list() 
    for i in range(len(y)//int(num_samples)):
        buf_y = y[i*num_samples:(i+1)*num_samples]
        buf_d = d[i*num_samples:(i+1)*num_samples]
        value = np.abs(np.dot(buf_y, buf_d))/(np.cdot(np.abs(buf_y),np.abs(buf_d))+np.finfo('float').eps)
        if value > 0.995:
            scores.append(1)
        else:
            scores.append(0)
    
    return scores

def get_VAD(sig, frame_size, Fs_Hz, mode=None):
    vad = webrtcvad.Vad(mode)
    num_samples = int(frame_size*Fs_Hz/1000)
    scores = list() 
    for i in range(len(sig)//num_samples):
        buf = float_to_byte(sig[i*num_samples:(i+1)*num_samples])
        score = vad.is_speech(buf,Fs_Hz)
        scores.append(score)

    return scores

def get_VAD_frames(sig, VAD_scores, frame_size, Fs_Hz):
    num_samples = int(frame_size*Fs_Hz/1000)
    short_sig = list()
    for i in range(len(VAD_scores)):
        if VAD_scores[i]:
            short_sig.append(sig[i*num_samples:(i+1)*num_samples])
    
    return np.array(short_sig)

def wbpesq(reference, degraded, sample_rate=None):
    """ Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.
    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).
    PESQ utility must be installed.
    """

    if platform.system() == 'Windows':
        program='./executables/wbpesq.exe'
    else:
        program='./executables/pesq_2018.exe'

    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
        import wave
        w = wave.open(reference, 'r')
        sample_rate = w.getframerate()
        w.close()
    if not sample_rate == 16000:
        return float(1.0)
        #raise ValueError('sample rate must be 16000 for wb pesq')

    args = [ program, '+%d' % sample_rate, '+wb', reference, degraded ]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    last_line = out.decode('utf-8').split('\n')[-2]
    if not last_line.startswith('P.862.2 Prediction'):
        #raise ValueError(last_line)
        print("Warning: PESQ failed, skipping file.")
        return float(1.0)
    else:
        return float(last_line.split()[-1])


def compute_PESQ(reference_signal, test_signal, fs, int_range_flag=True):
    # compute PESQ metric (according to ITU-T P.862.2) using numpy array representations of audio signals
    # reference_signal and test signal should be in range -1 to 1

    ref_name = os.path.join('.','temp','ref_temp.wav')
    test_name = os.path.join('.','temp','test_temp.wav')

    if int_range_flag:
        # renormalize audio data from int16 value domain to [-1, 1)
        max_value_int16 = 32768
        reference_signal = reference_signal / max_value_int16
        test_signal = test_signal / max_value_int16

    sf.write(ref_name, reference_signal, samplerate=fs)
    sf.write(test_name, test_signal, samplerate=fs)

    pesq_score = wbpesq(ref_name, test_name)

    os.remove(ref_name)
    os.remove(test_name)

    return pesq_score

def get_fft_frame(signal, offset, window, framing_params):

    start_idx = offset
    end_idx = offset + framing_params['frame_length']

    signal_windowed = signal[start_idx:end_idx] * window
    signal_fft_full = np.fft.rfft(signal_windowed, n=framing_params['K_fft'], axis=0)
    signal_fft = signal_fft_full[0: framing_params['K_mask']]

    return signal_fft

def compute_ERLE(y, s_hat, n, e=None, s_f = 0.99):
    d = n # known echo (aka "noise" in this framework)
    if e is not None:
        ePow = e**2
    else:
        d_est = y - s_hat # estimated echo
        ePow = (d-d_est)**2

    n_len = y.size
    ERLE = np.zeros((n_len,))
    dPow = d**2

    b = np.array([1-s_f])
    a = np.array([1,-s_f])

    dPow = signal.lfilter(b, a, dPow)
    ePow = signal.lfilter(b, a, ePow)

    ePow = np.maximum(np.spacing(np.float64(1)), ePow)
    div = np.divide(dPow, ePow)
    ERLE = 10 * np.log10(np.maximum(div, np.spacing(np.float64(1))))
    ERLE[ePow <= np.spacing(np.float64(1))] = 0
    mean_ERLE = np.mean(ERLE)

    return mean_ERLE, ERLE


def compute_LSD(reference_file, modified_file, L= 256):

    #--- Initialization of parameters
    L_        	= L // 2                                                        # overlap to previous frame
    Lplus      	= L // 2                                                        # overlap to successive frame
    Nw       	= L + L_ + Lplus                                                #Window length
    window      = np.hamming(Nw)                                                #blackman(Nw, 'periodic')	#hann(Nw, 'periodic')
    N_FFT       = int(2**(np.ceil(np.log2(Nw)+1)) )                             #Number of frequency bins for FFT computation (i.e., FFT length 1024)
    k_low       = 0                                                             #Lower frequency bin for UB LSD computation (i.e., 256 => 4kHz)
    k_up        = N_FFT // 2 + 1                                                #Upper frequency bin for UB LSD computation (i.e., 448 => 7kHz)
    LSD_low    	= 5                                                             #Lower LSD bound (dB) for percentage of outliers
    LSD_up    	= 10                                                            #Upper LSD bound (dB) for percentage of outliers

    #--- Load speech files
    ref_file    = reference_file                                                #Reference speech file
    ref_file    = np.concatenate([np.zeros(L_), ref_file, np.zeros(Lplus)])
    N_f_ref     = int(np.floor((len(ref_file)-L_-Lplus)/L))                     #Number of reference speech frames
    mod_file    = modified_file                                                 #Modified speech file
    mod_file    = np.concatenate([np.zeros(L_), mod_file, np.zeros(Lplus)])
    N_f_mod     = int(np.floor((len(mod_file)-L_-Lplus)/L))                     #Number of modified speech frames
    
    try:
        assert (N_f_ref == N_f_mod)
    except:
        raise IOError('Reference and modified speech files have different lengths!')

    #--- Initialization of UB LSD measure
    UB_LSD      = np.zeros(N_f_ref)              #Framewise UB log-spectral distortion measure
    vad         = np.zeros(N_f_ref)
    s_thres     = np.mean(ref_file**2 / 10)

    #--- Compute UB LSD
    offset    	= 0
    for idx in range(N_f_ref):

        if np.mean(ref_file[offset:offset+Nw]**2) < s_thres:
            vad[idx] = 0
        else:
            vad[idx] = 1

        P_ref       = np.abs(np.fft.rfft(ref_file[offset:offset+Nw]*window, n=N_FFT, axis = 0))**2	#Reference power spectrum (using a window function to avoid spectral leakage effects)
        P_mod       = np.abs(np.fft.rfft(mod_file[offset:offset+Nw]*window, n=N_FFT, axis = 0))**2	#Modified power spectrum (using a window function to avoid spectral leakage effects)
        UB_LSD[idx]	= np.sqrt( np.sum( (10 * np.log10( (P_ref[k_low:k_up] + np.finfo(np.float32).eps) / (P_mod[k_low:k_up] + np.finfo(np.float32).eps) ))**2 ) / (k_up - k_low) ) 
        offset	    = offset + L

    UB_LSD_mean	= np.sum(UB_LSD[vad==1]) / len(UB_LSD[vad==1])  #Mean UB LSD

    return UB_LSD, UB_LSD_mean