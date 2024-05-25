import os
import numpy as np
import warnings
import glob
import librosa
import scipy
from GetaecTestSet import GetaecTestSet
from scipy.io import loadmat
warnings.filterwarnings("ignore")
from audio_processing import *
from evaluation_metrics import *
import argparse

parser = argparse.ArgumentParser(description='Tensorflow script for training of Deep Learning Models for Echo Cancellation:')

parser.add_argument('--model_path', '-p', required=False, default='../06_Networks/weights/model', help='Path to model (can be relative).')
parser.add_argument('--echoless', required=False, action="store_true", help='Removes echo from test files.')
parser.add_argument('--noiseless', required=False, action="store_true", help='Removes noise from test files.')
parser.add_argument('--noNEAR', required=False, action="store_true", help='Disable near-end signal.')
parser.add_argument('--noaudio', required=False, action="store_true", help='Prevents writing of audio files.')
parser.add_argument('--size', required=False, default='Full', choices=['Full', 'Partial'], help='Minimizes testset for debugging with "Partial".')
parser.add_argument('--dataset', '-d', default='TUB_synth_test_TIMITnew_16', help="SNR of preprocessed data")
parser.add_argument('--model_select', '-m', default='Kalman', help='Model type in use.') 
parser.add_argument('--evalAECMOS', '-MOS', required=False, action="store_true", help='Compute AECMOS scores.')
parser.add_argument('--ERLEoverTime', '-EOT', required=False, action="store_true", help="Enable EoT computation for the entire file.")

parser.add_argument('--sampling_in', '-fsI', default=16000, help='Input sampling rate.')
parser.add_argument('--sampling_AEC', '-fsA', default=16000, help='AEC sampling rate.')

parser.add_argument('--cold_start', '-CS', required=False, action="store_true", help="Enable cold start.")
parser.add_argument('--operation_mode', '-OP', required=False, default='both', choices=['both', 'inference', 'evaluation'], help="Whether to perform inference, evlution, or both.")

parser.add_argument('--add_delay', '-delay', default=0, help='Input sampling rate.')
parser.add_argument('--SER_adjust', '-SER', default=[0], nargs='+',  help='Input sampling rate.')
parser.add_argument('--SNR_adjust', '-SNR', default=[0], nargs='+', help='Input sampling rate.')

args = parser.parse_args()

start_idx=0

###########################
# General Experiment setup
###########################

ScenarioChoice = args.size

Fs_Hz = args.sampling_in
eval_config = dict()
eval_config['target_noise'] = False
eval_config['noNear'] = args.noNEAR
eval_config['echoless'] = args.echoless
eval_config['noiseless'] = args.noiseless
eval_config['EOT'] = args.ERLEoverTime

DatasetType = args.dataset
Model_select = args.model_select

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

op_mode = args.operation_mode

FilePathBase, BB_avail, signalPaths, sectioned_test = GetaecTestSet(DatasetType, Model_select)

###########################
# Load trained model
###########################
print('Initializing Postprocessor Model...')

if Model_select == 'NLMS':

	frame_len = 512

	if op_mode != "evaluation":
		from models.custom_layers import NLMS_filter
		Filter = NLMS_filter(frame_len)
	
	Enhanced_FilePath = signalPaths['shat'] + '/NLMS'

elif Model_select == 'Kalman':

	frame_len = 512
	frame_hop = 128

	if op_mode != "evaluation":
		from models.custom_layers import Kalman_filter
		Filter = Kalman_filter(M=frame_len, L=frame_hop)

	Enhanced_FilePath = signalPaths['shat'] + '/Kalman'

# add new model:
# elif Model_select == '...':
#	 ...

else:
	settings_temp = __import__('settings')
	settings_dict = settings_temp.get_net_params()
	Enhanced_FilePath = signalPaths['shat'] + '/' + Model_select
	args.noaudio = True

####################################################################################################################

eval_config['evalAECMOS'] = args.evalAECMOS
if op_mode != "inference"and args.evalAECMOS:
	from aecmos_local import AECMOSEstimator
	import onnxruntime as ort

	eval_config['AECMOS'] = AECMOSEstimator(os.getcwd() +'/models/Run_1663915512_Stage_0.onnx')

# get config for blackbox algorithm/evaluation
if  BB_avail:
	framing_params_BB = get_framing_params_BB(f_s=args.sampling_AEC)

# folder structure
if 'INTERSPEECH' in DatasetType or 'ICASSP' in DatasetType: 
	mic = glob.glob(os.path.join(signalPaths['y'], '*_mic.wav'))
else:
	mic = glob.glob(os.path.join(signalPaths['y'], '*/*.wav'))

if ScenarioChoice == 'Partial':
	FileNumber = 5
else:
	FileNumber = len(mic)

# cold start setup
if args.cold_start:
	if sectioned_test == 1:
		sec_mode = 'DT'
	else:
		sec_mode = 'ST'
	eval_config['cold_start'] = True
	Enhanced_FilePath += '_CS'
else:
	eval_config['cold_start'] = False

eval_config['sectioned'] = sectioned_test

if not os.path.exists(Enhanced_FilePath):
	os.makedirs(Enhanced_FilePath)

metrics_final = dict()
ERLE_STFE_LEC = np.zeros(FileNumber)
ERLE_STFE_NN = np.zeros(FileNumber)

for SER_value_o in args.SER_adjust:
	for SNR_value_o in args.SNR_adjust:

		SER_value = int(SER_value_o)
		SNR_value = int(SNR_value_o)

		if (len(args.SER_adjust) * len(args.SNR_adjust) > 1) or (SER_value_o != 0) or (SNR_value_o != 0):
			Enhanced_FilePath_tmp = Enhanced_FilePath + '_SER' + str(SER_value) + '_SNR' + str(SNR_value)
		else:
			Enhanced_FilePath_tmp = Enhanced_FilePath

		if int(args.add_delay) > 0:
			Enhanced_FilePath_tmp = Enhanced_FilePath_tmp + '_delay' + args.add_delay

		######################################################
		# Run inference
		######################################################
		for File_indx in range(start_idx, FileNumber):
			NoisyFileNameIndx = mic[File_indx]

			string_array = np.array(list(NoisyFileNameIndx))
			char_indexes1 = np.where(string_array == os.sep)
			NoisyFileNameIndx0 = np.array(char_indexes1)[-1][-1]
			NoisyFileName_tmp = NoisyFileNameIndx[NoisyFileNameIndx0 + 1:]
			try:
				NoisyFileNameIndx1 = np.array(char_indexes1)[-1][-2]
				NoisySpeakerName_tmp = NoisyFileNameIndx[NoisyFileNameIndx1 + 1:NoisyFileNameIndx0]
			except:
				pass

			string_array4 = np.array(list(NoisyFileName_tmp))
			char_indexes4 = np.where(string_array4 == '_')
			string_array3 = np.array(list(NoisyFileName_tmp))
			char_indexes3 = np.where(string_array3 == '.')

			if 'INTERSPEECH' in DatasetType or 'ICASSP' in DatasetType:
				farend_FileName_tmp = NoisyFileName_tmp[0:np.array(char_indexes4)[-1][-1]] + '_lpb.wav'
				nearend_mic_EC_FileName = NoisyFileName_tmp
				mic_FileName = NoisyFileName_tmp
				Enh_FileName = NoisyFileName_tmp

			else:

				FileIDName = NoisySpeakerName_tmp + os.sep + NoisyFileName_tmp
				if not args.noaudio:
					if not os.path.exists(Enhanced_FilePath_tmp + os.sep + NoisySpeakerName_tmp):
						os.makedirs(Enhanced_FilePath_tmp + os.sep + NoisySpeakerName_tmp)

				farend_FileName_tmp = FileIDName
				mic_FileName = FileIDName
				Enh_FileName = FileIDName
				echo_FileName = FileIDName
				noise_FileName = FileIDName
				nearend_FileName = FileIDName
				nearend_mic_EC_FileName = FileIDName

				meta_File = FileIDName[:-4] + '.mat'

			micFileName 	= glob.glob(os.path.join(signalPaths['y'], mic_FileName))
			farend_FileName = glob.glob(os.path.join(signalPaths['x'], farend_FileName_tmp))

			###############################
			# Load farend and mic signals
			###############################
			mic_path_final 		= micFileName[0]
			farend_path_final 	= farend_FileName[0]

			y, fs = librosa.core.load(mic_path_final, sr=Fs_Hz, mono=True) # mic signal
			x, fs = librosa.core.load(farend_path_final, sr=Fs_Hz, mono=True) # farend signal

			if int(args.add_delay) > 0:
				x = np.roll(x, -int(args.add_delay))
				x[-int(args.add_delay):] = 0

			# length correction of the all three signals (some of files in INTERSPEECH DT were not the same length!)
			Lmin = min(len(y),len(x))
			y = y[0:Lmin]
			x = x[0:Lmin]

			y_in = y
			x_in = x

			if sectioned_test or args.cold_start:
				metaFileName 			= glob.glob(os.path.join(signalPaths['meta'], meta_File))
				meta_path_final 		= metaFileName[0]
				meta_data 				= loadmat(meta_path_final[:-4])
				eval_config['sections'] = meta_data['section_lengths']

			if args.cold_start:
				if sec_mode == 'DT':
					sec_len = eval_config['sections'][2][0]
				else:
					sec_len = eval_config['sections'][1][0]
				y_in = y_in[-sec_len:]
				x_in = x_in[-sec_len:]

			########################################################
			# Prepare BB components
			########################################################
			if BB_avail:
				echoFileName = glob.glob(os.path.join(signalPaths['d'], echo_FileName))
				noiseFileName = glob.glob(os.path.join(signalPaths['n'], noise_FileName))
				nearendFileName = glob.glob(os.path.join(signalPaths['s'], nearend_FileName))
				echo_path_final = echoFileName[0]
				noise_path_final = noiseFileName[0]
				nearend_path_final = nearendFileName[0]
				d = librosa.core.load(echo_path_final, sr=Fs_Hz, mono=True)[0]
				n = librosa.core.load(noise_path_final, sr=Fs_Hz, mono=True)[0]
				s = librosa.core.load(nearend_path_final, sr=Fs_Hz, mono=True)[0]

				d_in = d
				n_in = n
				s_in = s

				if args.cold_start:
					d_in = d_in[-sec_len:]
					n_in = n_in[-sec_len:]
					s_in = s_in[-sec_len:]

				if SER_value != 0:
					gain = np.power(10.0, -SER_value/20.0)
					d_in = gain * d_in
				
				if SNR_value != 0:
					gain = np.power(10.0, -SNR_value/20.0)
					n_in = gain * n_in

				if args.echoless:
					if args.noiseless:
						y_in = s_in
						n_in = np.zeros(len(s_in))
					else:
						y_in -= d_in
						y_in = np.minimum(y_in, 1)
						y_in = np.maximum(y_in, -1)
						n_in = y_in-s_in
						n_in = np.minimum(n_in, 1)
						n_in = np.maximum(n_in, -1)
					x_in = np.zeros(len(y_in))
				else:
					if args.noiseless:
						n_in = np.zeros(len(s_in))
						y_in = s_in+d_in
						y_in = np.minimum(y_in, 1)
						y_in = np.maximum(y_in, -1)
					else:
						y_in = n_in+s_in+d_in
						y_in = np.minimum(y_in, 1)
						y_in = np.maximum(y_in, -1)

				if eval_config['noNear']:
					y_in -= s_in
					y_in = np.minimum(y_in, 1)
					y_in = np.maximum(y_in, -1)
					s_in = np.zeros(len(s_in))

				signal_components = dict()
				signal_components['d'] = d_in
				signal_components['n'] = n_in
				signal_components['s'] = s_in
				signal_components['y'] = y_in
				signal_components['x'] = x_in
			else:
				signal_components = None
			xB_components = None

			if op_mode !=  "inference":
				# load section lengths and other info in multi-condition test scenarios
				if sectioned_test:
					metaFileName 			= glob.glob(os.path.join(signalPaths['meta'], meta_File))
					meta_path_final 		= metaFileName[0]
					meta_data 				= loadmat(meta_path_final[:-4])
					eval_config['sections'] = meta_data['section_lengths']

			###############################
			# Inference
			###############################
			EnhFileName = glob.glob(os.path.join(Enhanced_FilePath_tmp, Enh_FileName))
			out_path = os.path.join(Enhanced_FilePath_tmp+'\\')
			wavfile_Enhanced = os.path.join(out_path,Enh_FileName)

			if op_mode !=  "evaluation":

				if Model_select == 'NLMS' or Model_select == 'Kalman':
					if not args.sampling_in == args.sampling_AEC:
						y_in = librosa.resample(y_in, args.sampling_in, args.sampling_AEC)
						x_in = librosa.resample(x_in, args.sampling_in, args.sampling_AEC)
						d_in = librosa.resample(d_in, args.sampling_in, args.sampling_AEC)

					s_post = Filter.run(y_in, x_in, d=d_in, settings_dict=eval_config)
					
					if not args.sampling_in == args.sampling_AEC:
						s_post = librosa.resample(s_post, args.sampling_AEC, args.sampling_in)
					s_post = s_post[:Lmin]

				# add more models:
				# elif Model_select == '...':

				elif Model_select == 'Oracle_noisy':
					s_post = s_in + n_in
				elif Model_select == 'Oracle_clean':
					s_post = s_in
				elif Model_select == 'Microphone':
					s_post = y_in

				############################
				# Write enhanced signal
				############################
				if not args.noaudio:

					wavfile_Enhanced = os.path.join(Enhanced_FilePath_tmp, mic_FileName[0:-4] + '_y')
					write_wav(wavfile_Enhanced, y, Fs_Hz, int_range_flag=False)

					wavfile_Enhanced = os.path.join(Enhanced_FilePath_tmp, mic_FileName[0:-4] + '_e')
					write_wav(wavfile_Enhanced, s_post, args.sampling_AEC, int_range_flag=False)

					print('File no. ', File_indx, 'writing output wave file \t', mic_FileName)

				else:
					print('File no. ', File_indx, 'processing output wave file \t', mic_FileName)
		
			else:

				wavfile_Enhanced = os.path.join(Enhanced_FilePath_tmp, mic_FileName[0:-4] + '_e.wav')
				s_post, fs = librosa.core.load(wavfile_Enhanced, sr=Fs_Hz, mono=True)

				s_post = s_post[:Lmin]

				print('File no. ', File_indx, 'loading wave file \t', mic_FileName)

			if op_mode !=  "inference":
				# Evaluation
				if BB_avail:
					metrics = metric_eval(s_post, framing_params_BB, eval_config, signal_components, xB_components, f_in=Fs_Hz) # Performance Evaluation

					for entry in metrics:
						if not 'EoT' in entry:
							if File_indx == start_idx:
								metrics_final[entry] = list()
							metrics_final[entry].append(metrics[entry])

					if File_indx == start_idx:
						
						if sectioned_test == 1:
							EoT_len = len(metrics['EoT_DT_bb']) - frame_len
							metrics_final['EoT_DT_bb'] = metrics['EoT_DT_bb'][:EoT_len]
							metrics_final['EoT_DT'] = metrics['EoT_DT'][:EoT_len]
						elif sectioned_test == 2:
							EoT_len = len(metrics['EoT_FEST_bb']) - frame_len
							metrics_final['EoT_FEST_bb'] = metrics['EoT_FEST_bb'][:EoT_len]
							metrics_final['EoT_FEST'] = metrics['EoT_FEST'][:EoT_len]

						if eval_config['EOT']:
							metrics_final['EoT_full_bb'] 	= metrics['EoT_full_bb']
							metrics_final['EoT_full'] 		= metrics['EoT_full']

					else:
						if sectioned_test == 1:
							metrics_final['EoT_DT_bb'] += metrics['EoT_DT_bb'][:EoT_len]
							metrics_final['EoT_DT'] += metrics['EoT_DT'][:EoT_len]
						elif sectioned_test == 2:
							metrics_final['EoT_FEST_bb'] += metrics['EoT_FEST_bb'][:EoT_len]
							metrics_final['EoT_FEST'] += metrics['EoT_FEST'][:EoT_len]
						
						if eval_config['EOT']:
							metrics_final['EoT_full_bb'] 	+= metrics['EoT_full_bb']
							metrics_final['EoT_full'] 		+= metrics['EoT_full']
					

				# ERLE calculation for STFE
				if '_STFE' in DatasetType:
					Lmin = min(len(y),len(s_post))
					y = y[:Lmin]
					x = x[:Lmin]
					s_post = s_post[:Lmin]

					[mean_ERLE_NN, ERLE_NN] = compute_ERLE(y, s_post, y, e=None)
					metrics_final['ERLE_NN'].append(np.maximum(ERLE_NN, np.spacing(np.float64(1))))
					metrics_final['ERLE_STFE_NN'].append(mean_ERLE_NN)

		if op_mode !=  "inference":

			try:
				file_number = max(1, File_indx - start_idx)
			except:
				raise IOError('No data to evaluate found.')

			if sectioned_test == 1:
				metrics_final['EoT_DT_bb'] /= file_number
				metrics_final['EoT_DT'] /= file_number
			elif sectioned_test == 2:
				metrics_final['EoT_FEST_bb'] /= file_number
				metrics_final['EoT_FEST'] /= file_number

			if eval_config['EOT']:
				metrics_final['EoT_full_bb'] /= file_number
				metrics_final['EoT_full'] /= file_number

			if '_STFE' in DatasetType:
				print('\n LEC: ERLE STFE = ', np.mean(ERLE_STFE_LEC),' (dB)')
				print('\n NN: ERLE STFE = ', np.mean(ERLE_STFE_NN),' (dB)')

				metrics_final['ERLE_STFE'] = ERLE_STFE_NN

			if ScenarioChoice == 'Partial':
				Enhanced_FilePath_tmp += '_Part' + str(FileNumber)

			if BB_avail:
				Enhanced_FilePath_tmp += '_BB'
			
			if op_mode == "evaluation":
				Enhanced_FilePath_tmp += '_EV'

			metrics_final['file_list'] = mic
				
			MatFileName = Enhanced_FilePath_tmp + '.mat'
			scipy.io.savemat(MatFileName, metrics_final)
