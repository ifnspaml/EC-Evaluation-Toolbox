% --- General config file, include/run in all necessary scripts!
FILE_SEP = filesep;
conf.FILE_SEP = FILE_SEP;
conf.fsTarget = 16;

EXT_RAW	= GLOBAL.EXT_RAW;
EXT_WAV  = GLOBAL.EXT_WAV;
EXT_MAT  = GLOBAL.EXT_MAT;
TWO_GB_IN_B = 2*1024*1024*1024;

conf.doProfile									= false;
if conf.doProfile
	disp('!!!PROFILER ON!!!')
end

conf.scale_eps		= 0.1; % allowed deviation in dBov when normalizing files
conf.scale_alert	= 60;  % alert when a scaling factor > this is obtained, just a sanity check

assert(conf.fsTarget > 0, 'Please set conf.fsTarget before calling config.m');

conf.frame_length 	= 32*conf.fsTarget; % in ms, fsTarget = fs/1000
conf.frame_shift  	= conf.frame_length/2;
conf.fft_size		= conf.frame_length;
conf.K				= conf.fft_size/2+1;

conf.cp_start		= 2000 * conf.fsTarget / 500 + 1; % 33 for 8kHz this includes the 500Hz which are in bin 33! i checked it :), this is the bin where we start lookin for the pitch in the cepstrum
conf.cp_i_start		= 500 / (conf.fsTarget * 1000 / conf.fft_size) + 1; % 17 corresponding to the ifft spectrum :o -> make this dynamic in pitch.m
conf.Np				= 10;

conf.seed			= 1234;
conf.SNRs			= -5:1:15;
conf.CB_size		= 257;

conf.showPlots		= true;

% --- Preliminary NR config
conf.PNR.aPrioriSNREstimator		= 'DD';
conf.PNR.noisePowerEstimator		= 'MS';
conf.PNR.spectralWeightingRule	= 'MMSE-LSA';
conf.PNR.noiseOverEstimation		= 1.0;
conf.PNR.xi_min						= -15; % dB
conf.PNR.g_min							= -15; % dB
conf.PNR.dd_beta						= 0.975;
% ---

% --- Window calculation
window_name							= 'Hann';  % Window function selection
[nWin, w_norm, frame_shift]	= window_generation(conf.frame_length, conf.frame_shift, window_name);
conf.w								= nWin;
conf.w_sqrt							= sqrt(conf.w);
conf.w_norm							= w_norm;
% ---

conf.precision							= 'single'; % 'double', 'single'
% conf.featureType						= 'LOG_SPEC'; %'CC'; % CC, LOG_SPEC_Y_BAR, LOG_SPEC_Y_BAR_H, LOG_MEL_SPEC_NPE
% conf									= set_feature_type(conf.featureType, conf);
conf.VAonly								= false;
