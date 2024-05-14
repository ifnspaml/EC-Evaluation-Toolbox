% --- General config file, include/run in all necessary scripts!
FILE_SEP = filesep;
conf.FILE_SEP = FILE_SEP;
conf.fsTarget = 16;

if isunix
	conf.dir.base.root				= [FILE_SEP 'home' FILE_SEP 'Elshamy' FILE_SEP 'R' FILE_SEP];
elseif ispc
 	conf.dir.base.root              = ['C:' FILE_SEP 'Users' FILE_SEP 'Seidel' FILE_SEP 'Documents' FILE_SEP 'Hiwi' FILE_SEP 'Master' FILE_SEP 'Script' FILE_SEP '01_Preprocessing']; %['R:' FILE_SEP];
else
	disp('Platform not supported')
end

addpath(genpath([conf.dir.base.root '05DiSS' FILE_SEP '02SRC' FILE_SEP '00COMMON' FILE_SEP]));
addpath(genpath([conf.dir.base.root '05DiSS' FILE_SEP '02SRC' FILE_SEP 'SV_matlab_toolbox' FILE_SEP]));


EXT_RAW	= GLOBAL.EXT_RAW;
EXT_WAV  = GLOBAL.EXT_WAV;
EXT_MAT  = GLOBAL.EXT_MAT;
TWO_GB_IN_B = 2*1024*1024*1024;

conf.doProfile									= false;
if conf.doProfile
	disp('!!!PROFILER ON!!!')
end

%db = Database;
conf.dir.input.res.root						= [conf.dir.base.root '05DiSS' FILE_SEP '01RESOURCES' FILE_SEP];
conf.dir.input.res.clean					= [conf.dir.input.res.root 'clean' FILE_SEP];
conf.dir.input.res.noisy					= [conf.dir.input.res.root 'noisy' FILE_SEP];
conf.dir.input.res.org						= [conf.dir.input.res.clean 'Original' FILE_SEP];

conf.dir.input.res.test.clean.si			= [conf.dir.input.res.root 'test' FILE_SEP 'clean' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.test.noisy.si			= [conf.dir.input.res.root 'test' FILE_SEP 'noisy' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.test.noise				= [conf.dir.input.res.root 'test' FILE_SEP 'noise' FILE_SEP];
conf.dir.input.res.test.denoised.si		= [conf.dir.input.res.root 'test' FILE_SEP 'denoised' FILE_SEP 'SI' FILE_SEP];

% --- assumed to be SI
conf.dir.input.res.dev.clean.si			= [conf.dir.input.res.root 'dev' FILE_SEP 'clean' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.dev.clean.sd			= [conf.dir.input.res.root 'dev' FILE_SEP 'clean' FILE_SEP 'SD' FILE_SEP];
conf.dir.input.res.dev.noisy.si			= [conf.dir.input.res.root 'dev' FILE_SEP 'noisy' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.dev.noise			= [conf.dir.input.res.root 'dev' FILE_SEP 'noise' FILE_SEP];
conf.dir.input.res.dev.denoised.si		= [conf.dir.input.res.root 'dev' FILE_SEP 'denoised' FILE_SEP 'SI' FILE_SEP];
% ---

conf.dir.input.res.train.clean.sd		= [conf.dir.input.res.root 'train' FILE_SEP 'clean' FILE_SEP 'SD' FILE_SEP];
conf.dir.input.res.train.clean.si		= [conf.dir.input.res.root 'train' FILE_SEP 'clean' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.train.clean.root		= [conf.dir.input.res.root 'train' FILE_SEP 'clean' FILE_SEP];
conf.dir.input.res.train.noisy.si		= [conf.dir.input.res.root 'train' FILE_SEP 'noisy' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.train.noise.si		= [conf.dir.input.res.root 'train' FILE_SEP 'noise' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.train.denoised.si	= [conf.dir.input.res.root 'train' FILE_SEP 'denoised' FILE_SEP 'SI' FILE_SEP];




conf.dir.input.res.db								= [conf.dir.input.res.root '00DB' FILE_SEP];
conf.dir.input.res.tpl.sd							= [conf.dir.input.res.root '01TEMPLATES' FILE_SEP 'SD' FILE_SEP];
conf.dir.input.res.tpl.si							= [conf.dir.input.res.root '01TEMPLATES' FILE_SEP 'SI' FILE_SEP];
conf.dir.input.res.hmm								= [conf.dir.input.res.root '02HMM' FILE_SEP];
conf.dir.input.res.dnn								= [conf.dir.input.res.root '03DNN' FILE_SEP];

conf.dir.input.FIR_LP								= [conf.dir.base.root '05DiSS' FILE_SEP '02SRC' FILE_SEP '01PREPROCESSiNG' FILE_SEP 'FIR_LP' FILE_SEP];

conf.dir.output										= [conf.dir.base.root '05DiSS' FILE_SEP '03OUTPUT' FILE_SEP datestr(now, 'yymmdd_HHMMSS') FILE_SEP];

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
