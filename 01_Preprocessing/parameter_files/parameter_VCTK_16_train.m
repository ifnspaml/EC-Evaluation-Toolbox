% --- Parameters file, change according to training setup

FILE_SEP = filesep;
param.FILE_SEP = FILE_SEP;

srcPath   = ['..' FILE_SEP '00_Database' FILE_SEP 'CSTR-VCTK' FILE_SEP 'test_16' FILE_SEP];

param.out_path = ['..' FILE_SEP '02_Speech_Data' FILE_SEP 'CSTR-VCTK' FILE_SEP 'VCTK8_DEMAND_SEF_exp06_16_DT' FILE_SEP];

%% Data pattern
param.splitPattern      = [0.4; 0.3; 0.3];      % gender split: M-F, M-M, F-F
param.numFE             = 1;                    % number of FE signals to be generated for each speaker
param.numNE             = 125;                  % number of NE signals for each speaker

param.test_condition = 0;                       % Testing condition (with convergence periods)
                                                % 0 - off; 1 - DT; 2- STFE; 3 - STNE
param.len_sec = 10;                             % Condition section lengths

%if test_condition == 0
param.numConcatFE       = 3;                    % number of utterances concatenated for each FE signal
param.numConcatNE       = 2;                    % number of utterances concatenated for each NE signal

% number of FE/NE speaker pairs 
% (will result in x times 2 the output folders - speaker pairs and switched roles)
setsize_train   = 34;
setsize_dev     = 4;
setsize_test    = 0;
setsize_all     = setsize_train + setsize_dev + setsize_test;

param.fsTarget      = conf.fsTarget;

param.dBov          = -21;                      % base audio level (in dB)
param.refLevel       = [-36, -21];              % reference sound level range (in dB)

%% Echo settings
param.IR_pathChange       = [0,0];              % time frame ([start,stop] in s) in which RIR changes may happen
param.IR_fade             = [0.00, 0.00];       % fade-in time of new RIR ([min,max] in s)

param.SER      = [-10,-6,-3,0,3,6,10,99];       % randomly chosen from list
param.SER_mode = 'pseudo';                      % when to adjust SER level: 'pseudo'(speaker, before IR)/'d'(at microphone)

param.RIR_mode          = 'exp';                % RIR mode used: 'imagePRE'(image method, precalculated)
                                                % 'image' (image method, online calulation - slower),
                                                % 'exp' (WGN with exponential decay),
                                                % 'AIR' (Aachen IR DB)
                                                % 'AIR_DC' (delay-compensated AIR)
                                                % 'TUBS_dynIR' (TUBS continuously changing RIR after Jung)

param.generate_shortIR  = 512;                  % if > 0, generates an additional echo scenario with shortened RIR

%% image method-specific values
param.RIR.all.cs        = 340;                  % Sound velocity in m/s
param.RIR.all.fs        = param.fsTarget*1000;  % Sampling rate in Hz
param.RIR.all.n         = param.fsTarget*600;   % number of samples calculated for RIR (care for length!)
param.RIR.all.norm      = 0;                    % RIR normalization
param.RIR.all.order     = 20;                   % maximum reflection order (huge computational importance!)

% RT60:
param.RIR.train.beta{1} = [0.1, 1.0];
% reflection coefficients:
% param.RIR.train.beta{1}  = [0.85, 0.95];
% param.RIR.train.beta{2}  = [0.85, 0.95];
% param.RIR.train.beta{3}  = [0.85, 0.95];
% param.RIR.train.beta{4}  = [0.85, 0.95];
% param.RIR.train.beta{5}  = [0.85, 0.95, -0.7, -0.55];
% param.RIR.train.beta{6}  = [0.35, 0.45];

% SER-specific room modelling
param.RIR.train.SER_levels = [-10,-5,0,5,10,15,20,99];  %SER levels as defined, reference for speaker distances below
param.RIR.train.dist_SP  = [[0.08, 0.6]; [0.08, 0.6]; [0.08, 0.6]; [0.08, 0.6]; ...
    [0.08, 0.6]; [0.08, 0.6];[0.08, 0.6]];              % distance of NE speaker to mic
param.RIR.train.dist_LS  = [[0.08, 0.6]; [0.08, 0.6]; [0.08, 0.6]; [0.08, 0.6]; ...
    [0.08, 0.6]; [0.08, 0.6];[0.08, 0.6]];              % distance of loudspeaker to mic
param.RIR.train.r_mode   = 'dyn';                       %
param.RIR.train.a        = [2, 10];                     % room width
param.RIR.train.c        = [2.4, 3.2];                  % room height
param.RIR.train.pos_opt  = 1000;                        % options from pre-processing
param.RIR.train.z_mic    = [0.3, 1.8];                  % mic height

param.RIR.train.T        = [0.05, 0.6];                 % RIR length

% change if multiple datasets with different parameters should be generated at once
param.RIR.dev           = param.RIR.train;
param.RIR.test          = param.RIR.train;

%% TUBS dynIR-specific values
param.dyn_length = 4;   % length of accessed recording in s; [1, 4, 8, 20]
param.fin_length = 8;   % length of generated RIR in s
param.step       = 1;   % step size (skip x-1 samples in recording)
param.freeze     = 4;   % time in s after which RIR is frozen

%% Noise setup
param.enableMicNoise    = 1;                                    % enable noise at microphone
param.SNR               = [99, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27]; % randomly chosen from list
param.sepNoise          = 1;                                    % generate seperate NE noise datasets (for modular training)
param.useNoiseDB        = 1;                                    % use noise databases instead of generic white noise

noiseDB.train.name      = ['DEMAND_16+QUT-NOISE'];
noiseDB.train.fs        = 16;
noiseDB.train.ext       = '.wav';
noiseDB.train.add_white = 0;
noiseDB.train.randomcut = 1;

noiseDB.dev = noiseDB.train;
noiseDB.test = noiseDB.train;

%% FE noise
param.onlyFENoise           = 0;                            % replaces FE speaker with WGN
param.enableFENoise         = 0;                            % adds noise to FE signal (settings below)

% for enableFENoise
param.FE_SNR                = [10, 15, 20, 25];             % FE signal SNR

noiseDB_FE.train.name       = ['DNS' FILE_SEP 'train'];     % FE noise path
noiseDB_FE.train.fs         = param.fsTarget;               % sampling rate
noiseDB_FE.train.ext        = '.wav';  
noiseDB_FE.train.add_white  = 0;                            % add WGN as option
noiseDB_FE.train.randomcut  = 1;                            % randomly cut section from noise file

noiseDB_FE.dev = noiseDB_FE.train;
noiseDB_FE.dev.name         = ['DNS' FILE_SEP 'val'];

noiseDB_FE.test = noiseDB_FE.dev;
noiseDB_FE.test.name         = ['DNS' FILE_SEP 'test'];

%% Nonlinearities
param.nonlinear     = 1;                        % enable LS non-linearities
param.NLfunc{1}     = 'SEF';                    % SEF       - scaled error function
% param.NLfunc{2}     = 'memSeg';               % memSeg    - memoryless sigmoidal after Wang
% param.NLfunc{3}     = 'arctan';               % arctan    - arctan after Jung
% param.NLfunc        = param.NLfunc';          % if multiple options passed (random choice per file)

param.NLoptions     = [0.5, 1, 10, 999];        % parameter for SEF

param.enableIR_NE   = 0;                        % enable near-end impulse response
param.enableIR_FE   = 1;                        % enable far-end impulse response


%% SNR/SER randomization
generate_single     = 1;                        % only generate one dataset with randomly chosen SER/SNR; else: one per combination
param.secure_random = 1;                        % Ensures evenly distributed SNR/SER values


if debug
    param                   %   for Debugging: display parameter 
end

% might help with RAM issues (deprecated)
param.dataset_division = 1;