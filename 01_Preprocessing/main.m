clear all

addpath(genpath('scripts'))
addpath(genpath('parameter_files'))

%% ---------------------------
%
% Before running:
% ! Add 00COMMON and subfolders to path
% check config.m (usually no changes needed)
% ! carefully check parameter_{DATABASE NAME}.m (contains almost all important variables)
% if speaker dataset is changed, adapt regexp command (line 35, 48)
%
%% ---------------------------

%% Load Parameters
debug = 1;
run config.m                     % base config, usually not changed often
run parameter_VCTK_16_test.m             % individual dataset/corpus parameters
% run parameter_TIMIT.m  

if ~exist(param.out_path, 'dir')
    mkdir(param.out_path)
end
save([param.out_path 'settings'])

tic

param.output = ["wav"];     %wav/mat
generatorSettings = rng(1); %seeds the random number generator using the seed so that randi produces a predictable sequence of numbers

%% List and Organize Database

disp(['Reading input directory ' srcPath ' looking for ' EXT_WAV ' files.']);
evalc('dirList = listFiles(srcPath, [char("*") EXT_WAV], Inf, true);');

% female speakers
list_F      = regexp(dirList,['f[0-9]']);           %CSTR VCRK
% list_F = regexp(dirList,['\\DR\d*\\F']);      %TIMIT/
% list_F = regexp(dirList,['\\*\\F[0-9]']);      %NTT
female      = ~cellfun('isempty',list_F);
nFem        = nnz(female);
dirList_F   = cell(nFem,1);
j = 1;
for i = 1 : size(female)
    if female(i) ~= 0
        dirList_F(j) = dirList(i);
        j = j+1;
    end
end

% male speakers
list_M      = regexp(dirList,['m[0-9]']);           %CSTR VCRK
% list_M = regexp(dirList,['\\DR\d*\\M']);      %TIMIT
% list_M = regexp(dirList,['\\*\\M[0-9]']);      %NTT
male        = ~cellfun('isempty',list_M);
nMale       = nnz(male);
dirList_M   = cell(nMale,1);

j = 1;
for i = 1 : size(male)
    if male(i) ~= 0
        dirList_M(j) = dirList(i);
        j = j+1;
    end
end

random_F = randperm(numel(dirList_F),setsize_all).'; %choose female speakers
random_M = randperm(numel(dirList_M),setsize_all).'; %choose male speakers

if ~(param.SER_mode == "d" || param.SER_mode == "pseudo")
    error("SER application mode not defined. Use 'd' or 'pseudo'.")
end

%% Preload Room Impulse Responses
if param.RIR_mode == "imagePRE"
    if setsize_train > 0
        param.RIR.spec      = param.RIR.train;
        [param.RIR.train.options_FE ,param.RIR.train.options_NE, RT60]     = multiRIR(param.RIR);
%         save('TRAIN_FE_IR', 'param.RIR.train.options_FE', 'param.RIR.train.SER_levels');
    end
    if setsize_dev > 0
        param.RIR.spec      = param.RIR.dev;
        [param.RIR.dev.options_FE,param.RIR.dev.options_NE, param.RIR.dev.RT60]   = multiRIR(param.RIR);
    end
    if setsize_test > 0
        param.RIR.spec      = param.RIR.test;
        [param.RIR.test.options_FE,param.RIR.test.options_NE, param.RIR.dev.RT60]  = multiRIR(param.RIR);
%         save('TEST_FE_IR', 'param.RIR.test.options_FE', 'param.RIR.test.SER_levels');
    end
elseif param.RIR_mode == "AIR"

    %precomputed
    load('..\\AachenIR\\AIR_binaural_16');
    for i = 1:size(AIR_options,1)
        AIR_options{i}=AIR_options{i}(1:round(AIR_T60(i)*param.RIR.all.fs));
        AIR_options{i}=0.2*AIR_options{i}/max(AIR_options{i});
    end

elseif param.RIR_mode == "AIR_DC"

    %precomputed
    load('..\\AachenIR\\AIR_binaural_16');
    for i = 1:size(AIR_options,1)
        [max_val, max_idx] = max(AIR_options{i});
        AIR_options{i}=AIR_options{i}(max(max_idx-20,1):round(AIR_T60(i)*param.RIR.all.fs));
        AIR_options{i}=0.2*AIR_options{i}/max_val;
    end

elseif param.RIR_mode == "AIRshort"

    %precomputed
    load('AIR_binaural_16');
    for i = 1:size(AIR_options,1)
        [max_val, max_idx] = max(AIR_options{i});
        AIR_options{i}=AIR_options{i}(max(max_idx-20,1):max(max_idx-20,1)+512);
        AIR_options{i}=0.2*AIR_options{i}/max_val;
    end

elseif param.RIR_mode == "TUBS_dynIR"
        AIR_options{1} = load_dynIR(param);

elseif param.RIR_mode == "TUBS_dynIR_fixed" %static RIR as reference
        AIR_options{1} = load_dynIR(param);
        AIR_options{1} = AIR_options{1}(:,1)';

elseif param.RIR_mode == "RCWP"
    irPath = "../00_Database/RCWP_IR";
    ir_files = dir(irPath+'/*.wav');
    k = 1;
    for l = 1:numel(ir_files)
        IR_tmp = audioread([ir_files(l).folder param.FILE_SEP ir_files(l).name]);
        for m = 1:size(IR_tmp, 2)
            AIR_options{k} = IR_tmp(:,m);
            [max_val, max_idx] = max(AIR_options{k});
            AIR_options{k}=AIR_options{k}(max(max_idx-20,1):end);
            AIR_options{k}=0.2*AIR_options{k}/max_val;
            k = k + 1;
        end      
    end

end

% Divide pre-computed RIR options for computed datasets
if param.RIR_mode == "AIR" || param.RIR_mode == "AIR_DC" || param.RIR_mode == "AIRshort" || param.RIR_mode == "RCWP" || param.RIR_mode == "TUBS_dynIR" || param.RIR_mode == "TUBS_dynIR_fixed"
    fraction_train = round(size(AIR_options,1)*setsize_train/setsize_all);
    fraction_dev = min([fraction_train+round(size(AIR_options,1)*setsize_dev/setsize_all), size(AIR_options,1)]);
    fraction_test = min([fraction_train+fraction_dev+round(size(AIR_options,1)*setsize_test/setsize_all), size(AIR_options,1)]);

    if setsize_train > 0
        fraction_train = max([fraction_train, 1]);
        param.RIR.train.options_FE = AIR_options(1:fraction_train);
    end
    if setsize_dev > 0
        fraction_dev = max([fraction_dev, 1]);
        param.RIR.dev.options_FE = AIR_options(fraction_train+1:fraction_dev);
    end
    if setsize_test > 0
        fraction_test = max([fraction_test, 1]);
        param.RIR.test.options_FE = AIR_options(fraction_dev+1:fraction_test);
    end 
end

%% Generate Noise File List
if param.useNoiseDB
    param.noiseDB.train.files   = dir(['..' FILE_SEP '00_Database' FILE_SEP noiseDB.train.name FILE_SEP '*' noiseDB.train.ext]);
    if noiseDB.train.add_white
        param.noiseDB.train.files(end+1).name = 'gen_white';
    end
    if param.enableFENoise
        param.noiseDB_FE.train.files   = dir(['..' FILE_SEP '00_Database' FILE_SEP noiseDB_FE.train.name FILE_SEP '*' noiseDB_FE.train.ext]);
    end
    if setsize_dev > 0
        param.noiseDB.dev.files     = dir(['..' FILE_SEP '00_Database' FILE_SEP noiseDB.dev.name FILE_SEP '*' noiseDB.dev.ext]);
        if param.enableFENoise
            param.noiseDB_FE.dev.files   = dir(['..' FILE_SEP '00_Database' FILE_SEP noiseDB_FE.dev.name FILE_SEP '*' noiseDB_FE.dev.ext]);
        end
        if noiseDB.dev.add_white
            param.noiseDB.dev.files(end+1).name = 'gen_white';
        end
    end
    if setsize_test > 0
        param.noiseDB.test.files    = dir(['..' FILE_SEP '00_Database' FILE_SEP noiseDB.test.name FILE_SEP '*' noiseDB.test.ext]);
        if param.enableFENoise
            param.noiseDB_FE.test.files   = dir(['..' FILE_SEP '00_Database' FILE_SEP noiseDB_FE.test.name FILE_SEP '*' noiseDB_FE.test.ext]);
        end
        if noiseDB.test.add_white
            param.noiseDB.test.files(end+1).name = 'gen_white';
        end
    end
end
        
%% Train Set
if setsize_train > 0
    param.current = 'train';
    for d = 1:param.dataset_division
        param.d = d;
        disp('Creating training data...')
        param.RIR.spec      = param.RIR.train;
        param.noiseDB.spec  = param.noiseDB.train;
        if param.enableFENoise
            param.noiseDB_FE.spec  = param.noiseDB_FE.train;
        end
        speaker_mixing(param,setsize_train,dirList_M,dirList_F,random_M(1:setsize_train*param.dataset_division),random_F(1:setsize_train*param.dataset_division));

    end
end

%% Dev Set
if setsize_dev > 0
    param.current = 'dev';
    param.d = 1;
    disp('Creating development data...')

    param.RIR.spec      = param.RIR.dev;
    param.noiseDB.spec  = param.noiseDB.dev;
    if param.enableFENoise
        param.noiseDB_FE.spec  = param.noiseDB_FE.dev;
    end
    speaker_mixing(param,setsize_dev,dirList_M,dirList_F,random_M(setsize_train+1:setsize_train+setsize_dev),random_F(setsize_train+1:setsize_train+setsize_dev));
end

%% Test Set
if setsize_test > 0
    param.current = 'test';
    param.d = 1;
    disp('Creating test data...')

    param.RIR.spec      = param.RIR.test;
    param.noiseDB.spec  = param.noiseDB.test;
    if param.enableFENoise
        param.noiseDB_FE.spec  = param.noiseDB_FE.test;
    end
    speaker_mixing(param,setsize_test,dirList_M,dirList_F,random_M(setsize_train+setsize_dev+1:end),random_F(setsize_train+setsize_dev+1:end));
end

timeComplete = toc;

