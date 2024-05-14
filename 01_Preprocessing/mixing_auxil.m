function [mixedNE, mixedFE, noiseNE, returnRef] = mixing_auxil(fileListNE, fileListFE, param)
%MIXING_AUXIL Helper Function for Speaker Mixing using Image method
%   Function for speaker mixing of one speaker pairing (M-F,F-M,M-M,F-F)

if param.RIR_mode == "exp"
    seed_array  = randperm(10000);
    seed_var    = 1;
end

if param.IR_pathChange(2) > 0
    amount = 2;
else
    amount = 1;
end

%% homogenous randomization
if param.secure_random
    iter_num = param.numFE*param.numNE;
    if numel(param.SER) > 1
        SER_list = evenRandom(param.SER, iter_num, -1, 'full');
    end
    if numel(param.SNR) > 1
        SNR_list = evenRandom(param.SNR, iter_num, -1, 'full');
    end
    if param.enableFENoise
        if numel(param.FE_SNR) > 1
            FE_SNR_list = evenRandom(param.FE_SNR, iter_num, -1, 'full');
        end
    end
    if param.useNoiseDB
        param.noiseDB.spec.list         = evenRandom(1:numel(param.noiseDB.spec.files), iter_num, 2, 'full');
        if param.enableFENoise
            param.noiseDB_FE.spec.list         = evenRandom(1:numel(param.noiseDB_FE.spec.files), iter_num, 2, 'full');
        end
    end
    iter_num = iter_num*(param.enableIR_NE+param.enableIR_FE)*amount;
    if param.RIR_mode == "imagePRE" || param.RIR_mode == "AIR" || param.RIR_mode == "AIR_DC" ||param.RIR_mode == "realIR" || param.RIR_mode == "AIRshort" || param.RIR_mode == "RCWP" || param.RIR_mode == "dynIR_f"
        param.RIR.spec.FE_RIR_list     = evenRandom(param.RIR.spec.options_FE, iter_num, -1, 'full');
        if param.enableIR_NE
            param.RIR.spec.NE_RIR_list     = evenRandom(param.RIR.spec.options_NE, iter_num, -1, 'full');
        end
    end
    if param.RIR_mode == "dynIR"
        dyn_IR = param.RIR.spec.options_FE{1};
    end
end

%% Signal mixing
if ~param.test_condition
    selectNE = evenRandom(1:numel(fileListNE), param.numNE*param.numConcatNE*param.numFE).';      %allows numNE to exceed number of available NE files
end

% per NE speaker
for l = 1 : param.numNE
    
    % per FE speaker
    for j = 1 : param.numFE
        
        if ~param.test_condition
            scenario = "DT";
        
            start=(l-1)*param.numConcatNE*param.numFE+(j-1)*param.numConcatNE+1;
            % load near-end
            rawNE = audioread([getfield(fileListNE,{selectNE(start)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(start)},'name')]);
            for k = 1 : param.numConcatNE-1 %utterances 2 - param.numConcatFE are concatenated to the complete FE signal (differing in total length)
                rawNE = vertcat(rawNE,audioread([getfield(fileListNE,{selectNE(start+k)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(start+k)},'name')]));
            end

            % load and concatenate far-end
            selectFE = evenRandom(1:numel(fileListFE), param.numConcatFE).';      %allows numFE to exceed number of available NE files (not adviced)
            rawFE = audioread([getfield(fileListFE,{selectFE(1)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(1)},'name')]); %minimum of 1 utterance is required for FE signal
            for k = 2 : param.numConcatFE %utterances 2 - param.numConcatFE are concatenated to the complete FE signal (differing in total length)
                rawFE = vertcat(rawFE,audioread([getfield(fileListFE,{selectFE(k)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(k)},'name')]));
            end
            
            %% Align signal length
            if numel(rawFE) > numel(rawNE) % FE signal is longer than NE, therefore...
                diff = numel(rawFE)-numel(rawNE);
                position = randi(diff,1); % ... add zeros before and after the NE speech signal...
                mixedNE = vertcat(zeros(position,1), rawNE, zeros(diff-position,1)); % ...and place the NE signal randomly with zeros added before and after
                mixedFE = rawFE;
            elseif numel(rawFE) < numel(rawNE) % NE signal is longer than FE, therefore...
                diff = numel(rawNE)-numel(rawFE);
                position = randi(diff,1); % ... add zeros before and after the FE speech signal...
                mixedNE = rawNE;
                mixedFE = vertcat(zeros(position,1),rawFE, zeros(diff-position,1)); % place the FE signal randomly with zeros added before and after
            else
                mixedNE = rawNE;
                mixedFE = rawFE;
            end
            
        else
            selectNE = evenRandom(1:numel(fileListNE), 10).';
            selectFE = evenRandom(1:numel(fileListFE), 10).';
            len_section = param.len_sec * 1000 * param.fsTarget;
            if param.test_condition == 1
                scenario = "DT";
                rawNE = audioread([getfield(fileListNE,{selectNE(1)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(1)},'name')]);
                rawFE_1 = audioread([getfield(fileListFE,{selectFE(1)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(1)},'name')]);
                rawFE_2 = audioread([getfield(fileListFE,{selectFE(6)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(6)},'name')]);
                
                k = 1;
                while numel(rawNE) < 2 * len_section
                    k = k + 1;
                    rawNE = vertcat(rawNE,audioread([getfield(fileListNE,{selectNE(k)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(k)},'name')]));
                end
                
                rawNE = rawNE(1: 2 * len_section); %restrict len
                
                k = 1;
                while numel(rawFE_1) < len_section
                    k = k + 1;
                    rawFE_1 = vertcat(rawFE_1,audioread([getfield(fileListFE,{selectFE(k)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(k)},'name')]));
                end
                k = 6;
                while numel(rawFE_2) < len_section
                    k = k + 1;
                    rawFE_2 = vertcat(rawFE_2,audioread([getfield(fileListFE,{selectFE(k)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(k)},'name')]));
                end

                rawFE_1 = rawFE_1(1:len_section); %restrict len
                rawFE_2 = rawFE_2(1:len_section); %restrict len
                
                %% Align signal length
                mixedNE = vertcat(zeros(numel(rawFE_1),1),rawNE);
                diff = numel(rawNE)-numel(rawFE_2);
                mixedFE = vertcat(rawFE_1, zeros(diff,1),rawFE_2);
    
                section_lengths = [numel(rawFE_1); diff; numel(rawFE_2)];

            elseif param.test_condition == 2
                scenario = "STFE";
                rawFE_1 = audioread([getfield(fileListFE,{selectFE(1)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(1)},'name')]);
                rawFE_2 = audioread([getfield(fileListFE,{selectFE(6)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(6)},'name')]);
                
                k = 1;
                while numel(rawFE_1) < len_section
                    k = k + 1;
                    rawFE_1 = vertcat(rawFE_1,audioread([getfield(fileListFE,{selectFE(k)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(k)},'name')]));
                end
                rawFE_1 = rawFE_1(1:len_section); %restrict len

                k = 6;
                while numel(rawFE_2) < len_section
                    k = k + 1;
                    rawFE_2 = vertcat(rawFE_2,audioread([getfield(fileListFE,{selectFE(k)},'folder') param.FILE_SEP getfield(fileListFE,{selectFE(k)},'name')]));
                end
                rawFE_2 = rawFE_2(1:len_section); %restrict len

                %% Align signal length
                mixedFE = vertcat(rawFE_1, rawFE_2);
                mixedNE = zeros(numel(rawFE_1)+numel(rawFE_2),1);
                section_lengths = [numel(rawFE_1); numel(rawFE_2)];

            elseif param.test_condition == 3
                scenario = "STNE";
                rawNE_1 = audioread([getfield(fileListNE,{selectNE(1)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(1)},'name')]);
                rawNE_2 = audioread([getfield(fileListNE,{selectNE(6)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(6)},'name')]);
                
                k = 1;
                while numel(rawNE_1) < len_section
                    k = k + 1;
                    rawNE_1 = vertcat(rawNE_1,audioread([getfield(fileListNE,{selectNE(k)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(k)},'name')]));
                end
                rawNE_1 = rawNE_1(1:len_section); %restrict len
                k = 6;
                while numel(rawNE_2) < len_section
                    k = k + 1;
                    rawNE_2 = vertcat(rawNE_2,audioread([getfield(fileListNE,{selectNE(k)},'folder') param.FILE_SEP getfield(fileListNE,{selectNE(k)},'name')]));
                end
                rawNE_2 = rawNE_2(1: len_section); %restrict len
                
                %% Align signal length
                mixedNE = vertcat(rawNE_1, rawNE_2);
                mixedFE = zeros(numel(rawNE_1)+numel(rawNE_2),1);
                section_lengths = [numel(rawNE_1); numel(rawNE_2)];
            end
        end

        
        %% Choose SER/SNR
        if scenario == "STNE"
            local_SER = 999;
        else
            if numel(param.SER) > 1
                if param.secure_random
                    local_SER = SER_list(1);
                    SER_list = SER_list(2:end);
                else
                    num_SER = randi(numel(param.SER));
                    local_SER = param.SER(num_SER);
                end
            else
                local_SER = param.SER;
            end
        end
        if numel(param.SNR) > 1
            if param.secure_random
                local_SNR = SNR_list(1);
                SNR_list = SNR_list(2:end);
            else
                num_SNR = randi(numel(param.SNR));
                local_SNR = param.SNR(num_SNR);
            end
        else
            local_SNR = param.SNR;
        end
        
        if param.enableFENoise
            if param.secure_random
                local_FE_SNR = FE_SNR_list(1);
                FE_SNR_list = FE_SNR_list(2:end);
            else
                num_FE_SNR = randi(numel(param.FE_SNR));
                local_FE_SNR = param.SNR(num_FE_SNR);
            end
        end
        
        L_shift = min([0, local_SER, local_SNR, 1-randi(6)]);
        
        if size(param.dBov,2) > 1
            local_dBovNE = randi(param.dBov) + L_shift;
        else
            local_dBovNE = param.dBov + L_shift;
        end
        local_dBovFE = local_dBovNE - local_SER;
        local_noiseLevel = local_dBovNE - local_SNR;
        if param.enableFENoise
        	local_FEnoiseLevel = local_dBovFE - local_FE_SNR;
        end
        
        
        %% Adjust sound levels (sending side)
        local_dBovRef = randi(param.refLevel);
        if param.SER_mode == "pseudo"
            if local_dBovFE > -90
                if param.onlyFENoise
                    mixedFE = randn(numel(mixedFE),1);
                    [mixedFE, ~, ~, ~] = set_dBov(mixedFE, local_dBovRef, 0, param.fsTarget*1000);
                else
                    [mixedFE, ~, ~, ~] = set_dBov(mixedFE, local_dBovRef, 1, param.fsTarget*1000);
                end
            else
                mixedFE = zeros(numel(mixedFE),1);
            end
        end
        
        %% Generate noise
        if param.enableFENoise
            if local_FEnoiseLevel > -90
                if param.useNoiseDB
                    if param.secure_random
                        noise_file              = param.noiseDB_FE.spec.list(1);
                        param.noiseDB_FE.spec.list = param.noiseDB_FE.spec.list(2:end);
                    else
                        noise_file = randi(numel(param.noiseDB_FE.spec.files));
                    end
                    if getfield(param.noiseDB_FE.spec.files,{noise_file},'name') == "gen_white"
                        noise = randn(numel(mixedFE),1);
                    else
                        noise_aux               = audioread([getfield(param.noiseDB_FE.spec.files,{noise_file},'folder') param.FILE_SEP getfield(param.noiseDB_FE.spec.files,{noise_file},'name')]);
                        len_signal          = numel(mixedFE)-1;
                        len_noise           = numel(noise_aux);
                        if len_noise >= len_signal
                            noise_pos           = randi(len_noise-len_signal);
                            noise               = noise_aux;
                        else
                            if size(noise_aux, 2) > size(noise_aux, 1)
                                noise_aux = noise_aux';
                            end
                            noise   = noise_aux;
                            noise_pos = 1;
                            for i = 1:floor(len_signal/len_noise)
                                noise   = [noise; noise_aux];
                            end
                        end
                        noise               = noise(noise_pos:noise_pos+len_signal);
                    end
                else
                    noise = randn(numel(mixedFE),1);
                end
                [noise, ~, ~, ~] = set_dBov(noise, local_FEnoiseLevel, 0, param.fsTarget*1000);
                if size(noise, 2) > size(noise, 1)
                    noise = noise';
                end
                mixedFE = mixedFE + noise;
            end 
        end
        
        %% Reference signal
        if local_dBovFE < -90 && param.STstatic.enable
            returnRef = randn(numel(mixedFE),1);
            static_level = -randi([param.STstatic.level(1), param.STstatic.level(2)]);
            [returnRef, ~, ~, ~] = set_dBov(returnRef, static_level, 1, param.fsTarget*1000);
        else
            returnRef = mixedFE;
        end 
        
        %% Loudspeaker nonlinearities
        if param.nonlinear
            if size(param.NLfunc, 2) > 1
                curr_param      =  param.NLfunc{randi(size(param.NLfunc, 2))};
                [mixedFE, NL_param] = speaker_nonlin(mixedFE, 'double', curr_param, param.NLoptions);
            else
                [mixedFE, NL_param] = speaker_nonlin(mixedFE, 'double', param.NLfunc, param.NLoptions);
            end
        else
            NL_param = 'none';
        end
        
        %% Apply room impulse responses
        if param.enableIR_FE
            if local_dBovFE > -90
                
                for FE_idx=1:amount
                    switch param.RIR_mode
                        case {"imagePRE", "AIR", "AIR_DC", 'AIRshort', 'RCWP', 'dynIR_f'}
                            if param.secure_random
                                FE_IR = param.RIR.spec.FE_RIR_list{1};
                                if numel(param.RIR.spec.FE_RIR_list) > 1
                                    param.RIR.spec.FE_RIR_list = param.RIR.spec.FE_RIR_list(2:end);
                                end
                            else
                                if param.RIR.spec.r_mode == "dyn"
                                    temp_id = randi(param.RIR.spec.pos_opt);
                                    idx_IR_SER = find(param.RIR.spec.SER_levels == local_SER);
                                    FE_IR = param.RIR.spec.options_FE(idx_IR_SER, temp_id);
                                else
                                    FE_IR = datasample(param.RIR.spec.options_FE(:), 1);
                                end
                                FE_IR = FE_IR{1};
                            end
                        case "dynIR"
                            FE_IR = "dyn";
                            if param.generate_shortIR > 0
                                 short_echo = dyn_filter(dyn_IR(1:param.generate_shortIR,:),mixedFE);
                            end
                            mixedFE = dyn_filter(dyn_IR,mixedFE);
                        case "image"
                            if param.RIR.spec.r_mode == "dyn"
                                param.RIR.curr_SER_id = find(param.RIR.spec.SER_levels == local_SER);
                                [FE_IR, NE_IR] = RIRgen(param.RIR);
                            else
                                [FE_IR, NE_IR] = RIRgen(param.RIR);
                            end
                        case "exp"
                            seed = seed_array(seed_var);
                            seed_var = seed_var+1;
                            T_60 = abs(param.RIR.spec.T(1)+(param.RIR.spec.T(2)-param.RIR.spec.T(1))*rand);
                            IR_length = ceil(T_60*param.fsTarget*1000);
                            FE_IR = IR_GEN(IR_length,param.fsTarget*1000,seed,T_60);
                            FE_IR = 0.2 * FE_IR / max(FE_IR);
                        otherwise
                            error("RIR mode not defined. Use 'image', 'imagePRE' or 'exp'.")
                    end
                    if param.RIR_mode ~= "dynIR"
                        if param.generate_shortIR > 0
                             short_echo = fftfilt(FE_IR(1:param.generate_shortIR),mixedFE);
                        end
                        if param.IR_pathChange(2) > 0
                            FE_full{FE_idx} = 0.1*FE_IR/max(abs(FE_IR));
                            if param.generate_shortIR > 0
                                short_echo_full{FE_idx} = short_echo;
                            end
                        else
                            FE_full{FE_idx} = FE_IR;
                        end
                    else
                        FE_full{FE_idx} = FE_IR;
                    end
                end
                if param.RIR_mode ~= "dynIR"
                    if param.IR_pathChange(2) > 0
                    
                        changed_signals = zeros(1,numel(mixedFE));
                        IR_change_pos = zeros(1,1);
                        if param.test_condition
                            change_idx = numel(section_lengths);
                            end_change = section_lengths(change_idx);
                            start_change = numel(mixedFE) - end_change;
                        else
                            change_idx = 1;
                            end_change = numel(mixedFE);
                            start_change = 0;
                        end
    
                        pos2             = start_change + floor(param.RIR.all.fs*(param.IR_pathChange(1) + (end_change/param.RIR.all.fs-param.IR_pathChange(2)-param.IR_pathChange(1))*rand));
                        IR_change_pos(1) = pos2;
                        len2             = numel(mixedFE)-pos2;
                        fade_time        = floor(param.RIR.all.fs*(rand*(param.IR_fade(2)-param.IR_fade(1))+param.IR_fade(1)));
                        window_sig       = zeros(numel(mixedFE),1);
                        
                        window_sig(pos2+1:end)            = ones(len2,1);
                        if fade_time
                            win                               = hann(2*fade_time);
                            window_sig(pos2+1-fade_time:pos2) = win(1:fade_time);
                        end
                        
                        mix1                 = mixedFE.*(1-window_sig);
                        mix2                 = mixedFE.*window_sig;
                        changed_signals(1,:) = fftfilt(FE_full{2},mix1)+fftfilt(FE_full{1},mix2);   %late part fits original IR
    
                        mixedFE = fftfilt(FE_full{1},mixedFE);  %fastest solution, no notable difference
                        if param.generate_shortIR > 0
                            linFE   = fftfilt(FE_full{1}(1:param.generate_shortIR),returnRef);
                        else
                            linFE = fftfilt(FE_full{1},returnRef);
                        end
                    else
                        mixedFE = fftfilt(FE_IR,mixedFE);  %fastest solution, no notable difference
                        if param.generate_shortIR > 0
                            linFE = fftfilt(FE_IR(1:param.generate_shortIR),returnRef);
                        else
                            linFE = fftfilt(FE_IR,returnRef);
                        end
                    end
                end

            else
                FE_IR = [1];
                FE_full{1} = [1];
            end
        else
            FE_IR = [1];
            FE_full{1} = [1];
        end
        
        if param.enableIR_NE
            switch param.RIR_mode
                case {"AIR", "realIR", 'AIRshort'}
                    if param.secure_random
                        NE_IR = param.RIR.spec.NE_RIR_list{1};
                        mixedNE = fftfilt(NE_IR,mixedNE); 
                        param.RIR.spec.RIR_list = param.RIR.spec.NE_RIR_list(2:end);
                    else
                        NE_IR = datasample(param.RIR.spec.options_FE(:), 1);
                        mixedNE = fftfilt(NE_IR{1},mixedNE);                 
                    end
                case "imagePRE"
                    if param.RIR.spec.r_mode == "dyn"
                        if ~param.enableIR_FE || local_dBovFE <= -90
                            temp_id = randi(param.RIR.spec.pos_opt);
                            idx_IR_SER = find(param.RIR.spec.SER_levels == local_SER);
                        end
                        NE_IR = param.RIR.spec.options_NE(idx_IR_SER,temp_id);
                        NE_IR = NE_IR{1};
                        mixedNE = fftfilt(NE_IR,mixedNE);
                    else
                        if param.secure_random
                            NE_IR = param.RIR.spec.RIR_list{1};
                            mixedNE = fftfilt(NE_IR,mixedNE); 
                            param.RIR.spec.RIR_list = param.RIR.spec.RIR_list(2:end);
                        else
                            NE_IR = datasample(param.RIR.spec.options_FE(:), 1);
                            mixedNE = fftfilt(NE_IR{1},mixedNE);               
                        end
                    end
                case "image"
                    if param.RIR.spec.r_mode == "dyn"
                        mixedNE = fftfilt(NE_IR,mixedNE);
                    else
                        NE_IR = RIRgen(param.RIR);
                        mixedNE = fftfilt(NE_IR,mixedNE);  
                    end
                case "exp"
                    seed = seed_array(seed_var);
                    seed_var = seed_var+1;
                    T_60 = abs(param.RIR.train.T(1)+(param.RIR.spec.T(2)-param.RIR.spec.T(1))*rand);
                    IR_length = T_60*param.fsTarget*1000;
                    NE_IR = IR_GEN(IR_length,param.fsTarget,seed,T_60);
                    mixedNE = fftfilt(NE_IR,mixedNE);  %fastest solution, no notable difference
                otherwise
                    error("RIR mode not defined. Use 'AIR', 'image', 'imagePRE' or 'exp'.")    
            end
        else
            NE_IR = [1];
        end
        
        if param.IR_pathChange(2) == 0
            IR_change_pos = 0;
        end

        %% Adjust audio level (at microphone)
        if param.SER_mode == "d"
            if ~(scenario == "STFE")
                [mixedNE, ~, ~, ~] = set_dBov(mixedNE, local_dBovNE, 1, param.fsTarget*1000);
            end
            if local_dBovFE > -90
                if param.onlyFENoise
                    [mixedFE, ov, from, to] = set_dBov(mixedFE, local_dBovFE, 0, param.fsTarget*1000);
                else
                    [mixedFE, ov, from, to] = set_dBov(mixedFE, local_dBovFE, 1, param.fsTarget*1000);
                end
                FE_factor = 10^((to-from)/20);
                if param.IR_pathChange(2) > 0
                    changed_signals = changed_signals*FE_factor;
                end
                if param.RIR_mode ~= "dynIR"
                    linFE = linFE * FE_factor;
                end
            else
                FE_factor = 1;
                mixedFE = zeros(numel(mixedFE),1);
                if param.RIR_mode ~= "dynIR"
                    linFE = zeros(numel(mixedFE),1);
                end
            end
        elseif param.SER_mode == "pseudo"
            %rebalance to preserve SER at mic
            if ~(scenario == "STFE")
                [mixedNE, ~, ~, ~] = set_dBov(mixedNE, local_dBovNE, 1, param.fsTarget*1000);
            end
            if local_dBovFE > -90
                    if param.onlyFENoise
                        [mixedFE, ov, from, to] = set_dBov(mixedFE, local_dBovFE, 0, param.fsTarget*1000);
                    else
                        [mixedFE, ov, from, to] = set_dBov(mixedFE, local_dBovFE, 1, param.fsTarget*1000);
                    end
                    FE_factor = 10^((to-from)/20);
                    if param.IR_pathChange(2) > 0
                        changed_signals = changed_signals*FE_factor;
                    end
                    if param.RIR_mode ~= "dynIR"
                        linFE = linFE * FE_factor;
                    end
            else
                    FE_factor = 1;
                    if param.RIR_mode ~= "dynIR"
                        linFE = zeros(numel(mixedFE),1);
                    end
            end
        end
        FE_IR = FE_IR*FE_factor;
        for i = 1:size(FE_full,2); FE_full{i} = FE_full{i} * FE_factor; end

        %save meta information
        if ~param.test_condition
            save([param.curr_path 'meta' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.mat'], 'FE_full', 'FE_IR', 'NE_IR', 'IR_change_pos', 'local_SNR', 'local_SER', 'local_dBovRef', 'NL_param', 'scenario');
        else
            save([param.curr_path 'meta' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.mat'], 'FE_full', 'FE_IR', 'NE_IR', 'IR_change_pos', 'local_SNR', 'local_SER', 'local_dBovRef', 'NL_param', 'scenario', 'section_lengths');
        end
        
        %% Generate noise
        if param.enableMicNoise
            if local_noiseLevel > -90
                if param.useNoiseDB
                    if param.secure_random
                        noise_file              = param.noiseDB.spec.list(1);
                        param.noiseDB.spec.list = param.noiseDB.spec.list(2:end);
                    else
                        noise_file = randi(numel(param.noiseDB.spec.files));
                    end
                    if getfield(param.noiseDB.spec.files,{noise_file},'name') == "gen_white"
                        noise = randn(numel(mixedNE),1);
                    else
                        noise               = audioread([getfield(param.noiseDB.spec.files,{noise_file},'folder') param.FILE_SEP getfield(param.noiseDB.spec.files,{noise_file},'name')]);
                        if size(noise, 2) > 1
                            noise = noise(:,randi(size(noise,2)));
                        end
                        len_signal          = numel(mixedNE)-1;
                        
                        
                        while numel(noise)-len_signal < 0
                            noise = [noise; noise];
                        end
                        if numel(noise)-len_signal ~= 0
                            noise_pos           = randi(numel(noise)-len_signal);
                            noise               = noise(noise_pos:noise_pos+len_signal);
                        end
                    end
                else
                    noise = randn(numel(mixedNE),1);
                end

                [noise, ~, ~, ~] = set_dBov(noise, local_noiseLevel, 0, param.fsTarget*1000);

                if size(noise, 2) > size(noise, 1)
                    noise = noise';
                end
                noiseNE = noise;
            else
                noise = zeros(numel(mixedNE),1);
                if param.sepNoise
                    noiseNE = noise;
                end
            end 
        end
   
        if param.IR_pathChange(2) > 0

            max_ch = 1;

            for change_idx = 1:max_ch
                audiowrite([param.curr_path 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], changed_signals(change_idx,:), param.fsTarget*1000)
                mixed_signal = changed_signals(change_idx,:)' + mixedNE + noise;
                mixed_signal = max(-1,min(1,mixed_signal));
                audiowrite([param.curr_path 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], mixed_signal, param.fsTarget*1000)
            end
        end

        audiowrite([param.curr_path 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], mixedNE, param.fsTarget*1000)
        audiowrite([param.curr_path 'echo' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], mixedFE, param.fsTarget*1000)
        if param.RIR_mode ~= "dynIR"
            audiowrite([param.curr_path 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], linFE, param.fsTarget*1000)
        end
        if param.generate_shortIR
            if local_dBovFE <= -90
                short_echo = mixedFE;
            end
            audiowrite([param.curr_path 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], short_echo, param.fsTarget*1000) 
        end
        audiowrite([param.curr_path 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], returnRef, param.fsTarget*1000)
    
        audiowrite([param.curr_path 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], noiseNE, param.fsTarget*1000)
        mixed_signal = mixedFE + mixedNE + noise;
        mixed_signal = max(-1,min(1,mixed_signal));
        audiowrite([param.curr_path 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker) param.FILE_SEP 'file_' int2str(l) '_' int2str(j) '.wav'], mixed_signal, param.fsTarget*1000)
        
        
    end
end

end
