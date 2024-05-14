function d = speaker_mixing(param,setsize,dirList_M,dirList_F,random_M,random_F)
    
    d = param.d;

    generatorSettings = rng(1); %seeds the random number generator using the seed so that randi produces a predicatble sequence of numbers

    splitPattern = round(param.splitPattern * setsize);

    param.curr_path = [param.out_path param.FILE_SEP 'files' param.FILE_SEP param.current param.FILE_SEP];
    
    %% First half of Male and Female speakers of the random list used for FE
    for i = 1 : 0.5*splitPattern(1) %M-F
        disp(i)
        m = 0+i;
        t = (d-1)*0.5*splitPattern(1);
        n = splitPattern(1)*(param.dataset_division+1-i);
        
        fileListFE_M = dir([dirList_M{random_M(m+t)} '*.wav']);
        fileListFE_F = dir([dirList_F{random_F(m+t)} '*.wav']);
        fileListNE_M = dir([dirList_M{random_M(n-t)} '*.wav']);
        fileListNE_F = dir([dirList_F{random_F(n-t)} '*.wav']);

        param.speaker = m+t;
            
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            
            max_ch = 1;

            for change_idx = 1:max_ch
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
        
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])

        mixing_auxil(fileListNE_M, fileListFE_F, param);
        param.speaker = setsize+m+t;
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
        
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        mixing_auxil(fileListFE_M, fileListNE_F, param);
        
        param.speaker = n-t;
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
        
        mixing_auxil(fileListNE_F, fileListFE_M, param);
        param.speaker = setsize+n-t;
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
        
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        mixing_auxil(fileListFE_F, fileListNE_M, param);
    end
    
    %%
    for i = 1 : splitPattern(2) %M-M
        disp(i)
        m = splitPattern(1)*(param.dataset_division) + i;
        t = (d-1)*splitPattern(2);
        n = splitPattern(1)*(param.dataset_division)+splitPattern(2)*(param.dataset_division)+splitPattern(3)*(param.dataset_division)+1-i; % 70-56
        fileListFE_M = dir([dirList_M{random_M(m+t)} '*.wav']);
        fileListNE_M = dir([dirList_M{random_M(n-t)} '*.wav']);
        
        param.speaker = m+t;
        param.curr_path = [param.out_path param.FILE_SEP 'files' param.FILE_SEP param.current param.FILE_SEP];
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
            
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        mixing_auxil(fileListNE_M, fileListFE_M, param);
        param.speaker = setsize+m+t;
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
        
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        mixing_auxil(fileListFE_M, fileListNE_M, param);
    end
    
    %%
    for i = 1 : splitPattern(3) %F-F
        disp(i)
        m = splitPattern(1)*(param.dataset_division) + i;
        t = (d-1)*splitPattern(3);
        n = splitPattern(1)*(param.dataset_division)+splitPattern(2)*(param.dataset_division)+splitPattern(3)*(param.dataset_division)+1-i; % 70-56
        fileListFE_F = dir([dirList_F{random_F(m+t)} '*.wav']);
        fileListNE_F = dir([dirList_F{random_F(n-t)} '*.wav']);
        
        param.speaker = n-t;
        param.curr_path = [param.out_path param.FILE_SEP 'files' param.FILE_SEP param.current param.FILE_SEP];
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
            
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])

        mixing_auxil(fileListNE_F, fileListFE_F, param);
        param.speaker = setsize+n-t;
        
        mkdir([param.curr_path param.FILE_SEP 'echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'lin_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'farend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_speech' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_mic' param.FILE_SEP 'sp' int2str(param.speaker)])
        mkdir([param.curr_path param.FILE_SEP 'nearend_noise' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        if param.IR_pathChange(2) > 0
            for change_idx = 1:3
                mkdir([param.curr_path param.FILE_SEP 'echo_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
                mkdir([param.curr_path param.FILE_SEP 'mic_ch' int2str(change_idx) param.FILE_SEP 'sp' int2str(param.speaker)])
            end
        end
        if param.generate_shortIR
            mkdir([param.curr_path param.FILE_SEP 'short_echo' param.FILE_SEP 'sp' int2str(param.speaker)])
        end
        
        mkdir([param.curr_path param.FILE_SEP 'meta' param.FILE_SEP 'sp' int2str(param.speaker)])
        
        mixing_auxil(fileListFE_F, fileListNE_F, param);
    end
    
    
end   


     
