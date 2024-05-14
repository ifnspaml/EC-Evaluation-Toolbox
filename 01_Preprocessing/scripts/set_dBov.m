function [signal,ov,from,to] = set_dBov(signal_in,SLov,setASL, fs)
% [signal] = dBov(signal_in,SLov,setASL)
%
% "SLov = -26":  signal level below the dynamic range
% "signal_in": gives the signal that is to adjust
% "setASL = 1": makes the function to adjust the signal_in acording to
% the Active Speech Level, 'RMS' makes it adjust according to RMS
% ITU-T P.56

% if nargin < 4
%     fast_mode = 1;
% end

actlevel = 'initialize';
if nargin < 3
    setASL = 0;
end
if nargin < 4
    fs = 16000;
end

[current_path, ~] = fileparts(mfilename('fullpath'));

factor=0;
runs=0;
count=0;
%signal = loadshort(signal_in);
signal = signal_in;
signal = signal - mean(signal);

if sum(abs(signal)) == 0
    disp('input signal is equal zero');
    ov = 0;
    from = -999;
    to = -999;
else
    filenum=randi(100000,1,1); % to prevent problems when set_dBov is used by more than 1 functions/scripts at the same time
    while (exist( fullfile( [current_path '\temp_in' num2str(filenum) '.raw'] ), 'file' ) ~= 0)
        filenum=filenum+1;
    end
    filenum=num2str(filenum);
    while (round(factor*100)/100 ~= 1 && runs<50) %round(actlevel*10)/10 ~= SLov
        runs=runs+1;
        if factor ~= 0
            signal = factor .* signal;
        end
        count=count+1;

        saveshort(signal*32768,[current_path '\temp_in' filenum '.raw']);
        [~,result] = system(['"' current_path '\actlevel.exe" -q -sf ' int2str(fs) ' -rms "' current_path '\temp_in' filenum '.raw"']);

        ov(count)=max(abs(loadshort([current_path '\temp_in' filenum '.raw'])));

        if (setASL == 1)
            idx = strfind(result,'ActLev[dB]: ');
        else
            if (setASL == 0)
                idx = strfind(result,'RMSLev[dB]: ');
            end
        end
        actlevel = str2double(result(idx+12:idx+19));
        delete([current_path '\temp_in' filenum '.raw']);

        if runs == 1;
            from=actlevel;
        end

        factor = 10^((SLov - actlevel)/20);

    end
    % disp(count)       %DEBUG: number of iterations/ finished
    to=actlevel;

    signal = min(1, signal);
    signal = max(-1,signal);

end
