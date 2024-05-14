function [actlevel] = get_dBov(signal_in,setASL, fs)

if nargin < 2
    setASL = 0;
end
if nargin < 3
    fs = 16000;
end

[current_path, ~] = fileparts(mfilename('fullpath'));

signal = signal_in;
signal = signal - mean(signal);

if sum(abs(signal)) == 0
    actlevel = -200;
end

filenum=randi(100000,1,1); % to prevent problems when set_dBov is used by more than 1 functions/scripts at the same time
while (exist( fullfile( [current_path '\temp_in' num2str(filenum) '.raw'] ), 'file' ) ~= 0)
    filenum=filenum+1;
end
filenum=num2str(filenum);

saveshort(signal*32768,[current_path '\temp_in' filenum '.raw']);
[~,result] = system(['"' current_path '\actlevel.exe" -q -sf ' int2str(fs) '-rms "' current_path '\temp_in' filenum '.raw"']);
if (setASL == 1)
    idx = strfind(result,'ActLev[dB]: ');
else
    if (setASL == 0)
        idx = strfind(result,'RMSLev[dB]: ');
    end
end
actlevel = str2double(result(idx+12:idx+19));
delete([current_path '\temp_in' filenum '.raw']);

end
