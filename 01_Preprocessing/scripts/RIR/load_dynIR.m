function H_gesamt = load_dynIR(param)

h_length = 4000; %max
normalize_impulse_response = false;
signal_duration = param.fsTarget*1000*min(param.dyn_length);
data_per_segment = 5000;

%% load impulse response H
if ~exist('H','var')    
    disp('loading H... ');
    n_files = ceil(signal_duration/data_per_segment);
    H_gesamt = zeros(h_length,signal_duration);
    for n=1:n_files
        d_end = n*data_per_segment;
        if d_end > signal_duration
            load(['..\00_Data\TU_dynIR\11_automatic_sim_car_real_mirror_' num2str(param.dyn_length) '_s\rir_11_automatic_sim_car_real_mirror_' num2str(param.dyn_length) '_s_' num2str(signal_duration) '.mat'],'H');
        else
            load(['..\00_Data\TU_dynIR\11_automatic_sim_car_real_mirror_' num2str(param.dyn_length) '_s\rir_11_automatic_sim_car_real_mirror_' num2str(param.dyn_length) '_s_' num2str(n*data_per_segment) '.mat'],'H');
        end
        H_gesamt(:,(n-1)*data_per_segment+1:min(n*data_per_segment,signal_duration))=H(1:h_length,:);
        clear H;
        if ~mod(n,5000), fprintf('%s','.'), end
    end

    disp(' ...loading finished!');
end

if param.step > 1
    H_gesamt = H_gesamt(:,1:param.step:end);
    dyn_length = param.dyn_length/param.step;
else
    dyn_length = param.dyn_length;
end

if dyn_length < param.fin_length
    repeat = ceil(param.fin_length/(dyn_length));
    H_full = H_gesamt;

    for i=1:repeat
        H_full = [H_full, H_gesamt];
    end

    H_gesamt = H_full;

end

if dyn_length ~= param.fin_length
    H_gesamt = H_gesamt(:,1:param.fin_length*param.fsTarget*1000);
end

if param.freeze > 0
    freeze_point = H_gesamt(:,param.step*param.fsTarget*1000);
    freeze_range = (param.fin_length-param.step)*param.fsTarget*1000;
    H_gesamt(:,param.step*param.fsTarget*1000+1:end) = freeze_point .* ones(h_length,freeze_range);
end

%% normalize impulse responses by norm of "loudest" impulse response vector
if normalize_impulse_response
    max_norm = eps;
    parfor n=1:size(H_gesamt,2)
        max_norm = max( max_norm,norm(H_gesamt(:,n)) );
    end    
    H_gesamt = H_gesamt/max_norm;
end

if param.freeze == 0
    H_gesamt = H_gesamt(:,1)';
end

end

