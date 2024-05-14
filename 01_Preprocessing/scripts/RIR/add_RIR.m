function [mixed] = add_RIR(Signal,param)
%ADD_RIR Summary of this function goes here
%   Detailed explanation goes here

switch param.RIR_mode
    case "imagePRE"
        if param.secure_random
            mixed                   = fftfilt(param.RIR.spec.RIR_list{1},Signal);   %fastest solution
            param.RIR.spec.RIR_list = param.RIR.spec.RIR_list(2:end);
        else
            temp_IR = datasample(param.RIR.spec.options(:), 1);
            mixed   = fftfilt(temp_IR{1},Signal);                                   %fastest solution
        end
    case "image"
        temp_IR = RIRgen(param.RIR);
        mixed = fftfilt(temp_IR,Signal);                                            %fastest solution
    case "exp"
        seed = seed_array(seed_var);
        seed_var = seed_var+1;
        T_60 = abs(param.T_60(1)+(param.T_60(2)-param.T_60(1))*rand);
        IR_length = T_60*param.fsTarget*1000;
        temp_IR = IR_GEN(IR_length,param.fsTarget,seed,T_60);
        mixedNE{save_var,j,l} = fftfilt(temp_IR,mixedNE{save_var,j,l}); %fastest solution, no notable difference
        % [mixedNE{save_var,j,l}, ~, ~] = filter_IIR(temp_IR, 1,mixedNE{save_var,j,l}, zeros(length(temp_IR)-1, 1), []);
    otherwise
        error("RIR mode not defined. Use 'image', 'imagePRE' or 'exp'.")
end

end

