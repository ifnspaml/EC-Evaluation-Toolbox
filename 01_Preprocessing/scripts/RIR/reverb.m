function [t_reverb]=reverb(IR,fs,decay)

IR(5*fs+1:end) = [];  % max 5 sec!
if nargin == 2
    decay = -60;
end
if decay >= 0
    warning(decay_warn,'Positive decay value is seldomly appropriate!');
end
%   input:
%       IR = impulse response, array
%       fs = sampling rate, in Herz
%
%   output:
%       t_reverb = reverberation time (-60 dB or decay), in seconds
%


%% init
N = length(IR);     % length of impulse response array
if decay > -29
    dyn_range = [-5 -25];
elseif decay > -59
    dyn_range = [-5 -35];
else
    dyn_range = [-40 -60];
end
    
%% calculate EDC = energy decay curve
EDC_temp = zeros(1,N);
for idx = 1:N    
    EDC_temp(idx) = sum(IR(idx:end).^2);
end
EDC_temp = EDC_temp./(EDC_temp(1)+eps);     % norming to total energy of impulse response signal
EDC = 20*log10(EDC_temp);
clear EDC_temp EDC_norm;

%% linear regression analysis
[~,idx_s] = min(abs(EDC(:)-dyn_range(1)));
[~,idx_e] = min(abs(EDC(:)-dyn_range(2)));
q_opt = (EDC(idx_e)-EDC(idx_s))/(idx_e-idx_s+eps);
p_opt = EDC(idx_s)-q_opt*idx_s;
J = sum((EDC(idx_s:idx_e)-(p_opt+q_opt*(idx_s:idx_e))).^2);

for p = p_opt-5:0.5:p_opt+5
    for q = q_opt-1:0.01:min(q_opt+1,0)
        temp = sum((EDC(idx_s:idx_e)-(p+q*(idx_s:idx_e))).^2);
        if temp < J
            p_opt = p;
            q_opt = q;
            J = temp;
        end        
    end
end
clear idx_s idx_e temp J
t_reverb = (decay-p_opt)/(q_opt*fs+eps);
% figure;
% plot(EDC);
% hold on;
% plot(p_opt+q_opt.*(1:t_reverb*fs),'r');