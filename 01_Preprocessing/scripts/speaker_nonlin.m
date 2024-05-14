function [x_NL, NL_param] = speaker_nonlin(x, type, mode, options)
%SPEAKER_NONLIN Simulation of Nonlinearities
%   Models nonlinear properties of speakers

if nargin < 4
    options = [0.2, 0.5, 1, 10, 999];
end
if nargin < 3
    mode = 'Wang';
end

if nargin < 2
	% guess data type
    if max(abs(x)) > 1
        type = 'int16';
    else
        type = 'double';
    end
end

if mode == "Wang"
    % fit data range
    if type == "int16"
        x_in = x / 32768;
    elseif type == "double"
        x_in = x;
    end
    % speaker clipping
    x_max = 0.8;
    x_hard = min(x_in, x_max);
    x_hard = max(x_hard, -x_max);

    % nonlinear amplification
    b = 1.5 * x_hard - 0.3*x_hard.^2;
    gamma = 1;
    a = 4*ones(numel(x_in),1);
    for i = 1:numel(x_in)
        if b(i) <= 0
            a(i) = 0.5;
        end
    end
    c = -a.*b;
    div = 1+exp(c);
    x_NL = gamma*(2./div-1);
    
    % restore original data range
    if type == "int16"
        x_NL = x_NL * 32768;
    end
    NL_param = 'sigmoid';
elseif mode == "Jung"
    % fit data range
    if type == "int16"
        x_in = x;
    elseif type == "double"
        x_in = x * 32768;
    end
    alpha = 0.0001;
    x_NL = atan(alpha*x_in)/alpha;
    
    % restore original data range
    if type == "double"
        x_NL = x_NL / 32768;
    end
    NL_param = 'atan';
elseif mode == "SEF"
    % fit data range
    if type == "int16"
        x_in = x / 32768;
    elseif type == "double"
        x_in = x;
    end
    nu      = datasample(options, 1);
    if nu < 100
%         fun  = @(x) exp(-x.^2/(2*nu));
%         x_NL = zeros(size(x_in));
%         for i=1:numel(x_in)
%             x_NL(i) = integral(fun,0,x_in(i));
%         end
        x_NL=sqrt(pi*nu/2)*erf(x_in/(sqrt(2*nu)));
    else
        x_NL = x_in;
    end
    if type == "int16"
        x_NL = x_NL * 32768;
    end
    NL_param = nu;
end

end

