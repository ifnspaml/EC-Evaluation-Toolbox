function [samples] = evenRandom(data, samplesize, axis, mode)
%EVENRANDOM Evenly Distributed Random Sequence
%   Ensures even distribution of random sample sequence based on input data

if nargin < 4
    mode = 'periodic';
end

%% Rearrange Data: samples are picked on first order
if nargin < 3
    size_data = numel(data);
    data = data(:);
elseif axis < 1
    size_data = numel(data);
    data = data(:);
else
    size_data = size(data,axis);
    order = 1:ndims(data);
    order(axis) = [];
    order = [axis order];
    data = permute(data, order);
end

switch mode
    case "periodic"
        samples = [];
        remain = samplesize;
        
		% append random permutations of dataset
        while size_data < remain
            remain = remain - size_data;
            samples = [samples; datasample(data, size_data, 'Replace', false)]; 
        end
        samples = [samples; datasample(data, remain, 'Replace', false)];
        
    case "full"
		% repeat dataset to fit samplesize
        if size_data < samplesize
            factor = ceil(samplesize/size_data);
            data_full = repmat(data, factor, 1);
        else
            data_full = data;
        end
        
        samples = datasample(data_full, samplesize, 'Replace', false);
    otherwise
        error(['Mode ' mode ' not supported'])
end
            
end

