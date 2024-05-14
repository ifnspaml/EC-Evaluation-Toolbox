function [w, w_norm, fs] = window_generation(fl, fs, window)
% Window_generation - Computation of window function
%
% usage: [w, w_norm, fs] = window_generation(fl, fs, window)
%
%        Computation of window function w given frame length, frame shift,
%        and window name
%
%        fl     : frame length [# of samples]
%        fs     : frame shift  [# of samples]
%        window : window name
%  
%        w      : window function
%        w_norm : window normalization factor (if 0.5<fs<1, w_norm ~= 1)
%        fs     : frame shift [# of samples]
%
%        In window function computation, a series of '1' is inserted in the
%        middle of the window if the frame shift is bigger than 0.5.
% 
% Technische Universität Braunschweig, IfN, 2006 - 09 - 20 (Version 1.0)
% (c) Prof. Dr. -Ing. Tim Fingscheidt
%--------------------------------------------------------------------------

%--- Check window name
if isempty(strfind('recthannhammingblackman', lower(window))), 
    error('Input window name is not recognized. Please check it again!');
end; %if

%--- Check frame shift length
while (fs <= 0) || (fs >= fl),
    fprintf('Frame shift size must be smaller than frame length!');             
    fs = input('Please give a new value of frame shift: ');
end; % while

%--- Window definition
fs_percentage	= fs./fl; 
w					= ones(fl, 1);
if fs_percentage <= 0.5,
    eval(['w = ' lower(window) '(fl, ''periodic'');']);
else
    eval(['w([1:fs end-fs+1:end]) = ' lower(window) ...
        '(2.*fs, ''periodic'');']);
end; % if

%--- Computation of normalization factor for speech synthesis
window2 = [w; zeros(size(w))];
w_norm  = sum( window2( (1:round(fl./fs) ) .* fs) );