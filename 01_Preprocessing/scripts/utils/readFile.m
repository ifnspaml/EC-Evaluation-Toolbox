function [ filelist ] = readFile( filepath )
%READFILE Summary of this function goes here
%   Detailed explanation goes here

fid         = fopen(filepath,'r');
A           = textscan(fid, '%s');
fclose (fid);

filelist    = A {1,1};
end

