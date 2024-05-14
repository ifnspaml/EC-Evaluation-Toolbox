function count = saveshort(A, filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Write 16Bit short data to file on the disk.
%
%       Input...
%                   A:          MATLAB matrix or vector with data
%                   filename:   Name of file data will be written to
%       Output...
%                   count:      Number of elements written to file
%
%       Braunschweig, 05.10.2011
%   (c)	Prof. Dr.-Ing. Tim Fingscheidt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

infid = fopen(filename,'wb');                                               %Open file for writing
if infid == -1
   error(['SAVESHORT: File ', filename , ' could not be opened!']);
end

count = fwrite(infid, A, 'short');                                          %Write into file

if fclose(infid) ~= 0                                                       %Close file
   error(['SAVESHORT: File ', filename , ' was not closed properly!']);
end

