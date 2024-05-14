function A = loadshort(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Read file with 16Bit short data from the disk.
%
%       Input...
%                   filename:	String with input path and file name
%       Output...
%                   A:          Signal loaded into MATLAB
%
%       Braunschweig, 05.10.2011
%   (c)	Prof. Dr.-Ing. Tim Fingscheidt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

infid = fopen(filename,'r');                                                %Open file for reading
if infid == -1
   error(['LOADSHORT: File ', filename , ' could not be opened!']);
end

[A, count] = fread(infid, 'short');                                         %Read from file

if fclose(infid) ~= 0                                                       %Close file
   error(['LOADSHORT: File ', filename , ' is not closed properly!']);
end

