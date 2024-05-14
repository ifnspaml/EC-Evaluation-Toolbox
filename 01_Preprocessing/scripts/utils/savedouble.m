function c = savedouble(A, filename)

infid = fopen(filename, 'wb');

if infid == -1,
	error(['SAVESHORT: File ', filename, ' could not be opened!']);
end

c = fwrite(infid, A, 'double');

if fclose(infid) ~= 0,
	error(['SAVESHORT: File ', filename, ' was not closed properly!']);
end