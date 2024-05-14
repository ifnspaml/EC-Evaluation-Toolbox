function [ fileList ] = listFiles(d_path, pattern, depths, dirOnly)

if nargin == 2,
	depths = Inf;
	dirOnly = false;
end

if nargin == 3,
	dirOnly = false;
end

if ~exist(d_path, 'dir'),
	error([mfilename '.m: ' d_path ' not existing! Aborting.']);
end

% --- Extract toplevel directories
d				= dir(d_path);
isub			= [d(:).isdir];						% returns logical vector
dirList		= {d(isub).name}';
dirList(ismember(dirList,{'.','..'})) = [];
% ---

if dirOnly,
	tmpPathFileLists{1}	= cellfun(@(x) fullfile(d_path, x), dirList, 'UniformOutput', 0);
else	
	% --- Extract the current levels files
	tmpFileList				= dir(fullfile(d_path, pattern));
	% ---

	% --- Empty cell array to hold the sub directory file lists
	tmpPathFileLists		= cell(length(dirList)+1, 1);
	% ---
	
	% populate for top level dir cellfun is used to prepend path to file name
	tmpPathFileLists{1}	= cellfun(@(x) fullfile(d_path, x), {tmpFileList.name}.', 'UniformOutput', 0);
end

subLength							= 0;
if depths > 0,	
	for i = 1:length(dirList),	
			disp(['Scanning ' fullfile(d_path, dirList{i})]);
			% recursive call for this levels directories
			fileSubList					= listFiles(fullfile(d_path, dirList{i}), pattern, depths-1, dirOnly);
			tmpPathFileLists{i+1}	= fileSubList;
			subLength					= subLength + length(fileSubList);
	end
end
fileList	= cat(1, tmpPathFileLists{:});

if dirOnly && (subLength > 0 || depths == 1), % append trailing filesep
	fileList = strcat(fileList, filesep);
end

end

