if conf.fsTarget <= db.fs,
	if conf.fsTarget < db.fs,
		disp(['Target sampling rate ' num2str(conf.fsTarget) 'kHz < ' num2str(db.fs) 'kHz current sampling rate. Ok!']);

		% --- Load required FIR LP
		fir_lp_file				= [conf.dir.input.FIR_LP 'FIR_LP_' num2str(db.fs) 'kHz_to_' num2str(conf.fsTarget) 'kHz.mat'];
		if exist(fir_lp_file, 'file') ~= 2,
			error(['Required FIR LP mat file "' fir_lp_file '" not found! Aborting.']);
		else
			tmp_load				= load(fir_lp_file);
			tmp_vars				= fieldnames(tmp_load);
			if numel(tmp_vars) == 1
				 fir_lp_coeffs = tmp_load.(tmp_vars{1});
				 clear tmp_load tmp_vars
			else
				 error(['FIR LP mat file "' fir_lp_file '" having more than 2 variables! Aborting.']);
			end
		end
		% ---
	else
		disp(['Target sampling rate ' num2str(conf.fsTarget) 'kHz matches ' num2str(db.fs) 'kHz current sampling rate. Just moving files!']);
	end
else
	error(['Requested sample rate conversion (' num2str(db.fs) 'kHz => ' num2str(conf.fsTarget) 'kHz) not supported! Aborting.']);
end

for targetDir = processDirs,
	% --- Read input directory
	srcPath							= [db.root db.pathData targetDir{1} FILE_SEP];
	disp(['Reading input directory ' srcPath ' looking for ' db.fileExt ' files.']);
	dirList							= listFiles(srcPath, ['*' db.fileExt]);
	% ---
	
end