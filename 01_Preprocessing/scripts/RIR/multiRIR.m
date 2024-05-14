function [FE_options, NE_options, RT60] = multiRIR(RIRlib)
%MULTIRIR Multi-room RIR Generation using Image method
%   Coordinates RIR generation for multiple rooms.

k = 0;

%dynamic room generation
if RIRlib.spec.r_mode == "dyn"
    
    FE_options  = cell(numel(RIRlib.spec.SER_levels), RIRlib.spec.pos_opt);
    NE_options  = cell(numel(RIRlib.spec.SER_levels), RIRlib.spec.pos_opt);
    RT60        = cell(numel(RIRlib.spec.SER_levels), RIRlib.spec.pos_opt);
    
    for j = 1:numel(RIRlib.spec.SER_levels)
        for i = 1:RIRlib.spec.pos_opt
            disp(i)
            RIRlib.curr_SER_id = j;
            [FE_options{j,i}, NE_options{j,i},RT60{j,i}] = RIRgen(RIRlib);	% generate RIRs of one room
        end
    end

% computes RIR for all possible combinations of room dimensions a,b,c
else
    
    %no second, coupled NE IR list
    NE_options = [];
    
    if RIRlib.spec.r_mode == "comb"

    % store possible room dimensions
    all_a = RIRlib.spec.a;
    all_b = RIRlib.spec.b;
    all_c = RIRlib.spec.c;

    % compute number of RIRs
    num_rooms = numel(all_a) * numel(all_b) * numel(all_c);
    FE_options = cell(num_rooms, RIRlib.spec.pos_opt);

    % iterate through combinations
    for a = all_a
        RIRlib.spec.a = a;
        for b = all_b
            RIRlib.spec.b = b;
            for c = all_c
                RIRlib.spec.c = c;
                k = k + 1;
                for i = 1:RIRlib.spec.pos_opt
                    [FE_options{k,i},~,~] = RIRgen(RIRlib); % generate RIRs of one combination
                end
            end
        end
    end

    % computes RIR for a list of rooms r
    elseif RIRlib.spec.r_mode == "list"

        % store all possible rooms
        all_r = RIRlib.spec.r;

        % compute number of RIRs
        num_rooms = size(all_r,1);
        FE_options = cell(num_rooms, RIRlib.spec.pos_opt);

        % iterate through list
        for j = 1:num_rooms
            RIRlib.spec.r = all_r(j,:);
            for i = 1:RIRlib.spec.pos_opt
                [FE_options{j,i},~,~] = RIRgen(RIRlib);	% generate RIRs of one room
            end
        end   
    end
end

end

