function [RIR_FE, RIR_NE, RT60] = RIRgen(RIRlib)
%RIRGEN Summary of this function goes here
%   Detailed explanation goes here

cs = RIRlib.all.cs;
fs = RIRlib.all.fs;
n = RIRlib.all.n;
max_refl = RIRlib.all.order;    %reflection order

if RIRlib.spec.r_mode == "dyn"
    SER_id = RIRlib.curr_SER_id;
    
    dist_SP = RIRlib.spec.dist_SP(SER_id,:);
    dist_SP = random('Uniform', dist_SP(1), dist_SP(2));
    
    dist_LS = RIRlib.spec.dist_LS(SER_id,:);
    if RIRlib.spec.SER_levels(SER_id) > 0
        dist_LS(1) = max([dist_LS(1),dist_SP]);
        dist_LS(2) = max([dist_LS(1),dist_LS(2)]);
    else
        dist_LS(2) = min([dist_LS(2),dist_SP]);
        dist_LS(1) = min([dist_LS(2),dist_LS(1)]);
    end
    dist_LS = random('Uniform', dist_LS(1), dist_LS(2));
    
    a_min = max([2*dist_LS, RIRlib.spec.a(1)]);           %minimal room size for desired distance
    
    a = random('Uniform', a_min, RIRlib.spec.a(2));
    b = a + 2*rand;
    c = random('Uniform', RIRlib.spec.c(1), min([RIRlib.spec.c(2),b+0.5]));
    r = [a, b, c];
    
    z_mic = random('Uniform', RIRlib.spec.z_mic(1), RIRlib.spec.z_mic(2));
    MIC = [rand*a, rand*b, z_mic];
    
    theta = rand*2*pi;
    phi = 0.1*pi*(rand*1.5+0.5-z_mic);
    
    %speaker/loudspeaker position
    LS = MIC + dist_LS*[cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi)]; 
    if LS(1) > r(1) || LS(1) < 0
        LS(1) = MIC(1) - dist_LS*cos(theta);
    end
    if LS(2) > r(2) || LS(2) < 0
        LS(2) = MIC(2) - dist_LS*sin(theta);
    end
    theta = rand*2*pi;
    phi = 0.1*pi*(rand*1.5+0.5-z_mic);
    SP = MIC + dist_SP*[cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi)]; 
    if SP(1) > r(1) || SP(1) < 0
        SP(1) = MIC(1) - dist_SP*cos(theta);
    end
    if SP(2) > r(2) || SP(2) < 0
        SP(2) = MIC(2) - dist_SP*sin(theta);
    end
    
    if size(RIRlib.spec.beta,2) > 1
        % generate reflection coefficients
        for i = 1:6
            beta_tmp = RIRlib.spec.beta{i};
            if numel(beta_tmp) > 1
                if numel(beta_tmp) > 2
                    b_idx       = 2*randi(numel(beta_tmp)/2);
                    beta_tmp    = beta_tmp(b_idx-1:b_idx);
                end
                beta_tmp    = random('Uniform', beta_tmp(1), beta_tmp(2));
            end
            final_beta(i) = beta_tmp;
        end
        surface(1)  = a*c;
        surface(2)  = surface(1);
        surface(3)  = b*c;
        surface(4)  = surface(3);
        surface(5)  = a*b;
        surface(6)  = surface(5);
        mean_alpha  = sum(surface.*(1-final_beta.^2));
        RT60        = 0.161*a*b*c/mean_alpha;
    else
        %option for direct configuration of RT60
        beta_tmp    = RIRlib.spec.beta{1};
        final_beta  = beta_tmp(1) + (beta_tmp(2)-beta_tmp(1))* wblinv(min(((a-RIRlib.spec.a(1))/(RIRlib.spec.a(2)-RIRlib.spec.a(1))),0.999),0.4,1.5);
        RT60        = final_beta;
        n           = ceil(final_beta*fs);
    end
    
    evalc('RIR_FE = rir_generator(cs, fs, MIC, LS, r, final_beta, n, char("omnidirectional"), max_refl)');
    evalc('RIR_NE = rir_generator(cs, fs, MIC, SP, r, final_beta, n, char("omnidirectional"), max_refl)');
    
else
    beta = RIRlib.spec.T(randi(numel(RIRlib.spec.T)));
    RT60 = beta;
    
    if RIRlib.spec.r_mode == "comb"
        a = RIRlib.spec.a(randi(numel(RIRlib.spec.a)));
        b = RIRlib.spec.b(randi(numel(RIRlib.spec.b)));
        c = RIRlib.spec.c(randi(numel(RIRlib.spec.c)));
        r = [a, b, c];
    elseif RIRlib.spec.r_mode == "list"
        r = RIRlib.spec.r(randi(size(RIRlib.spec.r,1)),:);
    else
        error("Room mode not supported. Use 'list' or 'comb'")
    end

    %calculate speaker and microphone position
    if ischar(RIRlib.spec.m)
        if RIRlib.spec.m == "random"
            sm = RIRlib.spec.sm(randi(numel(RIRlib.spec.sm)));

            m = [rand*r(1), rand*r(2), rand*r(3)];

            theta = rand*2*pi;
            phi = rand*pi;

            %TODO: improve randomization algorithm
            s = m + sm*[cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi)]; 
            if s(1) > r(1) || s(1) < 0
                s(1) = m(1) - sm*cos(theta)*sin(phi);
            end
            if s(2) > r(2) || s(2) < 0
                s(2) = m(2) - sm*sin(theta)*sin(phi);
            end
            if s(3) > r(3) || s(3) < 0
                s(3) = m(3) - sm*cos(phi);
            end

        else
            error("Only random mic position implemented")
        end
    else
        error("Only random mic position implemented")
    end
    
    %calculate RIR
    evalc('RIR_FE = rir_generator(cs, fs, m, s, r, beta, n)');
    RIR_NE = [];
end

end

