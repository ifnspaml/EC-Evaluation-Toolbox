function [f_s,room,receiver,source,z_wall,c_sound,h_length,angle,highpass,velocity,trajectory_stepsize,trajectory_length,H_beta,center_circle,name,rirs_file,overwriteRIR] = RIRsetup()
%Parameters for the RIR-simulation
overwriteRIR = 0; % existing files will be overwritten without request
f_s = 16000; % Sampling frequency in Hz

room = [1.5,2.9,1]; % Room dimensions [x, y, z] in meter
% TOURAN %room = [1.46,2.87,1];
source = [1.5/2,0.48,0.8];

% TOURAN %source = [1.06,0.81,0.8]; % Position of source in room [x, y, z] in meter

iscircle = 1; % circle around source, set radius and parameters!
center_circle = [.4,1.05,0.8];
radius = 0.3;
H_beta = 0; % this can be added later with the fast add_centerRIR!

receiver_start = [0.1,0.1,0.7]; % Position of receiver in room [x, y, z] in meter
z_wall=[6,8,4;6,6,4];   % Specific wall impedance

% TOURAN %z_wall=[6,8,4;6,6,4];
% z_wall=[25.4,25.4,5.1;1,1,1]; % Specific wall impedance
% the wall normed by impedance of air) [x1, y1, z1; x2, y2, z2]
% Floor: Carpet, Walls: Plywood, Ceiling: Acoustic Tile
% walls with letter 1 are the walls starting in the origin
% ---- Z_Wall ---- %
% marble:               278
% concrete:             252.5
% brick:                977.9
% painted concrete:     48.8
% plaster on concrete:  56.3
% wood:                 43.8
% acoustic tile:        1.9
% heavy curtains:       4.9
% plywood:              25.4
% carpet:               5.1

c_sound = 343; % sound propagation time m/s
angle = 1;  % use angle
highpass = 0;
rirs_file = 5000;      %Maximum amount of RIRs saved to a single-file. maximum should be 5000 for compatibility with 32bit system.

name = '11_sim_car_real_mirror_4s'; % Name under which the impulse response is saved
h_length = 4000 ; % duration of impulse response in samples 4000 at 16kHZ == 1/4 sekunde
trajectory_stepsize = 1; % 1 equals one revolution per trajectory_length
trajectory_length = 16000; % Length of trajectory in samples. muss ganzzahliges von revolution_time sein

revolution_time = 4; % [s] of circle. Overwrites trajectory length if > 0



use_velocity = 0; % either use velocity to calculate receiver ending position flag 1 or give receiver end position

if use_velocity
    velocity = 6.2832/30*18/5; % in km/h (*18/5 for m/s).
    receiver_end = receiver_start+[0 (velocity*5/18*trajectory_length/(trajectory_stepsize*f_s)) 0];
else
    receiver_end = receiver_start+[1 1 0];
    velocity = sqrt((receiver_start(1)-receiver_end(1))^2+(receiver_start(2)-receiver_end(2))^2+(receiver_start(3)-receiver_end(3))^2)/(100*trajectory_length/(f_s*360));% in km/h
end

%% CIR
corner = 0;
if corner==1 % Calculation of Corner
    z_wall(2,:) = [1,1,1];
end



%% Computation of trajectory-points
disp('Computation of trajectory-points ...');

if (iscircle == 1)
    if revolution_time > 0
        trajectory_length = revolution_time * f_s;
    end
    receiver(:,1)= center_circle(1) + radius*cos(pi/2+2.*pi.*(trajectory_stepsize/trajectory_length:trajectory_stepsize/trajectory_length:trajectory_stepsize));
    receiver(:,2)= center_circle(2) + radius*-sin(pi/2+2.*pi.*(trajectory_stepsize/trajectory_length:trajectory_stepsize/trajectory_length:trajectory_stepsize));
    receiver(:,3) = center_circle(3);
else
    receiver = ones(trajectory_length/trajectory_stepsize,3);
    receiver(:,1) = receiver(:,1).*receiver_start(1);
    receiver(:,2) = linspace(receiver_start(2),receiver_end(2),trajectory_length/trajectory_stepsize);
    receiver(:,3) = receiver(:,3).*receiver_start(3);
    receiver(:,1) = linspace(receiver_start(1),receiver_end(1),trajectory_length/trajectory_stepsize);
    receiver(:,2) = linspace(receiver_start(2),receiver_end(2),trajectory_length/trajectory_stepsize);
    receiver(:,3) = linspace(receiver_start(3),receiver_end(3),trajectory_length/trajectory_stepsize);
end


%% 3D-Plot Room
figure('name', 'Plot of simulated room','color','w')
plot3(source(:,1),source(:,2),source(:,3),'g+','DisplayName','source');
hold on;
plot3(receiver(:,1),receiver(:,2),receiver(:,3),'r','DisplayName','receiver','LineWidth',1);

%     plot3(receiver(:,1),receiver(:,2),receiver(:,3),'r-','DisplayName','receiver','LineWidth',1);
%     plot3(receiver((1:end/4),1),receiver((1:end/4),2),receiver((1:end/4),3),'r-','DisplayName','receiver','LineWidth',3);
%
%     text(receiver(size(receiver,1)/4,1),receiver(size(receiver,1)/4,2),receiver(size(receiver,1)/4,3),'Ó','fontname','Wingdings 3','color','red','fontsize',22,'HorizontalAlignment','center','VerticalAlignment','Middle')
%     text(source(:,1),source(:,2),source(:,3),'W','fontname','Webdings','color','green','fontsize',20,'HorizontalAlignment','center')
% plot of numbers
%c = (1:size(receiver,1))';
%text(receiver(:,1),receiver(:,2),receiver(:,3)+c/10,num2str(c));

legend('show');
axis equal; axis([0 room(1) 0 room(2) 0 room(3)]);
box on; xlabel('[m]','FontSize',20,'FontName','Arial','interpreter','latex');ylabel('[m]','FontSize',20,'FontName','Arial','interpreter','latex'); zlabel('z-axis','FontSize',20,'FontName','Arial','interpreter','latex');

end


