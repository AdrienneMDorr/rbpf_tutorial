%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   RBPF MODEL               %
%   5-State Unicycle Model   %
%   Added Biases             %
%   Heat Map Meas Update     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all
format long

visualize_map
%% Define the model 

%generate v and w inputs (or measurements) for turtlebot
time = 0;
dt = 1;
vLimits = [0.1,0.5]; %m/s
wLimits = [-0.2618,0.2618]; %rad/s
a1 = rand; b1 = rand; %for traj shape for tb1

fn  = @(x,theta, v,omega) [x(1) + v*cos(theta)*dt;  % Nonlinear Dynamics
                           x(2) + v*sin(theta)*dt];

fl  = @(x,v,omega) [x(3) + omega*dt; % Linear Dynamics           
                    x(4);
                    x(5)];

h  = @(x)[x(1,:);       % Measurement model
          x(2,:)];           


x0 = [0;-10;pi/2;0.1;0.1];   % Initial state

P0 = [.001 0 0 0 0;
      0 .001 0 0 0;
      0 0 0.01 0 0;
      0 0 0 0 0;
      0 0 0 0 0];   % Initial Covariance
             

R  = 100; %0.01*eye(2);          % Measurement noise covariance
Qn = 0.01*eye(2);          % Process noise covariance for Nonlinear States
Ql = 0.01*eye(3);          % Process noise covariance for Linear States
Q  = 0.01*eye(3);          % Process noise covariance for all states

n = size(x0,1);             % State size
m = size(R,1);              % Measurement vector length
n_nl = 2;                   % State size for nonlinear
n_l  = 3;                   % Index of liner state

% Define matrices based on Eqs. 18 and 19.
An   = zeros(2,3);%[0;0];
Al   = zeros(3,3);
C    = [0 0 0];
Gn   = [1, 0;
       0, 1];
Gl   = [1 0 0;
        0 0 0;
        0 0 0];
%%%%
% Particle Filter
%%%%
N_Steps = 80;                  % Number of time steps
Np = 200;                      % Number of particles
x_mean = zeros(n,N_Steps);      % Array for posterior states
w = ones(1,Np);                 % Initialize Weights
x = zeros(n,N_Steps+1);         % Array for truth
y = zeros(m,N_Steps);           % Array for measurements

%% %%%%%%%%%%%%%%%% STEP 1: Initialization
  x(:,1) = x0; %mvnrnd(x0,P0,1)';             % Random draw to initialize state

  xnp = mvnrnd(x0(1:n_nl),P0(1:n_nl,1:n_nl),Np)'; % CREATE PARTICLES FOR NONlinear states 
  xlp = repmat(x0(n_l:end),1,Np);            % CREATE PARTICLES FOR linear states
  Pl  = P0(n_l:end,n_l:end);                     % Covariance for linear states
  Pp  = repmat(Pl,[1,1,Np]);             % Initial prior covariance 
  xlf = zeros(size(xlp));                % % Array for posterior LINEAR states 
  Pf  = zeros(size(Pp));                 % % Array for posterior "LINEAR" covariance
  accumulated_mean = 0;                  % % Variable to save x,y mean
  Pn = repmat(P0(1:n_nl,1:n_nl),[1,1,N_Steps]);  % % Array to save nonlinear covariance
  Pl = repmat(P0(n_l:end,n_l:end),[1,1,N_Steps]);            % % Array to save linear covariance

% Loop for Number of STATES
% k = # of states
% j = # of particles

for k = 1:N_Steps
    % Calculate velocity inputs
    v = (0.2* (vLimits(2) - vLimits(1)) * sin(a1*time) + 0.5 * (vLimits(2) - vLimits(1)) + vLimits(1));
    omega = 0.2* (wLimits(2) - wLimits(1)) * cos(b1*time) + 0.5 * (wLimits(2) - wLimits(1)) + wLimits(1);
    
    time = time + dt;

    %Propagate system dynamics
    % % Split up propagate steps into linear and nonlinear
    x(n_l:end,k+1)   = fl(x(:,k),v,omega) + Al*x(n_l:end,k) + Gl*randn(3,1)*0.001;     % Eq. (18b)
    x(1:n_nl,k+1) = fn(x(1:n_nl,k),x(3,k+1),v,omega) + An*x(n_l:end,k+1) + Gn*[0.0001;0.0001]*randn(1); % Eq. (18a)
   
    % % Plot to verify propoagate step
    % figure (100)
    % plot(time,x(3,k+1),'b*')
    % % plot(x(1,k+1),x(2,k+1),'b*')
    % hold on

    %Simulate measurements with true x
    y(:,k) = aMap(x(1,k),x(2,k)) +  mvnrnd(0,R,1)';
                            %%%%% y(:,k)   = h(x(:,k)) + C*x(n_l:end,k) + mvnrnd(zeros(2,1),R,1)';  % eq. (18c)
    %Measurement update
    ymeas = y(:,k);
    yhat = zeros(1,Np);
    for p = 1:Np
        yhat(:,p) = aMap(xnp(1,p),xnp(2,p));
    end


    % yhat = [xnp(1,:);
    %         xnp(2,:)];        %Expected measurement

    residual    = repmat(ymeas,1,Np) - yhat;    %Measurement residuals


    %% %%%%%%%%%%%%%%%% STEP 2: Calculate weights (25a)
    for j = 1:Np
        format long
      M = C*Pp(:,:,j)*C' + R; % Residual Covaraince
      like = -.5*randn; %-(1/2)*(residual(:,j)'*inv(M)*residual(:,j));
      w(j) = exp(like);
    end
    w = w/sum(w);           % Normalize the importance weights

    % Apply weights to the particles:
    for i = 1:size(w,2)
      accumulated_mean = accumulated_mean + w(i)*xnp(:,i);
    end
    x_mean(1:n_nl,k) = accumulated_mean;   %Compute mean for nonlinear states
    accumulated_mean = 0;   %reset for next loop
    
    % % Plot Particles
    figure(200)
    hold on
    estimated_plot = plot(x_mean(1,k),x_mean(2,k),'+g','MarkerSize',10,'LineWidth',3,'DisplayName','Estimated','HandleVisibility','off');
    particle_plots = plot(xnp(1,:),xnp(2,:),'.w', 'MarkerSize',1,'DisplayName','Particles','HandleVisibility','off');
    true_plots = plot(x(1,k),x(2,k),'+r','MarkerSize',10,'LineWidth',3,'DisplayName','True','HandleVisibility','off');
    % title('Heatmap with Trajectory Overlay')
    % legend('','Estimated','Particles','True')
    pause(0.2)

    % Estimate covariance for Nonlinear States:
    sumP = 0;
    for i = 1:size(w,2)
        dev = xnp(:,i)-x_mean(1:n_nl,k);
        sumP = sumP + w(i)*(dev*dev');
    end 
    Pn(:,:,k) = sumP; % Estimated Covaraince for nonlinear states     


    %%   STEP 3: Resample. We can insert a condition for resampling only when
    % necessary. For now, we resample at every update.
    index   = systematic_resampling(w);     
    xnp     = xnp(:,index);       % Resampled nonlinear particles
    xlp     = xlp(:,index);       % Resampled linear particles
    Pp      = Pp(:,:,index);      % Resampled covariance matrices
    xlf     = xlf(:,index);       % Resampled linear particles
    Pf      = Pf(:,:,index);      % Resampled covariance matrices     
    
    
    %%   STEP 4: Kalman Filter Measurement update  (4a) 
    for j = 1:Np
        M         = C*Pp(:,:,j)*C' + R;        % Eq. (22c) Residual Covaraiance
        K         = Pp(:,:,j)*C'*inv(M);       % Eq. (22d) Kalman Gain
        yhat      = aMap(xnp(1,j),xnp(2,j));   % Expexted Measurement
        xlf(:,j)  = xlp(:,j) + K*(ymeas - yhat); % Eq. (22a) Posterior linear state update
        Pf(:,:,j) = Pp(:,:,j) - K*M*K';          % Eq. (22b) Posterior linear cov update
    end
    x_mean(n_l:end,k) = mean(xlf,2);    % Compute estimate for the linear states
    Pl(:,:,k) = mean(Pf,'all');         % Compute mean cov linear state to save for plotting
    %% Algorithm step (4b), equation for PF prediction (25b)
    xnf = xnp;
    for j = 1:Np
      %Create the normal distribution with particles. Use mean and covariance provided in derivation.
      xnp(1:n_nl,j) = fn(xnf(:,j),xlf(1,j),v,omega) + An*xlf(:,j) + sqrtm(An*Pf(:,:,j)*An' + Gn*Qn*Gn')*randn(2,1);
    end

    %% Algorithm step (4c) Kalman filter Time Update
    for j = 1:Np
      N         = An*Pf(:,:,j)*An' + Gn*Qn*Gn';        % Eq. (23c)
      L         = Al*Pf(:,:,j)*An'*inv(N);           % Eq. (23d)
      z         = xnp(j) - fn(xnp(:,j),xlf(1,j),v,omega); %Eq (24d) 
      xlp(:,j)  = Al*xlf(:,j) + fl([xnp(:,j);xlf(:,j)],v,omega) + L*(z - An*xlf(:,j)); % Eq. (23a) 
      Pp(:,:,j) = Al*Pf(:,:,j)*Al' + Gl*Ql*Gl - L*N*L'; % Eq. (23b)
    end

    %% STEP 5: Start over (go to the top)    

end


%% Plotting section 
figure(200)
title('Heatmap with Trajectory Overlay')
legend('Estimated','Particles','True')



figure
plot(x_mean(1,:),x_mean(2,:),'b')
hold on
plot(x(1,:),x(2,:),'g')
hold on
xlabel('Meters (m)')
ylabel('Meters (m)')
legend ('RBPF','True')
title('Ground Robot Trajectory')

figure
subplot(3,1,1)
plot(x_mean(1,:),'b')
hold on;
plot(x(1,:),'r')
ylabel('Meters (m)')
xlabel('Time (s)')
title('X Position (x_1 nonlinear)')
legend ('RBPF','True')

subplot(3,1,2)
plot(x_mean(2,:),'b')
hold on;
plot(x(2,:),'r')
ylabel('Meters (m)')
xlabel('Time (s)')
title('Y Position (x_2 nonlinear)')
legend ('RBPF','True')
hold on;

subplot(3,1,3)
plot(x_mean(3,:),'b')
hold on;
plot(x(3,:),'r')
ylabel('Radians (rad)')
xlabel('Time (s)')
title('Heading/Theta (x_3 linear)')
legend ('RBPF','True')
hold on;

% 
% figure
% subplot(2,1,1)
% plot(x_mean(4,:),'b')
% hold on;
% plot(x(4,:),'r')
% xlabel('Time (s)')
% title('V Bias (x_4 linear)')
% legend ('RBPF','True')
% 
% subplot(2,1,2)
% plot(x_mean(5,:),'b')
% hold on;
% plot(x(5,:),'r')
% xlabel('Time (s)')
% title('Omega Bias (x_5 linear)')
% legend ('RBPF','True')
% hold on;

%%% Error Plots
sigx = num2str(sqrt(Pn(1,1,:)));
sigx = str2num(sigx);
sigy = num2str(sqrt(Pn(2,2,:)));
sigy = str2num(sigy);
sigt = num2str(sqrt(Pl(1,1,:)));
sigt = str2num(sigt);

figure
subplot(3,1,1)
plot(x(1,2:end)-x_mean(1,:),'b')
hold on
plot(3*sigx,'r');
hold on
plot(-3*sigx,'r')
ylabel('Meters (m)')
xlabel('Time (s)')
title('X-Position Error')
legend('Error','3-Sigma Error Bounds')
hold on

subplot(3,1,2)
plot(x(3,2:end)-x_mean(3,:),'b')
hold on
plot(3*sigy,'r')
hold on
plot(-3*sigy,'r')
ylabel('Meters (m)')
xlabel('Time (s)')
title('Y-Position Error')
legend('Error','3-Sigma Error Bounds')
hold on

subplot(3,1,3)
plot(x(3,2:end)-x_mean(3,:),'b')
hold on
plot(3*sigt,'r')
hold on
plot(-3*sigt,'r')
ylabel('Radians (rad)')
xlabel('Time (s)')
title('Heading Error')
legend('Error','3-Sigma Error Bounds')
hold on
