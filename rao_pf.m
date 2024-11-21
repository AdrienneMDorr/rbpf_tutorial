clc
clear all
close all
%% Define the model 
f  = @(x) [1*atan(x(1,:)) + 1*x(2,:);          % Dynamics
     1*x(2,:) + 0.3*x(3,:);
    0.92*x(3,:)-0.3*x(4,:);
    0.3*x(3,:)+0.92*x(4,:)];                

h  = @(x)[(0.1*x(1,:).^2).*sign(x(1,:));
     x(2,:) - x(3,:) + x(4,:)];           % Measurement model


x0 = zeros(4,1);             % Initial state

P0 = [1 0 0 0;
      0 1e-6 0 0;
      0 0 1e-6 0;
      0 0 0 1e-6];    % Covariance for the initial state
             

R  = 0.1*eye(2);             % Measurement noise covariance
Q  = 0.01*eye(4);            % Process noise covariance

n = size(x0,1);             % State size
m = size(R,1);                      % Measurement vector length

% Define matrices based on Eqs. 18 and 19.
An   = [1 0 0];
Al   = [1  0.3    0;
          0  0.92 -0.3; 
          0  0.3  0.92];
C    = [0 0  0;
          1 -1 1];
%%%%
% Particle Filter
%%%%
N_Steps = 250;                  % Number of time steps
Np = 100;
x_mean = zeros(4,N_Steps);   % Array for posterior states
w = ones(1,Np);
x = zeros(n,N_Steps+1);     % Array for truth
y = zeros(m,N_Steps);       % Array for measurements

%% %%%%%%%%%%%%%%%% STEP 1: Initialization
  x(:,1) = mvnrnd(x0,P0,1)';               % Random draw to initialize state
  xnp = mvnrnd(x0(1),P0(1,1),Np)';          % CREATE PARTICLES FOR NONlinear states 
  xlp = repmat(x0(2:4),1,Np);          % CREATE PARTICLES FOR linear states
  Pl  = P0(2:4,2:4);
  Pp  = repmat(Pl,[1,1,Np]);             % Initial prior covariance 
  xlf = zeros(size(xlp));                    % % Array for posterior LINEAR states 
  Pf  = zeros(size(Pp));                     % % Array for posterior "LINEAR" covariance


for k = 1:N_Steps
    %Propagate system dynamics
    x(:,k+1) = f(x(:,k)) + mvnrnd(zeros(4,1),Q,1)';
    %Simulate measurements with true x
    y(:,k)   = h(x(:,k)) + mvnrnd(zeros(2,1),R,1)';

    %Measurement update
    ymeas = y(:,k);
    yhat = [(0.1*xnp.^2).*sign(xnp);
            xlp(1,:) - xlp(2,:) + xlp(3,:)];        %Expected measurement

    residual    = repmat(ymeas,1,Np) - yhat;    %Measurement residuals

    %% %%%%%%%%%%%%%%%% STEP 2: Calculate weights (25a)
    for j = 1:Np
      residual_cov = C*Pp(:,:,j)*C' + R;
      w(j) =  exp(-(1/2)*(residual(:,j)'*inv(residual_cov)*residual(:,j)));
    end
    
      w = w/sum(w);           % Normalize the importance weights
      x_mean(1,k) = sum(w.*xnp,2);      % Compute estimate for the nonlinear states

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
      residual_cov         = C*Pp(:,:,j)*C' + R;    % Eq. (22c)
      K         = Pp(:,:,j)*C'*inv(residual_cov);       % Eq. (22d)
      
      yhat      = [(0.1*xnp(:,j).^2).*sign(xnp(:,j));
                   xlp(1,j) - xlp(2,j) + xlp(3,j)];

      xlf(:,j)  = xlp(:,j) + K*(ymeas - yhat);  % Eq. (22a)
      Pf(:,:,j) = Pp(:,:,j) - K*residual_cov*K';          % Eq. (22b)
    end
    x_mean(2:4,k) = mean(xlf,2);    % Compute estimate for the linear states

    %% Algorithm step (4b), equation for PF prediction (25b)
    xnf = xnp;
    for j = 1:Np
        %Create the normal distribution with particles. Use mean and covariance provided in derivation.
      xnp(j) = atan(xnf(j)) + xlf(1,j) + sqrt(An*Pf(:,:,j)*An' + Q(1,1))*randn(1);
    end

    %% Algorithm step (4c) Kalman filter Time Update
    for j = 1:Np
      N         = An*Pf(:,:,j)*An' + Q(1,1);        % Eq. (23c)
      L         = Al*Pf(:,:,j)*An'*inv(N);           % Eq. (23d)
      z         = xnp(j) - atan(xnf(j));                 % Eq. (24a)
      xlp(:,j)  = Al*xlf(:,j) + L*(z - An*xlf(:,j)); % Eq. (23a)
      Pp(:,:,j) = Al*Pf(:,:,j)*Al' + Q(2:end,2:end) - L*N*L'; % Eq. (23b)
    end

    %% STEP 5: Start over (go to the top)    

end


%% Plotting section 

figure(100)
hold on
subplot(2,2,1)
plot(x_mean(1,:),'b')
hold on;
plot(x(1,:),'r')
xlabel('Time (s)')
title('x_1 nonlinear')
xlim([0 250])
legend ('RBPF','True')


subplot(2,2,2)
plot(x_mean(2,:),'b')
hold on;
plot(x(2,:),'r')
xlabel('Time (s)')
title('x_2 linear')
xlim([0 250])
legend ('RBPF','True')
hold on;


subplot(2,2,3)
plot(x_mean(3,:),'b')
hold on;
plot(x(3,:),'r')
xlabel('Time (s)')
title('x_3 linear')
xlim([0 250])
legend ('RBPF','True')
hold on;


subplot(2,2,4)
plot(x_mean(4,:),'b')
hold on;
plot(x(4,:),'r')
xlabel('Time (s)')
title('x_4 linear')
xlim([0 250])
legend ('RBPF','True')
hold on;
%%%%%

figure
hold on
plot(x_mean(1,:),'b')
hold on;
plot(x(1,:),'r')
xlabel('Time (s)')
title('x_1 nonlinear')
xlim([0 250])
legend ('RBPF','True')


figure
plot(x_mean(2,:),'b')
hold on;
plot(x(2,:),'r')
xlabel('Time (s)')
title('x_2 linear')
xlim([0 250])
legend ('RBPF','True')
hold on;


figure
plot(x_mean(3,:),'b')
hold on;
plot(x(3,:),'r')
xlabel('Time (s)')
title('x_3 linear')
xlim([0 250])
legend ('RBPF','True')
hold on;


figure
plot(x_mean(4,:),'b')
hold on;
plot(x(4,:),'r')
xlabel('Time (s)')
title('x_4 linear')
xlim([0 250])
legend ('RBPF','True')
hold on;
