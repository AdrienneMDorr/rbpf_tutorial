%% Artificial map
%Authors: Vaisnav T and Humberto R.
%Created: July 23,2021
%%This function receives one x,y point and returns the magnetic field
%%intensity. The range for x and x is theoritically infinite but the map is
%%currently better defined in the range x =[-1,1]; y=[-3,3] approximately.
%%The ``operational range'' can be modified by redifining the mean and std
%%of the gaussian distributions.
% Example of use:
% scalar_intensity = aMap(1.2,3)


function ms_scalar = aMap(x,y)
    %Definition of the means and stds
    % mu1 = [-3.5; 12.15]; std1 = [0.5; 0.5].*eye(2);
    % mu2 = [-3; 10.6]; std2 = [1.2;2.3].*eye(2);
    % mu3 = [-0.4; -0.2]; std3 = 3*std1;
    % mu4 = [3; 2.7]; std4 = 1.5*std3;
    % mu5 = [4; 7.3]; std5 = 0.75*std3;
    % mu6 = [-2.5;0]; std6 = [3;3.3].*eye(2);
    % mu7 = [2; -3.5]; std7 = [1.5;2.4].*eye(2);
    % mu8 = [0.75; -2.5]; std8 = [3.5;3.4].*eye(2);
    % mu9 = [0.; -5]; std9 = [1.5;2.3].*eye(2);
    % mu10 = [-4; -8.5]; std10 = [1.5;1.9].*eye(2);


    mu1 = [-4.5; 12.15]; std1 = 3*[0.5; 0.2].*eye(2);
    mu2 = [-3.5; 7.6]; std2 = 3*[0.8; 0.4].*eye(2);
    mu3 = [-0.4; -0.2]; std3 = 3*3*std1;
    mu4 = [-4; -7.8]; std4 = 3*1.5*std3;
    mu5 = [-2.8; -9.9]; std5 = 3*0.75*std3;
    mu6 = [4;0]; std6 = 3*[1;1.3].*eye(2);
    mu7 = [0.8; -3.5]; std7 = 3*[0.5;0.4].*eye(2);
    mu8 = [0.75; -7.5]; std8 = 3*[0.5;0.4].*eye(2);
    mu9 = [3.4; -8]; std9 = 3*[0.5;0.3].*eye(2);
    mu10 =[-0.3; -2.5]; std10 = 3*[0.1;0.4].*eye(2);

    %Define the minimum measurement value to offset plot.
    min_vq = 4.2336e+04; 
    %Define maximum values.
    max1_vq = 5.3357e+04; max2_vq = 50934.7; max3_vq = 51596.7;
    max4_vq = 49244.5; max5_vq = 49545.6;
    %Some of the amplitudes are computed as differential values.
    amp1_vq = max1_vq - min_vq;
    amp2_vq = max2_vq - min_vq;
    amp3_vq = max3_vq - min_vq;
    amp4_vq = max4_vq - min_vq;
    amp5_vq = max5_vq - min_vq;
    %Some other will be assigned as percentages of the min value in the
    %plot.
    amp6_vq = 0.1*min_vq;
    amp7_vq = 0.12*min_vq;
    amp8_vq = 0.05*min_vq;
    amp9_vq = 0.1*min_vq;
    amp10_vq = 0.07*min_vq;
    
    %Construct the evaluation point as a vector.
    xv = [x;y];
    %Compute all the amplitudes at the xv point for all gaussians.
    f1 = amp1_vq*exp(-1/2*(xv - mu1)'*inv(std1^2)*(xv - mu1));
    f2 = amp2_vq*exp(-1/2*(xv - mu2)'*inv(std2^2)*(xv - mu2));
    f3 = amp3_vq*exp(-1/2*(xv - mu3)'*inv(std3^2)*(xv - mu3));
    f4 = amp4_vq*exp(-1/2*(xv - mu4)'*inv(std4^2)*(xv - mu4));
    f5 = amp5_vq*exp(-1/2*(xv - mu5)'*inv(std5^2)*(xv - mu5));
    f6 = amp6_vq*exp(-1/2*(xv - mu6)'*inv(std6^2)*(xv - mu6));
    f7 = amp7_vq*exp(-1/2*(xv - mu7)'*inv(std7^2)*(xv - mu7));
    
    %Extra peaks. These were not present in the original map.
    %The extra functions can be ignored by setting INCLUDE_EXTRAS = 0.
    INCLUDE_EXTRAS = 1; %This is not an efficient way but it is readable.
    f8 = amp8_vq*exp(-1/2*(xv - mu8)'*inv(std8^2)*(xv - mu8));
    f9 = amp9_vq*exp(-1/2*(xv - mu9)'*inv(std9^2)*(xv - mu9));
    f10 = amp10_vq*exp(-1/2*(xv - mu10)'*inv(std10^2)*(xv - mu10));

    f_extra = f8 + f9 + f10;
    
    %Addding all gaussians.
    gaussians_sum = f1 + f2 + f3 + f4 + f5 + f6 + f7 + INCLUDE_EXTRAS*f_extra;
    %Now offset the sum and return the intensity value at the requested x,y
    %point.
    ms_scalar = gaussians_sum + min_vq;
end
    