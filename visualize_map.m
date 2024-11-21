%% Artificial map amap.m
%Authors: Vaisnav T and Humberto R.
%Created: July 23,2021
%This script plots a contour map of the full map. The generated plot is
%only used for visualization purposes.

%Specify the range for plot.
x_range = -7:0.1:7;
y_range = -10:0.1:15;
%Initialize array to store height(Z) values
ms_fake = zeros(numel(y_range),numel(x_range));
%Compute intensities for the x and y ranges.
for j=1:length(x_range)
    for i=1:length(y_range)
        ms_fake(i,j) = aMap(x_range(j), y_range(i));
    end
end

%Plotting block code.
figure(200); clf;
contourf(x_range, y_range, ms_fake);
colorbar;
xlabel("x(m)");
ylabel("y(m)");