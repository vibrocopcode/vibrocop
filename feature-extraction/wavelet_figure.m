num = csvread('humancall/op9loudHuman1.csv');

% Delete unnecessary information    
num(2:2:end,:) = [];    

% Extract all axes
ax = num(:,2) ;
ay = num(:,3) ;
az = num(:,4) ;

% Compute the magnitude vector
acc = sqrt(ax.^2 + ay.^2 + az.^2); % in G's
acc_cmpersec = acc.*980; % convert to cm/s^2

num(:,5) = acc_cmpersec;

% Define time range (15th to 25th second)
start_time = 15; % Start time in seconds
end_time = 25;   % End time in seconds
indices = (num(:,1) >= start_time) & (num(:,1) <= end_time);
time = num(indices,1);
acc_data = num(indices,5);

% Continuous wavelet transform
scales = 1:1:floor(length(acc_data)/2); % Define scales for wavelet transform
coefs = cwt(acc_data, scales, 'morl', 'VoicesPerOctave',48);

% Plot wavelet coefficients
figure('DefaultAxesFontSize',14, 'DefaultAxesFontWeight', 'bold');
imagesc(time, scales, abs(coefs));
colorbar;
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Scale', 'FontSize', 14, 'FontWeight', 'bold');
title('Continuous Wavelet Transform', 'FontSize', 16, 'FontWeight', 'bold');
