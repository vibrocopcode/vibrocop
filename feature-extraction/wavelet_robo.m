% Load the accelerometer vibration data
data = readmatrix('robocall/data4.csv');

% Extract the acceleration values from the 7th second to the 17th second
t_start = 19; % Start time in seconds
t_end = 29; % End time in seconds
idx_start = find(data(:, 1) >= t_start, 1);
idx_end = find(data(:, 1) <= t_end, 1, 'last');
acceleration = data(idx_start:idx_end, 2:end);

% Define the wavelet function
wname = 'morl'; % Morlet wavelet
scales = 1:128; % Wavelet scales (powers of 2)

dt = mean(diff(data(:, 1))); % Sampling period

% Perform wavelet analysis on the acceleration data
[cfs, frequencies] = cwt(acceleration, scales, wname, 'SamplingPeriod', dt, 'scal');

% Plot the wavelet coefficients as a spectrogram
% Plot the wavelet coefficients as a spectrogram
t = data(idx_start:idx_end, 1);
figure;
imagesc(t(:), frequencies(:), abs(cfs).^2);
set(gca, 'YDir', 'normal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;