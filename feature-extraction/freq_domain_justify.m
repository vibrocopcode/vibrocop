% Load accelerometer data
num = csvread('humancall/op9RetakeEarRobo1.csv');

% Delete unnecessary information
num(2:2:end,:) = [];

% Extract time and z-axis acceleration data
time = num(:,1);
acc_z = num(:,4);

% Consider only data from 22nd second to 32nd second
start_time = 16; % Start time in seconds
end_time = 36;   % End time in seconds
indices = (time >= start_time) & (time <= end_time);
time = time(indices);
acc_z = acc_z(indices);

% High-pass filtering to remove gravity component
Fs = 1 / mean(diff(time));
acc_z_hp = highpass(acc_z, 18, Fs);

% Perform frequency analysis (Fast Fourier Transform)
N = length(acc_z_hp);
frequencies = (0:N-1) * Fs / N;
fft_acc_z = abs(fft(acc_z_hp));
fft_acc_z = fft_acc_z(1:N/2); % Take only the positive half of the spectrum
frequencies = frequencies(1:N/2); % Corresponding frequencies

% Calculate power spectral density
psd_values = fft_acc_z.^2 / N;

% Plot frequency spectrum
figure;
plot(frequencies, psd_values);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
title('PSD of Z-Axis Accelerometer Data');
grid on;