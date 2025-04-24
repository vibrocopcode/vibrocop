% Load accelerometer data
num = csvread('humancall/op9RetakeEarRobo10.csv');

% Delete unnecessary information
num(2:2:end,:) = [];

% Extract time and z-axis acceleration data
time = num(:,1);
acc_z = num(:,4);

% Consider only data from 22nd second to 32nd second
start_time = 26; % Start time in seconds
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

% Plot frequency spectrum
figure;
plot(frequencies, fft_acc_z);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum of Z-Axis Accelerometer Data');
grid on;

% Find dominant frequency
[max_mag, max_index] = max(fft_acc_z);
dominant_frequency = frequencies(max_index);

% Perform temporal analysis
% Find peak points
[peak_values, peak_indices] = findpeaks(acc_z_hp, time, 'MinPeakProminence', 0.01);

% Calculate temporal differences
temporal_diffs = diff(peak_indices);

% Plot temporal differences
figure;
histogram(temporal_diffs, 20);
xlabel('Temporal Difference (s)');
ylabel('Frequency');
title('Temporal Differences Between Peaks');

% Display dominant frequency
disp(['Dominant frequency: ', num2str(dominant_frequency), ' Hz']);

% Display temporal differences
disp(['Mean Temporal Difference: ', num2str(mean(temporal_diffs)), ' seconds']);
disp(['Standard Deviation of Temporal Differences: ', num2str(std(temporal_diffs)), ' seconds']);
