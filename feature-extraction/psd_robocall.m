% Load the accelerometer vibration data
%data = readmatrix('humancall/cdata50.csv');

data = readmatrix('robocall/data49.csv');

% Extract the acceleration values from the 7th second to the 17th second
t_start = 20; % Start time in seconds
t_end = 30; % End time in seconds
idx_start = find(data(:, 1) >= t_start, 1);
idx_end = find(data(:, 1) <= t_end, 1, 'last');
acceleration = data(idx_start:idx_end, 2:end);

% Compute the power spectral density (PSD) of the acceleration signal
fs = 1/mean(diff(data(:, 1))); % Sampling frequency
nfft = 2^nextpow2(size(acceleration, 1)); % Number of FFT points
window = hann(size(acceleration, 1)); % Hann window
[Pxx, f] = pwelch(acceleration, window, [], nfft, fs); % PSD estimate

% Count the number of spikes greater than 100 dB/Hz
num_spikes = sum(10*log10(Pxx(:, 1)) > 100) + sum(10*log10(Pxx(:, 2)) > 50);

% Print the results
fprintf('Number of spikes greater than 100 dB/Hz: %d\n', num_spikes);

% Plot the PSD estimate
plot(f, 10*log10(Pxx(:, 1)), 'b', f, 10*log10(Pxx(:, 2)), 'r')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')
legend('X-axis', 'Y-axis')