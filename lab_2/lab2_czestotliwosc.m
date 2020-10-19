[s,Fs] = audioread('220.wav');

scut = s(1000:2000);
ACOLS = xcorr(log(abs(fft(scut))));

[pks, locs]=findpeaks(ACOLS);
[peak_value, pi] = max(pks);
peak_index = locs(pi);
f0 = Fs/peak_index

fig = figure();
subplot(2,1,1)
plot(scut)
title('Sygna³')
subplot(2,1,2)
hold on
plot(ACOLS)
scatter(locs, pks)
hold off
title('Autokorelacja')

saveas(fig, 'spectral', 'eps')