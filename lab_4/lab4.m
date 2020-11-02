file = "5 WlazlKotek (pianino).wav";
[s,Fs] = audioread(file);


fig = figure();
subplot(5,1,1)
plot(s)
title(file)

subplot(5,1,2)
window_length = Fs * 0.1;
f0 = pitch(s, Fs, 'WindowLength', window_length, 'OverlapLength', 0, 'Method', 'CEP');
f0 = repelem(f0, window_length);
plot(f0)
title('CEP')

subplot(5,1,3)
f0 = pitch(s, Fs, 'WindowLength', window_length, 'OverlapLength', 0, 'Method', 'PEF','MedianFilterLength', 10);
f0 = repelem(f0, window_length);
plot(f0)
title('PEF')

subplot(5,1,4)
f0 = pitch(s, Fs, 'WindowLength', window_length, 'OverlapLength', 0, 'Method', 'NCF');
f0 = repelem(f0, window_length);
plot(f0)
title('NCF')

subplot(5,1,5)
f0 = pitch(s, Fs, 'WindowLength', window_length, 'OverlapLength', 0, 'Method', 'SRH');
f0 = repelem(f0, window_length);
plot(f0)
title('SRH')
