function F = calc_formant_forcsv(voice, Fs, label)
%CALC_FORMANT ‰Wyznaczanie formantów
%
%   F = calc_formant(voice, Fs) ‰Wyznaczenie formantów·
%
%       F : Wyznaczone wartości formantów
%       voice : Próbka wejściowa
%       Fs : Częstotliwość próbkowania
%
%   Copyright 2016 The MathWorks, Inc.

% 

voice_trgt = voice(:, 1);

% Określenie pasma zainteresowania

tap_b = [1 -0.96875];

voice_trgt = filter(tap_b, 1, voice_trgt);

% Okienkowanie

win_func = hamming(length(voice_trgt));
voice_trgt = win_func .* voice_trgt;

% Power Spectral Density (PSD) wyznaczone metodą Yule-Walker'a

Ndeg_ar = 10;
[P, f] = pyulear(voice_trgt, Ndeg_ar, 2048, Fs);
P = 10*log10(P);

% Wyszukiwanie maksimów identyfikujących formanty

[~, Nsmp_pks] = findpeaks(P);
Xhz_pks = f(Nsmp_pks);

F = Xhz_pks(Xhz_pks > 200);

F = F(1:2)'; % Należałoby wykorzystać tę funkcję do stworzenia swojego zbioru uczącego
