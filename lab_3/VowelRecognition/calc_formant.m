function F = calc_formant(voice, Fs)
%CALC_FORMANT �Wyznaczanie formant�w
%
%   F = calc_formant(voice, Fs) �Wyznaczenie formant�w�
%
%       F : Wyznaczone warto�ci formant�w
%       voice : Pr�bka wej�ciowa
%       Fs : Cz�stotliwo�� pr�bkowania
%
%   Copyright 2016 The MathWorks, Inc.

% 

voice_trgt = voice(:, 1);

% Okre�lenie pasma zainteresowania

tap_b = [1 -0.96875];

voice_trgt = filter(tap_b, 1, voice_trgt);

% Okienkowanie

win_func = hamming(length(voice_trgt));
voice_trgt = win_func .* voice_trgt;

% Power Spectral Density (PSD) wyznaczone metod� Yule-Walker'a

Ndeg_ar = 10;
[P, f] = pyulear(voice_trgt, Ndeg_ar, 2048, Fs);
P = 10*log10(P);

% Wyszukiwanie maksim�w identyfikuj�cych formanty

[~, Nsmp_pks] = findpeaks(P);
Xhz_pks = f(Nsmp_pks);

F = Xhz_pks(Xhz_pks > 200);

F = F(1:2)'; % Nale�a�oby wykorzysta� t� funkcj� do stworzenia swojego zbioru ucz�cego
