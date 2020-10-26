function data = record_voice_from_mic(Fs, T)
%RECORD_VOICE_FROM_MIC�
%
%   data = record_voice_from_mic(Fs, T)
%
%       data : Nagranie
%       Fs : Cz�stotliwo�� pr�bkowania
%       T : Czas trwania
%
%   Copyright 2016 The MathWorks, Inc.

recObj = audiorecorder(Fs, 24, 1);
disp('Zacznij m�wi� ...');
% mo�na wstawi� odliczanie
recordblocking(recObj, T);
disp('Koniec nagrywania');

% Przekazanie danych do zmiennej
data = getaudiodata(recObj);

% Odszumianie
windowSize = floor(Fs / 10);    % Ustalenie wielko�ci okna filtru
powerSmooth = filter(ones(1, windowSize) / windowSize, 1, data.^2); % Maska filtru

% Filtracja
threshold = max(powerSmooth) / 10;  % 
idx = powerSmooth > threshold;

voice_bgn = find(idx, 1, 'first');
voice_end = find(idx, 1, 'last');

% Ustalanie pocz�tku i ko�ca przedzia��w do wyci�cia
voice_bgn = max(voice_bgn - windowSize, 1);
voice_end = min(voice_end, length(data));

% Wiedzielony fragment d�wi�czny
data = data(voice_bgn:voice_end);
