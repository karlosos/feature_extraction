%% 
%
% Analiza formantów dla g³osek przechwytywanych z mikrofonu
%
% Copyright 2016 The MathWorks, Inc.

%% Porz¹dki

clear
close all

%% Czêstotliwoœæ próbkowania i czas
Fs = 8000;
T = 3;
formants=zeros(1,3);

label=0; %a e i o u y -> 1 2 3 4 5 6
gloski=['a' 'e' 'i' 'o' 'u' 'y']

for i=1:36
    label=label+1
    disp(['G³oska: ' int2str(label) ' ' gloski(label)]);
    data = record_voice_from_mic(Fs, T);

    %% Wyznaczenie formantów

    Y = calc_formant(data, Fs)
    formants=[formants; Y label];
    if label==6
        label=0;
    end
end;
csvwrite('karoldzialowski2.csv',formants)

