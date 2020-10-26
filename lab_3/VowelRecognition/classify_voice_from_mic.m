 %% 
%
% Klasyfikacja g³osek przechwytywanych z mikrofonu
%
% Copyright 2016 The MathWorks, Inc.

%% Porz¹dki

clear
close all

%% Czêstotliwoœæ próbkowania i czas
Fs = 8000;
T = 3;

data = record_voice_from_mic(Fs, T);

%% Wyznaczenie formantów

Y = calc_formant(data, Fs);

%% Wczytanie danych z formantami dla g³osek

D = readtable('karoldzialowski.csv');
X = D{:, 1:2};
T = D{:, 3};

% klasyfikator KNN
d = fitcknn(X, T, 'NumNeighbors', 4);

% 
% d = fitcdiscr(X, T, 'DiscrimType', 'linear');

% 
% d = fitcdiscr(X, T, 'DiscrimType', 'quadratic');

%% Rozpoznawanie g³osek

C = predict(d, Y);

V = {'a', 'e', 'i', 'o', 'u', 'y'};
disp('Rozpoznana g³oska')
disp(V(C))
