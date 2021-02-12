 %% 
%
% Klasyfikacja g�osek przechwytywanych z mikrofonu
%
% Copyright 2016 The MathWorks, Inc.

%% Porz�dki

clear
close all

%% Cz�stotliwo�� pr�bkowania i czas
Fs = 8000;
T = 3;

data = record_voice_from_mic(Fs, T);

%% Wyznaczenie formant�w

Y = calc_formant(data, Fs);

%% Wczytanie danych z formantami dla g�osek

D = readtable('karoldzialowski.csv');
X = D{:, 1:2};
T = D{:, 3};

% klasyfikator KNN
d = fitcknn(X, T, 'NumNeighbors', 4);

% 
% d = fitcdiscr(X, T, 'DiscrimType', 'linear');

% 
% d = fitcdiscr(X, T, 'DiscrimType', 'quadratic');

%% Rozpoznawanie g�osek

C = predict(d, Y);

V = {'a', 'e', 'i', 'o', 'u', 'y'};
disp('Rozpoznana g�oska')
disp(V(C))
