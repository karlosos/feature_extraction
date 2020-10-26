
%% Porz¹dki

clear
close all

%% Czêstotliwoœæ próbkowania i czas
Fs = 8000;
T = 3;

%% Wczytanie danych formantow
data = readtable('karoldzialowski2.csv', 'HeaderLines', 1);
Y = data{:, 1:2};
labels = data{:, 3};

%% Wczytanie danych z formantami dla g³osek

D1 = readtable('karoldzialowski.csv', 'HeaderLines', 1);
D2 = readtable('lukasik_marcin_formants.csv', 'HeaderLines', 1);
D3 = readtable('okonmichal.csv', 'HeaderLines', 1);
D4 = readtable('robertpiatek.csv', 'HeaderLines', 1);
D = [D1; D2; D3; D4]
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

c_mat = confusionmat(C,labels)
confusionchart(c_mat)