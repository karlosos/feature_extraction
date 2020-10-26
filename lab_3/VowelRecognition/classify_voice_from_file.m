
%% Porz�dki

clear
close all

%% Cz�stotliwo�� pr�bkowania i czas
Fs = 8000;
T = 3;

%% Wczytanie danych formantow
data = readtable('karoldzialowski2.csv', 'HeaderLines', 1);
Y = data{:, 1:2};
labels = data{:, 3};

%% Wczytanie danych z formantami dla g�osek

D = readtable('karoldzialowski.csv', 'HeaderLines', 1);
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

c_mat = confusionmat(C,labels)
confusionchart(c_mat)