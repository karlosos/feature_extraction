clear all

A = imread('DB1_102_2.tif');
A = histeq(A)

treshold = isodataAlgorithm(A);
binary = imbinarize(A, 'adaptive');
% binary = imcomplement(binary);
% binary = im2bw(A, iso);
thined = bwmorph(binary, 'thin', Inf);

subplot(1,3,1);
imshow(binary);
title("binary")

subplot(1,3,2); 
imshow(thined);
title("thinned")

spured = bwmorph(thined, 'clean');
spured = bwmorph(spured, 'spur');

subplot(1,3,3); 
imshow(spured);
title("spured")

ends = [];
forks = [];

[w, h] = size(spured);

for i=1:w-2
    for j=1:h-2
        sum = 0;
        for k=0:2
            for l=0:2
                sum = sum + spured(i+k, j+l);
            end
        end
        
        if (sum == 2)
            if (spured(i+1, j+1) == 1)
                ends = [ends; [j+1, i+1]];
            end
        end
        if (sum == 4)
            forks = [forks; [j, i]];
        end
    end
end

tol = 3;
% https://stackoverflow.com/questions/64264043/how-to-group-points-with-given-tolerance-in-matlab?noredirect=1#comment113643342_64264043
forks = uniquetol(forks, tol, 'ByRows', true, 'DataScale', 1);
ends = uniquetol(ends, tol, 'ByRows', true, 'DataScale', 1);

figure()
imshow(spured)
hold on
scatter(forks(:, 1), forks(:, 2), 'x', 'LineWidth', 2.5);
scatter(ends(:, 1), ends(:, 2), 'x', 'LineWidth', 2.5);
hold off