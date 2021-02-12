clear all

A = imread('images/DB3_102_6.tif');

filter = fspecial('average', 3);
A_filtered = imfilter(A,filter,'replicate');

level = graythresh(A);
binary = imbinarize(A_filtered,'adaptive','ForegroundPolarity','dark');
binary = imcomplement(binary);
thined = bwmorph(binary, 'thin', Inf);
spur = bwmorph(thined, 'spur');
clean = bwmorph(spur, 'clean');

fig = figure();
subplot(2,3,1);
imshow(A);
title("original image")

subplot(2,3,2);
imshow(A_filtered);
title("low pass")

subplot(2,3,3);
imshow(binary);
title("binarized")

subplot(2,3,4);
imshow(thined);
title("thined")

subplot(2,3,5);
imshow(spur);
title("spur")

subplot(2,3,6);
imshow(spur);
title("clean")

saveas(fig, 'output/binarization_steps', 'eps')
