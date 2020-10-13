clear all

A = imread('DB1_102_6.tif');

filter = fspecial('average', 3);
A_filtered = imfilter(A,filter,'replicate');

level = graythresh(A);
A_histeq = histeq(A_filtered);
binary = imbinarize(A_filtered,'adaptive','ForegroundPolarity','dark');
binary = imcomplement(binary);
binary_from_histeq = imbinarize(A_histeq, 'adaptive');

subplot(2,3,1);
imshow(A);
title("A")

subplot(2,3,2);
imshow(A_filtered);
title("filtered")

subplot(2,3,3);
imshow(A_histeq);
title("histeq")

subplot(2,3,4);
imshow(binary);
title("binary")

thined = bwmorph(binary, 'thin', Inf);
thined = bwmorph(thined, 'spur');
thined = bwmorph(thined, 'clean');

thined_from_histeq = bwmorph(binary_from_histeq, 'thin', Inf);
thined_from_histeq = bwmorph(binary_from_histeq, 'clean');
thined_from_histeq = bwmorph(binary_from_histeq, 'spur');

subplot(2,3,5);
imshow(thined);
title("thined")

subplot(2,3,6); 
imshow(thined_from_histeq);
title("thinned from histeq")

