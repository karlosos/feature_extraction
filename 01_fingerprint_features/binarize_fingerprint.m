function [binarized_image] = binarize_fingerprint(image)
    filter = fspecial('average', 3);
    image_filtered = imfilter(image, filter, 'replicate');

    binary = imbinarize(image_filtered,'adaptive','ForegroundPolarity','dark');
    binary = imcomplement(binary);

    binarized_image = bwmorph(binary, 'thin', Inf);
    binarized_image = bwmorph(binarized_image, 'spur');
    binarized_image = bwmorph(binarized_image, 'clean');
end

