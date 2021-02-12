function [forks, ends] = feature_extraction(binarized_image)
%FEATURE_EXTRACTION Summary of this function goes here
%   Detailed explanation goes here
    ends = [];
    forks = [];

    [w, h] = size(binarized_image);

    for i=1:w-2
        for j=1:h-2
            sum = 0;
            for k=0:2
                for l=0:2
                    sum = sum + binarized_image(i+k, j+l);
                end
            end

            if (sum == 2)
                if (binarized_image(i+1, j+1) == 1)
                    ends = [ends; [j+1, i+1]];
                end
            end
            if (sum == 4)
                if (binarized_image(i+1, j+1) == 1)
                    forks = [forks; [j+1, i+1]];
                end
            end
        end
    end

    tol = 3;
    % https://stackoverflow.com/questions/64264043/how-to-group-points-with-given-tolerance-in-matlab?noredirect=1#comment113643342_64264043
    forks = uniquetol(forks, tol, 'ByRows', true, 'DataScale', 1);
    ends = uniquetol(ends, tol, 'ByRows', true, 'DataScale', 1);
end

