extension = '.tif';
filesAndFolders = dir('images/*'); % contains only.m files

for i = 1:numel(filesAndFolders)
        if ~filesAndFolders(i).isdir
                filename = filesAndFolders(i).name;
                image = imread(['images/', filename]);
                binarized_image = binarize_fingerprint(image);
                fig_pair = figure('Name', filename);
                imshowpair(image, binarized_image, 'montage');
                
                [forks, ends] = feature_extraction(binarized_image);
                
                fig_features = figure('Name', [filename, ' features']);
                imshow(binarized_image)
                hold on
                scatter(forks(:, 1), forks(:, 2), '*g', 'LineWidth', 1);
                scatter(ends(:, 1), ends(:, 2), 'or', 'LineWidth', 1);
                hold off
                
                
                features = [forks ones(size(forks,1), 1)];
                features = [features; ends 2*ones(size(ends,1), 1)];
                
                saveas(fig_pair, ['output/', filename, '_pair.png'], 'png');
                saveas(fig_features, ['output/', filename, '_features.png'], 'png');
                close(fig_pair)
                close(fig_features)
        end
end