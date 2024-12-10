%% Code to read full size orthomosaic and cut it into smaller patches to be used for faster RCNN object detector
% Patches are saved as geotiffs
% This code needs to be run in advance of the damage assessment pipeline
% 1st September 2023


% Define the input GeoTIFF file
inputGeoTIFF = 'colonair_ortho_2.tif';

% Define the desired overlap percentage (e.g., 50%)
overlapPercentage = 50;

% Read the GeoTIFF
info = geotiffinfo(inputGeoTIFF);
[A, R] = geotiffread(inputGeoTIFF);

% Create a meshgrid based on the specified x and y ranges
[x, y] = meshgrid(R.XWorldLimits(1):R.CellExtentInWorldX:R.XWorldLimits(2), ...
                  R.YWorldLimits(2):-R.CellExtentInWorldY:R.YWorldLimits(1));

x = x(1:end-1, 1:end-1);
y = y(1:end-1, 1:end-1);


A = A(:,:,1:3);

% Define the size of the patches (e.g., 642 pixels)
patchSize = 950;

overlapSize = floor(patchSize * overlapPercentage / 100);


% Get the size of the GeoTIFF
[rows, cols, ~] = size(A);

% Calculate the number of patches in each dimension
numPatchesRows = floor((rows - overlapSize) / (patchSize - overlapSize));
numPatchesCols = floor((cols - overlapSize) / (patchSize - overlapSize));

tic
% Loop through rows and columns to cut the GeoTIFF into patches
for i = 1:numPatchesRows
    for j = 1:numPatchesCols
          % Calculate the starting and ending row and column indices for the patch
        startRow = (i - 1) * (patchSize - overlapSize) + 1;
        endRow = min(startRow + patchSize - 1, rows);
        startCol = (j - 1) * (patchSize - overlapSize) + 1;
        endCol = min(startCol + patchSize - 1, cols);
        
        % Extract the patch from the GeoTIFF
        patch = A(startRow:endRow, startCol:endCol, :);

        patchX = x(startRow:endRow, startCol:endCol, :);
        patchY = y(startRow:endRow, startCol:endCol, :);

        
         % Define the output file name for the patch (you can customize this)
        outputFileName = sprintf('Colonair2_patches_950_50overlap/Colonair2_patch_%d_%d.tif', i, j);

        % figure
        % imshow(patch)
    
        % Calculate the geographic extent of the patch
        xmin = min(min(patchX));
        xmax = max(max(patchX));
        ymin = min(min(patchY));
        ymax = max(max(patchY));

        % Create a maprefcells object for the patch
        mapRef = maprefcells([xmin, xmax], [ymin, ymax], size(patch));
        
        % Save the patch to a new GeoTIFF file with the maprefcells
        geotiffwrite(outputFileName, flipud(patch), mapRef, 'CoordRefSysCode', 32651)%, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
        
        % Store the maprefcells object in the cell array
        mapRefCellsArray{i, j} = mapRef;
        
        % Display progress
        fprintf('Saved patch %d_%d\n', i, j);
    end
end

toc