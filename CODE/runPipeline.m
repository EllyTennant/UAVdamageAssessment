%clear all, close all, clc
% 1. Run detector on everything (resizes images to 224 on the fly)
% 2. Scale up the detection output boxes to 950
% 3. Run sieve on everything
% 4. Run NMS on boxes
% 5. Run Classifier 1
% 6. If Class 1 = damaged run class 2.


% load geotiff image
% [A, R] = geotiffread('ortho_reprrojected_new.tif');
% A = A(:,:,1:3);


% this contains the list of the cleaned blocks, so we can skip lines 68-97
%load('blockListCleaned.mat')

%% Load trained models 
% fprintf('%s\n', 'Loading trained models ...')
% 
detector = load('/Users/eleanort/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/PhD/UAV damage detection/GitHub/UAVdamageAssessment/FINAL_MODELS/DETECTOR_FINAL0751.mat');
detector = detector.detector_1;

% Load classification network 1 (damaged vs not damaged)
trained_classifier_1 =load('/Users/eleanort/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/PhD/UAV damage detection/GitHub/UAVdamageAssessment/FINAL_MODELS/CLASSIFIER_1_FINAL0809.mat');
trained_classifier_1 = trained_classifier_1.trained_net_1;

% Load classification network 2 (moderate vs major damage)
trained_classifier_2 = load('/Users/eleanort/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/PhD/UAV damage detection/GitHub/UAVdamageAssessment/FINAL_MODELS/CLASSIFIER_2_FINAL0838.mat');
trained_classifier_2 = trained_classifier_2.trained_net_1;


% load sieve network
trained_sieve = load('/Users/eleanort/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/PhD/UAV damage detection/GitHub/UAVdamageAssessment/FINAL_MODELS/SIEVE_FINAL0977.mat');
trained_sieve = trained_sieve.best_model_1;

% Set box confidenc threshold for detections to be included. 
confThr = 0.5;
% 
scaledBlockSize = [224 224];
originalBlockSize = [950 950];
scale2 = originalBlockSize./scaledBlockSize;


% Run from within the folder that contains the image patches
% Setup storage space
allImages = dir;
allImages = extractfield(allImages, 'name')';
allImages = allImages(4:end);

%% Clean up the image patches to remove binary ones, and edge ones
idCheckIm = zeros(length(allImages),1);

for i = 1:length(allImages) 
 fprintf('%s%d%s%d\n', 'Cleaning up image ', i, '/', length(allImages));
im = allImages{i};
[I, R] = geotiffread(im);

% if image pixel values are all the same, i.e. we have a blank image,
% assign a 1 to the id
if min(min(I)) == max(max(I))
    idCheckIm(i) = 1;
end

% if the image is only black and white assign a 1 to the id
if all(I(:) == 0 | I(:) == 255)
    idCheckIm(i) = 1;
end

% if more than half of the image is black or white, assign a 1 to the id
binaryCount = sum(I(:) == 0 | I(:) == 255);
if binaryCount > numel(I) / 2;
    idCheckIm(i) = 1;

end
end

imagesToRun = find(idCheckIm==0);
allImagesNew = allImages(imagesToRun);
save('blockListCleaned')


%% Set up storage space
detections_boxes = cell(size(allImagesNew));
detections_scores = cell(size(allImagesNew));
geographic_box_coords_x = cell(size(allImagesNew));
geographic_box_coords_y = cell(size(allImagesNew));


scaledBoxes = cell(size(allImagesNew));
sieveCheck = cell(size(allImagesNew));

T = table(allImagesNew, detections_boxes, detections_scores,scaledBoxes, sieveCheck, geographic_box_coords_x,geographic_box_coords_y);


% load first image to get size and scale for resizing
im = allImagesNew{1};
[I, R] = geotiffread(im);
% Resize raster
scale = 224/size(I,2);


    % create empty shapefile
    m = mapshape;
    m.Geometry = 'polygon';
    mout = m;


%% Run detection on everything
tic
for i = 1:length(allImagesNew)
    fprintf('%s%d%s%d\n', 'Detecting boxes for image ', i, '/', length(allImagesNew));

im = allImagesNew{i};
[I, R] = geotiffread(im);

% figure
% mapshow(I,R)

[Ir,Rr] = mapresize(I,R,scale);
% figure
% imshow(Ir)
% Run detector on all images
[T.detections_boxes{i}, T.detections_scores{i}] = detect(detector,Ir,'Threshold',confThr);

end

% get rid of empty rows
empty = cellfun('isempty', T{:,'detections_boxes'} );
id = find(empty==0);
newtable = T(id,:);
T = newtable;

save('1_afterDetection_950')

%% Run sieve on everything
for i = 1:height(T)
     fprintf('%s%d%s%d\n', 'Sieving image ', i, '/', height(T));
 
 originalBoxes = T.detections_boxes{i};
    boxNewAll = [];

    %% Detector was run on small images, scale the boxes back up to 950 x 950 
  for j = 1:height(originalBoxes)

    boxNew = round(bboxresize(originalBoxes(j,:),scale2));
    boxNewAll = [boxNewAll; boxNew];

    T.scaledBoxes(i) = {boxNewAll};
 end

% Convert box coordinates to spatial coordinates
im = T.allImagesNew{i};
[I, R] = geotiffread(im);
% Create a meshgrid based on the specified x and y ranges
[x, y] = meshgrid(R.XWorldLimits(1):R.CellExtentInWorldX:R.XWorldLimits(2), ...
                  R.YWorldLimits(2):-R.CellExtentInWorldY:R.YWorldLimits(1));

x = x(1:end-1, 1:end-1);
y = y(1:end-1, 1:end-1);

% create space to store the geographic converted coords
boxesGeographic = [];

xs = [];
ys = [];


%% Crop patches using the boxes from the detector.
% and run through trained classification network.

YPred_sieve_All = [];
sieveProbs_all = [];
% for each box detected in an image block...
for k = 1:height(T.scaledBoxes{i})
    
    boxes = T.scaledBoxes{i};

    % Get the boxes relative coords
    boxToConv = boxes(k,:);

    % check the box x+w is not over 224
    if boxToConv(1)+boxToConv(3) > 950;

        dif = boxToConv(1)+boxToConv(3)-950;
        % subtract the difference(+1) from the width 
        boxToConv(3) = boxToConv(3)-(dif+1);
    end

    % check the box y+h is not over 224
    if boxToConv(2)+boxToConv(4) > 950
        dif = boxToConv(2)+boxToConv(4)-950;
        % subtract the difference(+1) from the width 
        boxToConv(4) = boxToConv(4)-(dif+1);
    end

    % convert x and y into geographic
    xBox = x(1,boxToConv(1));
    yBox = y(boxToConv(2),1);

    xmin = xBox; ymax = yBox;
    xmax = x(1,(boxToConv(1)+boxToConv(3)));
    ymin = y((boxToConv(2)+boxToConv(4)));

    xmin = min([xmin,xmax]); xmax = max(xmin, xmax);
    ymin = min([ymin,ymax]); ymax = max(ymin, ymax);

    % Reorder
    allx = [xmin xmax xmax xmin];
    ally = [ymin ymin ymax ymax];

    xs = [xs;allx];
    ys = [ys;ally];

    T.geographic_box_coords_x{i} = xs;
    T.geographic_box_coords_y{i} = ys;

    mout = append(mout,allx,ally);
    mout.Geometry = 'polygon';

        %% Crop image to box and run through sieve
        [Icrop,Rcrop] = mapcrop(I,R,[xmin, xmax],[ymin, ymax]);

        % figure
        % mapshow(Icrop,Rcrop)

            % pad and resize image so that it reaches correct dimensions
            [rows, cols, ~] = size(Icrop);
            
            % which dimension is larger
            rowslrger = rows > cols;
            colslrger = cols > rows;
            
            % get a random number, to be used to decide if the padding is pre or post
            rand = randi([1 2],1);
            location = {'pre', 'post'};
            % if more columns padd rows
            if colslrger == 1; 
                I_pad = padarray(Icrop,cols-rows, 0, location{rand});
                I_resize = imresize(I_pad, [224 224]);
            end
            
            if rowslrger == 1;
                I_pad = padarray(Icrop, rows-cols, 0, location{rand});
                I_resize = imresize(I_pad, [224 224]);
            end
            

% figure
% imshow(I_resize)

% Classify first image using sieve network
[YPred_sieve,probs_sieve] = classify(trained_sieve,I_resize);
YPred_sieve_All = [YPred_sieve_All; YPred_sieve];
sieveProbs_all = [sieveProbs_all; probs_sieve];

end
T.sieveCheck{i} = YPred_sieve_All;
T.sieveProbs{i} = sieveProbs_all;
end




%% Clean the table so that it contains only boxes that have passed the sieve check

afterSieving = table;

for i = 1:height(T)

    % sprintf('%s%d%s%d', 'Image #', i, '/', height(T))

    if isempty(T.detections_boxes{i,1}) == 0
    % Get the ids of the boxes that we want to keep (the background ones)
   id =  find(T.sieveCheck{i,1} == 'building');
   
   % files = repmat(T.allImagesNew{i, };
   boxes = T.detections_boxes{i};
   scores = T.detections_scores{i};
   scaledBoxes = T.scaledBoxes{i};
   geographic_box_coords_x = T.geographic_box_coords_x{i};
   geographic_box_coords_y = T.geographic_box_coords_y{i};
   sieveProbs = T.sieveProbs{i};
    
   afterSieving.allImagesNew{i} = T.allImagesNew{i};
   afterSieving.detections_boxes{i} = boxes(id,:);
   afterSieving.detections_scores{i} = scores(id,:);
   afterSieving.scaledBoxes{i} = scaledBoxes(id,:);
   afterSieving.geographic_box_coords_x{i} = geographic_box_coords_x(id,:);
   afterSieving.geographic_box_coords_y{i} = geographic_box_coords_y(id,:);
   afterSieving.sieveProbs{i} = sieveProbs(id,:);

    end

end

% Get rid of empty rows in table
% Get rid of images with no buildings in
empty = cellfun('isempty', afterSieving{:,'detections_boxes'} );
id = find(empty==0);
newtable = afterSieving(id,:);
afterSieving = newtable;

save('2_afterSieving_950')


% repeat the fileNames so that we can unwrap the cells in the table to a
% vector
allFiles = [];
for i = 1:height(afterSieving)

    repfiles = repmat({afterSieving.allImagesNew{i}},  size(afterSieving.detections_boxes{i},1),1);
    allFiles = [allFiles; repfiles];
end



%% Run non maximum suppression
% First we need to reorder the box coordinates into [x y w h]

scores = cell2mat(afterSieving.detections_scores);
xs = cell2mat(afterSieving.geographic_box_coords_x);
ys = cell2mat(afterSieving.geographic_box_coords_y);
scaledBoxes = cell2mat(afterSieving.scaledBoxes);
rectangles = cell(size(scores));
probsBuilding = cell2mat(afterSieving.sieveProbs);


for i = 1:length(scores)

    rectangles{i} = [xs(i,1), ys(i,1), (max(xs(i,:))-min(xs(i,:))), ...
        (max(ys(i,:))-min(ys(i,:)))];
end

boxes = cell2mat(rectangles);
% NMS
[selectedBbox,selectedScore, id] = selectStrongestBbox(boxes,scores,"RatioType","Min");

 afterNMS = table(allFiles(id), scaledBoxes(id), rectangles (id,:), xs(id,:), ys(id,:), scores(id), probsBuilding(id,2));

 afterNMS.Properties.VariableNames = {'allFiles', 'scaledBoxes', 'rectangles', 'xs','ys','scores', 'probsBuilding'};

 % create empty shapefile
    moutNew = mapshape;
    moutNew.Geometry = 'polygon';

% convert back to format to be used to make mapshape
for i = 1:height(afterNMS)
    disp(i)
    % Reorder
    allx = afterNMS.xs(i,:);
    ally = afterNMS.ys(i,:);

    moutNew = append(moutNew,allx,ally);
    moutNew.Geometry = 'polygon';

end

 save('3_afterNMS_950_0.9')


%% write the shape files and workspace

figure
subplot(1,2,1)
mapshow(mout, 'FaceAlpha', 0.1, 'LineWidth',1.5)
title('Pre NMS')
subplot(1,2,2)
mapshow(moutNew, 'FaceAlpha', 0.1, 'LineWidth',1.5)
title('Post NMS')
shapewrite(moutNew, 'postNMS_950_0.9.shp')
%shapewrite(mout, 'preNMS.shp')
% save('workspace')


% get the unique image namaes
U = unique(afterNMS.allFiles);

%% Run classification models
for i = 1:length(U) % for each unique image
    
   % find all rows of the table that are that images
   id = find(strcmp(afterNMS.allFiles, U(i)));


  
   [I, R] = geotiffread(U{i});


%% Crop patches using the boxes from the detector.
% and run through trained classification network.

% for each box detected in an image block...
for m = 1:length(id)

   fprintf('%s%d%s%d\n', 'Running classifiers for image ', id(m), '/', height(afterNMS));

    box = afterNMS.scaledBoxes(id(m),:);

    xmin = min(afterNMS.xs(id(m),:));
    xmax = max(afterNMS.xs(id(m),:));

    ymin = min(afterNMS.ys(id(m),:));
    ymax = max(afterNMS.ys(id(m),:));

        %% Crop image and run classification model
        [Icrop,Rcrop] = mapcrop(I,R,[xmin, xmax],[ymin, ymax]);

       
        % check the detected box isnt an edge box, by looking at the
        % proportion of white pixels. If more than 10% the pixels are
        % white or black, we assume this is an edge box, and assign a value
        % of 0 to the score, used later to remove rows
        binaryCount = sum(Icrop(:) == 0 | Icrop(:) == 255);
        if binaryCount > numel(Icrop) / 10;
            afterNMS.scores(id(m)) = 0;
            break
        end

        % figure
        %  imshow(Icrop)

            % pad and resize image so that it reaches correct dimensions
            [rows, cols, ~] = size(Icrop);

            % which dimension is larger
            rowslrger = rows > cols;
            colslrger = cols > rows;

            % get a random number, to be used to decide if the padding is pre or post
            rand = randi([1 2],1);
            location = {'pre', 'post'};
            % if more columns padd rows
            if colslrger == 1; 
                I_pad = padarray(Icrop,cols-rows, 0, location{rand});
                I_resize = imresize(I_pad, [224 224]);
            end

            if rowslrger == 1;
                I_pad = padarray(Icrop, rows-cols, 0, location{rand});
                I_resize = imresize(I_pad, [224 224]);
            end

% figure
% imshow(I_resize)
% Classify using first network
[ClassPred_1, ClassProb_1] = classify(trained_classifier_1,I_resize);


            % Convert categorical to number so that all data
            % are of the same type in table
            ytmp = cellstr(ClassPred_1);
            ytmp = ytmp{:};
            YPred =  str2double(ytmp);

afterNMS.ClassPred_1(id(m)) = YPred;
afterNMS.ClassProb_1{id(m)} = max(ClassProb_1);



           %% If the building is classified as damaged, we then run trained network number 2
            if afterNMS.ClassPred_1(id(m)) == 2

                [ClassPred_2, ClassProb_2] = classify(trained_classifier_2,I_resize);

                % Convert categorical to number so that all data
                % are of the same type in table
                ytmp = cellstr(ClassPred_2);
                ytmp = ytmp{:};
                YPred =  str2double(ytmp);

                afterNMS.ClassPred_2(id(m)) = YPred;
                afterNMS.ClassProb_2{id(m)} = max(ClassProb_2);

            end

            if afterNMS.ClassPred_1(id(m)) == 1
                afterNMS.ClassPred_2(id(m)) = nan;
                afterNMS.ClassProb_2{id(m)} = nan;
            end


 
end
end

rowRemove = find(afterNMS.scores==0);
afterNMS(rowRemove,:) = [];

rowRemove = find(afterNMS.ClassPred_1==0);
afterNMS(rowRemove,:) = [];

save('4_afterClassifying_950_0.9')


% create empty shapefile
    moutFinal = mapshape;
    moutFinal.Geometry = 'polygon';

% convert back to format to be used to make mapshape
for i = 1:height(afterNMS)
    disp(i);
    % Reorder
    allx = afterNMS.xs(i,:);
    ally = afterNMS.ys(i,:);

    moutFinal = append(moutFinal,allx,ally);
    moutFinal.Geometry = 'polygon';

end

figure
mapshow(moutFinal, 'FaceAlpha', 0.1, 'LineWidth',1.5)
shapewrite(moutFinal, 'afterClassification_950_0.9.shp')

%% Add attribute data to shapefile

% for the variables we want to include in the shapefile
% convert single to double

detections_scores = afterNMS.scores;
detections_scores = double(detections_scores);
probsBuilding = afterNMS.probsBuilding;

ClassPred_1 = afterNMS.ClassPred_1;
ClassProb_1 = afterNMS.ClassProb_1;
ClassProb_1 = cell2mat(cellfun(@double, ClassProb_1, 'UniformOutput', false));


ClassPred_2 = afterNMS.ClassPred_2;
ClassProb_2 = afterNMS.ClassProb_2;
ClassProb_2 = cell2mat(cellfun(@double, ClassProb_2, 'UniformOutput', false));

%Assign damage state of 0,1,2
damageState = max(afterNMS.ClassPred_1, afterNMS.ClassPred_2);
damageState(damageState==1)=0;
damageState(damageState==2)=1;
damageState(damageState==3)=2;

S = shaperead('afterClassification_950_0.9.shp');

% add score to table
for i = 1:length(detections_scores);
S(i).detection_scores = detections_scores(i);
S(i).probBuiling = probsBuilding(i);
S(i).ClassPred_1 = ClassPred_1(i);
S(i).ClassProb_1 = ClassProb_1(i);

S(i).ClassPred_2 = ClassPred_2(i);
S(i).ClassProb_2 = ClassProb_2(i);

S(i).damageState = damageState(i);
end




% Write the GeoStruct to a shapefile
shapewrite(S,'afterClassification_wAttributes_950_0.9.shp');