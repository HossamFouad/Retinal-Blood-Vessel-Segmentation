%% preprocessing
% Variables
K=11000; % number of superpixels per image
count=1; % counter for dataset blocks saving
label=[]; % output saving
%folder path
I.path_images="DRIVE\training\images\";  % vessel images
I.path_1st_manual="DRIVE\training\1st_manual\"; % ground truth
I.path_dataset="DRIVE\training\dataset\"; % dataset saving path
I.path_mask="DRIVE\training\mask\";  % images mask
% reading directory images path
I.files_images=dir(char(I.path_images+"*.tif"));
I.files_1st_manual=dir(char(I.path_1st_manual+"*.gif"));
I.files_mask=dir(char(I.path_mask+"*.gif"));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare dataset
for img=1:size(I.files_images,1)
    % image path
    imgpath=I.path_images+I.files_images(img).name; % reading input images
    manualpath=I.path_1st_manual+I.files_1st_manual(img).name; % reading ground truth
    % reading each image and selecting the green channel
    reading_image=imread(char(imgpath));
    enhanced_image=histeq(reading_image);
    enhanced_image = imgaussfilt(enhanced_image);
   % reading manual images used as ground truth
   manual_image=imread(char(manualpath));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % applying Simple Linear Iterative Clustering to obtain superpixels
   [l, Am, Sp, d] = slic(enhanced_image, K,35,1.5);
   % trunck to be suitable for chunking with 25*25 samples
   [Am_row,Am_col]=size(Am); 
   Am=Am(13:Am_row-12,13:Am_col-12);
   unique_Am=unique(Am); % unique values in superpixels after truncation
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    for i=1:size(unique_Am,1)
        x=unique_Am(i);
       % select one pixel randomly for one superpixel
      [index_row, index_col]=find(Am==x); 
      index_row=index_row+12;
      index_col=index_col+12;
      p = randperm(size(index_row,1),1);
      row=index_row(p,1); % row of random pixel
      col=index_col(p,1); % col of random pixel
      block= enhanced_image(row-12:row+12,col-12:col+12,:); % get block 25*25 sample
      block=block(:,:,2); %get the green channel
      file_name=I.path_dataset+count+".tif";
      imwrite(block,char(file_name)); % save image
      disp(count);
      count=count+1; 
      % save groundtruth corresponding of center pixel of block sample
      ground_truth=manual_image(row,col); 
      label=[label;ground_truth];
   end
end