% CODE FOR OPTIMIZING MACHINE LEARNING ALGORITHM FOR FM MONITOR

%% DATA LOADING AND PRE-PROCESSING ========================================
clear; clc;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data loading ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   This part of the code loads the data files for processing. Loading of
%   single or multiple data files are allowed. When a data file is loaded,
%   a variable named data_var is loaded in the workspace from where the
%   sensor and sensation data for that particular session is extracted and
%   stored in a cell matrix.

% Starting notification
fprintf('\n#Data file loading is going on...');

% Defining known parameters
Fs_sensor    = 1024; % Frequency of sensor data sampling in Hz
Fs_sensation = 1024; % Frequency of sensation data sampling in Hz
n_sensor     = 8;    % Total number of sensors

% Adding path for function files
addpath(genpath('SP_function_files'))
addpath(genpath('ML_function_files'))
addpath(genpath('Learned_models'))
addpath(genpath('matplotlib_function_files')) % Hold the custom function for some colormaps

% Locating and loading the data files
[data_file_names,path_name] = uigetfile('*.mat','Select the data files','MultiSelect','on'); % Returns file names with the extension in a cell, and path name
addpath(path_name); % Adds the path to the file
[Aclm_data1,Aclm_data2,Acstc_data1,Acstc_data2,Pzplt_data1,Pzplt_data2,Flexi_data,...
    IMU_data,sensation_data_SD1,~] = load_data_files(data_file_names); % Loads data files

% Claculate the no. of data files for different participants
n_data_files = length(Aclm_data1); % Total number of data files
[n_DF_P1,n_DF_P2,n_DF_P3,n_DF_P4,n_DF_P5] = deal(0,0,0,0,0); % Simulteneously initializing multiple variables with 0
for i = 1:n_data_files
    if n_data_files == 1
        DF_names = data_file_names;
    else
        DF_names = data_file_names{i};
    end
    switch DF_names(2)
        case '1'
            n_DF_P1 = n_DF_P1+1;
        case '2'
            n_DF_P2 = n_DF_P2+1;
        case '3'
            n_DF_P3 = n_DF_P3+1;
        case '4'
            n_DF_P4 = n_DF_P4+1;
        case '5'
            n_DF_P5 = n_DF_P5+1;
        otherwise
            disp('This Data file has naming inconsistency')
    end
end

% Arrange data files names in a table
T_data_file_names = table('Size',[n_data_files 1],'VariableTypes',"string");

for j = 1:n_data_files
    if ischar(data_file_names) % If there is only a single data file
        DFN = convertCharsToStrings(data_file_names);
    else
        DFN = cellstr(data_file_names{j}); % Converts data file names from cell elements to strings
    end
    T_data_file_names{j, :} = DFN; 
end

% Calculating the duration of each data files
duration_raw_data_files = zeros(n_data_files,1);
for i = 1:n_data_files
    duration_raw_data_files(i) = length(Acstc_data1{i})/(Fs_sensor*60); % Duration in minutes
end

% Ending notification
fprintf(['\nIn total, %.0f(P1=%.0f, P2=%.0f, P3=%.0f, P4=%.0f, P5=%.0f) data files \n' ...
    'have been uploaded.'],n_data_files,n_DF_P1,n_DF_P2,n_DF_P3,n_DF_P4,n_DF_P5);
fprintf('\nTotal length of loaded recording sessions: %.2f hrs\n',sum(duration_raw_data_files)/60);
if n_DF_P1+n_DF_P2+n_DF_P3+n_DF_P4+n_DF_P5 ~= n_data_files
    fprintf('Data files have naming inconsistency.\n')
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pre-processing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Pre-processing steps includes filtering/detrending and trimming the data
% Triming is done to remove some initial and final unwanted data and make
% data file lengths from SD1 and SD2 euqal.

% Preprocessing the data
[Aclm_data1_fltd,Aclm_data2_fltd,Acstc_data1_fltd,Acstc_data2_fltd,Pzplt_data1_fltd,...
    Pzplt_data2_fltd,Flexi_data_fltd,IMU_data_fltd,sensation_data_SD1_trimd] ...
    = get_preprocessed_data(Aclm_data1,Aclm_data2,Acstc_data1,Acstc_data2,Pzplt_data1,...
    Pzplt_data2,Flexi_data,IMU_data,sensation_data_SD1,Fs_sensor,Fs_sensation);

% Clearing unnecessary variables
clear Acstc_data1 Acstc_data2 Aclm_data1 Aclm_data2 Pzplt_data1 Pzplt_data2 ...
    Flexi_data IMU_data sensation_data_SD1 

% ~~~~~~~~~~~~~~~~~~~~~~~ Calculating general info ~~~~~~~~~~~~~~~~~~~~~~~~
% Extracting force sensor data during each recording session
Force_data_sample  = cell(1,n_data_files);
Force_mean         = zeros(n_data_files, 1);
Force_signal_power = zeros(1, n_data_files);
% sample_size      = 30; % sample size in seconds

for i = 1 : n_data_files
    %Force_data_sample{i} = Flexi_data_fltd{i}(1:sample_size*Fs_sensor);
    Force_data_sample{i}  = abs(Flexi_data_fltd{i});
    Force_mean(i)         = mean(Force_data_sample{i});
    Force_signal_power(i) = sum(Force_data_sample{i}.^2)/length(Force_data_sample{i});
end

% Duration of each data sample after trimming
duration_trimmed_data_files = zeros(n_data_files,1);
for i = 1:n_data_files
    duration_trimmed_data_files(i) = length(Acstc_data1_fltd{i})/(Fs_sensor*60); % Duration in minutes
end

% Clearing unnecessary variable
clear Force_data_sample

% Step completion notification
fprintf('\nData filtering and trimming are completed.\n')
%

%% DATA AND FEATURE EXTRACTION FOR MACHINE LEARNING =======================
%   This part of the code extracts data from all the sensors corresponding
%   to the TPDs and FPDs after combining detections from all the sensors by
%   by the logical OR operator.
%   Selected features were then collected from the extracted data sets.

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data extraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% Parameters for segmentation and detection matching
ext_backward     = 5.0; % Backward extension length in second
ext_forward      = 2.0; % Forward extension length in second
FM_dilation_time = 3.0; % Dilation size in seconds
n_all_FM_sensors = 6; % Number of FM sensors
FM_min_SN        = [30, 30, 30, 30, 30, 30]; % These values are selected to get SEN of 99%

% Variable decleration
IMU_map                      = cell(1,n_data_files);
M_sntn_map                   = cell(1,n_data_files);
n_Maternal_detected_movement = zeros(n_data_files,1);
threshold                    = zeros(n_data_files, n_all_FM_sensors);
TPD_extracted                = cell(1, n_data_files);
FPD_extracted                = cell(1, n_data_files);
extracted_TPD_weightage      = cell(1, n_data_files);
extracted_FPD_weightage      = cell(1, n_data_files);

fprintf("\n#Select the sensor for data extraction from the following options: ");
fprintf("\n1- Aclm, 2- Acstc, 3- Piezo, 4- Aclm+Acstc, \n5- Aclm+Piezo, 6- Acstc+Piezo, 7- All sensors.\n");
prompt           = "Selection: ";
sensor_selection = input(prompt);

% Starting notification
fprintf('\n#Data extraction is going on with the following settings: ')
fprintf('\n\tDetection matching time window = %.01f (s) + %.01f (s)',ext_backward,ext_forward);
fprintf('\n\tFM dilation period = %.01f (s)',FM_dilation_time);
fprintf('\n\tThreshold multiplier: \n\tAccelerometer = %.0f, Acoustic = %.0f, Piezoelectric = %.0f', ...
    FM_min_SN(1),FM_min_SN(3),FM_min_SN(5));

switch sensor_selection
    case 1
        fprintf('\n\tSensor combination: Accelerometers only.\n\t...');
    case 2
        fprintf('\n\tSensor combination: Acoustic sensors only.\n\t...');
    case 3
        fprintf('\n\tSensor combination: Piezoelectric diaphragms only.\n\t...');
    case 4
        fprintf('\n\tSensor combination: Accelerometers and acoustic sensors.\n\t...');
    case 5
        fprintf('\n\tSensor combination: Accelerometers and piezoelectric diaphragms.\n\t...');
    case 6
        fprintf('\n\tSensor combination: Acoustic sensors and piezoelectric diaphragms.\n\t...');
    case 7
        fprintf('\n\tSensor combination: All the sensors.\n\t...');
end

for i = 1 : n_data_files
    % Starting notification
    fprintf('\n\tCurrent data file: %.0f/%.0f',i,n_data_files)

    % --------- Segmentaiton of IMU data and creation of IMU_map ---------%
    % get_IMU_map() function is used here. It segments and dilates the IMU
    % data and returns the resultant data as IMU map. Settings for the
    % segmentation and dilation are given inside the function.
    %   Input variables:  IMU_data- A column vector of IMU data
    %                     data_file_names- a char variable with data file name
    %   Output variables: IMU_map- A column vector with segmented IMU data

    if n_data_files == 1
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i},data_file_names,Fs_sensor);
    else
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i},data_file_names{i},Fs_sensor);
    end

    % ----------------------- Creation of M_sensation_map ----------------%
    % get_sensation_map() function is used here. This function gives the
    % sensation map by dilating every detection to past and future. It also
    % revomes the windows that overlaps with IMU_map. Settings for the
    % extension are given inside the function.
    %   Input variables-  sensation_data- a column vector with the sensation data
    %                     IMU_map- a column vector
    %                     ext_backward, ext_forward- scalar values
    %                     Fs_sensor,Fs_sensation- scalar values    %
    %   Output variables: M_sntn_map- a column vector with all the sensation windows joined together

    M_sntn_map{i} = get_sensation_map(sensation_data_SD1_trimd{i},IMU_map{i},...
        ext_backward,ext_forward,Fs_sensor,Fs_sensation);

    % Calculating the number of maternal sensation detection
    sensation_data_labeled            = bwlabel(sensation_data_SD1_trimd{i});
    n_Maternal_detected_movement(i,1) = length(unique(sensation_data_labeled)) - 1; % 1 is deducted to remove the initial value of 0

    % ---------------------- Segmentation of FM data ---------------------%
    % get_segmented_data() function will be used here, which will
    % threshold the data, remove body movement, and dilate the data.
    % Setting for the threshold and dilation are given in the function.
    %   Input variables:  sensor_data- a cell variable;
    %                     min_SN- a vector/ a scalar;
    %                     IMU_map- a column vector
    %                     Fs_sensor, FM_dilation_time- a scalar
    %   Output variables: sensor_data_sgmntd- a cell variable of same size as the input variable sensor_data_fltd;
    %                     h- a vector

    switch sensor_selection
        case 1
            n_FM_sensors = 2;
            sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
        case 2
            n_FM_sensors = 2;
            sensor_data_fltd = {Acstc_data1_fltd{i}, Acstc_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
        case 3
            n_FM_sensors = 2;
            sensor_data_fltd = {Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
        case 4
            n_FM_sensors = 4;
            sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, ...
                Acstc_data1_fltd{i}, Acstc_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | ...
                sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};
        case 5
            n_FM_sensors = 4;
            sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, ...
                Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | ...
                sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};
        case 6
            n_FM_sensors = 4;
            sensor_data_fltd = {Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
                Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | ...
                sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};
        case 7
            n_FM_sensors = 6;
            sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
                Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
            [sensor_data_sgmntd, threshold(i,1:n_FM_sensors)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
            sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
                sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    end

    % ------------------- Extraction of TPDs and FPDs ---------------------
    sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    if (n_label) % When there is a detection by the sensor system
        [TPD_extracted{i},FPD_extracted{i},extracted_TPD_weightage{i},extracted_FPD_weightage{i}] = extract_detections(M_sntn_map{i},...
            sensor_data_fltd,sensor_data_sgmntd,sensor_data_sgmntd_cmbd_all_sensors_labeled,n_label,n_FM_sensors);
    end
end

% Clearing unnecessary variables
clear sensor_data_fltd sensor_data_sgmntd sensor_data_sgmntd_cmbd_all_sensors...
    sensor_data_sgmntd_cmbd_all_sensors_labeled

% Ending notification
fprintf('\nData extraction is completed.\n')
%

% ~~~~~~~~~~~~~~~~~~~~~~~~~ Feature extraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% In this part of the code, selected features are extracted from the
% the TPD and FPD data sets extracted in the previous step.

fprintf('\n#Feature extraction for machine learning is going on\n...')

[X_TPD,X_FPD,n_TPD,n_FPD,total_duration_TPD,total_duration_FPD] = ...
    extract_features(TPD_extracted,extracted_TPD_weightage,FPD_extracted,...
    extracted_FPD_weightage,threshold,Fs_sensor,n_FM_sensors);

fprintf('\nFeature extraction is completed.')
fprintf('\nIn total, %d features were collected forom all the sensor data.\n',size(X_TPD,2))
%

% Combining features from the TPD and FPD data sets
X            = [X_TPD; X_FPD];
Y            = zeros(n_TPD+n_FPD,1);
Y(1:n_TPD,1) = 1;

% Feature normalization
X_norm     = normalize_features(X);
X_TPD_norm = X_norm(1:size(X_TPD,1),:); % Normalize with respect to deviation (= max - min)
X_FPD_norm = X_norm(size(X_TPD,1)+1:end,:); % Normalize with respect to deviation (= max - min)

%% FEATURE RANKING BY NEIGHBOURHOOD COMPONENT ANALYSIS (NAC) ==============

%  Finding the best of lambda using K-fold cross-validation
K             = 5; % Number of folds
p             = 0.2; % fraction of observation in test set

rng default
% cvp         = cvpartition(Y,'kfold',K); % Stratified K-fold division
cvp           = cvpartition(Y,'HoldOut',p); % Stratified holdout division
numtestsets   = cvp.NumTestSets;
lambdavalues  = linspace(0,2,20)/length(Y); 

lossvalues    = zeros(length(lambdavalues),numtestsets);

for i = 1:length(lambdavalues)     
    for j = 1:numtestsets
        % Notify the iteration number
        fprintf('\n Iteration: %g/%g', (i-1)*j+j,length(lambdavalues)*numtestsets);

        % Extract the training set from the partition object
        Xtrain = X_norm(cvp.training(j),:);
        ytrain = Y(cvp.training(j),:);
        
        % Extract the test set from the partition object
        Xtest = X_norm(cvp.test(j),:);
        ytest = Y(cvp.test(j),:);
        
        % Train an nca model for classification using the training set
        ncaMdl = fscnca(Xtrain,ytrain,'FitMethod','exact','Verbose',1, ...
             'Solver','lbfgs','Lambda',lambdavalues(i), ...
             'IterationLimit',50,'GradientTolerance',1e-5);
        
        % Compute the classification loss for the test set using the nca model
        lossvalues(i,j) = loss(ncaMdl,Xtest,ytest,'LossFunction','quadratic');      
    end                          
end

% Plot lambda vs. loss
figure
plot(lambdavalues,mean(lossvalues,2),'ro-')
xlabel('Lambda values')
ylabel('Loss values')
grid on

[~,idx]    = min(mean(lossvalues,2)); % Find the index of minimum loass.
bestlambda = lambdavalues(idx); % Find the best lambda value.

% Applying the best model
ncaMdl = fscnca(X_norm,Y,'FitMethod','exact','Verbose',1,'Solver','lbfgs',...
    'Lambda',bestlambda);

figure
semilogx(ncaMdl.FeatureWeights,'ro')
xlabel('Feature index')
ylabel('Feature weight')   
grid on
%

% Extract the feature ranking information
feature_index   = (1:size(X_norm,2))'; % Creates a vector with feature index numbers
feature_ranking = [feature_index,ncaMdl.FeatureWeights]; % Combines feature index and weights in a matrix
[~,I_sort]      = sort(feature_ranking(:,2),'descend'); % creates a sorted index
feature_ranking = feature_ranking(I_sort,:); % Apply the sorted index to sort the features

%%
n_top_features_to_consider = 10;

index_top_features = feature_ranking(1:n_top_features_to_consider,1);

X_TPD_norm_ranked = X_TPD_norm(:,index_top_features);
X_FPD_norm_ranked = X_FPD_norm(:,index_top_features);

save('X_TPD_norm_ranked.txt', 'X_TPD_norm_ranked', '-ASCII');
save('X_FPD_norm_ranked.txt', 'X_FPD_norm_ranked', '-ASCII');
%

%% DATA VISUALIZATION USING DIMENSIONALITY REDUCTION ======================
%   In this part of the code, data is visualized using dimensionality
%   schems t-SNE and PCA.

% %  Before running PCA and t-SNE, it is important to first normalize X
% [X_norm, ~, ~] = featureNormalize(X); % Normalize with respect to deviation (= max - min)
% %
% % --------------------- Dimensionality reduction by PCA -------------------
% fprintf('\nRunning PCA and t-SNE on the whole dataset\n...');
% 
% % Run PCA
% [U, S] = run_PCA(X_norm);
% 
% % Determination of number of pinrciple components
% variance = 0.99; % select n_PC to retain 99% variance in the data 
% for i = 1:length(S)
%     S_P = S(1:i,:);
%     if (sum(S_P,'all')/sum(S,'all'))>= variance
%         n_PC_min = i;
%         break;
%     end
% end
% 
% %  Project the data onto K dimension
% Z = projectData(X_norm, U, n_PC_min);
% %
% % ---------------- Dimensionality reduction by t-SNE ----------------------
% % 2-D t-SNE
% rng default % for reproducibility
% [tSNE_X1,loss1] = tsne(X_norm,'Algorithm','exact');
% 
% % 3-D t-SNE
% rng default % for fair comparison
% [tSNE_X2,loss2] = tsne(X_norm,'Algorithm','exact','NumDimensions',3);
% 
% fprintf(['\nIn t-SNE 2-D embedding has loss %g, and 3-D embedding has loss...' ...
%     ' %g.\n'],loss1,loss2)
% %
% fprintf('\nPCA and t-SNE analyses are completed.\n');
% %
% % ---------------------- Visualization generation -------------------------
% fprintf('\nCreating visualization using PCA and t-SNE data.');
% 
% % Visualization by PCA
% figure
% plotData(Z, Y, '+g'); % Visualize only 1st 3 components in a 3-D plot
% title('PCA');
% %
% 
% % Visualization by t-SNE
% %   Plot settings
% B_width = 3; % Width of the box
% m_size = 25; % Marker size
% F_name = 'Times New Roman'; % Font type
% F_size = 40; % Font size
% 
% %   2-D plotting
% figure
% gscatter(tSNE_X1(:,1),tSNE_X1(:,2),Y, 'rgb','.',m_size)
% legend('non-FM', 'FM', 'Location', 'northwest');
% legend boxoff
% set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', B_width) % Sets the font type and size of the labels
% xlim([-120 120]) % This limit gives the best figure
% ylim([-120 100]) % This limit gives the best figure
% set(gca,'XTick',[], 'YTick',[]); % To get rid of all the tick marks
% set(gca,'Xticklabel',[],'Yticklabel',[]) % To get rid of all the labels
% 
% %   3-D plotting
% figure
% m_size = 40; % Marker size
% v = double(categorical(Y)); % Need to change the category from 0,1 to 1,2
% Y_v = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
% scatter3(tSNE_X2(:,1),tSNE_X2(:,2),tSNE_X2(:,3),m_size,Y_v,'filled')
% % legend('non-FM', 'FM', 'Location', 'best');
% % legend boxoff
% view(-50,8)
% set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', B_width) % Sets the font type and size of the labels
% % xlim([-90 100]) % This limit gives the best figure
% % ylim([-120 100]) % This limit gives the best figure
% % zlim([-100 90]) % This limit gives the best figure
% set(gca,'XTick',[], 'YTick',[], 'ZTick',[]); % To get rid of all the tick marks
% set(gca,'Xticklabel',[],'Yticklabel',[],'Zticklabel',[]) % To get rid of all the labels
% box on; % Turn on the 3D box
% set(gca, 'BoxStyle', 'full') % Set the 3D box to full 
% %
% 
% fprintf('\nCreation of visualizations are completed.\n');
% %


