% Code for analyzing fetal movement data obtained from the new multi-sensor
% fetal movement monitor
%

%% DATA LOADING AND PREPROCESSING =========================================
clear; clc;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data loading ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   This part of the code loads the data files for processing. Loading of
%   single or multiple data files are allowed. When a data file is loaded,
%   a variable named data_var is loaded in the workspace from where the
%   sensor and sensation data for that particular session is extracted and
%   stored in a cell matrix.

% Notify starting of this step
fprintf('\n#Data file loading is going on...');

% Define known parameters
Fs_sensor    = 1024; % Frequency of sensor data sampling in Hz
Fs_sensation = 1024; % Frequency of sensation data sampling in Hz
n_sensor     = 8;    % Total number of sensors

% Add path for function files
addpath(genpath('SP_function_files'))
addpath(genpath('matplotlib_function_files')) % Hold the custom function for some colormaps

% Locate and load the data files
[data_file_names,path_name] = uigetfile('*.mat','Select the data files','MultiSelect','on'); % Returns file names with the extension in a cell, and path name
addpath(path_name); % Adds the path to the file
[Aclm_data1,Aclm_data2,Acstc_data1,Acstc_data2,Pzplt_data1,Pzplt_data2,Flexi_data,...
    IMU_data,sensation_data_SD1,~] = load_data_files(data_file_names); % Loads data files

% Claculate the no. of data files for different participants
[n_data_files,n_DF_P1,n_DF_P2,n_DF_P3,n_DF_P4,n_DF_P5,...
    duration_raw_data_files] = get_dataFile_info(data_file_names,Fs_sensor,Aclm_data1);

% Notify completion of this step
fprintf(['\nIn total, %.0f(P1= %.0f, P2= %.0f, P3= %.0f, P4= %.0f, P5= %.0f) data files \n' ...
    'have been uploaded.'],n_data_files,n_DF_P1,n_DF_P2,n_DF_P3,n_DF_P4,n_DF_P5);
fprintf('\nTotal length of loaded recording sessions: %.2f hrs\n',sum(duration_raw_data_files)/60);
if n_DF_P1+n_DF_P2+n_DF_P3+n_DF_P4+n_DF_P5 ~= n_data_files
    fprintf('Data files have naming inconsistency.\n')
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Pre-processing steps includes filtering/detrending and trimming the data
% Triming is done to remove some initial and final unwanted data and make
% data file lengths from SD1 and SD2 euqal.

% Preprocess the data
[Aclm_data1_fltd,Aclm_data2_fltd,Acstc_data1_fltd,Acstc_data2_fltd,Pzplt_data1_fltd,...
    Pzplt_data2_fltd,Flexi_data_fltd,IMU_data_fltd,sensation_data_SD1_trimd] ...
    = get_preprocessed_data(Aclm_data1,Aclm_data2,Acstc_data1,Acstc_data2,Pzplt_data1,...
    Pzplt_data2,Flexi_data,IMU_data,sensation_data_SD1,Fs_sensor,Fs_sensation);

% Calculate general info after pre-processing
[duration_trimmed_data_files,Force_mean] = get_info_from_preprocessed_data(n_data_files,Flexi_data_fltd, Fs_sensor);

% Clear unnecessary variables
clear Acstc_data1 Acstc_data2 Aclm_data1 Aclm_data2 Pzplt_data1 Pzplt_data2 ...
    Flexi_data IMU_data sensation_data_SD1 

% Notify the completion of the step
fprintf('\nData filtering and trimming are completed.\n')
%

%% OPTIMIZATION OF THE THRESHOLD MULTIPLIER ===============================

% In this section the data analysis algorithm is run for a range of values
% of threshold multiplier to find the optimum value of threshold multiplier
% for different sensors. Maximization of F1_score is considered as the
% condition for selecting the optimum threshold multiplier.

% Starting notification
fprintf('\nThreshold optimization is going on... ')

% ----------- Calculation that will not vary across the iterations --------
% The IMU_map and the sensation_map are determined in this step.

% Parameters for creating sensation map and detection matching
ext_backward = 5.0; % Backward extension length in second
ext_forward  = 2.0 ; % Forward extension length in second

% Variable decleration
IMU_map    = cell(1,n_data_files);   
M_sntn_map = cell(1,n_data_files);  
n_activity = zeros(n_data_files, 1); % Cell matrix to hold the number of fetal activities
n_Maternal_detected_movement = zeros(n_data_files, 1);

for i = 1 : n_data_files

    % --------- Segmentaiton of IMU data and creation of IMU_map ---------%
    % get_IMU_map() function is used here. It segments and dilates the IMU 
    % data and returns the resultant data as IMU map. Settings for the
    % segmentation and dilation are given inside the function.
    %   Input variables:  IMU_data- A column vector of IMU data 
    %                     data_file_names- a char variable with data file name
    %   Output variables: IMU_map- A column vector with segmented IMU data 

    if n_data_files == 1
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i}, data_file_names, Fs_sensor);
    else
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i}, data_file_names{i}, Fs_sensor);
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

    M_sntn_map{i} = get_sensation_map(sensation_data_SD1_trimd{i}, IMU_map{i}, ext_backward, ext_forward, Fs_sensor, Fs_sensation);

    % ---------------- Determination of detection statistics --------------
    M_sntn_Map_labeled = bwlabel(M_sntn_map{i});
    n_activity(i)      = length(unique(M_sntn_Map_labeled)) - 1; % 1 is deducted to remove the first element, which is 0
    
    sensation_data_labeled          = bwlabel(sensation_data_SD1_trimd{i});
    n_Maternal_detected_movement(i) = length(unique(sensation_data_labeled)) - 1; % 1 is deducted to remove the initial value of 0
    %

end

% ----------- Calculations that will vary across the iterations -----------
% Parameters for segmentation
n_FM_sensors      = 6; % number of FM sensors
FM_dilation_time  = 3; % Dilation time for FM signal in second

% Paramters for ROC analysis
FM_min_SN_initial = 10; % Initial value of the minimum signal to noise ratio that is used to get the threshold value
FM_min_SN         = FM_min_SN_initial;
max_SN            = 200;
SN_increment      = 10;
n_iter            = 1+round((max_SN-FM_min_SN_initial)/SN_increment);

% New variable decleration
h                 = cell(1,n_iter); % cell matrix to contain threshold values
TPD_all_cmbd_indv = cell(1,n_FM_sensors/2);
FPD_all_cmbd_indv = cell(1,n_FM_sensors/2);
TND_all_cmbd_indv = cell(1,n_FM_sensors/2);
FND_all_cmbd_indv = cell(1,n_FM_sensors/2);

for r = 1 : n_iter        
    fprintf('\n\tIteration: %.0f/%.0f',r,n_iter) % Displaying the current value of the multiplier on screen

    h{r} = zeros(n_data_files,n_FM_sensors); % matrix to contain threshold values          
    for i = 1:n_data_files        
        % ~~~~~~~~~~ SEGMENTATION AND EXCLUSION OF BODY MOVEMENT ~~~~~~~~~%
        % get_segmented_data() function will be used here, which will
        % threshold the data, remove body movement, and dilate the data.
        % Setting for the threshold and dilation are given in the function.
        %   Input variables:  sensor_data- a cell variable;
        %                     min_SN- a vector/ a scalar;
        %                     IMU_map- a column vector
        %                     Fs_sensor, FM_dilation_time- a scalar
        %   Output variables: sensor_data_sgmntd- a cell variable of same size as the input variable sensor_data_fltd;
        %                     h- a vector
        
        sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
            Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};        
        [sensor_data_sgmntd, h{r}(i,:)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);
        
        % ~~~~~~ COMBINING SEGMENTED DATA FROM SAME TYPE OF SENSORS ~~~~~~~
        sensor_data_sgmntd_cmbd{1} = double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2}); % Combined segmented Aclm data
        sensor_data_sgmntd_cmbd{2} = double(sensor_data_sgmntd{3} | sensor_data_sgmntd{4}); % Combined segmented Acstc data
        sensor_data_sgmntd_cmbd{3} = double(sensor_data_sgmntd{5} | sensor_data_sgmntd{6}); % Combined segmented Pzplt data

        % ~~~~~~~~~~~~~~~ MATCHING WITH MATERNAL SENSATION ~~~~~~~~~~~~~~~%            
        [TPD_cmbd, FPD_cmbd, TND_cmbd, FND_cmbd] = match_with_m_sensation(sensor_data_sgmntd_cmbd, ...
            sensation_data_SD1_trimd{i},IMU_map{i},M_sntn_map{i},ext_backward,ext_forward, ...
            FM_dilation_time,Fs_sensor,Fs_sensation); % This is for combined sensor of each type
        
        % ~~~~~~~~~~~~~~~~~~ DATA STORING FOR POST-PROCESSING ~~~~~~~~~~~~%
        for k = 1 : n_FM_sensors/2
            TPD_all_cmbd_indv{k}(i,r) = TPD_cmbd(k); % This is for combined sensor of each type
            FPD_all_cmbd_indv{k}(i,r) = FPD_cmbd(k);
            TND_all_cmbd_indv{k}(i,r) = TND_cmbd(k);
            FND_all_cmbd_indv{k}(i,r) = FND_cmbd(k);
        end
        % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
    end
            
    FM_min_SN = FM_min_SN + SN_increment; % Updating the FM_min_SN at the end of each iterations
end

% End notification
fprintf('\nThreshold optimization is completed.\n\n')
%

% ======================== Performance analysis ===========================

% ---------------------- For individual data sets -------------------------
% Determination of performance parameters for each data sets across each iteration 
[SEN_all_cmbd_indv,PPV_all_cmbd_indv,SPE_all_cmbd_indv,ACC_all_cmbd_indv,FS_all_cmbd_indv,FPR_all_cmbd_indv]...
    = get_performance_params(TPD_all_cmbd_indv,FPD_all_cmbd_indv,TND_all_cmbd_indv,FND_all_cmbd_indv);
% get_performance_params() will return cell variables with parameters for
% each sensor in each cell. In a cell each row will have values for each
% data file and each column will have values for each iteration.

% Determination of the best multiplier and the corresponding performance parameters for each type of sensors
[r_best_cmbd_indv,FM_min_SN_best_cmbd_indv,Threshold_best_cmbd_indv,TPD_best_cmbd_indv,FPD_best_cmbd_indv,TND_best_cmbd_indv,FND_best_cmbd_indv,...
    SEN_best_cmbd_indv,PPV_best_cmbd_indv,SPE_best_cmbd_indv,ACC_best_cmbd_indv,FS_best_cmbd_indv] = get_optimum_params(n_data_files,n_FM_sensors/2,...
    FM_min_SN_initial,SN_increment,TPD_all_cmbd_indv,FPD_all_cmbd_indv,TND_all_cmbd_indv,FND_all_cmbd_indv,SEN_all_cmbd_indv,PPV_all_cmbd_indv,...
    SPE_all_cmbd_indv,ACC_all_cmbd_indv,FS_all_cmbd_indv,FPR_all_cmbd_indv, h); % This is for combined sensor of each type
% get_optimum_params() function will return matrix variables with each row
% representing result for each data set and each column will represent
% result for each sensor

% --------------------- For all the data sets combinedly -----------------%
% Determination of performance parameters for all the data sets combinedly, but across all the iteration 
TPD_all_cmbd_overall = cell(1, n_FM_sensors/2);
FPD_all_cmbd_overall = cell(1, n_FM_sensors/2);
TND_all_cmbd_overall = cell(1, n_FM_sensors/2);
FND_all_cmbd_overall = cell(1, n_FM_sensors/2);

for i = 1 : n_FM_sensors/2
    TPD_all_cmbd_overall{i} = sum(TPD_all_cmbd_indv{i} ,1); % This is for combined sensor of each type
    FPD_all_cmbd_overall{i} = sum(FPD_all_cmbd_indv{i} ,1);
    TND_all_cmbd_overall{i} = sum(TND_all_cmbd_indv{i} ,1);
    FND_all_cmbd_overall{i} = sum(FND_all_cmbd_indv{i} ,1);
end

[SEN_all_cmbd_overall, PPV_all_cmbd_overall, SPE_all_cmbd_overall, ACC_all_cmbd_overall, FS_all_cmbd_overall, FPR_all_cmbd_overall] ...
    = get_performance_params(TPD_all_cmbd_overall, FPD_all_cmbd_overall, TND_all_cmbd_overall, FND_all_cmbd_overall);

% Determination of the best multiplier and the corresponding performance parameters for each type of sensors
[r_best_cmbd_overall, FM_min_SN_best_cmbd_overall, ~ , TPD_best_cmbd_overall, FPD_best_cmbd_overall, TND_best_cmbd_overall, ...
    FND_best_cmbd_overall, SEN_best_cmbd_overall, PPV_best_cmbd_overall, SPE_best_cmbd_overall, ACC_best_cmbd_overall, FS_best_cmbd_overall, FPR_best_cmbd_overall] ...
    = get_optimum_params(1, n_FM_sensors/2, FM_min_SN_initial, SN_increment, TPD_all_cmbd_overall, FPD_all_cmbd_overall, TND_all_cmbd_overall, ...
    FND_all_cmbd_overall, SEN_all_cmbd_overall, PPV_all_cmbd_overall, SPE_all_cmbd_overall, ACC_all_cmbd_overall, FS_all_cmbd_overall, FPR_all_cmbd_overall, h); % This is for combined sensor of each type

% in case of overall performance, n_data_files is set to 1 while passing input to the get_optimum_params() function. 
% This is because, in this case detections from all the files are summed up to a single row vector.
% Additionally, the output parameter for the threshold is kept blank as it is not meaningful in case of overall performance 
% as the value will vary across the data files for the same value of threshold multiplier.
%

% ---------------------- Plotting ROC & PR curves ------------------------%
TPR_all_cmbd_overall = SEN_all_cmbd_overall; % SEN and TPR are same things 
legend_all           = {'Accelerometer', 'Acoustic sensor', 'Piezoelectric diaphragm'};
ROC_AUC_all          = zeros(1, length(TPR_all_cmbd_overall)); % Variable to store area under ROCs
PR_AUC_all           = zeros(1, length(TPR_all_cmbd_overall)); % Variable to store AUC of PR curves

fig1 = figure();
for i = 1 : length(TPR_all_cmbd_overall)

    % Plotting ROC curve
    ax1 = subplot(2,1,1); % ax1 is necessary to assign legend later
    plot(ax1, FPR_all_cmbd_overall{i}, TPR_all_cmbd_overall{i},'LineWidth', 1)
    hold on;
    xlabel('FPR');
    ylabel('TPR');
    title('ROC curves for different sensors');
    
    % Plotting PR curve
    ax2 = subplot(2,1,2);
    plot(ax2, TPR_all_cmbd_overall{i}, PPV_all_cmbd_overall{i}, 'LineWidth', 1)
    hold on;
    xlabel('Recall');
    ylabel('Precision');
    title('Precision vs. recall curves for different sensors');
    
    % Determining area under ROC curves
    end_index1     = find(~isnan(FPR_all_cmbd_overall{i}(:)), 1, 'last' );% To avoid any NAN value in FPR
    end_index2     = find(~isnan(TPR_all_cmbd_overall{i}(:)), 1, 'last' );% To avoid any NAN value in TPR
    end_index      = min(end_index1, end_index2);    
    ROC_AUC_all(i) = abs(trapz(FPR_all_cmbd_overall{i}(1:end_index), TPR_all_cmbd_overall{i}(1:end_index)));
    % Because the numbers are in descending order, abs() is used to avoid negative AUC
    
    % Determining area under PR curves
    end_index1    = find(~isnan(TPR_all_cmbd_overall{i}(:)), 1, 'last' );% To avoid any NAN value in FPR
    end_index2    = find(~isnan(PPV_all_cmbd_overall{i}(:)), 1, 'last' );% To avoid any NAN value in FPR
    end_index     = min(end_index1, end_index2);
    PR_AUC_all(i) = abs(trapz(TPR_all_cmbd_overall{i}(1:end_index), PPV_all_cmbd_overall{i}(1:end_index)));
    
end

legend (ax1, legend_all, 'Location', 'best');
legend (ax1, 'boxoff');
hold off;
legend (ax2, legend_all,'Location', 'best');
legend (ax2, 'boxoff');
hold off;
%

% --------- Determining Partial area under the ROC and PR curves --------%
% To compare between the AUC from different sensors, partial AUC is
% computed within the range of x-coordinate that is common to all. This 
% range is first computed by finding the max start value and min end value 
% of x-coordinate for different sensors.
min_x_ROC   = max(cellfun(@min, FPR_all_cmbd_overall)); % X-coordinate starts from the max value of the min x-coordinates for all the sensors
max_x_ROC   = min(cellfun(@max, FPR_all_cmbd_overall)); % X-coordinate ends at the min value of the max x-coordinates for all the sensors
x_coord_ROC = min_x_ROC : 0.01 : max_x_ROC; % This is a common x-coordinate for all the sensors

min_x_PR    = max(cellfun(@min, TPR_all_cmbd_overall)); % X-coordinate starts from the max value of the min x-coordinates for all the sensors
max_x_PR    = min(cellfun(@max, TPR_all_cmbd_overall)); % X-coordinate ends at the min value of the max x-coordinates for all the sensors
x_coord_PR  = min_x_PR : 0.01 : max_x_PR; % This is a common x-coordinate for all the sensors

% To compute the area under this common range, interpolation is necessary
% for all the curves. For the interpolation to work, the unique values
% of sampling points need to be removed from the data set. Once the
% interpolation is done, the partial AUC can simply be calculated by
% trapz() function.
FPR_all_cmbd_overall_unique       = cell(1, length(TPR_all_cmbd_overall));
TPR_all_cmbd_overall_unique       = cell(1, length(TPR_all_cmbd_overall));
TPR_all_cmbd_overall_interpolated = cell(1, length(TPR_all_cmbd_overall)); 
PPV_all_cmbd_overall_unique       = cell(1, length(TPR_all_cmbd_overall));
PPV_all_cmbd_overall_interpolated = cell(1, length(TPR_all_cmbd_overall));

ROC_AUC_partial_all = zeros(1, length(TPR_all_cmbd_overall));
PR_AUC_partial_all  = zeros(1, length(TPR_all_cmbd_overall));

for i = 1 : length(TPR_all_cmbd_overall)
    
    % Removing multiple occurance of same x-coordinate for ROC curves
    [FPR_all_cmbd_overall_unique{i}, ia] = unique(FPR_all_cmbd_overall{i}); % ia holds the index values
    TPR_all_cmbd_overall_unique{i}       = TPR_all_cmbd_overall{i}(ia); % Getting the same values for the TPR data
    
    % Calculating the partial AUC for ROC curves
    TPR_all_cmbd_overall_interpolated{i} = interp1(FPR_all_cmbd_overall_unique{i}, TPR_all_cmbd_overall_unique{i}, ...
        x_coord_ROC); % Interpolated TPR values for the given x coordinates
    ROC_AUC_partial_all(i) = trapz(x_coord_ROC, TPR_all_cmbd_overall_interpolated{i});
    
    % Removing multiple occurance of same x-coordinate for PR curves
    [TPR_all_cmbd_overall_unique{i}, ib] = unique(TPR_all_cmbd_overall{i}); % ia holds the index values
    PPV_all_cmbd_overall_unique{i}       = PPV_all_cmbd_overall{i}(ib); % Getting the same values for the TPR data
    
    % Calculating the partial AUC for PR curves
    PPV_all_cmbd_overall_interpolated{i} = interp1(TPR_all_cmbd_overall_unique{i}, PPV_all_cmbd_overall_unique{i}, ...
        x_coord_PR); % Interpolated TPR values for the given x coordinates
    PR_AUC_partial_all(i) = trapz(x_coord_PR, PPV_all_cmbd_overall_interpolated{i});

end
%

% ====================== Storing data in a table ==========================
% --------------------------- Table for general info ----------------------
T_general_info = table('Size',[n_data_files 4], 'VariableTypes', {'string', 'double', 'double', 'double'}) ;
for i = 1 : n_data_files
    if ischar(data_file_names) % If there is only a single data file
        DFN = data_file_names;
    else
        DFN = cellstr(data_file_names{i}(1:end-4)); % Data file names are converted from cell elements to strings
    end
    T_general_info(i,:) = [DFN, duration_raw_data_files(i), Force_mean(i), n_Maternal_detected_movement(i)];
end
T_general_info.Properties.VariableNames = {'Data file', 'Duration (min)', 'Mean Force (au)', 'No. of maternal sensation'}; % Assigns column names to the table
%

% --------------------- Storing data for individual sensors --------------%
sensor_names = ["Accelerometer"; "Acoustic sensor"; "Piezoelectric diaphragm"];

% Initialization of the table
T_all_cmbd_indv    = table('Size',[n_data_files*length(sensor_names) 15], 'VariableTypes', {'string', 'string', 'double', 'double', 'double', 'double', 'double', 'double', ...
    'double', 'double', 'double', 'double', 'double', 'double', 'double'}) ; 
T_all_cmbd_overall = table('Size',[length(sensor_names) 21], 'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', ...
    'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', ...
    'double', 'double', 'double' }) ; 

% Loop for storing the data
for i = 1 : length(sensor_names)
    
    for j = 1 : n_data_files        
        if ischar(data_file_names) % If there is only a single data file
            DFN = data_file_names;
        else
            DFN = cellstr(data_file_names{j}); % Data file names are converted from cell elements to strings
        end

        T_all_cmbd_indv{(i-1)*n_data_files + j, :} = [sensor_names(i), DFN, duration_raw_data_files(j), Force_mean(j), FM_min_SN_best_cmbd_indv(j,i), ...
            Threshold_best_cmbd_indv(j,i), FS_best_cmbd_indv(j,i), SEN_best_cmbd_indv(j,i), PPV_best_cmbd_indv(j,i), SPE_best_cmbd_indv(j,i), ...
            ACC_best_cmbd_indv(j,i), TPD_best_cmbd_indv(j,i), FPD_best_cmbd_indv(j,i), TND_best_cmbd_indv(j,i), FND_best_cmbd_indv(j,i)];        
    end
    
    T_all_cmbd_overall{i,:} = [sensor_names(i), sum(duration_raw_data_files), mean(Force_mean), FM_min_SN_best_cmbd_overall(i), FS_best_cmbd_overall(i), ...
        SEN_best_cmbd_overall(i), PPV_best_cmbd_overall(i), SPE_best_cmbd_overall(i), ACC_best_cmbd_overall(i), TPD_best_cmbd_overall(i), ...
        FPD_best_cmbd_overall(i), TND_best_cmbd_overall(i), FND_best_cmbd_overall(i), ROC_AUC_all(i), ROC_AUC_partial_all(i), min_x_ROC, ...
        max_x_ROC, PR_AUC_all(i), PR_AUC_partial_all(i), min_x_PR, max_x_PR];
    
end

T_all_cmbd_indv.Properties.VariableNames    = {'Sensor', 'Data file', 'Duration (min)', 'Mean Force (au)', 'Threshold multiplier', 'Threshold value',...
    'F Score', 'Sensitivity', 'PPV', 'Specificity', 'Accuracy', 'TPD', 'FPD', 'TND', 'FND'}; % Assigns column names to the table
T_all_cmbd_overall.Properties.VariableNames = {'Sensor Name', 'Duration (min)', 'Mean Force (au)', 'Threshold multiplier', 'F Score', ...
    'Sensitivity', 'PPV', 'Specificity', 'Accuracy', 'TPD', 'FPD', 'TND', 'FND', 'ROC_AUC', 'ROC_AUC_partial', 'Range_ROC_AUC_min', ...
    'Range_ROC_AUC_max', 'PR_AUC', 'PR_AUC_partial', 'Range_PR_AUC_min', 'Range_PR_AUC_max'}; % Assigns column names to the table
%

% --------------------- Displaying the stored data -----------------------%
disp('General info')
disp(T_general_info)
pause(2);

disp('Performance for Individual data sets: ');
disp(T_all_cmbd_indv)
pause(2);

disp('Performance for all the data sets combined: ');
disp(T_all_cmbd_overall)
pause(2);
%

%% PERFORMANCE EVALUATION BASED ON OPTIMUM MULTIPLIER =====================
% Starting notification
fprintf('\nPerformance analysis is going on... ')

% ----------- Calculation that will not vary across the iterations --------
% The IMU_map and the sensation_map are determined in this step.

% Parameters for creating sensation map and detection matching
ext_backward = 5.0; % Backward extension length in second 
ext_forward  = 2.0 ; % Forward extension length in second

% Variable decleration
IMU_map    = cell(1,n_data_files);   
M_sntn_map = cell(1,n_data_files);  
n_activity = zeros(n_data_files, 1); % Cell matrix to hold the number of fetal activities
n_Maternal_detected_movement = zeros(n_data_files, 1);

% Starting notification
fprintf('\n\tGenerating maternal sensation map...')

for i = 1 : n_data_files
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

        % ---------------- Determination of detection statistics --------------
        M_sntn_Map_labeled = bwlabel(M_sntn_map{i});
        n_activity(i)      = length(unique(M_sntn_Map_labeled)) - 1; % 1 is deducted to remove the first element, which is 0

        sensation_data_labeled          = bwlabel(sensation_data_SD1_trimd{i});
        n_Maternal_detected_movement(i) = length(unique(sensation_data_labeled)) - 1; % 1 is deducted to remove the initial value of 0
        %
end

fprintf('\n\tMaternal sensation map generation is completed.')

% ----------- Calculations that will vary across the iterations -----------
% Starting notification
fprintf('\n\tCalculation across individual data files is going on...\n')

% Parameters for FM data segmentation
n_FM_sensors     = 6; % number of FM sensors
FM_dilation_time = 3; % Dilation time for FM signal in second

% Variable decleration
TPD_all_indv_sensors_indv = cell(1, n_FM_sensors);
FPD_all_indv_sensors_indv = cell(1, n_FM_sensors);
TND_all_indv_sensors_indv = cell(1, n_FM_sensors);
FND_all_indv_sensors_indv = cell(1, n_FM_sensors);

threshold  = zeros(n_data_files, n_FM_sensors); % Variable for thresold
ROC_option = 1; % if 1, perform ROc, if 0 don't perform ROC

if ROC_option == 0
    n_iter    = 1;
    FM_min_SN = [30,30,30,30,30,30];
    % FM_min_SN = [FM_min_SN_best_cmbd_overall(1),FM_min_SN_best_cmbd_overall(1), ...
    %     FM_min_SN_best_cmbd_overall(2),FM_min_SN_best_cmbd_overall(2), ...
    %     FM_min_SN_best_cmbd_overall(3),FM_min_SN_best_cmbd_overall(3)];
    % Best multipliers for combined sensors of each types are considered here
    % Best multipliers are: Aclm = 30, Acstc = 50, Piezo = 40

else
    % Paramters for ROC analysis
    FM_min_SN_initial = 2000; % Initial value of the minimum signal to noise ratio that is used to get the threshold value
    FM_min_SN         = FM_min_SN_initial;
    max_SN            = 2000;
    SN_increment      = 100;
    n_iter            = 1+round((max_SN-FM_min_SN_initial)/SN_increment);
end

% Variable decteration
Overall_detection_Aclm = zeros(n_iter,4);
Overall_detection_Acstc = zeros(n_iter,4);
Overall_detection_Pzplt = zeros(n_iter,4);

Overall_detection_Aclm_OR_Acstc  = zeros(n_iter,4);
Overall_detection_Aclm_OR_Pzplt  = zeros(n_iter,4);
Overall_detection_Acstc_OR_Pzplt = zeros(n_iter,4);

Overall_detection_atleast_1_type = zeros(n_iter,4);
Overall_detection_atleast_2_type = zeros(n_iter,4);
Overall_detection_atleast_3_type = zeros(n_iter,4);

for r = 1 : n_iter      
    for i = 1 : n_data_files
        fprintf('\n\t\tCurrent data file: %.0f/%.0f (Iteration: %.0f/%.0f)',i,n_data_files,r, n_iter)

        % ~~~~~~~~~~~~~~~~~~~~~~ Segmentation of FM data ~~~~~~~~~~~~~~~~~~~~~%
        % get_segmented_data() function will be used here, which will
        % threshold the data, remove body movement, and dilate the data.
        % Setting for the threshold and dilation are given in the function.
        %   Input variables:  sensor_data- a cell variable;
        %                     min_SN- a vector/ a scalar;
        %                     IMU_map- a column vector
        %                     Fs_sensor, FM_dilation_time- a scalar
        %   Output variables: sensor_data_sgmntd- a cell variable of same size as the input variable sensor_data_fltd;
        %                     h- a vector

        sensor_data_fltd = {Aclm_data1_fltd{i},Aclm_data2_fltd{i},Acstc_data1_fltd{i},Acstc_data2_fltd{i},...
            Pzplt_data1_fltd{i},Pzplt_data2_fltd{i}};
        [sensor_data_sgmntd,threshold(i,:)] = get_segmented_data(sensor_data_fltd,FM_min_SN,IMU_map{i},...
            FM_dilation_time,Fs_sensor);

        % ~~~~~~~~~~~~~~~~~~~~~~~~~~ Sensor fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        % First fuse the left and right sensors of each type using OR
        sensor_data_sgmntd_Aclm  = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
        sensor_data_sgmntd_Acstc = {double(sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};
        sensor_data_sgmntd_Pzplt = {double(sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};

        % Then apply different sensor fusion scheme to combine different types of sensors

        % SENSOR FUSION SCHEME 1: Combination based on logical OR operation
        %   In this scheme, data fusion is performed before dilation.
        %   Combined data are stored as a cell to make it compatable with the
        %   function related to matcing with maternal sensation
        sensor_data_sgmntd_Aclm_OR_Acstc  = {double(sensor_data_sgmntd_Aclm{1}  | sensor_data_sgmntd_Acstc{1})};
        sensor_data_sgmntd_Aclm_OR_Pzplt  = {double(sensor_data_sgmntd_Aclm{1}  | sensor_data_sgmntd_Pzplt{1})};
        sensor_data_sgmntd_Acstc_OR_Pzplt = {double(sensor_data_sgmntd_Acstc{1} | sensor_data_sgmntd_Pzplt{1})};

        sensor_data_sgmntd_cmbd_all_sensors_OR = {double(sensor_data_sgmntd_Aclm{1} | sensor_data_sgmntd_Acstc{1} | sensor_data_sgmntd_Pzplt{1})};
        %

        % SENSOR FUSION SCHEME 2: Combination based on logical AND operation
        %   In this scheme, data fusion is performed after dilation.
        %   Combined data are stored as a cell to make it compatable with the
        %   function related to matcing with maternal sensation

        %         sensor_data_sgmntd_Aclm_AND_Acstc  = {double(sensor_data_sgmntd_Aclm{1}  & sensor_data_sgmntd_Acstc{1})};
        %         sensor_data_sgmntd_Aclm_AND_Pzplt  = {double(sensor_data_sgmntd_Aclm{1}  & sensor_data_sgmntd_Pzplt{1})};
        %         sensor_data_sgmntd_Acstc_AND_Pzplt = {double(sensor_data_sgmntd_Acstc{1} & sensor_data_sgmntd_Pzplt{1})};
        %
        %         sensor_data_sgmntd_cmbd_all_sensors_AND = {double(sensor_data_sgmntd_Aclm{1} & sensor_data_sgmntd_Acstc{1} & sensor_data_sgmntd_Pzplt{1})};
        %         

        % SENSOR FUSION SCHEME 3: Combination based on detection by atleaset n sensors
        %   In this scheme, data fusion is performed after dilation.
        %   All the sensor data are first combined with logical OR. Each
        %   non-zero sengment in the combined data is then checked against
        %   individual sensor data to find its presence in that data set.
        %   Combined data are stored as a cell to make it compatable with the
        %   function related to matcing with maternal sensation.
        sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
            sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
        sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
        n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

        sensor_data_sgmntd_cmbd_multi_type_sensors_OR = [sensor_data_sgmntd_Acstc, sensor_data_sgmntd_Aclm, sensor_data_sgmntd_Pzplt];

        % Initialization of variables
        %         sensor_data_sgmntd_atleast_1_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);
        %         sensor_data_sgmntd_atleast_2_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);
        %         sensor_data_sgmntd_atleast_3_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);
        %         sensor_data_sgmntd_atleast_4_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);
        %         sensor_data_sgmntd_atleast_5_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);
        %         sensor_data_sgmntd_atleast_6_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);

        sensor_data_sgmntd_atleast_1_type{1}   = zeros(length(sensor_data_sgmntd{1}),1);
        sensor_data_sgmntd_atleast_2_type{1}   = zeros(length(sensor_data_sgmntd{1}),1);
        sensor_data_sgmntd_atleast_3_type{1}   = zeros(length(sensor_data_sgmntd{1}),1);

        if (n_label) % When there is a detection by the sensor system

            for k = 1 : n_label

                L_min = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1 ); % Sample no. corresponding to start of the label
                L_max = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1, 'last' ); % Sample no. corresponding to end of the label

                indv_detection_map = zeros(length(sensor_data_sgmntd{1}),1); % Need to be initialized before every detection matching
                indv_detection_map(L_min:L_max) = 1; % mapping individual sensation data

                % For detection by at least n sensors
                %                 tmp_var = 0; % Variable to hold number of common sensors for each detection
                %                 for j = 1:n_FM_sensors
                %                     if (sum(indv_detection_map.*sensor_data_sgmntd{j})) % Non-zero value indicates intersection
                %                         tmp_var = tmp_var + 1;
                %                     end
                %                 end
                %
                %                 switch tmp_var
                %                     case 1
                %                         sensor_data_sgmntd_atleast_1_sensor{1}(L_min:L_max) = 1;
                %                     case 2
                %                         sensor_data_sgmntd_atleast_1_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_2_sensor{1}(L_min:L_max) = 1;
                %                     case 3
                %                         sensor_data_sgmntd_atleast_1_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_2_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_3_sensor{1}(L_min:L_max) = 1;
                %                     case 4
                %                         sensor_data_sgmntd_atleast_1_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_2_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_3_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_4_sensor{1}(L_min:L_max) = 1;
                %                     case 5
                %                         sensor_data_sgmntd_atleast_1_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_2_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_3_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_4_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_5_sensor{1}(L_min:L_max) = 1;
                %                     case 6
                %                         sensor_data_sgmntd_atleast_1_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_2_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_3_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_4_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_5_sensor{1}(L_min:L_max) = 1;
                %                         sensor_data_sgmntd_atleast_6_sensor{1}(L_min:L_max) = 1;
                %                     otherwise
                %                         disp('This would never print.')
                %                 end

                % For detection by at least n type of sensors
                tmp_var = 0; % Variable to hold number of common sensors for each detection
                for j = 1:n_FM_sensors/2
                    if (sum(indv_detection_map.*sensor_data_sgmntd_cmbd_multi_type_sensors_OR{j})) % Non-zero value indicates intersection
                        tmp_var = tmp_var + 1;
                    end
                end

                switch tmp_var
                    case 1
                        sensor_data_sgmntd_atleast_1_type{1}(L_min:L_max) = 1;
                    case 2
                        sensor_data_sgmntd_atleast_1_type{1}(L_min:L_max) = 1;
                        sensor_data_sgmntd_atleast_2_type{1}(L_min:L_max) = 1;
                    case 3
                        sensor_data_sgmntd_atleast_1_type{1}(L_min:L_max) = 1;
                        sensor_data_sgmntd_atleast_2_type{1}(L_min:L_max) = 1;
                        sensor_data_sgmntd_atleast_3_type{1}(L_min:L_max) = 1;
                    otherwise
                        disp('This should never print.')
                end
            end
        end
        %

        % ~~~~~~~~~~~~~~~~~ Matching with maternal sensation ~~~~~~~~~~~~~~~~~%
        %   match_with_m_sensation() function will be used here-
        %   Input variables:  sensor_data_sgmntd- A cell variable with single cell/multiple cells.
        %                                         Each cell contains data from a sensor or a combination.
        %                     sensation_data, IMU_map, M_sntn_Map- cell variables with single cell
        %                     ext_bakward, ext_forward, FM_dilation_time- scalar values
        %                     Fs_sensor, Fs_sensation- scalar values
        %   Output variables: TPD, FPD, TND, FND- vectors with number of rows equal to the
        %                                         number of cells in the sensor_data_sgmntd

        % For individual sensors
        %   The input argument 'sensor_data_sgmntd' will be a multi-cell
        %   variable with number of cell = number of sensor data.
        %   Hence, the function will return 6 x 1 vectors for each output
        %   argument because the input data has 6 cells.
        %   The values are then stored in cell variables to make it compatable
        %   with the performance analysis section.

        %         [TPD_indv_sensors, FPD_indv_sensors, TND_indv_sensors, FND_indv_sensors] = match_with_m_sensation(sensor_data_sgmntd, sensation_data_SD1_trimd{i}, ...
        %             IMU_map{i}, M_sntn_map{i}, ext_backward, ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         for j = 1 : n_FM_sensors
        %             TPD_all_indv_sensors_indv{j}(i,1) = TPD_indv_sensors(j);
        %             FPD_all_indv_sensors_indv{j}(i,1) = FPD_indv_sensors(j);
        %             TND_all_indv_sensors_indv{j}(i,1) = TND_indv_sensors(j);
        %             FND_all_indv_sensors_indv{j}(i,1) = FND_indv_sensors(j);
        %             % Above variables will have 6 cell arrays, each for one sensors.
        %             % Each cell will contain a vector with row numbers = n_data_files.
        %         end
        %

        % For combined sensors
        %   input argument 'sensor_data_sgmntd' will be a single-cell variable.
        %   Hence the function will return a scalar value for each output argument.
        %   Cell variable is used to store the output data to make it
        %   compatible with the performance analysis section.
        %
        % Combined left and right sensors
        [TPD_Acstc_indv{1}(i,1), FPD_Acstc_indv{1}(i,1), TND_Acstc_indv{1}(i,1), FND_Acstc_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_Acstc, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_Aclm_indv{1}(i,1), FPD_Aclm_indv{1}(i,1), TND_Aclm_indv{1}(i,1), FND_Aclm_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_Aclm, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_Pzplt_indv{1}(i,1), FPD_Pzplt_indv{1}(i,1), TND_Pzplt_indv{1}(i,1), FND_Pzplt_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_Pzplt, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);

        % Combination with sensor fusion scheme 1
        [TPD_Aclm_OR_Acstc_indv{1}(i,1), FPD_Aclm_OR_Acstc_indv{1}(i,1), TND_Aclm_OR_Acstc_indv{1}(i,1), FND_Aclm_OR_Acstc_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_Aclm_OR_Acstc, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_Aclm_OR_Pzplt_indv{1}(i,1), FPD_Aclm_OR_Pzplt_indv{1}(i,1), TND_Aclm_OR_Pzplt_indv{1}(i,1), FND_Aclm_OR_Pzplt_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_Aclm_OR_Pzplt, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_Acstc_OR_Pzplt_indv{1}(i,1), FPD_Acstc_OR_Pzplt_indv{1}(i,1), TND_Acstc_OR_Pzplt_indv{1}(i,1), FND_Acstc_OR_Pzplt_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_Acstc_OR_Pzplt, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_all_sensors_OR_cmbd_indv{1}(i,1), FPD_all_sensors_OR_cmbd_indv{1}(i,1), TND_all_sensors_OR_cmbd_indv{1}(i,1), FND_all_sensors_OR_cmbd_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_cmbd_all_sensors_OR, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        
        % Combination with sensor fusion scheme 2
        %         [TPD_Aclm_AND_Acstc_indv{1}(i,1), FPD_Aclm_AND_Acstc_indv{1}(i,1), TND_Aclm_AND_Acstc_indv{1}(i,1), FND_Aclm_AND_Acstc_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_Aclm_AND_Acstc, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_Aclm_AND_Pzplt_indv{1}(i,1), FPD_Aclm_AND_Pzplt_indv{1}(i,1), TND_Aclm_AND_Pzplt_indv{1}(i,1), FND_Aclm_AND_Pzplt_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_Aclm_AND_Pzplt, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_Acstc_AND_Pzplt_indv{1}(i,1), FPD_Acstc_AND_Pzplt_indv{1}(i,1), TND_Acstc_AND_Pzplt_indv{1}(i,1), FND_Acstc_AND_Pzplt_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_Acstc_AND_Pzplt, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_all_sensors_AND_cmbd_indv{1}(i,1), FPD_all_sensors_AND_cmbd_indv{1}(i,1), TND_all_sensors_AND_cmbd_indv{1}(i,1), FND_all_sensors_AND_cmbd_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_cmbd_all_sensors_AND, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        
        % Combination with sensor fusion scheme 3
        %         [TPD_atleast_1_sensor_indv{1}(i,1), FPD_atleast_1_sensor_indv{1}(i,1), TND_atleast_1_sensor_indv{1}(i,1), FND_atleast_1_sensor_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_atleast_1_sensor, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_atleast_2_sensor_indv{1}(i,1), FPD_atleast_2_sensor_indv{1}(i,1), TND_atleast_2_sensor_indv{1}(i,1), FND_atleast_2_sensor_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_atleast_2_sensor, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_atleast_3_sensor_indv{1}(i,1), FPD_atleast_3_sensor_indv{1}(i,1), TND_atleast_3_sensor_indv{1}(i,1), FND_atleast_3_sensor_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_atleast_3_sensor, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_atleast_4_sensor_indv{1}(i,1), FPD_atleast_4_sensor_indv{1}(i,1), TND_atleast_4_sensor_indv{1}(i,1), FND_atleast_4_sensor_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_atleast_4_sensor, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_atleast_5_sensor_indv{1}(i,1), FPD_atleast_5_sensor_indv{1}(i,1), TND_atleast_5_sensor_indv{1}(i,1), FND_atleast_5_sensor_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_atleast_5_sensor, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %         [TPD_atleast_6_sensor_indv{1}(i,1), FPD_atleast_6_sensor_indv{1}(i,1), TND_atleast_6_sensor_indv{1}(i,1), FND_atleast_6_sensor_indv{1}(i,1)] ...
        %             = match_with_m_sensation(sensor_data_sgmntd_atleast_6_sensor, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
        %             ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);

        [TPD_atleast_1_type_indv{1}(i,1), FPD_atleast_1_type_indv{1}(i,1), TND_atleast_1_type_indv{1}(i,1), FND_atleast_1_type_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_atleast_1_type, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_atleast_2_type_indv{1}(i,1), FPD_atleast_2_type_indv{1}(i,1), TND_atleast_2_type_indv{1}(i,1), FND_atleast_2_type_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_atleast_2_type, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        [TPD_atleast_3_type_indv{1}(i,1), FPD_atleast_3_type_indv{1}(i,1), TND_atleast_3_type_indv{1}(i,1), FND_atleast_3_type_indv{1}(i,1)] ...
            = match_with_m_sensation(sensor_data_sgmntd_atleast_3_type, sensation_data_SD1_trimd{i}, IMU_map{i}, M_sntn_map{i}, ext_backward, ...
            ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
        %
    end

    % ----------------------- Performance analysis ---------------------------%
    % This section will use get_performance_params() function
    %   Input variables:  TPD_all, FPD_all, TND_all, FND_all- single cell/multi-cell variable.
    %                     Number of cell indicates number of sensor data or
    %                     combination data provided together.
    %                     Each cell containes a vector with row size = n_data_files
    %
    %   Output variables: SEN_all,PPV_all,SPE_all,ACC_all,FS_all,FPR_all- cell variable with
    %                     size same as the input variables.

    % For individual data sets
    % For individual sensors
    %     [SEN_all_indv_sensors_indv, PPV_all_indv_sensors_indv, SPE_all_indv_sensors_indv, ACC_all_indv_sensors_indv, FS_all_indv_sensors_indv, FPR_all_indv_sensors_indv] ...
    %         = get_performance_params(TPD_all_indv_sensors_indv, FPD_all_indv_sensors_indv, TND_all_indv_sensors_indv, FND_all_indv_sensors_indv);

    % For combined individual sensors
    [SEN_Aclm_indv, PPV_Aclm_indv, SPE_Aclm_indv, ACC_Aclm_indv, FS_Aclm_indv, FPR_Aclm_indv] ...
        = get_performance_params(TPD_Aclm_indv, FPD_Aclm_indv, TND_Aclm_indv, FND_Aclm_indv);
    [SEN_Acstc_indv, PPV_Acstc_indv, SPE_Acstc_indv, ACC_Acstc_indv, FS_Acstc_indv, FPR_Acstc_indv] ...
        = get_performance_params(TPD_Acstc_indv, FPD_Acstc_indv, TND_Acstc_indv, FND_Acstc_indv);
    [SEN_Pzplt_indv, PPV_Pzplt_indv, SPE_Pzplt_indv, ACC_Pzplt_indv, FS_Pzplt_indv, FPR_Pzplt_indv] ...
        = get_performance_params(TPD_Pzplt_indv, FPD_Pzplt_indv, TND_Pzplt_indv, FND_Pzplt_indv);

    % For sensor combinaitons
    [SEN_Aclm_OR_Acstc_indv, PPV_Aclm_OR_Acstc_indv, SPE_Aclm_OR_Acstc_indv, ACC_Aclm_OR_Acstc_indv, FS_Aclm_OR_Acstc_indv, FPR_Aclm_OR_Acstc_indv] ...
        = get_performance_params(TPD_Aclm_OR_Acstc_indv, FPD_Aclm_OR_Acstc_indv, TND_Aclm_OR_Acstc_indv, FND_Aclm_OR_Acstc_indv);
    [SEN_Aclm_OR_Pzplt_indv, PPV_Aclm_OR_Pzplt_indv, SPE_Aclm_OR_Pzplt_indv, ACC_Aclm_OR_Pzplt_indv, FS_Aclm_OR_Pzplt_indv, FPR_Aclm_OR_Pzplt_indv] ...
        = get_performance_params(TPD_Aclm_OR_Pzplt_indv, FPD_Aclm_OR_Pzplt_indv, TND_Aclm_OR_Pzplt_indv, FND_Aclm_OR_Pzplt_indv);
    [SEN_Acstc_OR_Pzplt_indv, PPV_Acstc_OR_Pzplt_indv, SPE_Acstc_OR_Pzplt_indv, ACC_Acstc_OR_Pzplt_indv, FS_Acstc_OR_Pzplt_indv, FPR_Acstc_OR_Pzplt_indv] ...
        = get_performance_params(TPD_Acstc_OR_Pzplt_indv, FPD_Acstc_OR_Pzplt_indv, TND_Acstc_OR_Pzplt_indv, FND_Acstc_OR_Pzplt_indv);
    [SEN_all_sensors_OR_cmbd_indv, PPV_all_sensors_OR_cmbd_indv, SPE_all_sensors_OR_cmbd_indv, ACC_all_sensors_OR_cmbd_indv, FS_all_sensors_OR_cmbd_indv, FPR_all_OR_sensors_cmbd_indv] ...
        =  get_performance_params(TPD_all_sensors_OR_cmbd_indv, FPD_all_sensors_OR_cmbd_indv, TND_all_sensors_OR_cmbd_indv, FND_all_sensors_OR_cmbd_indv);

    %     [SEN_Aclm_AND_Acstc_indv, PPV_Aclm_AND_Acstc_indv, SPE_Aclm_AND_Acstc_indv, ACC_Aclm_AND_Acstc_indv, FS_Aclm_AND_Acstc_indv, FPR_Aclm_AND_Acstc_indv] ...
    %         = get_performance_params(TPD_Aclm_AND_Acstc_indv, FPD_Aclm_AND_Acstc_indv, TND_Aclm_AND_Acstc_indv, FND_Aclm_AND_Acstc_indv);
    %     [SEN_Aclm_AND_Pzplt_indv, PPV_Aclm_AND_Pzplt_indv, SPE_Aclm_AND_Pzplt_indv, ACC_Aclm_AND_Pzplt_indv, FS_Aclm_AND_Pzplt_indv, FPR_Aclm_AND_Pzplt_indv] ...
    %         = get_performance_params(TPD_Aclm_AND_Pzplt_indv, FPD_Aclm_AND_Pzplt_indv, TND_Aclm_AND_Pzplt_indv, FND_Aclm_AND_Pzplt_indv);
    %     [SEN_Acstc_AND_Pzplt_indv, PPV_Acstc_AND_Pzplt_indv, SPE_Acstc_AND_Pzplt_indv, ACC_Acstc_AND_Pzplt_indv, FS_Acstc_AND_Pzplt_indv, FPR_Acstc_AND_Pzplt_indv] ...
    %         = get_performance_params(TPD_Acstc_AND_Pzplt_indv, FPD_Acstc_AND_Pzplt_indv, TND_Acstc_AND_Pzplt_indv, FND_Acstc_AND_Pzplt_indv);
    %     [SEN_all_sensors_AND_cmbd_indv, PPV_all_sensors_AND_cmbd_indv, SPE_all_sensors_AND_cmbd_indv, ACC_all_sensors_AND_cmbd_indv, FS_all_sensors_AND_cmbd_indv, FPR_all_AND_sensors_cmbd_indv] ...
    %         =  get_performance_params(TPD_all_sensors_AND_cmbd_indv, FPD_all_sensors_AND_cmbd_indv, TND_all_sensors_AND_cmbd_indv, FND_all_sensors_AND_cmbd_indv);

    %     [SEN_atleast_1_sensor_indv, PPV_atleast_1_sensor_indv, SPE_atleast_1_sensor_indv, ACC_atleast_1_sensor_indv, FS_atleast_1_sensor_indv, FPR_atleast_1_sensor_indv] ...
    %         = get_performance_params(TPD_atleast_1_sensor_indv, FPD_atleast_1_sensor_indv, TND_atleast_1_sensor_indv, FND_atleast_1_sensor_indv);
    %     [SEN_atleast_2_sensor_indv, PPV_atleast_2_sensor_indv, SPE_atleast_2_sensor_indv, ACC_atleast_2_sensor_indv, FS_atleast_2_sensor_indv, FPR_atleast_2_sensor_indv] ...
    %         = get_performance_params(TPD_atleast_2_sensor_indv, FPD_atleast_2_sensor_indv, TND_atleast_2_sensor_indv, FND_atleast_2_sensor_indv);
    %     [SEN_atleast_3_sensor_indv, PPV_atleast_3_sensor_indv, SPE_atleast_3_sensor_indv, ACC_atleast_3_sensor_indv, FS_atleast_3_sensor_indv, FPR_atleast_3_sensor_indv] ...
    %         = get_performance_params(TPD_atleast_3_sensor_indv, FPD_atleast_3_sensor_indv, TND_atleast_3_sensor_indv, FND_atleast_3_sensor_indv);
    %     [SEN_atleast_4_sensor_indv, PPV_atleast_4_sensor_indv, SPE_atleast_4_sensor_indv, ACC_atleast_4_sensor_indv, FS_atleast_4_sensor_indv, FPR_atleast_4_sensor_indv] ...
    %         = get_performance_params(TPD_atleast_4_sensor_indv, FPD_atleast_4_sensor_indv, TND_atleast_4_sensor_indv, FND_atleast_4_sensor_indv);
    %     [SEN_atleast_5_sensor_indv, PPV_atleast_5_sensor_indv, SPE_atleast_5_sensor_indv, ACC_atleast_5_sensor_indv, FS_atleast_5_sensor_indv, FPR_atleast_5_sensor_indv] ...
    %         = get_performance_params(TPD_atleast_5_sensor_indv, FPD_atleast_5_sensor_indv, TND_atleast_5_sensor_indv, FND_atleast_5_sensor_indv);
    %     [SEN_atleast_6_sensor_indv, PPV_atleast_6_sensor_indv, SPE_atleast_6_sensor_indv, ACC_atleast_6_sensor_indv, FS_atleast_6_sensor_indv, FPR_atleast_6_sensor_indv] ...
    %         = get_performance_params(TPD_atleast_6_sensor_indv, FPD_atleast_6_sensor_indv, TND_atleast_6_sensor_indv, FND_atleast_6_sensor_indv);

    [SEN_atleast_1_type_indv, PPV_atleast_1_type_indv, SPE_atleast_1_type_indv, ACC_atleast_1_type_indv, FS_atleast_1_type_indv, FPR_atleast_1_type_indv] ...
        = get_performance_params(TPD_atleast_1_type_indv, FPD_atleast_1_type_indv, TND_atleast_1_type_indv, FND_atleast_1_type_indv);
    [SEN_atleast_2_type_indv, PPV_atleast_2_type_indv, SPE_atleast_2_type_indv, ACC_atleast_2_type_indv, FS_atleast_2_type_indv, FPR_atleast_2_type_indv] ...
        = get_performance_params(TPD_atleast_2_type_indv, FPD_atleast_2_type_indv, TND_atleast_2_type_indv, FND_atleast_2_type_indv);
    [SEN_atleast_3_type_indv, PPV_atleast_3_type_indv, SPE_atleast_3_type_indv, ACC_atleast_3_type_indv, FS_atleast_3_type_indv, FPR_atleast_3_type_indv] ...
        = get_performance_params(TPD_atleast_3_type_indv, FPD_atleast_3_type_indv, TND_atleast_3_type_indv, FND_atleast_3_type_indv);

    % For all the data sets combinedly
    %     TPD_all_indv_sensors_overall = cell(1, n_FM_sensors);
    %     FPD_all_indv_sensors_overall = cell(1, n_FM_sensors);
    %     TND_all_indv_sensors_overall = cell(1, n_FM_sensors);
    %     FND_all_indv_sensors_overall = cell(1, n_FM_sensors);
    %
    %     for j = 1 : n_FM_sensors
    %         TPD_all_indv_sensors_overall{j} = sum(TPD_all_indv_sensors_indv{j} ,1); % Sums up the elements of each column and returns a row vector
    %         FPD_all_indv_sensors_overall{j} = sum(FPD_all_indv_sensors_indv{j} ,1);
    %         TND_all_indv_sensors_overall{j} = sum(TND_all_indv_sensors_indv{j} ,1);
    %         FND_all_indv_sensors_overall{j} = sum(FND_all_indv_sensors_indv{j} ,1);
    %     end

    TPD_Aclm_overall{1}  = sum(TPD_Aclm_indv{1} ,1); % Sums up across a column
    FPD_Aclm_overall{1}  = sum(FPD_Aclm_indv{1} ,1);
    TND_Aclm_overall{1}  = sum(TND_Aclm_indv{1} ,1);
    FND_Aclm_overall{1}  = sum(FND_Aclm_indv{1} ,1);
    TPD_Acstc_overall{1} = sum(TPD_Acstc_indv{1} ,1);
    FPD_Acstc_overall{1} = sum(FPD_Acstc_indv{1} ,1);
    TND_Acstc_overall{1} = sum(TND_Acstc_indv{1} ,1);
    FND_Acstc_overall{1} = sum(FND_Acstc_indv{1} ,1);
    TPD_Pzplt_overall{1} = sum(TPD_Pzplt_indv{1} ,1);
    FPD_Pzplt_overall{1} = sum(FPD_Pzplt_indv{1} ,1);
    TND_Pzplt_overall{1} = sum(TND_Pzplt_indv{1} ,1);
    FND_Pzplt_overall{1} = sum(FND_Pzplt_indv{1} ,1);

    TPD_Aclm_OR_Acstc_overall{1}        = sum(TPD_Aclm_OR_Acstc_indv{1} ,1);
    FPD_Aclm_OR_Acstc_overall{1}        = sum(FPD_Aclm_OR_Acstc_indv{1} ,1);
    TND_Aclm_OR_Acstc_overall{1}        = sum(TND_Aclm_OR_Acstc_indv{1} ,1);
    FND_Aclm_OR_Acstc_overall{1}        = sum(FND_Aclm_OR_Acstc_indv{1} ,1);
    TPD_Aclm_OR_Pzplt_overall{1}        = sum(TPD_Aclm_OR_Pzplt_indv{1} ,1);
    FPD_Aclm_OR_Pzplt_overall{1}        = sum(FPD_Aclm_OR_Pzplt_indv{1} ,1);
    TND_Aclm_OR_Pzplt_overall{1}        = sum(TND_Aclm_OR_Pzplt_indv{1} ,1);
    FND_Aclm_OR_Pzplt_overall{1}        = sum(FND_Aclm_OR_Pzplt_indv{1} ,1);
    TPD_Acstc_OR_Pzplt_overall{1}       = sum(TPD_Acstc_OR_Pzplt_indv{1} ,1);
    FPD_Acstc_OR_Pzplt_overall{1}       = sum(FPD_Acstc_OR_Pzplt_indv{1} ,1);
    TND_Acstc_OR_Pzplt_overall{1}       = sum(TND_Acstc_OR_Pzplt_indv{1} ,1);
    FND_Acstc_OR_Pzplt_overall{1}       = sum(FND_Acstc_OR_Pzplt_indv{1} ,1);
    TPD_all_sensors_OR_cmbd_overall{1}  = sum(TPD_all_sensors_OR_cmbd_indv{1} ,1);
    FPD_all_sensors_OR_cmbd_overall{1}  = sum(FPD_all_sensors_OR_cmbd_indv{1} ,1);
    TND_all_sensors_OR_cmbd_overall{1}  = sum(TND_all_sensors_OR_cmbd_indv{1} ,1);
    FND_all_sensors_OR_cmbd_overall{1}  = sum(FND_all_sensors_OR_cmbd_indv{1} ,1);

    %     TPD_Aclm_AND_Acstc_overall{1}       = sum(TPD_Aclm_AND_Acstc_indv{1} ,1);
    %     FPD_Aclm_AND_Acstc_overall{1}       = sum(FPD_Aclm_AND_Acstc_indv{1} ,1);
    %     TND_Aclm_AND_Acstc_overall{1}       = sum(TND_Aclm_AND_Acstc_indv{1} ,1);
    %     FND_Aclm_AND_Acstc_overall{1}       = sum(FND_Aclm_AND_Acstc_indv{1} ,1);
    %     TPD_Aclm_AND_Pzplt_overall{1}       = sum(TPD_Aclm_AND_Pzplt_indv{1} ,1);
    %     FPD_Aclm_AND_Pzplt_overall{1}       = sum(FPD_Aclm_AND_Pzplt_indv{1} ,1);
    %     TND_Aclm_AND_Pzplt_overall{1}       = sum(TND_Aclm_AND_Pzplt_indv{1} ,1);
    %     FND_Aclm_AND_Pzplt_overall{1}       = sum(FND_Aclm_AND_Pzplt_indv{1} ,1);
    %     TPD_Acstc_AND_Pzplt_overall{1}      = sum(TPD_Acstc_AND_Pzplt_indv{1} ,1);
    %     FPD_Acstc_AND_Pzplt_overall{1}      = sum(FPD_Acstc_AND_Pzplt_indv{1} ,1);
    %     TND_Acstc_AND_Pzplt_overall{1}      = sum(TND_Acstc_AND_Pzplt_indv{1} ,1);
    %     FND_Acstc_AND_Pzplt_overall{1}      = sum(FND_Acstc_AND_Pzplt_indv{1} ,1);
    %     TPD_all_sensors_AND_cmbd_overall{1} = sum(TPD_all_sensors_AND_cmbd_indv{1} ,1);
    %     FPD_all_sensors_AND_cmbd_overall{1} = sum(FPD_all_sensors_AND_cmbd_indv{1} ,1);
    %     TND_all_sensors_AND_cmbd_overall{1} = sum(TND_all_sensors_AND_cmbd_indv{1} ,1);
    %     FND_all_sensors_AND_cmbd_overall{1} = sum(FND_all_sensors_AND_cmbd_indv{1} ,1);

    %     TPD_atleast_1_sensor_overall{1}     = sum(TPD_atleast_1_sensor_indv{1} ,1); % Sums up across a column
    %     FPD_atleast_1_sensor_overall{1}     = sum(FPD_atleast_1_sensor_indv{1} ,1);
    %     TND_atleast_1_sensor_overall{1}     = sum(TND_atleast_1_sensor_indv{1} ,1);
    %     FND_atleast_1_sensor_overall{1}     = sum(FND_atleast_1_sensor_indv{1} ,1);
    %     TPD_atleast_2_sensor_overall{1}     = sum(TPD_atleast_2_sensor_indv{1} ,1);
    %     FPD_atleast_2_sensor_overall{1}     = sum(FPD_atleast_2_sensor_indv{1} ,1);
    %     TND_atleast_2_sensor_overall{1}     = sum(TND_atleast_2_sensor_indv{1} ,1);
    %     FND_atleast_2_sensor_overall{1}     = sum(FND_atleast_2_sensor_indv{1} ,1);
    %     TPD_atleast_3_sensor_overall{1}     = sum(TPD_atleast_3_sensor_indv{1} ,1);
    %     FPD_atleast_3_sensor_overall{1}     = sum(FPD_atleast_3_sensor_indv{1} ,1);
    %     TND_atleast_3_sensor_overall{1}     = sum(TND_atleast_3_sensor_indv{1} ,1);
    %     FND_atleast_3_sensor_overall{1}     = sum(FND_atleast_3_sensor_indv{1} ,1);
    %     TPD_atleast_4_sensor_overall{1}     = sum(TPD_atleast_4_sensor_indv{1} ,1);
    %     FPD_atleast_4_sensor_overall{1}     = sum(FPD_atleast_4_sensor_indv{1} ,1);
    %     TND_atleast_4_sensor_overall{1}     = sum(TND_atleast_4_sensor_indv{1} ,1);
    %     FND_atleast_4_sensor_overall{1}     = sum(FND_atleast_4_sensor_indv{1} ,1);
    %     TPD_atleast_5_sensor_overall{1}     = sum(TPD_atleast_5_sensor_indv{1} ,1);
    %     FPD_atleast_5_sensor_overall{1}     = sum(FPD_atleast_5_sensor_indv{1} ,1);
    %     TND_atleast_5_sensor_overall{1}     = sum(TND_atleast_5_sensor_indv{1} ,1);
    %     FND_atleast_5_sensor_overall{1}     = sum(FND_atleast_5_sensor_indv{1} ,1);
    %     TPD_atleast_6_sensor_overall{1}     = sum(TPD_atleast_6_sensor_indv{1} ,1);
    %     FPD_atleast_6_sensor_overall{1}     = sum(FPD_atleast_6_sensor_indv{1} ,1);
    %     TND_atleast_6_sensor_overall{1}     = sum(TND_atleast_6_sensor_indv{1} ,1);
    %     FND_atleast_6_sensor_overall{1}     = sum(FND_atleast_6_sensor_indv{1} ,1);

    TPD_atleast_1_type_overall{1}       = sum(TPD_atleast_1_type_indv{1} ,1); % Sums up across a column
    FPD_atleast_1_type_overall{1}       = sum(FPD_atleast_1_type_indv{1} ,1);
    TND_atleast_1_type_overall{1}       = sum(TND_atleast_1_type_indv{1} ,1);
    FND_atleast_1_type_overall{1}       = sum(FND_atleast_1_type_indv{1} ,1);
    TPD_atleast_2_type_overall{1}       = sum(TPD_atleast_2_type_indv{1} ,1);
    FPD_atleast_2_type_overall{1}       = sum(FPD_atleast_2_type_indv{1} ,1);
    TND_atleast_2_type_overall{1}       = sum(TND_atleast_2_type_indv{1} ,1);
    FND_atleast_2_type_overall{1}       = sum(FND_atleast_2_type_indv{1} ,1);
    TPD_atleast_3_type_overall{1}       = sum(TPD_atleast_3_type_indv{1} ,1);
    FPD_atleast_3_type_overall{1}       = sum(FPD_atleast_3_type_indv{1} ,1);
    TND_atleast_3_type_overall{1}       = sum(TND_atleast_3_type_indv{1} ,1);
    FND_atleast_3_type_overall{1}       = sum(FND_atleast_3_type_indv{1} ,1);

    % For ROC & PR curves
    Overall_detection_Aclm (r,:)  = [TPD_Aclm_overall{1},FPD_Aclm_overall{1},TND_Aclm_overall{1},FND_Aclm_overall{1}];
    Overall_detection_Acstc (r,:) = [TPD_Acstc_overall{1},FPD_Acstc_overall{1},TND_Acstc_overall{1},FND_Acstc_overall{1}];
    Overall_detection_Pzplt (r,:) = [TPD_Pzplt_overall{1},FPD_Pzplt_overall{1},TND_Pzplt_overall{1},FND_Pzplt_overall{1}];

    Overall_detection_Aclm_OR_Acstc  (r,:) = [TPD_Aclm_OR_Acstc_overall{1},FPD_Aclm_OR_Acstc_overall{1},TND_Aclm_OR_Acstc_overall{1},FND_Aclm_OR_Acstc_overall{1}];
    Overall_detection_Aclm_OR_Pzplt  (r,:) = [TPD_Aclm_OR_Pzplt_overall{1},FPD_Aclm_OR_Pzplt_overall{1},TND_Aclm_OR_Pzplt_overall{1},FND_Aclm_OR_Pzplt_overall{1}];
    Overall_detection_Acstc_OR_Pzplt (r,:) = [TPD_Acstc_OR_Pzplt_overall{1},FPD_Acstc_OR_Pzplt_overall{1},TND_Acstc_OR_Pzplt_overall{1},FND_Acstc_OR_Pzplt_overall{1}];

    Overall_detection_atleast_1_type (r,:) = [TPD_atleast_1_type_overall{1},FPD_atleast_1_type_overall{1},TND_atleast_1_type_overall{1},FND_atleast_1_type_overall{1}];
    Overall_detection_atleast_2_type (r,:) = [TPD_atleast_2_type_overall{1},FPD_atleast_2_type_overall{1},TND_atleast_2_type_overall{1},FND_atleast_2_type_overall{1}];
    Overall_detection_atleast_3_type (r,:) = [TPD_atleast_3_type_overall{1},FPD_atleast_3_type_overall{1},TND_atleast_3_type_overall{1},FND_atleast_3_type_overall{1}];

    FM_min_SN = FM_min_SN + SN_increment; % Updating the FM_min_SN at the end of each iterations
end

fprintf('\nPerformance analysis completed.\n\n')
%

%% =========================== Post-processing ============================

[SEN_all_indv_sensors_overall, PPV_all_indv_sensors_overall, SPE_all_indv_sensors_overall, ACC_all_indv_sensors_overall, FS_all_indv_sensors_overall, FPR_all_indv_sensors_overall] ...
    = get_performance_params(TPD_all_indv_sensors_overall, FPD_all_indv_sensors_overall, TND_all_indv_sensors_overall, FND_all_indv_sensors_overall); % Values for each sensor will be stored in each cell

[SEN_Acstc_overall, PPV_Acstc_overall, SPE_Acstc_overall, ACC_Acstc_overall, FS_Acstc_overall, FPR_Acstc_overall] ...
    = get_performance_params(TPD_Acstc_overall, FPD_Acstc_overall, TND_Acstc_overall, FND_Acstc_overall);
[SEN_Aclm_overall, PPV_Aclm_overall, SPE_Aclm_overall, ACC_Aclm_overall, FS_Aclm_overall, FPR_Aclm_overall] ...
    = get_performance_params(TPD_Aclm_overall, FPD_Aclm_overall, TND_Aclm_overall, FND_Aclm_overall);
[SEN_Pzplt_overall, PPV_Pzplt_overall, SPE_Pzplt_overall, ACC_Pzplt_overall, FS_Pzplt_overall, FPR_Pzplt_overall] ...
    = get_performance_params(TPD_Pzplt_overall, FPD_Pzplt_overall, TND_Pzplt_overall, FND_Pzplt_overall);
[SEN_Aclm_OR_Acstc_overall, PPV_Aclm_OR_Acstc_overall, SPE_Aclm_OR_Acstc_overall, ACC_Aclm_OR_Acstc_overall, FS_Aclm_OR_Acstc_overall, FPR_Aclm_OR_Acstc_overall] ...
    = get_performance_params(TPD_Aclm_OR_Acstc_overall, FPD_Aclm_OR_Acstc_overall, TND_Aclm_OR_Acstc_overall, FND_Aclm_OR_Acstc_overall);
[SEN_Aclm_OR_Pzplt_overall, PPV_Aclm_OR_Pzplt_overall, SPE_Aclm_OR_Pzplt_overall, ACC_Aclm_OR_Pzplt_overall, FS_Aclm_OR_Pzplt_overall, FPR_Aclm_OR_Pzplt_overall] ...
    = get_performance_params(TPD_Aclm_OR_Pzplt_overall, FPD_Aclm_OR_Pzplt_overall, TND_Aclm_OR_Pzplt_overall, FND_Aclm_OR_Pzplt_overall);
[SEN_Acstc_OR_Pzplt_overall, PPV_Acstc_OR_Pzplt_overall, SPE_Acstc_OR_Pzplt_overall, ACC_Acstc_OR_Pzplt_overall, FS_Acstc_OR_Pzplt_overall, FPR_Acstc_OR_Pzplt_overall] ...
    = get_performance_params(TPD_Acstc_OR_Pzplt_overall, FPD_Acstc_OR_Pzplt_overall, TND_Acstc_OR_Pzplt_overall, FND_Acstc_OR_Pzplt_overall);
[SEN_all_sensors_OR_cmbd_overall, PPV_all_sensors_OR_cmbd_overall, SPE_all_sensors_OR_cmbd_overall, ACC_all_sensors_OR_cmbd_overall, FS_all_sensors_OR_cmbd_overall, FPR_all_sensors_OR_cmbd_overall] ...
    = get_performance_params(TPD_all_sensors_OR_cmbd_overall, FPD_all_sensors_OR_cmbd_overall, TND_all_sensors_OR_cmbd_overall, FND_all_sensors_OR_cmbd_overall);

[SEN_Aclm_AND_Acstc_overall, PPV_Aclm_AND_Acstc_overall, SPE_Aclm_AND_Acstc_overall, ACC_Aclm_AND_Acstc_overall, FS_Aclm_AND_Acstc_overall, FPR_Aclm_AND_Acstc_overall] ...
    = get_performance_params(TPD_Aclm_AND_Acstc_overall, FPD_Aclm_AND_Acstc_overall, TND_Aclm_AND_Acstc_overall, FND_Aclm_AND_Acstc_overall);
[SEN_Aclm_AND_Pzplt_overall, PPV_Aclm_AND_Pzplt_overall, SPE_Aclm_AND_Pzplt_overall, ACC_Aclm_AND_Pzplt_overall, FS_Aclm_AND_Pzplt_overall, FPR_Aclm_AND_Pzplt_overall] ...
    = get_performance_params(TPD_Aclm_AND_Pzplt_overall, FPD_Aclm_AND_Pzplt_overall, TND_Aclm_AND_Pzplt_overall, FND_Aclm_AND_Pzplt_overall);
[SEN_Acstc_AND_Pzplt_overall, PPV_Acstc_AND_Pzplt_overall, SPE_Acstc_AND_Pzplt_overall, ACC_Acstc_AND_Pzplt_overall, FS_Acstc_AND_Pzplt_overall, FPR_Acstc_AND_Pzplt_overall] ...
    = get_performance_params(TPD_Acstc_AND_Pzplt_overall, FPD_Acstc_AND_Pzplt_overall, TND_Acstc_AND_Pzplt_overall, FND_Acstc_AND_Pzplt_overall);
[SEN_all_sensors_AND_cmbd_overall, PPV_all_sensors_AND_cmbd_overall, SPE_all_sensors_AND_cmbd_overall, ACC_all_sensors_AND_cmbd_overall, FS_all_sensors_AND_cmbd_overall, FPR_all_sensors_AND_cmbd_overall] ...
    = get_performance_params(TPD_all_sensors_AND_cmbd_overall, FPD_all_sensors_AND_cmbd_overall, TND_all_sensors_AND_cmbd_overall, FND_all_sensors_AND_cmbd_overall);

[SEN_atleast_1_sensor_overall, PPV_atleast_1_sensor_overall, SPE_atleast_1_sensor_overall, ACC_atleast_1_sensor_overall, FS_atleast_1_sensor_overall, FPR_atleast_1_sensor_overall] ...
    = get_performance_params(TPD_atleast_1_sensor_overall, FPD_atleast_1_sensor_overall, TND_atleast_1_sensor_overall, FND_atleast_1_sensor_overall);
[SEN_atleast_2_sensor_overall, PPV_atleast_2_sensor_overall, SPE_atleast_2_sensor_overall, ACC_atleast_2_sensor_overall, FS_atleast_2_sensor_overall, FPR_atleast_2_sensor_overall] ...
    = get_performance_params(TPD_atleast_2_sensor_overall, FPD_atleast_2_sensor_overall, TND_atleast_2_sensor_overall, FND_atleast_2_sensor_overall);
[SEN_atleast_3_sensor_overall, PPV_atleast_3_sensor_overall, SPE_atleast_3_sensor_overall, ACC_atleast_3_sensor_overall, FS_atleast_3_sensor_overall, FPR_atleast_3_sensor_overall] ...
    = get_performance_params(TPD_atleast_3_sensor_overall, FPD_atleast_3_sensor_overall, TND_atleast_3_sensor_overall, FND_atleast_3_sensor_overall);
[SEN_atleast_4_sensor_overall, PPV_atleast_4_sensor_overall, SPE_atleast_4_sensor_overall, ACC_atleast_4_sensor_overall, FS_atleast_4_sensor_overall, FPR_atleast_4_sensor_overall] ...
    = get_performance_params(TPD_atleast_4_sensor_overall, FPD_atleast_4_sensor_overall, TND_atleast_4_sensor_overall, FND_atleast_4_sensor_overall);
[SEN_atleast_5_sensor_overall, PPV_atleast_5_sensor_overall, SPE_atleast_5_sensor_overall, ACC_atleast_5_sensor_overall, FS_atleast_5_sensor_overall, FPR_atleast_5_sensor_overall] ...
    = get_performance_params(TPD_atleast_5_sensor_overall, FPD_atleast_5_sensor_overall, TND_atleast_5_sensor_overall, FND_atleast_5_sensor_overall);
[SEN_atleast_6_sensor_overall, PPV_atleast_6_sensor_overall, SPE_atleast_6_sensor_overall, ACC_atleast_6_sensor_overall, FS_atleast_6_sensor_overall, FPR_atleast_6_sensor_overall] ...
    = get_performance_params(TPD_atleast_6_sensor_overall, FPD_atleast_6_sensor_overall, TND_atleast_6_sensor_overall, FND_atleast_6_sensor_overall);

[SEN_atleast_1_type_overall, PPV_atleast_1_type_overall, SPE_atleast_1_type_overall, ACC_atleast_1_type_overall, FS_atleast_1_type_overall, FPR_atleast_1_type_overall] ...
    = get_performance_params(TPD_atleast_1_type_overall, FPD_atleast_1_type_overall, TND_atleast_1_type_overall, FND_atleast_1_type_overall);
[SEN_atleast_2_type_overall, PPV_atleast_2_type_overall, SPE_atleast_2_type_overall, ACC_atleast_2_type_overall, FS_atleast_2_type_overall, FPR_atleast_2_type_overall] ...
    = get_performance_params(TPD_atleast_2_type_overall, FPD_atleast_2_type_overall, TND_atleast_2_type_overall, FND_atleast_2_type_overall);
[SEN_atleast_3_type_overall, PPV_atleast_3_type_overall, SPE_atleast_3_type_overall, ACC_atleast_3_type_overall, FS_atleast_3_type_overall, FPR_atleast_3_type_overall] ...
    = get_performance_params(TPD_atleast_3_type_overall, FPD_atleast_3_type_overall, TND_atleast_3_type_overall, FND_atleast_3_type_overall);
%

% ====================== Storing data in a table =========================
% % Determination of duration of each data files
% duration_data_files = zeros(n_data_files, 1);
% 
% for i = 1 : n_data_files
%     duration_data_files(i,1) = length(Acstc_data1_fltd{i})/(Fs_sensor*60); % Duration in minutes of the original data file before trimming
% end
% %

% Combining information for differen sensors
sensor_combinations = ["Left Accelerometer","Right Accelerometer","Left Acoustic","Right Acoustic","Left Piezo","Right Piezo","Combined Accelerometers","Combined Acoustics","Combined Piezos",...
    "Accelerometers OR Acoustics", "Accelerometers OR Piezoelectrics", "Acoustics OR Piezoelectrics", "All sensors OR",...
    "Accelerometers AND Acoustics","Accelerometers AND Piezoelectrics","Acoustics AND Piezoelectrics","All sensors AND",...
    "At least 1 sensor",         "At least 2 sensors",           "At least 3 sensors","At least 4 sensors","At least 5 sensors","At least 6 sensors",...
    "At least 1 type of sensor", "At least 2 types of sensor",   "At least 3 types of sensor"];

FM_min_SN_used = "indv sensor type based";

FS_indv_total  = [FS_all_indv_sensors_indv, FS_Aclm_indv, FS_Acstc_indv, FS_Pzplt_indv,...
    FS_Aclm_OR_Acstc_indv,    FS_Aclm_OR_Pzplt_indv,    FS_Acstc_OR_Pzplt_indv,   FS_all_sensors_OR_cmbd_indv, ...
    FS_Aclm_AND_Acstc_indv,   FS_Aclm_AND_Pzplt_indv,   FS_Acstc_AND_Pzplt_indv,  FS_all_sensors_AND_cmbd_indv, ...
    FS_atleast_1_sensor_indv, FS_atleast_2_sensor_indv, FS_atleast_3_sensor_indv, FS_atleast_4_sensor_indv, FS_atleast_5_sensor_indv, FS_atleast_6_sensor_indv, ...
    FS_atleast_1_type_indv,   FS_atleast_2_type_indv,   FS_atleast_3_type_indv];
SEN_indv_total  = [SEN_all_indv_sensors_indv, SEN_Aclm_indv, SEN_Acstc_indv, SEN_Pzplt_indv,...
    SEN_Aclm_OR_Acstc_indv,    SEN_Aclm_OR_Pzplt_indv,    SEN_Acstc_OR_Pzplt_indv,   SEN_all_sensors_OR_cmbd_indv, ...
    SEN_Aclm_AND_Acstc_indv,   SEN_Aclm_AND_Pzplt_indv,   SEN_Acstc_AND_Pzplt_indv,  SEN_all_sensors_AND_cmbd_indv, ...
    SEN_atleast_1_sensor_indv, SEN_atleast_2_sensor_indv, SEN_atleast_3_sensor_indv, SEN_atleast_4_sensor_indv, SEN_atleast_5_sensor_indv, SEN_atleast_6_sensor_indv, ...
    SEN_atleast_1_type_indv,   SEN_atleast_2_type_indv,   SEN_atleast_3_type_indv];
PPV_indv_total  = [PPV_all_indv_sensors_indv, PPV_Aclm_indv, PPV_Acstc_indv, PPV_Pzplt_indv,...
    PPV_Aclm_OR_Acstc_indv,    PPV_Aclm_OR_Pzplt_indv,    PPV_Acstc_OR_Pzplt_indv,   PPV_all_sensors_OR_cmbd_indv, ...
    PPV_Aclm_AND_Acstc_indv,   PPV_Aclm_AND_Pzplt_indv,   PPV_Acstc_AND_Pzplt_indv,  PPV_all_sensors_AND_cmbd_indv, ...
    PPV_atleast_1_sensor_indv, PPV_atleast_2_sensor_indv, PPV_atleast_3_sensor_indv, PPV_atleast_4_sensor_indv, PPV_atleast_5_sensor_indv, PPV_atleast_6_sensor_indv, ...
    PPV_atleast_1_type_indv,   PPV_atleast_2_type_indv,   PPV_atleast_3_type_indv];
SPE_indv_total  = [SPE_all_indv_sensors_indv, SPE_Aclm_indv, SPE_Acstc_indv, SPE_Pzplt_indv,...
    SPE_Aclm_OR_Acstc_indv,    SPE_Aclm_OR_Pzplt_indv,    SPE_Acstc_OR_Pzplt_indv,   SPE_all_sensors_OR_cmbd_indv, ...
    SPE_Aclm_AND_Acstc_indv,   SPE_Aclm_AND_Pzplt_indv,   SPE_Acstc_AND_Pzplt_indv,  SPE_all_sensors_AND_cmbd_indv, ...
    SPE_atleast_1_sensor_indv, SPE_atleast_2_sensor_indv, SPE_atleast_3_sensor_indv, SPE_atleast_4_sensor_indv, SPE_atleast_5_sensor_indv, SPE_atleast_6_sensor_indv, ...
    SPE_atleast_1_type_indv,   SPE_atleast_2_type_indv,   SPE_atleast_3_type_indv];
ACC_indv_total  = [ACC_all_indv_sensors_indv, ACC_Aclm_indv, ACC_Acstc_indv, ACC_Pzplt_indv,...
    ACC_Aclm_OR_Acstc_indv,    ACC_Aclm_OR_Pzplt_indv,    ACC_Acstc_OR_Pzplt_indv,   ACC_all_sensors_OR_cmbd_indv, ...
    ACC_Aclm_AND_Acstc_indv,   ACC_Aclm_AND_Pzplt_indv,   ACC_Acstc_AND_Pzplt_indv,  ACC_all_sensors_AND_cmbd_indv, ...
    ACC_atleast_1_sensor_indv, ACC_atleast_2_sensor_indv, ACC_atleast_3_sensor_indv, ACC_atleast_4_sensor_indv, ACC_atleast_5_sensor_indv, ACC_atleast_6_sensor_indv, ...
    ACC_atleast_1_type_indv,   ACC_atleast_2_type_indv,   ACC_atleast_3_type_indv];
TPD_indv_total  = [TPD_all_indv_sensors_indv, TPD_Aclm_indv, TPD_Acstc_indv, TPD_Pzplt_indv,...
    TPD_Aclm_OR_Acstc_indv,    TPD_Aclm_OR_Pzplt_indv,    TPD_Acstc_OR_Pzplt_indv,   TPD_all_sensors_OR_cmbd_indv, ...
    TPD_Aclm_AND_Acstc_indv,   TPD_Aclm_AND_Pzplt_indv,   TPD_Acstc_AND_Pzplt_indv,  TPD_all_sensors_AND_cmbd_indv, ...
    TPD_atleast_1_sensor_indv, TPD_atleast_2_sensor_indv, TPD_atleast_3_sensor_indv, TPD_atleast_4_sensor_indv, TPD_atleast_5_sensor_indv, TPD_atleast_6_sensor_indv, ...
    TPD_atleast_1_type_indv,   TPD_atleast_2_type_indv,   TPD_atleast_3_type_indv];
FPD_indv_total  = [FPD_all_indv_sensors_indv, FPD_Aclm_indv, FPD_Acstc_indv, FPD_Pzplt_indv,...
    FPD_Aclm_OR_Acstc_indv,    FPD_Aclm_OR_Pzplt_indv,    FPD_Acstc_OR_Pzplt_indv,   FPD_all_sensors_OR_cmbd_indv, ...
    FPD_Aclm_AND_Acstc_indv,   FPD_Aclm_AND_Pzplt_indv,   FPD_Acstc_AND_Pzplt_indv,  FPD_all_sensors_AND_cmbd_indv, ...
    FPD_atleast_1_sensor_indv, FPD_atleast_2_sensor_indv, FPD_atleast_3_sensor_indv, FPD_atleast_4_sensor_indv, FPD_atleast_5_sensor_indv, FPD_atleast_6_sensor_indv, ...
    FPD_atleast_1_type_indv,   FPD_atleast_2_type_indv,   FPD_atleast_3_type_indv];
TND_indv_total  = [TND_all_indv_sensors_indv, TND_Aclm_indv, TND_Acstc_indv, TND_Pzplt_indv,...
    TND_Aclm_OR_Acstc_indv,    TND_Aclm_OR_Pzplt_indv,    TND_Acstc_OR_Pzplt_indv,   TND_all_sensors_OR_cmbd_indv, ...
    TND_Aclm_AND_Acstc_indv,   TND_Aclm_AND_Pzplt_indv,   TND_Acstc_AND_Pzplt_indv,  TND_all_sensors_AND_cmbd_indv, ...
    TND_atleast_1_sensor_indv, TND_atleast_2_sensor_indv, TND_atleast_3_sensor_indv, TND_atleast_4_sensor_indv, TND_atleast_5_sensor_indv, TND_atleast_6_sensor_indv, ...
    TND_atleast_1_type_indv,   TND_atleast_2_type_indv,   TND_atleast_3_type_indv];
FND_indv_total  = [FND_all_indv_sensors_indv, FND_Aclm_indv, FND_Acstc_indv, FND_Pzplt_indv,...
    FND_Aclm_OR_Acstc_indv,    FND_Aclm_OR_Pzplt_indv,    FND_Acstc_OR_Pzplt_indv,   FND_all_sensors_OR_cmbd_indv, ...
    FND_Aclm_AND_Acstc_indv,   FND_Aclm_AND_Pzplt_indv,   FND_Acstc_AND_Pzplt_indv,  FND_all_sensors_AND_cmbd_indv, ...
    FND_atleast_1_sensor_indv, FND_atleast_2_sensor_indv, FND_atleast_3_sensor_indv, FND_atleast_4_sensor_indv, FND_atleast_5_sensor_indv, FND_atleast_6_sensor_indv, ...
    FND_atleast_1_type_indv,   FND_atleast_2_type_indv,   FND_atleast_3_type_indv];

FS_overall_total  = [FS_all_indv_sensors_overall, FS_Aclm_overall, FS_Acstc_overall, FS_Pzplt_overall,...
    FS_Aclm_OR_Acstc_overall,    FS_Aclm_OR_Pzplt_overall,    FS_Acstc_OR_Pzplt_overall,   FS_all_sensors_OR_cmbd_overall, ...
    FS_Aclm_AND_Acstc_overall,   FS_Aclm_AND_Pzplt_overall,   FS_Acstc_AND_Pzplt_overall,  FS_all_sensors_AND_cmbd_overall, ...
    FS_atleast_1_sensor_overall, FS_atleast_2_sensor_overall, FS_atleast_3_sensor_overall, FS_atleast_4_sensor_overall, FS_atleast_5_sensor_overall, FS_atleast_6_sensor_overall, ...
    FS_atleast_1_type_overall,   FS_atleast_2_type_overall,   FS_atleast_3_type_overall];
SEN_overall_total  = [SEN_all_indv_sensors_overall, SEN_Aclm_overall, SEN_Acstc_overall, SEN_Pzplt_overall,...
    SEN_Aclm_OR_Acstc_overall,    SEN_Aclm_OR_Pzplt_overall,    SEN_Acstc_OR_Pzplt_overall,   SEN_all_sensors_OR_cmbd_overall, ...
    SEN_Aclm_AND_Acstc_overall,   SEN_Aclm_AND_Pzplt_overall,   SEN_Acstc_AND_Pzplt_overall,  SEN_all_sensors_AND_cmbd_overall, ...
    SEN_atleast_1_sensor_overall, SEN_atleast_2_sensor_overall, SEN_atleast_3_sensor_overall, SEN_atleast_4_sensor_overall, SEN_atleast_5_sensor_overall, SEN_atleast_6_sensor_overall, ...
    SEN_atleast_1_type_overall,   SEN_atleast_2_type_overall,   SEN_atleast_3_type_overall];
PPV_overall_total  = [PPV_all_indv_sensors_overall, PPV_Aclm_overall, PPV_Acstc_overall, PPV_Pzplt_overall,...
    PPV_Aclm_OR_Acstc_overall,    PPV_Aclm_OR_Pzplt_overall,    PPV_Acstc_OR_Pzplt_overall,   PPV_all_sensors_OR_cmbd_overall, ...
    PPV_Aclm_AND_Acstc_overall,   PPV_Aclm_AND_Pzplt_overall,   PPV_Acstc_AND_Pzplt_overall,  PPV_all_sensors_AND_cmbd_overall, ...
    PPV_atleast_1_sensor_overall, PPV_atleast_2_sensor_overall, PPV_atleast_3_sensor_overall, PPV_atleast_4_sensor_overall, PPV_atleast_5_sensor_overall, PPV_atleast_6_sensor_overall, ...
    PPV_atleast_1_type_overall,   PPV_atleast_2_type_overall,   PPV_atleast_3_type_overall];
SPE_overall_total  = [SPE_all_indv_sensors_overall, SPE_Aclm_overall, SPE_Acstc_overall, SPE_Pzplt_overall,...
    SPE_Aclm_OR_Acstc_overall,    SPE_Aclm_OR_Pzplt_overall,    SPE_Acstc_OR_Pzplt_overall,   SPE_all_sensors_OR_cmbd_overall, ...
    SPE_Aclm_AND_Acstc_overall,   SPE_Aclm_AND_Pzplt_overall,   SPE_Acstc_AND_Pzplt_overall,  SPE_all_sensors_AND_cmbd_overall, ...
    SPE_atleast_1_sensor_overall, SPE_atleast_2_sensor_overall, SPE_atleast_3_sensor_overall, SPE_atleast_4_sensor_overall, SPE_atleast_5_sensor_overall, SPE_atleast_6_sensor_overall, ...
    SPE_atleast_1_type_overall,   SPE_atleast_2_type_overall,   SPE_atleast_3_type_overall];
ACC_overall_total  = [ACC_all_indv_sensors_overall, ACC_Aclm_overall, ACC_Acstc_overall, ACC_Pzplt_overall,...
    ACC_Aclm_OR_Acstc_overall,    ACC_Aclm_OR_Pzplt_overall,    ACC_Acstc_OR_Pzplt_overall,   ACC_all_sensors_OR_cmbd_overall, ...
    ACC_Aclm_AND_Acstc_overall,   ACC_Aclm_AND_Pzplt_overall,   ACC_Acstc_AND_Pzplt_overall,  ACC_all_sensors_AND_cmbd_overall, ...
    ACC_atleast_1_sensor_overall, ACC_atleast_2_sensor_overall, ACC_atleast_3_sensor_overall, ACC_atleast_4_sensor_overall, ACC_atleast_5_sensor_overall, ACC_atleast_6_sensor_overall, ...
    ACC_atleast_1_type_overall,   ACC_atleast_2_type_overall,   ACC_atleast_3_type_overall];
TPD_overall_total  = [TPD_all_indv_sensors_overall, TPD_Aclm_overall, TPD_Acstc_overall, TPD_Pzplt_overall,...
    TPD_Aclm_OR_Acstc_overall,    TPD_Aclm_OR_Pzplt_overall,    TPD_Acstc_OR_Pzplt_overall,   TPD_all_sensors_OR_cmbd_overall, ...
    TPD_Aclm_AND_Acstc_overall,   TPD_Aclm_AND_Pzplt_overall,   TPD_Acstc_AND_Pzplt_overall,  TPD_all_sensors_AND_cmbd_overall, ...
    TPD_atleast_1_sensor_overall, TPD_atleast_2_sensor_overall, TPD_atleast_3_sensor_overall, TPD_atleast_4_sensor_overall, TPD_atleast_5_sensor_overall, TPD_atleast_6_sensor_overall, ...
    TPD_atleast_1_type_overall,   TPD_atleast_2_type_overall,   TPD_atleast_3_type_overall];
FPD_overall_total  = [FPD_all_indv_sensors_overall, FPD_Aclm_overall, FPD_Acstc_overall, FPD_Pzplt_overall,...
    FPD_Aclm_OR_Acstc_overall,    FPD_Aclm_OR_Pzplt_overall,    FPD_Acstc_OR_Pzplt_overall,   FPD_all_sensors_OR_cmbd_overall, ...
    FPD_Aclm_AND_Acstc_overall,   FPD_Aclm_AND_Pzplt_overall,   FPD_Acstc_AND_Pzplt_overall,  FPD_all_sensors_AND_cmbd_overall, ...
    FPD_atleast_1_sensor_overall, FPD_atleast_2_sensor_overall, FPD_atleast_3_sensor_overall, FPD_atleast_4_sensor_overall, FPD_atleast_5_sensor_overall, FPD_atleast_6_sensor_overall, ...
    FPD_atleast_1_type_overall,   FPD_atleast_2_type_overall,   FPD_atleast_3_type_overall];
TND_overall_total  = [TND_all_indv_sensors_overall, TND_Aclm_overall, TND_Acstc_overall, TND_Pzplt_overall,...
    TND_Aclm_OR_Acstc_overall,    TND_Aclm_OR_Pzplt_overall,    TND_Acstc_OR_Pzplt_overall,   TND_all_sensors_OR_cmbd_overall, ...
    TND_Aclm_AND_Acstc_overall,   TND_Aclm_AND_Pzplt_overall,   TND_Acstc_AND_Pzplt_overall,  TND_all_sensors_AND_cmbd_overall, ...
    TND_atleast_1_sensor_overall, TND_atleast_2_sensor_overall, TND_atleast_3_sensor_overall, TND_atleast_4_sensor_overall, TND_atleast_5_sensor_overall, TND_atleast_6_sensor_overall, ...
    TND_atleast_1_type_overall,   TND_atleast_2_type_overall,   TND_atleast_3_type_overall];
FND_overall_total  = [FND_all_indv_sensors_overall, FND_Aclm_overall, FND_Acstc_overall, FND_Pzplt_overall,...
    FND_Aclm_OR_Acstc_overall,    FND_Aclm_OR_Pzplt_overall,    FND_Acstc_OR_Pzplt_overall,   FND_all_sensors_OR_cmbd_overall, ...
    FND_Aclm_AND_Acstc_overall,   FND_Aclm_AND_Pzplt_overall,   FND_Acstc_AND_Pzplt_overall,  FND_all_sensors_AND_cmbd_overall, ...
    FND_atleast_1_sensor_overall, FND_atleast_2_sensor_overall, FND_atleast_3_sensor_overall, FND_atleast_4_sensor_overall, FND_atleast_5_sensor_overall, FND_atleast_6_sensor_overall, ...
    FND_atleast_1_type_overall,   FND_atleast_2_type_overall,   FND_atleast_3_type_overall];

% Initialization of the table
T_sensor_combinations_overall = table('Size',[length(sensor_combinations) 13], 'VariableTypes', {'string', 'double', 'double', 'string', 'double', ...
    'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'}) ; 
T_sensor_combinations_indv = table('Size',[length(sensor_combinations)*n_data_files 14], 'VariableTypes', {'string', 'string', 'double', 'double', ...
    'string', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'}) ; 


% Loop for storing the data
for i = 1 : length(sensor_combinations)
    T_sensor_combinations_overall{i,:} = [sensor_combinations(i), sum(duration_raw_data_files), mean(Force_mean), FM_min_SN_used, FS_overall_total(i), ...
        SEN_overall_total(i), PPV_overall_total(i), SPE_overall_total(i), ACC_overall_total(i), TPD_overall_total(i), FPD_overall_total(i), ...
        TND_overall_total(i), FND_overall_total(i)];
    for j = 1 : n_data_files
        if ischar(data_file_names) % If there is only a single data file
            DFN = data_file_names;
        else
            DFN = cellstr(data_file_names{j}); % Data file names are converted from cell elements to strings
        end

        T_sensor_combinations_indv{(i-1)*n_data_files + j, :} = [sensor_combinations(i), DFN, duration_raw_data_files(j), Force_mean(j), ...
            FM_min_SN_used, FS_indv_total{i}(j), SEN_indv_total{i}(j), PPV_indv_total{i}(j), SPE_indv_total{i}(j), ACC_indv_total{i}(j), ...
            TPD_indv_total{i}(j), FPD_indv_total{i}(j), TND_indv_total{i}(j), FND_indv_total{i}(j)];
    end
end

T_sensor_combinations_overall.Properties.VariableNames = {'Sensor Name','Duration (min)','Mean Force (au)','Threshold multiplier','F Score',...
    'Sensitivity', 'PPV', 'Specificity', 'Accuracy', 'TPD', 'FPD', 'TND', 'FND'}; % Assigns column names to the table
T_sensor_combinations_indv.Properties.VariableNames    = {'Sensor Name','Data file','Duration (min)','Mean Force (au)','Threshold multiplier','F Score',...
    'Sensitivity','PPV','Specificity','Accuracy','TPD','FPD','TND','FND'}; % Assigns column names to the table

% --------------------- Displaying the stored data ---------------------- %
% disp('Performance for Individual data sets: ');
% disp(T_sensor_combinations_indv)
disp('Performance for all the data sets combined: ');
disp(T_sensor_combinations_overall)
%

% Detection stats for the selected sensor combination
detection_stats = [SEN_atleast_3_type_overall, PPV_atleast_3_type_overall, FS_atleast_3_type_overall, SPE_atleast_3_type_overall, ACC_atleast_3_type_overall];
%

%% DETERMINATION OF DETECTION STATS FOR BIOPHYSICAL PROFILING =============

n_movements           = zeros(n_data_files,1);
total_FM_duration     = zeros(n_data_files,1); % This will hold the duration the fetus was moving in each data file
mean_FM_duration      = zeros(n_data_files,1); % Thisi will hold the mean duration of FM in each data file
median_onset_interval = zeros(n_data_files,1); % Thisi will hold the median of the interval between each onset of FM for each data file

FM_dilation_time_new = 2; % Detections within this time (s) will be considered as the same detection

for i = 1:n_data_files

    % Segment the sensor data
    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, ~] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time_new, Fs_sensor); % Segmentation based on new dilation time
    
    % Fuse the left and right sensors of each type using OR
    sensor_data_sgmntd_Aclm  = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
    sensor_data_sgmntd_Acstc = {double(sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};    
    sensor_data_sgmntd_Pzplt = {double(sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    
    % Combining multi-type sensors under a single variable
    sensor_data_sgmntd_cmbd_multi_type_sensors_OR = [sensor_data_sgmntd_Acstc, sensor_data_sgmntd_Aclm, sensor_data_sgmntd_Pzplt];

    % Combining all sensors    
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    % Variable decleration
    sensor_data_sgmntd_atleast_3_type{1}   = zeros(length(sensor_data_sgmntd{1}),1);

    if (n_label) % When there is a detection by the sensor system

        for k = 1 : n_label

            L_min = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1 ); % Sample no. corresponding to start of the label
            L_max = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1, 'last' ); % Sample no. corresponding to end of the label

            indv_detection_map = zeros(length(sensor_data_sgmntd{1}),1); % Need to be initialized before every detection matching
            indv_detection_map(L_min:L_max) = 1; % mapping individual sensation data

            % For detection by at least n type of sensors
            tmp_var = 0; % Variable to hold number of common sensors for each detection
            for j = 1:n_FM_sensors/2
                if (sum(indv_detection_map.*sensor_data_sgmntd_cmbd_multi_type_sensors_OR{j})) % Non-zero value indicates intersection
                    tmp_var = tmp_var + 1;
                end
            end

            if tmp_var == 3
                sensor_data_sgmntd_atleast_3_type{1}(L_min:L_max) = 1;
            end
        end
    end

    detection_map         = sensor_data_sgmntd_atleast_3_type{1}; % This indicates detections by Machine Learning algorithm
    detection_map_labeled = bwlabel(detection_map);
    n_movements (i)       = max(detection_map_labeled);

    detection_only         = detection_map(detection_map == 1); % Keeps only the detected segments
    total_FM_duration(i)   = (length(detection_only)/Fs_sensor-n_movements(i)*FM_dilation_time_new/2)/60; % Dilation length in sides of each detection is removed and coverted in minutes
    mean_FM_duration(i)    = total_FM_duration(i)*60/n_movements(i); % mean duration of FM in each data file in s
    %     total_FM_duration(i)   = (length(detection_only)/Fs_sensor)/60; % Dilation length in sides of each detection is removed and coverted in minutes
    %     mean_FM_duration(i)    = (length(detection_only)/Fs_sensor)/n_movements(i); % mean duration of FM in each data file

    onset_interval = zeros(n_movements(i)-1,1);
    for j = 1:(n_movements(i)-1)
        onset1 = find(detection_map_labeled == j,1); % Sample no. corresponding to start of the label
        onset2 = find(detection_map_labeled == j+1,1); % Sample no. corresponding to start of the next label

        onset_interval(j) = (onset2-onset1)/Fs_sensor; % onset to onset interval in seconds
    end

    median_onset_interval(i) = median(onset_interval); % Gives median onset interval in s
end

total_nonFM_duration = duration_trimmed_data_files - total_FM_duration; % Time fetus was not moving

% Storing all the BPP parameters in a table 
T_BPP_parameters = table('Size',[n_data_files 6],'VariableTypes',{'string','double','double','double','double','double'},...
    'VariableNames',{'Data file name','Total FM duration','Total non-FM duration','Mean FM duration', 'Median onset interval', 'No. of FM'});

for i = 1:n_data_files
    if ischar(data_file_names) % If there is only a single data file
        DFN = convertCharsToStrings(data_file_names);
    else
        DFN = cellstr(data_file_names{i}); % Converts data file names from cell elements to strings
    end
    T_BPP_parameters{i,1} = DFN; % Data file names in the first column
    T_BPP_parameters{i,2} = total_FM_duration(i); % time the fetus was active 
    T_BPP_parameters{i,3} = total_nonFM_duration(i); % time the fetus was inactive
    T_BPP_parameters{i,4} = mean_FM_duration(i); % mean duration of fetus
    T_BPP_parameters{i,5} = median_onset_interval(i); % meadian intervals between the onsets of movements
    T_BPP_parameters{i,6} = n_movements(i);
end
%

%% MEDIAN ONSET INTERVALS FOR COMBINED DATA SETS ========================== 
% Because the median interval calculated above is for individual data sets,
% a different method is necessary to calculate that for a particular week
% for each participants.

n1 = 126;
n2 = 131;
onset_interval_all = 0;

for i = n1 : n2
    % Segment the sensor data
    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, ~] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time_new, Fs_sensor); % Segmentation based on new dilation time

    % Fuse the left and right sensors of each type using OR
    sensor_data_sgmntd_Aclm  = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
    sensor_data_sgmntd_Acstc = {double(sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};
    sensor_data_sgmntd_Pzplt = {double(sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};

    % Combining multi-type sensors under a single variable
    sensor_data_sgmntd_cmbd_multi_type_sensors_OR = [sensor_data_sgmntd_Acstc, sensor_data_sgmntd_Aclm, sensor_data_sgmntd_Pzplt];

    % Combining all sensors
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    % Variable decleration
    sensor_data_sgmntd_atleast_3_type{1}   = zeros(length(sensor_data_sgmntd{1}),1);

    if (n_label) % When there is a detection by the sensor system

        for k = 1 : n_label

            L_min = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1 ); % Sample no. corresponding to start of the label
            L_max = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1, 'last' ); % Sample no. corresponding to end of the label

            indv_detection_map = zeros(length(sensor_data_sgmntd{1}),1); % Need to be initialized before every detection matching
            indv_detection_map(L_min:L_max) = 1; % mapping individual sensation data

            % For detection by at least n type of sensors
            tmp_var = 0; % Variable to hold number of common sensors for each detection
            for j = 1:n_FM_sensors/2
                if (sum(indv_detection_map.*sensor_data_sgmntd_cmbd_multi_type_sensors_OR{j})) % Non-zero value indicates intersection
                    tmp_var = tmp_var + 1;
                end
            end

            if tmp_var == 3
                sensor_data_sgmntd_atleast_3_type{1}(L_min:L_max) = 1;
            end
        end
    end

    detection_map         = sensor_data_sgmntd_atleast_3_type{1}; % This indicates detections by Machine Learning algorithm
    detection_map_labeled = bwlabel(detection_map);
    n_movements (i)       = max(detection_map_labeled);

    onset_interval = zeros(n_movements(i)-1,1);
    for j = 1:(n_movements(i)-1)
        onset1 = find(detection_map_labeled == j,1); % Sample no. corresponding to start of the label
        onset2 = find(detection_map_labeled == j+1,1); % Sample no. corresponding to start of the next label

        onset_interval(j) = (onset2-onset1)/Fs_sensor; % onset to onset interval in seconds
    end

    median_onset_interval(i) = median(onset_interval); % Gives median onset interval in s
    onset_interval_all = [onset_interval_all;onset_interval];

end

onset_interval_all = onset_interval_all(2:end);

median_onset_interval_combined = median(onset_interval_all) % Gives median onset interval in s
%

%% PERFROMANCE PLOTTING ===================================================
% Plot settings
legend_all={'Accelerometer','Acoustic sensor','Piezoelectric diaphragm'};
L_width   = 3; % Width of the box
p_L_width = 4; % Width of the lines in the plot
F_name    = 'Times New Roman';
F_size    = 32; % Font size
Font_size_legend = 22;
Font_size_text   = 14;

SEN_OR = [SEN_Aclm_OR_Acstc_overall{1}, SEN_Aclm_OR_Pzplt_overall{1}, SEN_Acstc_OR_Pzplt_overall{1}, SEN_all_sensors_OR_cmbd_overall{1}];
PPV_OR = [PPV_Aclm_OR_Acstc_overall{1}, PPV_Aclm_OR_Pzplt_overall{1}, PPV_Acstc_OR_Pzplt_overall{1}, PPV_all_sensors_OR_cmbd_overall{1}];
FS_OR  = [FS_Aclm_OR_Acstc_overall{1},  FS_Aclm_OR_Pzplt_overall{1},  FS_Acstc_OR_Pzplt_overall{1},  FS_all_sensors_OR_cmbd_overall{1}];

SEN_AND = [SEN_Aclm_AND_Acstc_overall{1}, SEN_Aclm_AND_Pzplt_overall{1}, SEN_Acstc_AND_Pzplt_overall{1}, SEN_all_sensors_AND_cmbd_overall{1}];
PPV_AND = [PPV_Aclm_AND_Acstc_overall{1}, PPV_Aclm_AND_Pzplt_overall{1}, PPV_Acstc_AND_Pzplt_overall{1}, PPV_all_sensors_AND_cmbd_overall{1}];
FS_AND  = [FS_Aclm_AND_Acstc_overall{1},  FS_Aclm_AND_Pzplt_overall{1},  FS_Acstc_AND_Pzplt_overall{1},  FS_all_sensors_AND_cmbd_overall{1}];

SEN_atleast = [SEN_atleast_1_type_overall{1}, SEN_atleast_2_type_overall{1}, SEN_atleast_3_type_overall{1}];
PPV_atleast = [PPV_atleast_1_type_overall{1}, PPV_atleast_2_type_overall{1}, PPV_atleast_3_type_overall{1}];
FS_atleast  = [FS_atleast_1_type_overall{1}, FS_atleast_2_type_overall{1}, FS_atleast_3_type_overall{1}];

tiledlayout(3,3,'Padding','compact','TileSpacing','compact'); % 3x3 tile in most compact fitting

x = categorical({'SEN', 'PPV', 'F1 score'});
x = reordercats(x,{'SEN', 'PPV', 'F1 score'}); % To maintain the order in the plot

nexttile
b = bar(x, [SEN_OR; PPV_OR; FS_OR]);
for j = 1:4
    xtips1 = b(j).XEndPoints;
    ytips1 = b(j).YEndPoints;
    labels1 = [sprintf('%.2f', b(j).YData(1)); sprintf('%.2f', b(j).YData(2)); sprintf('%.2f', b(j).YData(3))];
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', Font_size_text);
end
yticks(0:0.2:1);
xlim(categorical({'SEN', 'F1 score'}))
set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', L_width) % Sets the font type and size of the labels
lgd = legend({'Accelerometers OR Acoustics',  'Accelerometers OR Piezoelectrics', 'Acoustics OR Piezoelectrics',  'All sensors OR'});
lgd.FontSize = Font_size_legend;
lgd.Location = 'northoutside';
lgd.NumColumns = 1;
legend boxoff

nexttile
b = bar(x, [SEN_AND; PPV_AND; FS_AND]);
for j = 1:4
    xtips1 = b(j).XEndPoints;
    ytips1 = b(j).YEndPoints;
    labels1 = [sprintf('%.2f', b(j).YData(1)); sprintf('%.2f', b(j).YData(2)); sprintf('%.2f', b(j).YData(3))];
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', Font_size_text);
end
yticks(0:0.2:1);
xlim(categorical({'SEN', 'F1 score'}))
set(gca, 'YTickLabel', [], 'FontName', F_name, 'FontSize', F_size, 'linewidth', L_width) % Sets the font type and size of the labels
lgd = legend({'Accelerometers AND Acoustics',  'Accelerometers AND Piezoelectrics', 'Acoustics AND Piezoelectrics',  'All sensors AND'});
lgd.FontSize = Font_size_legend;
lgd.Location = 'northoutside';
lgd.NumColumns = 1;
legend boxoff

nexttile
b = bar(x, [SEN_atleast; PPV_atleast; FS_atleast]);
for j = 1:3
    xtips1 = b(j).XEndPoints;
    ytips1 = b(j).YEndPoints;
    labels1 = [sprintf('%.2f', b(j).YData(1)); sprintf('%.2f', b(j).YData(2)); sprintf('%.2f', b(j).YData(3))];
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', Font_size_text);
end
yticks(0:0.2:1);
xlim(categorical({'SEN', 'F1 score'}))
set(gca, 'YTickLabel', [], 'FontName', F_name, 'FontSize', F_size, 'linewidth', L_width) % Sets the font type and size of the labels
lgd = legend({'At least 1 type',  'At least 2 type', 'At least 3 type'});
lgd.FontSize = Font_size_legend;
lgd.Location = 'northoutside';
lgd.NumColumns = 1;
legend boxoff

%% EXTRACTION OF DETECTIONS FOR T-F ANALYSIS ==============================
% TPDs and FPDs that are common to all the sensors are extracted for T-F
% analysis

% Starting notification
disp('T-F analysis is going on... ')

% Parameters and variables for segmentation and detection matching
ext_backward = 5.0; % Backward extension length in second
ext_forward = 2.0; % Forward extension length in second
FM_dilation_time = 3.0;% Dilation size in seconds
n_FM_sensors = 6; % number of FM sensors
FM_min_SN = [30, 30, 60, 60, 50, 50];

IMU_map = cell(1,n_data_files);   
M_sntn_map = cell(1,n_data_files);  
threshold = zeros(n_data_files, n_FM_sensors);

% Parameters and variables for PSD estimation
w = hann(Fs_sensor); % Hann window of length 1s
n_overlap = Fs_sensor/2; % 50% overlap between the segments
p_overlap = 0.5; % 50% overlap between the segments
nfft = Fs_sensor; % number of fourier points will be same as sampling freq

Pxx_TPD_avg = zeros(Fs_sensor/2 + 1, n_FM_sensors);
Pxx_detrended_TPD_avg = zeros(Fs_sensor/2 + 1, n_FM_sensors);
Pxx_FPD_avg = zeros(Fs_sensor/2 + 1, n_FM_sensors);
Pxx_detrended_FPD_avg = zeros(Fs_sensor/2 + 1, n_FM_sensors);
n_TPD_cycle = 0;
n_FPD_cycle = 0;

TPD_extracted = cell(1, n_data_files);
FPD_extracted = cell(1, n_data_files);
 
for i = 1 : n_data_files

    % Starting notification
    fprintf('Current data file: %.0f/%.0f\n', i,n_data_files)

    % --------- Segmentaiton of IMU data and creation of IMU_map ---------%
    % get_IMU_map() function is used here. It segments and dilates the IMU 
    % data and returns the resultant data as IMU map. Settings for the
    % segmentation and dilation are given inside the function.
    %   Input variables:  IMU_data- A column vector of IMU data 
    %                     data_file_names- a char variable with data file name
    %   Output variables: IMU_map- A column vector with segmented IMU data 

    if n_data_files == 1
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i}, data_file_names, Fs_sensor);
    else
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i}, data_file_names{i}, Fs_sensor);
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

    M_sntn_map{i} = get_sensation_map(sensation_data_SD1_trimd{i}, IMU_map{i}, ext_backward, ext_forward, Fs_sensor, Fs_sensation);

    % ~~~~~~~~~~~~~~~~~~~~~~ Segmentation of FM data ~~~~~~~~~~~~~~~~~~~~~%
    % get_segmented_data() function will be used here, which will
    % threshold the data, remove body movement, and dilate the data.
    % Setting for the threshold and dilation are given in the function.
    %   Input variables:  sensor_data- a cell variable;
    %                     min_SN- a vector/ a scalar;
    %                     IMU_map- a column vector
    %                     Fs_sensor, FM_dilation_time- a scalar
    %   Output variables: sensor_data_sgmntd- a cell variable of same size as the input variable sensor_data_fltd;
    %                     h- a vector

    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, threshold(i,:)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);

    % --------------- Detections common to all the sensors ----------------
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    % Initialization of variables
    sensor_data_sgmntd_atleast_6_sensor{1} = zeros(length(sensor_data_sgmntd{1}),1);

    if (n_label) % When there is a detection by the sensor system

        for k = 1 : n_label
            L_min = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1 ); % Sample no. corresponding to start of the label
            L_max = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1, 'last' ); % Sample no. corresponding to end of the label

            indv_detection_map = zeros(length(sensor_data_sgmntd{1}),1); % Need to be initialized before every detection matching
            indv_detection_map(L_min:L_max) = 1; % mapping individual sensation data

            % For detection by at least n sensors
            tmp_var = 0; % Variable to hold number of common sensors for each detection
            for j = 1:n_FM_sensors
                if (sum(indv_detection_map.*sensor_data_sgmntd{j})) % Non-zero value indicates intersection
                    tmp_var = tmp_var + 1;
                end
            end

            switch tmp_var
                case 6
                    sensor_data_sgmntd_atleast_6_sensor{1}(L_min:L_max) = 1;
            end
        end
    end

    % ------------------- Extraction of common TPDs and FPDs---------------
    sensor_data_sgmntd_atleast_6_sensor_labeled = bwlabel(sensor_data_sgmntd_atleast_6_sensor{1});
    n_detection = length(unique(sensor_data_sgmntd_atleast_6_sensor_labeled)) - 1; % Number of total detections 
    n_candidate_TPD = length(unique(sensor_data_sgmntd_atleast_6_sensor_labeled.*M_sntn_map{i})) - 1; % Number of detections that intersects with maternal sensation
    n_candidate_FPD =  n_detection - n_candidate_TPD;

    current_file_TPD_extraction_cell = cell(1, n_candidate_TPD); % Each cell will contain 1 TPD for all 6 sensors
    current_file_FPD_extraction_cell = cell(1, n_candidate_FPD);

    if n_detection
        k_TPD = 0;
        k_FPD = 0;
        for k = 1:n_detection
            L_min = find(sensor_data_sgmntd_atleast_6_sensor_labeled == k, 1 ); % Sample no. corresponding to the start of the label
            L_max = find(sensor_data_sgmntd_atleast_6_sensor_labeled == k, 1, 'last' ); % Sample no. corresponding to the end of the label
            indv_window = zeros(length(M_sntn_map{i}),1); % This window represents the current detection, which will be compared with M_sntn_map to find if it is TPD or FPD
            indv_window(L_min:L_max) = 1;
            X = sum(indv_window.*M_sntn_map{i}); % Checks the overlap with the maternal sensation

            if X
                k_TPD = k_TPD + 1;
                current_TPD_extraction = zeros(L_max-L_min+1, n_FM_sensors);
                for j = 1:n_FM_sensors
                    current_TPD_extraction(:,j) = sensor_data_fltd{j}(L_min:L_max); % Each 

                    [Pxx_TPD, f_TPD] = pwelch(current_TPD_extraction(:,j), w, n_overlap, nfft, Fs_sensor);
                    [Pxx_detrended_TPD, f_detrended_TPD] = pwelch_new(current_TPD_extraction(:,j), w, p_overlap, nfft, Fs_sensor, 'half', 'plot', 'mean');
                    % pwelch_new() is a user defined function that allows detrending.
                    %   Input conditions: half - one sided PSD; plot- normal plotting; 
                    %                     mean-  remove the mean value of each segment from each segment of the data.

                    Pxx_TPD_avg(:,j) = Pxx_TPD_avg(:,j) + Pxx_TPD; % Summing the PSD for each TPD
                    Pxx_detrended_TPD_avg(:,j) = Pxx_detrended_TPD_avg(:,j) + Pxx_detrended_TPD;
                end
                current_file_TPD_extraction_cell {k_TPD} = current_TPD_extraction;
                n_TPD_cycle = n_TPD_cycle + 1;
            else
                k_FPD = k_FPD + 1;
                current_FPD_extraction = zeros(L_max-L_min+1, n_FM_sensors);
                for j = 1:n_FM_sensors
                    current_FPD_extraction(:,j) = sensor_data_fltd{j}(L_min:L_max);

                    [Pxx_FPD, f_FPD] = pwelch(current_FPD_extraction(:,j), w, n_overlap, nfft, Fs_sensor);
                    [Pxx_detrended_FPD, f_detrended_FPD] = pwelch_new(current_FPD_extraction(:,j), w, p_overlap, nfft, Fs_sensor, 'half', 'plot', 'mean');

                    Pxx_FPD_avg(:,j) = Pxx_FPD_avg(:,j) + Pxx_FPD;
                    Pxx_detrended_FPD_avg(:,j) = Pxx_detrended_FPD_avg(:,j) + Pxx_detrended_FPD;
                end
                current_file_FPD_extraction_cell {k_FPD} = current_FPD_extraction;
                n_FPD_cycle = n_FPD_cycle + 1;
            end
        end
    end

    TPD_extracted{i} = current_file_TPD_extraction_cell; % Each cell will contain wll the extracted TPD in a particular data set
    FPD_extracted{i} = current_file_FPD_extraction_cell;
end

% Averaging the PSD summation
Pxx_TPD_avg = Pxx_TPD_avg/n_TPD_cycle; 
Pxx_detrended_TPD_avg =  Pxx_detrended_TPD_avg/n_TPD_cycle;
Pxx_FPD_avg = Pxx_FPD_avg/n_FPD_cycle;
Pxx_detrended_FPD_avg =  Pxx_detrended_FPD_avg/n_FPD_cycle;

% Plotting the PSD
x_lim = 100;
subplot(2,1,1)
for i = 1:n_FM_sensors    
    plot(f_detrended_TPD, Pxx_detrended_TPD_avg(:,i))
    xlim([0 x_lim])
    hold on;
    plot(f_TPD, Pxx_TPD_avg(:,i))
end
hold off;

subplot(2,1,2)
for i = 1:n_FM_sensors    
    plot(f_detrended_FPD, Pxx_detrended_FPD_avg(:,i))
    xlim([0 x_lim])
    hold on;
    plot(f_FPD, Pxx_FPD_avg(:,i))
end
hold off;
%

%% SPECTROGRAM ============================================================

% Parameters and variables for PSD estimation
s_size = Fs_sensor/2; % Size of each segment in the STFT
w = hann(s_size); % Hann window of length 0.5 S
n_overlap = floor(s_size/1.25); % 80% overlap between the segments
nfft = s_size*2; % number of fourier points will be twice sampling freq. This pads 0 at the end to increase the frequency resolution
% For the above setting, the obtained time resolution = 100 ms, and frequency resolution = 1 Hz 

% Selection of data file and TPD
data_file_no = 5;
TPD_no = 7;
t_start = 2; % Starting time in s
t_end = 7; % Ending time in s

% Good candidates: (File_no, TPD_no): (3,3), (4,2), (4,3) , (5,2), (5,4), (5,5)
% excellent,  (5,6, excellent^2), (5,7, excellent^3), (5,11, too good),
% (6,6, too good), (7,2, super)
% Finally used: (5,7)

% Plot settings
legend_all={'Accelerometer','Acoustic sensor','Piezoelectric diaphragm'};
L_width = 3; % Width of the box
p_L_width = 4; % Width of the lines in the plot
F_size = 28; % Font size
F_name = 'Times New Roman';

tiledlayout(3,3,'Padding','tight','TileSpacing','tight'); % 3x3 tile in most compact fitting

% Plot the data in the first row
for j = 1:3
    % Get the X and Y-axis data
    data = TPD_extracted{1, data_file_no}{1, TPD_no}(:, 2*j); % TO get the data for the right sensor only
    % data = data(t_start*Fs_sensor:t_end*Fs_sensor+1,1); % uncomment if you want to plot within a range defined by t_start:t_end
    time_vec = (0:length(data)-1)/Fs_sensor;

    % Plot the data
    nexttile
    plot(time_vec, data, 'LineWidth', p_L_width)    

    % Set the axis properties
    ax = gca; % To get the coordinate axis
    if j == 1
        ylabel('Amplitude (a.u.)'); % Ylabel is only applied to the leftmost axis
        ax.YAxis.Exponent = -2;
        ylim([-0.02 0.01])
    end
    if j == 2
        ax.YAxis.Exponent = -2;
        ylim([-0.06 0.06])
        yticks(-0.06:0.03:0.06)
    end
    if j == 3
        ax.YAxis.Exponent = -1;
        ylim([-0.3 0.3])
        yticks(-0.3:0.1:0.3)
    end

    xlabel('Time(s)');
    xlim([0 8])
    xticks(0:2:10);
    set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', L_width) % Sets the font type and size of the labels
end

% Plot the spectrogram in the seond row
for j=1:3   
    % Get the X and Y-axis data
    data = TPD_extracted{1,data_file_no}{1,TPD_no}(:,2*j);
    % data = data(t_start*Fs_sensor:t_end*Fs_sensor+1,1); % Uncomment if you want to plot within a range defined by t_start:t_end
    [~,f,t,P] = spectrogram(data, w, n_overlap, nfft, Fs_sensor, 'psd', 'yaxis'); % P is the PSD 

    % Plot the data
    nexttile
    % spectrogram(data,w_spec,n_overlap_spec,nfft_spec,Fs_sensor,'yaxis') % Plot using spectrogram
    imagesc(t, f, (P+eps)) % Plot using imagesc. 
    %      Add eps like pspectrogram does: eps is the distance between 1.0 and the next largest double precision number (eps = 2.2204e-16)
    axis xy % Corrects the Y-axis order: By default imagesc() puts Y-axis to high-to-low order
    h = colorbar; % To display the colorbar on the left    

    % Set the axis properties
    if j == 3
        h.Label.String = 'PSD (a.u.^2/Hz)'; % Labels the colorbar
        h.Ticks = 3*10^-3:3*10^-3:12*10^-3  ; % To manage colorbar ticks
    end
    colormap(jet)
    ylim([0,30]);
    yticks(0:10:30);
    if j==1
        ylabel('Frequency (Hz)')
    end
    xlim([-inf 8])
    xlabel('Time (s)'); 
    set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', L_width) % Sets the font type & size, and the line width
end

% Plot the PSD in the third row
for j=1:3
    % Plot the data
    nexttile
    plot(f_detrended_FPD, Pxx_detrended_FPD_avg(:,2*j),'LineWidth', p_L_width)

    % Set the axis properties
    if j == 1
        ylabel('PSD(a.u.^2/Hz)')
        ylim([0 8*10^-7])
        yticks(0:2*10^-7:8*10^-7)
    end
    if j == 2
        ylim([0 3*10^-6])
        yticks(0:1*10^-6:3*10^-6)
    end
    if j == 3
        ylim([0 8*10^-5])
        yticks(0:2*10^-5:8*10^-5)
    end

    xlim([0 30])
    xticks(0:10:30)
    xlabel('Frequency (Hz)')
    set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', L_width) % Sets the font type & size, and the line width
end
%

%% SAVING TPD AND FPD DATA AS TEST FILES ==================================
data_file_no = 1;
TPD_no = 1;
data = TPD_extracted{1,data_file_no}{TPD_no,1};
save('TPDfile.txt','data','-ascii')

%% EXTRACTION OF DATA FOR MACHINE LEARNING ================================
% TPDs and FPDs that are detected by any sensor are extracted for machine
% learning

% Starting notification
disp('Data extraction for machine learning is going on... ')

% Parameters and variables for segmentation and detection matching
ext_backward = 5.0; % Backward extension length in second
ext_forward = 2.0; % Forward extension length in second
FM_dilation_time = 3.0; % Dilation size in seconds
n_FM_sensors = 6; % Number of FM sensors
FM_min_SN = [30, 30, 40, 40, 40, 40]; % These values are selected to get SEN of 95%

IMU_map = cell(1,n_data_files);   
M_sntn_map = cell(1,n_data_files);  
threshold = zeros(n_data_files, n_FM_sensors);

TPD_extracted = cell(1, n_data_files);
FPD_extracted = cell(1, n_data_files);
 
for i = 1 : n_data_files

    % Starting notification
    fprintf('Current data file: %.0f/%.0f\n', i,n_data_files)

    % --------- Segmentaiton of IMU data and creation of IMU_map ---------%
    % get_IMU_map() function is used here. It segments and dilates the IMU 
    % data and returns the resultant data as IMU map. Settings for the
    % segmentation and dilation are given inside the function.
    %   Input variables:  IMU_data- A column vector of IMU data 
    %                     data_file_names- a char variable with data file name
    %   Output variables: IMU_map- A column vector with segmented IMU data 

    if n_data_files == 1
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i}, data_file_names, Fs_sensor);
    else
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i}, data_file_names{i}, Fs_sensor);
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

    M_sntn_map{i} = get_sensation_map(sensation_data_SD1_trimd{i}, IMU_map{i}, ext_backward, ext_forward, Fs_sensor, Fs_sensation);

    % ~~~~~~~~~~~~~~~~~~~~~~ Segmentation of FM data ~~~~~~~~~~~~~~~~~~~~~%
    % get_segmented_data() function will be used here, which will
    % threshold the data, remove body movement, and dilate the data.
    % Setting for the threshold and dilation are given in the function.
    %   Input variables:  sensor_data- a cell variable;
    %                     min_SN- a vector/ a scalar;
    %                     IMU_map- a column vector
    %                     Fs_sensor, FM_dilation_time- a scalar
    %   Output variables: sensor_data_sgmntd- a cell variable of same size as the input variable sensor_data_fltd;
    %                     h- a vector

    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, threshold(i,:)] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);

    % ------------------ Detections from all the sensors ------------------
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    % ------------------- Extraction of TPDs and FPDs ---------------------
    n_candidate_TPD = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled.*M_sntn_map{i})) - 1; % Number of detections that intersects with maternal sensation
    n_candidate_FPD =  n_label - n_candidate_TPD;

    current_file_TPD_extraction_cell = cell(1, n_candidate_TPD); % Each cell will contain 1 TPD for all 6 sensors
    current_file_FPD_extraction_cell = cell(1, n_candidate_FPD);

    if (n_label) % When there is a detection by the sensor system
        k_TPD = 0;
        k_FPD = 0;
        for k = 1:n_label
            L_min = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1 ); % Sample no. corresponding to the start of the label
            L_max = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1, 'last' ); % Sample no. corresponding to the end of the label
            indv_window = zeros(length(M_sntn_map{i}),1); % This window represents the current detection, which will be compared with M_sntn_map to find if it is TPD or FPD
            indv_window(L_min:L_max) = 1;
            X = sum(indv_window.*M_sntn_map{i}); % Checks the overlap with the maternal sensation

            if X
                k_TPD = k_TPD + 1;
                current_TPD_extraction = zeros(L_max-L_min+1, n_FM_sensors);
                for j = 1:n_FM_sensors
                    current_TPD_extraction(:,j) = sensor_data_fltd{j}(L_min:L_max); % Each sensor in each column
                end

                current_file_TPD_extraction_cell {k_TPD} = current_TPD_extraction;
            else
                k_FPD = k_FPD + 1;
                current_FPD_extraction = zeros(L_max-L_min+1, n_FM_sensors);
                for j = 1:n_FM_sensors
                    current_FPD_extraction(:,j) = sensor_data_fltd{j}(L_min:L_max);
                end
                current_file_FPD_extraction_cell {k_FPD} = current_FPD_extraction;
            end
        end
    end

    TPD_extracted{i} = current_file_TPD_extraction_cell; % Each cell will contain wll the extracted TPD in a particular data set
    FPD_extracted{i} = current_file_FPD_extraction_cell;
end

%% SAVING THE EXTRACTED DATA ==============================================
fileName = "TPD_FPD.mat";
save(fileName, 'TPD_extracted', 'FPD_extracted', 'threshold')

