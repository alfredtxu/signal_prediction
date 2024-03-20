T

%% PERFORMANCE ANALYSIS ===================================================
fprintf('\nPerformance analysis of the algorithm is going on...\n');

% Select the option to perform ROC & PR analysis
ROC_curve_option = 0; % 1: generates ROC & PR curves, 0: doesn't generate ROC & PR curves
if ROC_curve_option == 1
    % Define parameter for ROC and PR curve generation
    n_division_thd = 5;
    thd = linspace(0.95,1,n_division_thd+1);
    thd = thd (2:end-1); % removes the first and last elements as they are 0 and 1, respectively
    n_ROC_iter = n_division_thd-1;
else
    thd = 0.5;
    n_ROC_iter = 1;
end
%
overall_detections = zeros(n_ROC_iter,4);
%
for k = 1:n_ROC_iter

    fprintf('\nIteration for ROC: %.0f/%.0f\n',k,n_ROC_iter)

    % ------------ Getting the prediction for the overall data set -----------%
    load_prediction_option = 1; % if 1 load the prediction from the current directory

    if load_prediction_option == 1
        prediction_overall_dataset_TPD_score = load('Y_TPD_from_python.txt');
        prediction_overall_dataset_FPD_score = load('Y_FPD_from_python.txt');

        prediction_overall_dataset_TPD = prediction_overall_dataset_TPD_score >= thd(k);
        prediction_overall_dataset_FPD = prediction_overall_dataset_FPD_score >= thd(k);

        prediction_overall_dataset = [prediction_overall_dataset_TPD; prediction_overall_dataset_FPD];

    else
        for j = 1:length(final_test_prediction)
            % Convert test scores to classification probabilities
            if classifier_option == 2
                test_scores = sigmoid (final_test_scores{j});
            else
                test_scores = final_test_scores{j};
            end
            
            % Calculate the final test prediction
            final_test_prediction{j} = test_scores(:,2) >= thd(k);
        end

        % Get the prediction of overall data set based on data_div_option
        if data_div_option == 1 % Based on holdout method
            Z = X_norm; % The trained classifier is applied on the whole data set
            if classifier_option == 1 % Logistic regression
                if dim_reduction_option == 1
                    Z = projectData(X_norm, U{I_max_LR}, n_PC(I_max_LR)); % Applying PCA on the whole data set
                end
                prediction_overall_dataset = predict(LR_model_selected{I_max_LR}, Z);

            elseif classifier_option == 2 % SVM
                if dim_reduction_option == 1
                    Z = projectData(X_norm, U{I_max_SVM}, n_PC(I_max_SVM)); % Applying PCA on the whole data set
                end
                prediction_overall_dataset = predict(SVM_model_selected{I_max_SVM}, Z);

            elseif classifier_option == 3 % NN
                if dim_reduction_option == 1
                    Z = projectData(X_norm, U{I_max_NN}, n_PC(I_max_NN)); % Applying PCA on the whole data set
                end
                prediction_overall_dataset = predict(NN_model_selected{I_max_NN}, Z);

            elseif classifier_option == 4 % RF
                if dim_reduction_option == 1
                    Z = projectData(X_norm, U{I_max_RF}, n_PC(I_max_RF)); % Applying PCA on the whole data set
                end
                prediction_overall_dataset = predict(RF_model_selected{I_max_RF}, Z);
            end
            prediction_overall_dataset_TPD = prediction_overall_dataset(1:n_TPD);
            prediction_overall_dataset_FPD = prediction_overall_dataset(n_TPD+1:end);

        elseif (data_div_option == 2)||(data_div_option == 3) % Based on stratified K-fold division
            [prediction_overall_dataset_TPD,prediction_overall_dataset_FPD] = ...
                get_overall_test_prediction(final_test_prediction,n_TPD,n_data_TPD_each_fold,...
                n_data_TPD_last_fold,n_FPD,n_data_FPD_each_fold,n_data_FPD_last_fold,...
                n_iter,rand_num_TPD,rand_num_FPD);

            prediction_overall_dataset = [prediction_overall_dataset_TPD;prediction_overall_dataset_FPD];

        elseif data_div_option == 4 % Based on data division by participants
            prediction_overall_dataset_TPD = zeros(1,1); % Initialization
            prediction_overall_dataset_FPD = zeros(1,1);
            for i = 1:n_participant
                n_TPD_current = size(X_TPD_by_participant{i},1);
                n_FPD_current = size(X_FPD_by_participant{i},1);

                prediction_overall_dataset_TPD = [prediction_overall_dataset_TPD;final_test_prediction{i}(1:n_TPD_current)];
                prediction_overall_dataset_FPD = [prediction_overall_dataset_FPD;final_test_prediction{i}(n_TPD_current+1:end)];
            end

            prediction_overall_dataset_TPD = prediction_overall_dataset_TPD(2:end); % Removing the initialized 1st row
            prediction_overall_dataset_FPD = prediction_overall_dataset_FPD(2:end);

            prediction_overall_dataset     = [prediction_overall_dataset_TPD;prediction_overall_dataset_FPD];

        else
            fprintf('\nWrong data division option chosen. \n');
            return
        end
    end

    % Accuracy of for the overall prediction
    Accuracy_overall_dataset     = sum((prediction_overall_dataset == Y))/length(Y); % Y will be same as we are using non-randomized data
    Accuracy_overall_dataset_TPD = sum((prediction_overall_dataset_TPD == Y(1:n_TPD)))/n_TPD;
    Accuracy_overall_dataset_FPD = sum((prediction_overall_dataset_FPD == Y(n_TPD+1:end)))/n_FPD;

    % Getting the detection stats
    matching_index_TPD = 1;
    matching_index_FPD = 1;
    sensor_data_sgmntd_cmbd_all_sensors_ML = cell(1, n_data_files);

    for i = 1 : n_data_files

        % Starting notification
        fprintf('\tCurrent data file: %.0f/%.0f\n',i,n_data_files)

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

        sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
        n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled
        %

        % ------------------ Mapping the detection by ML ------------------
        [sensor_data_sgmntd_cmbd_all_sensors_ML{i},matching_index_TPD,matching_index_FPD]...
            = map_ML_detections(sensor_data_sgmntd_cmbd_all_sensors_labeled,M_sntn_map{i},n_label,...
            prediction_overall_dataset_TPD,prediction_overall_dataset_FPD,matching_index_TPD,matching_index_FPD);
        %

        % ~~~~~~~~~~~~~~~~~ Matching with maternal sensation ~~~~~~~~~~~~~~~~~%
        %   match_with_m_sensation() function will be used here-
        %   Input variables:  sensor_data_sgmntd- A cell variable with single cell/multiple cells.
        %                                         Each cell contains data from a sensor or a combination.
        %                     sensation_data, IMU_map, M_sntn_Map- cell variables with single cell
        %                     ext_bakward, ext_forward, FM_dilation_time- scalar values
        %                     Fs_sensor, Fs_sensation- scalar values
        %   Output variables: TPD, FPD, TND, FND- vectors with number of rows equal to the
        %

        % For combined data
        %   input argument 'sensor_data_sgmntd' will be a single-cell variable.
        %   Hence the function will return a scalar value for each output argument.
        %   Cell variable is used to store the output data to make it
        %   compatible with the performance analysis section.

        current_ML_detection_map{1} = sensor_data_sgmntd_cmbd_all_sensors_ML{i};
        [TPD_indv{1}(i,1), FPD_indv{1}(i,1), TND_indv{1}(i,1), FND_indv{1}(i,1)] ...
            = match_with_m_sensation(current_ML_detection_map, sensation_data_SD1_trimd{i}, ...
            IMU_map{i}, M_sntn_map{i}, ext_backward, ext_forward, FM_dilation_time, Fs_sensor, Fs_sensation);
    end

    % ----------------------- Performance analysis ---------------------------%
    % This section will use get_performance_params() function
    %   Input variables:  TPD_all, FPD_all, TND_all, FND_all- single cell/multi-cell variable.
    %                     Number of cell indicates number of sensor data or
    %                     combination data provided together.
    %                     Each cell containes a vector with no. of rows = n_data_files
    %
    %   Output variables: SEN_all,PPV_all,SPE_all,ACC_all,FS_all,FPR_all- cell variable with
    %                     size same as the input variables.

    % For individual data sets
    [SEN_indv, PPV_indv, SPE_indv, ACC_indv, FS_indv, FPR_indv] = get_performance_params(TPD_indv, FPD_indv, TND_indv, FND_indv);
    % all the returned variables are 1x1 cell arrays

    indv_detections = [TPD_indv{1,1},FPD_indv{1,1},TND_indv{1,1},FND_indv{1,1}]; % Combines all the detections in a matrix

    % For the overall data sets
    TPD_overall{1}  = sum(TPD_indv{1},1);
    FPD_overall{1}  = sum(FPD_indv{1},1);
    TND_overall{1}  = sum(TND_indv{1},1);
    FND_overall{1}  = sum(FND_indv{1},1);

    overall_detections(k,:) = [TPD_overall{1,1},FPD_overall{1,1},TND_overall{1,1},FND_overall{1,1}];

    [SEN_overall, PPV_overall, SPE_overall, ACC_overall, FS_overall, FPR_overall] ...
        = get_performance_params(TPD_overall, FPD_overall, TND_overall, FND_overall);
    PABAK_overall = 2*ACC_overall{1}-1;
    detection_stats = [SEN_overall{1}, PPV_overall{1}, FS_overall, ...
        SPE_overall{1}, ACC_overall{1}, PABAK_overall];
end

fprintf('\nPerformance analysis is completed.\n')
%

% ------------------- Displaying performance matrics ----------------------
fprintf('\nSettings for the algorithm were: ');
fprintf(['\n\tThreshold multiplier:\n\t\tAccelerometer = %.0f, Acoustic = %.0f,' ...
    '\n\t\tPiezoelectric diaphragm = %.0f.'], FM_min_SN(1), FM_min_SN(3), FM_min_SN(5));
% fprintf(['\n\tData division option for training and testing: %g \n\t\t(1- holdout; ' ...
%     '2,3- K-fold; 4-by participants)'], data_div_option);
% fprintf('\n\tFPD/TPD ratio in the training data set: %.2f', FPD_TPD_ratio);
% fprintf('\n\tCost of getting TPD wrong: %.2f', cost_TPD);
% fprintf('\n\tClassifier: %g (1- LR, 2- SVM, 3-NN, 4- RF)', classifier_option);
% fprintf('\n\tPCA: %g (1- on; 0- off)', dim_reduction_option);

fprintf('\nDetection stats:\n\tSEN = %.3f, PPV = %.3f, F1 score = %.3f,', SEN_overall{1}, PPV_overall{1}, FS_overall{1});
fprintf('\n\tSPE = %.3f, ACC = %.3f, PABAK = %.3f.\n', SPE_overall{1}, ACC_overall{1}, PABAK_overall);

% Clearing variable
clear sensor_data_fltd sensor_data_sgmntd sensor_data_sgmntd_cmbd_all_sensors...
    sensor_data_sgmntd_cmbd_all_sensors_labeled


%% DETERMINATION OF DETECTION STATS FOR BIOPHYSICAL PROFILING =============

n_movements           = zeros(n_data_files,1);
total_FM_duration     = zeros(n_data_files,1); % This will hold the duration the fetus was moving in each data file
mean_FM_duration      = zeros(n_data_files,1); % Thisi will hold the mean duration of FM in each data file
median_onset_interval = zeros(n_data_files,1); % Thisi will hold the median of the interval between each onset of FM for each data file

FM_dilation_time_new = 2; % Detections within this s will be considered as the same detection

for i = 1:n_data_files
    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, ~] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time_new, Fs_sensor); % Segmentation based on new dilation time
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};

    ML_detection_map              = sensor_data_sgmntd_cmbd_all_sensors_ML{i}; % This indicates detections by Machine Learning algorithm
    reduced_detection_map         = ML_detection_map.*sensor_data_sgmntd_cmbd_all_sensors{1}; % Reduced, because of the new dilation length
    reduced_detection_map_labeled = bwlabel(reduced_detection_map);
    n_movements (i)               = max(reduced_detection_map_labeled);

    detection_only         = reduced_detection_map(reduced_detection_map == 1); % Keeps only the detected segments
    total_FM_duration(i)   = (length(detection_only)/Fs_sensor-n_movements(i)*FM_dilation_time_new/2)/60; % Dilation length in sides of each detection is removed and coverted in minutes
    mean_FM_duration(i)    = total_FM_duration(i)*60/n_movements(i); % mean duration of FM in each data file in s
    %     total_FM_duration(i)   = (length(detection_only)/Fs_sensor)/60; % Dilation length in sides of each detection is removed and coverted in minutes
    %     mean_FM_duration(i)    = (length(detection_only)/Fs_sensor)/n_movements(i); % mean duration of FM in each data file

    onset_interval = zeros(n_movements(i)-1,1);
    for j = 1:(n_movements(i)-1)
        onset1 = find(reduced_detection_map_labeled == j,1); % Sample no. corresponding to start of the label
        onset2 = find(reduced_detection_map_labeled == j+1,1); % Sample no. corresponding to start of the next label

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

%% MEDIAN ONSET INTERVALS FOR COMBINED DATA SETS ========================== 
% Because the median interval calculated above is for individual data sets,
% a different method is necessary to calculate that for a particular week
% for each participants.

n1 = 126;
n2 = 131;
onset_interval_all = 0;
for i = n1 : n2

    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i}, ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, ~] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time_new, Fs_sensor); % Segmentation based on new dilation time
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};

    ML_detection_map              = sensor_data_sgmntd_cmbd_all_sensors_ML{i}; % This indicates detections by Machine Learning algorithm
    reduced_detection_map         = ML_detection_map.*sensor_data_sgmntd_cmbd_all_sensors{1}; % Reduced, because of the new dilation length
    reduced_detection_map_labeled = bwlabel(reduced_detection_map);
    n_movements (i)               = max(reduced_detection_map_labeled);

    onset_interval = zeros(n_movements(i)-1,1);
    for j = 1:(n_movements(i)-1)
        onset1 = find(reduced_detection_map_labeled == j,1); % Sample no. corresponding to start of the label
        onset2 = find(reduced_detection_map_labeled == j+1,1); % Sample no. corresponding to start of the next label

        onset_interval(j) = (onset2-onset1)/Fs_sensor; % onset to onset interval in seconds
    end
    onset_interval_all = [onset_interval_all;onset_interval];
   
end

onset_interval_all = onset_interval_all(2:end);

 median_onset_interval_combined = median(onset_interval_all) % Gives median onset interval in s



%% MODEL DIAGNOSTICS ======================================================
% This section performs disgnostic on the learned classifiers

% ---------------------------- Learning curve -----------------------------
fprintf('\nLearning curve generation is going on...\n');

% Dividing the data set into training and testing data sets
training_protion = 0.8; % Training portion in case of holdout method- option 1
[X_train_LC,Y_train_LC,X_test_LC,Y_test_LC,n_training_data_TPD,...
    n_training_data_FPD] = divide_by_holdout(X_TPD_norm_ranked,X_FPD_norm_ranked,training_protion);

% Randomization of training and testing data
%   This randomization is necessary as the Y are first all 1 and then all 0.
%   This randomization will remove that problem.
rng default;
rand_num   = randperm(size(X_train_LC,1));
X_train_LC = X_train_LC(rand_num,:);
Y_train_LC = Y_train_LC(rand_num,:);

rng default;
rand_num  = randperm(size(X_test_LC,1));
X_test_LC = X_test_LC(rand_num,:);
Y_test_LC = Y_test_LC(rand_num,:);

increment = 100; % Increment of training size in each iterration
n_iter    = floor((n_training_data_TPD+n_training_data_FPD)/increment); % Number of iteration

LR_train_error_LC  = zeros(n_iter,1);
LR_test_error_LC   = zeros(n_iter,1);
SVM_train_error_LC = zeros(n_iter,1);
SVM_test_error_LC  = zeros(n_iter,1);
NN_train_error_LC  = zeros(n_iter,1);
NN_test_error_LC   = zeros(n_iter,1);
RF_train_error_LC  = zeros(n_iter,1);
RF_test_error_LC   = zeros(n_iter,1);

test_all_models = 0; % 1- all the models are tested simulteneously; 0- only 1 model is tested
classifier_option = 3;
load_NN_LC_result_option = 1;

for i = 1:n_iter
    % Generating training data set
    X_train_temp = X_train_LC(1:i*increment,:);
    Y_train_temp = Y_train_LC(1:i*increment,:);

    % Applying learning models
    if (test_all_models == 1)||(classifier_option == 2)
        % --------------------------------- SVM -------------------------------
        [SVM_Z_train_temp,SVM_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
            X_test_LC,dim_reduction_option,U,n_PC,I_max_SVM);
        rng default; % sets random seed to default value. This is necessary for reproducibility.
        SVM_model_LC = fitcsvm(SVM_Z_train_temp,Y_train_temp,'KernelFunction','rbf','Cost', ...
            cost_function,'BoxConstraint',C,'KernelScale',sigma);

        % Evaluation of traning and testing errors
        SVM_train_prediction_temp = predict(SVM_model_LC, SVM_Z_train_temp);
        SVM_train_error_LC(i) = 1 - sum((SVM_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

        SVM_test_prediction_temp = predict(SVM_model_LC, SVM_Z_test_temp);
        SVM_test_error_LC(i) = 1- sum((SVM_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
        %
    end

    if (test_all_models == 1)||(classifier_option == 3)
        % --------------------------------- NN _-------------------------------
        if load_NN_LC_result_option == 0
            [NN_Z_train_temp,NN_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
                X_test_LC,dim_reduction_option,U,n_PC,I_max_NN);
            rng default; % sets random seed to default value. This is necessary for reproducibility.
            NN_model_LC = fitcnet(NN_Z_train_temp,Y_train_temp,'LayerSizes', NN_layer_size,...
                'Activation','sigmoid','Lambda',NN_lambda);

            % Evaluation of traning and testing errors
            NN_train_prediction_temp = predict(NN_model_LC, NN_Z_train_temp);
            NN_train_error_LC(i) = 1 - sum((NN_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

            NN_test_prediction_temp = predict(NN_model_LC, NN_Z_test_temp);
            NN_test_error_LC(i) = 1- sum((NN_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
            %
        else
            NN_train_error_LC = load('LC_trainError_from_python.txt');
            NN_test_error_LC  = load('LC_testError_from_python.txt');
        end
    end

    if (test_all_models == 1)||(classifier_option == 4)
        % --------------------------------- RF --------------------------------
        [RF_Z_train_temp,RF_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
            X_test_LC,dim_reduction_option,U,n_PC,I_max_RF);
        rng default; % sets random seed to default value. This is necessary for reproducibility.
        RF_model_LC = fitcensemble(RF_Z_train_temp,Y_train_temp,'Cost', cost_function,...
            'Learners', t_selected,'Method','Bag','NumlearningCycles',100,'Resample','on','Replace','on','Fresample',1);

        % Evaluation of traning and testing errors
        RF_train_prediction_temp = predict(RF_model_LC, RF_Z_train_temp);
        RF_train_error_LC(i) = 1 - sum((RF_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

        RF_test_prediction_temp = predict(RF_model_LC, RF_Z_test_temp);
        RF_test_error_LC(i) = 1- sum((RF_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
        %
    end

    if (test_all_models == 1)||(classifier_option == 1)
        % ---------------------- Logistic regression --------------------------
        [LR_Z_train_temp,LR_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
            X_test_LC,dim_reduction_option,U,n_PC,I_max_LR);
        rng default;
        LR_model_LC = fitclinear(LR_Z_train_temp, Y_train_temp,'Learner','logistic',...
            'Lambda', lambda_selected,'Cost', cost_function);

        % Evaluation of traning and testing errors
        %     LR_train_prediction_temp = predict_LR(LR_theta_LC, LR_Z_train_temp);
        LR_train_prediction_temp = predict(LR_model_LC, LR_Z_train_temp);
        LR_train_error_LC(i) = 1 - sum((LR_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

        %     LR_test_prediction_temp = predict_LR(LR_theta_LC, LR_Z_test_temp);
        LR_test_prediction_temp = predict(LR_model_LC, LR_Z_test_temp);
        LR_test_error_LC(i) = 1- sum((LR_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
        %
    end

    fprintf('End of iteration: %.0f/%.0f\n',i,n_iter);
end

% Plotting the learning curve 
%   Plot settings
B_width = 3; % Width of the box
L_width = 4; % Width of the plot line
F_name = 'Times New Roman'; % Font type
F_size = 32; % Font size
y_limit = 40;

tiledlayout(2,2,'Padding','tight','TileSpacing','tight');
x_coordinate = (1:n_iter)*increment;

%   Tile 1
nexttile
plot(x_coordinate,LR_train_error_LC*100,x_coordinate,LR_test_error_LC*100,'LineWidth',L_width)
ylabel('Error (%)');
legend('Training error','Testing error');
legend('Training error', 'Test error', 'Location', 'best');
legend boxoff
ylim([0 y_limit])
yticks(0:10:y_limit)
set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', B_width) % Sets the font type and size of the plot

%   Tile 2
nexttile
plot(x_coordinate,SVM_train_error_LC*100,x_coordinate,SVM_test_error_LC*100, 'LineWidth',L_width)
legend('Training error','Testing error');
legend('Training error', 'Test error', 'Location', 'best');
legend boxoff
ylim([0 y_limit])
yticks(0:10:y_limit)
set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', B_width) % Sets the font type and size of the plot

%   Tile 3
nexttile
plot(x_coordinate,NN_train_error_LC*100,x_coordinate,NN_test_error_LC*100, 'LineWidth',L_width)
xlabel('No. of training examples');
ylabel('Error (%)');
legend('Training error','Testing error');
legend('Training error', 'Test error', 'Location', 'best');
legend boxoff
ylim([0 y_limit])
yticks(0:10:y_limit)
set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', B_width) % Sets the font type and size of the plot

%   Tile 4
nexttile
plot(x_coordinate,RF_train_error_LC*100,x_coordinate,RF_test_error_LC*100, 'LineWidth',L_width)
xlabel('No. of training examples');
legend('Training error','Testing error');
legend('Training error', 'Test error', 'Location', 'best');
legend boxoff
ylim([0 y_limit])
yticks(0:10:y_limit)
set(gca, 'FontName', F_name, 'FontSize', F_size, 'linewidth', B_width) % Sets the font type and size of the plot
%

fprintf('\nLearning curve generation is completed.\n');

%% VISUALIZATION OF THE ALGORITHM =========================================
% Filtered and segmented data from all the sensors are plotted in a subplot
% along with the maternal sensation markings and the body movement map

% Starting notification
disp('Generating the visualization... ')

% Parameters for creating sensation map and detection matching
FM_min_SN = [30, 30, 30, 30, 30, 30]; % These values are selected to get SEN of 98%
IMU_threshold = [0.003 0.002]; % fixed threshold value obtained through seperate testing
% threshold = zeros(n_data_files, n_FM_sensors); % Variable for thresold

n1 = 7; % 31 and 7 for the examples used in the paper,68 for ML
n2 = 1;

% Used in the paper: n1 = 7: S1_Day5_dataFile_000 (200s - 400s)

for i = n1 : n1

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
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i},data_file_names,Fs_sensor);
    else
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i},data_file_names{i},Fs_sensor);
    end

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

    sensor_data_fltd = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i},  ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, ~] = get_segmented_data(sensor_data_fltd, FM_min_SN, IMU_map{i}, FM_dilation_time, Fs_sensor);

    % ~~~~~~~~~~~~~~~~~~~~~~~~~~ Sensor fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~%
    % Combining left and right sensors of each type
    sensor_data_sgmntd_Left_OR_Right_Aclm  = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2})};
    sensor_data_sgmntd_Left_OR_Right_Acstc = {double(sensor_data_sgmntd{3} | sensor_data_sgmntd{4})};
    sensor_data_sgmntd_Left_OR_Right_Pzplt = {double(sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};

    % Combining sensor data of different types
    % SENSOR FUSION SCHEME 1: Combination based on logical OR operation
    %   Combined data are stored as a cell to make it compatable with the
    %   function related to matcing with maternal sensation
    sensor_data_sgmntd_cmbd_all_sensors_OR = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    %

    % SENSOR FUSION SCHEME 2: Combination based on logical AND operation
    %   Combined data are stored as a cell to make it compatable with the
    %   function related to matcing with maternal sensation
    sensor_data_sgmntd_cmbd_all_sensors_AND = {double((sensor_data_sgmntd{1} | sensor_data_sgmntd{2}) & (sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4}) & (sensor_data_sgmntd{5} | sensor_data_sgmntd{6}))};
    %

    % SENSOR FUSION SCHEME 3: Combination based on detection by atleaset n sensors
    %   All the sensor data are first combined with logical OR. Each
    %   non-zero sengment in the combined data is then checked against
    %   individual sensor data to find its presence in that data set.
    %   Combined data are stored as a cell to make it compatable with the
    %   function related to matcing with maternal sensation.
    sensor_data_sgmntd_cmbd_all_sensors = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    sensor_data_sgmntd_cmbd_all_sensors_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    n_label = length(unique(sensor_data_sgmntd_cmbd_all_sensors_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    sensor_data_sgmntd_cmbd_multi_type_sensors_OR = [sensor_data_sgmntd_Left_OR_Right_Acstc, sensor_data_sgmntd_Left_OR_Right_Aclm, sensor_data_sgmntd_Left_OR_Right_Pzplt];

    % Initialization of variables
    sensor_data_sgmntd_atleast_1_type{1} = zeros(length(sensor_data_sgmntd{1}),1);
    sensor_data_sgmntd_atleast_2_type{1} = zeros(length(sensor_data_sgmntd{1}),1);
    sensor_data_sgmntd_atleast_3_type{1} = zeros(length(sensor_data_sgmntd{1}),1);

    if (n_label) % When there is a detection by the sensor system
        for k = 1 : n_label
            L_min = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1 ); % Sample no. corresponding to the start of the label
            L_max = find(sensor_data_sgmntd_cmbd_all_sensors_labeled == k, 1, 'last' ); % Sample no. corresponding to the end of the label

            indv_detection_map = zeros(length(sensor_data_sgmntd{1}),1); % Need to be initialized before every detection matching
            indv_detection_map(L_min:L_max) = 1; % mapping individual sensation data

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
                    disp('This would never print.')
            end
        end
    end

    % ------------------------- Plotting the data ------------------------%
    % Parameters for plotting
    segmentation_multiplier = 0.9; % To control the height of the segmentation map in the plot
    m_sntn_multiplier       = segmentation_multiplier/2; % To control the height of the maternal sensation marks
    scheme_multiplier       = 1.5; % To control the height of the overall detection map for different scheme in the plot
    y_lim_multiplier        = 1.05; % To control the Y-axis limit

    B_width = 4; % Width of the box (for paper = 4)
    L_width = 3; % Width of the lines in the plot (for paper = 4)

    Font_size_labels = 34; %(for paper = 36)
    Font_size_legend = 28; % (for paper = 28)
    Font_type_legend = 'Times New Roman';
    n_column_legend  = 1; % Number of columns in the legend
    location_legend  = 'northeastoutside';

    legend_option = 0; % To turn on (= 1) and off (= 0) the legends
    time_limit_option = 1; % If 1, it will allow to put a limit on the time axis value

    if time_limit_option == 1
        xlim_1 = 200; % 336 for the example used in the paper (data set no. 31) 250.5
        xlim_2 = 400; % 536 for the example used in the paper (data set no. 31) 390.5
    else
        xlim_1 = 0;
        xlim_2 = (length(Acstc_data1_fltd{i})-1)/Fs_sensor;
    end

    xlim_index1               = xlim_1*Fs_sensor+1; % 1 is added because Matlab index starts from 1. If xlim_1 = 0, index = 1
    xlim_index2               = xlim_2*Fs_sensor+1;
    sensorData_time_vector    = (xlim_index1-1:xlim_index2-1)/Fs_sensor; % time vector for the data in SD1. 1 is deducted because time vector starts from 0 s.
    sensationData_time_vector = (xlim_index1-1:xlim_index2-1)/Fs_sensation; % either of sensation_data_SD1 or SD2 could have been used here

    sensation_matrix = [sensationData_time_vector', sensation_data_SD1_trimd{i}(xlim_index1:xlim_index2)]; % holds both sensation data and time vector
    sensation_index  = find (sensation_matrix(:,2)); % index of the non-zero elements in the sensation data
    sensation_only   = sensation_matrix(sensation_index,:); % non-zero sensation data in column 2 and the corresponding timings in column 1

    sensorData_time_vector = sensorData_time_vector-xlim_1; % To make the time vector start from 0.
    sensation_only(:,1)    = sensation_only(:,1)-xlim_1;

    % Plotting using tiles
    tiledlayout(9,1,'Padding','tight','TileSpacing','none');

    % Tile 1
    nexttile
    plot(sensorData_time_vector, IMU_data_fltd{i}(xlim_index1:xlim_index2), 'LineWidth', L_width); % Plots the IMU data at the first row
    hold on;
    plot(sensorData_time_vector, IMU_map{i}(xlim_index1:xlim_index2)*IMU_threshold(1), 'Color', [0 0 0], 'LineWidth', L_width); % Plots the body movement map
    hold off;
    ylim([min(IMU_data_fltd{i}(xlim_index1:xlim_index2))*y_lim_multiplier, ...
        max(IMU_data_fltd{i}(xlim_index1:xlim_index2))*y_lim_multiplier]); % Setting the Y limit to get the best view
    axis off; % Puts both the axes off

    if legend_option == 1
        lgd = legend('IMU accelerometer', 'Maternal body movement map');
        lgd.FontName = Font_type_legend;
        lgd.FontSize = Font_size_legend;
        lgd.NumColumns = n_column_legend;
        legend('Location',location_legend)
        legend boxoff;
    end

    % Tile 2 - 7
    for j = 1:1:n_FM_sensors
        nexttile
        plot(sensorData_time_vector, sensor_data_fltd{j}(xlim_index1:xlim_index2),'LineWidth', L_width)
        hold on;
        %         plot(sensorData_time_vector, sensor_data_sgmntd{j}(xlim_index1:xlim_index2)...
        %             *max(sensor_data_fltd{j}(xlim_index1:xlim_index2))*segmentation_multiplier,'color', [0.4660 0.6740 0.1880], 'LineWidth', L_width)
        hold on;
        plot(sensation_only(:,1), sensation_only(:,2)*max(sensor_data_fltd{j}(xlim_index1:xlim_index2))*m_sntn_multiplier, 'r*', 'LineWidth', L_width);
        hold off;
        ylim([min(sensor_data_fltd{j}(xlim_index1:xlim_index2))*y_lim_multiplier, max(sensor_data_fltd{j}(xlim_index1:xlim_index2))*y_lim_multiplier]);
        axis off;

        if (j == 2)&&(legend_option==1)
            lgd            = legend ('Sensor response', 'Detection by maternal sensation');
            lgd.FontName   = Font_type_legend;
            lgd.FontSize   = Font_size_legend;
            lgd.NumColumns = n_column_legend;

            legend('Location',location_legend)
            legend boxoff;
        end
    end

    % Tile 8
    nexttile
    plot(sensorData_time_vector, sensor_data_sgmntd_cmbd_all_sensors_OR{1}(xlim_index1:xlim_index2)*scheme_multiplier,...
        'color', [0.4940, 0.1840, 0.5560], 'LineWidth', L_width); % Plotting sensor fusion scheme 1
    hold on
    plot(sensation_only(:,1), sensation_only(:,2)*scheme_multiplier/2, 'r*', 'LineWidth', L_width);
    ylim([-1, 2*y_lim_multiplier]);
    axis off;

    if legend_option == 1
        lgd = legend('Detection by thresholding-based algorithm');
        lgd.FontName = Font_type_legend;
        lgd.FontSize = Font_size_legend;
        lgd.NumColumns = n_column_legend;
        legend('Location',location_legend)
        legend boxoff;
    end

%     % Tile 9
%     nexttile
%     plot(sensorData_time_vector, sensor_data_sgmntd_cmbd_all_sensors_AND{1}(xlim_index1:xlim_index2)*scheme_multiplier,...
%         'color', [0.8500, 0.3250, 0.0980], 'LineWidth', L_width);  % Plotting sensor fusion scheme 2
%     hold on
%     plot(sensation_only(:,1), sensation_only(:,2)*scheme_multiplier/2, 'r*', 'LineWidth', L_width);
%     ylim([-1, 2*y_lim_multiplier]);
%     axis off;
% 
%     if legend_option == 1
%         lgd = legend('All types of sensors');
%         lgd.FontName = Font_type_legend;
%         lgd.FontSize = Font_size_legend;
%         lgd.NumColumns = n_column_legend;
%         legend('Location',location_legend)
%         legend boxoff;
%     end
% 
    % Tile 10
%     nexttile
%     plot(sensorData_time_vector, sensor_data_sgmntd_atleast_3_type{1}(xlim_index1:xlim_index2)*scheme_multiplier,...
%         'color', [0.9290 0.6940 0.1250], 'LineWidth', L_width);  % Plotting sensor fusion scheme 3
%     hold on
%     plot(sensation_only(:,1), sensation_only(:,2)*scheme_multiplier/2, 'r*', 'LineWidth', L_width);
%     ylim([-1, 2*y_lim_multiplier]);
%     axis off
% 
%     if legend_option == 1
%         lgd = legend('Thresholding-based detection');
%         lgd.FontName = Font_type_legend;
%         lgd.FontSize = Font_size_legend;
%         lgd.NumColumns = n_column_legend;
%         legend('Location',location_legend)
%         legend boxoff;
%     end

    % Tile 11
    nexttile
    plot(sensorData_time_vector, sensor_data_sgmntd_cmbd_all_sensors_ML{i}(xlim_index1:xlim_index2)*scheme_multiplier,...
        'color', [0.3010 0.7450 0.9330], 'LineWidth', L_width);  % Plotting sensor fusion scheme 3
    hold on
    plot(sensation_only(:,1), sensation_only(:,2)*scheme_multiplier/2, 'r*', 'LineWidth', L_width);
    ylim([-1, 2*y_lim_multiplier]);
    %     xlim([-inf, max(sensorData_time_vector)]);
    h               = gca;
    h.YAxis.Visible = 'off'; % Puts the Y-axis off
    box off % Removes the box of the plot
    xlabel('Time (s)');

    if legend_option == 1
        lgd            = legend('Detection by machine learning-based algorithm');
        lgd.FontName   = Font_type_legend;
        lgd.FontSize   = Font_size_legend;
        lgd.NumColumns = n_column_legend;

        legend('Location',location_legend)
        legend boxoff;
    end

    % Setting up the labels and ticks
    set(gca, 'FontName','Times New Roman','FontSize',Font_size_labels) % Sets the font type and size of the labels
    set(gca,'linewidth',B_width)
    set(gca,'TickLength',[0.005, 0.01])
    %

end

fprintf('Data visualization is completed.\n');
%












