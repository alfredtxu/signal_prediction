% This script is for optimising the machine learning model on FM data
clc
clear
close all

curr_dir = pwd;
cd(curr_dir)
% -------------------------------------------------------------------------------

%% Parameter definition
freq_sensor = 1024;
freq_sensation = 1024;
num_FMsensor = 6;

ext_backward = 5.0; % Backward extension length in second
ext_forward = 2.0; % Forward extension length in second
FM_dilation_time = 3.0; % Dilation size in seconds

FM_min_SN = [30, 30, 30, 30, 30, 30];

% the cohort of participant
participants = {'S1', 'S2', 'S3', 'S4', 'S5'};

% Add path for function files
% old sub-directories
% addpath(genpath('SP_function_files'))
% addpath(genpath('ML_function_files'))
% addpath(genpath('matplotlib_function_files'))
% addpath(genpath('Learned_models'))

% new sub-directories
addpath(genpath('z10_olddata_mat_raw'))
addpath(genpath('z11_olddata_mat_preproc'))
addpath(genpath('z12_olddata_mat_proc'))
addpath(genpath('z90_ML_functions'))
addpath(genpath('z91_ML_python_files'))
% ----------------------------------------------------------------------------------------------

%% Data loading: pre-processed data
% filtered, trimmed and equalized the length between two SD cards (*SD1 & SD2 for this case)
for i = 1 : size(participants, 2)

    tmp_mat = ['sensor_data_suite_' participants{i} '_preproc.mat'];
    load(tmp_mat);

    all_forc{i,:} = forcFilTEq;
    all_acceL{i,:} = acceLNetFilTEq;
    all_acceR{i,:} = acceRNetFilTEq;
    all_acouL{i,:} = acouLFilTEq;
    all_acouR{i,:} = acouRFilTEq;
    all_piezL{i,:} = piezLFilTEq;
    all_piezR{i,:} = piezRFilTEq;
    all_IMUacce{i,:} = IMUacceNetFilTEq;

    all_sens1{i,:} = sens1TEq;
    all_sens2{i,:} = sens2TEq;
    all_sensMtx{i,:} = sens1TEq_mtxP;

    all_timeV{i,:} = time_vecTEq;
    
    all_nfile(i,:) = size(forcFilTEq, 1);

    fprintf('Loaded pre-processed data ... %d (%d) - %s ... \n', i, all_nfile(i,:), tmp_mat);

    clear acce* acou* forc* IMU* piez* sens* time* tmp_mat
end

% the number of data files in total
num_dfiles = sum(all_nfile);

for i = 1 : size(all_forc, 1)

    tmp_forc = all_forc{i};

    for j = 1 : size(tmp_forc, 1)

        tmp_forc_sub = abs(tmp_forc{j});
        info_forc(j).mu = mean(tmp_forc_sub);
        info_forc(j).spower = sum(tmp_forc_sub.^2)/length(tmp_forc_sub);
        info_forc(j).dura = length(tmp_forc_sub)/(freq_sensor*60);
        clear tmp_forc_sub
    end

    info_dfile{i, :} = info_forc;

    clear tmp_forc info_forc
end

% merge the data files from the cohort of paricipants
all_IMUacce_cat = cat(1,all_IMUacce{:});
all_forc_cat = cat(1,all_forc{:});

all_acceL_cat = cat(1,all_acceL{:});
all_acceR_cat = cat(1,all_acceR{:});
all_acouL_cat = cat(1,all_acouL{:});
all_acouR_cat = cat(1,all_acouR{:});
all_piezL_cat = cat(1,all_piezL{:});
all_piezR_cat = cat(1,all_piezR{:});

all_sens1_cat = cat(1,all_sens1{:});
all_sens2_cat = cat(1,all_sens2{:});
all_sensMtx_cat = cat(1,all_sensMtx{:});
all_timeV_cat = cat(1,all_timeV{:});
% ----------------------------------------------------------------------------------------------

%% Data loading: processed data
% signal segmentation and dilation
for i = 1 : size(participants, 2)

    tmp_mat = ['sensor_data_suite_' participants{i} '_proc.mat'];
    load(tmp_mat);

    % single type FM sensor (operator: OR)
    all_fusion1_accl{i,:} = sensor_fusion1_accl;
    all_fusion1_acou{i,:} = sensor_fusion1_acou;
    all_fusion1_piez{i,:} = sensor_fusion1_piez;

    % two-type combined FM sensors (operator: OR)
    all_fusion1_accl_acou{i,:} = sensor_fusion1_accl_acou;
    all_fusion1_accl_piez{i,:} = sensor_fusion1_accl_piez;
    all_fusion1_acou_piez{i,:} = sensor_fusion1_acou_piez;
    
    % three-type combined FM sensors (operator: OR)
    all_fusion1_accl_acou_piez{i,:} = sensor_fusion1_all;
    all_fusion1_accl_acou_piez_bw{i,:} = sensor_fusion1_all_bw;
    all_fusion1_accl_acou_piez_comp{i,:} = sensor_fusion1_all_comp;

    % segmented FM sensor suite & according thresholds
    all_seg{i,:} = sensor_suite_segmented;
    all_FM_thresh{i,:} = sensor_suite_thresh;

    % IMU and sensation maps
    all_IMUacce_map{i,:} = IMUacce_map;
    all_sensation_map{i,:} = sensation_map;
    all_sens_map_label{i,:} = sens_map_label;
    all_sens_label{i,:} = sens_label;

    fprintf('Loaded processed data ... %d (%d) - %s ... \n', i, size(sensor_fusion1_accl,1), tmp_mat);

    clear sensor_fusion* tmp_mat
end

% merge the data files from the cohort of paricipants
all_seg_cat = cat(1, all_seg{:});

all_fusion1_accl_cat = cat(1,all_fusion1_accl{:});
all_fusion1_acou_cat = cat(1,all_fusion1_acou{:});
all_fusion1_piez_cat = cat(1,all_fusion1_piez{:});

all_fusion1_accl_acou_cat = cat(1,all_fusion1_accl_acou{:});
all_fusion1_accl_piez_cat = cat(1,all_fusion1_accl_piez{:});
all_fusion1_acou_piez_cat = cat(1,all_fusion1_acou_piez{:});

all_fusion1_accl_acou_piez_cat = cat(1,all_fusion1_accl_acou_piez{:});
all_fusion1_accl_acou_piez_bw_cat = cat(1,all_fusion1_accl_acou_piez_bw{:});
all_fusion1_accl_acou_piez_comp_cat = cat(1,all_fusion1_accl_acou_piez_comp{:});

all_IMUacce_map_cat = cat(1,all_IMUacce_map{:});
all_sensation_map_cat = cat(1,all_sensation_map{:});
all_sens_map_label_cat = cat(1,all_sens_map_label{:});
all_sens_label_cat = cat(1,all_sens_label{:});

% the number of maternal sensation detection in each data file (remove the 1st element valusd at 0)
for i = 1 : size(all_sens_label_cat, 1) 
    all_sens_labelU_cat(i,:) = length(unique(all_sens_label_cat{i,:})) - 1;
end

% threshold of each sensor
for i = 1 : size(all_FM_thresh, 1)
    
    tmp_thresh = all_FM_thresh{i,:};
    
    all_acceL_thresh{i,:} = tmp_thresh(:,1);
    all_acceR_thresh{i,:} = tmp_thresh(:,2);
    all_acouL_thresh{i,:} = tmp_thresh(:,3);
    all_acouR_thresh{i,:} = tmp_thresh(:,4);
    all_piezL_thresh{i,:} = tmp_thresh(:,5);
    all_piezR_thresh{i,:} = tmp_thresh(:,6);
    
    clear tmp_thresh
end

all_acceL_thresh_cat = cat(1,all_acceL_thresh{:});
all_acceR_thresh_cat = cat(1,all_acceR_thresh{:});
all_acouL_thresh_cat = cat(1,all_acouL_thresh{:});
all_acouR_thresh_cat = cat(1,all_acouR_thresh{:});
all_piezL_thresh_cat = cat(1,all_piezL_thresh{:});
all_piezR_thresh_cat = cat(1,all_piezR_thresh{:});

%% Sensor fusion selection
% Options: single / double / three types of FM sensor combination
sensor_selection = 'accl_acou_piez';

all_acceL_cat = cat(1,all_acceL{:});
all_acceR_cat = cat(1,all_acceR{:});
all_acouL_cat = cat(1,all_acouL{:});
all_acouR_cat = cat(1,all_acouR{:});
all_piezL_cat = cat(1,all_piezL{:});
all_piezR_cat = cat(1,all_piezR{:});

switch sensor_selection
    case 'accl'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat};
        selected_data_proc = all_fusion1_accl_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat);
        selected_sensorT = 1;
        selected_sensorN = 2;
    case 'acou'
        selected_data_preproc = {all_acouL_cat, all_acouR_cat};
        selected_data_proc = all_fusion1_acou_cat;
        selected_thresh = cat(2, all_acouL_thresh_cat, all_acouR_thresh_cat);
        selected_sensorT = 1;
        selected_sensorN = 2;
    case 'piez'
        selected_data_preproc = {all_piezL_cat, all_piezR_cat};
        selected_data_proc = all_fusion1_piez_cat;
        selected_thresh = cat(2, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 1;
        selected_sensorN = 2;
    case 'accl_acou'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_acouL_cat, all_acouR_cat};
        selected_data_proc = all_fusion1_accl_acou_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat);
        selected_sensorT = 2;
        selected_sensorN = 4;
    case 'accl_piez'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_piezL_cat, all_piezR_cat};
        selected_data_proc = all_fusion1_accl_piez_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 2;
        selected_sensorN = 4;
    case 'acou_piez'
        selected_data_preproc = {all_acouL_cat, all_acouR_cat, all_acouL_cat, all_acouR_cat};
        selected_data_proc = all_fusion1_acou_piez_cat;
        selected_thresh = cat(2, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 2;
        selected_sensorN = 4;
    case 'accl_acou_piez'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_acouL_cat, all_acouR_cat, all_piezL_cat, all_piezR_cat};
        selected_data_proc = all_fusion1_accl_acou_piez_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 3;
        selected_sensorN = 6;
    otherwise
        disp('Selected FM sensor(s) is out of range ...');
end

%% Extract detection
% Indicate ture positive (TP) & false positive (FP) classes
for i = 1 : num_dfiles

    tmp_lab = bwlabel(selected_data_proc{i,:});
    tmp_numLab = length(unique(tmp_lab)) - 1;
    tmp_detection_numTP = length(unique(tmp_lab .* all_sensation_map_cat{i,:})) - 1; 
    tmp_detection_numFP = tmp_numLab - tmp_detection_numTP;

    tmp_detection_TPc = cell(1, tmp_detection_numTP);
    tmp_detection_FPc = cell(1, tmp_detection_numFP);
    tmp_detection_TPw = zeros(tmp_detection_numTP, 1);
    tmp_detection_FPw = zeros(tmp_detection_numFP, 1);

    tmp_cTP = 0;
    tmp_cFP = 0;
    
    for j = 1 : tmp_numLab

        tmp_idxS = find(tmp_lab == j, 1); 
        tmp_idxE = find(tmp_lab == j, 1, 'last'); 

        tmp_mtx = zeros(length(all_sensation_map_cat{i,:}),1);
        tmp_mtx(tmp_idxS : tmp_idxE) = 1;
        
        % case: sensor_selection - 'accl_acou_piez'
        % all_fusion1_accl_cat, all_fusion1_acou_cat, all_fusion1_piez_cat
        tmp_ntype = 0;
        if selected_sensorT == 3
            tmp_all_sensor = {all_fusion1_accl_cat{i, :}, all_fusion1_acou_cat{i, :}, all_fusion1_piez_cat{i, :}};
            for k = 1 : selected_sensorT
                if (sum(tmp_mtx .* tmp_all_sensor{k})) 
                    tmp_ntype = tmp_ntype + 1;
                end
            end
            clear tmp_all_sensor
        end

        % Current (labelled) section: TPD vs FPD class
        tmp_tp = sum(tmp_mtx .* all_sensation_map_cat{i,:});

        if tmp_tp > 0
            
            tmp_cTP = tmp_cTP + 1;
            tmp_TPextraction = zeros(tmp_idxE-tmp_idxS+1, selected_sensorN);

            for k = 1 : selected_sensorN
                tmp_data_preproc = selected_data_preproc{k};
                tmp_TPextraction(:, k) = tmp_data_preproc{i, :}(tmp_idxS:tmp_idxE); 
                clear tmp_data_preproc
            end

            tmp_detection_TPc{tmp_cTP} = tmp_TPextraction;
            tmp_detection_TPw(tmp_cTP) = tmp_ntype;

        else
            tmp_cFP = tmp_cFP + 1;
            tmp_FPextraction = zeros(tmp_idxE-tmp_idxS+1, selected_sensorN);

            for k = 1 : selected_sensorN
                tmp_data_preproc = selected_data_preproc{k};
                tmp_FPextraction(:, k) = tmp_data_preproc{i, :}(tmp_idxS:tmp_idxE);
                clear tmp_data_preproc
            end
            
            tmp_detection_FPc{tmp_cFP} = tmp_FPextraction;
            tmp_detection_FPw(tmp_cFP) = tmp_ntype;
        end

        clear tmp_idxS tmp_idxE tmp_mtx tmp_ntype tmp_TPextraction tmp_FPextraction tmp_tp 
    end

    detection_TPc{i, :} = tmp_detection_TPc;
    detection_TPw{i,:} = tmp_detection_TPw;

    detection_FPc{i, :} = tmp_detection_FPc;
    detection_FPw{i,:} = tmp_detection_FPw;

    % sensation detection summary
    fprintf('Data file: %d - the number of labels is %d (TP: %d vs FP: %d) ... \n', i, tmp_numLab, length(tmp_detection_TPw), length(tmp_detection_FPw));

    clear tmp_lab tmp_numLab ...
          tmp_cTP tmp_cFP ...
          tmp_detection_numTP tmp_detection_numFP ...
          tmp_detection_TPc tmp_detection_FPc ...
          tmp_detection_TPw tmp_detection_FPw
end

%% Data feature extraction for machine learning
% the total number of true & false positive (TP & FP) detections
num_TP = 0;
num_FP = 0;
dura_TP = 0;
dura_FP = 0;

% Loop through the cohort of data files
for i = 1 : size(detection_TPc, 1)
    num_TP = num_TP + size(detection_TPc{i, :}, 2);
    num_FP = num_FP + size(detection_FPc{i, :}, 2);

    for j = 1: size(detection_TPc{i, :}, 2)
        dura_TP = dura_TP + size(detection_TPc{i, :}{j}, 1);
    end
    for j = 1: size(detection_FPc{i, :}, 2)
        dura_FP = dura_FP + size(detection_FPc{i, :}{j}, 1);
    end
end

% duration: in hours
dura_TP = dura_TP/1024/3600; 
dura_FP = dura_FP/1024/3600; 

% feature extraction: TP & FP
feature_TP = zeros(num_TP,1);
feature_FP = zeros(num_FP,1);

% feature indice (16 feature in total)
idx_TP = 1;
idx_FP = 1;
idx_num = 16;

% feature extraction: TP class
for i = 1 : size(detection_TPc, 1)

    for j = 1 : size(detection_TPc{i, :}, 2)
       
        feature_TP(idx_TP, 1) = length(detection_TPc{i, :}{j})/freq_sensor; 

        for k = 1 : selected_sensorN
            
            tmp_signal = detection_TPc{i, :}{j}(:,k);
            tmp_signal_thresh = abs(tmp_signal) - selected_thresh(i, k);
            tmp_signal_threshGt = tmp_signal_thresh(tmp_signal_thresh>0);

            % Time domain features
            feature_TP(idx_TP, (k-1)*idx_num+2) = max(tmp_signal_thresh); % Max value
            feature_TP(idx_TP, (k-1)*idx_num+3) = mean(tmp_signal_thresh); % Mean value
            feature_TP(idx_TP, (k-1)*idx_num+4) = sum(tmp_signal_thresh.^2); % Energy
            feature_TP(idx_TP, (k-1)*idx_num+5) = std(tmp_signal_thresh); % Standard deviation
            feature_TP(idx_TP, (k-1)*idx_num+6) = iqr(tmp_signal_thresh); % Interquartile range
            feature_TP(idx_TP, (k-1)*idx_num+7) = skewness(tmp_signal_thresh); % Skewness
            feature_TP(idx_TP, (k-1)*idx_num+8) = kurtosis(tmp_signal_thresh); % Kurtosis

            if isempty(tmp_signal_threshGt)
                feature_TP(idx_TP, (k-1)*idx_num+9) = 0; % Duration above threshold
                feature_TP(idx_TP, (k-1)*idx_num+10) = 0; % Mean above threshold value
                feature_TP(idx_TP, (k-1)*idx_num+11) = 0; % Energy above threshold value

            else
                feature_TP(idx_TP,(k-1)*idx_num+9) = length(tmp_signal_threshGt); % Duration above threshold
                feature_TP(idx_TP,(k-1)*idx_num+10) = mean(tmp_signal_threshGt); % Mean above threshold
                feature_TP(idx_TP,(k-1)*idx_num+11) = sum(tmp_signal_threshGt.^2); % Energy above threshold
            end

            % Frequency domain features: the main frequency mode above 1 Hz
            [~,~,feature_TP(idx_TP, (k-1)*idx_num+12)] = ML_get_frequency_mode(tmp_signal, freq_sensor, 1);
            feature_TP(idx_TP, (k-1)*idx_num+13) = ML_getPSD(tmp_signal,freq_sensor, 1, 2);
            feature_TP(idx_TP, (k-1)*idx_num+14) = ML_getPSD(tmp_signal,freq_sensor, 2, 5);
            feature_TP(idx_TP, (k-1)*idx_num+15) = ML_getPSD(tmp_signal,freq_sensor, 5, 10);
            feature_TP(idx_TP, (k-1)*idx_num+16) = ML_getPSD(tmp_signal,freq_sensor, 10, 20);
            feature_TP(idx_TP, (k-1)*idx_num+17) = ML_getPSD(tmp_signal,freq_sensor, 20, 30);

            clear tmp_signal tmp_signal_thresh tmp_signal_threshGt
        end

        idx_TP = idx_TP + 1;

    end

end

% feature extraction: FP class
for i = 1 : length(detection_FPc)
    
    for j = 1 : length(detection_FPc{i, :})

        feature_FP(idx_FP, 1) = length(detection_FPc{i, :}{j})/freq_sensor;
        
        for k = 1 : selected_sensorN

            tmp_signal = detection_FPc{i, :}{j}(:,k);
            tmp_signal_thresh = abs(tmp_signal) - selected_thresh(i, k);
            tmp_signal_threshGt = tmp_signal_thresh(tmp_signal_thresh>0);

            feature_FP(idx_FP, (k-1)*idx_num+2) = max(tmp_signal_thresh); % Max value
            feature_FP(idx_FP, (k-1)*idx_num+3) = mean(tmp_signal_thresh); % Mean value
            feature_FP(idx_FP, (k-1)*idx_num+4) = sum(tmp_signal_thresh.^2); % Energy
            feature_FP(idx_FP, (k-1)*idx_num+5) = std(tmp_signal_thresh); % Standard deviation
            feature_FP(idx_FP, (k-1)*idx_num+6) = iqr(tmp_signal_thresh); % Interquartile range
            feature_FP(idx_FP, (k-1)*idx_num+7) = skewness(tmp_signal_thresh); % Skewness
            feature_FP(idx_FP, (k-1)*idx_num+8) = kurtosis(tmp_signal_thresh); % Kurtosis

            if isempty(tmp_signal_threshGt)
                feature_FP(idx_FP, (k-1)*idx_num+9) = 0; % Duration above threshold
                feature_FP(idx_FP, (k-1)*idx_num+10) = 0; % Mean above threshold value
                feature_FP(idx_FP, (k-1)*idx_num+11) = 0; % Energy above threshold value
            else
                feature_FP(idx_FP, (k-1)*idx_num+9) = length(tmp_signal_threshGt); % Duration above threshold
                feature_FP(idx_FP, (k-1)*idx_num+10) = mean(tmp_signal_threshGt); % Mean above threshold
                feature_FP(idx_FP, (k-1)*idx_num+11) = sum(tmp_signal_threshGt.^2); % Energy above threshold
            end

            [~,~,feature_FP(idx_FP,(k-1)*idx_num+12)] = ML_get_frequency_mode(tmp_signal, freq_sensor, 1);
            feature_FP(idx_FP,(k-1)*idx_num+13) = ML_getPSD(tmp_signal, freq_sensor, 1, 2);
            feature_FP(idx_FP,(k-1)*idx_num+14) = ML_getPSD(tmp_signal, freq_sensor, 2, 5);
            feature_FP(idx_FP,(k-1)*idx_num+15) = ML_getPSD(tmp_signal, freq_sensor, 5, 10);
            feature_FP(idx_FP,(k-1)*idx_num+16) = ML_getPSD(tmp_signal, freq_sensor, 10, 20);
            feature_FP(idx_FP,(k-1)*idx_num+17) = ML_getPSD(tmp_signal, freq_sensor, 20, 30);

            clear tmp_signal tmp_signal_thresh tmp_signal_threshGt
        end

        idx_FP = idx_FP + 1;

    end
end

% Combination of feature extraction
X_features = [feature_TP; feature_FP];
Y_labels = zeros(num_TP + num_FP, 1);
Y_labels(1 : num_TP,1) = 1;

% Feature normalization: norm = (X - Xmin) / (Xman - Xmin)
X_norm = (X_features - min(X_features)) ./ (max(X_features) - min(X_features));
X_norm_TPD = X_norm(1:size(feature_TP,1), :);
X_norm_FPD = X_norm(size(feature_TP,1)+1 : end, :);

% Feature standarlization (z-score normalization): stan = (X - Xmean) / Xstd
X_zscore = (X_features - mean(X_features)) ./ std(X_features);
X_zscore_TPD = X_zscore(1:size(feature_TP,1), :);
X_zscore_FPD = X_zscore(size(feature_TP,1)+1 : end, :);

%% Features' ranking by Neighbourhood Component Analysis (NAC) 
% optimise hyper-parameter: lambda with K-fold cross-validation

% Sets random seed to default value. This is necessary for reproducibility.
rng default  

% number of folds
K = 5;

% fraction of observation in test set
P = 0.2; 

% Stratified K-fold division / holdout division
cvp_K = cvpartition(Y_labels,'kfold', K); 
cvp_P = cvpartition(Y_labels,'HoldOut', P);
cvp_L = cvpartition(Y_labels,'Leaveout');
cvp_R = cvpartition(Y_labels,'Resubstitution');
cvp = cvp_P;

numtestsets = cvp.NumTestSets;
lambdavalues = linspace(0, 2, 20) / length(Y_labels); 
lossvalues = zeros(length(lambdavalues), numtestsets);

% optimization
for i = 1 : length(lambdavalues)   

    for j = 1 : numtestsets
        
        fprintf('Iterating... lambda values > %d of NumTestSets > %d \n', i, j);

        % Extract the training / test set from the partition object
        tmp_X_train = X_norm(cvp.training(j), :);
        tmp_Y_train = Y_labels(cvp.training(j), :);
        tmp_X_test = X_norm(cvp.test(j), :);
        tmp_Y_test = Y_labels(cvp.test(j), :);
        
        % Train an NCA model for classification
        tmp_ncaMdl = fscnca(tmp_X_train, tmp_Y_train, ...
                            'FitMethod', 'exact', 'Verbose', 1, ...
                            'Solver', 'lbfgs', ...
                            'Lambda', lambdavalues(i), ...
                            'IterationLimit', 50, ...
                            'GradientTolerance', 1e-5);
        
        % Compute the classification loss for the test set using the nca model
        lossvalues(i, j) = loss(tmp_ncaMdl, tmp_X_test, tmp_Y_test, 'LossFunction', 'quadratic');      

        clear tmp_X_train tmp_Y_train tmp_X_test tmp_Y_test tmp_ncaMdl
    end                          
end

% Plot lambda vs. loss
figure
plot(lambdavalues, mean(lossvalues,2), 'ro-')
xlabel('Lambda values')
ylabel('Loss values')
grid on

% the index of minimum loass > the best lambda value.
[~,idx] = min(mean(lossvalues,2));
bestlambda = lambdavalues(idx); 

% Use the selected lambda to optimise NCA model
% Stratified
% for all the sensors. 1.144441194796607e-04 from previous run; 1.202855e-04 from another run
% bestlambda = 0.000133548792051176; 
ncaMdl = fscnca(X_norm, Y_labels, ...
                'FitMethod', 'exact', ...
                'Verbose', 1, ...
                'Solver', 'lbfgs', ...
                'Lambda', bestlambda);

figure
semilogx(ncaMdl.FeatureWeights, 'ro')
xlabel('Feature index')
ylabel('Feature weight')   
grid on

% Extract the feature ranking information
% Combines feature index and weights in a matrix
num_top_features = 30;
feature_idx = (1 : size(X_norm,2))'; 
feature_ranking = [feature_idx, ncaMdl.FeatureWeights]; 
[~,I_sort] = sort(feature_ranking(:,2), 'descend');
feature_rankingS = feature_ranking(I_sort,:); 

index_top_features = feature_rankingS(1:num_top_features,1);
X_norm_TPD_ranked = X_norm_TPD(:,index_top_features);
X_norm_FPD_ranked = X_norm_FPD(:,index_top_features);

time_stamp = datestr(now, 'yyyymmddHHMMSS');
save(['X_TPD_norm_ranked_' time_stamp '.txt'], 'X_norm_TPD_ranked', '-ASCII');
save(['X_FPD_norm_ranked_' time_stamp '.txt'], 'X_norm_FPD_ranked', '-ASCII');

%% Training % Test datasets
% Division into training & test sets
% 1: Divide by hold out method 
% 2: K-fold with original ratio of FPD and TPD in each fold, 
% 3: K-fold with custom ratio of FPD and TPD in each fold. 
% 4: Divide by participants
% * cases 1-3: create stratified division (each division will have the same ratio
% of FPD and TPD.)
data_div_option = 2;

if data_div_option == 1 
    
    training_protion = 0.8; 
    
    [X_train, Y_train, X_test, Y_test, ...
     num_training_TP, num_training_FP, num_test_TP, num_test_FP] ...
     = ML_divide_by_holdout(X_norm_TPD_ranked, X_norm_FPD_ranked, training_protion);
    
% Dividing data into stratified K-folds
elseif (data_div_option == 2)||(data_div_option == 3) 
    
    K = 5;

    [X_Kfold, Y_Kfold, ...
     num_TP_each_fold, num_TP_last_fold,...
     num_FP_each_fold, num_FP_last_fold, ...
     TP_FP_ratio, rand_num_TP, rand_num_FP] ...
     = ML_divide_by_Kfolds(X_norm_TPD_ranked, X_norm_FPD_ranked, data_div_option, K);

% Dividing data by participant    
elseif data_div_option == 4 

    [X_TP_by_participant, X_FP_by_participant] = ...
        ML_divide_by_participants(all_nfile, detection_TPc, detection_FPc, ...
                                  X_norm_TPD_ranked, X_norm_FPD_ranked);
else
    fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', data_div_option);
end

%% Cross-validation to find the optimum hyperparameters
fprintf('Optimising hyperparameters through the process of cross-validation ...\n');

if data_div_option == 1
    num_iter = 1; 
elseif (data_div_option == 2)||(data_div_option == 3)
    num_iter = K;
elseif data_div_option == 4
    num_iter = size(participants, 2);
end

% Parameters for PCA
% > Number of principal components
% > U matrix in PCA
num_pca_comp = zeros(1, num_iter);
U_mtx = cell(1, num_iter); 

accu_train = zeros(1, num_iter);
accu_test = zeros(1, num_iter);
accu_testTP = zeros(1, num_iter);
accu_testFP = zeros(1, num_iter);

% Higher waitage for TPD 
% cost_TPD = FPD_TPD_ratio; 
cost_TP = 2;

% Defines the waitage for getting wrong
cost_func = [0, 1; cost_TP, 0]; 

% A range of classification models
SVM_mdl = cell(1, num_iter); 
NN_mdl = cell(1, num_iter); 
LR_mdl = cell(1, num_iter); 
RF_mdl = cell(1, num_iter); 

% Options for classification models
% 1 - Loagistic Regression (LR)
% 2 - Support Vector Machine (SVM )
% 3 - Neural Network (NN)
% 4 - Random Forest (RF)
mdl_option = 2; 

% Dimensionality reduction by PCA: 1 - on, 0 - off.
dim_reduction_option = 0; 

% Cross-validation: K x K-fold - 1; K-fold - 0
cv_option = 1; 

for i = 1 : num_iter
    
    fprintf('Current iteration ... %d / %d ... \n', i, num_iter)
    
    % training & test datasets
    % Split options: 1 - holdout, 2/3 - stratified K-fold, 4 - by participant
    if data_div_option == 1 
        
        tmp_iter_Xtrain = X_train;
        tmp_iter_Ytrain = Y_train;

        tmp_iter_Xtest = X_test;
        tmp_iter_Ytest = Y_test;

        tmp_iter_num_test_TP = num_test_TP;
        tmp_iter_num_test_FP = num_test_FP;

    elseif (data_div_option == 2) || (data_div_option == 3)

        % Partitioning data for K x K nested cross-validation
        % the i-th fold as the test data and rest of the folds as the training data
        tmp_iter_Xtest = X_Kfold{i};
        tmp_iter_Ytest = Y_Kfold{i};

        tmp_iter_Xtrain = [];
        tmp_iter_Ytrain = [];

        for j = 1 : K
            if j ~= i
                tmp_iter_Xtrain = [tmp_iter_Xtrain; X_Kfold{j}];
                tmp_iter_Ytrain = [tmp_iter_Ytrain; Y_Kfold{j}];
            end
        end
        
        if i==num_iter
            tmp_iter_num_test_TP = num_TP_last_fold;
            tmp_iter_num_test_FP = num_FP_last_fold;
        else
            tmp_iter_num_test_TP = num_TP_each_fold;
            tmp_iter_num_test_FP = num_FP_each_fold;
        end

    elseif data_div_option == 4 

        tmp_iter_Xtest = [X_TP_by_participant{i}; X_FP_by_participant{i}];
        tmp_iter_Ytest = zeros(size(tmp_iter_Xtest,1),1);
        tmp_iter_Ytest(1:size(X_TP_by_participant{i},1)) = 1;

        tmp_iter_Xtrain = [];
        tmp_iter_Ytrain = [];
        
        for j = 1 : size(participants, 2)
            if j ~= i
                tmp_iter_Xtrain = [tmp_iter_Xtrain; X_TP_by_participant{j}; X_FP_by_participant{j}];
                tmp_iter_Ytrain = [tmp_iter_Ytrain; ones(size(X_TP_by_participant{j},1), 1); zeros(size(X_FP_by_participant{j},1), 1)];
            end
        end

        tmp_iter_num_test_TP = size(X_TP_by_participant{i},1);
        tmp_iter_num_test_FP = size(X_FP_by_participant{i},1);
    else
        fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', data_div_option);
    end

    % dimensionality reduction: PCA
    if dim_reduction_option == 1
        
        % Generating PCA model from training data set
        tmp_iter_sigma = (tmp_iter_Xtrain' * tmp_iter_Xtrain) / size(tmp_iter_Xtrain, 1);
        [U_mtx{i}, tmp_iter_signal, ~] = svd(tmp_iter_sigma);
        % [U_mtx{i}, tmp_iter_signal] = ML_run_PCA(tmp_iter_Xtrain);

        % retain 99% variance in the data
        variance = 0.99; 

        for P = 1 : length(tmp_iter_signal)

            tmp_S_P = tmp_iter_signal(1:P,:);
            
            if (sum(tmp_S_P,'all')/sum(tmp_iter_signal,'all')) >= variance
                
                % number of principal components
                num_pca_comp(i) = P; 
                break;
            end
        end

        % Project the data onto n_PC dimensions
        % Z = projectData(X, U, K)
        % tmp_iter_Xtrain = ML_projectData(tmp_iter_Xtrain, U_mtx{i}, num_comp(i));
        % tmp_iter_Xtest  = ML_projectData(tmp_iter_Xtest, U_mtx{i}, num_comp(i));
        tmp_iter_Xtrain = tmp_iter_Xtrain * U_mtx{i}(:, 1:num_pca_comp(i));
        tmp_iter_Xtest  = tmp_iter_Xtest * U_mtx{i}(:, 1:num_pca_comp(i));

        clear tmp_iter_sigma tmp_iter_signal tmp_S_P
    end

    % Model cross-validation: stratified K-fold partitioning on the training dataset
    rng default;
    cvp_P = cvpartition(tmp_iter_Ytrain, 'Kfold', 5);
    
    % Logistic regression
    if mdl_option == 1 
        
        rng default
        
        % Gets the cross-validated logistic regression model
        % Lambda is optimized through K-fold cross-validation
        % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
        LR_mdl{i} = fitclinear(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                               'Learner', 'logistic', 'Cost', cost_func,...
                               'OptimizeHyperparameters','Lambda', ...
                               'HyperparameterOptimizationOptions', ...
                               struct('AcquisitionFunctionName', ...
                                      'expected-improvement-plus', ...
                                      'CVPartition', cvp_P, ...
                                      'ShowPlots', false));

        % Prediction
        [accu_train(i), accu_test(i), ...
         accu_testTP(i), accu_testFP(i)]...
            = ML_get_prediction_accuracy(LR_mdl{i}, ...
                                         tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                         tmp_iter_Xtest, tmp_iter_Ytest, ...
                                         tmp_iter_num_test_TP, tmp_iter_num_test_FP);
        
    % SVM (Gausian kernel)
    elseif mdl_option == 2
        
        rng default;

        % Hyperparameters (BoxConstraint(C), and KernelScale(mu)) are determined based on a K-fold cross-validation.
        % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
        SVM_mdl{i} = fitcsvm(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                             'KernelFunction', 'rbf', 'Cost', cost_func, ...
                             'OptimizeHyperparameters','auto', ...
                             'HyperparameterOptimizationOptions', ... 
                             struct('AcquisitionFunctionName', ...
                                    'expected-improvement-plus', ...
                                    'CVPartition', cvp_P, ...
                                    'ShowPlots', false));
        
        % Prediction using the selected model
        [accu_train(i), accu_test(i), ... 
         accu_testTP(i),accu_testFP(i)]...
            = ML_get_prediction_accuracy(SVM_mdl{i}, ...
                                         tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                         tmp_iter_Xtest, tmp_iter_Ytest, ...
                                         tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    % Neural Network
    elseif mdl_option == 3

        rng default;

        % Sigmoid activation function is used.
        % Hyperparameters (lambda, and layer size) are determined based on a K-fold cross-validation.
        % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
        NN_mdl{i} = fitcnet(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                            'Activation', 'sigmoid', ...
                            'OptimizeHyperparameters', {'Lambda', 'LayerSizes'}, ...
                            'HyperparameterOptimizationOptions', ...
                            struct('AcquisitionFunctionName', ...
                                   'expected-improvement-plus', ...
                                   'CVPartition', cvp_P, ...
                                   'ShowPlots', false));
        
        % Prediction using the selected model
        [accu_train(i), accu_test(i), ...
         accu_testTP(i), accu_testFP(i)]...
            = ML_get_prediction_accuracy(NN_mdl{i}, ...
                                         tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                         tmp_iter_Xtest, tmp_iter_Ytest, ...
                                         tmp_iter_num_test_TP, tmp_iter_num_test_FP);
        
    % Random forest
    elseif mdl_option == 4 

        rng default;

        % Random forest algorithm is used for tree ensemble.
        % Minimum size of the leaf node is used as the stopping criterion.
        % Number of features to sample in each tree are determined based on a K-fold cross-validation.
        % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
        tmp_tt = templateTree('Type', 'classification', 'Reproducible', true, 'MinLeafSize', 50);
        RF_mdl{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                 'Cost', cost_func, ...
                                 'Learners', tmp_tt, ...
                                 'Replace', 'on', ...
                                 'Resample', 'on', ...
                                 'Fresample', 1, ...
                                 'NumLearningCycles', 100,...
                                 'Method', 'Bag', ...
                                 'OptimizeHyperparameters', {'NumVariablesToSample'}, ...
                                 'HyperparameterOptimizationOptions',...
                                 struct('AcquisitionFunctionName', ...
                                        'expected-improvement-plus', ...
                                        'CVPartition', cvp_P, ...
                                        'ShowPlots', false));
        
        % Prediction using the selected model
        [accu_train(i), accu_test(i), ...
         accu_testTP(i), accu_testFP(i)]...
            = ML_get_prediction_accuracy(RF_mdl{i}, ...
                                         tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                         tmp_iter_Xtest, tmp_iter_Ytest, ...
                                         tmp_iter_num_test_TP, tmp_iter_num_test_FP);

        clear tmp_tt
    end

    % One iteration only (break after 1st loop)
    if cv_option == 0 
        break; 
    end

end

% the best model: the maximum test accuracy
[bestMdl_val, bestMdl_idx] = max(accu_test); 

%% The model with the optimal hyperparameters
final_test_pred = cell (1, num_iter);
final_test_scores = cell (1, num_iter);
final_train_accu = zeros(1, num_iter);
final_test_accu = zeros(1, num_iter);
final_testTP_accu = zeros(1, num_iter);
final_testFP_accu = zeros(1, num_iter);

% The number of principal components
num_pca_comp = zeros(1, num_iter); 
U_mtx = cell(1, num_iter);

% ML model selection
if mdl_option == 1
    model_str = 'Logistic Regression';
    model_optimised_LR = cell (1, num_iter);
elseif mdl_option == 2
    model_str = 'Support Vector Machine';
    model_optimised_SVM = cell (1, num_iter);
elseif mdl_option == 3
    model_str = 'Neural Network';
    model_optimised_NN = cell (1, num_iter);
elseif mdl_option == 4
    model_str = 'Random Forest';
    model_optimised_RF = cell (1, num_iter);
end

% Iteration
for i = 1 : num_iter
    
    fprintf('The optimised model - %s > current iteration: %d / %d ... \n', model_str, i, num_iter);

    % training and test data sets
    if data_div_option == 1
        
        tmp_iter_Xtrain = X_train;
        tmp_iter_Ytrain = Y_train;

        tmp_iter_Xtest = X_test;
        tmp_iter_Ytest = Y_test;

        tmp_iter_num_test_TP = num_test_TP;
        tmp_iter_num_test_FP = num_test_FP;

    elseif (data_div_option == 2) || (data_div_option == 3)

        % Partitioning data for K x K nested cross-validation
        % the i-th fold as the test data and rest of the folds as the training data
        tmp_iter_Xtest = X_Kfold{i};
        tmp_iter_Ytest = Y_Kfold{i};

        tmp_iter_Xtrain = [];
        tmp_iter_Ytrain = [];

        for j = 1 : K
            if j ~= i
                tmp_iter_Xtrain = [tmp_iter_Xtrain; X_Kfold{j}];
                tmp_iter_Ytrain = [tmp_iter_Ytrain; Y_Kfold{j}];
            end
        end

        if i == num_iter
            tmp_iter_num_test_TP = num_TP_last_fold;
            tmp_iter_num_test_FP = num_FP_last_fold;
        else
            tmp_iter_num_test_TP = num_TP_each_fold;
            tmp_iter_num_test_FP = num_FP_each_fold;
        end

    elseif data_div_option == 4

        tmp_iter_Xtest = [X_TP_by_participant{i}; X_FP_by_participant{i}];
        tmp_iter_Ytest = zeros(size(tmp_iter_Xtest,1),1);
        tmp_iter_Ytest(1 : size(X_TP_by_participant{i},1)) = 1;

        tmp_iter_Xtrain = [];
        tmp_iter_Ytrain = [];

        for j = 1:size(participants, 2)
            if j ~= i
                tmp_iter_Xtrain = [tmp_iter_Xtrain; X_TP_by_participant{j}; X_FP_by_participant{j}];
                tmp_iter_Ytrain = [tmp_iter_Ytrain; ones(size(X_TP_by_participant{j},1), 1); zeros(size(X_FP_by_participant{j},1), 1)];
            end
        end

        tmp_iter_num_test_TP = size(X_TP_by_participant{i}, 1);
        tmp_iter_num_test_FP = size(X_FP_by_participant{i}, 1);
    else
        fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', data_div_option);
    end

    % PCA for dimensionality reduction
    if dim_reduction_option == 1

        % Generating PCA model from training data set
        tmp_iter_sigma = (tmp_iter_Xtrain' * tmp_iter_Xtrain) / size(tmp_iter_Xtrain, 1);
        [U_mtx{i}, tmp_iter_signal, ~] = svd(tmp_iter_sigma);
        % [U_mtx{i}, tmp_iter_signal] = ML_run_PCA(tmp_iter_Xtrain);

        % retain 99% variance in the data
        variance = 0.99;
        
        for P = 1 : length(tmp_signal)

            tmp_S_P = tmp_signal(1:P, :);
            
            if (sum(tmp_S_P, 'all') / sum(tmp_signal, 'all')) >= variance

                % number of principal components
                num_pca_comp(i) = P; 
                break;
            end
        end

        % Project the data onto n_PC dimensions
        % Z = projectData(X, U, K)
        % tmp_iter_Xtrain = ML_projectData(tmp_iter_Xtrain, U_mtx{i}, num_comp(i));
        % tmp_iter_Xtest  = ML_projectData(tmp_iter_Xtest, U_mtx{i}, num_comp(i));
        tmp_iter_Xtrain = tmp_iter_Xtrain * U_mtx{i}(:, 1:num_pca_comp(i));
        tmp_iter_Xtest  = tmp_iter_Xtest * U_mtx{i}(:, 1:num_pca_comp(i));

        clear tmp_iter_sigma tmp_iter_signal tmp_S_P
    end

    % Training the model (optimised selection)
    % Logistic regression
    if mdl_option == 1 

        % lambda with lowest test error
        lambda_best = LR_mdl{bestMdl_idx}.ModelParameters.Lambda; 
        model_optimised_LR{i} = fitclinear(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                           'Learner', 'logistic', ...
                                           'Lambda', lambda_best, ...
                                           'Cost', cost_func);

        % Prediction
        [final_train_accu(i), final_test_accu(i), ... 
         final_testTP_accu(i), final_testFP_accu(i)] ...
            = ML_get_prediction_accuracy(model_optimised_LR{i}, ...
                                      tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                      tmp_iter_Xtest, tmp_iter_Ytest, ...
                                      tmp_iter_num_test_TP, tmp_iter_num_test_FP);
        [final_test_pred{i}, final_test_scores{i}] = predict(model_optimised_LR{i}, tmp_iter_Xtest);
        
    elseif mdl_option == 2

        tmp_svmC = SVM_mdl{bestMdl_idx}.ModelParameters.BoxConstraint; 
        tmp_svmSigma = SVM_mdl{bestMdl_idx}.ModelParameters.KernelScale;

        rng default;
        model_optimised_SVM{i} = fitcsvm(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                         'KernelFunction', 'rbf',...
                                         'Cost', cost_func, ...
                                         'BoxConstraint', tmp_svmC, ...
                                         'KernelScale', tmp_svmSigma);
        
        % Prediction
        [final_train_accu(i), final_test_accu(i), ...
         final_testTP_accu(i), final_testFP_accu(i)] ...
            = ML_get_prediction_accuracy(model_optimised_SVM{i}, ...
                                         tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                         tmp_iter_Xtest, tmp_iter_Ytest, ...
                                         tmp_iter_num_test_TP, tmp_iter_num_test_FP);

        [final_test_pred{i}, final_test_scores{i}] = predict(model_optimised_SVM{i}, tmp_iter_Xtest);
        
        clear tmp_svmC tmp_svmSigma

    elseif mdl_option == 3
        
        tmp_NN_layerSize = NN_mdl{bestMdl_idx}.ModelParameters.LayerSizes;
        tmp_NN_lambda = NN_mdl{bestMdl_idx}.ModelParameters.Lambda;

        rng default;
        model_optimised_NN{i} = fitcnet(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                        'LayerSizes', tmp_NN_layerSize, ...
                                        'Activation', 'sigmoid', ...
                                        'Lambda', tmp_NN_lambda);
        
        % Prediction
        [final_train_accu(i), final_test_accu(i), ...
         final_testTP_accu(i), final_testFP_accu(i)] ...
            = ML_get_prediction_accuracy(model_optimised_NN{i}, ...
                                         tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                         tmp_iter_Xtest, tmp_iter_Ytest, ...
                                         tmp_iter_num_test_TP, tmp_iter_num_test_FP);

        [final_test_pred{i}, final_test_scores{i}] = predict(model_optimised_NN{i}, tmp_iter_Xtest);
        
        clear tmp_NN_layerSize tmp_NN_lambda

    elseif mdl_option == 4
        
        % RF_NLC = RF_model{I_max}.ModelParameters.NLearn; % Number of trees in the ensemble
        % NumVariablesToSample to sample and MinLeafSize are included here
        tmp_RF_template = RF_mdl{bestMdl_idx}.ModelParameters.LearnerTemplates{1}; 
        tmp_RF_cycles = 100;
        tmp_RF_fSam = 1;

        rng default; 
        model_optimised_RF{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                             'Cost', cost_func, ...
                                             'Learners', tmp_RF_template, ...
                                             'NumLearningCycles', tmp_RF_cycles, ...
                                             'Replace', 'on', ...
                                             'Resample', 'on', ...
                                             'Fresample', tmp_RF_fSam, ...
                                             'Method', 'Bag');

        % Prediction using the model
        [final_train_accu(i),final_test_accu(i),final_testTP_accu(i),...
            final_testFP_accu(i)] = ML_get_prediction_accuracy(model_optimised_RF{i},tmp_iter_Xtrain,...
            tmp_iter_Ytrain,tmp_iter_Xtest,tmp_iter_Ytest,tmp_iter_num_test_TP,tmp_iter_num_test_FP);

        [final_test_pred{i},final_test_scores{i}] = predict(model_optimised_RF{i}, tmp_iter_Xtest);
        
        clear tmp_RF_template tmp_RF_cycles tmp_RF_fSam
    end
end

% Statistics of the classification accuracies
train_accuAvg = mean(final_train_accu);
train_accuStd = std(final_train_accu);

test_accuAvg = mean(final_test_accu);
test_accuStd = std(final_test_accu);

test_accuAvg_TP = mean(final_testTP_accu);
test_accuStd_TP = std(final_testTP_accu); 
test_accuAvg_FP = mean(final_testFP_accu);
test_accuStd_FP = std(final_testFP_accu); 

% Selection of best trained model
[maxFinal_val,maxFinal_idx] = max(final_test_accu); 

% Classifier selection
% Logistic regression
if mdl_option == 1 
    bestIdx_LR = maxFinal_idx;
    accuracy_LR = [train_accuAvg, train_accuStd, test_accuAvg, test_accuStd, ...
                   test_accuAvg_TP, test_accuStd_TP, test_accuAvg_FP, test_accuStd_FP];
% SVM
elseif mdl_option == 2 
    bestIdx_SVM = maxFinal_idx;
    accuracy_SVM = [train_accuAvg, train_accuStd, test_accuAvg, test_accuStd, ...
                    test_accuAvg_TP, test_accuStd_TP, test_accuAvg_FP, test_accuStd_FP];
% Neural Network
elseif mdl_option == 3 
    bestIdx_NN = maxFinal_idx;
    accuracy_NN = [train_accuAvg, train_accuStd, test_accuAvg, test_accuStd, ...
                   test_accuAvg_TP, test_accuStd_TP, test_accuAvg_FP, test_accuStd_FP];
% Random Forest
elseif mdl_option == 4 
    bestIdx_RF = maxFinal_idx;
    accuracy_RF = [train_accuAvg, train_accuStd, test_accuAvg, test_accuStd, ...
                   test_accuAvg_TP, test_accuStd_TP, test_accuAvg_FP, test_accuStd_FP];
end

% Print related information in the console windows
fprintf('Data division option: %d ... \n', data_div_option);
fprintf('Cost of getting TP wrong: %.2f ... \n', cost_TP);
fprintf('Classifier option: %d ... \n', mdl_option);
fprintf('PCA dimension reduction: %d ... \n', dim_reduction_option);

fprintf('Training accuracy: %.3f (%.3f) ... \n', train_accuAvg, train_accuStd);
fprintf('Test accuracy: %.3f(%.3f) ... \n', test_accuAvg, test_accuStd);
fprintf('Test accuracy (True Positive): %.3f(%.3f) ... \n', test_accuAvg_TP, test_accuStd_TP);
fprintf('Test accuracy (Flase Positive): %.3f(%.3f) ... \n', test_accuAvg_FP, test_accuStd_FP);

% Saving the trained models and performance
% ... add later

%% Performance evaluation 
% 1: ROC & PR > on, 0: ROC & PR > off
switch_ROC = 0; 

if switch_ROC == 1
    num_div_ROC = 5;
    ROC_thresh = linspace(0.95, 1, num_div_ROC+1);

    % removes two ends (0 & 1)
    ROC_thresh = ROC_thresh(2 : end-1); 
    num_iter_ROC = num_div_ROC - 1;
else
    ROC_thresh = 0.5;
    num_iter_ROC = 1;
end

overall_detection_matrix = zeros(num_iter_ROC, 4);

for k = 1 : num_iter_ROC

    fprintf('Iterating ROC: %d / %d ... \n', k, num_iter_ROC)

    % Prediction for the overall dataset
    % 1: load the prediction from the current directory
    load_prediction_option = 1; 

    if load_prediction_option == 1
        prediction_overall_TPscore = load('Y_TPD_from_python.txt');
        prediction_overall_FPscore = load('Y_FPD_from_python.txt');
        prediction_overall_TP = prediction_overall_TPscore >= ROC_thresh(k);
        prediction_overall_FP = prediction_overall_FPscore >= ROC_thresh(k);
        prediction_overall = [prediction_overall_TP; prediction_overall_FP];
    else
        for j = 1 : length(final_test_pred)

            % Convert test scores to classification probabilities
            if mdl_option == 2
                test_scores = sigmoid(final_test_scores{j});
            else
                test_scores = final_test_scores{j};
            end
            
            % Calculate the final test prediction
            final_test_pred{j} = test_scores(:, 2) >= ROC_thresh(k);
        end

        % Prediction of overall dataset based on "data_div_option"
        % holdout method
        if data_div_option == 1

            % The trained classifier is applied on the whole dataset
            Z = X_norm; 
            
            % Logistic regression
            if mdl_option == 1 

                % PCA for dimensionality reduction
                if dim_reduction_option == 1
                    Z = ML_projectData(X_norm, U_mtx{bestIdx_LR}, num_pca_comp(bestIdx_LR)); 
                end
                prediction_overall = predict(model_optimised_LR{bestIdx_LR}, Z);

            % SVM
            elseif mdl_option == 2 
                if dim_reduction_option == 1
                    Z = ML_projectData(X_norm, U_mtx{bestIdx_SVM}, num_pca_comp(bestIdx_SVM));
                end
                prediction_overall = predict(model_optimised_SVM{bestIdx_SVM}, Z);

            % NN
            elseif mdl_option == 3 
                if dim_reduction_option == 1
                    Z = projectData(X_norm, U_mtx{bestIdx_NN}, num_pca_comp(bestIdx_NN));
                end
                prediction_overall = predict(model_optimised_NN{bestIdx_NN}, Z);
            
            % Random Forest
            elseif mdl_option == 4
                if dim_reduction_option == 1
                    Z = projectData(X_norm, U_mtx{bestIdx_RF}, num_pca_comp(bestIdx_RF));
                end
                prediction_overall = predict(model_optimised_RF{bestIdx_RF}, Z);
            end

            % Get 
            prediction_overall_TP = prediction_overall(1 : num_TP);
            prediction_overall_FP = prediction_overall(num_TP+1 : end);

        % stratified K-fold division
        elseif (data_div_option == 2) || (data_div_option == 3) 
            [prediction_overall_TP, prediction_overall_FP] = ...
                ML_get_overall_test_prediction(final_test_pred, ...
                                               num_TP, num_TP_each_fold, num_TP_last_fold, ...
                                               num_FP, num_FP_each_fold, num_FP_last_fold, ...
                                               num_iter, rand_num_TP, rand_num_FP);

            prediction_overall = [prediction_overall_TP; prediction_overall_FP];

        % division by participants
        elseif data_div_option == 4 

            prediction_overall_TP = []; 
            prediction_overall_FP = [];
            
            for i = 1 : size(participants, 2)
                prediction_overall_TP = [prediction_overall_TP; final_test_pred{i}(1 : size(X_TP_by_participant{i}, 1))];
                prediction_overall_FP = [prediction_overall_FP; final_test_pred{i}(size(X_TP_by_participant{i}, 1)+1 : end)];
            end

            prediction_overall = [prediction_overall_TP; prediction_overall_FP];

        else
            fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', data_div_option);
        end
    end

    % Accuracy of for the overall prediction
    % * Y is the same as non-randomized data
    accuracy_overall = sum((prediction_overall == Y_labels)) / length(Y_labels); 
    accuracy_overall_TP = sum((prediction_overall_TP == Y_labels(1 : num_TP))) / num_TP;
    accuracy_overall_FP = sum((prediction_overall_FP == Y_labels(num_TP+1 : end))) / num_FP;

    % Statistics: FM detections
    matching_idx_TP = 1;
    matching_idx_FP = 1;
    FM_segmented_all_ML = cell(num_dfiles, 1);

    for i = 1 : num_dfiles

        fprintf('Statistics: FM detections - current data file: %d / %d ... \n', i, num_dfiles);

        switch sensor_selection
            case 'accl'
                selected_data_preproc = {all_acceL_cat, all_acceR_cat};
                selected_data_proc = all_fusion1_accl_cat;
                selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat);
                selected_sensorT = 1;
                selected_sensorN = 2;
            case 'acou'
                selected_data_preproc = {all_acouL_cat, all_acouR_cat};
                selected_data_proc = all_fusion1_acou_cat;
                selected_thresh = cat(2, all_acouL_thresh_cat, all_acouR_thresh_cat);
                selected_sensorT = 1;
                selected_sensorN = 2;
            case 'piez'
                selected_data_preproc = {all_piezL_cat, all_piezR_cat};
                selected_data_proc = all_fusion1_piez_cat;
                selected_thresh = cat(2, all_piezL_thresh_cat, all_piezR_thresh_cat);
                selected_sensorT = 1;
                selected_sensorN = 2;
            case 'accl_acou'
                selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_acouL_cat, all_acouR_cat};
                selected_data_proc = all_fusion1_accl_acou_cat;
                selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat);
                selected_sensorT = 2;
                selected_sensorN = 4;
            case 'accl_piez'
                selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_piezL_cat, all_piezR_cat};
                selected_data_proc = all_fusion1_accl_piez_cat;
                selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
                selected_sensorT = 2;
                selected_sensorN = 4;
            case 'acou_piez'
                selected_data_preproc = {all_acouL_cat, all_acouR_cat, all_acouL_cat, all_acouR_cat};
                selected_data_proc = all_fusion1_acou_piez_cat;
                selected_thresh = cat(2, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
                selected_sensorT = 2;
                selected_sensorN = 4;
            case 'accl_acou_piez'
                selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_acouL_cat, all_acouR_cat, all_piezL_cat, all_piezR_cat};
                selected_data_proc = all_fusion1_accl_acou_piez_cat;
                selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
                selected_sensorT = 3;
                selected_sensorN = 6;
            otherwise
                disp('Selected FM sensor(s) is out of range ...');
        end
        
        selected_data_proc_labeled = bwlabel(selected_data_proc{1});
        selected_data_proc_numLabel = length(unique(selected_data_proc_labeled)) - 1; 
        
        % Mapping the detection by ML
        [FM_segmented_all_ML{i,:}, ... 
         matching_idx_TP, matching_idx_FP]...
            = ML_map_mlDetections(selected_data_proc_labeled, ...
                                  all_sensation_map_cat{i,:}, ...
                                  selected_data_proc_numLabel, ...
                                  prediction_overall_TP, ...
                                  prediction_overall_FP, ...
                                  matching_idx_TP, ...
                                  matching_idx_FP);
        
        % Matching with maternal sensation > ML_match_sensation()
        [TP_detection(i, :), ...
         FP_detection(i, :), ...
         TN_detection(i, :), ...
         FN_detection(i, :)] ...
            = ML_match_sensation(all_seg_cat{i,:}, ...
                                 all_sens1_cat{i,:}, ...
                                 all_IMUacce_map_cat{i,:}, ...
                                 all_sensation_map_cat{i,:}, ...
                                 ext_backward, ext_forward, ...
                                 FM_dilation_time, ...
                                 freq_sensor, freq_sensation);
    end

    % Performance analysis > ML_get_performance_params()
    [performance_sens, ...
     performance_ppv, ...
     performance_spec, ...
     performance_accu, ...
     performance_fscore, ...
     performance_fdr] ...
        = ML_get_performance_params(TP_detection, ...
                                    FP_detection, ...
                                    TN_detection, ...
                                    FN_detection);

    indv_detections = [TP_detection{1,1},FP_detection{1,1},TN_detection{1,1},FN_detection{1,1}];

    % Combine all the data files
    TP_detectionAll = sum(TP_detection, 1);
    FP_detectionAll = sum(FP_detection, 1);
    TN_detectionAll = sum(TN_detection, 1);
    FN_detectionAll = sum(FN_detection, 1);

    overall_detection_matrix(k,:) = [TP_detectionAll, FP_detectionAll, TN_detectionAll, FN_detectionAll];

    [performanceAll_sens, ...
     performanceAll_ppv, ...
     performanceAll_spec, ...
     performanceAll_accu, ...
     performanceAll_fscore, ...
     performanceAll_fdr] ...
        = get_performance_params(TP_detectionAll, ...
                                 FP_detectionAll, ...
                                 TN_detectionAll, ...
                                 FN_detectionAll);

    performanceAll_pabak = 2 * performanceAll_accu{1}-1;

    overall_performance_matrix = [performanceAll_sens, performanceAll_spec, ...
                                  performanceAll_accu, performanceAll_ppv, ...
                                  performanceAll_fscore, performanceAll_fdr, ...
                                  performanceAll_pabak];
end

% Display performance matrix
fprintf('ML model settings are as follows: ... \n');
fprintf(['Threshold multiplier: \n ' ...
                               '>> Accelerometer = %.00f ... \n' ...
                               '>> Acoustic = %.00f ... \n' ...
                               '>> Piezoelectric diaphragm = %.00f... \n'], ...
                               FM_min_SN(1), FM_min_SN(3), FM_min_SN(5));
fprintf('Data division option for training and testing: %d (1- holdout; 2,3- K-fold; 4-by participants) ... \n', data_div_option);
fprintf('FP/TP ratio in the training data set: %.2f ... \n', TP_FP_ratio);
fprintf('Cost of getting TP wrong: %.2f ... \n', cost_TP);
fprintf('Classifier: %d (1- LR, 2- SVM, 3-NN, 4- RF) ... \n', mdl_option);
fprintf('PCA: %d (1- on; 0- off) ... \n', dim_reduction_option);

fprintf(['Detection stats:\n ...' ...
                         ' >> ACC = %.3f, SEN = %.3f, SPE = %.3f ... \n'], ...
                         performanceAll_accu{1}, performanceAll_sens{1}, performanceAll_spec{1});
fprintf('PPV = %.3f, F1 score = %.3f, PABAK = %.3f ... \n', ...
        performanceAll_ppv{1}, performanceAll_fscore{1}, performanceAll_pabak);

% Clearing variable
% clear selected_data_preproc sensor_data_sgmntd ...
%       selected_data_proc...
%       selected_data_proc_labeled


%% Detection stats: biophysical profile 
% the number of FM movements
% the duration the active fetus (moving) in each data file
% the mean duration of FM in each data file
% the median of the interval between each onset of FM for each data file
num_movements = zeros(num_dfiles,1);
total_FM_duration = zeros(num_dfiles,1);
mean_FM_duration = zeros(num_dfiles,1);
median_onset_interval = zeros(num_dfiles,1);

% Detections within 2 seconds are treated as the same detection
FM_dilation_duplicate = 2; 

% * fwise: individual data file wise
for i = 1 : num_dfiles

    % pre-processed data
    selected_data_preproc_fwise = {all_acceL_cat{i,:}, all_acceR_cat{i,:}, ...
                                   all_acouL_cat{i,:}, all_acouR_cat{i,:}, ...
                                   all_piezL_cat{i,:}, all_piezR_cat{i,:}};
    
    % segmented data
    selected_data_proc_fwise = all_fusion1_accl_acou_piez_cat{i,:};

    % the detections by Machine Learning algorithm
    ML_detection_map_fwise = FM_segmented_all_ML{i}; 

    % Reduced, because of the new dilation length
    reduced_detection_map = ML_detection_map_fwise.*selected_data_proc_fwise; 
    reduced_detection_map_labeled = bwlabel(reduced_detection_map);
    num_movements(i, :) = max(reduced_detection_map_labeled);

    % Extract the postive detection
    reduced_detection_map_posi = reduced_detection_map(reduced_detection_map == 1); 
    
    % Dilation length in sides of each detection is removed and coverted in minutes
    total_FM_duration(i, :) = (length(reduced_detection_map_posi) / freq_sensor - num_movements(i, :) * FM_dilation_duplicate / 2) / 60;
    % total_FM_duration(i, :) = (length(reduced_detection_map_posi) / Fs_sensor) / 60;

    % mean duration of FM in each data file in seconds
    mean_FM_duration(i, :) = total_FM_duration(i) * 60 / num_movements(i, :); 
    % mean_FM_duration(i, :) = (length(reduced_detection_map_posi) / Fs_sensor) / num_movements(i, :);

    % interval between FM movements in seconds
    onset_interval = zeros(num_movements(i, :) - 1, 1);
    for j = 1 : (num_movements(i, :) - 1)
        tmp_onset1 = find(reduced_detection_map_labeled == j, 1); 
        tmp_onset2 = find(reduced_detection_map_labeled == j+1, 1);
        onset_interval(j, :) = (tmp_onset2 - tmp_onset1) / freq_sensor;
        clear tmp_onset1 tmp_onset2
    end

    % the median onset interval in seconds
    median_onset_interval(i, :) = median(onset_interval); 

    % the duration (trimmed) of each data file
    duration_dataT_fwise(i, :) = max(all_timeV_cat{i,:}) / 60;
end

% non-fetus duration in each data file
total_nonFM_duration = duration_dataT_fwise - total_FM_duration; 

% data table: the BPP parameters
T_BPP_para = table('Size', [num_dfiles 6], ...
                   'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
                   'VariableNames', {'Data file name', 'Total FM duration', 'Total non-FM duration', 'Mean FM duration', 'Median onset interval', 'No. of FM'});

for i = 1 : num_dfiles
    T_BPP_para{i, 1} = all_data_fname{i, :}; 
    T_BPP_para{i, 2} = num2str(total_FM_duration(i, :)); 
    T_BPP_para{i, 3} = num2str(total_nonFM_duration(i, :)); 
    T_BPP_para{i, 4} = num2str(mean_FM_duration(i, :)); 
    T_BPP_para{i, 5} = num2str(median_onset_interval(i, :)); 
    T_BPP_para{i, 6} = num2str(num_movements(i, :));
end

%% The median onset intervals based on the combined data sets 
n1 = 126;
n2 = 131;
onset_interval_all = 0;
for i = n1 : n2

    % the pre-processed and processed (segmented) data for the selected range
    selected_data_preproc_fwiseSub = {all_acceL_cat{i,:}, all_acceR_cat{i,:}, ...
                                      all_acouL_cat{i,:}, all_acouR_cat{i,:}, ...
                                      all_piezL_cat{i,:}, all_piezR_cat{i,:}};
    selected_data_proc_fwiseSub = all_fusion1_accl_acou_piez_cat{i,:};

    ML_detection_map_fwiseSub = FM_segmented_all_ML{i};
    reduced_detection_map = ML_detection_map_fwiseSub.*selected_data_proc_fwiseSub;
    reduced_detection_map_labeled = bwlabel(reduced_detection_map);
    num_movements (i)               = max(reduced_detection_map_labeled);

    onset_interval = zeros(num_movements(i)-1,1);
    for j = 1:(num_movements(i)-1)
        tmp_onset1 = find(reduced_detection_map_labeled == j,1); % Sample no. corresponding to start of the label
        tmp_onset2 = find(reduced_detection_map_labeled == j+1,1); % Sample no. corresponding to start of the next label

        onset_interval(j) = (tmp_onset2-tmp_onset1)/freq_sensor; % onset to onset interval in seconds
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
[X_train_LC,Y_train_LC,X_test_LC,Y_test_LC,num_training_TP,...
    num_training_FP] = divide_by_holdout(X_norm_TPD_ranked,X_norm_FPD_ranked,training_protion);

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
num_iter    = floor((num_training_TP+num_training_FP)/increment); % Number of iteration

LR_train_error_LC  = zeros(num_iter,1);
LR_test_error_LC   = zeros(num_iter,1);
SVM_train_error_LC = zeros(num_iter,1);
SVM_test_error_LC  = zeros(num_iter,1);
NN_train_error_LC  = zeros(num_iter,1);
NN_test_error_LC   = zeros(num_iter,1);
RF_train_error_LC  = zeros(num_iter,1);
RF_test_error_LC   = zeros(num_iter,1);

test_all_models = 0; % 1- all the models are tested simulteneously; 0- only 1 model is tested
mdl_option = 3;
load_NN_LC_result_option = 1;

for i = 1:num_iter
    % Generating training data set
    X_train_temp = X_train_LC(1:i*increment,:);
    Y_train_temp = Y_train_LC(1:i*increment,:);

    % Applying learning models
    if (test_all_models == 1)||(mdl_option == 2)
        % --------------------------------- SVM -------------------------------
        [SVM_Z_train_temp,SVM_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
            X_test_LC,dim_reduction_option,U_mtx,num_pca_comp,bestIdx_SVM);
        rng default; % sets random seed to default value. This is necessary for reproducibility.
        SVM_model_LC = fitcsvm(SVM_Z_train_temp,Y_train_temp,'KernelFunction','rbf','Cost', ...
            cost_func,'BoxConstraint',tmp_svmC,'KernelScale',tmp_svmSigma);

        % Evaluation of traning and testing errors
        SVM_train_prediction_temp = predict(SVM_model_LC, SVM_Z_train_temp);
        SVM_train_error_LC(i) = 1 - sum((SVM_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

        SVM_test_prediction_temp = predict(SVM_model_LC, SVM_Z_test_temp);
        SVM_test_error_LC(i) = 1- sum((SVM_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
        %
    end

    if (test_all_models == 1)||(mdl_option == 3)
        % --------------------------------- NN _-------------------------------
        if load_NN_LC_result_option == 0
            [NN_Z_train_temp,NN_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
                X_test_LC,dim_reduction_option,U_mtx,num_pca_comp,bestIdx_NN);
            rng default; % sets random seed to default value. This is necessary for reproducibility.
            NN_model_LC = fitcnet(NN_Z_train_temp,Y_train_temp,'LayerSizes', tmp_NN_layerSize,...
                'Activation','sigmoid','Lambda',tmp_NN_lambda);

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

    if (test_all_models == 1)||(mdl_option == 4)
        % --------------------------------- RF --------------------------------
        [RF_Z_train_temp,RF_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
            X_test_LC,dim_reduction_option,U_mtx,num_pca_comp,bestIdx_RF);
        rng default; % sets random seed to default value. This is necessary for reproducibility.
        RF_model_LC = fitcensemble(RF_Z_train_temp,Y_train_temp,'Cost', cost_func,...
            'Learners', t_selected,'Method','Bag','NumlearningCycles',100,'Resample','on','Replace','on','Fresample',1);

        % Evaluation of traning and testing errors
        RF_train_prediction_temp = predict(RF_model_LC, RF_Z_train_temp);
        RF_train_error_LC(i) = 1 - sum((RF_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

        RF_test_prediction_temp = predict(RF_model_LC, RF_Z_test_temp);
        RF_test_error_LC(i) = 1- sum((RF_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
        %
    end

    if (test_all_models == 1)||(mdl_option == 1)
        % ---------------------- Logistic regression --------------------------
        [LR_Z_train_temp,LR_Z_test_temp] = reduce_dimension_by_PCA(X_train_temp,...
            X_test_LC,dim_reduction_option,U_mtx,num_pca_comp,bestIdx_LR);
        rng default;
        LR_model_LC = fitclinear(LR_Z_train_temp, Y_train_temp,'Learner','logistic',...
            'Lambda', lambda_best,'Cost', cost_func);

        % Evaluation of traning and testing errors
        %     LR_train_prediction_temp = predict_LR(LR_theta_LC, LR_Z_train_temp);
        LR_train_prediction_temp = predict(LR_model_LC, LR_Z_train_temp);
        LR_train_error_LC(i) = 1 - sum((LR_train_prediction_temp == Y_train_temp))/length(Y_train_temp);

        %     LR_test_prediction_temp = predict_LR(LR_theta_LC, LR_Z_test_temp);
        LR_test_prediction_temp = predict(LR_model_LC, LR_Z_test_temp);
        LR_test_error_LC(i) = 1- sum((LR_test_prediction_temp == Y_test_LC))/length(Y_test_LC);
        %
    end

    fprintf('End of iteration: %.0f/%.0f\n',i,num_iter);
end

% Plotting the learning curve 
%   Plot settings
B_width = 3; % Width of the box
L_width = 4; % Width of the plot line
F_name = 'Times New Roman'; % Font type
F_size = 32; % Font size
y_limit = 40;

tiledlayout(2,2,'Padding','tight','TileSpacing','tight');
x_coordinate = (1:num_iter)*increment;

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
    fprintf('Current data file: %.0f/%.0f\n', i,num_dfiles)

    % --------- Segmentaiton of IMU data and creation of IMU_map ---------%
    % get_IMU_map() function is used here. It segments and dilates the IMU
    % data and returns the resultant data as IMU map. Settings for the
    % segmentation and dilation are given inside the function.
    %   Input variables:  IMU_data- A column vector of IMU data
    %                     data_file_names- a char variable with data file name
    %   Output variables: IMU_map- A column vector with segmented IMU data

    if num_dfiles == 1
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i},data_file_names,freq_sensor);
    else
        IMU_map{i} = get_IMU_map(IMU_data_fltd{i},data_file_names{i},freq_sensor);
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

    selected_data_preproc = {Aclm_data1_fltd{i}, Aclm_data2_fltd{i}, Acstc_data1_fltd{i}, Acstc_data2_fltd{i},  ...
        Pzplt_data1_fltd{i}, Pzplt_data2_fltd{i}};
    [sensor_data_sgmntd, ~] = get_segmented_data(selected_data_preproc, FM_min_SN, IMU_map{i}, FM_dilation_time, freq_sensor);

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
    selected_data_proc = {double(sensor_data_sgmntd{1} | sensor_data_sgmntd{2} | sensor_data_sgmntd{3} | ...
        sensor_data_sgmntd{4} | sensor_data_sgmntd{5} | sensor_data_sgmntd{6})};
    selected_data_proc_labeled = bwlabel(selected_data_proc{1});
    selected_data_proc_numLabel = length(unique(selected_data_proc_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    tmp_all_sensor = [sensor_data_sgmntd_Left_OR_Right_Acstc, sensor_data_sgmntd_Left_OR_Right_Aclm, sensor_data_sgmntd_Left_OR_Right_Pzplt];

    % Initialization of variables
    sensor_data_sgmntd_atleast_1_type{1} = zeros(length(sensor_data_sgmntd{1}),1);
    sensor_data_sgmntd_atleast_2_type{1} = zeros(length(sensor_data_sgmntd{1}),1);
    sensor_data_sgmntd_atleast_3_type{1} = zeros(length(sensor_data_sgmntd{1}),1);

    if (selected_data_proc_numLabel) % When there is a detection by the sensor system
        for k = 1 : selected_data_proc_numLabel
            tmp_idxS = find(selected_data_proc_labeled == k, 1 ); % Sample no. corresponding to the start of the label
            tmp_idxE = find(selected_data_proc_labeled == k, 1, 'last' ); % Sample no. corresponding to the end of the label

            indv_detection_map = zeros(length(sensor_data_sgmntd{1}),1); % Need to be initialized before every detection matching
            indv_detection_map(tmp_idxS:tmp_idxE) = 1; % mapping individual sensation data

            % For detection by at least n type of sensors
            tmp_ntype = 0; % Variable to hold number of common sensors for each detection
            for j = 1:selected_sensorN/2
                if (sum(indv_detection_map.*tmp_all_sensor{j})) % Non-zero value indicates intersection
                    tmp_ntype = tmp_ntype + 1;
                end
            end

            switch tmp_ntype
                case 1
                    sensor_data_sgmntd_atleast_1_type{1}(tmp_idxS:tmp_idxE) = 1;
                case 2
                    sensor_data_sgmntd_atleast_1_type{1}(tmp_idxS:tmp_idxE) = 1;
                    sensor_data_sgmntd_atleast_2_type{1}(tmp_idxS:tmp_idxE) = 1;
                case 3
                    sensor_data_sgmntd_atleast_1_type{1}(tmp_idxS:tmp_idxE) = 1;
                    sensor_data_sgmntd_atleast_2_type{1}(tmp_idxS:tmp_idxE) = 1;
                    sensor_data_sgmntd_atleast_3_type{1}(tmp_idxS:tmp_idxE) = 1;
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
        xlim_2 = (length(Acstc_data1_fltd{i})-1)/freq_sensor;
    end

    xlim_index1               = xlim_1*freq_sensor+1; % 1 is added because Matlab index starts from 1. If xlim_1 = 0, index = 1
    xlim_index2               = xlim_2*freq_sensor+1;
    sensorData_time_vector    = (xlim_index1-1:xlim_index2-1)/freq_sensor; % time vector for the data in SD1. 1 is deducted because time vector starts from 0 s.
    sensationData_time_vector = (xlim_index1-1:xlim_index2-1)/freq_sensation; % either of sensation_data_SD1 or SD2 could have been used here

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
    for j = 1:1:selected_sensorN
        nexttile
        plot(sensorData_time_vector, selected_data_preproc{j}(xlim_index1:xlim_index2),'LineWidth', L_width)
        hold on;
        %         plot(sensorData_time_vector, sensor_data_sgmntd{j}(xlim_index1:xlim_index2)...
        %             *max(sensor_data_fltd{j}(xlim_index1:xlim_index2))*segmentation_multiplier,'color', [0.4660 0.6740 0.1880], 'LineWidth', L_width)
        hold on;
        plot(sensation_only(:,1), sensation_only(:,2)*max(selected_data_preproc{j}(xlim_index1:xlim_index2))*m_sntn_multiplier, 'r*', 'LineWidth', L_width);
        hold off;
        ylim([min(selected_data_preproc{j}(xlim_index1:xlim_index2))*y_lim_multiplier, max(selected_data_preproc{j}(xlim_index1:xlim_index2))*y_lim_multiplier]);
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
    plot(sensorData_time_vector, FM_segmented_all_ML{i}(xlim_index1:xlim_index2)*scheme_multiplier,...
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












