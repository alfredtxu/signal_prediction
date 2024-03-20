% This script is for optimising the machine learning model on FM data
clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

% Parameter definition
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

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ******************************** section 1 ******************************
% ***************************** data preparation **************************
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    % segmented FM sensor suite & according thresholds
    all_seg{i,:} = sensor_suite_segmented;
    all_FM_thresh{i,:} = sensor_suite_thresh;

    % IMU and sensation maps
    all_IMUacce_map{i,:} = IMUacce_map;
    all_sensation_map{i,:} = sensation_map;
    all_sens_map_label{i,:} = sens_map_label;
    all_sens_label{i,:} = sens_label;

    fprintf('Loaded processed data ... %d (%d) - %s ... \n', i, size(sensor_fusion1_accl,1), tmp_mat);

    clear tmp_mat ...
          sensor_fusion1_accl sensor_fusion1_acou sensor_fusion1_piez ...
          sensor_fusion1_accl_acou sensor_fusion1_accl_piez sensor_fusion1_acou_piez ...
          sensor_fusion1_all ...
          sensor_suite_segmented sensor_suite_thresh ...
          IMUacce_map sensation_map sens_map_label sens_label
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
% selected_sensor_suite = 'accl';
% selected_sensor_suite = 'acou';
% selected_sensor_suite = 'piez';
% selected_sensor_suite = 'accl_acou';
% selected_sensor_suite = 'accl_piez';
selected_sensor_suite = 'acou_piez';
% selected_sensor_suite = 'accl_acou_piez';

switch selected_sensor_suite
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

selelcted_results_folder = ['D:\a_workplace\FMM_data_' selected_sensor_suite];
if ~isdir(selelcted_results_folder)    
    system(['mkdir ' selelcted_results_folder]);
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ******************************** section 1 ******************************
% ***************************** feature extraction ************************
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        else
            tmp_cFP = tmp_cFP + 1;
            tmp_FPextraction = zeros(tmp_idxE-tmp_idxS+1, selected_sensorN);

            for k = 1 : selected_sensorN
                tmp_data_preproc = selected_data_preproc{k};
                tmp_FPextraction(:, k) = tmp_data_preproc{i, :}(tmp_idxS:tmp_idxE);
                clear tmp_data_preproc
            end
            
            tmp_detection_FPc{tmp_cFP} = tmp_FPextraction;
        end

        clear tmp_idxS tmp_idxE tmp_mtx tmp_TPextraction tmp_FPextraction tmp_tp 
    end

    detection_TPc{i, :} = tmp_detection_TPc;
    detection_FPc{i, :} = tmp_detection_FPc;

    % sensation detection summary
    fprintf('Data file: %d - the number of labels is %d ... \n', i, tmp_numLab);

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

%% corrections: 
% Calculate correlation coefficients 
% rank-based measures - Spearman's or Kendall's tau 
% for any type of monotonic relationship) between variables. 
% rho_spearman = corr(X_norm, 'Type', 'Spearman');c
% rho_kendall = corr(X_norm, 'Type', 'Kendall');
% 
% rho_spearman_mu = mean(mean(rho_spearman, 1));
% rho_kendall_mu = mean(mean(rho_kendall, 1));

%% Feature ranking by Neighbourhood Component Analysis (NCA) 
% optimise hyper-parameter: lambda with K-fold cross-validation
% lambda values
lambdaV1 = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 0, 1, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5] / length(Y_labels); 
lambdaV2 = linspace(0, 20, 20) / length(Y_labels); 
lambdaV = cat(2, lambdaV1, lambdaV2);

% Gradient Tolerance values
GradientToleranceV = [1e-4, 1e-6];

% Sets random seed to default value. This is necessary for reproducibility.
rng default  

% Stratified K-fold division / holdout division
cvp_K = cvpartition(Y_labels,'kfold', 5); 
cvp_P = cvpartition(Y_labels,'HoldOut', 0.2);
cvp_L = cvpartition(Y_labels,'Leaveout');
cvp_R = cvpartition(Y_labels,'Resubstitution');

div_op = 1;

switch div_op
    case 1
        tmp_cvp = cvp_K;
    case 2
        tmp_cvp = cvp_P;
    case 3
        tmp_cvp = cvp_L;
    case 4
        tmp_cvp = cvp_R;
end

numtestsets = tmp_cvp.NumTestSets;
lossvalues = zeros(length(lambdaV),length(GradientToleranceV), numtestsets);

% optimization
for i = 1 : length(lambdaV)

    for g = 1 : length(GradientToleranceV)

        for j = 1 : numtestsets

            fprintf('Iterating... lambda values > %d; gradient tolerance > %d; NumTestSets > %d ... \n', ...
                    i, g, j);

            % Extract the training / test set from the partition object
            tmp_X_train = X_norm(tmp_cvp.training(j), :);
            tmp_Y_train = Y_labels(tmp_cvp.training(j), :);
            tmp_X_test = X_norm(tmp_cvp.test(j), :);
            tmp_Y_test = Y_labels(tmp_cvp.test(j), :);

            % Train an NCA model for classification
            tmp_ncaMdl = fscnca(tmp_X_train, tmp_Y_train, ...
                'FitMethod', 'exact', ...
                'Verbose', 1, ...
                'Solver', 'lbfgs', ...
                'Lambda', lambdaV(i), ...
                'IterationLimit', 1000, ...
                'GradientTolerance', GradientToleranceV(g));

            % Compute the classification loss for the test set using the nca model
            lossvalues(i, g, j) = loss(tmp_ncaMdl, tmp_X_test, tmp_Y_test, 'LossFunction', 'quadratic');

            clear tmp_X_train tmp_Y_train tmp_X_test tmp_Y_test tmp_ncaMdl
        end
    end
end

% mean of k-fold / iterations
tmp_muLoss = mean(lossvalues,3); 

% Plot lambda vs. loss
figure
plot(lambdaV, mean(tmp_muLoss, 2), 'ro-')
xlabel('Lambda values')
ylabel('Loss values')
grid on

% the index of minimum loass > the best lambda value.
[~, fscnca_bestLambdaIdx] = min(mean(tmp_muLoss, 2));
[~, fscnca_gradientToleranceIdx] = min(tmp_muLoss(fscnca_bestLambdaIdx,:)); 
fscnca_bestlambda = lambdaV(fscnca_bestLambdaIdx);
fscnca_bestGT = GradientToleranceV(fscnca_gradientToleranceIdx);
fscnca_bestlossvalues = tmp_muLoss(fscnca_bestLambdaIdx, fscnca_gradientToleranceIdx);

clear fscnca_bestLambdaIdx fscnca_gradientToleranceIdx tmp_cvp tmp_muLoss

% Use the selected lambda to optimise NCA model
% Stratified
% for all the sensors. 1.144441194796607e-04 from previous run; 1.202855e-04 from another run
% bestlambda = 0.000133548792051176; 
ncaMdl_final = fscnca(X_norm, Y_labels, ...
                'FitMethod', 'exact', ...
                'Verbose', 1, ...
                'Solver', 'lbfgs', ...
                'Lambda', fscnca_bestlambda, ...
                'GradientTolerance', fscnca_bestGT);

figure
semilogx(ncaMdl_final.FeatureWeights, 'ro')
xlabel('Feature index')
ylabel('Feature weight')   
grid on

figure
histogram(ncaMdl_final.FeatureWeights, length(ncaMdl_final.FeatureWeights))
grid on 

% Extract the feature ranking information
% Combines feature index and weights in a matrix
thresh_feature = 0.05;
feature_idx = (1 : size(X_norm,2))'; 
feature_ranking = [feature_idx, ncaMdl_final.FeatureWeights]; 
num_top_features = length(find(feature_ranking(:,2) >= thresh_feature));

[~,I_sort] = sort(feature_ranking(:,2), 'descend');
feature_rankingS = feature_ranking(I_sort,:); 

index_top_features = feature_rankingS(1:num_top_features,1);
X_norm_TPD_ranked = X_norm_TPD(:,index_top_features);
X_norm_FPD_ranked = X_norm_FPD(:,index_top_features);

time_stamp = datestr(now, 'yyyymmddHHMMSS');
cd(selelcted_results_folder);
save(['X_TPD_norm_ranked_' selected_sensor_suite '_' time_stamp '.txt'], 'X_norm_TPD_ranked', '-ASCII');
save(['X_FPD_norm_ranked_' selected_sensor_suite '_' time_stamp '.txt'], 'X_norm_FPD_ranked', '-ASCII');
cd(curr_dir)

% Summary for dimensionality reduction
fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('The the optimised feature reduction is as follows: \n');
fprintf('> Best Lambda : %d ... \n ', fscnca_bestlambda);
fprintf('> Best Gradient tolerance : %d ... \n ', fscnca_bestGT);
fprintf('> Best Loss value : %d ... \n ', fscnca_bestlossvalues);
fprintf('> Reduce features (of %d) : %d ... \n ', num_top_features, length(feature_idx));
fprintf('> TP vs FP : %d vs %d ... \n ', size(X_norm_TPD_ranked, 1), size(X_norm_FPD_ranked, 1));
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n');

% Save feature selection outcomes
cd(selelcted_results_folder);
save([selected_sensor_suite '_feature_selection_results.mat', 'X_features', 'Y_labels', ...
                        'X_norm', 'X_norm_TPD', 'X_norm_FPD', ... 
                        'X_zscore', 'X_zscore_TPD', 'X_zscore_FPD', ...
                        'lossvalues', 'fscnca_bestLambdaIdx', 'fscnca_gradientToleranceIdx', ...
                        'fscnca_bestlambda', 'fscnca_bestGT', 'fscnca_bestlossvalues', ...
                        'ncaMdl_final', 'index_top_features', 'X_norm_TPD_ranked', 'X_norm_FPD_ranked', ...
                        '-v7.3'])
cd(curr_dir)

% ----------------------------------------------------------------------------------------
% Release memory
% ----------------------------------------------------------------------------------------
%% clear unnecesary variables
clear all_acceL all_acceL_cat all_acceL_thresh all_acceL_thresh_cat ...
      all_acceR all_acceR_cat all_acceR_thresh all_acceR_thresh_cat ...
      all_acouL all_acouL_cat all_acouL_thresh all_acouL_thresh_cat ...
      all_acouR all_acouR_cat all_acouR_thresh all_acouR_thresh_cat ...
      all_piezL all_piezL_cat all_piezL_thresh all_piezL_thresh_cat ...
      all_piezR all_piezR_cat all_piezR_thresh all_piezR_thresh_cat   

clear all_FM_thresh all_forc all_forc_cat ...
      all_nfile ...
      all_IMUacce all_IMUacce_cat all_IMUacce_map all_IMUacce_map_cat ...
      all_seg all_seg_cat ...
      all_sens1 all_sens1_cat ...
      all_sens2 all_sens2_cat ...
      all_sens_label all_sens_label_cat all_sens_labelU_cat ...
      all_sens_map_label all_sens_map_label_cat ...
      all_sensation_map all_sensation_map_cat ...
      all_sensMtx all_sensMtx_cat ...
      all_timeV all_timeV_cat

clear all_fusion1_accl all_fusion1_accl_cat ...
      all_fusion1_acou all_fusion1_acou_cat ...      
      all_fusion1_piez all_fusion1_piez_cat ...
      all_fusion1_accl_acou all_fusion1_accl_acou_cat ...
      all_fusion1_accl_piez all_fusion1_accl_piez_cat ...
      all_fusion1_acou_piez all_fusion1_acou_piez_cat ...
      all_fusion1_accl_acou_piez all_fusion1_accl_acou_piez_cat

clear cvp_K cvp_P cvp_L cvp_R

clear feature_TP feature_FP X_features Y_labels ...
      X_zscore X_zscore_TPD X_zscore_FPD

% ----------------------------------------------------------------------------------------
% Training machine learning models
% ----------------------------------------------------------------------------------------
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
    
    num_iter = 1;

    [X_train, Y_train, X_test, Y_test, ...
     num_training_TP, num_training_FP, num_test_TP, num_test_FP] ...
     = ML_divide_by_holdout(X_norm_TPD_ranked, X_norm_FPD_ranked, 0.8);

elseif (data_div_option == 2) || (data_div_option == 3) 
    
    num_iter = 5;

    [X_Kfold, Y_Kfold, ...
     num_TP_each_fold, num_TP_last_fold,...
     num_FP_each_fold, num_FP_last_fold, ...
     TP_FP_ratio, ...
     rand_num_TP, rand_num_FP] ...
     = ML_divide_by_Kfolds(X_norm_TPD_ranked, X_norm_FPD_ranked, data_div_option, num_iter);

% Dividing data by participant    
elseif data_div_option == 4 

    num_iter = size(participants, 2);

    [X_TP_by_participant, X_FP_by_participant] = ...
        ML_divide_by_participants(all_nfile, detection_TPc, detection_FPc, ...
                                  X_norm_TPD_ranked, X_norm_FPD_ranked);
else
    num_iter = 0;
    fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', data_div_option);
end

%% Cross-validation to find the optimum hyperparameters
fprintf('Optimising hyperparameters through the process of cross-validation ...\n');

% Parameters for PCA
% > Number of principal components
% > U matrix in PCA
num_pca_comp = zeros(1, num_iter);
U_mtx = cell(1, num_iter); 

train_accu_LR = zeros(1, num_iter);
test_accu_LR = zeros(1, num_iter);
testTP_accu_LR = zeros(1, num_iter);
testFP_accu_LR = zeros(1, num_iter);

train_accu_SVM = zeros(1, num_iter);
test_accu_SVM = zeros(1, num_iter);
testTP_accu_SVM = zeros(1, num_iter);
testFP_accu_SVM = zeros(1, num_iter);

train_accu_RF = zeros(size(rf_method, 2), num_iter);
test_accu_RF = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF = zeros(size(rf_method, 2), num_iter);

train_accu_DA = zeros(1, num_iter);
test_accu_DA = zeros(1, num_iter);
testTP_accu_DA = zeros(1, num_iter);
testFP_accu_DA = zeros(1, num_iter);

train_accu_SNN = zeros(1, num_iter);
test_accu_SNN = zeros(1, num_iter);
testTP_accu_SNN = zeros(1, num_iter);
testFP_accu_SNN = zeros(1, num_iter);
% --------------------------------------------------------------------

% Higher waitage for TPD 
cost_FPD = round(size(X_norm_FPD, 1) / size(X_norm_TPD, 1), 1); 
% cost_TP = 2;

% Defines the waitage for getting wrong
cost_func = [0, 1; cost_FPD, 0]; 

% A range of classification models
% the number of learning models
num_mdl = 5;

% 'Generalized Additive Model', ...
mdl_str = {'Logistic Regression', ...
           'Support Vector Machine', ...
           'Random Forest', ...
           'Discriminant Analysis', ...
           'Shallow Neural Network'};

mdl_LR = cell(1, num_iter); % logistic regress 
mdl_SVM = cell(1, num_iter); % support vector machine
mdl_RF = cell(1, num_iter); % decision tree ensemble (random forest)
mdl_DA = cell(1, num_iter); % discriminant analysis
% mdl_GAM = cell(1, num_iter); % generalized additive model 
mdl_SNN = cell(1, num_iter); % shallow neural network

dim_reduction_option = 0; % Dimensionality reduction by PCA: 1 > on, 0 > off.
cv_option = 1; % Cross-validation: 1 > K x K-fold; 0 . K-fold

% 'RUSBoost' - Err - Sampling observations and boosting by majority undersampling at the same time not supported.
rf_method = {'Bag', ...
             'AdaBoostM1','GentleBoost','LogitBoost','RobustBoost', ...
             'LPBoost','TotalBoost'};
% rf_tree = templateTree('Type', 'classification', 'Reproducible', true, 'MinLeafSize', 50);
rf_tree = templateTree('Type', 'classification', 'Reproducible', true, 'Surrogate','on');

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

        for j = 1 : num_iter
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
        % [U_mtx{i}, tmp_iter_signal] = ML_runPCA(tmp_iter_Xtrain);

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
    cvp_mdl = cvpartition(tmp_iter_Ytrain, 'Kfold', 5);
       
    % Gets the cross-validated logistic regression model
    % Lambda is optimized through K-fold cross-validation
    % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
    mdl_LR{i} = fitclinear(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                            'Learner', 'logistic', ...
                            'Cost', cost_func,...
                            'OptimizeHyperparameters','Lambda', ...
                            'HyperparameterOptimizationOptions', ...
                            struct('AcquisitionFunctionName', ...
                            'expected-improvement-plus', ...
                            'CVPartition', cvp_mdl, ...
                            'ShowPlots', false));
    
    % Prediction
    [train_accu_LR(i), test_accu_LR(i), ...
     testTP_accu_LR(i), testFP_accu_LR(i)]...
        = ML_get_prediction_accuracy(mdl_LR{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);

    fprintf('Model Validation Iter %d > LR: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_LR(i), test_accu_LR(i), testTP_accu_LR(i), testFP_accu_LR(i));
    % ----------------------------------------------------------------------------------

    % Hyperparameters (BoxConstraint(C), and KernelScale(mu)) are determined based on a K-fold cross-validation.
    % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
    mdl_SVM{i} = fitcsvm(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
        'KernelFunction', 'rbf', 'Cost', cost_func, ...
        'OptimizeHyperparameters','auto', ...
        'HyperparameterOptimizationOptions', ...
        struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', ...
        'CVPartition', cvp_mdl, ...
        'ShowPlots', false));

    % Prediction using the selected model
    [train_accu_SVM(i), test_accu_SVM(i), ...
        testTP_accu_SVM(i), testFP_accu_SVM(i)]...
        = ML_get_prediction_accuracy(mdl_SVM{i}, ...
        tmp_iter_Xtrain, tmp_iter_Ytrain, ...
        tmp_iter_Xtest, tmp_iter_Ytest, ...
        tmp_iter_num_test_TP, tmp_iter_num_test_FP);

    fprintf('Model Validation Iter %d> SVM: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_SVM(i), test_accu_SVM(i), testTP_accu_SVM(i), testFP_accu_SVM(i));
    % ----------------------------------------------------------------------------------    
    
    % Random forest algorithm is used for tree ensemble.
    % Minimum size of the leaf node is used as the stopping criterion.
    % Number of features to sample in each tree are determined based on a K-fold cross-validation.
    % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
    for rf = 1 : size(rf_method, 2)

        if rf == 1 || rf == 5 || rf == 6 || rf == 7
            mdl_RF{i, rf} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                'Method', rf_method{rf}, ...
                'Cost', cost_func, ...
                'Learners', rf_tree, ...
                'Replace', 'on', ...
                'Resample', 'on', ...
                'FResample', 1, ...
                'OptimizeHyperparameters', {'NumVariablesToSample', 'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits'}, ...
                'HyperparameterOptimizationOptions',...
                struct('AcquisitionFunctionName', ...
                'expected-improvement-plus', ...
                'CVPartition', cvp_mdl, ...
                'ShowPlots', false));
        else
            mdl_RF{i, rf} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                'Method', rf_method{rf}, ...
                'Cost', cost_func, ...
                'Learners', rf_tree, ...
                'Replace', 'on', ...
                'Resample', 'on', ...
                'FResample', 1, ...
                'OptimizeHyperparameters', {'NumVariablesToSample', 'NumLearningCycles', 'LearnRate', 'MinLeafSize', 'MaxNumSplits'}, ...
                'HyperparameterOptimizationOptions',...
                struct('AcquisitionFunctionName', ...
                'expected-improvement-plus', ...
                'CVPartition', cvp_mdl, ...
                'ShowPlots', false));
        end

        % Prediction using the selected model
        [train_accu_RF(i, rf), test_accu_RF(i, rf), ...
         testTP_accu_RF(i, rf), testFP_accu_RF(i, rf)]...
            = ML_get_prediction_accuracy(mdl_RF{i, rf}, ...
            tmp_iter_Xtrain, tmp_iter_Ytrain,...
            tmp_iter_Xtest, tmp_iter_Ytest, ...
            tmp_iter_num_test_TP, tmp_iter_num_test_FP);

        fprintf('Model Validation Iter %d> RF (%s): train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
                i, rf_method{rf}, train_accu_RF(i, rf), test_accu_RF(i, rf), testTP_accu_RF(i, rf), testFP_accu_RF(i, rf));
    end
    % ----------------------------------------------------------------------------------    

    % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
    mdl_DA{i} = fitcdiscr(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
        'Cost', cost_func, ...
        'ScoreTransform', 'logit', ...
        'OptimizeHyperparameters', 'all', ...
        'HyperparameterOptimizationOptions',...
        struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', ...
        'CVPartition', cvp_mdl, ...
        'ShowPlots', false));

    % Prediction using the selected model
    [train_accu_DA(i), test_accu_DA(i), ...
        testTP_accu_DA(i), testFP_accu_DA(i)]...
        = ML_get_prediction_accuracy(mdl_DA{i}, ...
        tmp_iter_Xtrain, tmp_iter_Ytrain,...
        tmp_iter_Xtest, tmp_iter_Ytest, ...
        tmp_iter_num_test_TP, tmp_iter_num_test_FP);

    fprintf('Model Validation Iter %d> DA: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_DA(i), test_accu_DA(i), testTP_accu_DA(i), testFP_accu_DA(i));
    % ----------------------------------------------------------------------------------    
        
    % Sigmoid activation function is used.
    % Hyperparameters (lambda, and layer size) are determined based on a K-fold cross-validation.
    % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
    mdl_SNN{i} = fitcnet(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
        'OptimizeHyperparameters', 'all', ...
        'HyperparameterOptimizationOptions', ...
        struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', ...
        'CVPartition', cvp_mdl, ...
        'ShowPlots', false));

    tmp_iteration = mdl_SNN{i}.TrainingHistory.Iteration;
    tmp_trainLosses = mdl_SNN{i}.TrainingHistory.TrainingLoss;
    tmp_valLosses = mdl_SNN{i}.TrainingHistory.ValidationLoss;
    
    figure
    plot(tmp_iteration, tmp_trainLosses, tmp_iteration, tmp_valLosses)
    legend(["Training", "Validation"])
    xlabel("Iteration")
    ylabel("Cross-Entropy Loss")

    % Prediction using the selected model
    [train_accu_SNN(i), test_accu_SNN(i), ...
     testTP_accu_SNN(i), testFP_accu_SNN(i)]...
        = ML_get_prediction_accuracy(mdl_SNN{i}, ...
        tmp_iter_Xtrain, tmp_iter_Ytrain,...
        tmp_iter_Xtest, tmp_iter_Ytest, ...
        tmp_iter_num_test_TP, tmp_iter_num_test_FP);

    clear tmp_iteration tmp_trainLosses tmp_valLosses

    fprintf('Model Validation Iter %d> SNN: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_SNN(i), test_accu_SNN(i), testTP_accu_SNN(i), testFP_accu_SNN(i));
    % ----------------------------------------------------------------------------------    

    clear tmp_iter_Xtrain tmp_iter_Ytrain tmp_iter_Xtest tmp_iter_Ytest tmp_iter_num_test_TP tmp_iter_num_test_FP

% end of the loop for CV iterations
end

% the best model: the maximum test accuracy
[best_cvMdl_LR_val, best_cvMdl_LR_idx] = max(test_accu_LR); 
[best_cvMdl_SVM_val, best_cvMdl_SVM_idx] = max(test_accu_SVM); 
[best_cvMdl_RF1_val, best_cvMdl_RF1_idx] = max(test_accu_RF(:,1)); 
[best_cvMdl_RF2_val, best_cvMdl_RF2_idx] = max(test_accu_RF(:,2)); 
[best_cvMdl_RF3_val, best_cvMdl_RF3_idx] = max(test_accu_RF(:,3)); 
[best_cvMdl_RF4_val, best_cvMdl_RF4_idx] = max(test_accu_RF(:,4)); 
[best_cvMdl_RF5_val, best_cvMdl_RF5_idx] = max(test_accu_RF(:,4)); 
[best_cvMdl_RF6_val, best_cvMdl_RF6_idx] = max(test_accu_RF(:,5)); 
[best_cvMdl_RF7_val, best_cvMdl_RF7_idx] = max(test_accu_RF(:,7)); 
[best_cvMdl_DA_val, best_cvMdl_DA_idx] = max(test_accu_DA); 
% [best_cvMdl_GAM_val, best_cvMdl_GAM_idx] = max(test_accu_GAM); 
[best_cvMdl_SNN_val, best_cvMdl_SNN_idx] = max(test_accu_SNN); 

fprintf('Best  performance LR - test accu: %d ... \n', best_cvMdl_LR_val);
fprintf('Best  performance SVM - test accu: %d ... \n', best_cvMdl_SVM_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{1}, best_cvMdl_RF1_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{2}, best_cvMdl_RF2_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{3}, best_cvMdl_RF3_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{4}, best_cvMdl_RF4_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{5}, best_cvMdl_RF5_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{6}, best_cvMdl_RF6_val);
fprintf('Best  performance RF (%s) - test accu: %d ... \n', rf_method{7}, best_cvMdl_RF7_val);
fprintf('Best  performance LR - test accu: %d ... \n', best_cvMdl_DA_val);
fprintf('Best  performance LR - test accu: %d ... \n', best_cvMdl_SNN_val);
fprintf(' ~~~~~~~~~~~~~~~~~~~~~~~ CV completed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n');

% Save ML models - cross validation
cd(selelcted_results_folder);
save([selected_sensor_suite '_mdl_CV.mat'], 'mdl_LR', 'mdl_SVM','mdl_RF', 'mdl_DA', 'mdl_SNN', ...
                                            'train_accu_LR', 'test_accu_LR', 'testTP_accu_LR', 'testFP_accu_LR', ...
                                            'train_accu_SVM', 'test_accu_SVM', 'testTP_accu_SVM', 'testFP_accu_SVM', ...
                                            'train_accu_RF', 'test_accu_RF', 'testTP_accu_RF', 'testFP_accu_RF', ...
                                            'train_accu_DA', 'test_accu_DA', 'testTP_accu_DA', 'testFP_accu_DA', ...
                                            'train_accu_SNN', 'test_accu_SNN', 'testTP_accu_SNN', 'testFP_accu_SNN', ...
                                            '-v7.3');
cd(curr_dir)

% ------------------------------------------------------------------
%% The model with the optimal hyperparameters
% ------------------------------------------------------------------
train_accu_LR_final = zeros(1, num_iter);
test_accu_LR_final = zeros(1, num_iter);
testTP_accu_LR_final = zeros(1, num_iter);
testFP_accu_LR_final = zeros(1, num_iter);
test_pred_LR_final = cell(1, num_iter);
test_scores_LR_final = cell(1, num_iter);

train_accu_SVM_final = zeros(1, num_iter);
test_accu_SVM_final = zeros(1, num_iter);
testTP_accu_SVM_final = zeros(1, num_iter);
testFP_accu_SVM_final = zeros(1, num_iter);
test_pred_SVM_final = cell(1, num_iter);
test_scores_SVM_final = cell(1, num_iter);

train_accu_RF1_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF1_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF1_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF1_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF1_final = cell(size(rf_method, 2), num_iter);
test_scores_RF1_final = cell(size(rf_method, 2), num_iter);

train_accu_RF2_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF2_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF2_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF2_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF2_final = cell(size(rf_method, 2), num_iter);
test_scores_RF2_final = cell(size(rf_method, 2), num_iter);

train_accu_RF3_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF3_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF3_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF3_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF3_final = cell(size(rf_method, 2), num_iter);
test_scores_RF3_final = cell(size(rf_method, 2), num_iter);

train_accu_RF4_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF4_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF4_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF4_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF4_final = cell(size(rf_method, 2), num_iter);
test_scores_RF4_final = cell(size(rf_method, 2), num_iter);

train_accu_RF5_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF5_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF5_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF5_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF5_final = cell(size(rf_method, 2), num_iter);
test_scores_RF5_final = cell(size(rf_method, 2), num_iter);

train_accu_RF6_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF6_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF6_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF6_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF6_final = cell(size(rf_method, 2), num_iter);
test_scores_RF6_final = cell(size(rf_method, 2), num_iter);

train_accu_RF7_final = zeros(size(rf_method, 2), num_iter);
test_accu_RF7_final = zeros(size(rf_method, 2), num_iter);
testTP_accu_RF7_final = zeros(size(rf_method, 2), num_iter);
testFP_accu_RF7_final = zeros(size(rf_method, 2), num_iter);
test_pred_RF7_final = cell(size(rf_method, 2), num_iter);
test_scores_RF7_final = cell(size(rf_method, 2), num_iter);

train_accu_DA_final = zeros(1, num_iter);
test_accu_DA_final = zeros(1, num_iter);
testTP_accu_DA_final = zeros(1, num_iter);
testFP_accu_DA_final = zeros(1, num_iter);
test_pred_DA_final = cell(1, num_iter);
test_scores_DA_final = cell(1, num_iter);

train_accu_SNN_final = zeros(1, num_iter);
test_accu_SNN_final = zeros(1, num_iter);
testTP_accu_SNN_final = zeros(1, num_iter);
testFP_accu_SNN_final = zeros(1, num_iter);
test_pred_SNN_final = cell(1, num_iter);
test_scores_SNN_final = cell(1, num_iter);
% --------------------------------------------------------------------

% The number of principal components
num_pca_comp = zeros(1, num_iter); 
U_mtx = cell(1, num_iter);

% ML model list
mdl_LR_final = cell (1, num_iter);
mdl_SVM_final = cell (1, num_iter);
mdl_RF1_final = cell (1, num_iter);
mdl_RF2_final = cell (1, num_iter);
mdl_RF3_final = cell (1, num_iter);
mdl_RF4_final = cell (1, num_iter);
mdl_RF5_final = cell (1, num_iter);
mdl_RF6_final = cell (1, num_iter);
mdl_RF7_final = cell (1, num_iter);
mdl_DA_final = cell (1, num_iter);
% mdl_GAM_final = cell (1, num_iter);
mdl_SNN_final = cell (1, num_iter);

% Iteration
for i = 1 : num_iter
    
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

        for j = 1 : num_iter
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
    
    % -----------------------------------------------------
    % Training the model (optimised selection)
    fprintf('Best model iteration: %d / %d ... \n', i, num_iter);
    rng default;

    % lambda with lowest test error
    tmp_lambda_best_LR = mdl_LR{best_cvMdl_LR_idx}.ModelParameters.Lambda;
    mdl_LR_final{i} = fitclinear(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                'Learner', 'logistic', ...
                                'Cost', cost_func, ...
                                'Lambda', tmp_lambda_best_LR);

    % Prediction
    [train_accu_LR_final(i), test_accu_LR_final(i), ...
     testTP_accu_LR_final(i), testFP_accu_LR_final(i)] ...
        = ML_get_prediction_accuracy(mdl_LR_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_LR_final{i}, test_scores_LR_final{i}] = predict(mdl_LR_final{i}, tmp_iter_Xtest);
    
    clear tmp_lambda_best_LR

    fprintf('Optimised model performance LR > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_LR_final(i), test_accu_LR_final(i), testTP_accu_LR_final(i), testFP_accu_LR_final(i));
    % -------------------------------------------------------------------------------------------

    tmp_svmC_best_SVM = mdl_SVM{best_cvMdl_SVM_idx}.ModelParameters.BoxConstraint;
    tmp_svmSigma_best_SVM = mdl_SVM{best_cvMdl_SVM_idx}.ModelParameters.KernelScale;
    mdl_SVM_final{i} = fitcsvm(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                'KernelFunction', 'rbf',...
                                'Cost', cost_func, ...
                                'BoxConstraint', tmp_svmC_best_SVM, ...
                                'KernelScale', tmp_svmSigma_best_SVM);

    % Prediction
    [train_accu_SVM_final(i), test_accu_SVM_final(i), ...
     testTP_accu_SVM_final(i), testFP_accu_SVM_final(i)] ...
        = ML_get_prediction_accuracy(mdl_SVM_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_SVM_final{i}, test_scores_SVM_final{i}] = predict(mdl_SVM_final{i}, tmp_iter_Xtest);

    clear tmp_svmC_best_SVM tmp_svmSigma_best_SVM

    fprintf('Optimised model performance SVM > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_SVM_final(i), test_accu_SVM_final(i), testTP_accu_SVM_final(i), testFP_accu_SVM_final(i));
    % -------------------------------------------------------------------------------------------

    % rf == 1 || rf == 5 || rf == 6 || rf == 7
    % RF - Bag
    tmp_rf1_method = mdl_RF{best_cvMdl_RF1_idx, 1}.ModelParameters.Method;
    tmp_rf1_learner = mdl_RF{best_cvMdl_RF1_idx, 1}.ModelParameters.LearnerTemplates;
    tmp_rf1_NumLearnCycles = mdl_RF{best_cvMdl_RF1_idx, 1}.ModelParameters.NLearn;
    
    mdl_RF1_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf1_method, ...
                                    'Learners', tmp_rf1_learner, ...
                                    'NumLearningCycles', tmp_rf1_NumLearnCycles);

    % Prediction
    [train_accu_RF1_final(i), test_accu_RF1_final(i), ...
     testTP_accu_RF1_final(i), testFP_accu_RF1_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF1_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF1_final{i}, test_scores_RF1_final{i}] = predict(mdl_RF1_final{i}, tmp_iter_Xtest);

    clear tmp_rf1_method tmp_rf1_learner tmp_rf1_NumLearnCycles

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{1}, i, train_accu_RF1_final(i), test_accu_RF1_final(i), testTP_accu_RF1_final(i), testFP_accu_RF1_final(i));
    % -------------------------------------------------------------------------------------------

    % RF - RobustBoost
    tmp_rf5_method = mdl_RF{best_cvMdl_RF5_idx, 5}.ModelParameters.Method;
    tmp_rf5_learner = mdl_RF{best_cvMdl_RF5_idx, 5}.ModelParameters.LearnerTemplates;
    tmp_rf5_NumLearnCycles = mdl_RF{best_cvMdl_RF5_idx, 5}.ModelParameters.NLearn;
    
    mdl_RF5_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf5_method, ...
                                    'Learners', tmp_rf5_learner, ...
                                    'NumLearningCycles', tmp_rf5_NumLearnCycles);

    % Prediction
    [train_accu_RF5_final(i), test_accu_RF5_final(i), ...
     testTP_accu_RF5_final(i), testFP_accu_RF5_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF5_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF5_final{i}, test_scores_RF5_final{i}] = predict(mdl_RF5_final{i}, tmp_iter_Xtest);

    clear tmp_rf5_method tmp_rf5_learner tmp_rf5_NumLearnCycles

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{5}, i, train_accu_RF5_final(i), test_accu_RF5_final(i), testTP_accu_RF5_final(i), testFP_accu_RF5_final(i));
    % -------------------------------------------------------------------------------------------
    
    % RF - LPBoost
    tmp_rf6_method = mdl_RF{best_cvMdl_RF6_idx, 6}.ModelParameters.Method;
    tmp_rf6_learner = mdl_RF{best_cvMdl_RF6_idx, 6}.ModelParameters.LearnerTemplates;
    tmp_rf6_NumLearnCycles = mdl_RF{best_cvMdl_RF6_idx, 6}.ModelParameters.NLearn;
    
    mdl_RF6_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf6_method, ...
                                    'Learners', tmp_rf6_learner, ...
                                    'NumLearningCycles', tmp_rf6_NumLearnCycles);

    % Prediction
    [train_accu_RF6_final(i), test_accu_RF6_final(i), ...
     testTP_accu_RF6_final(i), testFP_accu_RF6_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF6_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF6_final{i}, test_scores_RF6_final{i}] = predict(mdl_RF6_final{i}, tmp_iter_Xtest);

    clear tmp_rf6_method tmp_rf6_learner tmp_rf6_NumLearnCycles

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{6}, i, train_accu_RF6_final(i), test_accu_RF6_final(i), testTP_accu_RF6_final(i), testFP_accu_RF6_final(i));
    % -------------------------------------------------------------------------------------------

    % RF - TotalBoost
    tmp_rf7_method = mdl_RF{best_cvMdl_RF7_idx, 7}.ModelParameters.Method;
    tmp_rf7_learner = mdl_RF{best_cvMdl_RF7_idx, 7}.ModelParameters.LearnerTemplates;
    tmp_rf7_NumLearnCycles = mdl_RF{best_cvMdl_RF7_idx, 7}.ModelParameters.NLearn;
    
    mdl_RF7_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf7_method, ...
                                    'Learners', tmp_rf7_learner, ...
                                    'NumLearningCycles', tmp_rf7_NumLearnCycles);

    % Prediction
    [train_accu_RF7_final(i), test_accu_RF7_final(i), ...
     testTP_accu_RF7_final(i), testFP_accu_RF7_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF7_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF7_final{i}, test_scores_RF7_final{i}] = predict(mdl_RF7_final{i}, tmp_iter_Xtest);

    clear tmp_rf7_method tmp_rf7_learner tmp_rf7_NumLearnCycles

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{7}, i, train_accu_RF7_final(i), test_accu_RF7_final(i), testTP_accu_RF7_final(i), testFP_accu_RF7_final(i));
    % -------------------------------------------------------------------------------------------
    
    % rf == 2 || rf == 3 || rf == 4
    % RF - AdaBoostM1
    tmp_rf2_method = mdl_RF{best_cvMdl_RF2_idx, 2}.ModelParameters.Method;
    tmp_rf2_learner = mdl_RF{best_cvMdl_RF2_idx, 2}.ModelParameters.LearnerTemplates;
    tmp_rf2_NumLearnCycles = mdl_RF{best_cvMdl_RF2_idx, 2}.ModelParameters.NLearn;
    tmp_rf2_learnrate = mdl_RF{best_cvMdl_RF2_idx, 1}.ModelParameters.LearnRate;
    mdl_RF2_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf2_method, ...
                                    'Learners', tmp_rf2_learner, ...
                                    'NumLearningCycles', tmp_rf2_NumLearnCycles, ...
                                    'LearnRate', tmp_rf2_learnrate);

    % Prediction
    [train_accu_RF2_final(i), test_accu_RF2_final(i), ...
     testTP_accu_RF2_final(i), testFP_accu_RF2_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF2_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF2_final{i}, test_scores_RF2_final{i}] = predict(mdl_RF2_final{i}, tmp_iter_Xtest);

    clear tmp_rf2_method tmp_rf2_learner tmp_rf2_NumLearnCycles tmp_rf2_learnrate

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{2}, i, train_accu_RF2_final(i), test_accu_RF2_final(i), testTP_accu_RF2_final(i), testFP_accu_RF2_final(i));
    % -------------------------------------------------------------------------------------------

    % RF - GentleBoost
    tmp_rf3_method = mdl_RF{best_cvMdl_RF3_idx, 3}.ModelParameters.Method;
    tmp_rf3_learner = mdl_RF{best_cvMdl_RF3_idx, 3}.ModelParameters.LearnerTemplates;
    tmp_rf3_NumLearnCycles = mdl_RF{best_cvMdl_RF3_idx, 3}.ModelParameters.NLearn;
    tmp_rf3_learnrate = mdl_RF{best_cvMdl_RF3_idx, 1}.ModelParameters.LearnRate;
    mdl_RF3_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf3_method, ...
                                    'Learners', tmp_rf3_learner, ...
                                    'NumLearningCycles', tmp_rf3_NumLearnCycles, ...
                                    'LearnRate', tmp_rf3_learnrate);

    % Prediction
    [train_accu_RF3_final(i), test_accu_RF3_final(i), ...
     testTP_accu_RF3_final(i), testFP_accu_RF3_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF3_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF3_final{i}, test_scores_RF3_final{i}] = predict(mdl_RF3_final{i}, tmp_iter_Xtest);

    clear tmp_rf3_method tmp_rf3_learner tmp_rf3_NumLearnCycles tmp_rf3_learnrate

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{3}, i, train_accu_RF3_final(i), test_accu_RF3_final(i), testTP_accu_RF3_final(i), testFP_accu_RF3_final(i));
    % -------------------------------------------------------------------------------------------

    % RF - LogitBoost
    tmp_rf4_method = mdl_RF{best_cvMdl_RF4_idx, 4}.ModelParameters.Method;
    tmp_rf4_learner = mdl_RF{best_cvMdl_RF4_idx, 4}.ModelParameters.LearnerTemplates;
    tmp_rf4_NumLearnCycles = mdl_RF{best_cvMdl_RF4_idx, 4}.ModelParameters.NLearn;
    tmp_rf4_learnrate = mdl_RF{best_cvMdl_RF4_idx, 1}.ModelParameters.LearnRate;
    mdl_RF4_final{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    'Cost', cost_func, ...
                                    'Replace', 'on', ...
                                    'Resample', 'on', ...
                                    'FResample', 1, ...
                                    'Method', tmp_rf4_method, ...
                                    'Learners', tmp_rf4_learner, ...
                                    'NumLearningCycles', tmp_rf4_NumLearnCycles, ...
                                    'LearnRate', tmp_rf4_learnrate);

    % Prediction
    [train_accu_RF4_final(i), test_accu_RF4_final(i), ...
     testTP_accu_RF4_final(i), testFP_accu_RF4_final(i)] ...
        = ML_get_prediction_accuracy(mdl_RF4_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    [test_pred_RF4_final{i}, test_scores_RF4_final{i}] = predict(mdl_RF4_final{i}, tmp_iter_Xtest);

    clear tmp_rf4_method tmp_rf4_learner tmp_rf4_NumLearnCycles tmp_rf4_learnrate

    fprintf('Optimised model performance RF - %s > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            rf_method{4}, i, train_accu_RF4_final(i), test_accu_RF4_final(i), testTP_accu_RF4_final(i), testFP_accu_RF4_final(i));
    % -------------------------------------------------------------------------------------------

    % Discriminate analysis
    tmp_DA_DiscrimType = mdl_DA{best_cvMdl_DA_idx}.ModelParameters.DiscrimType;
    tmp_DA_Gamma = mdl_DA{best_cvMdl_DA_idx}.ModelParameters.Gamma;
    tmp_DA_Delta = mdl_DA{best_cvMdl_DA_idx}.ModelParameters.Delta;

    mdl_DA_final{i} = fitcdiscr(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                'Cost', cost_func, ...
                                'ScoreTransform', 'logit', ...
                                'DiscrimType', tmp_DA_DiscrimType, ...
                                'Gamma', tmp_DA_Gamma, ...
                                'Delta', tmp_DA_Delta);

    % Prediction using the model
    [train_accu_DA_final(i),test_accu_DA_final(i), ...
     testTP_accu_DA_final(i), testFP_accu_DA_final(i)] ...
     = ML_get_prediction_accuracy(mdl_DA_final{i}, ...
                                  tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                  tmp_iter_Xtest,tmp_iter_Ytest, ...
                                  tmp_iter_num_test_TP,tmp_iter_num_test_FP);

    [test_pred_DA_final{i},test_scores_DA_final{i}] = predict(mdl_DA_final{i}, tmp_iter_Xtest);

    clear tmp_DA_DiscrimType tmp_DA_Gamma tmp_DA_Delta

    fprintf('Optimised model performance DA > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_DA_final(i), test_accu_DA_final(i), testTP_accu_DA_final(i), testFP_accu_DA_final(i));
    % -------------------------------------------------------------------------------------------

    % Shallow Neural Network
    tmp_SNN_layersize = mdl_SNN{best_cvMdl_SNN_idx}.ModelParameters.LayerSizes;
    tmp_SNN_activation = mdl_SNN{best_cvMdl_SNN_idx}.ModelParameters.Activations;
    tmp_SNN_lambda = mdl_SNN{best_cvMdl_SNN_idx}.ModelParameters.Lambda;
    tmp_SNN_LayerWeightsInit = mdl_SNN{best_cvMdl_SNN_idx}.ModelParameters.LayerWeightsInitializer;
    tmp_SNN_LayerBiasesInit = mdl_SNN{best_cvMdl_SNN_idx}.ModelParameters.LayerBiasesInitializer;
    
    mdl_SNN_final{i} = fitcnet(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                'Standardize', false, ...
                                'LayerSizes', tmp_SNN_layersize, ...
                                'Activation', tmp_SNN_activation, ...
                                'Lambda', tmp_SNN_lambda, ...
                                'LayerWeightsInitializer', tmp_SNN_LayerWeightsInit, ...
                                'LayerBiasesInitializer', tmp_SNN_LayerBiasesInit);

    % Prediction
    [train_accu_SNN_final(i), test_accu_SNN_final(i), ...
     testTP_accu_SNN_final(i), testFP_accu_SNN_final(i)] ...
        = ML_get_prediction_accuracy(mdl_SNN_final{i}, ...
                                    tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                    tmp_iter_Xtest, tmp_iter_Ytest, ...
                                    tmp_iter_num_test_TP, tmp_iter_num_test_FP);

    [test_pred_SNN_final{i}, test_scores_SNN_final{i}] = predict(mdl_SNN_final{i}, tmp_iter_Xtest);

    clear tmp_SNN_layersize tmp_SNN_activation tmp_SNN_lambda tmp_SNN_LayerWeightsInit tmp_SNN_LayerBiasesInit

    fprintf('Optimised model performance SNN > Iter %d: train accu: %d; test accu: %d; test TP accu: %d; test FP accu: %d ... \n', ...
            i, train_accu_SNN_final(i), test_accu_SNN_final(i), testTP_accu_SNN_final(i), testFP_accu_SNN_final(i));
    % -------------------------------------------------------------------------------------------

    clear tmp_iter_Xtrain tmp_iter_Ytrain tmp_iter_Xtest tmp_iter_Ytest tmp_iter_num_test_TP tmp_iter_num_test_FP

% end of best model interations  
end

% Statistics of the classification accuracies
% model: LR
train_accu_LR_final_avg = mean(train_accu_LR_final);
train_accu_LR_final_std = std(train_accu_LR_final);

test_accu_LR_avg = mean(test_accu_LR_final);
test_accu_LR_std = std(test_accu_LR_final);

test_accu_TP_LR_avg = mean(testTP_accu_LR_final);
test_accu_TP_LR_std = std(testTP_accu_LR_final); 
test_accu_FP_LR_avg = mean(testFP_accu_LR_final);
test_accu_FP_LR_std = std(testFP_accu_LR_final); 

% Selection of best trained model
[maxFinal_LR_val, maxFinal_LR_idx] = max(test_accu_LR_final); 
performance_LR = [train_accu_LR_final_avg, train_accu_LR_final_std, ...
                  test_accu_LR_avg, test_accu_LR_std, ...
                  test_accu_TP_LR_avg, test_accu_TP_LR_std, ...
                  test_accu_FP_LR_avg, test_accu_FP_LR_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_LR_val, train_accu_LR_final_avg, test_accu_LR_avg, test_accu_TP_LR_avg, test_accu_FP_LR_avg);
% -------------------------------------------------------------------

train_accu_SVM_final_avg = mean(train_accu_SVM_final);
train_accu_SVM_final_std = std(train_accu_SVM_final);

test_accu_SVM_avg = mean(test_accu_SVM_final);
test_accu_SVM_std = std(test_accu_SVM_final);

test_accu_TP_SVM_avg = mean(testTP_accu_SVM_final);
test_accu_TP_SVM_std = std(testTP_accu_SVM_final); 
test_accu_FP_SVM_avg = mean(testFP_accu_SVM_final);
test_accu_FP_SVM_std = std(testFP_accu_SVM_final); 

% Selection of best trained model
[maxFinal_SVM_val, maxFinal_SVM_idx] = max(test_accu_SVM_final); 
performance_SVM = [train_accu_SVM_final_avg, train_accu_SVM_final_std, ...
                  test_accu_SVM_avg, test_accu_SVM_std, ...
                  test_accu_TP_SVM_avg, test_accu_TP_SVM_std, ...
                  test_accu_FP_SVM_avg, test_accu_FP_SVM_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_SVM_val, train_accu_SVM_final_avg, test_accu_SVM_avg, test_accu_TP_SVM_avg, test_accu_FP_SVM_avg);
% -------------------------------------------------------------------

train_accu_rf1_final_avg = mean(train_accu_RF1_final);
train_accu_rf1_final_std = std(train_accu_RF1_final);

test_accu_rf1_avg = mean(test_accu_RF1_final);
test_accu_rf1_std = std(test_accu_RF1_final);

test_accu_TP_rf1_avg = mean(testTP_accu_RF1_final);
test_accu_TP_rf1_std = std(testTP_accu_RF1_final); 
test_accu_FP_rf1_avg = mean(testFP_accu_RF1_final);
test_accu_FP_rf1_std = std(testFP_accu_RF1_final); 

% Selection of best trained model
[maxFinal_rf1_val, maxFinal_rf1_idx] = max(test_accu_RF1_final); 
performance_rf1 = [train_accu_rf1_final_avg, train_accu_rf1_final_std, ...
                  test_accu_rf1_avg, test_accu_rf1_std, ...
                  test_accu_TP_rf1_avg, test_accu_TP_rf1_std, ...
                  test_accu_FP_rf1_avg, test_accu_FP_rf1_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf1_val, train_accu_rf1_final_avg, test_accu_rf1_avg, test_accu_TP_rf1_avg, test_accu_FP_rf1_avg);
% -------------------------------------------------------------------

train_accu_rf5_final_avg = mean(train_accu_RF5_final);
train_accu_rf5_final_std = std(train_accu_RF5_final);

test_accu_rf5_avg = mean(test_accu_RF5_final);
test_accu_rf5_std = std(test_accu_RF5_final);

test_accu_TP_rf5_avg = mean(testTP_accu_RF5_final);
test_accu_TP_rf5_std = std(testTP_accu_RF5_final); 
test_accu_FP_rf5_avg = mean(testFP_accu_RF5_final);
test_accu_FP_rf5_std = std(testFP_accu_RF5_final); 

% Selection of best trained model
[maxFinal_rf5_val, maxFinal_rf5_idx] = max(test_accu_RF5_final); 
performance_rf5 = [train_accu_rf5_final_avg, train_accu_rf5_final_std, ...
                  test_accu_rf5_avg, test_accu_rf5_std, ...
                  test_accu_TP_rf5_avg, test_accu_TP_rf5_std, ...
                  test_accu_FP_rf5_avg, test_accu_FP_rf5_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf5_val, train_accu_rf5_final_avg, test_accu_rf5_avg, test_accu_TP_rf5_avg, test_accu_FP_rf5_avg);
% -------------------------------------------------------------------

train_accu_rf6_final_avg = mean(train_accu_RF6_final);
train_accu_rf6_final_std = std(train_accu_RF6_final);

test_accu_rf6_avg = mean(test_accu_RF6_final);
test_accu_rf6_std = std(test_accu_RF6_final);

test_accu_TP_rf6_avg = mean(testTP_accu_RF6_final);
test_accu_TP_rf6_std = std(testTP_accu_RF6_final); 
test_accu_FP_rf6_avg = mean(testFP_accu_RF6_final);
test_accu_FP_rf6_std = std(testFP_accu_RF6_final); 

% Selection of best trained model
[maxFinal_rf6_val, maxFinal_rf6_idx] = max(test_accu_RF6_final); 
performance_rf6 = [train_accu_rf6_final_avg, train_accu_rf6_final_std, ...
                  test_accu_rf6_avg, test_accu_rf6_std, ...
                  test_accu_TP_rf6_avg, test_accu_TP_rf6_std, ...
                  test_accu_FP_rf6_avg, test_accu_FP_rf6_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf6_val, train_accu_rf6_final_avg, test_accu_rf6_avg, test_accu_TP_rf6_avg, test_accu_FP_rf6_avg);
% -------------------------------------------------------------------

train_accu_rf7_final_avg = mean(train_accu_RF7_final);
train_accu_rf7_final_std = std(train_accu_RF7_final);

test_accu_rf7_avg = mean(test_accu_RF7_final);
test_accu_rf7_std = std(test_accu_RF7_final);

test_accu_TP_rf7_avg = mean(testTP_accu_RF7_final);
test_accu_TP_rf7_std = std(testTP_accu_RF7_final); 
test_accu_FP_rf7_avg = mean(testFP_accu_RF7_final);
test_accu_FP_rf7_std = std(testFP_accu_RF7_final); 

% Selection of best trained model
[maxFinal_rf7_val, maxFinal_rf7_idx] = max(test_accu_RF7_final); 
performance_rf7 = [train_accu_rf7_final_avg, train_accu_rf7_final_std, ...
                  test_accu_rf7_avg, test_accu_rf7_std, ...
                  test_accu_TP_rf7_avg, test_accu_TP_rf7_std, ...
                  test_accu_FP_rf7_avg, test_accu_FP_rf7_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf7_val, train_accu_rf7_final_avg, test_accu_rf7_avg, test_accu_TP_rf7_avg, test_accu_FP_rf7_avg);
% -------------------------------------------------------------------

train_accu_rf2_final_avg = mean(train_accu_RF2_final);
train_accu_rf2_final_std = std(train_accu_RF2_final);

test_accu_rf2_avg = mean(test_accu_RF2_final);
test_accu_rf2_std = std(test_accu_RF2_final);

test_accu_TP_rf2_avg = mean(testTP_accu_RF2_final);
test_accu_TP_rf2_std = std(testTP_accu_RF2_final); 
test_accu_FP_rf2_avg = mean(testFP_accu_RF2_final);
test_accu_FP_rf2_std = std(testFP_accu_RF2_final); 

% Selection of best trained model
[maxFinal_rf2_val, maxFinal_rf2_idx] = max(test_accu_RF2_final); 
performance_rf2 = [train_accu_rf2_final_avg, train_accu_rf2_final_std, ...
                  test_accu_rf2_avg, test_accu_rf2_std, ...
                  test_accu_TP_rf2_avg, test_accu_TP_rf2_std, ...
                  test_accu_FP_rf2_avg, test_accu_FP_rf2_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf2_val, train_accu_rf2_final_avg, test_accu_rf2_avg, test_accu_TP_rf2_avg, test_accu_FP_rf2_avg);
% -------------------------------------------------------------------

train_accu_rf3_final_avg = mean(train_accu_RF3_final);
train_accu_rf3_final_std = std(train_accu_RF3_final);

test_accu_rf3_avg = mean(test_accu_RF3_final);
test_accu_rf3_std = std(test_accu_RF3_final);

test_accu_TP_rf3_avg = mean(testTP_accu_RF3_final);
test_accu_TP_rf3_std = std(testTP_accu_RF3_final); 
test_accu_FP_rf3_avg = mean(testFP_accu_RF3_final);
test_accu_FP_rf3_std = std(testFP_accu_RF3_final); 

% Selection of best trained model
[maxFinal_rf3_val, maxFinal_rf3_idx] = max(test_accu_RF3_final); 
performance_rf3 = [train_accu_rf3_final_avg, train_accu_rf3_final_std, ...
                  test_accu_rf3_avg, test_accu_rf3_std, ...
                  test_accu_TP_rf3_avg, test_accu_TP_rf3_std, ...
                  test_accu_FP_rf3_avg, test_accu_FP_rf3_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf3_val, train_accu_rf3_final_avg, test_accu_rf3_avg, test_accu_TP_rf3_avg, test_accu_FP_rf3_avg);
% -------------------------------------------------------------------

train_accu_rf4_final_avg = mean(train_accu_RF4_final);
train_accu_rf4_final_std = std(train_accu_RF4_final);

test_accu_rf4_avg = mean(test_accu_RF4_final);
test_accu_rf4_std = std(test_accu_RF4_final);

test_accu_TP_rf4_avg = mean(testTP_accu_RF4_final);
test_accu_TP_rf4_std = std(testTP_accu_RF4_final); 
test_accu_FP_rf4_avg = mean(testFP_accu_RF4_final);
test_accu_FP_rf4_std = std(testFP_accu_RF4_final); 

% Selection of best trained model
[maxFinal_rf4_val, maxFinal_rf4_idx] = max(test_accu_RF4_final); 
performance_rf4 = [train_accu_rf4_final_avg, train_accu_rf4_final_std, ...
                  test_accu_rf4_avg, test_accu_rf4_std, ...
                  test_accu_TP_rf4_avg, test_accu_TP_rf4_std, ...
                  test_accu_FP_rf4_avg, test_accu_FP_rf4_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_rf4_val, train_accu_rf4_final_avg, test_accu_rf4_avg, test_accu_TP_rf4_avg, test_accu_FP_rf4_avg);
% -------------------------------------------------------------------

train_accu_DA_final_avg = mean(train_accu_DA_final);
train_accu_DA_final_std = std(train_accu_DA_final);

test_accu_DA_avg = mean(test_accu_DA_final);
test_accu_DA_std = std(test_accu_DA_final);

test_accu_TP_DA_avg = mean(testTP_accu_DA_final);
test_accu_TP_DA_std = std(testTP_accu_DA_final); 
test_accu_FP_DA_avg = mean(testFP_accu_DA_final);
test_accu_FP_DA_std = std(testFP_accu_DA_final); 

% Selection of best trained model
[maxFinal_DA_val, maxFinal_DA_idx] = max(test_accu_DA_final); 
performance_DA = [train_accu_DA_final_avg, train_accu_DA_final_std, ...
                  test_accu_DA_avg, test_accu_DA_std, ...
                  test_accu_TP_DA_avg, test_accu_TP_DA_std, ...
                  test_accu_FP_DA_avg, test_accu_FP_DA_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_DA_val, train_accu_DA_final_avg, test_accu_DA_avg, test_accu_TP_DA_avg, test_accu_FP_DA_avg);
% -------------------------------------------------------------------

train_accu_SNN_final_avg = mean(train_accu_SNN_final);
train_accu_SNN_final_std = std(train_accu_SNN_final);

test_accu_SNN_avg = mean(test_accu_SNN_final);
test_accu_SNN_std = std(test_accu_SNN_final);

test_accu_TP_SNN_avg = mean(testTP_accu_SNN_final);
test_accu_TP_SNN_std = std(testTP_accu_SNN_final); 
test_accu_FP_SNN_avg = mean(testFP_accu_SNN_final);
test_accu_FP_SNN_std = std(testFP_accu_SNN_final); 

% Selection of best trained model
[maxFinal_SNN_val, maxFinal_SNN_idx] = max(test_accu_SNN_final); 
performance_SNN = [train_accu_SNN_final_avg, train_accu_SNN_final_std, ...
                  test_accu_SNN_avg, test_accu_SNN_std, ...
                  test_accu_TP_SNN_avg, test_accu_TP_SNN_std, ...
                  test_accu_FP_SNN_avg, test_accu_FP_SNN_std];

fprintf('Summary of final model performance > LR: Max test: %d; mean train: %d; mean test: %d; mean test TP: %d; mean test FP: %d ... \n', ...
        maxFinal_SNN_val, train_accu_SNN_final_avg, test_accu_SNN_avg, test_accu_TP_SNN_avg, test_accu_FP_SNN_avg);
% -------------------------------------------------------------------

%% Saving the trained models and performance
cd(selelcted_results_folder);
save([selected_sensor_suite '_mdl_final.mat'], ...
     'mdl_LR_final', 'mdl_SVM_final', ...
     'mdl_RF1_final', 'mdl_RF2_final', 'mdl_RF3_final', 'mdl_RF4_final', 'mdl_RF5_final', 'mdl_RF6_final', 'mdl_RF7_final', ...
     'mdl_DA_final', 'mdl_SNN_final', ...
     'train_accu_LR_final', ...
     'test_accu_LR_final', ...
     'testTP_accu_LR_final', ...
     'testFP_accu_LR_final', ...
     'test_pred_LR_final', ...
     'test_scores_LR_final', ...
     'train_accu_SVM_final', ...
     'test_accu_SVM_final', ...
     'testTP_accu_SVM_final', ...
     'testFP_accu_SVM_final', ...
     'test_pred_SVM_final', ...
     'test_scores_SVM_final', ...
     'train_accu_RF1_final', ...
     'test_accu_RF1_final', ...
     'testTP_accu_RF1_final', ...
     'testFP_accu_RF1_final', ...
     'test_pred_RF1_final', ...
     'test_scores_RF1_final', ...
     'train_accu_RF2_final', ...
     'test_accu_RF2_final', ...
     'testTP_accu_RF2_final', ...
     'testFP_accu_RF2_final', ...
     'test_pred_RF2_final', ...
     'test_scores_RF2_final', ...
     'train_accu_RF3_final', ...
     'test_accu_RF3_final', ...
     'testTP_accu_RF3_final', ...
     'testFP_accu_RF3_final', ...
     'test_pred_RF3_final', ...
     'test_scores_RF3_final', ...
     'train_accu_RF4_final', ...
     'test_accu_RF4_final', ...
     'testTP_accu_RF4_final', ...
     'testFP_accu_RF4_final', ...
     'test_pred_RF4_final', ...
     'test_scores_RF4_final', ...
     'train_accu_RF5_final', ...
     'test_accu_RF5_final', ...
     'testTP_accu_RF5_final', ...
     'testFP_accu_RF5_final', ...
     'test_pred_RF5_final', ...
     'test_scores_RF5_final', ...
     'train_accu_RF6_final', ...
     'test_accu_RF6_final', ...
     'testTP_accu_RF6_final', ...
     'testFP_accu_RF6_final', ...
     'test_pred_RF6_final', ...
     'test_scores_RF6_final', ...
     'train_accu_RF7_final', ...
     'test_accu_RF7_final', ...
     'testTP_accu_RF7_final', ...
     'testFP_accu_RF7_final', ...
     'test_pred_RF7_final', ...
     'test_scores_RF7_final', ...
     'train_accu_DA_final', ...
     'test_accu_DA_final', ...
     'testTP_accu_DA_final', ...
     'testFP_accu_DA_final', ...
     'test_pred_DA_final', ...
     'test_scores_DA_final', ...
     'train_accu_SNN_final', ...
     'test_accu_SNN_final', ...
     'testTP_accu_SNN_final', ...
     'testFP_accu_SNN_final', ...
     'test_pred_SNN_final', ...
     'test_scores_SNN_final', ...
     'performance_LR','performance_SVM', ...
     'performance_rf1', 'performance_rf2', 'performance_rf3', 'performance_rf4', 'performance_rf5', 'performance_rf6', 'performance_rf7', ...
     'performance_DA', 'performance_SNN', ...
     '-v7.3');
cd(curr_dir)



