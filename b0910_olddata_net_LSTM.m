% This script is for optimising the machine learning model on FM data
clc
clear
close all

curr_dir = pwd;
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
          tmp_detection_TPc tmp_detection_FPc
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
rho_spearman = corr(X_norm, 'Type', 'Spearman');
rho_kendall = corr(X_norm, 'Type', 'Kendall');

rho_spearman_mu = mean(mean(rho_spearman, 1));
rho_kendall_mu = mean(mean(rho_kendall, 1));

%% Feature ranking by Neighbourhood Component Analysis (NCA) 
% optimise hyper-parameter: lambda with K-fold cross-validation
% lambda values
% lambdaV1 = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 0, 1, 1.0e1, 1.0e2, 1.0e3, 1.0e4]; 
lambdaV2 = linspace(0, 20, 20) / length(Y_labels); 

% Sets random seed to default value. This is necessary for reproducibility.
rng default  

% number of folds
K1 = 10;

% fraction of observation in test set
P = 0.2; 

% Stratified K-fold division / holdout division
cvp_K1 = cvpartition(Y_labels,'kfold', K1); 
cvp_P = cvpartition(Y_labels,'HoldOut', P);
cvp_L = cvpartition(Y_labels,'Leaveout');
cvp_R = cvpartition(Y_labels,'Resubstitution');

for div_op = 1

    switch div_op
    case 1
        tmp_cvp = cvp_K1;
    case 2
        tmp_cvp = cvp_P;
    case 3
        tmp_cvp = cvp_L;
    case 4
        tmp_cvp = cvp_R;
    end

    numtestsets = tmp_cvp.NumTestSets;    
    lossvalues = zeros(length(lambdaV2), numtestsets);
    
    % optimization
    for i = 1 : length(lambdaV2)   
    
        for j = 1 : numtestsets
            
            fprintf('Iterating... lambda values > %d of NumTestSets > %d \n', i, j);
    
            % Extract the training / test set from the partition object
            tmp_X_train = X_norm(tmp_cvp.training(j), :);
            tmp_Y_train = Y_labels(tmp_cvp.training(j), :);
            tmp_X_test = X_norm(tmp_cvp.test(j), :);
            tmp_Y_test = Y_labels(tmp_cvp.test(j), :);
            
            % Train an NCA model for classification
            tmp_ncaMdl = fscnca(tmp_X_train, tmp_Y_train, ...
                                'FitMethod', 'exact', 'Verbose', 1, ...
                                'Solver', 'lbfgs', ...
                                'Lambda', lambdaV2(i), ...
                                'IterationLimit', 50, ...
                                'GradientTolerance', 1e-5);
            
            % Compute the classification loss for the test set using the nca model
            lossvalues(i, j) = loss(tmp_ncaMdl, tmp_X_test, tmp_Y_test, 'LossFunction', 'quadratic');      
    
            clear tmp_X_train tmp_Y_train tmp_X_test tmp_Y_test tmp_ncaMdl
        end                          
    end
    
    % Plot lambda vs. loss
    figure
    plot(lambdaV2, mean(lossvalues,2), 'ro-')
    xlabel('Lambda values')
    ylabel('Loss values')
    grid on
    
    % the index of minimum loass > the best lambda value.
    [~,tmp_lossIdx] = min(mean(lossvalues,2));
    bestlambda_op(div_op) = lambdaV2(tmp_lossIdx); 
    lossvalues_op{div_op, :} = lossvalues;

    clear tmp_lossIdx tmp_cvp

% end of the loop of div_op
end

% best lambda from the series of stratified division
bestlambda = bestlambda_op(1);

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
thresh_feature = 0.1;
feature_idx = (1 : size(X_norm,2))'; 
feature_ranking = [feature_idx, ncaMdl.FeatureWeights]; 
num_top_features = length(find(feature_ranking(:,2) >= thresh_feature));

[~,I_sort] = sort(feature_ranking(:,2), 'descend');
feature_rankingS = feature_ranking(I_sort,:); 

index_top_features = feature_rankingS(1:num_top_features,1);
X_norm_TPD_ranked = X_norm_TPD(:,index_top_features);
X_norm_FPD_ranked = X_norm_FPD(:,index_top_features);

time_stamp = datestr(now, 'yyyymmddHHMMSS');
save(['X_TPD_norm_ranked_' time_stamp '.txt'], 'X_norm_TPD_ranked', '-ASCII');
save(['X_FPD_norm_ranked_' time_stamp '.txt'], 'X_norm_FPD_ranked', '-ASCII');

% kernel matrix (RBF)
% Parameter for Gaussian RBF kernel
% sigma = 1.0;
% kernel_type = 'gaussian';
% 
% X_norm_TPD_ranked_transformed = ML_transformed_kernel_matrix(kernel_type, sigma, X_norm_TPD_ranked);
% X_norm_FPD_ranked_transformed = ML_transformed_kernel_matrix(kernel_type, sigma, X_norm_FPD_ranked);

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
    
    K = 10;

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
cost_FPD = round(size(X_norm_FPD, 1) / size(X_norm_TPD, 1), 1); 
% cost_TP = 2;

% Defines the waitage for getting wrong
cost_func = [0, 1; cost_FPD, 0]; 

% A range of classification models
% 1 - Loagistic Regression (LR)
% 2 - Support Vector Machine (SVM )
% 3 - Neural Network (NN)
% 4 - Random Forest (RF)
LR_mdl = cell(1, num_iter); % logistic regress 
SVM_mdl = cell(1, num_iter); % support vector machine
RF_mdl = cell(1, num_iter); % decision tree ensemble (random forest)
DA_mdel = cell(1, num_iter); % discriminant analysis
GAM_mdel = cell(1, num_iter); % generalized additive model 
SNN_mdel = cell(1, num_iter); % shallow neural network
LTSM_mdel = cell(1, num_iter); % deep neural network - LTSM
GRU_mdel = cell(1, num_iter); % deep neural network - GRU

% the number of learning models
num_mdl = 8;

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
    cvp_mdl = cvpartition(tmp_iter_Ytrain, 'Kfold', 10);
    
    for mdl_option = 1 : num_mdl

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
                                          'CVPartition', cvp_mdl, ...
                                          'ShowPlots', false));
    
            % Prediction
            [accu_train_LR(i), accu_test_LR(i), ...
             accu_testTP_LR(i), accu_testFP_LR(i)]...
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
                                        'CVPartition', cvp_mdl, ...
                                        'ShowPlots', false));
            
            % Prediction using the selected model
            [accu_train_SVM(i), accu_test_SVM(i), ... 
             accu_testTP_SVM(i),accu_testFP_SVM(i)]...
                = ML_get_prediction_accuracy(SVM_mdl{i}, ...
                                             tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                             tmp_iter_Xtest, tmp_iter_Ytest, ...
                                             tmp_iter_num_test_TP, tmp_iter_num_test_FP);
        
        % (Decision Tree Ensemble) Random forest
        elseif mdl_option == 3 
    
            rng default;
    
            % Random forest algorithm is used for tree ensemble.
            % Minimum size of the leaf node is used as the stopping criterion.
            % Number of features to sample in each tree are determined based on a K-fold cross-validation.
            % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
            tmp_tree = templateTree('Type', 'classification', 'Reproducible', true, 'MinLeafSize', 50);
            RF_mdl{i} = fitcensemble(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                     'Cost', cost_func, ...
                                     'Learners', tmp_tree, ...
                                     'Replace', 'on', ...
                                     'Resample', 'on', ...
                                     'FResample', 1, ...
                                     'NumLearningCycles', 100,...
                                     'Method', 'Bag', ...
                                     'OptimizeHyperparameters', {'NumVariablesToSample'}, ...
                                     'HyperparameterOptimizationOptions',...
                                     struct('AcquisitionFunctionName', ...
                                            'expected-improvement-plus', ...
                                            'CVPartition', cvp_mdl, ...
                                            'ShowPlots', false));
            
            % Prediction using the selected model
            [accu_train_RF(i), accu_test_RF(i), ...
             accu_testTP_RF(i), accu_testFP_RF(i)]...
                = ML_get_prediction_accuracy(RF_mdl{i}, ...
                                             tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                             tmp_iter_Xtest, tmp_iter_Ytest, ...
                                             tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    
            clear tmp_tree

        % Discriminant analysis
        elseif mdl_option == 4 
    
            rng default;
    
            % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
            DA_mdl{i} = fitcdiscr(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                     'DiscrimType', 'linear', ...
                                     'Cost', cost_func, ...
                                     'Gamma', 1, 'Delta', '1', ...
                                     'ScoreTransform', 'logit', ... 
                                     'OptimizeHyperparameters', 'auto', ...
                                     'HyperparameterOptimizationOptions',...
                                     struct('AcquisitionFunctionName', ...
                                            'expected-improvement-plus', ...
                                            'CVPartition', cvp_mdl, ...
                                            'ShowPlots', false));
            
            % Prediction using the selected model
            [accu_train_DA(i), accu_test_DA(i), ...
             accu_testTP_DA(i), accu_testFP_DA(i)]...
                = ML_get_prediction_accuracy(DA_mdl{i}, ...
                                             tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                             tmp_iter_Xtest, tmp_iter_Ytest, ...
                                             tmp_iter_num_test_TP, tmp_iter_num_test_FP);
    
        % Generalized Additive Model (gradient boosting algorithm)
        elseif mdl_option == 5 
    
            rng default;
    
            % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
            GAM_mdl{i} = fitcgam(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                 'Cost', cost_func, ...
                                 'ScoreTransform', 'logit', ... 
                                 'OptimizeHyperparameters', 'auto', ...
                                 'HyperparameterOptimizationOptions',...
                                 struct('Optimizer', 'bayesopt', ...
                                        'AcquisitionFunctionName', 'expected-improvement-plus', ...
                                        'Repartition', true, ...
                                        'CVPartition', cvp_mdl, ...
                                        'ShowPlots', false));
            
            % Prediction using the selected model
            [accu_train_GAM(i), accu_test_GAM(i), ...
             accu_testTP_GAM(i), accu_testFP_GAM(i)]...
                = ML_get_prediction_accuracy(GAM_mdl{i}, ...
                                             tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                             tmp_iter_Xtest, tmp_iter_Ytest, ...
                                             tmp_iter_num_test_TP, tmp_iter_num_test_FP);    

        % Shallow Neural Network
        elseif mdl_option == 6
    
            rng default;
    
            % Sigmoid activation function is used.
            % Hyperparameters (lambda, and layer size) are determined based on a K-fold cross-validation.
            % AcquisitionFunctionName is set to expected-improvement-plus for reproducibility.
            SNN_mdl{i} = fitcnet(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                'OptimizeHyperparameters', 'auto', ...
                                'HyperparameterOptimizationOptions', ...
                                struct('AcquisitionFunctionName', ...
                                       'expected-improvement-plus', ...
                                       "MaxObjectiveEvaluations", 100, ...
                                       'CVPartition', cvp_mdl, ...
                                       'ShowPlots', false));
            
            tmp_iteration = SNN_mdl{i}.TrainingHistory.Iteration;
            tmp_trainLosses = SNN_mdl{i}.TrainingHistory.TrainingLoss;
            tmp_valLosses = SNN_mdl{i}.TrainingHistory.ValidationLoss;
            
            plot(tmp_iteration, tmp_trainLosses, tmp_iteration, tmp_valLosses)
            legend(["Training", "Validation"])
            xlabel("Iteration")
            ylabel("Cross-Entropy Loss")

            % Prediction using the selected model
            [accu_train_SNN(i), accu_test_SNN(i), ...
             accu_testTP_SNN(i), accu_testFP_SNN(i)]...
                = ML_get_prediction_accuracy(SNN_mdl{i}, ...
                                             tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                             tmp_iter_Xtest, tmp_iter_Ytest, ...
                                             tmp_iter_num_test_TP, tmp_iter_num_test_FP);

        % Deep Neural Network
        % a recurrent neural network (RNN) such as a long short-term memory (LSTM) 
        % or a gated recurrent unit (GRU) neural network for sequence and time-series data
        elseif mdl_option == 7
    
            rng default;
    
            % net = trainNetwork(sequences,layers,options)
            


            
            LSTM_net{i} = trainNetwork(tmp_iter_Xtrain, tmp_iter_Ytrain, ...
                                'OptimizeHyperparameters', 'auto', ... 
                                'HyperparameterOptimizationOptions', ...
                                struct('AcquisitionFunctionName', ...
                                       'expected-improvement-plus', ...
                                       "MaxObjectiveEvaluations", 100, ...
                                       'CVPartition', cvp_mdl, ...
                                       'ShowPlots', false));
            
            tmp_iteration = LSTM_net{i}.TrainingHistory.Iteration;
            tmp_trainLosses = LSTM_net{i}.TrainingHistory.TrainingLoss;
            tmp_valLosses = LSTM_net{i}.TrainingHistory.ValidationLoss;
            
            plot(tmp_iteration, tmp_trainLosses, tmp_iteration, tmp_valLosses)
            legend(["Training", "Validation"])
            xlabel("Iteration")
            ylabel("Cross-Entropy Loss")

            % Prediction using the selected model
            [accu_train_DNN(i), accu_test_DNN(i), ...
             accu_testTP_DNN(i), accu_testFP_DNN(i)]...
                = ML_get_prediction_accuracy(LSTM_net{i}, ...
                                             tmp_iter_Xtrain, tmp_iter_Ytrain,...
                                             tmp_iter_Xtest, tmp_iter_Ytest, ...
                                             tmp_iter_num_test_TP, tmp_iter_num_test_FP);
        end


    end
    
    
    % One iteration only (break after 1st loop)
    if cv_option == 0 
        break; 
    end

end

% the best model: the maximum test accuracy
[bestMdl_val, bestMdl_idx] = max(accu_test); 

%% The model with the optimal hyperparameters
final_test_pred = cell(1, num_iter);
final_test_scores = cell(1, num_iter);
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
    load_prediction_option = 0; 

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
                test_scores = sigmoid(dlarray(final_test_scores{j}));
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
        
        selected_data_proc_labeled = bwlabel(selected_data_proc{i});
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
            = ML_match_sensation(all_seg_cat(i,:), ...
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

    % indv_detections = [TP_detection{1,1},FP_detection{1,1},TN_detection{1,1},FN_detection{1,1}];

    % Combine all the data files
    TP_detectionAll(k, :) = sum(TP_detection, 1);
    FP_detectionAll(k, :) = sum(FP_detection, 1);
    TN_detectionAll(k, :) = sum(TN_detection, 1);
    FN_detectionAll(k, :) = sum(FN_detection, 1);

    % overall_detection_matrix(k,:) = [TP_detectionAll, FP_detectionAll, TN_detectionAll, FN_detectionAll];

    [performanceAll_sens, ...
     performanceAll_ppv, ...
     performanceAll_spec, ...
     performanceAll_accu, ...
     performanceAll_fscore, ...
     performanceAll_fdr] ...
        = ML_get_performance_params(TP_detectionAll(k, :), ...
                                 FP_detectionAll(k, :), ...
                                 TN_detectionAll(k, :), ...
                                 FN_detectionAll(k, :));

    performanceAll_pabak = 2 * performanceAll_accu - 1;

    overall_performance_matrix(k, :, :) = [performanceAll_sens; performanceAll_spec; ...
                                        performanceAll_accu; performanceAll_ppv; ...
                                        performanceAll_fscore; performanceAll_fdr; ...
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

% fprintf(['Detection stats:\n ...' ...
%                          ' >> ACC = %.3f, SEN = %.3f, SPE = %.3f ... \n'], ...
%                          performanceAll_accu{1}, performanceAll_sens{1}, performanceAll_spec{1});
% fprintf('PPV = %.3f, F1 score = %.3f, PABAK = %.3f ... \n', ...
%         performanceAll_ppv{1}, performanceAll_fscore{1}, performanceAll_pabak);

% Clearing variable
% clear selected_data_preproc sensor_data_sgmntd ...
%       selected_data_proc...
%       selected_data_proc_labeled

% LSTM
% -----------------------------------------------------------------------------------------
LSTM_inputSize = 24; % Define the input size (e.g., number of features)
LSTM_numHiddenUnits = 100; % Define the number of hidden units in the LSTM layer
LSTM_numClasses = 2; % Define the number of output classes

% * lstm / gru
% lstmLayer(numHiddenUnits, 'OutputMode', 'last')
% gruLayer(numHiddenUnits, 'OutputMode', 'last')

% * bi- / unidirectional LSTM layers
% bilstmLayer(LSTM_numHiddenUnits, 'OutputMode', 'last')
% lstmLayer(LSTM_numHiddenUnits, 'OutputMode', 'last')
LSTM_layers = [sequenceInputLayer(LSTM_inputSize)
          lstmLayer(LSTM_numHiddenUnits, 'OutputMode', 'last')
          fullyConnectedLayer(LSTM_numClasses)
          softmaxLayer
          classificationLayer];

LSTM_options = trainingOptions('adam', ...
                          'MaxEpochs', 100, ...
                          'MiniBatchSize', 64, ...
                          'InitialLearnRate', 0.01, ...
                          'GradientThreshold', 1, ...
                          'ExecutionEnvironment',"auto",...
                          'plots','training-progress', ...
                          'Verbose',false);

LSTM_net = trainNetwork(tmp_iter_Xtrain_lstm, tmp_iter_Ytrain3, LSTM_layers, LSTM_options);
LSTM_YPred = classify(LSTM_net, tmp_iter_Xtrain);
accuracy = mean(LSTM_YPred == YValidation);




