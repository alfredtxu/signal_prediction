% This script is for optimising the machine learning model on FM data
clc
clear
close all

% current working directory
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

% save results for backing up
res_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab\y10_results';
seq_save = 1; % sprintf('%03d', seq_save)

% the cohort of participant
participants = {'S1', 'S2', 'S3', 'S4', 'S5'};

% Parameter definition
freq_sensor = 1024;
freq_sensation = 1024;
num_FMsensor = 6;

% (Linear) dilation of sensation map (5s backward & 2s forward)
sens_dilationB = 5.0; 
sens_dilationF = 2.0; 

% (Linear) dilation of FM sensor data
FM_dilation_time = 3.0; 
FM_min_SN = [30, 30, 30, 30, 30, 30];

% Add paths for function files
addpath(genpath('y10_results'))
addpath(genpath('y11_ML_models'))
addpath(genpath('y12_visualization'))
addpath(genpath('z10_olddata_mat_raw'))
addpath(genpath('z11_olddata_mat_preproc'))
% addpath(genpath('z12_olddata_mat_proc'))
addpath(genpath('z13_olddata_mat_proc_revision'))
addpath(genpath('z90_ML_functions'))
addpath(genpath('z91_ML_python_files'))

% time stamp
dateStr = datestr(datenum(datetime), 'yyyymmdd');

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ******************************** section 1 ******************************
% ***************************** data preparation **************************
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%% Data loading: pre-processed data
% filtered, trimmed and equalized the length between two SD cards (*SD1 & SD2 for this case)
for i = 1 : size(participants, 2)

    tmp_mat = ['sensor_data_suite_' participants{i} '_preproc.mat'];
    load(tmp_mat);

    all_acceL{i, :} = acceLNetFilTEq;
    all_acceR{i, :} = acceRNetFilTEq;
    all_acouL{i, :} = acouLFilTEq;
    all_acouR{i, :} = acouRFilTEq;
    all_piezL{i, :} = piezLFilTEq;
    all_piezR{i, :} = piezRFilTEq;

    all_forc{i, :} = forcFilTEq;
    all_IMUacce{i, :} = IMUacceNetFilTEq;

    all_sens1{i, :} = sens1TEq;
    all_sens2{i, :} = sens2TEq;
    all_sensMtx{i, :} = sens1TEq_mtxP;

    all_timeV{i, :} = time_vecTEq;
    all_nfile(i, :) = size(forcFilTEq, 1);

    fprintf('Loaded pre-processed data ... %d (%d) - %s ... \n', i, all_nfile(i, :), tmp_mat);

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
all_acceL_cat = cat(1,all_acceL{:});
all_acceR_cat = cat(1,all_acceR{:});
all_acouL_cat = cat(1,all_acouL{:});
all_acouR_cat = cat(1,all_acouR{:});
all_piezL_cat = cat(1,all_piezL{:});
all_piezR_cat = cat(1,all_piezR{:});

all_forc_cat = cat(1,all_forc{:});
all_IMUacce_cat = cat(1,all_IMUacce{:});

all_sens1_cat = cat(1,all_sens1{:});
all_sens2_cat = cat(1,all_sens2{:});
all_sensMtx_cat = cat(1,all_sensMtx{:});
all_timeV_cat = cat(1,all_timeV{:});

% release system memory
clear all_acceL all_acceR all_acouL all_acouR all_piezL all_piezR ...
      all_forc all_IMUacce all_sens1 all_sens2 all_sensMtx all_timeV
% ------------------------------------------------------------------------------------

%% Data loading: processed data
% signal segmentation and dilation
for i = 1 : size(participants, 2)

    tmp_mat = ['sensor_data_suite_' participants{i} '_procRevision.mat'];
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
    all_sens_map{i,:} = sensation_map;
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

all_fusionOR_accl_cat = cat(1,all_fusion1_accl{:});
all_fusionOR_acou_cat = cat(1,all_fusion1_acou{:});
all_fusionOR_piez_cat = cat(1,all_fusion1_piez{:});
all_fusionOR_accl_acou_cat = cat(1,all_fusion1_accl_acou{:});
all_fusionOR_accl_piez_cat = cat(1,all_fusion1_accl_piez{:});
all_fusionOR_acou_piez_cat = cat(1,all_fusion1_acou_piez{:});
all_fusionOR_accl_acou_piez_cat = cat(1,all_fusion1_accl_acou_piez{:});

all_IMUacce_map_cat = cat(1,all_IMUacce_map{:});
all_sens_map_cat = cat(1,all_sens_map{:});

% combine the number of maternal sensation detection in each data file (remove the 1st element valusd at 0)
all_sens_map_label_cat = cat(1,all_sens_map_label{:});
all_sens_label_cat = cat(1,all_sens_label{:});
for i = 1 : size(all_sens_label_cat, 1) 
    all_sens_labelU_cat(i,:) = length(unique(all_sens_label_cat{i,:})) - 1;
end
% ------------------------------------------------------------------------------------

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
% ------------------------------------------------------------------------------------

%% Sensor fusion selection
% Options: single / double / three types of FM sensor combination
% selected_sensor_suite = 'accl';
% selected_sensor_suite = 'acou';
% selected_sensor_suite = 'piez';
% selected_sensor_suite = 'accl_acou';
% selected_sensor_suite = 'accl_piez';
% selected_sensor_suite = 'acou_piez';
selected_sensor_suite = 'accl_acou_piez';

switch selected_sensor_suite
    case 'accl'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat};
        selected_data_proc = all_fusionOR_accl_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat);
        selected_sensorT = 1;
        selected_sensorN = 2;
    case 'acou'
        selected_data_preproc = {all_acouL_cat, all_acouR_cat};
        selected_data_proc = all_fusionOR_acou_cat;
        selected_thresh = cat(2, all_acouL_thresh_cat, all_acouR_thresh_cat);
        selected_sensorT = 1;
        selected_sensorN = 2;
    case 'piez'
        selected_data_preproc = {all_piezL_cat, all_piezR_cat};
        selected_data_proc = all_fusionOR_piez_cat;
        selected_thresh = cat(2, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 1;
        selected_sensorN = 2;
    case 'accl_acou'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_acouL_cat, all_acouR_cat};
        selected_data_proc = all_fusionOR_accl_acou_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat);
        selected_sensorT = 2;
        selected_sensorN = 4;
    case 'accl_piez'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_piezL_cat, all_piezR_cat};
        selected_data_proc = all_fusionOR_accl_piez_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 2;
        selected_sensorN = 4;
    case 'acou_piez'
        selected_data_preproc = {all_acouL_cat, all_acouR_cat, all_acouL_cat, all_acouR_cat};
        selected_data_proc = all_fusionOR_acou_piez_cat;
        selected_thresh = cat(2, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 2;
        selected_sensorN = 4;
    case 'accl_acou_piez'
        selected_data_preproc = {all_acceL_cat, all_acceR_cat, all_acouL_cat, all_acouR_cat, all_piezL_cat, all_piezR_cat};
        selected_data_proc = all_fusionOR_accl_acou_piez_cat;
        selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);
        selected_sensorT = 3;
        selected_sensorN = 6;
    otherwise
        disp('Selected FM sensor(s) is out of range ... \n');
end

selected_results_folder = [res_dir '_' selected_sensor_suite '_' dateStr];
if ~isdir(selected_results_folder)    
    system(['mkdir ' selected_results_folder]);
end

% save pre- and processed data
cd(selected_results_folder)
save(['R' sprintf('%03d', seq_save) '_all_FMsensor_preproc.mat'], 'all_acceL_cat', 'all_acceR_cat', 'all_acceL_thresh_cat', 'all_acceR_thresh_cat', ...
                                 'all_acouL_cat', 'all_acouR_cat', 'all_acouL_thresh_cat', 'all_acouR_thresh_cat', ...
                                 'all_piezL_cat', 'all_piezL_thresh_cat', 'all_piezR_cat', 'all_piezR_thresh_cat', ...
                                 '-v7.3');
seq_save = seq_save + 1;

save(['R' sprintf('%03d', seq_save) '_all_sens_IMU.mat'], 'all_IMUacce_cat', 'all_IMUacce_map_cat', ...
                         'all_sens1_cat', 'all_sens2_cat', 'all_sens_label_cat', ...
                         'all_sens_map_cat', 'all_sens_map_label_cat', ...
                         '-v7.3');
seq_save = seq_save + 1;

save(['R' sprintf('%03d', seq_save) '_all_FMsensor_proc_fusion.mat'], 'all_fusionOR_accl_cat', 'all_fusionOR_acou_cat', 'all_fusionOR_piez_cat', ...
                                     'all_fusionOR_accl_acou_cat', 'all_fusionOR_accl_piez_cat', 'all_fusionOR_acou_piez_cat', ...
                                     'all_fusionOR_accl_acou_piez_cat', '-v7.3');
seq_save = seq_save + 1;
cd(curr_dir)

% release system memory
clear all_acceL all_acceL_cat all_acceL_thresh all_acceL_thresh_cat ...
      all_acceR all_acceR_cat all_acceR_thresh all_acceR_thresh_cat ...
      all_acouL all_acouL_cat all_acouL_thresh all_acouL_thresh_cat ...
      all_acouR all_acouR_cat all_acouR_thresh all_acouR_thresh_cat ...
      all_piezL all_piezL_cat all_piezL_thresh all_piezL_thresh_cat ...
      all_piezR all_piezR_cat all_piezR_thresh all_piezR_thresh_cat   

clear all_FM_thresh ...
      all_forc all_forc_cat ...
      all_IMUacce all_IMUacce_cat all_IMUacce_map_cat ...
      all_seg all_seg_cat ...
      all_sens1 all_sens1_cat ...
      all_sens2 all_sens2_cat ...
      all_sens_label all_sens_label_cat all_sens_labelU_cat ...
      all_sens_map_cat all_sens_map_label all_sens_map_label_cat ...
      all_sensMtx all_sensMtx_cat ...
      all_timeV all_timeV_cat

clear all_fusion1_accl all_fusionOR_accl_cat ...
      all_fusion1_acou all_fusionOR_acou_cat ...      
      all_fusion1_piez all_fusionOR_piez_cat ...
      all_fusion1_accl_acou all_fusionOR_accl_acou_cat ...
      all_fusion1_accl_piez all_fusionOR_accl_piez_cat ...
      all_fusion1_acou_piez all_fusionOR_acou_piez_cat ...
      all_fusion1_accl_acou_piez all_fusionOR_accl_acou_piez_cat

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ******************************** section 1 ******************************
% ***************************** feature extraction ************************
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%% Extract detection - Indicate ture positive (TP) & false positive (FP) classes
for i = 1 : num_dfiles

    [tmp_FMdata_lab, tmp_FMdata_comp] = bwlabel(selected_data_proc{i,:});
    tmp_detection_numTP = length(unique(tmp_FMdata_lab .* all_sens_map_cat{i,:})) - 1; 
    tmp_detection_numFP = tmp_FMdata_comp - tmp_detection_numTP;

    tmp_detection_TPc = cell(1, tmp_detection_numTP);
    tmp_detection_FPc = cell(1, tmp_detection_numFP);
    tmp_detection_TPw = zeros(tmp_detection_numTP, 1);
    tmp_detection_FPw = zeros(tmp_detection_numFP, 1);

    tmp_cTP = 0;
    tmp_cFP = 0;
    
    for j = 1 : tmp_FMdata_comp

        tmp_idxS = find(tmp_FMdata_lab == j, 1); 
        tmp_idxE = find(tmp_FMdata_lab == j, 1, 'last'); 

        tmp_mtx = zeros(length(all_sens_map_cat{i,:}),1);
        tmp_mtx(tmp_idxS : tmp_idxE) = 1;
        
        % Current (labelled) section: TPD vs FPD class
        tmp_tp = sum(tmp_mtx .* all_sens_map_cat{i,:});

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
    fprintf('Data file: %d - the number of labels is %d ... \n', i, tmp_FMdata_comp);

    clear tmp_FMdata_lab tmp_FMdata_comp ...
          tmp_cTP tmp_cFP ...
          tmp_detection_numTP tmp_detection_numFP ...
          tmp_detection_TPc tmp_detection_FPc ...
          tmp_detection_TPw tmp_detection_FPw
end

%% Data feature extraction for machine learning
% the number of true & false positive (TP & FP) detections for each participant
num_TP_p1 = 0;
num_TP_p2 = 0;
num_TP_P3 = 0;
num_TP_p4 = 0;
num_TP_p5 = 0;

num_FP_p1 = 0;
num_FP_p2 = 0;
num_FP_p3 = 0;
num_FP_p4 = 0;
num_FP_p5 = 0;

num_dfiles_p1 = num_data_files(1, :);
num_dfiles_p2 = num_data_files(2, :);
num_dfiles_p3 = num_data_files(3, :);
num_dfiles_p4 = num_data_files(4, :);
num_dfiles_p5 = num_data_files(5, :);

num_range_p1 = [1, num_dfiles_p1];
num_range_p2 = [(num_dfiles_p1+1), (num_dfiles_p1+num_dfiles_p2)];
num_range_p3 = [(num_dfiles_p1+num_dfiles_p2+1), (num_dfiles_p1+num_dfiles_p2+num_dfiles_p3)];
num_range_p4 = [(num_dfiles_p1+num_dfiles_p2+num_dfiles_p3+1), (num_dfiles_p1+num_dfiles_p2+num_dfiles_p3+num_dfiles_p4)];
num_range_p5 = [(num_dfiles_p1+num_dfiles_p2+num_dfiles_p3+num_dfiles_p4+1), (num_dfiles_p1+num_dfiles_p2+num_dfiles_p3+num_dfiles_p4+num_dfiles_p5)];

for i = num_range_p1(1) : num_range_p1(2)
    num_TP_p1 = num_TP_p1 + length(detection_TPc{i});
    num_FP_p1 = num_FP_p1 + length(detection_FPc{i});
end
for i = num_range_p2(1) : num_range_p2(2)
    num_TP_p2 = num_TP_p2 + length(detection_TPc{i});
    num_FP_p2 = num_FP_p2 + length(detection_FPc{i});
end
for i = num_range_p3(1) : num_range_p3(2)
    num_TP_P3 = num_TP_P3 + length(detection_TPc{i});
    num_FP_p3 = num_FP_p3 + length(detection_FPc{i});
end
for i = num_range_p4(1) : num_range_p4(2)
    num_TP_p4 = num_TP_p4 + length(detection_TPc{i});
    num_FP_p4 = num_FP_p4 + length(detection_FPc{i});
end
for i = num_range_p5(1) : num_range_p5(2)
    num_TP_p5 = num_TP_p5 + length(detection_TPc{i});
    num_FP_p5 = num_FP_p5 + length(detection_FPc{i});
end

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

% feature indice (the feature of entire duration + 22 adaptive features in total)
idx_TP = 1;
idx_FP = 1;
idx_num = 22;

% feature extraction: TP class
for i = 1 : size(detection_TPc, 1)

    for j = 1 : size(detection_TPc{i, :}, 2)
       
        feature_TP(idx_TP, 1) = length(detection_TPc{i, :}{j})/freq_sensor; % the entire duration

        for k = 1 : selected_sensorN
            
            tmp_signal = detection_TPc{i, :}{j}(:,k);
            tmp_signal_thresh = abs(tmp_signal) - selected_thresh(i, k);
            tmp_signal_threshGt = tmp_signal_thresh(tmp_signal_thresh > 0);
            tmp_signal_threshLe = tmp_signal_thresh(tmp_signal_thresh <= 0);

            % Time domain features - tmp_signal_thresh
            feature_TP(idx_TP, (k-1)*idx_num+2) = max(tmp_signal_thresh); % Max value
            feature_TP(idx_TP, (k-1)*idx_num+3) = min(tmp_signal_thresh); % Min value
            feature_TP(idx_TP, (k-1)*idx_num+4) = mean(tmp_signal_thresh); % Mean value
            feature_TP(idx_TP, (k-1)*idx_num+5) = median(tmp_signal_thresh); % Median value
            feature_TP(idx_TP, (k-1)*idx_num+6) = sum(tmp_signal_thresh.^2); % Energy
            feature_TP(idx_TP, (k-1)*idx_num+7) = std(tmp_signal_thresh); % Standard deviation
            feature_TP(idx_TP, (k-1)*idx_num+8) = iqr(tmp_signal_thresh); % Interquartile range
            feature_TP(idx_TP, (k-1)*idx_num+9) = skewness(tmp_signal_thresh); % Skewness
            feature_TP(idx_TP, (k-1)*idx_num+10) = kurtosis(tmp_signal_thresh); % Kurtosis

            % Time domain features - tmp_signal_threshGt
            if isempty(tmp_signal_threshGt)
                feature_TP(idx_TP, (k-1)*idx_num+11) = 0; % Duration above threshold
                feature_TP(idx_TP, (k-1)*idx_num+12) = 0; % Mean above threshold value
                feature_TP(idx_TP, (k-1)*idx_num+13) = 0; % Mesian above threshold value
                feature_TP(idx_TP, (k-1)*idx_num+14) = 0; % Energy above threshold value
            else
                feature_TP(idx_TP,(k-1)*idx_num+11) = length(tmp_signal_threshGt); % Duration above threshold
                feature_TP(idx_TP,(k-1)*idx_num+12) = mean(tmp_signal_threshGt); % Mean above threshold
                feature_TP(idx_TP,(k-1)*idx_num+13) = median(tmp_signal_threshGt); % Median above threshold
                feature_TP(idx_TP,(k-1)*idx_num+14) = sum(tmp_signal_threshGt.^2); % Energy above threshold
            end

            % Time domain features - tmp_signal_threshLe
            if isempty(tmp_signal_threshLe)
                feature_TP(idx_TP, (k-1)*idx_num+15) = 0; % Mean above threshold value
                feature_TP(idx_TP, (k-1)*idx_num+16) = 0; % Mesian above threshold value
                feature_TP(idx_TP, (k-1)*idx_num+17) = 0; % Energy above threshold value
            else
                feature_TP(idx_TP,(k-1)*idx_num+15) = mean(tmp_signal_threshLe); % Mean above threshold
                feature_TP(idx_TP,(k-1)*idx_num+16) = median(tmp_signal_threshLe); % Median above threshold
                feature_TP(idx_TP,(k-1)*idx_num+17) = sum(tmp_signal_threshLe.^2); % Energy above threshold
            end

            % Frequency domain features: the main frequency mode above 1 Hz
            [~,~,feature_TP(idx_TP, (k-1)*idx_num+18)] = ML_get_frequency_mode(tmp_signal, freq_sensor, 1);
            feature_TP(idx_TP, (k-1)*idx_num+19) = ML_getPSD(tmp_signal,freq_sensor, 1, 2);
            feature_TP(idx_TP, (k-1)*idx_num+20) = ML_getPSD(tmp_signal,freq_sensor, 2, 5);
            feature_TP(idx_TP, (k-1)*idx_num+21) = ML_getPSD(tmp_signal,freq_sensor, 5, 10);
            feature_TP(idx_TP, (k-1)*idx_num+22) = ML_getPSD(tmp_signal,freq_sensor, 10, 20);
            feature_TP(idx_TP, (k-1)*idx_num+23) = ML_getPSD(tmp_signal,freq_sensor, 20, 30);

            clear tmp_signal tmp_signal_thresh tmp_signal_threshGt tmp_signal_threshLe
        end

        idx_TP = idx_TP + 1;

    end

end

% feature extraction: FP class
for i = 1 : length(detection_FPc)
    
    for j = 1 : length(detection_FPc{i, :})

        feature_FP(idx_FP, 1) = length(detection_FPc{i, :}{j})/freq_sensor;
        
        for k = 1 : selected_sensorN

            tmp_signal = detection_FPc{i, :}{j}(:, k);
            tmp_signal_thresh = abs(tmp_signal) - selected_thresh(i, k);
            tmp_signal_threshGt = tmp_signal_thresh(tmp_signal_thresh > 0);
            tmp_signal_threshLe = tmp_signal_thresh(tmp_signal_thresh <= 0);

            feature_FP(idx_FP, (k-1)*idx_num+2) = max(tmp_signal_thresh); % Max value
            feature_FP(idx_FP, (k-1)*idx_num+3) = min(tmp_signal_thresh); % Min value
            feature_FP(idx_FP, (k-1)*idx_num+4) = mean(tmp_signal_thresh); % Mean value
            feature_FP(idx_FP, (k-1)*idx_num+5) = median(tmp_signal_thresh); % Median value
            feature_FP(idx_FP, (k-1)*idx_num+6) = sum(tmp_signal_thresh.^2); % Energy
            feature_FP(idx_FP, (k-1)*idx_num+7) = std(tmp_signal_thresh); % Standard deviation
            feature_FP(idx_FP, (k-1)*idx_num+8) = iqr(tmp_signal_thresh); % Interquartile range
            feature_FP(idx_FP, (k-1)*idx_num+9) = skewness(tmp_signal_thresh); % Skewness
            feature_FP(idx_FP, (k-1)*idx_num+10) = kurtosis(tmp_signal_thresh); % Kurtosis

            if isempty(tmp_signal_threshGt)
                feature_FP(idx_FP, (k-1)*idx_num+11) = 0; % Duration above threshold
                feature_FP(idx_FP, (k-1)*idx_num+12) = 0; % Mean above threshold value
                feature_FP(idx_FP, (k-1)*idx_num+13) = 0; % Median above threshold value
                feature_FP(idx_FP, (k-1)*idx_num+14) = 0; % Energy above threshold value
            else
                feature_FP(idx_FP, (k-1)*idx_num+11) = length(tmp_signal_threshGt); % Duration above threshold
                feature_FP(idx_FP, (k-1)*idx_num+12) = mean(tmp_signal_threshGt); % Mean above threshold
                feature_FP(idx_FP, (k-1)*idx_num+13) = median(tmp_signal_threshGt); % Median above threshold
                feature_FP(idx_FP, (k-1)*idx_num+14) = sum(tmp_signal_threshGt.^2); % Energy above threshold
            end

            if isempty(tmp_signal_threshGt)
                feature_FP(idx_FP, (k-1)*idx_num+15) = 0; % Mean above threshold value
                feature_FP(idx_FP, (k-1)*idx_num+16) = 0; % Median above threshold value
                feature_FP(idx_FP, (k-1)*idx_num+17) = 0; % Energy above threshold value
            else
                feature_FP(idx_FP, (k-1)*idx_num+15) = mean(tmp_signal_threshLe); % Mean above threshold
                feature_FP(idx_FP, (k-1)*idx_num+16) = median(tmp_signal_threshLe); % Median above threshold
                feature_FP(idx_FP, (k-1)*idx_num+17) = sum(tmp_signal_threshLe.^2); % Energy above threshold
            end

            [~,~,feature_FP(idx_FP,(k-1)*idx_num+18)] = ML_get_frequency_mode(tmp_signal, freq_sensor, 1);
            feature_FP(idx_FP,(k-1)*idx_num+19) = ML_getPSD(tmp_signal, freq_sensor, 1, 2);
            feature_FP(idx_FP,(k-1)*idx_num+20) = ML_getPSD(tmp_signal, freq_sensor, 2, 5);
            feature_FP(idx_FP,(k-1)*idx_num+21) = ML_getPSD(tmp_signal, freq_sensor, 5, 10);
            feature_FP(idx_FP,(k-1)*idx_num+22) = ML_getPSD(tmp_signal, freq_sensor, 10, 20);
            feature_FP(idx_FP,(k-1)*idx_num+23) = ML_getPSD(tmp_signal, freq_sensor, 20, 30);

            clear tmp_signal tmp_signal_thresh tmp_signal_threshGt tmp_signal_threshLe
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
X_norm_TP = X_norm(1:size(feature_TP,1), :);
X_norm_FP = X_norm(size(feature_TP,1)+1 : end, :);

% Feature standarlization (z-score normalization): stan = (X - Xmean) / Xstd
% X_zscore = (X_features - mean(X_features)) ./ std(X_features);
% X_zscore_TP = X_zscore(1:size(feature_TP,1), :);
% X_zscore_FP = X_zscore(size(feature_TP,1)+1 : end, :);

% save data feaures
cd(selected_results_folder);
save(['R' sprintf('%03d', seq_save) '_' selected_sensor_suite '_feature_selection_rawFeatures.mat', 'X_features', 'feature_TP', 'feature_FP', '-v7.3']);
seq_save = seq_save + 1;
cd(curr_dir)

% release system memory
clear detection_TPc detection_FPc
clear feature_TP feature_FP X_features 
% clear X_zscore X_zscore_TP X_zscore_FP     
% ----------------------------------------------------------------------------------------------------

%% Feature ranking by Neighbourhood Component Analysis (NCA) 
% optimise hyper-parameter: lambda with K-fold cross-validation
% lambda values
nca_lambdaV1 = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 0, 1, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5] / length(Y_labels); 
nca_lambdaV2 = linspace(0, 20, 20) / length(Y_labels); 
nca_lambdaV = cat(2, nca_lambdaV1, nca_lambdaV2);

% Gradient Tolerance values
nca_gt = [1e-4, 1e-5, 1e-6];

% Sets random seed to default value. This is necessary for reproducibility.
rng default  

% Stratified K-fold division / holdout division
nca_cvpK = cvpartition(Y_labels,'kfold', 5); 
nca_cvpP = cvpartition(Y_labels,'HoldOut', 0.2);
nca_cvpL = cvpartition(Y_labels,'Leaveout');
nca_cvpR = cvpartition(Y_labels,'Resubstitution');

nca_div = 1;
switch nca_div
    case 1
        nca_curr_cvp = nca_cvpK;
    case 2
        nca_curr_cvp = nca_cvpP;
    case 3
        nca_curr_cvp = nca_cvpL;
    case 4
        nca_curr_cvp = nca_cvpR;
end

% optimization - feature selection
nca_numtestsets = nca_curr_cvp.NumTestSets;
nca_lossvalues = zeros(length(nca_lambdaV), length(nca_gt), nca_numtestsets);
for i = 1 : length(nca_lambdaV)
    for g = 1 : length(nca_gt)
        for j = 1 : nca_numtestsets

            fprintf('Iterating... lambda values > %d; gradient tolerance > %d; NumTestSets > %d ... \n', i, g, j);

            % Extract the training / test set from the partition object
            tmp_X_train = X_norm(nca_curr_cvp.training(j), :);
            tmp_Y_train = Y_labels(nca_curr_cvp.training(j), :);
            tmp_X_test = X_norm(nca_curr_cvp.test(j), :);
            tmp_Y_test = Y_labels(nca_curr_cvp.test(j), :);

            % Train an NCA model for classification
            tmp_ncaMdl = fscnca(tmp_X_train, tmp_Y_train, ...
                                'FitMethod', 'exact', ...
                                'Verbose', 1, ...
                                'Solver', 'lbfgs', ...
                                'Lambda', nca_lambdaV(i), ...
                                'IterationLimit', 1000, ...
                                'GradientTolerance', nca_gt(g));

            % Compute the classification loss for the test set using the nca model
            nca_lossvalues(i, g, j) = loss(tmp_ncaMdl, tmp_X_test, tmp_Y_test, 'LossFunction', 'quadratic');

            clear tmp_X_train tmp_Y_train tmp_X_test tmp_Y_test tmp_ncaMdl

        end % loop for iteration
    end % loop for gradient tolerance
end % loop for lambda

% mean of k-fold / iterations
nca_cvp_muLoss = mean(nca_lossvalues,3); 

% the index of minimum loss: the best lambda value.
[~, nca_bestLambdaIdx] = min(mean(nca_cvp_muLoss, 2));
[~, nca_bestGtIdx] = min(nca_cvp_muLoss(nca_bestLambdaIdx,:)); 
nca_bestLambda = nca_lambdaV(nca_bestLambdaIdx);
nca_bestGt = nca_gt(nca_bestGtIdx);
nca_bestLoss = nca_cvp_muLoss(nca_bestLambdaIdx, nca_bestGtIdx);

% Plot lambda vs. loss
figure
plot(nca_lambdaV, mean(nca_cvp_muLoss, 2), 'ro-')
xlabel('Lambda values')
ylabel('Loss values')
grid on

% Use the selected lambda to optimise NCA model
% Stratified
% for all the sensors. 1.144441194796607e-04 from previous run; 1.202855e-04 from another run
% bestlambda = 0.000133548792051176; 
nca_mdl_final = fscnca(X_norm, Y_labels, ...
                     'FitMethod', 'exact', ...
                     'Verbose', 1, ...
                     'Solver', 'lbfgs', ...
                     'Lambda', nca_bestLambda, ...
                     'GradientTolerance', nca_bestGt);

figure
semilogx(nca_mdl_final.FeatureWeights, 'ro')
xlabel('Feature index')
ylabel('Feature weight')   
grid on

figure
histogram(nca_mdl_final.FeatureWeights, length(nca_mdl_final.FeatureWeights))
grid on 

% Extract the feature ranking information
% Combines feature index and weights in a matrix
nca_thresh_feature = 0.05;
nca_feature_idx = (1 : size(X_norm,2))'; 
nca_feature_ranking = [nca_feature_idx, nca_mdl_final.FeatureWeights]; 
nca_top_features = length(find(nca_feature_ranking(:,2) >= nca_thresh_feature));

[~, nca_sort_idx] = sort(nca_feature_ranking(:,2), 'descend');
nca_feature_rankingS = nca_feature_ranking(nca_sort_idx,:); 

nca_index_top_features = nca_feature_rankingS(1:nca_top_features,1);
X_norm_ranked = X_norm(:, nca_index_top_features);
X_norm_TP_ranked = X_norm_TP(:, nca_index_top_features);
X_norm_FP_ranked = X_norm_FP(:, nca_index_top_features);

% save feature rankings into text files
% cd(selected_results_folder);
% save(['X_TP_norm_ranked_' selected_sensor_suite '_' dateStr '.txt'], 'X_norm_TP_ranked', '-ASCII');
% save(['X_FP_norm_ranked_' selected_sensor_suite '_' dateStr '.txt'], 'X_norm_FP_ranked', '-ASCII');
% cd(curr_dir)

% Summary for dimensionality reduction
fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('The the optimised feature reduction is as follows: \n');
fprintf('> Best Lambda : %d ... \n ', nca_bestLambda);
fprintf('> Best Gradient tolerance : %d ... \n ', nca_bestGt);
fprintf('> Best Loss value : %d ... \n ', nca_bestLoss);
fprintf('> Reduce features (of %d) : %d ... \n ', nca_top_features, length(nca_feature_idx));
fprintf('> TP vs FP : %d vs %d ... \n ', size(X_norm_TP_ranked, 1), size(X_norm_FP_ranked, 1));
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% Save feature selection outcomes
% * save the optimised results and optimzing models separately to improve the flexibility
cd(selected_results_folder);
save(['R' sprintf('%03d', seq_save) '_' selected_sensor_suite '_feature_selection_results.mat', ...
                        'X_norm', 'X_norm_TP', 'X_norm_FP', 'X_norm_TP_ranked', 'X_norm_FP_ranked', 'Y_labels', ...
                        'nca_lossvalues', 'nca_bestlossvalues', ...
                        'nca_bestLambdaIdx', 'nca_bestLambda', ...
                        'nca_gradientToleranceIdx', 'nca_bestGT', ...
                        'nca_index_top_features',  ...
                        '-v7.3'])
seq_save = seq_save + 1;
save(['R' sprintf('%03d', seq_save) '_' selected_sensor_suite '_feature_selection_nca_mdl.mat', 'nca_mdl_final', '-v7.3']);
seq_save = seq_save + 1;
cd(curr_dir)

% Release memory
clear nca_lambdaV1 nca_lambdaV2 nca_lambdaV nca_gt ...
      nca_div nca_curr_cvp nca_cvpK nca_cvpP nca_cvpL nca_cvpR ...
      nca_numtestsets nca_cvp_muLoss ...
      nca_bestLambdaIdx nca_bestGtIdx nca_bestLambda nca_bestGt nca_lossvalues nca_bestLoss ...
      nca_mdl_final ...
      nca_thresh_feature nca_feature_idx nca_feature_ranking ...
      nca_top_features nca_sort_idx nca_feature_rankingS nca_index_top_features
% ----------------------------------------------------------------------------------------

%% Training % Test datasets
% Division into training & test sets
% 1: Divide by hold out method 
% 2: K-fold with original ratio of FPD and TPD in each fold, 
% 3: K-fold with custom ratio of FPD and TPD in each fold. 
% 4: Divide by participants
% * cases 1-3: create stratified division (each division will have the same ratio
% of FPD and TPD.)
%% Cross-validation to find the optimum hyperparameters
fprintf('Optimising hyperparameters through the process of cross-validation ...\n');

% data division for cross-valiation
ml_cv_data_div = 2;

if ml_cv_data_div == 1 
    
    num_iter = 1;

    [X_train, Y_train, X_test, Y_test, ...
     num_training_TP, num_training_FP, num_test_TP, num_test_FP] ...
     = ML_divide_by_holdout(X_norm_TP_ranked, X_norm_FP_ranked, 0.8);

    fprintf('Cross-validation data division: Hold-out * iteration(s) is %d ...\n', num_iter);

elseif (ml_cv_data_div == 2) || (ml_cv_data_div == 3) 
    
    num_iter = 5;

    [X_Kfold, Y_Kfold, ...
     num_TP_each_fold, num_TP_last_fold,...
     num_FP_each_fold, num_FP_last_fold, ...
     TP_FP_ratio, ...
     rand_num_TP, rand_num_FP] ...
     = ML_divide_by_Kfolds(X_norm_TP_ranked, X_norm_FP_ranked, ml_cv_data_div, num_iter);

    fprintf('Cross-validation data division: K-fold * iteration(s) is %d ...\n', num_iter);

% Dividing data by participant    
elseif ml_cv_data_div == 4 

    num_iter = size(participants, 2);

    [X_TP_by_participant, X_FP_by_participant] = ...
        ML_divide_by_participants(all_nfile, X_norm_TP_ranked, X_norm_FP_ranked);

    fprintf('Cross-validation data division: by participant * iteration(s) is %d ...\n', num_iter);

else
    num_iter = 0;
    fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', ml_cv_data_div);
end

% Parameters for PCA
% > Number of principal components
% > U matrix in PCA
pca_option_cv = 0; % Dimensionality reduction by PCA: 1 > on, 0 > off.
num_pca_comp = zeros(1, num_iter);
U_mtx = cell(1, num_iter); 

% cv_option = 1; % Cross-validation: 1 > K x K-fold; 0 . K-fold
% -----------------------------------------------------------------------------------------------------------

% rf_tree = templateTree('Type', 'classification', 'Reproducible', true, 'MinLeafSize', 50);
rf_tree = templateTree('Type', 'classification', 'Reproducible', true, 'Surrogate','on');

% 'RUSBoost' - Err - Sampling observations and boosting by majority undersampling at the same time not supported.
rf_method = {'Bag', ...
             'AdaBoostM1','GentleBoost','LogitBoost','RobustBoost', ...
             'LPBoost','TotalBoost'};

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

% Defines the waitage for getting wrong: higher waitage for TPD 
% cost_TP = 2;
cost_FPD = round(size(X_norm_FP, 1) / size(X_norm_TP, 1), 1); 
cost_func = [0, 1; cost_FPD, 0]; 

% A range of classification models
% the number of learning models
% num_mdl_1 = 5; % LR, SVM, RF, DA, SNN
% num_mdl_2 = 11; % LR, SVM, RF1-7, DA, SNN

% 'Generalized Additive Model', ...
% mdl_str = {'Logistic Regression', ...
%            'Support Vector Machine', ...
%            'Random Forest', ...
%            'Discriminant Analysis', ...
%            'Shallow Neural Network'};
mdl_LR = cell(1, num_iter); % logistic regress 
mdl_SVM = cell(1, num_iter); % support vector machine
mdl_RF = cell(1, num_iter); % decision tree ensemble (random forest)
mdl_DA = cell(1, num_iter); % discriminant analysis
% mdl_GAM = cell(1, num_iter); % generalized additive model 
mdl_SNN = cell(1, num_iter); % shallow neural network
% ----------------------------------------------------------------------------------------------------

% Evaluation through the interations
for i = 1 : num_iter
    
    fprintf('Current iteration ... %d / %d ... \n', i, num_iter)
    
    % training & test datasets
    % Split options: 1 - holdout, 2/3 - stratified K-fold, 4 - by participant
    if ml_cv_data_div == 1 
        
        tmp_iter_Xtrain = X_train;
        tmp_iter_Ytrain = Y_train;

        tmp_iter_Xtest = X_test;
        tmp_iter_Ytest = Y_test;

        tmp_iter_num_test_TP = num_test_TP;
        tmp_iter_num_test_FP = num_test_FP;

    elseif (ml_cv_data_div == 2) || (ml_cv_data_div == 3)

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

    elseif ml_cv_data_div == 4 

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
        fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', ml_cv_data_div);
    end

    % dimensionality reduction: PCA
    if pca_option_cv == 1
        
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

    clear tmp_iter_Xtrain tmp_iter_Ytrain tmp_iter_Xtest tmp_iter_Ytest tmp_iter_num_test_TP tmp_iter_num_test_FP cvp_mdl

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
cd(selected_results_folder);
save(['R' sprintf('%03d', seq_save) '_' selected_sensor_suite '_mdl_CV.mat'], 'mdl_LR', 'mdl_SVM','mdl_RF', 'mdl_DA', 'mdl_SNN', '-v7.3');
seq_save = seq_save + 1;
save(['R' sprintf('%03d', seq_save) '_' selected_sensor_suite '_accuracy_CV.mat'], ...
                                            'train_accu_LR', 'test_accu_LR', 'testTP_accu_LR', 'testFP_accu_LR', ...
                                            'train_accu_SVM', 'test_accu_SVM', 'testTP_accu_SVM', 'testFP_accu_SVM', ...
                                            'train_accu_RF', 'test_accu_RF', 'testTP_accu_RF', 'testFP_accu_RF', ...
                                            'train_accu_DA', 'test_accu_DA', 'testTP_accu_DA', 'testFP_accu_DA', ...
                                            'train_accu_SNN', 'test_accu_SNN', 'testTP_accu_SNN', 'testFP_accu_SNN', ...
                                            '-v7.3');
seq_save = seq_save + 1;
cd(curr_dir)

% clear variables to release systemn memory
clear mdl_LR mdl_SVM mdl_RF mdl_DA mdl_SNN

% clear variables to avoid conflicts
clear X_train Y_train X_test Y_test ...
      X_Kfold Y_Kfold num_TP_each_fold num_TP_last_fold num_FP_each_fold num_FP_last_fold ...
      TP_FP_ratio rand_num_TP rand_num_FP ...
      X_TP_by_participant X_FP_by_participant ...
      num_iter num_pca_comp U_mtx
% ------------------------------------------------------------------

%% The model with the optimal hyperparameters
% data division for cross-valiation
ml_final_data_div = 2;

if ml_final_data_div == 1 
    
    num_iter = 1;

    [X_train, Y_train, X_test, Y_test, ...
     num_training_TP, num_training_FP, num_test_TP, num_test_FP] ...
     = ML_divide_by_holdout(X_norm_TP_ranked, X_norm_FP_ranked, 0.8);

    fprintf('Cross-validation data division: Hold-out * iteration(s) is %d ...\n', num_iter);

elseif (ml_final_data_div == 2) || (ml_final_data_div == 3) 
    
    num_iter = 5;

    [X_Kfold, Y_Kfold, ...
     num_TP_each_fold, num_TP_last_fold,...
     num_FP_each_fold, num_FP_last_fold, ...
     TP_FP_ratio, ...
     rand_num_TP, rand_num_FP] ...
     = ML_divide_by_Kfolds(X_norm_TP_ranked, X_norm_FP_ranked, ml_final_data_div, num_iter);

    fprintf('Cross-validation data division: K-fold * iteration(s) is %d ...\n', num_iter);

% Dividing data by participant    
elseif ml_final_data_div == 4 

    num_iter = size(participants, 2);

    [X_TP_by_participant, X_FP_by_participant] = ...
        ML_divide_by_participants(all_nfile, X_norm_TP_ranked, X_norm_FP_ranked);

    fprintf('Cross-validation data division: by participant * iteration(s) is %d ...\n', num_iter);

else
    num_iter = 0;
    fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', ml_final_data_div);
end

% Parameters for PCA
% > Number of principal components
% > U matrix in PCA
pca_option_final = 0; % Dimensionality reduction by PCA: 1 > on, 0 > off.
num_pca_comp = zeros(1, num_iter);
U_mtx = cell(1, num_iter); 

% cv_option = 1; % Cross-validation: 1 > K x K-fold; 0 . K-fold
% -----------------------------------------------------------------------------------------------------------

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
    if ml_final_data_div == 1

        tmp_iter_Xtrain = X_train;
        tmp_iter_Ytrain = Y_train;

        tmp_iter_Xtest = X_test;
        tmp_iter_Ytest = Y_test;

        tmp_iter_num_test_TP = num_test_TP;
        tmp_iter_num_test_FP = num_test_FP;

    elseif (ml_final_data_div == 2) || (ml_final_data_div == 3)

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

    elseif ml_final_data_div == 4

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
        fprintf('Warning: Input option is %d... \n ... Please check data division option.\n', ml_cv_data_div);
    end

    % PCA for dimensionality reduction
    if pca_option_final == 1

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
% * The best ML model is adopted here for further visualization
% SVM: test_pred_SVM_final, test_scores_SVM_final
% ** will completed this part later ...
test_pred_final_best = test_pred_SVM_final;

%% Saving the trained models and performance
cd(selected_results_folder);
save(['R' sprintf('%03d', seq_save) '_' selected_sensor_suite '_mdl_final.mat'], ...
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
seq_save = seq_save + 1;
cd(curr_dir)
% -----------------------------------------------------------------------------------------------------------------------------------

% clear variable to avoid conflicts
% clear X_train Y_train X_test Y_test ...
%       X_Kfold Y_Kfold num_TP_each_fold num_TP_last_fold num_FP_each_fold num_FP_last_fold ...
%       TP_FP_ratio rand_num_TP rand_num_FP ...
%       X_TP_by_participant X_FP_by_participant ...
%       num_iter 
% clear num_pca_comp U_mtx
% -------------------------------------------------------------------------------------------------------

%% Post-actions: results processing
fprintf('Plotting ROC curve ... \n\n');
pca_option_roc = 0;
roc_flg = 0; 
switch roc_flg
    case 0
        roc_thresh = 0.5;
        roc_iter = 1;
    case 1
        roc_div = 10;
        roc_thresh = linspace(0, 1, roc_div+1);
        roc_thresh = roc_thresh(2 : end-1); % removes the first and last elements as they are 0 and 1
        roc_iter = roc_div - 1;
    otherwise
        fprintf('Warning: ROC option is not in the range ...\n');
end

% initialization
roc_detections = zeros(roc_iter, 4);

for k = 1 : roc_iter
    
    for j = 1 : size(test_pred_final_best, 2)
        
        % Convert test scores to classification probabilities
        % For SVM: Sigmoid activation: sigmoid(test_scores_SVM_final{j})
        roc_test_scores = 1 ./ (1 + exp(-1 .* test_scores_SVM_final{j}));

        % For other ML methods
        % roc_test_scores = test_scores_SVM_final{j};

        % Calculate the final test prediction
        roc_test_pred{j} = roc_test_scores(:, 2) >= roc_thresh(k);
    end

    % Get the prediction for the entire dataset
    if ml_final_data_div == 1 % holdout method
        
        % full_features: X_norm
        % Applying PCA on the whole data set
        % PCA: U_mtx, num_pca_comp
        if pca_option_roc == 1

            % ML_projectData(X_norm, U_mtx{maxFinal_SVM_idx}, num_pca_comp(maxFinal_SVM_idx)); 
            roc_alldata = X_norm_ranked * U_mtx{maxFinal_SVM_idx}(:, 1:num_pca_comp(maxFinal_SVM_idx));
        else
            roc_alldata = X_norm_ranked;
        end
        
        % The trained classifier is applied on the whole dataset
        roc_pred_alldata = predict(mdl_SVM_final{maxFinal_SVM_idx}, roc_alldata);
        roc_pred_alldata_TP = roc_pred_alldata(1 : num_TP);
        roc_pred_alldata_FP = roc_pred_alldata(num_TP + 1 : end);

    elseif (ml_final_data_div == 2) || (ml_final_data_div == 3) % stratified K-fold division
        
        [roc_pred_alldata_TP, ...
         roc_pred_alldata_FP] = ...
            ML_get_overall_test_prediction(roc_test_pred, ...
                                           num_TP, num_TP_each_fold, num_TP_last_fold, ...
                                           num_FP, num_FP_each_fold, num_FP_last_fold,...
                                           num_iter, rand_num_TP, rand_num_FP);

        % The trained classifier is applied on the whole data set
        roc_pred_alldata = [roc_pred_alldata_TP; roc_pred_alldata_FP];

    elseif ml_final_data_div == 4 % data division by participants

        % Initialization
        roc_pred_alldata_TP = []; 
        roc_pred_alldata_FP = [];
        
        for p = 1 : size(participants, 2)
            tmp_curr_TP = size(X_TP_by_participant{p},1);
            tmp_curr_FP = size(X_FP_by_participant{p},1);

            roc_pred_alldata_TP = [roc_pred_alldata_TP; roc_test_pred{p}(1:tmp_curr_TP)];
            roc_pred_alldata_FP = [roc_pred_alldata_FP; roc_test_pred{p}(tmp_curr_TP+1:end)];

            clear tmp_curr_TP tmp_curr_FP
        end

        roc_pred_alldata = [roc_pred_alldata_TP; roc_pred_alldata_FP];

    else
        fprintf('Warning > The data division option is out of range ... \n\n');
    end

    % Accuracy of for the overall prediction
    % * Y_labels is paird with X_norm before randomizing for the division of training and test sub-dataset
    accu_alldata = sum((roc_pred_alldata == Y_labels)) / length(Y_labels); 
    accu_alldata_TP = sum((roc_pred_alldata_TP == Y(1 : num_TP))) / num_TP;
    accu_alldata_FP = sum((roc_pred_alldata_FP == Y(num_TP+1 : end))) / num_FP;
    % ---------------------------------------------------------------------------------------------------------

    % the statistics of detections
    selected_data_proc_MLdetections = cell(num_dfiles, 1);
    
    % matching the index the beginning (idx == 1)
    idx_TP_match = 1; 
    idx_FP_match = 1; 
    
    for i = 1 : num_dfiles

        fprintf('the statistics of detections - current data file: %d / %d ... \n', i, num_dfiles);

        [selected_data_proc_labeled{i, :}, selected_data_proc_comp(i, :)] = bwlabel(selected_data_proc{i, :});

        % Mapping the detection by machine learning models
        % > FM overlapped with sensation & corrected predicted by the model - TP
        % > FM overlapped with sensation & incorrected predicted by the model - FN
        % > FM non-overlapped with sensation & corrected predicted by the model - FP
        % > FM non-overlapped with sensation & incorrected predicted by the model - TN
        [selected_data_proc_MLdetections{i, :}, ...
         selected_data_proc_MLdetections_mark{i, :}, ...
         idx_TP_match, idx_FP_match, ...
         mdl_TP(i, :), mdl_FN(i, :), mdl_FP(i, :), mdl_TN(i, :)]...
            = ML_map_mdl_detections(selected_data_proc_labeled{i, :}, ...
                                    selected_data_proc_comp(i, :),...
                                    all_sens_map_cat{i, :}, ...
                                    roc_pred_alldata_TP, ...
                                    roc_pred_alldata_FP, ...
                                    idx_TP_match, ...
                                    idx_FP_match);
        
        % Mapping the machine learning detection with sensation maps
        % > Corrected predicted by the model ovelapped with sensations - TP
        % > All sensations excluding TP above - FN
        % > Corrected predicted by the model non-ovelapped with sensations - FP
        % > Non-sensation & non-sensor segments at a unit of 7s (dialtion thresholds for senstations) - TN
        [all_TP(i, :), all_FN(i, :), all_FP(i, :), all_TN(i, :)] ...
            = ML_match_sensation_sensorFusion(selected_data_proc_MLdetections{i, :}, ...
                                              selected_data_proc_MLdetections_mark{i, :}, ...
                                              all_sens_map_cat{i, :}, ...
                                              sens_dilationB, sens_dilationF, ...
                                              freq_sensor, freq_sensation);
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
    [SEN_indv, PPV_indv, SPE_indv, ACC_indv, FS_indv, FPR_indv] = get_performance_params(mdl_TP, mdl_FP, mdl_TN, mdl_FN);
    % all the returned variables are 1x1 cell arrays

    indv_detections = [mdl_TP{1,1},mdl_FP{1,1},mdl_TN{1,1},mdl_FN{1,1}]; % Combines all the detections in a matrix

    % For the overall data sets
    TPD_overall{1}  = sum(mdl_TP{1},1);
    FPD_overall{1}  = sum(mdl_FP{1},1);
    TND_overall{1}  = sum(mdl_TN{1},1);
    FND_overall{1}  = sum(mdl_FN{1},1);

    roc_detections(k,:) = [TPD_overall{1,1},FPD_overall{1,1},TND_overall{1,1},FND_overall{1,1}];

    [SEN_overall, PPV_overall, SPE_overall, ACC_overall, FS_overall, FPR_overall] ...
        = get_performance_params(TPD_overall, FPD_overall, TND_overall, FND_overall);
    PABAK_overall = 2*ACC_overall{1}-1;
    detection_stats = [SEN_overall{1}, PPV_overall{1}, FS_overall, ...
        SPE_overall{1}, ACC_overall{1}, PABAK_overall];

% end of loop for roc_iter
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
    selected_data_proc_labeled


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

    ML_detection_map              = selected_data_proc_MLdetections{i}; % This indicates detections by Machine Learning algorithm
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

    ML_detection_map              = selected_data_proc_MLdetections{i}; % This indicates detections by Machine Learning algorithm
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
            X_test_LC,pca_option_cv,U,n_PC,I_max_SVM);
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
                X_test_LC,pca_option_cv,U,n_PC,I_max_NN);
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
            X_test_LC,pca_option_cv,U,n_PC,I_max_RF);
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
            X_test_LC,pca_option_cv,U,n_PC,I_max_LR);
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
    selected_data_proc_labeled = bwlabel(sensor_data_sgmntd_cmbd_all_sensors{1});
    selected_data_proc_comp = length(unique(selected_data_proc_labeled)) - 1; % Number of labels in the sensor_data_cmbd_all_sensors_labeled

    sensor_data_sgmntd_cmbd_multi_type_sensors_OR = [sensor_data_sgmntd_Left_OR_Right_Acstc, sensor_data_sgmntd_Left_OR_Right_Aclm, sensor_data_sgmntd_Left_OR_Right_Pzplt];

    % Initialization of variables
    sensor_data_sgmntd_atleast_1_type{1} = zeros(length(sensor_data_sgmntd{1}),1);
    sensor_data_sgmntd_atleast_2_type{1} = zeros(length(sensor_data_sgmntd{1}),1);
    sensor_data_sgmntd_atleast_3_type{1} = zeros(length(sensor_data_sgmntd{1}),1);

    if (selected_data_proc_comp) % When there is a detection by the sensor system
        for k = 1 : selected_data_proc_comp
            L_min = find(selected_data_proc_labeled == k, 1 ); % Sample no. corresponding to the start of the label
            L_max = find(selected_data_proc_labeled == k, 1, 'last' ); % Sample no. corresponding to the end of the label

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
    plot(sensorData_time_vector, selected_data_proc_MLdetections{i}(xlim_index1:xlim_index2)*scheme_multiplier,...
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

































