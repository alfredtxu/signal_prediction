% This script is for optimising the machine learning model on FM data
clc
clear
close all

% current working directory
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

% data directory
participant = '001RB';
res_dir = ['D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_' participant];

% Parameter definition
freq = 400;
num_FMsensor = 6;

% (Linear) dilation of sensation map (5s backward & 2s forward)
sens_dilationB = 5.0; 
sens_dilationF = 2.0; 

% (Linear) dilation of FM sensor data
FM_dilation_time = 3.0; 

cd(res_dir)
load(['FM01_b_mat_' participant '_focus_preproc.mat']);
cd(curr_dir)

all_acouL = acouL_TF;
all_acouM = acouM_TF;
all_acouR = acouR_TF;
all_piezL = piezL_TF;
all_piezM = piezM_TF;
all_piezR = piezR_TF;

all_forc = forc_TF;
all_IMUa = IMUaccNet_TF;
all_IMUg = IMUgyrNet_TF;

all_sens = sens_T;
all_sensMtx = sens_T_mtxP;

all_nfile = size(forc_TF, 1);

clear acou* forc* IMU* piez* sens* 
% ------------------------------------------------------------------------------------

%% Data loading: processed data
% signal segmentation and dilation
cd(res_dir)
load(['FM02_b_mat_' participant '_focus_proc.mat']);
cd(curr_dir)

% FM sensor fusion
all_fusion_acou = sensor_IMUag_fusionOR_acou;
all_fusion_piez = sensor_IMUag_fusionOR_piez;
all_fusion_acou_piez = sensor_IMUag_fusionOR_acouORpiez;

% segmented FM sensor suite & according thresholds
all_FM_seg = FM_segmented_IMUag;
all_FM_thresh = FM_thresh;

% IMU and sensation maps
all_IMUa_map = IMUa_map;
all_IMUg_map = IMUg_map;

all_sens_map = sensT_mapIMUag;
all_sens_map_label = sens_mapIMUag_label;
all_sens_label = sens_label;
% ------------------------------------------------------------------------------------

selected_data_preproc = {all_acouL, all_acouM, all_acouR, ...
                         all_piezL, all_piezM, all_piezR};
selected_data_proc = all_fusion_acou_piez;
        
%% Extract detection - Indicate ture positive (TP) & false positive (FP) classes
for i = 1 : all_nfile

    [tmp_FMdata_lab, tmp_FMdata_comp] = bwlabel(selected_data_proc{i,:});
    tmp_detection_numTP = length(unique(tmp_FMdata_lab .* all_sens_map{i,:})) - 1; 
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

        tmp_mtx = zeros(length(all_sens_map{i,:}),1);
        tmp_mtx(tmp_idxS : tmp_idxE) = 1;
        
        % Current (labelled) section: TPD vs FPD class
        tmp_tp = sum(tmp_mtx .* all_sens_map{i,:});

        if tmp_tp > 0
            
            tmp_cTP = tmp_cTP + 1;
            tmp_TPextraction = zeros(tmp_idxE-tmp_idxS+1, num_FMsensor);

            for k = 1 : num_FMsensor
                tmp_data_preproc = selected_data_preproc{k};
                tmp_TPextraction(:, k) = tmp_data_preproc{i, :}(tmp_idxS:tmp_idxE); 
                clear tmp_data_preproc
            end

            tmp_detection_TPc{tmp_cTP} = tmp_TPextraction;
        else
            tmp_cFP = tmp_cFP + 1;
            tmp_FPextraction = zeros(tmp_idxE-tmp_idxS+1, num_FMsensor);

            for k = 1 : num_FMsensor
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
% the total number of true & false positive (TP & FP) detections
num_TP = 0;
num_FP = 0;

% Loop through the cohort of data files
for i = 1 : size(detection_TPc, 1)
    num_TP = num_TP + size(detection_TPc{i, :}, 2);
    num_FP = num_FP + size(detection_FPc{i, :}, 2);
end

for i = 1 : size(all_sens_map, 1)
    [tmpL, tmpC] = bwlabel(all_sens_map{i, :});
    num_sens(i, :) = tmpC;
    clear tmpL tmpC
end
num_sensP = sum(num_sens);

cd(res_dir)
save(['TP_FP_sens_' participant '.mat'], 'num_TP', 'num_FP', 'num_sensP', ...
                                         'detection_TPc', 'detection_FPc', 'num_sens', '-v7.3');
cd(curr_dir)





