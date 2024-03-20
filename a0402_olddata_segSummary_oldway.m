clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

%% Parameter definition
% the cohort of participant
participants = {'S1', 'S2', 'S3', 'S4', 'S5'};

% the number of data files in total
num_dfiles = 131;

% new sub-directories
addpath(genpath('z12_olddata_mat_proc'))
addpath(genpath('z15_olddata_mat_proc_new_summary'))
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
selected_data_proc = all_fusion1_accl_acou_piez_cat;
selected_thresh = cat(2, all_acceL_thresh_cat, all_acceR_thresh_cat, all_acouL_thresh_cat, all_acouR_thresh_cat, all_piezL_thresh_cat, all_piezR_thresh_cat);

%% Extract detection
% Indicate ture positive (TP) & false positive (FP) classes
for i = 1 : num_dfiles

    tmp_lab = bwlabel(selected_data_proc{i,:});
    tmp_numLab = length(unique(tmp_lab)) - 1;

    tmp_detection_numTP = length(unique(tmp_lab .* all_sensation_map_cat{i,:})) - 1; 
    tmp_detection_numFP = tmp_numLab - tmp_detection_numTP;

    detection_numTP(i, :) = tmp_detection_numTP;
    detection_numFP(i, :) = tmp_detection_numFP;

    detection_mtx(i, 1) = tmp_detection_numTP;
    detection_mtx(i, 2) = all_sens_labelU_cat(i,:);
    detection_mtx(i, 3) = tmp_numLab;

    % sensation detection summary
    fprintf('Data file: %d - TP: %d of %d vs %d) ... \n', i, tmp_detection_numTP, all_sens_labelU_cat(i,:), tmp_numLab);

    clear tmp_lab tmp_numLab ...
          tmp_detection_numTP tmp_detection_numFP
end

