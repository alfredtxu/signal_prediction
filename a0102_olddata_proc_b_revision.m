% -------------------------------------------------------------------------
% SD1:
% Channel 1: Flexi force sensor
% Channel 2: Piezoelectric plate sensor 1 (Left belly)
% Channel 3: Piezoelectric plate sensor 2 (Right belly)
% Channel 4: Acoustic sensor 1 (Left belly)
% Channel 5-7: IMU data (Accelerometer)
% Channel 8: Maternal senstation
% 
% SD2:
% Channel 1: Accoustic sensor 2 (Right belly)
% Channel 2-4: Accelerometer 2 (Right belly)
% Channel 5-7: Accelerometer 1 (Left belly)
% Channel 8: Maternal sensation
% -------------------------------------------------------------------------
clc
clear
close all

curr_dir = pwd;
cd(curr_dir);

% Add paths of the folders with all necessary function files
% SP_function_files: signal processing algorithms
% ML_function_files: machine learning algorithms
% Learned_models: learned models
% matplotlib colormaps: matplotlib_function_files
addpath(genpath('SP_function_files')) 
addpath(genpath('ML_function_files')) 
addpath(genpath('Learned_models')) 
addpath(genpath('matplotlib_function_files')) 
addpath(genpath('z10_olddata_mat_raw')) 
addpath(genpath('z11_olddata_mat_preproc')) 
addpath(genpath('z12_olddata_mat_proc')) 
addpath(genpath('z13_olddata_mat_proc_revision')) 

% LOADING PRE-PROCESSED DATA
% ** set current loading data portion
% group the data matrice by individuals
participant = 'S5';
load(['sensor_data_suite_' participant '_preproc.mat']);

% the number of data file
nfile = size(sens1T, 1);
%-------------------------------------------------------------------------

% Define known parameters
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
fres_sensor = 1024; 
fres_sensation = 1024; 
num_channel = 8;    
num_FMsensors = 6; 
% -------------------------------------------------------------------------

% Dilation time for FM signal in second
FM_dilation_time = 3.0; 
dilation_size = round (FM_dilation_time * fres_sensor); 

% linear element necessary for dilation operation
FM_lse = strel('line', dilation_size, 90); 
low_signal_quantile = 0.25;
segmentation_cutoff = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];
FM_min = [30, 30, 30, 30, 30, 30];

% Best multipliers for combined sensors of each types are considered
% FM_min_SN = [FM_min_SN_best_cmbd_overall(1), FM_min_SN_best_cmbd_overall(1), ...
%              FM_min_SN_best_cmbd_overall(2), FM_min_SN_best_cmbd_overall(2), ...
%              FM_min_SN_best_cmbd_overall(3), FM_min_SN_best_cmbd_overall(3)]; 

% Parameters: sensation map and detection matching
% Backward / Forward extension length in second
ext_backward = 5.0; 
ext_forward = 2.0 ;
dil_backward = round(ext_backward*fres_sensation); 
dil_forward = round(ext_forward*fres_sensation); 

sensation_map = cell(nfile,1);
num_sens_FMactivity = zeros(nfile, 1); 
num_sens_activity = zeros(nfile, 1);
threshold = zeros(nfile, num_FMsensors); 

% percentage of overlap between senstation and maternal movement
olap_per = 0.1;

% Parameters: IMU accelerometer data map
IMUacce_map = cell(nfile,1);

% Fixed threshold value obtained through seperate testing
% Subject 1 & 2: 0.003
% Subject 3, 4 & 5: 0.002
switch participant
    case 'S1'
        IMU_threshold=0.003;
    case 'S2'
        IMU_threshold=0.003;
    case 'S3'
        IMU_threshold=0.002;
    case 'S4'
        IMU_threshold=0.002;
    case 'S5'
        IMU_threshold=0.002;
    otherwise
        disp('Error: the number of sensor types is beyond pre-setting...')
end

% Dilation length in seconds
IMU_dilation_time = 4.0; 

% Dilation lenght in sample number
IMU_dilation_size = round(IMU_dilation_time*fres_sensor); 

% Create a linear structuring element (vertical, as deg=90) 
% that will have values 1 with a length of dilation_size;
IMU_lse = strel('line', IMU_dilation_size, 90); 

% the number of files involved
n1 = 1; 
n2 = nfile;

% initialize sensor suite
sensor_suite_thresh = zeros(n2, num_FMsensors); 
sensor_suite_segmented = cell(n2, num_FMsensors);
% -------------------------------------------------------------------------

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : nfile

    fprintf('Current data file from %s: %d of %d ...\n', participant, i, nfile);

    % Segment & dilate the IMU data and to generate an IMU map. 
    % Dilating IMU accelerometer data
    % Dilate or expand the ROI's (points with value = 1) by dilation_size 
    % (half above and half below) defined by SE
    tmp_IMUacce = IMUacceNetFilTEq{i,:};
    tmp_IMUacce_map = abs(tmp_IMUacce) >= IMU_threshold;
    IMUacce_map{i,:} = imdilate(tmp_IMUacce_map, IMU_lse); 
    % -------------------------------------------------------------------------

    % Sensation map: dilating every detection to the backward and forward range. 
    % Revome the overlapping with IMUacce_map (body movements). 
    % maternal sensation detection
    tmp_sens = sens1TEq{i,:};
    tmp_sens_idx = find(tmp_sens>0);

    % Initializing the map with zeros everywhere
    tmp_sens_map = zeros(length(tmp_sens),1); 

    for j1 = 1 : length(tmp_sens_idx) 

        % Getting the index values for the map
        tmp_idx = tmp_sens_idx(j1); 

        % starting/ending point of this sensation in the map
        tmp_idxS = tmp_idx - dil_backward; 
        tmp_idxE = tmp_idx + dil_forward; 

        % avoid index exceeding
        tmp_idxS = max(tmp_idxS, 1);
        tmp_idxE = min(tmp_idxE, length(tmp_sens_map));
        
        % Generating sensation map: a single vector with all the sensation
        % Assigns 1 to the diliated range of [L1:L2]
        tmp_sens_map(tmp_idxS:tmp_idxE) = 1; 
    
        % Removal of the maternal sensation that has coincided with body
        % movement (IMU accelerometre data)
        tmp_overlaps = sum(tmp_sens_map(tmp_idxS:tmp_idxE) .* tmp_IMUacce_map(tmp_idxS:tmp_idxE));
        if (tmp_overlaps >= olap_per*abs(tmp_idxE-tmp_idxS+1))
            tmp_sens_map(tmp_idxS:tmp_idxE) = 0; 
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_overlaps
    end

    % sensation map
    sensation_map{i, :} = tmp_sens_map;

    % detection statistics: connected matrix & the number of connected components
    % * Remove the first component(, which is background valued as 0)
    % FM & maternal senstation
    [tmp_sens_map_label, tmp_num_sens_FMcomp] = bwlabel(tmp_sens_map);
    sens_map_label{i,:} = tmp_sens_map_label;
    num_sens_FMactivity(i,:) = tmp_num_sens_FMcomp; 

    [tmp_sens_label, tmp_num_sens_comp] = bwlabel(tmp_sens);
    sens_label{i,:} = tmp_sens_label;
    num_sens_activity(i,:) = tmp_num_sens_comp; 

    clear tmp_IMUacce tmp_IMUacce_map
    clear tmp_sens tmp_sens_idx tmp_sens_map
    clear tmp_num_sens_FMcomp tmp_num_sens_comp
    clear tmp_sens_map_label tmp_sens_label
    % -------------------------------------------------------------------------

    % Segmenting FM sensor data
    % > decide the threshold
    % > remove body movement
    % > dilate the data.
    sensor_suite{i, 1} = acceLNetFilTEq{i,:};
    sensor_suite{i, 2} = acceRNetFilTEq{i,:};
    sensor_suite{i, 3} = acouLFilTEq{i,:};
    sensor_suite{i, 4} = acouRFilTEq{i,:};
    sensor_suite{i, 5} = piezLFilTEq{i,:};
    sensor_suite{i, 6} = piezRFilTEq{i,:};

    % Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(sensor_suite{i, j2});

        % Returns the quantile value referred as the noise threshold
        tmp_low_cutoff = quantile(tmp_sensor_indiv,low_signal_quantile); 
        tmp_sensor_suite_low = tmp_sensor_indiv(tmp_sensor_indiv <= tmp_low_cutoff); 
        
        % determine the individual/adaptive threshold
        sensor_suite_thresh(i, j2) = FM_min(j2) * median(tmp_sensor_suite_low);
        
        if isnan(sensor_suite_thresh(i, j2)) 
            sensor_suite_thresh(i, j2) = Inf;
        end

        % Precaution against too noisy signal
        if sensor_suite_thresh(i, j2) < segmentation_cutoff(j2)
            sensor_suite_thresh(i, j2) = Inf; 
        end

        % binarising: above threshold - 1 otherwise - 0
        sensor_suite_segmented{i, j2} = (tmp_sensor_indiv >= sensor_suite_thresh(i, j2)); 

        % Dilation of the thresholded data
        sensor_suite_segmented{i, j2} = imdilate(sensor_suite_segmented{i, j2}, FM_lse); 

        % Exclusion of body movement data
        sensor_suite_segmented{i, j2} = sensor_suite_segmented{i, j2} .* (1-IMUacce_map{i,:}); 

        clear tmp_sensor_indiv tmp_low_cutoff tmp_sensor_suite_low
    end
    % -------------------------------------------------------------------------
    
    % Sensor fusion 1
    % * Each sensor combined with logical 'OR' operator
    % * Data fusion performed after dilation.
    % Three groups:
    % > single sensor 
    % > combinatorial two-type sensor 
    % > all threee type of senors
    sensor_fusion1_accl{i,:} = double(sensor_suite_segmented{i, 1} | sensor_suite_segmented{i, 2});
    sensor_fusion1_acou{i,:} = double(sensor_suite_segmented{i, 3} | sensor_suite_segmented{i, 4});
    sensor_fusion1_piez{i,:} = double(sensor_suite_segmented{i, 5} | sensor_suite_segmented{i, 6});
    % sensor_fusion1_comb{i,:} = cat(1,sensor_fusion1_accl{i,:}, sensor_fusion1_acou{i,:}, sensor_fusion1_piez{i,:});

    sensor_fusion1_accl_acou{i,:}  = double(sensor_fusion1_accl{i,:} | sensor_fusion1_acou{i,:});
    sensor_fusion1_accl_piez{i,:} = double(sensor_fusion1_accl{i,:} | sensor_fusion1_piez{i,:});
    sensor_fusion1_acou_piez{i,:}  = double(sensor_fusion1_acou{i,:} | sensor_fusion1_piez{i,:});

    sensor_fusion1_all{i,:} = double(sensor_fusion1_accl{i,:} | sensor_fusion1_acou{i,:} | sensor_fusion1_piez{i,:});
    [tmp_sensor_fusion1_all_bw, tmp_sensor_fusion1_all_comp] = bwlabel(sensor_fusion1_all{i,:});
    sensor_fusion1_all_bw{i,:} = tmp_sensor_fusion1_all_bw;
    sensor_fusion1_all_comp{i,:} = tmp_sensor_fusion1_all_comp;

    clear tmp_sensor_fusion1_all_bw tmp_sensor_fusion1_all_comp
    % -------------------------------------------------------------------------

    % Sensor fusion 2 
    % * Each type of sensor combined with logical 'AND' operator
    % * Data fusion performed after dilation.
    % Two groups:
    % > combinatorial two-type sensor 
    % > all threee type of senors
    sensor_fusion2_accl_acou{i,:} = double(sensor_fusion1_accl{i,:} & sensor_fusion1_acou{i,:});
    sensor_fusion2_accl_piez{i,:} = double(sensor_fusion1_accl{i,:} & sensor_fusion1_piez{i,:});
    sensor_fusion2_acou_piez{i,:} = double(sensor_fusion1_acou{i,:} & sensor_fusion1_piez{i,:});

    sensor_fusion2_all{i,:} = double(sensor_fusion1_accl{i,:} & sensor_fusion1_acou{i,:} & sensor_fusion1_piez{i,:});

    [tmp_sensor_fusion2_all_bw, tmp_sensor_fusion2_all_comp] = bwlabel(sensor_fusion2_all{i,:});
    sensor_fusion2_all_bw{i,:} = tmp_sensor_fusion2_all_bw;
    sensor_fusion2_all_comp{i,:} = tmp_sensor_fusion2_all_comp;

    clear tmp_sensor_fusion2_all_bw tmp_sensor_fusion2_all_comp
    % -------------------------------------------------------------------------

    % Sensor fusion 3
    % * Each individual sensor combined with logical 'AND' operator
    % * Data fusion performed after dilation.
    sensor_fusion3_accl{i,:} = double(sensor_suite_segmented{i, 1} & sensor_suite_segmented{i, 2});
    sensor_fusion3_acou{i,:} = double(sensor_suite_segmented{i, 3} & sensor_suite_segmented{i, 4});
    sensor_fusion3_piez{i,:} = double(sensor_suite_segmented{i, 5} & sensor_suite_segmented{i, 6});

    sensor_fusion3_accl_acou{i,:} = double(sensor_fusion3_accl{i,:} & sensor_fusion3_acou{i,:});
    sensor_fusion3_accl_piez{i,:} = double(sensor_fusion3_accl{i,:} & sensor_fusion3_piez{i,:});
    sensor_fusion3_acou_piez{i,:} = double(sensor_fusion3_acou{i,:} & sensor_fusion3_piez{i,:});

    sensor_fusion3_all{i,:} = double(sensor_fusion3_accl{i,:} & sensor_fusion3_acou{i,:} & sensor_fusion3_piez{i,:});

    [tmp_sensor_fusion3_all_bw, tmp_sensor_fusion3_all_comp] = bwlabel(sensor_fusion3_all{i,:});
    sensor_fusion3_all_bw{i,:} = tmp_sensor_fusion3_all_bw;
    sensor_fusion3_all_comp{i,:} = tmp_sensor_fusion3_all_comp;

    clear tmp_sensor_fusion3_all_bw tmp_sensor_fusion3_all_comp
    % -------------------------------------------------------------------------
 
end

% save segmented and fused data
cd('z13_olddata_mat_proc_revision')
save(['sensor_data_suite_' participant '_procRevsion.mat'], ...
      'IMUacce_map', ...
      'sensation_map', ...
      'sens_map_label', ...
      'num_sens_FMactivity', ...
      'sens_label', ...
      'num_sens_activity', ...
      'sensor_suite_thresh', ...
      'sensor_suite_segmented', ...
      'sensor_fusion1_accl', ...
      'sensor_fusion1_acou', ...
      'sensor_fusion1_piez', ...
      'sensor_fusion1_accl_acou', ...
      'sensor_fusion1_accl_piez', ...
      'sensor_fusion1_acou_piez', ...
      'sensor_fusion1_all', ...
      'sensor_fusion1_all_bw', ...
      'sensor_fusion1_all_comp', ...
      'sensor_fusion2_accl_acou', ...
      'sensor_fusion2_accl_piez', ...
      'sensor_fusion2_acou_piez', ...
      'sensor_fusion2_all', ...
      'sensor_fusion2_all_bw', ...
      'sensor_fusion2_all_comp', ...
      'sensor_fusion3_accl', ...
      'sensor_fusion3_acou', ...
      'sensor_fusion3_piez', ...
      'sensor_fusion3_accl_acou', ...
      'sensor_fusion3_accl_piez', ...
      'sensor_fusion3_acou_piez', ...
      'sensor_fusion3_all', ...
      'sensor_fusion3_all_bw', ...
      'sensor_fusion3_all_comp', ...
      '-v7.3');
cd(curr_dir)











