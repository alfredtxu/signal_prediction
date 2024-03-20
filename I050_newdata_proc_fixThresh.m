clc
clear 
close all

% HOME
% curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';

% OFFICE
curr_dir = pwd;

cd(curr_dir);

%% Pre-setting & data loading
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
freq = 400;
num_channel = 16;
num_FMsensors = 6;

% data file directory
fprintf('Loading data files ...\n');

% HOME
% fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

% OFFICE
fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007', 
% data category: F - focus; D - day; N - night
data_folder = 'b_mat_003AM';
data_processed = 'c_processed_003AM';
data_category = 'focus';

fdir_p = [fdir '\' data_processed];
load([fdir_p '\FM01_' data_folder '_' data_category '_preproc.mat']);

% the number of data file
num_files = size(sens_T, 1);

%% FM 
% linear element necessary for dilation operation
FM_dilation_time = 3.0;
FM_dilation_size = round (FM_dilation_time * freq);
FM_lse = strel('line', FM_dilation_size, 90);
low_signal_quantile = 0.25;
noise_level = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];
FM_min = [30, 30, 30, 30, 30, 30];

% initialize sensor suite
FM_thresh = zeros(num_files, num_FMsensors);
FM_segmented = cell(num_files, num_FMsensors);

%% IMU
% Create a linear structuring element (vertical, as deg=90)
% that will have values 1 with a length of dilation_size (in seconds);
IMU_dilation_time = 4.0;
IMU_dilation_size = round(IMU_dilation_time*freq);
IMU_lse = strel('line', IMU_dilation_size, 90);

% empirical threshold
% * old data IMU_threshold: 0.003;
% method 1: decided by designated percentile
% perc_IMUa = 95;
% perc_IMUg = 95;
% IMUa_thresh = prctile(IMUaccNet_TF, perc_IMUa);
% IMUg_thresh = prctile(IMUgyrNet_TF, perc_IMUg);

% method 2: solid threshold
IMUa_thresh = 0.010;
IMUg_thresh = 0.005;

IMUa_map = cell(num_files, 1);
IMUg_map = cell(num_files, 1);

%% sensation
sens_map = cell(num_files, 1);
sens_map_activity = zeros(num_files, 1); 
sens_activity = zeros(num_files, 1);

% Parameters: sensation map and detection matching (Backward/Forward extension length in second)
ext_backward = 5.0;
ext_forward = 2.0 ;
sens_dil_backward = round(ext_backward*freq);
sens_dil_forward = round(ext_forward*freq);

% percentage of overlap between senstation and maternal movement (IMU)
overlap_perc = 0.1;

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    fprintf('Current data file from %s (%s): %d of %d ...\n', data_folder, data_category, i, num_files);

    % Segment & dilate the IMU data and to generate an IMU map. 
    % Dilating IMU accelerometer data
    % Dilate or expand the ROI's (points with value = 1) by dilation_size 
    % (half above and half below) defined by SE
    tmp_IMUa = IMUaccNet_TF{i, :};
    tmp_IMUa_map = abs(tmp_IMUa) >= IMUa_thresh;
    IMUa_map{i, :} = imdilate(tmp_IMUa_map, IMU_lse); 

    tmp_IMUg = IMUgyrNet_TF{i, :};
    tmp_IMUg_map = abs(tmp_IMUg) >= IMUg_thresh;
    IMUg_map{i, :} = imdilate(tmp_IMUg_map, IMU_lse); 

    % Sensation map: dilating every detection to the backward and forward range. 
    % Revome the overlapping with IMUacce_map (body movements). 
    % maternal sensation detection
    tmp_sens = sens_T{i, :};
    tmp_sens_idx = find(tmp_sens == 1);

    % Initializing the map with zeros everywhere
    tmp_sens_map = zeros(length(tmp_sens), 1); 
    tmp_sens_mapIMUa = zeros(length(tmp_sens), 1); 
    tmp_sens_mapIMUg = zeros(length(tmp_sens), 1); 
    tmp_sens_mapIMUag = zeros(length(tmp_sens), 1); 

    for j1 = 1 : length(tmp_sens_idx) 

        % Getting the index values for the map
        tmp_idx = tmp_sens_idx(j1); 

        % starting/ending point of this sensation in the map
        tmp_idxS = tmp_idx - sens_dil_backward; 
        tmp_idxE = tmp_idx + sens_dil_forward; 

        % avoid index exceeding
        tmp_idxS = max(tmp_idxS, 1);
        tmp_idxE = min(tmp_idxE, length(tmp_sens_map));
        
        % Generating sensation map: a single vector with all the sensation
        % Assigns 1 to the diliated range of [L1:L2]
        tmp_sens_map(tmp_idxS:tmp_idxE) = 1; 
        tmp_sens_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sens_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sens_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        % Removal of the maternal sensation that has coincided with body
        % movement (IMU accelerometre data)
        tmp_overlaps_acc = sum(tmp_sens_map(tmp_idxS:tmp_idxE) .* IMUa_map{i, :}(tmp_idxS:tmp_idxE));
        if (tmp_overlaps_acc >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sens_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sens_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
    
        tmp_overlaps_gyr = sum(tmp_sens_map(tmp_idxS:tmp_idxE) .* IMUg_map{i, :}(tmp_idxS:tmp_idxE));
        if (tmp_overlaps_gyr >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sens_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sens_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_overlaps_acc tmp_overlaps_gyr
    end

    % sensation components and maps
    % detection statistics: connected matrix & the number of connected components
    % * Remove the first component(, which is background valued as 0)
    % FM & maternal senstation
    [tmp_sens_label, tmp_sens_comp] = bwlabel(tmp_sens);
    sens_label{i,:} = tmp_sens_label;
    sens_activity(i,:) = tmp_sens_comp;

    [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(tmp_sens_map);
    sens_map_label{i,:} = tmp_sens_map_label;
    sens_map_activity(i,:) = tmp_sens_map_comp; 

    [tmp_sens_mapIMUa_label, tmp_sens_mapIMUa_comp] = bwlabel(tmp_sens_mapIMUa);
    sens_mapIMUa_label{i,:} = tmp_sens_mapIMUa_label;
    sens_mapIMUa_activity(i,:) = tmp_sens_mapIMUa_comp; 

    [tmp_sens_mapIMUg_label, tmp_sens_mapIMUg_FMcomp] = bwlabel(tmp_sens_mapIMUg);
    sens_mapIMUg_label{i,:} = tmp_sens_mapIMUg_label;
    sens_mapIMUg_activity(i,:) = tmp_sens_mapIMUg_FMcomp; 

    [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(tmp_sens_mapIMUag);
    sens_mapIMUag_label{i,:} = tmp_sens_mapIMUag_label;
    sens_mapIMUag_activity(i,:) = tmp_sens_mapIMUag_comp; 

    sens_map{i, :} = tmp_sens_map;
    sensT_mapIMUa{i, :} = tmp_sens_mapIMUa;
    sensT_mapIMUg{i, :} = tmp_sens_mapIMUg;
    sensT_mapIMUag{i, :} = tmp_sens_mapIMUag;
    
    % clear temporal variables
    clear tmp_IMUa tmp_IMUa_map ...
          tmp_IMUg tmp_IMUg_map ...
          tmp_sens tmp_sens_idx ...
          tmp_sens_map tmp_sens_mapIMUa tmp_sens_mapIMUg tmp_sens_mapIMUag ...
          tmp_sens_label tmp_sens_comp ... 
          tmp_sens_map_label tmp_sens_map_comp ...
          tmp_sens_mapIMUa_label tmp_sens_mapIMUa_comp ...
          tmp_sens_mapIMUg_label tmp_sens_mapIMUg_FMcomp ...
          tmp_sens_mapIMUag_label tmp_sens_mapIMUag_comp
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ sensation maps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    % Segmenting FM sensor data
    % > decide the threshold
    % > remove body movement
    % > dilate the data.
    sensor_suite{i, 1} = acouL_TF{i,:};
    sensor_suite{i, 2} = acouM_TF{i,:};
    sensor_suite{i, 3} = acouR_TF{i,:};
    sensor_suite{i, 4} = piezL_TF{i,:};
    sensor_suite{i, 5} = piezM_TF{i,:};
    sensor_suite{i, 6} = piezR_TF{i,:};

    % Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(sensor_suite{i, j2});

        % Returns the quantile value referred as the noise threshold
        tmp_low_cutoff = quantile(tmp_sensor_indiv, low_signal_quantile); 
        tmp_sensor_suite_low = tmp_sensor_indiv(tmp_sensor_indiv <= tmp_low_cutoff); 
        
        % determine the individual/adaptive threshold
        FM_thresh(i, j2) = FM_min(j2) * median(tmp_sensor_suite_low);
        
        if isnan(FM_thresh(i, j2)) 
            FM_thresh(i, j2) = Inf;
        end

        % Precaution against too noisy signal
        if FM_thresh(i, j2) < noise_level(j2)
            FM_thresh(i, j2) = Inf; 
        end

        % binarising: above threshold - 1 otherwise - 0
        % Dilation of the thresholded data
        FM_segmented{i, j2} = (tmp_sensor_indiv >= FM_thresh(i, j2)); 
        FM_segmented{i, j2} = imdilate(FM_segmented{i, j2}, IMU_lse); 

        % Exclusion of body movement data
        FM_segmented_IMUa{i, j2} = FM_segmented{i, j2} .* (1-IMUa_map{i,:}); 
        FM_segmented_IMUg{i, j2} = FM_segmented{i, j2} .* (1-IMUg_map{i,:}); 
        FM_segmented_IMUa{i, j2} = FM_segmented{i, j2} .* (1-IMUa_map{i,:}) .* (1-IMUg_map{i,:}); 
        
        clear tmp_sensor_indiv tmp_low_cutoff tmp_sensor_suite_low
    end
    % -------------------------------------------------------------------------
    
    % Sensor fusion group I:
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'OR'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    sensor_fusionOR_acou{i, :} = double(FM_segmented{i, 1} | FM_segmented{i, 2} | FM_segmented{i, 3});
    sensor_fusionOR_piez{i, :} = double(FM_segmented{i, 4} | FM_segmented{i, 5} | FM_segmented{i, 6});
    sensor_fusionOR_acouORpiez{i, :}  = double(sensor_fusionOR_acou{i,:} | sensor_fusionOR_piez{i, :});
    sensor_fusionOR_acouANDpiez{i, :} = double(sensor_fusionOR_acou{i,:} & sensor_fusionOR_piez{i, :});
    
    % the label and components
    [tmp_sensor_fusionOR_acou_label, tmp_sensor_fusionOR_acou_comp] = bwlabel(sensor_fusionOR_acou{i, :});
    sensor_fusionOR_acou_label{i, :} = tmp_sensor_fusionOR_acou_label;
    sensor_fusionOR_acou_comp{i, :} = tmp_sensor_fusionOR_acou_comp;

    [tmp_sensor_fusionOR_piez_label, tmp_sensor_fusionOR_piez_comp] = bwlabel(sensor_fusionOR_piez{i, :});
    sensor_fusionOR_piez_label{i, :} = tmp_sensor_fusionOR_piez_label;
    sensor_fusionOR_piez_comp{i, :} = tmp_sensor_fusionOR_piez_comp;

    [tmp_sensor_fusionOR_acouORpiez_label, tmp_sensor_fusionOR_acouORpiez_comp] = bwlabel(sensor_fusionOR_acouORpiez{i, :});
    sensor_fusionOR_acouORpiez_label{i, :} = tmp_sensor_fusionOR_acouORpiez_label;
    sensor_fusionOR_acouORpiez_comp{i, :} = tmp_sensor_fusionOR_acouORpiez_comp;

    [tmp_sensor_fusionOR_acouANDpiez_label, tmp_sensor_fusionOR_acouANDpiez_comp] = bwlabel(sensor_fusionOR_acouANDpiez{i, :});
    sensor_fusionOR_acouANDpiez_label{i, :} = tmp_sensor_fusionOR_acouANDpiez_label;
    sensor_fusionOR_acouANDpiez_comp{i, :} = tmp_sensor_fusionOR_acouANDpiez_comp;

    clear tmp_sensor_fusionOR_acou_label tmp_sensor_fusionOR_acou_comp ...
          tmp_sensor_fusionOR_piez_label tmp_sensor_fusionOR_piez_comp ...
          tmp_sensor_fusionOR_acouORpiez_label tmp_sensor_fusionOR_acouORpiez_comp ...    
          tmp_sensor_fusionOR_acouANDpiez_label tmp_sensor_fusionOR_acouANDpiez_comp
    % -------------------------------------------------------------------------

    % Sensor fusion group II:
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'AND'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    sensor_fusionAND_acou{i, :} = double(FM_segmented{i, 1} & FM_segmented{i, 2} & FM_segmented{i, 3});
    sensor_fusionAND_piez{i, :} = double(FM_segmented{i, 4} & FM_segmented{i, 5} & FM_segmented{i, 6});
    sensor_fusionAND_acouORpiez{i, :} = double(sensor_fusionAND_acou{i, :} | sensor_fusionAND_piez{i, :});
    sensor_fusionAND_acouANDpiez{i, :} = double(sensor_fusionAND_acou{i, :} & sensor_fusionAND_piez{i, :});

    % the label and components
    [tmp_sensor_fusionAND_acou_label, tmp_sensor_fusionAND_acou_comp] = bwlabel(sensor_fusionAND_acou{i, :});
    sensor_fusionAND_acou_label{i, :} = tmp_sensor_fusionAND_acou_label;
    sensor_fusionAND_acou_comp{i, :} = tmp_sensor_fusionAND_acou_comp;

    [tmp_sensor_fusionAND_piez_label, tmp_sensor_fusionAND_piez_comp] = bwlabel(sensor_fusionAND_piez{i, :});
    sensor_fusionAND_piez_label{i, :} = tmp_sensor_fusionAND_piez_label;
    sensor_fusionAND_piez_comp{i, :} = tmp_sensor_fusionAND_piez_comp;

    [tmp_sensor_fusionAND_acouORpiez_label, tmp_sensor_fusionAND_acouORpiez_comp] = bwlabel(sensor_fusionAND_acouORpiez{i, :});
    sensor_fusionAND_acouORpiez_label{i, :} = tmp_sensor_fusionAND_acouORpiez_label;
    sensor_fusionAND_acouORpiez_comp{i, :} = tmp_sensor_fusionAND_acouORpiez_comp;

    [tmp_sensor_fusionAND_acouANDpiez_label, tmp_sensor_fusionAND_acouANDpiez_comp] = bwlabel(sensor_fusionAND_acouANDpiez{i, :});
    sensor_fusionAND_acouANDpiez_label{i, :} = tmp_sensor_fusionAND_acouANDpiez_label;
    sensor_fusionAND_acouANDpiez_comp{i, :} = tmp_sensor_fusionAND_acouANDpiez_comp;

    clear tmp_sensor_fusionAND_acou_label tmp_sensor_fusionAND_acou_comp ...
          tmp_sensor_fusionAND_piez_label tmp_sensor_fusionAND_piez_comp ...
          tmp_sensor_fusionAND_acouORpiez_label tmp_sensor_fusionAND_acouORpiez_comp ...
          tmp_sensor_fusionAND_acouANDpiez_label tmp_sensor_fusionAND_acouANDpiez_comp 
    % -------------------------------------------------------------------------

% end of loop for the files
end

% save segmented and fused data
cd([fdir '\' data_processed]);
save(['FM02_' data_folder '_' data_category '_proc.mat'], ...
      'FM_thresh', 'FM_segmented', 'IMUa_map', 'IMUg_map', ...
      'sens_map', 'sensT_mapIMUa',  'sensT_mapIMUg',  'sensT_mapIMUag', ...
      'sens_label', 'sens_activity', 'sens_map_label', 'sens_map_activity', ...
      'sens_mapIMUa_label', 'sens_mapIMUa_activity', ...
      'sens_mapIMUg_label', 'sens_mapIMUg_activity', ...
      'sens_mapIMUag_label', 'sens_mapIMUag_activity', ...
      'sensor_suite', ...
      'sensor_fusionOR_acou', 'sensor_fusionOR_piez', 'sensor_fusionOR_acouORpiez', 'sensor_fusionOR_acouANDpiez', ...
      'sensor_fusionOR_acou_label', 'sensor_fusionOR_acou_comp', 'sensor_fusionOR_piez_label', 'sensor_fusionOR_piez_comp', ...
      'sensor_fusionOR_acouORpiez_label', 'sensor_fusionOR_acouORpiez_comp', 'sensor_fusionOR_acouANDpiez_label', 'sensor_fusionOR_acouANDpiez_comp', ...
      'sensor_fusionAND_acou', 'sensor_fusionAND_piez', 'sensor_fusionAND_acouORpiez', 'sensor_fusionAND_acouANDpiez', ...
      'sensor_fusionAND_acou_label', 'sensor_fusionAND_acou_comp', 'sensor_fusionAND_piez_label', 'sensor_fusionAND_piez_comp', ...
      'sensor_fusionAND_acouORpiez_label', 'sensor_fusionAND_acouORpiez_comp', 'sensor_fusionAND_acouANDpiez_label', 'sensor_fusionAND_acouANDpiez_comp', ...
      '-v7.3');
cd(curr_dir)











