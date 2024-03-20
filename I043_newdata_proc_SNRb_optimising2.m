clc
clear 
close all

% HOME
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';

% OFFICE
% curr_dir = pwd;

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
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

% OFFICE
% fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007', 
% data category: F - focus; D - day; N - night
data_folder = 'b_mat_003AM';
data_processed = 'c_processed_003AM';
data_category = 'focus';

fdir_p = [fdir '\' data_processed];
load([fdir_p '\FM01_' data_folder '_' data_category '_preproc.mat']);

% the number of data file
num_files = size(sens_T, 1);

%% Threshold
std_times = 4;
low_signal_quantile = 0.25;
acou_min = 30;
piez_min = 40;

%% FM 
% linear element necessary for dilation operation
FM_dilation_time = 3.0;
FM_dilation_size = round (FM_dilation_time * freq);
FM_lse = strel('line', FM_dilation_size, 90);

% low_signal_quantile = 0.25;
% FM_min = [30, 30, 30, 30, 30, 30];
noise_level = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];

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
% IMUa_thresh = 0.010;
% IMUg_thresh = 0.005;

%% sensation
% Parameters: sensation map and detection matching (Backward/Forward extension length in second)
ext_backward = 5.0;
ext_forward = 2.0 ;
sens_dil_backward = round(ext_backward*freq);
sens_dil_forward = round(ext_forward*freq);

% percentage of overlap between senstation and maternal movement (IMU)
overlap_perc = 0.20;
% -------------------------------------------------------------------------------------------------------

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    fprintf('Current data file from %s (%s): %d of %d ...\n', data_folder, data_category, i, num_files);

    % Segment & dilate the IMU data and to generate an IMU map. 
    % Dilating IMU accelerometer data
    % Dilate or expand the ROI's (points with value = 1) by dilation_size 
    % (half above and half below) defined by SE
    tmp_IMUa = abs(IMUaccNet_TF{i, :});

    tmp_IMUa_mu = mean(tmp_IMUa);
    tmp_IMUa_std = std(tmp_IMUa);
    IMUa_thresh(i, :) = tmp_IMUa_mu + std_times * tmp_IMUa_std;

    tmp_IMUa_P = tmp_IMUa(tmp_IMUa >= IMUa_thresh(i, :));
    tmp_IMUa_N = tmp_IMUa(tmp_IMUa < IMUa_thresh(i, :));

    p_signal_IMUa = 0;
    for pa = 1 : length(tmp_IMUa_P)
        p_signal_IMUa = p_signal_IMUa + tmp_IMUa_P(pa)^2;
    end

    n_signal_IMUa = 0;
    for na = 1 : length(tmp_IMUa_N)
        n_signal_IMUa = n_signal_IMUa + tmp_IMUa_N(na)^2;
    end

    SNR_IMUa(i, :) = 10 * log10((p_signal_IMUa / length(tmp_IMUa_P)) / (n_signal_IMUa / length(tmp_IMUa_N)));

    tmp_IMUa_map = tmp_IMUa >= IMUa_thresh(i, :);
    IMUa_map{i, :} = imdilate(tmp_IMUa_map, IMU_lse);
    % -----------------------------------------------------------------------------------------------

    tmp_IMUg = abs(IMUgyrNet_TF{i, :});

    tmp_IMUg_mu = mean(tmp_IMUg);
    tmp_IMUg_std = std(tmp_IMUg);
    IMUg_thresh(i, :) = tmp_IMUg_mu + std_times * tmp_IMUg_std;

    tmp_IMUg_P = tmp_IMUg(tmp_IMUg >= IMUg_thresh(i, :));
    tmp_IMUg_N = tmp_IMUg(tmp_IMUg < IMUg_thresh(i, :));

    p_signal_IMUg = 0;
    for pg = 1 : length(tmp_IMUg_P)
        p_signal_IMUg = p_signal_IMUg + tmp_IMUg_P(pg)^2;
    end

    n_signal_IMUg = 0;
    for ng = 1 : length(tmp_IMUg_N)
        n_signal_IMUg = n_signal_IMUg + tmp_IMUg_N(ng)^2;
    end

    SNR_IMUg(i) = 10 * log10((p_signal_IMUg / length(tmp_IMUg_P)) / (n_signal_IMUg / length(tmp_IMUg_N)));

    tmp_IMUg_map = tmp_IMUg >= IMUg_thresh(i, :);
    IMUg_map{i, :} = imdilate(tmp_IMUg_map, IMU_lse);
    % -----------------------------------------------------------------------------------------------

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
    clear tmp_IMUa tmp_IMUa_mu tmp_IMUa_std ... 
          tmp_IMUa_P tmp_IMUa_N p_signal_IMUa n_signal_IMUa tmp_IMUa_map ... 
          tmp_IMUg tmp_IMUg_mu tmp_IMUg_std ... 
          tmp_IMUg_P tmp_IMUg_N p_signal_IMUg n_signal_IMUg tmp_IMUg_map ...
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
    sensor_suite_preproc{i, 1} = acouL_TF{i,:};
    sensor_suite_preproc{i, 2} = acouM_TF{i,:};
    sensor_suite_preproc{i, 3} = acouR_TF{i,:};
    sensor_suite_preproc{i, 4} = piezL_TF{i,:};
    sensor_suite_preproc{i, 5} = piezM_TF{i,:};
    sensor_suite_preproc{i, 6} = piezR_TF{i,:};

    % Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(sensor_suite_preproc{i, j2});
        tmp_low_cutoff = quantile(tmp_sensor_indiv, low_signal_quantile); 
        tmp_sensor_suite_low = tmp_sensor_indiv(tmp_sensor_indiv <= tmp_low_cutoff); 

        % determine the individual/adaptive threshold
        if j2 <= num_FMsensors / 2
            FM_thresh(i, j2) = acou_min * median(tmp_sensor_suite_low);
        else
            FM_thresh(i, j2) = piez_min * median(tmp_sensor_suite_low);
        end

        if isnan(FM_thresh(i, j2))
            FM_thresh(i, j2) = Inf;
        elseif FM_thresh(i, j2) < noise_level
            FM_thresh(i, j2) = Inf;
        end

        % binarising: above threshold - 1 otherwise - 0
        tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < FM_thresh(i, j2));
        tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= FM_thresh(i, j2));

        % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
        % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
        % SNR = 10 * log10(P_signal / P_noise)
        p_signal = 0;
        for p = 1 : length(tmp_FM_segmented)
            p_signal = p_signal + tmp_FM_segmented(p)^2;
        end

        n_signal = 0;
        for n = 1 : length(tmp_FM_removed)
            n_signal = n_signal + tmp_FM_removed(n)^2;
        end

        SNR_FM(i, j2) = 10 * log10((p_signal / length(tmp_FM_segmented)) / (n_signal / length(tmp_FM_removed)));

        FM_segmented{i, j2} = (tmp_sensor_indiv >= FM_thresh(i, j2));

        % Exclusion of body movement data
        FM_segmented_IMUa{i, j2} = FM_segmented{i, j2} .* (1-IMUa_map{i,:}); 
        FM_segmented_IMUa{i, j2} = imdilate(FM_segmented_IMUa{i, j2}, FM_lse); 

        FM_segmented_IMUg{i, j2} = FM_segmented{i, j2} .* (1-IMUg_map{i,:}); 
        FM_segmented_IMUg{i, j2} = imdilate(FM_segmented_IMUg{i, j2}, FM_lse); 

        FM_segmented_IMUag{i, j2} = FM_segmented{i, j2} .* (1-IMUa_map{i,:}) .* (1-IMUg_map{i,:}); 
        FM_segmented_IMUag{i, j2} = imdilate(FM_segmented_IMUag{i, j2}, FM_lse); 

        FM_segmented{i, j2} = imdilate(FM_segmented{i, j2}, FM_lse); 
        
        %% match of FM and sensation
        [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(sens_map{i, :});
        [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(sensT_mapIMUag{i, :});
    
        [tmp_sensor_label, tmp_sensor_comp] = bwlabel(FM_segmented{i, j2});
        [tmp_sensor_IMUag_label, tmp_sensor_IMUag_comp] = bwlabel(FM_segmented_IMUag{i, j2});
    
        
        match_indiv_IMUag_sensation{i, j2} = tmp_sensor_IMUag_label .* sensT_mapIMUag{i, :};
        match_indiv_IMUag_sensation_num(i, j2) = length(unique(match_indiv_IMUag_sensation{i, j2})) - 1;
    
        match_indiv_IMUag_sensation_INV{i, j2} = tmp_sens_mapIMUag_label .* FM_segmented_IMUag{i, j2};
        match_indiv_IMUag_sensation_num_INV(i, j2) = length(unique(match_indiv_IMUag_sensation_INV{i, j2})) - 1;
        
        fprintf('Signal segmentation file %d - sensor %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
                i, j2, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_indiv_IMUag_sensation_num(i, j2));
    
        fprintf('Signal segmentation file %d - sensor %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
                i, j2, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_indiv_IMUag_sensation_num_INV(i, j2));


        clear tmp_sensor_indiv tmp_low_cutoff tmp_sensor_suite_low ...
              tmp_sens_map_label tmp_sens_map_comp tmp_sens_mapIMUag_label tmp_sens_mapIMUag_comp ...
              tmp_sensor_label tmp_sensor_comp tmp_sensor_IMUag_label tmp_sensor_IMUag_comp
    end
    % -------------------------------------------------------------------------
    
    % Sensor fusion group I:
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'OR'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    sensor_fusionOR_acou{i, :} = double(FM_segmented{i, 1} | FM_segmented{i, 2} | FM_segmented{i, 3});
    sensor_fusionOR_piez{i, :} = double(FM_segmented{i, 4} | FM_segmented{i, 5} | FM_segmented{i, 6});
    sensor_fusionOR_acouORpiez{i, :}  = double(sensor_fusionOR_acou{i,:} | sensor_fusionOR_piez{i, :});
    
    sensor_IMUag_fusionOR_acou{i, :} = double(FM_segmented_IMUag{i, 1} | FM_segmented_IMUag{i, 2} | FM_segmented_IMUag{i, 3});
    sensor_IMUag_fusionOR_piez{i, :} = double(FM_segmented_IMUag{i, 4} | FM_segmented_IMUag{i, 5} | FM_segmented_IMUag{i, 6});
    sensor_IMUag_fusionOR_acouORpiez{i, :}  = double(sensor_IMUag_fusionOR_acou{i,:} | sensor_IMUag_fusionOR_piez{i, :});
    % -------------------------------------------------------------------------

    %% match of FM and sensation
    [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(sens_map{i, :});
    [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(sensT_mapIMUag{i, :});

    [tmp_sensor_label, tmp_sensor_comp] = bwlabel(sensor_fusionOR_acouORpiez{i, :});
    [tmp_sensor_IMUag_label, tmp_sensor_IMUag_comp] = bwlabel(sensor_IMUag_fusionOR_acouORpiez{i, :});

    match_sensor_IMUag_sensation{i, :} = tmp_sensor_IMUag_label .* sensT_mapIMUag{i, :};
    match_sensor_IMUag_sensation_num(i, :) = length(unique(match_sensor_IMUag_sensation{i, :})) - 1;

    match_sensor_IMUag_sensation_INV{i, :} = tmp_sens_mapIMUag_label .* sensor_IMUag_fusionOR_acouORpiez{i, :};
    match_sensor_IMUag_sensation_num_INV(i, :) = length(unique(match_sensor_IMUag_sensation_INV{i, :})) - 1;
    % ---------------------------------------------------------------------------------------------------

    fprintf('Fusion > Signal segmentation file %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
            i, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_sensor_IMUag_sensation_num(i, :));

    fprintf('Fusion > Signal segmentation file %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
            i, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_sensor_IMUag_sensation_num_INV(i, :));

% end of loop for the files
end

% save segmented and fused data
% cd([fdir '\' data_processed]);
% save(['FM02_' data_folder '_' data_category '_proc.mat'], ...
%       'IMUa_thresh', 'SNR_IMUa', 'IMUa_map', 'IMUg_thresh', 'SNR_IMUg', 'IMUg_map', ...
%       'sens', 'sens_map', 'sensT_mapIMUa', 'sensT_mapIMUg', 'sensT_mapIMUag', ...
%       'sens_label', 'sens_activity', 'sens_map_label', 'sens_map_activity', ...
%       'sens_mapIMUa_label', 'sens_mapIMUa_activity', 'sens_mapIMUg_label', 'sens_mapIMUg_activity', ...
%       'sens_mapIMUag_label', 'sens_mapIMUag_activity', ...
%       'sensor_suite_preproc', ... 
%       'FM_thresh', 'SNR_FM', ...
%       'FM_segmented', 'FM_segmented_IMUa', 'FM_segmented_IMUg', 'FM_segmented_IMUag', ...
%       'sensor_fusionOR_acou', 'sensor_fusionOR_piez', 'sensor_fusionOR_acouORpiez', ...
%       'sensor_IMUag_fusionOR_acou', 'sensor_IMUag_fusionOR_piez', 'sensor_IMUag_fusionOR_acouORpiez', ...
%       'match_sensor_sensation_num', 'match_sensor_IMUag_sensation_num', ...
%       'match_sensorFusion_sensation_num', 'match_sensorFusion_IMUag_sensation_num', ...
%       '-v7.3');
% cd(curr_dir)











