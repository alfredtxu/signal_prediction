clc
clear 
% close all

% HOME / OFFICE
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% curr_dir = pwd;

cd(curr_dir);

%% Pre-setting & data loading
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
freq = 400;
num_channel = 16;
num_FMsensors = 6;

% HOME / OFFICE
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';
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

%% Thresholds for FM sensor & IMU
std_times_acou = 4;
std_times_piez = 4;
noise_level = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];

std_times_IMU = 4;

%% FM dilation
% Linear structuring element (vertical, as deg=90) that will have values 1 with a length of dilation_size (in seconds);
FM_dilation_time = 3.0;
FM_dilation_size = round (FM_dilation_time * freq);
FM_lse = strel('line', FM_dilation_size, 90);

%% IMU dilation
% Linear structuring element (vertical, as deg=90) that will have values 1 with a length of dilation_size (in seconds);
IMU_dilation_time = 4.0;
IMU_dilation_size = round(IMU_dilation_time*freq);
IMU_lse = strel('line', IMU_dilation_size, 90);

%% sensation dilation
% Parameters: sensation map and detection matching (Backward/Forward extension length in second)
sens_dilationB = 5.0;
sens_dilationF = 2.0 ;
sens_dilation_sizeB = round(sens_dilationB*freq);
sens_dilation_sizeF = round(sens_dilationF*freq);

%% FM moving average window size
FM_window = 3.0;

% percentage of overlap between senstation and maternal movement (IMU)
overlap_perc = 0.15;
% -------------------------------------------------------------------------------------------------------

% Plotting parameters
ylim_multiplier = 1.2;

sntn_multiplier = 1;
IMU_multiplier = 1;
fm_multiplier = 0.9;

line_width = 1;
box_width = 1; 

Font_size_labels = 16;
Font_size_legend = 14;

col_code1 = [0 0 0];
col_code2 = [0.4660 0.6740 0.1880];
col_code3 = [0.4940 0.1840 0.5560];
col_code4 = [0.8500 0.3250 0.0980];
col_code5 = [0.9290 0.6940 0.1250];

%% SEGMENTING PRE-PROCESSED DATA
% for i = 1 : num_files
for i = 8

    fprintf('Current data file from %s (%s): %d of %d ...\n', data_folder, data_category, i, num_files);
    
    % time vector
    tmp_timeV = (0 : length(IMUaccNet_TF{i,:}) - 1) / freq;

    % Segment & dilate the IMU data and to generate an IMU map. 
    tmp_IMUa = abs(IMUaccNet_TF{i, :});
    tmp_IMUa_mu = mean(tmp_IMUa);
    tmp_IMUa_std = std(tmp_IMUa);
    IMUa_thresh(i, :) = tmp_IMUa_mu + std_times_IMU * tmp_IMUa_std;
    tmp_IMUa_map = tmp_IMUa >= IMUa_thresh(i, :);
    IMUa_map_ori{i, :} = tmp_IMUa_map;
    IMUa_map{i, :} = imdilate(tmp_IMUa_map, IMU_lse);

    % SNR
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
    % -----------------------------------------------------------------------------------------------

    tmp_IMUg = abs(IMUgyrNet_TF{i, :});
    tmp_IMUg_mu = mean(tmp_IMUg);
    tmp_IMUg_std = std(tmp_IMUg);
    IMUg_thresh(i, :) = tmp_IMUg_mu + std_times_IMU * tmp_IMUg_std;
    tmp_IMUg_map = tmp_IMUg >= IMUg_thresh(i, :);
    IMUg_map_ori{i, :} = tmp_IMUg_map;
    IMUg_map{i, :} = imdilate(tmp_IMUg_map, IMU_lse);

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

    % Plot IMUs
    figure
    title('IMU');
    tiledlayout(6, 1, 'Padding', 'tight', 'TileSpacing', 'none');
    
    % Tile 1-8: IMUs
    % T1: orinal IMUa
    nexttile
    hold on;
    plot(tmp_timeV, IMUaccNet_TF{i, :}, 'LineWidth', line_width); 
    ylim([min(IMUaccNet_TF{i,:})*ylim_multiplier, max(IMUaccNet_TF{i,:})*ylim_multiplier]);
    hold off;
    
    % T2: orinal absoulate IMUa
    nexttile
    hold on;
    plot(tmp_timeV, tmp_IMUa, 'LineWidth', line_width); 
    ylim([0, max(tmp_IMUa)*ylim_multiplier]);
    hold off;

    % T3: dilated & thresholded absoulate IMUa
    nexttile
    hold on;
    plot(tmp_timeV, IMUa_map{i, :}, 'Color', col_code1, 'LineWidth', line_width);
    ylim([0, 1*ylim_multiplier]);
    hold off;

    % T4: orinal IMUg
    nexttile
    hold on;
    plot(tmp_timeV, IMUgyrNet_TF{i, :}, 'LineWidth', line_width); 
    ylim([min(IMUgyrNet_TF{i,:})*ylim_multiplier, max(IMUgyrNet_TF{i,:})*ylim_multiplier]);
    hold off;
    
    % T5: orinal absoulate IMU6
    nexttile
    hold on;
    plot(tmp_timeV, tmp_IMUg, 'LineWidth', line_width); 
    ylim([0, max(tmp_IMUg)*ylim_multiplier]);
    hold off;

    % T6: dilated & thresholded absoulate IMUg
    nexttile
    hold on;
    plot(tmp_timeV, IMUg_map{i, :}, 'Color', col_code1, 'LineWidth', line_width);
    ylim([0, 1*ylim_multiplier]);
    hold off;  
    % -----------------------------------------------------------------------------------------------

    % Sensation map: dilating every detection to the backward and forward range. 
    % Revome the overlapping with IMUacce_map (body movements). 
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
        tmp_idxS = tmp_idx - sens_dilation_sizeB; 
        tmp_idxE = tmp_idx + sens_dilation_sizeF; 

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

    % plot 
    figure
    tiledlayout(5, 1, 'Padding', 'tight', 'TileSpacing', 'none');
    
    % Tile 1-5: sensations
    % T1: orinal sensation
    nexttile
    hold on;
    plot(tmp_timeV, tmp_sens, 'LineWidth', line_width); 
    ylim([0, 1*ylim_multiplier]);
    hold off;

    % T2: dilated sensation maps
    nexttile
    hold on;
    plot(tmp_timeV, tmp_sens_map, 'LineWidth', line_width); 
    ylim([0, 1*ylim_multiplier]);
    hold off;

    % T1: dilated sensation maps excluded IMUa
    nexttile
    hold on;
    plot(tmp_timeV, tmp_sens_mapIMUa, 'LineWidth', line_width); 
    ylim([0, 1*ylim_multiplier]);
    hold off;

    % T1: dilated sensation maps excluded IMUg
    nexttile
    hold on;
    plot(tmp_timeV, tmp_sens_mapIMUg, 'LineWidth', line_width); 
    ylim([0, 1*ylim_multiplier]);
    hold off;

    % T1: dilated sensation maps excluded IMUa & IMUg
    nexttile
    hold on;
    plot(tmp_timeV, tmp_sens_mapIMUag, 'LineWidth', line_width); 
    ylim([0, 1*ylim_multiplier]);
    hold off;

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

        % moving average
        tmp_mLen = length(tmp_sensor_indiv) - FM_window * freq + 1;
        for w = 1 : tmp_mLen
            tmp_sensor_indiv_MA(w, :) = mean(tmp_sensor_indiv(w : w+FM_window * freq-1));
        end

        tmp_timeV_MA = (0 : length(tmp_sensor_indiv_MA) - 1) / freq;
        tmp_sensor_indiv_MAmu = mean(tmp_sensor_indiv_MA);
        tmp_sensor_indiv_MAstd = std(tmp_sensor_indiv_MA);

        % determine the individual/adaptive threshold
        if j2 <= num_FMsensors / 2
            tmp_std_times_FM = std_times_acou;
        else
            tmp_std_times_FM = std_times_piez;
        end
        FM_thresh(i, j2) = tmp_sensor_indiv_MAmu + tmp_std_times_FM * tmp_sensor_indiv_MAstd;

        if isnan(FM_thresh(i, j2))
            FM_thresh(i, j2) = Inf;
        elseif FM_thresh(i, j2) < noise_level
            FM_thresh(i, j2) = Inf;
        end

        % binarising: above threshold - 1 otherwise - 0
        tmp_FM_noise = tmp_sensor_indiv_MA(tmp_sensor_indiv_MA < FM_thresh(i, j2));
        tmp_FM_signals = tmp_sensor_indiv_MA(tmp_sensor_indiv_MA >= FM_thresh(i, j2));

        % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
        % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
        % SNR = 10 * log10(P_signal / P_noise)
        p_signal = 0;
        for p = 1 : length(tmp_FM_signals)
            p_signal = p_signal + tmp_FM_signals(p)^2;
        end

        n_signal = 0;
        for n = 1 : length(tmp_FM_noise)
            n_signal = n_signal + tmp_FM_noise(n)^2;
        end

        SNR_FM(i, j2) = 10 * log10((p_signal / length(tmp_FM_signals)) / (n_signal / length(tmp_FM_noise)));

        tmp_FM_segmented = (tmp_sensor_indiv_MA >= FM_thresh(i, j2));
        tmp_FM_segmented_dia = imdilate(tmp_FM_segmented, FM_lse); 

        % FM_segmented{i, j2} = tmp_FM_segmented;
        FM_segmented{i, j2} = tmp_FM_segmented_dia; 

        % Exclusion of body movement data
        FM_segmented_IMUa{i, j2} = tmp_FM_segmented_dia .* (1-IMUa_map{i,:}); 
        % FM_segmented_IMUa{i, j2} = imdilate(FM_segmented_IMUa{i, j2}, FM_lse); 

        FM_segmented_IMUg{i, j2} = tmp_FM_segmented_dia .* (1-IMUg_map{i,:}); 
        % FM_segmented_IMUg{i, j2} = imdilate(FM_segmented_IMUg{i, j2}, FM_lse); 

        FM_segmented_IMUag{i, j2} = tmp_FM_segmented_dia .* (1-IMUa_map{i,:}) .* (1-IMUg_map{i,:}); 
        % FM_segmented_IMUag{i, j2} = imdilate(FM_segmented_IMUag{i, j2}, FM_lse); 

        %% match of FM and sensation
        [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(sens_map{i, :});
        [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(sensT_mapIMUag{i, :});
    
        [tmp_sensor_label, tmp_sensor_comp] = bwlabel(FM_segmented{i, j2});
        [tmp_sensor_IMUag_label, tmp_sensor_IMUag_comp] = bwlabel(FM_segmented_IMUag{i, j2});
    
        match_indiv_sensation{i, j2} = tmp_sensor_label .* sens_map{i, :};
        match_indiv_sensation_num(i, j2) = length(unique(match_indiv_sensation{i, j2})) - 1;
        
        match_indiv_sensation_INV{i, j2} = tmp_sens_map_label .* FM_segmented{i, j2};
        match_indiv_sensation_num_INV(i, j2) = length(unique(match_indiv_sensation_INV{i, j2})) - 1;
    
        match_indiv_IMUag_sensation{i, j2} = tmp_sensor_IMUag_label .* sensT_mapIMUag{i, :};
        match_indiv_IMUag_sensation_num(i, j2) = length(unique(match_indiv_IMUag_sensation{i, j2})) - 1;
    
        match_indiv_IMUag_sensation_INV{i, j2} = tmp_sens_mapIMUag_label .* FM_segmented_IMUag{i, j2};
        match_indiv_IMUag_sensation_num_INV(i, j2) = length(unique(match_indiv_IMUag_sensation_INV{i, j2})) - 1;
    
%         fprintf('Signal segmentation file %d - sensor %d >> comp (sens / FM): %d / %d; match: %d ...\n', ...
%                 i, j2, tmp_sens_map_comp, tmp_sensor_comp, match_indiv_sensation_num(i, j2));
    
        fprintf('Signal segmentation file %d - sensor %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
                i, j2, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_indiv_IMUag_sensation_num(i, j2));

%         fprintf('Signal segmentation file %d - sensor %d >> comp (sens / FM): %d / %d; match: %d ...\n', ...
%                 i, j2, tmp_sens_map_comp, tmp_sensor_comp, match_indiv_sensation_num_INV(i, j2));
    
        fprintf('Signal segmentation file %d - sensor %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
                i, j2, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_indiv_IMUag_sensation_num_INV(i, j2));

        % plot FM sensors
        % plot 
        figure
        tiledlayout(9, 1, 'Padding', 'tight', 'TileSpacing', 'none');
        
        nexttile
        hold on;
        plot(tmp_timeV, IMUa_map{i, :}, 'LineWidth', line_width); 
        ylim([0, 1*ylim_multiplier]);
        hold off;

        nexttile
        hold on;
        plot(tmp_timeV, IMUg_map{i, :}, 'LineWidth', line_width); 
        ylim([0, 1*ylim_multiplier]);
        hold off;

        nexttile
        hold on;
        plot(tmp_timeV, sensT_mapIMUag{i, :}, 'LineWidth', line_width); 
        ylim([0, 1*ylim_multiplier]);
        hold off;

        % T1: orignal sensor data
        nexttile
        hold on;
        plot(tmp_timeV_MA, tmp_sensor_indiv_MA, 'LineWidth', line_width); 
        ylim([0, max(tmp_sensor_indiv_MA)*ylim_multiplier]);
        hold off;
         
        % T2: segmented
        nexttile
        hold on;
        plot(tmp_timeV_MA, tmp_sensor_indiv_MA.*tmp_FM_segmented, 'LineWidth', line_width); 
        ylim([0, max(tmp_sensor_indiv_MA)*ylim_multiplier]);
        hold off;
         

        % T3: dialiated
        nexttile
        hold on;
        plot(tmp_timeV_MA, tmp_sensor_indiv_MA.*tmp_FM_segmented_dia, 'LineWidth', line_width); 
        ylim([0, max(tmp_sensor_indiv_MA)*ylim_multiplier]);
        hold off;
         

        % T4: dialiated + IMua
        nexttile
        hold on;
        plot(tmp_timeV, tmp_sensor_indiv_MA.*FM_segmented_IMUa{i, j2}, 'LineWidth', line_width); 
        ylim([0, max(tmp_sensor_indiv_MA)*ylim_multiplier]);
        hold off;
         

        % T5: dialiated + IMUg
        nexttile
        hold on;
        plot(tmp_timeV, tmp_sensor_indiv_MA.*FM_segmented_IMUg{i, j2}, 'LineWidth', line_width); 
        ylim([0, max(tmp_sensor_indiv_MA)*ylim_multiplier]);
        hold off;
         

        % T6: dialiated + IMUa + IMUg
        nexttile
        hold on;
        plot(tmp_timeV, tmp_sensor_indiv_MA.*FM_segmented_IMUag{i, j2}, 'LineWidth', line_width); 
        ylim([0, max(tmp_sensor_indiv_MA)*ylim_multiplier]);
        hold off;
         

        clear tmp_sensor_indiv_MA tmp_low_cutoff tmp_sensor_suite_low ...
              tmp_sens_map_label tmp_sens_map_comp tmp_sens_mapIMUag_label tmp_sens_mapIMUag_comp ...
              tmp_sensor_label tmp_sensor_comp tmp_sensor_IMUag_label tmp_sensor_IMUag_comp ...
              tmp_std_times_FM tmp_FM_segmented tmp_FM_segmented_dia
    end
    % -------------------------------------------------------------------------
    
    
% end of loop for the files
end











