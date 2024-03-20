% This function is desinated for signal segmentation with a range of empricial thresholds for optimzing the segmenting performance
% 2023.10.02
% T.Xu

clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir);

%% Pre-setting & data loading
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
freq = 400;
num_channel = 16;
num_FMsensors = 6;

% data file directory
fprintf('Loading data files ...\n');
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007',
% data category: F - focus; D - day; N - night
data_folder = 'b_mat_003AM';
data_processed = 'c_processed_003AM';
data_category = 'focus';

fdir_p = [fdir '\' data_processed];
load([fdir_p '\FM01_' data_folder '_' data_category '_preproc.mat']);

% the number of data file
num_files = size(sens_T, 1);

%% IMU
IMU_dilation_time = 4.0;
IMU_dilation_size = round(IMU_dilation_time*freq);
IMU_lse = strel('line', IMU_dilation_size, 90);

% 3 /4 apprears to be the best performance
% std_times = 1 : 5;
std_times = 4;

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
overlap_perc = 0.15;

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    for t = 1 : length(std_times)

        tmp_IMUa = abs(IMUaccNet_TF{i, :});
        
        tmp_IMUa_mu = mean(tmp_IMUa);
        tmp_IMUa_std = std(tmp_IMUa);
        IMUa_thresh(i, t) = tmp_IMUa_mu + std_times(t) * tmp_IMUa_std;

        tmp_IMUa_P = tmp_IMUa(abs(tmp_IMUa) >= IMUa_thresh(i, t));
        tmp_IMUa_N = tmp_IMUa(abs(tmp_IMUa) < IMUa_thresh(i, t));

        p_signal_IMUa = 0;
        for pa = 1 : length(tmp_IMUa_P)
            p_signal_IMUa = p_signal_IMUa + tmp_IMUa_P(pa)^2;
        end

        n_signal_IMUa = 0;
        for na = 1 : length(tmp_IMUa_N)
            n_signal_IMUa = n_signal_IMUa + tmp_IMUa_N(na)^2;
        end

        SNR_IMUa(i, t) = 10 * log10((p_signal_IMUa / length(tmp_IMUa_P)) / (n_signal_IMUa / length(tmp_IMUa_N)));

        tmp_IMUa_map = abs(tmp_IMUa) >= IMUa_thresh(i, t);
        IMUa_map{i, t, :} = imdilate(tmp_IMUa_map, IMU_lse);
        % -----------------------------------------------------------------------------------------------

        tmp_IMUg = abs(IMUgyrNet_TF{i, :});

        tmp_IMUg_mu = mean(tmp_IMUg);
        tmp_IMUg_std = std(tmp_IMUg);
        IMUg_thresh(i, t) = tmp_IMUg_mu + std_times(t) * tmp_IMUg_std;

        tmp_IMUg_P = tmp_IMUg(abs(tmp_IMUg) >= IMUg_thresh(i, t));
        tmp_IMUg_N = tmp_IMUg(abs(tmp_IMUg) < IMUg_thresh(i, t));

        p_signal_IMUg = 0;
        for pg = 1 : length(tmp_IMUg_P)
            p_signal_IMUg = p_signal_IMUg + tmp_IMUg_P(pg)^2;
        end

        n_signal_IMUg = 0;
        for ng = 1 : length(tmp_IMUg_N)
            n_signal_IMUg = n_signal_IMUg + tmp_IMUg_N(ng)^2;
        end

        SNR_IMUg(i, t) = 10 * log10((p_signal_IMUg / length(tmp_IMUg_P)) / (n_signal_IMUg / length(tmp_IMUg_N)));

        tmp_IMUg_map = abs(tmp_IMUg) >= IMUg_thresh(i, t);
        IMUg_map{i, t, :} = imdilate(tmp_IMUg_map, IMU_lse);

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
            tmp_overlaps_acc = sum(tmp_sens_map(tmp_idxS:tmp_idxE) .* IMUa_map{i, t, :}(tmp_idxS:tmp_idxE));
            if (tmp_overlaps_acc >= overlap_perc*(tmp_idxE-tmp_idxS+1))
                tmp_sens_mapIMUa(tmp_idxS:tmp_idxE) = 0;
                tmp_sens_mapIMUag(tmp_idxS:tmp_idxE) = 0;
            end

            tmp_overlaps_gyr = sum(tmp_sens_map(tmp_idxS:tmp_idxE) .* IMUg_map{i, t, :}(tmp_idxS:tmp_idxE));
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
        %
        % sens: orginal maternal button pressed
        % sens map: the dialiated linear data from sens (7s unit [-5 2])
        % sens map IMUa: sens maps excluded overlap (threshold: 10%) with IMU acc
        % sens map IMUg: sens maps excluded overlap (threshold: 10%) with IMU gyr
        [tmp_sens_label, tmp_sens_comp] = bwlabel(tmp_sens);
        sens_label{i, t, :} = tmp_sens_label;
        sens_activity(i, t, :) = tmp_sens_comp;

        [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(tmp_sens_map);
        sens_map_label{i, t, :} = tmp_sens_map_label;
        sens_map_activity(i, t, :) = tmp_sens_map_comp;

        [tmp_sens_mapIMUa_label, tmp_sens_mapIMUa_comp] = bwlabel(tmp_sens_mapIMUa);
        sens_mapIMUa_label{i, t, :} = tmp_sens_mapIMUa_label;
        sens_mapIMUa_activity(i, t, :) = tmp_sens_mapIMUa_comp;

        [tmp_sens_mapIMUg_label, tmp_sens_mapIMUg_FMcomp] = bwlabel(tmp_sens_mapIMUg);
        sens_mapIMUg_label{i, t, :} = tmp_sens_mapIMUg_label;
        sens_mapIMUg_activity(i, t, :) = tmp_sens_mapIMUg_FMcomp;

        [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(tmp_sens_mapIMUag);
        sens_mapIMUag_label{i, t, :} = tmp_sens_mapIMUag_label;
        sens_mapIMUag_activity(i, t, :) = tmp_sens_mapIMUag_comp;

        sens_map{i, t, :} = tmp_sens_map;
        sensT_mapIMUa{i, t, :} = tmp_sens_mapIMUa;
        sensT_mapIMUg{i, t, :} = tmp_sens_mapIMUg;
        sensT_mapIMUag{i, t, :} = tmp_sens_mapIMUag;

        fprintf(['Current data file from %s (%s): %d of %d >> ' ...
                 'IMUa: %d; IMUg: %d; ' ...
                 'SNR IMUa: %d; SNR IMUa: %d; ' ...
                 'sens_activity: %d; ' ...
                 'sens_map_activity: %d; ' ...
                 'sens_mapIMUa_activeity: %d; ' ...
                 'sens_mapIMUg_activeity: %d; ' ...
                 'sens_mapIMUag_activeity: %d ... \n'], ...
                 data_folder, data_category, i, num_files, ...
                 IMUa_thresh(t), IMUg_thresh(t), ...
                 SNR_IMUa(i, t), SNR_IMUg(i, t), ...
                 tmp_sens_comp, tmp_sens_map_comp, ...
                 tmp_sens_mapIMUa_comp, tmp_sens_mapIMUg_FMcomp, tmp_sens_mapIMUag_comp);


        % clear temporal variables
        clear tmp_IMUa tmp_IMUa_P tmp_IMUa_N tmp_IMUa_map ...
              tmp_IMUg tmp_IMUg_P tmp_IMUg_N tmp_IMUg_map ...
              p_signal_IMUa n_signal_IMUa p_signal_IMUg p_signal_IMUg ...
              tmp_sens tmp_sens_idx ...
              tmp_sens_map tmp_sens_mapIMUa tmp_sens_mapIMUg tmp_sens_mapIMUag ...
              tmp_sens_label tmp_sens_comp ...
              tmp_sens_map_label tmp_sens_map_comp ...
              tmp_sens_mapIMUa_label tmp_sens_mapIMUa_comp ...
              tmp_sens_mapIMUg_label tmp_sens_mapIMUg_FMcomp ...
              tmp_sens_mapIMUag_label tmp_sens_mapIMUag_comp       
 
    end
end














