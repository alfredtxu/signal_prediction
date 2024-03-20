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

%% FM 
low_signal_quantile = 0.25;
noise_level = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];
% noise_level = [0.0005 0.0005 0.0005 0.0005 0.0005 0.0005];

% 3 / 4 apprear the best performance
% std_times = 1 : 5;
std_times = 4;

% initialize sensor suite
FM_thresh = zeros(num_files, num_FMsensors);
tmp_FM_segmented = cell(num_files, num_FMsensors);

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    for t = 1 : length(std_times)

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
            tmp_sensor_indiv_mu = mean(tmp_sensor_indiv);
            tmp_sensor_indiv_std = std(tmp_sensor_indiv);
    
            % determine the individual/adaptive threshold
            FM_thresh(i, t, j2) = tmp_sensor_indiv_mu + std_times(t) * tmp_sensor_indiv_std;
            
            if isnan(FM_thresh(i, t, j2)) 
                FM_thresh(i, t, j2) = Inf;
            elseif FM_thresh(i, t, j2) < noise_level
                FM_thresh(i, t, j2) = Inf; 
            end
    
            % binarising: above threshold - 1 otherwise - 0
            tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < FM_thresh(i, t, j2)); 
            tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= FM_thresh(i, t, j2)); 
            
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

            SNR(i, t, j2) = 10 * log10((p_signal / length(tmp_FM_segmented)) / (n_signal / length(tmp_FM_removed)));
            FM_segmented{i, t, j2} = (tmp_sensor_indiv >= FM_thresh(i, t, j2)); 

            fprintf('Current data file from %s (%s): %d of %d >> FM threshold: %d - SNR: %d - ratio: %d ...\n', ...
                data_folder, data_category, i, num_files, std_times(t), SNR(i, t, j2), length(tmp_FM_removed) / length(tmp_FM_segmented));

            clear tmp_sensor_indiv tmp_low_cutoff tmp_sensor_suite_low tmp_FM_removed tmp_FM_segmented
        end
        % -------------------------------------------------------------------------

    % end of loop for the IMUa / IMUg thresholds
    end

% end of loop for the files
end

SNA_mu = mean(SNR, 3);
SNA_std = mean(SNR, 3);






