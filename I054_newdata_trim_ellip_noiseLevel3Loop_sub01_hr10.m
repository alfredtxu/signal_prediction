% 1. remove the singals below noise level (extreme low singals)
% 2. remove the lowest 25% signals (FM sensors only), then calculating the adaptive thresholds on the remaining 75%
clc
clear
close all

% HOME
% curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

% OFFICE
curr_dir = pwd;
fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

cd(curr_dir);

%% Variables
% Frequency (Hz): sensor / sensation
freq = 400;

% the number of channels / FM sensors
num_channel = 16;
num_FMsensors = 6;

% thresholds: noise and low quantile level, and corresponding weights
noise_quan = 0.05;
low_quan = 0.25;

noise_quanW = 1.10;
low_quanW = 1.10;

% FM: Morphological structuring element for dilating the FM segmentation
FM_dilation_time = 3.0;
FM_dilation_size = round (FM_dilation_time * freq);
FM_lse = strel('line', FM_dilation_size, 90);

% IMU: Morphological structuring element for dilating the body movement segmentation
IMU_dilation_time = 4.0;
IMU_dilation_size = round(IMU_dilation_time*freq);
IMU_lse = strel('line', IMU_dilation_size, 90);

% Sensation: Morphological structuring element for dilating the maternal button press segmentation
sens_dilation_timeB = 5.0;
sens_dilation_timeF = 2.0 ;
sens_dilation_sizeB = round(sens_dilation_timeB*freq);
sens_dilation_sizeF = round(sens_dilation_timeF*freq);

% thresholds (mu + N * std)
Nstd_IMU_TE = 3;
Nstd_acou_TE = 1;
Nstd_piez_TE = 1;

% thresholds: singal-noise ratio
snr_IMU_th = 25;
snr_acou_th = 20:25;
snr_piez_th = 20:25;

% percentage of overlap between senstation and maternal movement (IMU)
overlap_percSens = 0.20;
overlap_percFM = 0.01;
% -------------------------------------------------------------------------------------------------------

%% Data loading
% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007',
% data category: F - focus; D - day; N - night
participant = 'all';
data_folder = ['c_processed_' participant];
data_category = 'focus';
data_serial_common = 'common';
data_serial_case = 'ellip';

fdir_p = [fdir '\' data_folder '_F_hr10'];
preproc_dfile_common = ['Preproc011_' participant '_' data_category '_' data_serial_common '.mat'];
preproc_dfile_ellip = ['Preproc013_' participant '_' data_category '_' data_serial_case '.mat'];

load([fdir_p '\' preproc_dfile_common]);
load([fdir_p '\' preproc_dfile_ellip]);

% Renaming
forc = forc_hr10 ;
forcT = forcT_hr10 ;
forcTE = forcTE_hr10 ;

acouL = acouL_hr10 ;
acouLT = acouLT_hr10 ;
acouLTE = acouLTE_hr10 ;
acouM = acouM_hr10 ;
acouMT = acouMT_hr10 ;
acouMTE = acouMTE_hr10 ;
acouR = acouR_hr10 ;
acouRT = acouRT_hr10 ;
acouRTE = acouRTE_hr10 ;

piezL = piezL_hr10 ;
piezLT = piezLT_hr10 ;
piezLTE = piezLTE_hr10 ;
piezM = piezM_hr10 ;
piezMT = piezMT_hr10 ;
piezMTE = piezMTE_hr10 ;
piezR = piezR_hr10 ;
piezRT = piezRT_hr10 ;
piezRTE = piezRTE_hr10 ;

IMUacc = IMUacc_hr10 ;
IMUaccNet = IMUaccNet_hr10 ;
IMUaccNetT = IMUaccNetT_hr10 ;
IMUaccNetTE = IMUaccNetTE_hr10 ;
IMUaccT = IMUaccT_hr10 ;
IMUgyr = IMUgyr_hr10 ;
IMUgyrNet = IMUgyrNet_hr10 ;
IMUgyrNetT = IMUgyrNetT_hr10 ;
IMUgyrNetTE = IMUgyrNetTE_hr10 ;
IMUgyrT = IMUgyrT_hr10 ;

sens = sens_hr10 ;
sensT = sensT_hr10 ;
sensT_idxP = sensT_idxP_hr10 ;
sensT_MA_B = sensT_MA_B_hr10 ;
sensT_MA = sensT_MA_hr10 ;
sensT_MA_idxP = sensT_MA_idxP_hr10 ;
sensT_MA_mtx = sensT_MA_mtx_hr10 ;
sensT_MA_mtxP = sensT_MA_mtxP_hr10 ;
sensT_MA_thresh = sensT_MA_thresh_hr10 ;
sensT_MA_thresh_idxP = sensT_MA_thresh_idxP_hr10 ;
sensT_MA_thresh_mtx = sensT_MA_thresh_mtx_hr10 ;
sensT_MA_thresh_mtxP = sensT_MA_thresh_mtxP_hr10 ;
sensT_MA_threshB = sensT_MA_threshB_hr10 ;
sensT_MA_threshB_idxP = sensT_MA_threshB_idxP_hr10 ;
sensT_MA_threshB_mtx = sensT_MA_threshB_mtx_hr10 ;
sensT_MA_threshB_mtxP = sensT_MA_threshB_mtxP_hr10 ;
sensT_MALoss = sensT_MALoss_hr10 ;
sensT_mtx = sensT_mtx_hr10 ;
sensT_mtxP = sensT_mtxP_hr10 ;

clear acouL_hr10 acouLT_hr10 acouLTE_hr10 ...
      acouM_hr10 acouMT_hr10 acouMTE_hr10 ...
      acouR_hr10 acouRT_hr10 acouRTE_hr10 ...
      forc_hr10 forcT_hr10 forcTE_hr10 ...
      IMUacc_hr10 IMUaccNet_hr10 IMUaccNetT_hr10 IMUaccNetTE_hr10 IMUaccT_hr10 ...
      IMUgyr_hr10 IMUgyrNet_hr10 IMUgyrNetT_hr10 IMUgyrNetTE_hr10 IMUgyrT_hr10 ...
      piezL_hr10 piezLT_hr10 piezLTE_hr10 ...
      piezM_hr10 piezMT_hr10 piezMTE_hr10 ...
      piezR_hr10 piezRT_hr10 piezRTE_hr10 ...
      sens_hr10 sensT_hr10 sensT_idxP_hr10 ...
      sensT_MA_B_hr10 sensT_MA_hr10 sensT_MA_idxP_hr10 ...
      sensT_MA_mtx_hr10 sensT_MA_mtxP_hr10 ...
      sensT_MA_thresh_hr10 sensT_MA_thresh_idxP_hr10 sensT_MA_thresh_mtx_hr10 sensT_MA_thresh_mtxP_hr10 ...
      sensT_MA_threshB_hr10 sensT_MA_threshB_idxP_hr10 sensT_MA_threshB_mtx_hr10 sensT_MA_threshB_mtxP_hr10 ...
      sensT_MALoss_hr10 sensT_mtx_hr10 sensT_mtxP_hr10

% the number of data files
num_files = size(sensT, 1);

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    fprintf('Current data file from %d (%d) ...\n', i, num_files);

    for n1 = 1 : length(Nstd_IMU_TE)

        %% IMUaTE thrsholding
        % SNR: IMUaTE
        % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
        % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
        % SNR = 10 * log10(P_signal / P_noise)
        tmp_SNR_IMUaTE = 0;
        tmp_Nstd_IMU_TE = Nstd_IMU_TE(n1);
        tmp_IMUaTE = abs(IMUaccNetTE{i, :});

        while tmp_SNR_IMUaTE <= snr_IMU_th

            tmp_IMUaTE_th = mean(tmp_IMUaTE) + tmp_Nstd_IMU_TE * std(tmp_IMUaTE);

            tmp_IMUaTE_P = tmp_IMUaTE(tmp_IMUaTE >= tmp_IMUaTE_th);
            tmp_IMUaTE_N = tmp_IMUaTE(tmp_IMUaTE < tmp_IMUaTE_th);

            p_signal_IMUaTE = 0;
            for pa = 1 : length(tmp_IMUaTE_P)
                p_signal_IMUaTE = p_signal_IMUaTE + tmp_IMUaTE_P(pa)^2;
            end
            n_signal_IMUaTE = 0;
            for na = 1 : length(tmp_IMUaTE_N)
                n_signal_IMUaTE = n_signal_IMUaTE + tmp_IMUaTE_N(na)^2;
            end

            tmp_SNR_IMUaTE = 10 * log10((p_signal_IMUaTE / length(tmp_IMUaTE_P)) / (n_signal_IMUaTE / length(tmp_IMUaTE_N)));

            % increase the threshold weights if the SNR is not sufficient
            tmp_Nstd_IMU_TE = tmp_Nstd_IMU_TE + 1;

        end

        IMUaTE_th(i, n1).thresh = tmp_IMUaTE_th;
        IMUaTE_th(i, n1).Nstd = tmp_Nstd_IMU_TE;
        IMUaTE_th(i, n1).SNR = tmp_SNR_IMUaTE;

        fprintf('---- IMUa ellip filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUaTE_th, tmp_Nstd_IMU_TE-1, tmp_SNR_IMUaTE);

        % IMUa maps
        IMUaTE_map{i, n1, :} = tmp_IMUaTE >= IMUaTE_th(i, n1).thresh;
        IMUaTE_mapDil{i, n1, :} = imdilate(IMUaTE_map{i, n1, :}, IMU_lse);

        clear tmp_SNR_IMUaTE tmp_Nstd_IMU_TE tmp_IMUaTE tmp_IMUaTE_th tmp_IMUaTE_P tmp_IMUaTE_N
        % -----------------------------------------------------------------------------------------------

        %% IMUgTE thrsholding
        % SNR: IMUgTE
        tmp_SNR_IMUgTE = 0;
        tmp_Nstd_IMU_TE = Nstd_IMU_TE(n1);
        tmp_IMUgTE = abs(IMUgyrNetTE{i, :});

        while tmp_SNR_IMUgTE <= snr_IMU_th

            tmp_IMUgTE_th = mean(tmp_IMUgTE) + tmp_Nstd_IMU_TE * std(tmp_IMUgTE);

            tmp_IMUgTE_P = tmp_IMUgTE(tmp_IMUgTE >= tmp_IMUgTE_th);
            tmp_IMUgTE_N = tmp_IMUgTE(tmp_IMUgTE < tmp_IMUgTE_th);

            p_signal_IMUgTE = 0;
            for pa = 1 : length(tmp_IMUgTE_P)
                p_signal_IMUgTE = p_signal_IMUgTE + tmp_IMUgTE_P(pa)^2;
            end
            n_signal_IMUgTE = 0;
            for na = 1 : length(tmp_IMUgTE_N)
                n_signal_IMUgTE = n_signal_IMUgTE + tmp_IMUgTE_N(na)^2;
            end

            tmp_SNR_IMUgTE = 10 * log10((p_signal_IMUgTE / length(tmp_IMUgTE_P)) / (n_signal_IMUgTE / length(tmp_IMUgTE_N)));

            % increase the threshold weights if the SNR is not sufficient
            tmp_Nstd_IMU_TE = tmp_Nstd_IMU_TE + 1;

        end

        IMUgTE_th(i, n1).thresh = tmp_IMUgTE_th;
        IMUgTE_th(i, n1).Nstd = tmp_Nstd_IMU_TE;
        IMUgTE_th(i, n1).SNR = tmp_SNR_IMUgTE;

        fprintf('---- IMUg ellip filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUgTE_th, tmp_Nstd_IMU_TE-1, tmp_SNR_IMUgTE);

        % IMUg maps
        IMUgTE_map{i, n1, :} = tmp_IMUgTE >= IMUgTE_th(i, n1).thresh;
        IMUgTE_mapDil{i, n1, :} = imdilate(IMUgTE_map{i, n1, :}, IMU_lse);

        clear tmp_SNR_IMUgTE tmp_Nstd_IMU_TE tmp_IMUgTE tmp_IMUgTE_th tmp_IMUgTE_P tmp_IMUgTE_N
        % -----------------------------------------------------------------------------------------------

        %% Sensation
        % Initializing the map: sensT
        % senstation labels by connected components
        tmp_sensT = sensT{i, :};
        [tmp_sensTL, tmp_sensTC] = bwlabel(tmp_sensT);

        % initialization of sensation maps
        tmp_sensT_map = zeros(length(tmp_sensT), 1);
        tmp_sensTE_mapIMUa = zeros(length(tmp_sensT), 1);
        tmp_sensTE_mapIMUg = zeros(length(tmp_sensT), 1);
        tmp_sensTE_mapIMUag = zeros(length(tmp_sensT), 1);

        for j1 = 1 : tmp_sensTC

            % the idx range of the current cluster (component)
            tmp_idx = find(tmp_sensTL == j1);

            tmp_idxS = min(tmp_idx) - sens_dilation_sizeB;
            tmp_idxE = max(tmp_idx) + sens_dilation_sizeF;

            tmp_idxS = max(tmp_idxS, 1);
            tmp_idxE = min(tmp_idxE, length(tmp_sensT_map));

            % sensation map
            tmp_sensT_map(tmp_idxS:tmp_idxE) = 1;

            tmp_sensTE_mapIMUa(tmp_idxS:tmp_idxE) = 1;
            tmp_sensTE_mapIMUg(tmp_idxS:tmp_idxE) = 1;
            tmp_sensTE_mapIMUag(tmp_idxS:tmp_idxE) = 1;

            % Remove maternal sensation coincided with body movement (IMU a/g) - ellip
            tmp_xIMUaTE = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUaTE_mapDil{i, n1, :}(tmp_idxS:tmp_idxE));
            tmp_xIMUgTE = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUgTE_mapDil{i, n1, :}(tmp_idxS:tmp_idxE));

            if (tmp_xIMUaTE >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
                tmp_sensTE_mapIMUa(tmp_idxS:tmp_idxE) = 0;
                tmp_sensTE_mapIMUag(tmp_idxS:tmp_idxE) = 0;
            end

            if (tmp_xIMUgTE >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
                tmp_sensTE_mapIMUg(tmp_idxS:tmp_idxE) = 0;
                tmp_sensTE_mapIMUag(tmp_idxS:tmp_idxE) = 0;
            end

            clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTE tmp_xIMUgTE
        end

        % sensation maps
        sensT_map{i, n1, :} = tmp_sensT_map;
        sensTE_mapIMUa{i, n1, :} = tmp_sensTE_mapIMUa;
        sensTE_mapIMUg{i, n1, :} = tmp_sensTE_mapIMUg;
        sensTE_mapIMUag{i, n1, :} = tmp_sensTE_mapIMUag;

        [tmp_sensT_mapL, tmp_sensT_mapC] = bwlabel(tmp_sensT_map);
        [tmp_sensTE_mapIMUaL, tmp_sensTE_mapIMUaC] = bwlabel(tmp_sensTE_mapIMUa);
        [tmp_sensTE_mapIMUgL, tmp_sensTE_mapIMUgC] = bwlabel(tmp_sensTE_mapIMUg);
        [tmp_sensTE_mapIMUagL, tmp_sensTE_mapIMUagC] = bwlabel(tmp_sensTE_mapIMUag);

        sensT_mapL{i, n1, :} = tmp_sensT_mapL;
        sensT_mapC(i, n1, :) = tmp_sensT_mapC;
        sensTE_mapIMUaL{i, n1, :} = tmp_sensTE_mapIMUaL;
        sensTE_mapIMUaC(i, n1, :) = tmp_sensTE_mapIMUaC;
        sensTE_mapIMUgL{i, n1, :} = tmp_sensTE_mapIMUgL;
        sensTE_mapIMUgC(i, n1, :) = tmp_sensTE_mapIMUgC;
        sensTE_mapIMUagL{i, n1, :} = tmp_sensTE_mapIMUagL;
        sensTE_mapIMUagC(i, n1, :) = tmp_sensTE_mapIMUagC;

        fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d - %d - %d ... \n', ...
            sensT_mapC(i, n1, :), sensTE_mapIMUaC(i, n1, :), sensTE_mapIMUgC(i, n1, :), sensTE_mapIMUagC(i, n1, :));

        clear tmp_sensT tmp_sensTL tmp_sensTC tmp_sensT_map ...
            tmp_sensTE_mapIMUa tmp_sensTE_mapIMUg tmp_sensTE_mapIMUag ...
            tmp_sensT_mapL tmp_sensT_mapC ...
            tmp_sensTE_mapIMUaL tmp_sensTE_mapIMUaC ...
            tmp_sensTE_mapIMUgL tmp_sensTE_mapIMUgC ...
            tmp_sensTE_mapIMUagL tmp_sensTE_mapIMUagC
        % --------------------------------------------------------------------------------

        %% Segmenting FM sensors
        % 1. apply for an adaptive threshold
        % 2. remove body movement
        % 3. dilate the fetal movements at desinated duration
        FM_suiteTE{i, 1} = acouLTE{i,:};
        FM_suiteTE{i, 2} = acouMTE{i,:};
        FM_suiteTE{i, 3} = acouRTE{i,:};
        FM_suiteTE{i, 4} = piezLTE{i,:};
        FM_suiteTE{i, 5} = piezMTE{i,:};
        FM_suiteTE{i, 6} = piezRTE{i,:};

        for n2 = 1 : length(snr_acou_th)

            for w1 = 1 : length(noise_quanW)

                for w2 = 1 : length(low_quanW)

                    % FM Sensors Thresholding
                    for j2 = 1 : num_FMsensors

                        tmp_sensor_indiv = abs(FM_suiteTE{i, j2});

                        % remove noise level
                        tmp_noise_cutoff = quantile(tmp_sensor_indiv, noise_quan);
                        tmp_sensor_indiv_noise = tmp_sensor_indiv;
                        tmp_sensor_indiv_noise(tmp_sensor_indiv_noise<=tmp_noise_cutoff*noise_quanW(w1), :) = [];

                        % remove low level
                        tmp_low_cutoff = quantile(tmp_sensor_indiv_noise, low_quan);
                        tmp_sensor_indiv_cutoff = tmp_sensor_indiv_noise;
                        tmp_sensor_indiv_cutoff(tmp_sensor_indiv_cutoff<=tmp_low_cutoff*low_quanW(w2), :) = [];

                        tmp_SNR_sensor = 0;

                        if j2 <= num_FMsensors / 2
                            tmp_std_times_FM = Nstd_acou_TE;
                            tmp_snr_FM = snr_acou_th(n2);
                        else
                            tmp_std_times_FM = Nstd_piez_TE;
                            tmp_snr_FM = snr_piez_th(n2);
                        end

                        while tmp_SNR_sensor <= tmp_snr_FM

                            tmp_FM_threshTE = mean(tmp_sensor_indiv_cutoff) + tmp_std_times_FM * std(tmp_sensor_indiv_cutoff);
                            if isnan(tmp_FM_threshTE)
                                tmp_FM_threshTE = 0;
                            end

                            tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < tmp_FM_threshTE);
                            tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= tmp_FM_threshTE);

                            p_signal = 0;
                            for p = 1 : length(tmp_FM_segmented)
                                p_signal = p_signal + tmp_FM_segmented(p)^2;
                            end

                            n_signal = 0;
                            for n = 1 : length(tmp_FM_removed)
                                n_signal = n_signal + tmp_FM_removed(n)^2;
                            end

                            tmp_SNR_sensor = 10 * log10((p_signal / length(tmp_FM_segmented)) / (n_signal / length(tmp_FM_removed)));

                            % increase the threshold weights if the SNR is not sufficient
                            tmp_std_times_FM = tmp_std_times_FM + 1;
                        end

                        FM_TE_th(i, n1, n2, w1, w2, j2).thresh = tmp_FM_threshTE;
                        FM_TE_th(i, n1, n2, w1, w2, j2).Nstd = tmp_std_times_FM;
                        FM_TE_th(i, n1, n2, w1, w2, j2).SNE = tmp_SNR_sensor;

                        fprintf('---- FM(%d-th sensor) ellip filtering: threshold > %d; SNR (%d*std): %d ... \n', ...
                            j2, tmp_FM_threshTE, tmp_std_times_FM-1, tmp_SNR_sensor);

                        clear tmp_SNR_sensor tmp_std_times_FM tmp_snr_FM ...
                            tmp_FM_removed tmp_FM_segmented tmp_FM_threshTE
                        % --------------------------------------------------------------------------------------------------------------

                        %% segmentation by thresholding
                        FM_segTE{i, n1, n2, w1, w2, j2} = (tmp_sensor_indiv >= FM_TE_th(i, n1, n2, w1, w2, j2).thresh);
                        FM_segTE_dil{i, n1, n2, w1, w2, j2} = double(imdilate(FM_segTE{i, n1, n2, w1, w2, j2}, FM_lse));

                        % labels by connected components
                        [tmp_LTE, tmp_CTE] = bwlabel(FM_segTE{i, n1, n2, w1, w2, j2});
                        [tmp_LTE_dil, tmp_CTE_dil] = bwlabel(FM_segTE_dil{i, n1, n2, w1, w2, j2});

                        % Method 1 - denoise segmentations by removing movements
                        FM_segTE_dil_IMUa_m1{i, n1, n2, w1, w2, j2} = FM_segTE_dil{i, n1, n2, w1, w2, j2} .* (1-IMUaTE_mapDil{i,n1,:});
                        FM_segTE_dil_IMUg_m1{i, n1, n2, w1, w2, j2} = FM_segTE_dil{i, n1, n2, w1, w2, j2} .* (1-IMUgTE_mapDil{i,n1,:});
                        FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, j2} = FM_segTE_dil{i, n1, n2, w1, w2, j2} .* (1-IMUaTE_mapDil{i,n1,:}) .* (1-IMUgTE_mapDil{i,n1,:});

                        % labels by connected components
                        [tmp_LTE_IMUa_m1, tmp_CTE_IMUa_m1] = bwlabel(FM_segTE_dil_IMUa_m1{i, n1, n2, w1, w2, j2});
                        [tmp_LTE_IMUg_m1, tmp_CTE_IMUg_m1] = bwlabel(FM_segTE_dil_IMUg_m1{i, n1, n2, w1, w2, j2});
                        [tmp_LTE_IMUag_m1, tmp_CTE_IMUag_m1] = bwlabel(FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, j2});
                        % ------------------------------------------------------------------------------------------------------

                        % Method 2 - further segmentations by removing movements
                        tmp_segTE_dil_IMUa_m2 = zeros(length(FM_segTE_dil{i, n1, n2, w1, w2, j2}), 1);
                        tmp_segTE_dil_IMUg_m2 = zeros(length(FM_segTE_dil{i, n1, n2, w1, w2, j2}), 1);
                        tmp_segTE_dil_IMUag_m2 = zeros(length(FM_segTE_dil{i, n1, n2, w1, w2, j2}), 1);

                        for c = 1: tmp_CTE_dil

                            tmp_idx = find(tmp_LTE_dil == c);

                            tmp_segTE_dil_IMUa_m2(tmp_idx, :) = 1;
                            tmp_segTE_dil_IMUg_m2(tmp_idx, :) = 1;
                            tmp_segTE_dil_IMUag_m2(tmp_idx, :) = 1;

                            tmp_xIMUaTE = sum(FM_segTE_dil{i, n1, n2, w1, w2, j2}(tmp_idx) .* IMUaTE_mapDil{i, n1, :}(tmp_idx));
                            tmp_xIMUgTE = sum(FM_segTE_dil{i, n1, n2, w1, w2, j2}(tmp_idx) .* IMUgTE_mapDil{i, n1, :}(tmp_idx));

                            if tmp_xIMUaTE >= overlap_percFM*length(tmp_idx)
                                tmp_segTE_dil_IMUa_m2(tmp_idx) = 0;
                                tmp_segTE_dil_IMUag_m2(tmp_idx) = 0;
                            end

                            if tmp_xIMUgTE >= overlap_percFM*length(tmp_idx)
                                tmp_segTE_dil_IMUg_m2(tmp_idx) = 0;
                                tmp_segTE_dil_IMUag_m2(tmp_idx) = 0;
                            end

                            clear tmp_idx tmp_xIMUaTE tmp_xIMUgTE
                        end

                        [tmp_LTE_IMUa_m2, tmp_CTE_IMUa_m2] = bwlabel(tmp_segTE_dil_IMUa_m2);
                        [tmp_LTE_IMUg_m2, tmp_CTE_IMUg_m2] = bwlabel(tmp_segTE_dil_IMUg_m2);
                        [tmp_LTE_IMUag_m2, tmp_CTE_IMUag_m2] = bwlabel(tmp_segTE_dil_IMUag_m2);

                        FM_segTE_dil_IMUa_m2{i, n1, n2, w1, w2, j2} = tmp_segTE_dil_IMUa_m2;
                        FM_segTE_dil_IMUg_m2{i, n1, n2, w1, w2, j2} = tmp_segTE_dil_IMUg_m2;
                        FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, j2} = tmp_segTE_dil_IMUag_m2;

                        % display instant performance
                        fprintf('---- FM(%d-th sensor) ellip filtering > Denoising: method 1 vs method 2: IMUa > %d - %d; IMUg > %d - %d; IMUag > %d - %d ...\n', ...
                            j2, tmp_CTE_IMUa_m1, tmp_CTE_IMUa_m2, tmp_CTE_IMUg_m1, tmp_CTE_IMUg_m2, tmp_CTE_IMUag_m1, tmp_CTE_IMUag_m2);
                        % -------------------------------------------------------------------------------------------------------

                        clear tmp_sensor_indiv_ori tmp_sensor_indiv tmp_low_cutoff  ...
                            tmp_LTE tmp_CTE tmp_LTE_dil tmp_CTE_dil ...
                            tmp_LTE_IMUa_m1 tmp_CTE_IMUa_m1 tmp_LTE_IMUg_m1 tmp_CTE_IMUg_m1 tmp_LTE_IMUag_m1 tmp_CTE_IMUag_m1 ...
                            tmp_segTE_dil_IMUa_m2 tmp_segTE_dil_IMUg_m2 tmp_segTE_dil_IMUag_m2 ...
                            tmp_LTE_IMUa_m2 tmp_CTE_IMUa_m2 tmp_LTE_IMUg_m2 tmp_CTE_IMUg_m2 tmp_LTE_IMUag_m2 tmp_CTE_IMUag_m2
                    end
                    % -------------------------------------------------------------------------

                    %% Sensor fusion
                    % * Data fusion performed after dilation.
                    % For sensor type, they are combined with logical 'OR'
                    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
                    FM_segTE_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTE{i, n1, n2, w1, w2, 1} | FM_segTE{i, n1, n2, w1, w2, 2} | FM_segTE{i, n1, n2, w1, w2, 3});
                    FM_segTE_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTE{i, n1, n2, w1, w2, 4} | FM_segTE{i, n1, n2, w1, w2, 5} | FM_segTE{i, n1, n2, w1, w2, 6});
                    FM_segTE_fOR_acouORpiez{i, n1, n2, w1, w2, :}  = double(FM_segTE_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTE_fOR_piez{i, n1, n2, w1, w2, :});

                    FM_segTE_dil_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTE_dil{i, n1, n2, w1, w2, 1} | FM_segTE_dil{i, n1, n2, w1, w2, 2} | FM_segTE_dil{i, n1, n2, w1, w2, 3});
                    FM_segTE_dil_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTE_dil{i, n1, n2, w1, w2, 4} | FM_segTE_dil{i, n1, n2, w1, w2, 5} | FM_segTE_dil{i, n1, n2, w1, w2, 6});
                    FM_segTE_dil_fOR_acouORpiez{i, n1, n2, w1, w2, :} = double(FM_segTE_dil_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTE_dil_fOR_piez{i, n1, n2, w1, w2, :});

                    FM_segTE_dil_IMUag_m1_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, 1} | FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, 2} | FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, 3});
                    FM_segTE_dil_IMUag_m1_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, 4} | FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, 5} | FM_segTE_dil_IMUag_m1{i, n1, n2, w1, w2, 6});
                    FM_segTE_dil_IMUag_m1_fOR_acouORpiez{i, n1, n2, w1, w2, :}  = double(FM_segTE_dil_IMUag_m1_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTE_dil_IMUag_m1_fOR_piez{i, n1, n2, w1, w2, :});

                    FM_segTE_dil_IMUag_m2_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, 1} | FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, 2} | FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, 3});
                    FM_segTE_dil_IMUag_m2_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, 4} | FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, 5} | FM_segTE_dil_IMUag_m2{i, n1, n2, w1, w2, 6});
                    FM_segTE_dil_IMUag_m2_fOR_acouORpiez{i, n1, n2, w1, w2, :}  = double(FM_segTE_dil_IMUag_m2_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTE_dil_IMUag_m2_fOR_piez{i, n1, n2, w1, w2, :});

                    [tmp_FM_segTE_dil_fOR_acouORpiezL, tmp_FM_segTE_dil_fOR_acouORpiezC] = bwlabel(FM_segTE_dil_fOR_acouORpiez{i, n1, n2, w1, w2, :});
                    [tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezL, tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezC] = bwlabel(FM_segTE_dil_IMUag_m1_fOR_acouORpiez{i, n1, n2, w1, w2, :});
                    [tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezL, tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezC] = bwlabel(FM_segTE_dil_IMUag_m2_fOR_acouORpiez{i, n1, n2, w1, w2, :});

                    FM_segTE_dil_fOR_acouORpiezL{i, n1, n2, w1, w2, :} = tmp_FM_segTE_dil_fOR_acouORpiezL;
                    FM_segTE_dil_fOR_acouORpiezC(i, n1, n2, w1, w2, :) = tmp_FM_segTE_dil_fOR_acouORpiezC;
                    FM_segTE_dil_IMUag_m1_fOR_acouORpiezL{i, n1, n2, w1, w2, :} = tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezL;
                    FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, n1, n2, w1, w2, :) = tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezC;
                    FM_segTE_dil_IMUag_m2_fOR_acouORpiezL{i, n1, n2, w1, w2, :} = tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezL;
                    FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, n1, n2, w1, w2, :) = tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezC;

                    clear tmp_FM_segTE_dil_fOR_acouORpiezL tmp_FM_segTE_dil_fOR_acouORpiezC ...
                        tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezL tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezC ...
                        tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezL tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezC
                    % ---------------------------------------------------------------------------------------------------------------------------------

                    %% match of FM and sensation
                    matchTE_FM_Sens{i, n1, n2, w1, w2, :} = FM_segTE_dil_fOR_acouORpiezL{i, n1, n2, w1, w2, :} .* sensT_map{i, n1, :};
                    matchTE_FM_Sens_num(i, n1, n2, w1, w2, :) = length(unique(matchTE_FM_Sens{i, n1, n2, w1, w2, :})) - 1;

                    fprintf('---- FM_segTE_dil_fOR_acouORpiez: %d vs sensT_map: %d > match: %d ...\n', ...
                        FM_segTE_dil_fOR_acouORpiezC(i, n1, n2, w1, w2, :), sensT_mapC(i, n1, :), matchTE_FM_Sens_num(i, n1, n2, w1, w2, :));
                    % ---------------------------------------------------------------------------------------------------

                    matchTE_FM_Sens_INV{i, n1, n2, w1, w2, :} = sensT_mapL{i, n1, :} .* FM_segTE_dil_fOR_acouORpiez{i, n1, n2, w1, w2, :};
                    matchTE_FM_Sens_num_INV(i, n1, n2, w1, w2, :) = length(unique(matchTE_FM_Sens_INV{i, n1, n2, w1, w2, :})) - 1;

                    fprintf('---- *Inversed* sensT_map: %d vs FM_segTE_dil_fOR_acouORpiez: %d > match: %d ...\n', ...
                        sensT_mapC(i, n1, :), FM_segTE_dil_fOR_acouORpiezC(i, n1, n2, w1, w2, :), matchTE_FM_Sens_num_INV(i, n1, n2, w1, w2, :));

                    % sensitivity & precision
                    matchTE_FM_Sens_sensi(i, n1, n2, w1, w2, :) = matchTE_FM_Sens_num(i, n1, n2, w1, w2, :) / sensT_mapC(i, n1, :);
                    matchTE_FM_Sens_preci(i, n1, n2, w1, w2, :) = matchTE_FM_Sens_num(i, n1, n2, w1, w2, :) / FM_segTE_dil_fOR_acouORpiezC(i, n1, n2, w1, w2, :);
                    fprintf('---- **** FM_segTE_dil_fOR_acouORpiez - sensitivity & precision: %d & %d ...\n', ...
                        matchTE_FM_Sens_sensi(i, n1, n2, w1, w2, :), matchTE_FM_Sens_preci(i, n1, n2, w1, w2, :));
                    % ---------------------------------------------------------------------------------------------------

                    matchTE_FM_Sens_IMUag_m1{i, n1, n2, w1, w2, :} = FM_segTE_dil_IMUag_m1_fOR_acouORpiezL{i, n1, n2, w1, w2, :} .* sensTE_mapIMUag{i, n1, :};
                    matchTE_FM_Sens_IMUag_m1_num(i, n1, n2, w1, w2, :) = length(unique(matchTE_FM_Sens_IMUag_m1{i, n1, n2, w1, w2, :})) - 1;

                    fprintf('---- FM_segTE_dil_IMUag_m1_fOR_acouORpiez: %d vs sensTE_mapIMUag: %d > match: %d ...\n', ...
                        FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, n1, n2, w1, w2, :), sensTE_mapIMUagC(i, n1, :), matchTE_FM_Sens_IMUag_m1_num(i, n1, n2, w1, w2, :));
                    % ---------------------------------------------------------------------------------------------------

                    matchTE_FM_Sens_IMUag_m1_INV{i, n1, n2, w1, w2, :} = sensTE_mapIMUagL{i, n1, :} .* FM_segTE_dil_IMUag_m1_fOR_acouORpiez{i, n1, n2, w1, w2, :};
                    matchTE_FM_Sens_IMUag_m1_num_INV(i, n1, n2, w1, w2, :) = length(unique(matchTE_FM_Sens_IMUag_m1_INV{i, n1, n2, w1, w2, :})) - 1;

                    fprintf('---- *Inversed* sensT_map: %d vs FM_segTE_dil_IMUag_m1_fOR_acouORpiez: %d > match: %d ...\n', ...
                        sensTE_mapIMUagC(i, n1, :), FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, n1, n2, w1, w2, :), matchTE_FM_Sens_IMUag_m1_num(i, n1, n2, w1, w2, :));

                    % sensitivity & precision
                    matchTE_FM_Sens_IMUag_m1_sensi(i, n1, n2, w1, w2, :) = matchTE_FM_Sens_IMUag_m1_num(i, n1, n2, w1, w2, :) / sensTE_mapIMUagC(i, n1, :);
                    matchTE_FM_Sens_IMUag_m1_preci(i, n1, n2, w1, w2, :) = matchTE_FM_Sens_IMUag_m1_num(i, n1, n2, w1, w2, :) / FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, n1, n2, w1, w2, :);
                    fprintf('---- **** FM_segTE_dil_IMUag_m1_fOR_acouORpiezC - sensitivity & precision: %d & %d ...\n', ...
                        matchTE_FM_Sens_IMUag_m1_sensi(i, n1, n2, w1, w2, :), matchTE_FM_Sens_IMUag_m1_preci(i, n1, n2, w1, w2, :));
                    % -----------------------------------------------------------------------------------------------------

                    matchTE_FM_Sens_IMUag_m2{i, n1, n2, w1, w2, :} = FM_segTE_dil_IMUag_m2_fOR_acouORpiezL{i, n1, n2, w1, w2, :} .* sensTE_mapIMUag{i, n1, :};
                    matchTE_FM_Sens_IMUag_m2_num(i, n1, n2, w1, w2, :) = length(unique(matchTE_FM_Sens_IMUag_m2{i, n1, n2, w1, w2, :})) - 1;

                    fprintf('---- FM_segTE_dil_IMUag_m2_fOR_acouORpiez: %d vs sensTE_mapIMUag: %d > match: %d ...\n', ...
                        FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, n1, n2, w1, w2, :), sensTE_mapIMUagC(i, n1, :), matchTE_FM_Sens_IMUag_m2_num(i, n1, n2, w1, w2, :));
                    % ---------------------------------------------------------------------------------------------------

                    matchTE_FM_Sens_IMUag_m2_INV{i, n1, n2, w1, w2, :} = sensTE_mapIMUagL{i, n1, :} .* FM_segTE_dil_IMUag_m2_fOR_acouORpiez{i, n1, n2, w1, w2, :};
                    matchTE_FM_Sens_IMUag_m2_num_INV(i, n1, n2, w1, w2, :) = length(unique(matchTE_FM_Sens_IMUag_m2_INV{i, n1, n2, w1, w2, :})) - 1;

                    fprintf('---- *Inversed* sensTE_mapIMUag: %d vs FM_segTE_dil_IMUag_m2_fOR_acouORpiez: %d > match: %d ...\n', ...
                        sensTE_mapIMUagC(i, n1, :), FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, n1, n2, w1, w2, :), matchTE_FM_Sens_IMUag_m2_num(i, n1, n2, w1, w2, :));

                    % sensitivity & precision
                    matchTE_FM_Sens_IMUag_m2_sensi(i, n1, n2, w1, w2, :) = matchTE_FM_Sens_IMUag_m2_num(i, n1, n2, w1, w2, :) / sensTE_mapIMUagC(i, n1, :);
                    matchTE_FM_Sens_IMUag_m2_preci(i, n1, n2, w1, w2, :) = matchTE_FM_Sens_IMUag_m2_num(i, n1, n2, w1, w2, :) / FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, n1, n2, w1, w2, :);
                    fprintf('---- **** FM_segTE_dil_IMUag_m2_fOR_acouORpiezC - sensitivity & precision: %d & %d ...\n', ...
                        matchTE_FM_Sens_IMUag_m2_sensi(i, n1, n2, w1, w2, :), matchTE_FM_Sens_IMUag_m2_preci(i, n1, n2, w1, w2, :));
                    % ---------------------------------------------------------------------------------------------------

                    fprintf('**** N*std IMU: %d; SNR acour/piez: %d, weights for noise/low singal levels (%d & %d) ...\n\n\n', ...
                            Nstd_IMU_TE(n1), snr_acou_th(n2), noise_quanW(w1),low_quanW(w2));

                % end of loop w1
                end
            % end of loop w2
            end
            % end of loop n2
        end
        % end of loop: n1
    end
    % end of loop for the files
end

% save segmented and fused data
cd(fdir_p);
save(['Seg02_ellip_' participant '_' data_category ...
      '_FMsnr' num2str(max(snr_acou_th)) ...
      '_IMUx' num2str(Nstd_IMU_TE) ...
      '_OLs' num2str(overlap_percSens*100) ...
      '_OLf' num2str(overlap_percFM*100) ...
      '_WN' num2str(noise_quanW*100) ...
      '_WL' num2str(low_quanW*100) '_IMUSens.mat'], ...
      'IMUaTE_th', 'IMUaTE_map', 'IMUaTE_mapDil', ...
      'IMUgTE_th', 'IMUgTE_map', 'IMUgTE_mapDil', ...
      'sensT_map', 'sensT_mapL', 'sensT_mapC', ...
      'sensTE_mapIMUa', 'sensTE_mapIMUaL', 'sensTE_mapIMUaC', ...
      'sensTE_mapIMUg', 'sensTE_mapIMUgL', 'sensTE_mapIMUgC', ...
      'sensTE_mapIMUag', 'sensTE_mapIMUagL', 'sensTE_mapIMUagC', ...
      '-v7.3');
save(['Seg02_ellip_' participant '_' data_category ...
      '_FMsnr' num2str(max(snr_acou_th)) ...
      '_IMUx' num2str(Nstd_IMU_TE) ...
      '_OLs' num2str(overlap_percSens*100) ...
      '_OLf' num2str(overlap_percFM*100) ...
      '_WN' num2str(noise_quanW*100) ...
      '_WL' num2str(low_quanW*100) '_FM.mat'], ...
      'FM_suiteTE', ...
      'FM_TE_th', 'FM_segTE', 'FM_segTE_dil', ...
      'FM_segTE_dil_IMUa_m1', 'FM_segTE_dil_IMUg_m1', 'FM_segTE_dil_IMUag_m1', ...
      'FM_segTE_dil_IMUa_m2', 'FM_segTE_dil_IMUg_m2', 'FM_segTE_dil_IMUag_m2', ...
      'FM_segTE_fOR_acou', 'FM_segTE_fOR_piez', 'FM_segTE_fOR_acouORpiez', ...
      'FM_segTE_dil_fOR_acou', 'FM_segTE_dil_fOR_piez', 'FM_segTE_dil_fOR_acouORpiez', ...
      'FM_segTE_dil_IMUag_m1_fOR_acou', 'FM_segTE_dil_IMUag_m1_fOR_piez', 'FM_segTE_dil_IMUag_m1_fOR_acouORpiez', ...
      'FM_segTE_dil_IMUag_m2_fOR_acou', 'FM_segTE_dil_IMUag_m2_fOR_piez', 'FM_segTE_dil_IMUag_m2_fOR_acouORpiez', ...
      '-v7.3');
save(['Seg02_ellip_' participant '_' data_category ...
      '_FMsnr' num2str(max(snr_acou_th)) ...
      '_IMUx' num2str(Nstd_IMU_TE) ...
      '_OLs' num2str(overlap_percSens*100) ...
      '_OLf' num2str(overlap_percFM*100) ...
      '_WN' num2str(noise_quanW*100) ...
      '_WL' num2str(low_quanW*100) '_FM_label.mat'], ...
      'FM_segTE_dil_fOR_acouORpiezL', 'FM_segTE_dil_fOR_acouORpiezC', ...
      'FM_segTE_dil_IMUag_m1_fOR_acouORpiezL', 'FM_segTE_dil_IMUag_m1_fOR_acouORpiezC', ...
      'FM_segTE_dil_IMUag_m2_fOR_acouORpiezL', 'FM_segTE_dil_IMUag_m2_fOR_acouORpiezC', ...
      '-v7.3');
save(['Seg02_ellip_' participant '_' data_category ...
      '_FMsnr' num2str(max(snr_acou_th)) ...
      '_IMUx' num2str(Nstd_IMU_TE) ...
      '_OLs' num2str(overlap_percSens*100) ...
      '_OLf' num2str(overlap_percFM*100) ...
      '_WN' num2str(noise_quanW*100) ...
      '_WL' num2str(low_quanW*100) '_Match.mat'], ...
      'matchTE_FM_Sens', 'matchTE_FM_Sens_num', ...
      'matchTE_FM_Sens_INV', 'matchTE_FM_Sens_num_INV', ...
      'matchTE_FM_Sens_IMUag_m1', 'matchTE_FM_Sens_IMUag_m1_num', ...
      'matchTE_FM_Sens_IMUag_m1_INV', 'matchTE_FM_Sens_IMUag_m1_num_INV', ...
      'matchTE_FM_Sens_IMUag_m2', 'matchTE_FM_Sens_IMUag_m2_num', ...
      'matchTE_FM_Sens_IMUag_m2_INV', 'matchTE_FM_Sens_IMUag_m2_num_INV', ...
      'matchTE_FM_Sens_sensi', 'matchTE_FM_Sens_preci', ...
      'matchTE_FM_Sens_IMUag_m1_sensi', 'matchTE_FM_Sens_IMUag_m1_preci', ...
      'matchTE_FM_Sens_IMUag_m2_sensi', 'matchTE_FM_Sens_IMUag_m2_preci', ...
      '-v7.3');
cd(curr_dir)











