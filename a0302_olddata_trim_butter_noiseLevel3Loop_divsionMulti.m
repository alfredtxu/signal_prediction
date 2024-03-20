% 1. remove the singals below noise level (extreme low singals)
% 2. remove the lowest 25% signals (FM sensors only), then calculating the adaptive thresholds on the remaining 75%
clc
clear
close all

% HOME / OFFICE
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% curr_dir = 'G:\My Drive\ic_welcom_leap_fm\b_src_matlab';
fdir = [curr_dir '\z14_olddata_mat_proc_new'];
cd(curr_dir);

% Frequency (Hz): sensor / sensation
freq = 400;

% the number of channels / FM sensors
num_channel = 16;
num_FMsensors = 6;

% Add paths for function files
% the cohort of participant
participants = {'S1', 'S2', 'S3', 'S4', 'S5'};
group = 'all';
data_category = 'olddata';

addpath(genpath('z10_olddata_mat_raw'))
addpath(genpath('z11_olddata_mat_preproc'))
% -------------------------------------------------------------------------------------------------------

%% Data loading - preprocessed
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

    clear acceLNetFil acceLNetFilT acceLNetFilTEq_cat acceLNetT acceLNetTEq ...
        acceRNetFil acceRNetFilT acceRNetFilTEq_cat acceRNetT acceRNetTEq ...
        acouLFil acouLFilT acouLFilTEq_cat acouLT acouLTEq ...
        acouRFil acouRFilT acouRFilTEq_cat acouRT acouRTEq ...
        acceLNetFilTEq acceRNetFilTEq acouLFilTEq acouRFilTEq ...
        forcFil forcFilT forcFilTEq forcFilTEq_cat forcT forcTEq ...
        IMUacceNetFil IMUacceNetFilT IMUacceNetFilTEq IMUacceNetFilTEq_cat IMUacceNetT IMUacceNetTEq ...
        piezLFil piezLFilT piezLFilTEq piezLFilTEq_cat piezLT piezLTEq ...
        piezRFil piezRFilT piezRFilTEq piezRFilTEq_cat piezRT piezRTEq ...
        sens1T sens1TEq sens1TEq_cat sens1TEq_mtxP sens2T sens2TEq sens2TEq_cat ...
        time_vecTEq tmp_mat
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

% the number of data files in total
num_files = sum(all_nfile);
% =========================================================================================================
% =========================================================================================================

%% Variables for thresholding and segmenting
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

for nstd = 2 : 4

    for olf = 0.05 : 0.05 : 0.10

        % thresholds: noise and low quantile level, and corresponding weights
        noise_quanL = 0.05;
        low_quanL = 0.25;

        noise_quanW = 1.00;
        low_quanW = 1.00;

        % thresholds (mu + N * std) - 2,3,4
        Nstd_IMU_TB = nstd;
        Nstd_acou_TB = 1;
        Nstd_piez_TB = 1;

        % thresholds: singal-noise ratio
        snr_IMU_th = 25;
        snr_acou_th = 20:23;
        snr_piez_th = 20:23;

        % percentage of overlap between senstation and maternal movement (IMU)
        overlap_percSens = 0.20;
        overlap_percFM = olf; % 0.05, 0.10

        %% divide into portions
        num_p = 6;
        fper = ceil(num_files / num_p);

        for por = 1 : num_p

            curr_p = por;

            % file sequence for each portion
            fidx_s = fper*(curr_p-1)+1;

            if curr_p == num_p
                fidx_e = num_files;
            else
                fidx_e = fper*curr_p;
            end

            num_filesP = fidx_e - fidx_s + 1;

            if curr_p < num_p
                all_acceL_catP = all_acceL_cat(fidx_s:fidx_e, :);
                all_acceR_catP = all_acceR_cat(fidx_s:fidx_e, :);
                all_acouL_catP = all_acouL_cat(fidx_s:fidx_e, :);
                all_acouR_catP = all_acouR_cat(fidx_s:fidx_e, :);
                all_piezL_catP = all_piezL_cat(fidx_s:fidx_e, :);
                all_piezR_catP = all_piezR_cat(fidx_s:fidx_e, :);

                all_forc_catP = all_forc_cat(fidx_s:fidx_e, :);
                all_IMUacce_catP = all_IMUacce_cat(fidx_s:fidx_e, :);
                all_sens1_catP = all_sens1_cat(fidx_s:fidx_e, :);
                all_sens2_catP = all_sens2_cat(fidx_s:fidx_e, :);
                all_sensMtx_catP = all_sensMtx_cat(fidx_s:fidx_e, :);
                all_timeV_catP = all_timeV_cat(fidx_s:fidx_e, :);
            else
                all_acceL_catP = all_acceL_cat(fidx_s:end, :);
                all_acceR_catP = all_acceR_cat(fidx_s:end, :);
                all_acouL_catP = all_acouL_cat(fidx_s:end, :);
                all_acouR_catP = all_acouR_cat(fidx_s:end, :);
                all_piezL_catP = all_piezL_cat(fidx_s:end, :);
                all_piezR_catP = all_piezR_cat(fidx_s:end, :);

                all_forc_catP = all_forc_cat(fidx_s:end, :);
                all_IMUacce_catP = all_IMUacce_cat(fidx_s:end, :);
                all_sens1_catP = all_sens1_cat(fidx_s:end, :);
                all_sens2_catP = all_sens2_cat(fidx_s:end, :);
                all_sensMtx_catP = all_sensMtx_cat(fidx_s:end, :);
                all_timeV_catP = all_timeV_cat(fidx_s:end, :);
            end

            % release system memory
            clear all_acceL all_acceR all_acouL all_acouR all_piezL all_piezR ...
                all_forc all_IMUacce all_sens1 all_sens2 all_sensMtx all_timeV
            % ------------------------------------------------------------------------------------

            %% SEGMENTING PRE-PROCESSED DATA
            for i = 1 : num_filesP

                fprintf('Current data file (portion: %d) from %d of %d ...\n', curr_p, i, num_filesP);

                for n1 = 1 : length(Nstd_IMU_TB)

                    %% IMUaTB thrsholding
                    % SNR: IMUaTB
                    % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
                    % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
                    % SNR = 10 * log10(P_signal / P_noise)
                    tmp_SNR_IMUaTB = 0;
                    tmp_Nstd_IMU_TB = Nstd_IMU_TB(n1);
                    tmp_IMUaTB = abs(all_IMUacce_catP{i, :});

                    while tmp_SNR_IMUaTB <= snr_IMU_th

                        tmp_IMUaTB_th = mean(tmp_IMUaTB) + tmp_Nstd_IMU_TB * std(tmp_IMUaTB);

                        tmp_IMUaTB_P = tmp_IMUaTB(tmp_IMUaTB >= tmp_IMUaTB_th);
                        tmp_IMUaTB_N = tmp_IMUaTB(tmp_IMUaTB < tmp_IMUaTB_th);

                        p_signal_IMUaTB = 0;
                        for pa = 1 : length(tmp_IMUaTB_P)
                            p_signal_IMUaTB = p_signal_IMUaTB + tmp_IMUaTB_P(pa)^2;
                        end
                        n_signal_IMUaTB = 0;
                        for na = 1 : length(tmp_IMUaTB_N)
                            n_signal_IMUaTB = n_signal_IMUaTB + tmp_IMUaTB_N(na)^2;
                        end

                        tmp_SNR_IMUaTB = 10 * log10((p_signal_IMUaTB / length(tmp_IMUaTB_P)) / (n_signal_IMUaTB / length(tmp_IMUaTB_N)));

                        % increase the threshold weights if the SNR is not sufficient
                        tmp_Nstd_IMU_TB = tmp_Nstd_IMU_TB + 1;

                    end

                    IMUaTB_th(i, n1).thresh = tmp_IMUaTB_th;
                    IMUaTB_th(i, n1).Nstd = tmp_Nstd_IMU_TB;
                    IMUaTB_th(i, n1).SNR = tmp_SNR_IMUaTB;

                    fprintf('---- IMUa Butterworth filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUaTB_th, tmp_Nstd_IMU_TB-1, tmp_SNR_IMUaTB);

                    % IMUa maps
                    IMUaTB_map{i, n1, :} = tmp_IMUaTB >= IMUaTB_th(i, n1).thresh;
                    IMUaTB_mapDil{i, n1, :} = imdilate(IMUaTB_map{i, n1, :}, IMU_lse);

                    clear tmp_SNR_IMUaTB tmp_Nstd_IMU_TB tmp_IMUaTB tmp_IMUaTB_th tmp_IMUaTB_P tmp_IMUaTB_N
                    % -----------------------------------------------------------------------------------------------

                    %% Sensation
                    % Initializing the map: sensT
                    % senstation labels by connected components
                    tmp_sensT = all_sens1_catP{i, :};
                    [tmp_sensTL, tmp_sensTC] = bwlabel(tmp_sensT);

                    % initialization of sensation maps
                    tmp_sensT_map = zeros(length(tmp_sensT), 1);
                    tmp_sensTB_mapIMUa = zeros(length(tmp_sensT), 1);

                    for j1 = 1 : tmp_sensTC

                        % the idx range of the current cluster (component)
                        tmp_idx = find(tmp_sensTL == j1);

                        tmp_idxS = min(tmp_idx) - sens_dilation_sizeB;
                        tmp_idxE = max(tmp_idx) + sens_dilation_sizeF;

                        tmp_idxS = max(tmp_idxS, 1);
                        tmp_idxE = min(tmp_idxE, length(tmp_sensT_map));

                        % sensation map
                        tmp_sensT_map(tmp_idxS:tmp_idxE) = 1;

                        tmp_sensTB_mapIMUa(tmp_idxS:tmp_idxE) = 1;

                        % Remove maternal sensation coincided with body movement (IMU a/g) - butterworth
                        tmp_xIMUaTB = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUaTB_mapDil{i, n1, :}(tmp_idxS:tmp_idxE));

                        if (tmp_xIMUaTB >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
                            tmp_sensTB_mapIMUa(tmp_idxS:tmp_idxE) = 0;
                        end

                        clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTB
                    end

                    % sensation maps
                    sensT_map{i, n1, :} = tmp_sensT_map;
                    sensTB_mapIMUa{i, n1, :} = tmp_sensTB_mapIMUa;

                    [tmp_sensT_mapL, tmp_sensT_mapC] = bwlabel(tmp_sensT_map);
                    [tmp_sensTB_mapIMUaL, tmp_sensTB_mapIMUaC] = bwlabel(tmp_sensTB_mapIMUa);

                    sensT_mapL{i, n1, :} = tmp_sensT_mapL;
                    sensT_mapC(i, n1, :) = tmp_sensT_mapC;
                    sensTB_mapIMUaL{i, n1, :} = tmp_sensTB_mapIMUaL;
                    sensTB_mapIMUaC(i, n1, :) = tmp_sensTB_mapIMUaC;

                    fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d ... \n', ...
                        sensT_mapC(i, n1, :), sensTB_mapIMUaC(i, n1, :));

                    clear tmp_sensT tmp_sensTL tmp_sensTC ...
                        tmp_sensT_map tmp_sensT_mapL tmp_sensT_mapC ...
                        tmp_sensTB_mapIMUa tmp_sensTB_mapIMUaL tmp_sensTB_mapIMUaC
                    % --------------------------------------------------------------------------------

                    %% Segmenting FM sensors
                    % 1. apply for an adaptive threshold
                    % 2. remove body movement
                    % 3. dilate the fetal movements at desinated duration
                    FM_suiteTB{i, 1} = all_acceL_catP{i,:};
                    FM_suiteTB{i, 2} = all_acceR_catP{i,:};
                    FM_suiteTB{i, 3} = all_acouL_catP{i,:};
                    FM_suiteTB{i, 4} = all_acouR_catP{i,:};
                    FM_suiteTB{i, 5} = all_piezL_catP{i,:};
                    FM_suiteTB{i, 6} = all_piezR_catP{i,:};

                    for n2 = 1 : length(snr_acou_th)

                        for w1 = 1 : length(noise_quanW)

                            for w2 = 1 : length(low_quanW)

                                % FM Sensors Thresholding
                                for j2 = 1 : num_FMsensors

                                    tmp_sensor_indiv = abs(FM_suiteTB{i, j2});

                                    % remove noise level
                                    tmp_noise_cutoff = quantile(tmp_sensor_indiv, noise_quanL);
                                    tmp_sensor_indiv_noise = tmp_sensor_indiv;
                                    tmp_sensor_indiv_noise(tmp_sensor_indiv_noise<=tmp_noise_cutoff*noise_quanW(w1), :) = [];

                                    % remove low level
                                    tmp_low_cutoff = quantile(tmp_sensor_indiv_noise, low_quanL);
                                    tmp_sensor_indiv_cutoff = tmp_sensor_indiv_noise;
                                    tmp_sensor_indiv_cutoff(tmp_sensor_indiv_cutoff<=tmp_low_cutoff*low_quanW(w2), :) = [];

                                    tmp_SNR_sensor = 0;

                                    if j2 <= num_FMsensors / 2
                                        tmp_std_times_FM = Nstd_acou_TB;
                                        tmp_snr_FM = snr_acou_th(n2);
                                    else
                                        tmp_std_times_FM = Nstd_piez_TB;
                                        tmp_snr_FM = snr_piez_th(n2);
                                    end

                                    while tmp_SNR_sensor <= tmp_snr_FM

                                        tmp_FM_threshTB = mean(tmp_sensor_indiv_cutoff) + tmp_std_times_FM * std(tmp_sensor_indiv_cutoff);
                                        if isnan(tmp_FM_threshTB)
                                            tmp_FM_threshTB = 0;
                                        end

                                        tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < tmp_FM_threshTB);
                                        tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= tmp_FM_threshTB);

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

                                    FM_TB_th(i, n1, n2, w1, w2, j2).thresh = tmp_FM_threshTB;
                                    FM_TB_th(i, n1, n2, w1, w2, j2).Nstd = tmp_std_times_FM;
                                    FM_TB_th(i, n1, n2, w1, w2, j2).SNE = tmp_SNR_sensor;

                                    fprintf('---- FM(%d-th sensor) Butterworth filtering: threshold > %d; SNR (%d*std): %d ... \n', ...
                                        j2, tmp_FM_threshTB, tmp_std_times_FM-1, tmp_SNR_sensor);

                                    clear tmp_SNR_sensor tmp_std_times_FM tmp_snr_FM ...
                                        tmp_FM_removed tmp_FM_segmented tmp_FM_threshTB
                                    % --------------------------------------------------------------------------------------------------------------

                                    %% segmentation by thresholding
                                    FM_segTB{i, n1, n2, w1, w2, j2} = (tmp_sensor_indiv >= FM_TB_th(i, n1, n2, w1, w2, j2).thresh);
                                    FM_segTB_dil{i, n1, n2, w1, w2, j2} = double(imdilate(FM_segTB{i, n1, n2, w1, w2, j2}, FM_lse));

                                    % labels by connected components
                                    [tmp_LTB, tmp_CTB] = bwlabel(FM_segTB{i, n1, n2, w1, w2, j2});
                                    [tmp_LTB_dil, tmp_CTB_dil] = bwlabel(FM_segTB_dil{i, n1, n2, w1, w2, j2});

                                    % Method 1 - denoise segmentations by removing movements
                                    FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, j2} = FM_segTB_dil{i, n1, n2, w1, w2, j2} .* (1-IMUaTB_mapDil{i,n1,:});

                                    % labels by connected components
                                    [tmp_LTB_IMUa_m1, tmp_CTB_IMUa_m1] = bwlabel(FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, j2});
                                    % ------------------------------------------------------------------------------------------------------

                                    % Method 2 - further segmentations by removing movements
                                    tmp_segTB_dil_IMUa_m2 = zeros(length(FM_segTB_dil{i, n1, n2, w1, w2, j2}), 1);

                                    for c = 1: tmp_CTB_dil

                                        tmp_idx = find(tmp_LTB_dil == c);

                                        tmp_segTB_dil_IMUa_m2(tmp_idx, :) = 1;
                                        tmp_xIMUaTB = sum(FM_segTB_dil{i, n1, n2, w1, w2, j2}(tmp_idx) .* IMUaTB_mapDil{i, n1, :}(tmp_idx));

                                        if tmp_xIMUaTB >= overlap_percFM*length(tmp_idx)
                                            tmp_segTB_dil_IMUa_m2(tmp_idx) = 0;
                                        end

                                        clear tmp_idx tmp_xIMUaTB
                                    end

                                    [tmp_LTB_IMUa_m2, tmp_CTB_IMUa_m2] = bwlabel(tmp_segTB_dil_IMUa_m2);
                                    FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, j2} = tmp_segTB_dil_IMUa_m2;

                                    % display instant performance
                                    fprintf('---- FM(%d-th sensor) Butterworth filtering > Denoising: method 1 vs method 2: IMUa > %d - %d ...\n', ...
                                        j2, tmp_CTB_IMUa_m1, tmp_CTB_IMUa_m2);
                                    % -------------------------------------------------------------------------------------------------------

                                    clear tmp_sensor_indiv_ori tmp_sensor_indiv tmp_low_cutoff  ...
                                        tmp_LTB tmp_CTB tmp_LTB_dil tmp_CTB_dil ...
                                        tmp_LTB_IMUa_m1 tmp_CTB_IMUa_m1 ...
                                        tmp_segTB_dil_IMUa_m2 ...
                                        tmp_LTB_IMUa_m2 tmp_CTB_IMUa_m2
                                end
                                % -------------------------------------------------------------------------

                                %% Sensor fusion
                                % * Data fusion performed after dilation. (a1-acce; a2-acou; p-piez)
                                % For sensor type, they are combined with logical 'OR'
                                % Between the types of sensors, they are combined with logical 'OR' / 'AND'
                                FM_segTB_fOR_acce{i, n1, n2, w1, w2, :} = double(FM_segTB{i, n1, n2, w1, w2, 1} | FM_segTB{i, n1, n2, w1, w2, 2});
                                FM_segTB_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTB{i, n1, n2, w1, w2, 3} | FM_segTB{i, n1, n2, w1, w2, 4});
                                FM_segTB_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTB{i, n1, n2, w1, w2, 5} | FM_segTB{i, n1, n2, w1, w2, 6});
                                FM_segTB_fOR_a1a2{i, n1, n2, w1, w2, :}  = double(FM_segTB_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_fOR_acou{i, n1, n2, w1, w2, :});
                                FM_segTB_fOR_a1p{i, n1, n2, w1, w2, :}  = double(FM_segTB_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_fOR_a2p{i, n1, n2, w1, w2, :}  = double(FM_segTB_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_fOR_a1a2p{i, n1, n2, w1, w2, :}  = double(FM_segTB_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_fOR_piez{i, n1, n2, w1, w2, :});

                                FM_segTB_dil_fOR_acce{i, n1, n2, w1, w2, :} = double(FM_segTB_dil{i, n1, n2, w1, w2, 1} | FM_segTB_dil{i, n1, n2, w1, w2, 2});
                                FM_segTB_dil_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTB_dil{i, n1, n2, w1, w2, 3} | FM_segTB_dil{i, n1, n2, w1, w2, 4});
                                FM_segTB_dil_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTB_dil{i, n1, n2, w1, w2, 5} | FM_segTB_dil{i, n1, n2, w1, w2, 6});
                                FM_segTB_dil_fOR_a1a2{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_fOR_acou{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_fOR_a1p{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_fOR_a2p{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_dil_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_fOR_a1a2p{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_dil_fOR_piez{i, n1, n2, w1, w2, :});

                                FM_segTB_dil_IMUa_m1_fOR_acce{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, 1} | FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, 2});
                                FM_segTB_dil_IMUa_m1_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, 3} | FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, 4});
                                FM_segTB_dil_IMUa_m1_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, 5} | FM_segTB_dil_IMUa_m1{i, n1, n2, w1, w2, 6});
                                FM_segTB_dil_IMUa_m1_fOR_a1a2{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m1_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m1_fOR_acou{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_IMUa_m1_fOR_a1p{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m1_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m1_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_IMUa_m1_fOR_a2p{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m1_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m1_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_IMUa_m1_fOR_a1a2p{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m1_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m1_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m1_fOR_piez{i, n1, n2, w1, w2, :});

                                FM_segTB_dil_IMUa_m2_fOR_acce{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, 1} | FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, 2});
                                FM_segTB_dil_IMUa_m2_fOR_acou{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, 3} | FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, 4});
                                FM_segTB_dil_IMUa_m2_fOR_piez{i, n1, n2, w1, w2, :} = double(FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, 5} | FM_segTB_dil_IMUa_m2{i, n1, n2, w1, w2, 6});
                                FM_segTB_dil_IMUa_m2_fOR_a1a2{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m2_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m2_fOR_acou{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_IMUa_m2_fOR_a1p{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m2_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m2_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_IMUa_m2_fOR_a2p{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m2_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m2_fOR_piez{i, n1, n2, w1, w2, :});
                                FM_segTB_dil_IMUa_m2_fOR_a1a2p{i, n1, n2, w1, w2, :}  = double(FM_segTB_dil_IMUa_m2_fOR_acce{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m2_fOR_acou{i, n1, n2, w1, w2, :} | FM_segTB_dil_IMUa_m2_fOR_piez{i, n1, n2, w1, w2, :});

                                [tmp_FM_segTB_dil_fOR_a1a2pL, tmp_FM_segTB_dil_fOR_a1a2pC] = bwlabel(FM_segTB_dil_fOR_a1a2p{i, n1, n2, w1, w2, :});
                                [tmp_FM_segTB_dil_IMUa_m1_fOR_a1a2pL, tmp_FM_segTB_dil_IMUa_m1_fOR_a1a2pC] = bwlabel(FM_segTB_dil_IMUa_m1_fOR_a1a2p{i, n1, n2, w1, w2, :});
                                [tmp_FM_segTB_dil_IMUa_m2_fOR_a1a2pL, tmp_FM_segTB_dil_IMUa_m2_fOR_a1a2pC] = bwlabel(FM_segTB_dil_IMUa_m2_fOR_a1a2p{i, n1, n2, w1, w2, :});

                                FM_segTB_dil_fOR_a1a2pL{i, n1, n2, w1, w2, :} = tmp_FM_segTB_dil_fOR_a1a2pL;
                                FM_segTB_dil_fOR_a1a2pC(i, n1, n2, w1, w2, :) = tmp_FM_segTB_dil_fOR_a1a2pC;
                                FM_segTB_dil_IMUa_m1_fOR_a1a2pL{i, n1, n2, w1, w2, :} = tmp_FM_segTB_dil_IMUa_m1_fOR_a1a2pL;
                                FM_segTB_dil_IMUa_m1_fOR_a1a2pC(i, n1, n2, w1, w2, :) = tmp_FM_segTB_dil_IMUa_m1_fOR_a1a2pC;
                                FM_segTB_dil_IMUa_m2_fOR_a1a2pL{i, n1, n2, w1, w2, :} = tmp_FM_segTB_dil_IMUa_m2_fOR_a1a2pL;
                                FM_segTB_dil_IMUa_m2_fOR_a1a2pC(i, n1, n2, w1, w2, :) = tmp_FM_segTB_dil_IMUa_m2_fOR_a1a2pC;

                                clear tmp_FM_segTB_dil_fOR_a1a2pL tmp_FM_segTB_dil_fOR_a1a2pC ...
                                    tmp_FM_segTB_dil_IMUa_m1_fOR_a1a2pL tmp_FM_segTB_dil_IMUa_m1_fOR_a1a2pC ...
                                    tmp_FM_segTB_dil_IMUa_m2_fOR_a1a2pL tmp_FM_segTB_dil_IMUa_m2_fOR_a1a2pC
                                % ---------------------------------------------------------------------------------------------------------------------------------

                                %% match of FM and sensation
                                matchTB_FM_Sens{i, n1, n2, w1, w2, :} = FM_segTB_dil_fOR_a1a2pL{i, n1, n2, w1, w2, :} .* sensT_map{i, n1, :};
                                matchTB_FM_Sens_num(i, n1, n2, w1, w2, :) = length(unique(matchTB_FM_Sens{i, n1, n2, w1, w2, :})) - 1;

                                fprintf('---- FM_segTB_dil_fOR_acceORacouORpiel: %d vs sensT_map: %d > match: %d ...\n', ...
                                    FM_segTB_dil_fOR_a1a2pC(i, n1, n2, w1, w2, :), sensT_mapC(i, n1, :), matchTB_FM_Sens_num(i, n1, n2, w1, w2, :));
                                % ---------------------------------------------------------------------------------------------------

                                matchTB_FM_Sens_INV{i, n1, n2, w1, w2, :} = sensT_mapL{i, n1, :} .* FM_segTB_dil_fOR_a1a2p{i, n1, n2, w1, w2, :};
                                matchTB_FM_Sens_num_INV(i, n1, n2, w1, w2, :) = length(unique(matchTB_FM_Sens_INV{i, n1, n2, w1, w2, :})) - 1;

                                fprintf('---- *Inversed* sensT_map: %d vs FM_segTB_dil_fOR_acceORacouORpiel: %d > match: %d ...\n', ...
                                    sensT_mapC(i, n1, :), FM_segTB_dil_fOR_a1a2pC(i, n1, n2, w1, w2, :), matchTB_FM_Sens_num_INV(i, n1, n2, w1, w2, :));

                                % sensitivity & precision
                                matchTB_FM_Sens_sensi(i, n1, n2, w1, w2, :) = matchTB_FM_Sens_num(i, n1, n2, w1, w2, :) / sensT_mapC(i, n1, :);
                                matchTB_FM_Sens_preci(i, n1, n2, w1, w2, :) = matchTB_FM_Sens_num(i, n1, n2, w1, w2, :) / FM_segTB_dil_fOR_a1a2pC(i, n1, n2, w1, w2, :);
                                fprintf('---- **** FM_segTB_dil_fOR_acceORacouORpiel - sensitivity & precision: %d & %d ...\n', ...
                                    matchTB_FM_Sens_sensi(i, n1, n2, w1, w2, :), matchTB_FM_Sens_preci(i, n1, n2, w1, w2, :));
                                % ---------------------------------------------------------------------------------------------------

                                matchTB_FM_Sens_IMUa_m1{i, n1, n2, w1, w2, :} = FM_segTB_dil_IMUa_m1_fOR_a1a2pL{i, n1, n2, w1, w2, :} .* sensTB_mapIMUa{i, n1, :};
                                matchTB_FM_Sens_IMUa_m1_num(i, n1, n2, w1, w2, :) = length(unique(matchTB_FM_Sens_IMUa_m1{i, n1, n2, w1, w2, :})) - 1;

                                fprintf('---- FM_segTB_dil_IMUa_m1_fOR_acceORacouORpiel: %d vs sensTB_mapIMUa: %d > match: %d ...\n', ...
                                    FM_segTB_dil_IMUa_m1_fOR_a1a2pC(i, n1, n2, w1, w2, :), sensTB_mapIMUaC(i, n1, :), matchTB_FM_Sens_IMUa_m1_num(i, n1, n2, w1, w2, :));
                                % ---------------------------------------------------------------------------------------------------

                                matchTB_FM_Sens_IMUa_m1_INV{i, n1, n2, w1, w2, :} = sensTB_mapIMUaL{i, n1, :} .* FM_segTB_dil_IMUa_m1_fOR_a1a2p{i, n1, n2, w1, w2, :};
                                matchTB_FM_Sens_IMUa_m1_num_INV(i, n1, n2, w1, w2, :) = length(unique(matchTB_FM_Sens_IMUa_m1_INV{i, n1, n2, w1, w2, :})) - 1;

                                fprintf('---- *Inversed* sensT_map: %d vs FM_segTB_dil_IMUa_m1_fOR_acceORacouORpiel: %d > match: %d ...\n', ...
                                    sensTB_mapIMUaC(i, n1, :), FM_segTB_dil_IMUa_m1_fOR_a1a2pC(i, n1, n2, w1, w2, :), matchTB_FM_Sens_IMUa_m1_num(i, n1, n2, w1, w2, :));

                                % sensitivity & precision
                                matchTB_FM_Sens_IMUa_m1_sensi(i, n1, n2, w1, w2, :) = matchTB_FM_Sens_IMUa_m1_num(i, n1, n2, w1, w2, :) / sensTB_mapIMUaC(i, n1, :);
                                matchTB_FM_Sens_IMUa_m1_preci(i, n1, n2, w1, w2, :) = matchTB_FM_Sens_IMUa_m1_num(i, n1, n2, w1, w2, :) / FM_segTB_dil_IMUa_m1_fOR_a1a2pC(i, n1, n2, w1, w2, :);

                                fprintf('---- **** FM_segTB_dil_IMUa_m1_fOR_acceORacouORpielC - sensitivity & precision: %d & %d ...\n', ...
                                    matchTB_FM_Sens_IMUa_m1_sensi(i, n1, n2, w1, w2, :), matchTB_FM_Sens_IMUa_m1_preci(i, n1, n2, w1, w2, :));
                                % -----------------------------------------------------------------------------------------------------

                                matchTB_FM_Sens_IMUa_m2{i, n1, n2, w1, w2, :} = FM_segTB_dil_IMUa_m2_fOR_a1a2pL{i, n1, n2, w1, w2, :} .* sensTB_mapIMUa{i, n1, :};
                                matchTB_FM_Sens_IMUa_m2_num(i, n1, n2, w1, w2, :) = length(unique(matchTB_FM_Sens_IMUa_m2{i, n1, n2, w1, w2, :})) - 1;

                                fprintf('---- FM_segTB_dil_IMUa_m2_fOR_acceORacouORpiel: %d vs sensTB_mapIMUa: %d > match: %d ...\n', ...
                                    FM_segTB_dil_IMUa_m2_fOR_a1a2pC(i, n1, n2, w1, w2, :), sensTB_mapIMUaC(i, n1, :), matchTB_FM_Sens_IMUa_m2_num(i, n1, n2, w1, w2, :));
                                % ---------------------------------------------------------------------------------------------------

                                matchTB_FM_Sens_IMUa_m2_INV{i, n1, n2, w1, w2, :} = sensTB_mapIMUaL{i, n1, :} .* FM_segTB_dil_IMUa_m2_fOR_a1a2p{i, n1, n2, w1, w2, :};
                                matchTB_FM_Sens_IMUa_m2_num_INV(i, n1, n2, w1, w2, :) = length(unique(matchTB_FM_Sens_IMUa_m2_INV{i, n1, n2, w1, w2, :})) - 1;

                                fprintf('---- *Inversed* sensTB_mapIMUa: %d vs FM_segTB_dil_IMUa_m2_fOR_acceORacouORpiel: %d > match: %d ...\n', ...
                                    sensTB_mapIMUaC(i, n1, :), FM_segTB_dil_IMUa_m2_fOR_a1a2pC(i, n1, n2, w1, w2, :), matchTB_FM_Sens_IMUa_m2_num(i, n1, n2, w1, w2, :));

                                % sensitivity & precision
                                matchTB_FM_Sens_IMUa_m2_sensi(i, n1, n2, w1, w2, :) = matchTB_FM_Sens_IMUa_m2_num(i, n1, n2, w1, w2, :) / sensTB_mapIMUaC(i, n1, :);
                                matchTB_FM_Sens_IMUa_m2_preci(i, n1, n2, w1, w2, :) = matchTB_FM_Sens_IMUa_m2_num(i, n1, n2, w1, w2, :) / FM_segTB_dil_IMUa_m2_fOR_a1a2pC(i, n1, n2, w1, w2, :);

                                fprintf('---- **** FM_segTB_dil_IMUa_m2_fOR_acceORacouORpielC - sensitivity & precision: %d & %d ...\n', ...
                                    matchTB_FM_Sens_IMUa_m2_sensi(i, n1, n2, w1, w2, :), matchTB_FM_Sens_IMUa_m2_preci(i, n1, n2, w1, w2, :));
                                % ---------------------------------------------------------------------------------------------------

                                fprintf('**** N*std IMU: %d; SNR acour/piez: %d, weights for noise/low singal levels (%d & %d) ...\n\n\n', ...
                                    Nstd_IMU_TB(n1), snr_acou_th(n2), noise_quanW(w1), low_quanW(w2));

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
            cd(fdir);
            save(['RA302_butter_' group '_' data_category '_P' sprintf('%03d',fidx_s) '_' sprintf('%03d',fidx_e) ...
                '_FMsnr' num2str(max(snr_acou_th)) ...
                '_IMUx' num2str(Nstd_IMU_TB) ...
                '_OLs' num2str(overlap_percSens*100) ...
                '_OLf' num2str(overlap_percFM*100) ...
                '_LN' num2str(noise_quanL*100) ...
                '_LL' num2str(low_quanL*100) ...
                '_WN' num2str(noise_quanW*100) ...
                '_WL' num2str(low_quanW*100) '_IMUSens.mat'], ...
                'IMUaTB_th', 'IMUaTB_map', 'IMUaTB_mapDil', ...
                'sensT_map', 'sensT_mapL', 'sensT_mapC', ...
                'sensTB_mapIMUa', 'sensTB_mapIMUaL', 'sensTB_mapIMUaC', ...
                '-v7.3');
            save(['RA302_butter_' group '_' data_category '_P' sprintf('%03d',fidx_s) '_' sprintf('%03d',fidx_e) ...
                '_FMsnr' num2str(max(snr_acou_th)) ...
                '_IMUx' num2str(Nstd_IMU_TB) ...
                '_OLs' num2str(overlap_percSens*100) ...
                '_OLf' num2str(overlap_percFM*100) ...
                '_LN' num2str(noise_quanL*100) ...
                '_LL' num2str(low_quanL*100) ...
                '_WN' num2str(noise_quanW*100) ...
                '_WL' num2str(low_quanW*100) '_FM.mat'], ...
                'FM_suiteTB', ...
                'FM_TB_th', 'FM_segTB', 'FM_segTB_dil', ...
                'FM_segTB_dil_IMUa_m1', ...
                'FM_segTB_dil_IMUa_m2', ...
                'FM_segTB_dil_fOR_a1a2', 'FM_segTB_dil_fOR_a1p', 'FM_segTB_dil_fOR_a2p', ...
                'FM_segTB_dil_fOR_acce', 'FM_segTB_dil_IMUa_m1_fOR_a1a2', 'FM_segTB_dil_IMUa_m1_fOR_a1p', ...
                'FM_segTB_dil_IMUa_m1_fOR_a2p', 'FM_segTB_dil_IMUa_m1_fOR_acce', 'FM_segTB_dil_IMUa_m2_fOR_a1a2', 'FM_segTB_dil_IMUa_m2_fOR_a1p', ...
                'FM_segTB_dil_IMUa_m2_fOR_a2p', 'FM_segTB_dil_IMUa_m2_fOR_acce', 'FM_segTB_fOR_a1a2', 'FM_segTB_fOR_a1p', 'FM_segTB_fOR_a2p', 'FM_segTB_fOR_acce', ...
                'FM_segTB_fOR_acou', 'FM_segTB_fOR_piez', 'FM_segTB_fOR_a1a2p', ...
                'FM_segTB_dil_fOR_acou', 'FM_segTB_dil_fOR_piez', 'FM_segTB_dil_fOR_a1a2p', ...
                'FM_segTB_dil_IMUa_m1_fOR_acou', 'FM_segTB_dil_IMUa_m1_fOR_piez', 'FM_segTB_dil_IMUa_m1_fOR_a1a2p', ...
                'FM_segTB_dil_IMUa_m2_fOR_acou', 'FM_segTB_dil_IMUa_m2_fOR_piez', 'FM_segTB_dil_IMUa_m2_fOR_a1a2p', ...
                'FM_segTB_dil_fOR_a1a2pL', 'FM_segTB_dil_fOR_a1a2pC', ...
                'FM_segTB_dil_IMUa_m1_fOR_a1a2pL', 'FM_segTB_dil_IMUa_m1_fOR_a1a2pC', ...
                'FM_segTB_dil_IMUa_m2_fOR_a1a2pL', 'FM_segTB_dil_IMUa_m2_fOR_a1a2pC', ...
                '-v7.3');
            save(['RA302_butter_' group '_' data_category '_P' sprintf('%03d',fidx_s) '_' sprintf('%03d',fidx_e) ...
                '_FMsnr' num2str(max(snr_acou_th)) ...
                '_IMUx' num2str(Nstd_IMU_TB) ...
                '_OLs' num2str(overlap_percSens*100) ...
                '_OLf' num2str(overlap_percFM*100) ...
                '_LN' num2str(noise_quanL*100) ...
                '_LL' num2str(low_quanL*100) ...
                '_WN' num2str(noise_quanW*100) ...
                '_WL' num2str(low_quanW*100) '_FM_label.mat'], ...
                'FM_segTB_dil_fOR_a1a2pL', 'FM_segTB_dil_fOR_a1a2pC', ...
                'FM_segTB_dil_IMUa_m1_fOR_a1a2pL', 'FM_segTB_dil_IMUa_m1_fOR_a1a2pC', ...
                'FM_segTB_dil_IMUa_m2_fOR_a1a2pL', 'FM_segTB_dil_IMUa_m2_fOR_a1a2pC', ...
                '-v7.3');
            save(['RA302_butter_' group '_' data_category '_P' sprintf('%03d',fidx_s) '_' sprintf('%03d',fidx_e) ...
                '_FMsnr' num2str(max(snr_acou_th)) ...
                '_IMUx' num2str(Nstd_IMU_TB) ...
                '_OLs' num2str(overlap_percSens*100) ...
                '_OLf' num2str(overlap_percFM*100) ...
                '_LN' num2str(noise_quanL*100) ...
                '_LL' num2str(low_quanL*100) ...
                '_WN' num2str(noise_quanW*100) ...
                '_WL' num2str(low_quanW*100) '_Match.mat'], ...
                'matchTB_FM_Sens', 'matchTB_FM_Sens_num', ...
                'matchTB_FM_Sens_INV', 'matchTB_FM_Sens_num_INV', ...
                'matchTB_FM_Sens_IMUa_m1', 'matchTB_FM_Sens_IMUa_m1_num', ...
                'matchTB_FM_Sens_IMUa_m1_INV', 'matchTB_FM_Sens_IMUa_m1_num_INV', ...
                'matchTB_FM_Sens_IMUa_m2', 'matchTB_FM_Sens_IMUa_m2_num', ...
                'matchTB_FM_Sens_IMUa_m2_INV', 'matchTB_FM_Sens_IMUa_m2_num_INV', ...
                'matchTB_FM_Sens_sensi', 'matchTB_FM_Sens_preci', ...
                'matchTB_FM_Sens_IMUa_m1_sensi', 'matchTB_FM_Sens_IMUa_m1_preci', ...
                'matchTB_FM_Sens_IMUa_m2_sensi', 'matchTB_FM_Sens_IMUa_m2_preci', ...
                '-v7.3');
            cd(curr_dir)


            %% clear working space
            clear curr_p fidx_s fidx_e num_filesP ...
                all_acceL_catP all_acceR_catP all_acouL_catP all_acouR_catP all_piezL_catP all_piezR_catP ...
                all_forc_catP all_IMUacce_catP ...
                all_sens1_catP all_sens2_catP all_sensMtx_catP all_timeV_catP

            clear IMUaTB_th IMUaTB_map IMUaTB_mapDil ...
                sensT_map sensT_mapL sensT_mapC ...
                sensTB_mapIMUa sensTB_mapIMUaL sensTB_mapIMUaC

            clear FM_suiteTB ...
                FM_TB_th FM_segTB FM_segTB_dil ...
                FM_segTB_dil_IMUa_m1 FM_segTB_dil_IMUa_m2 ...
                FM_segTB_fOR_acce FM_segTB_fOR_acou FM_segTB_fOR_piez ...
                FM_segTB_fOR_a1a2p ...
                FM_segTB_dil_fOR_acce FM_segTB_dil_fOR_acou FM_segTB_dil_fOR_piez FM_segTB_dil_fOR_a1a2p ...
                FM_segTB_dil_IMUa_m1_fOR_acce FM_segTB_dil_IMUa_m1_fOR_acou FM_segTB_dil_IMUa_m1_fOR_piez ...
                FM_segTB_dil_IMUa_m1_fOR_a1a2 FM_segTB_dil_IMUa_m1_fOR_a1p FM_segTB_dil_IMUa_m1_fOR_a2p FM_segTB_dil_IMUa_m1_fOR_a1a2p ...
                FM_segTB_dil_IMUa_m2_fOR_acou FM_segTB_dil_IMUa_m2_fOR_piez FM_segTB_dil_IMUa_m2_fOR_a1a2p ...
                FM_segTB_dil_fOR_a1a2pL FM_segTB_dil_fOR_a1a2pC ...
                FM_segTB_dil_IMUa_m1_fOR_a1a2pL FM_segTB_dil_IMUa_m1_fOR_a1a2pC ...
                FM_segTB_dil_IMUa_m2_fOR_a1a2pL FM_segTB_dil_IMUa_m2_fOR_a1a2pC ...
                FM_segTB_dil_IMUa_m2_fOR_a1a2 FM_segTB_dil_IMUa_m2_fOR_a1p FM_segTB_dil_IMUa_m2_fOR_a2p FM_segTB_dil_IMUa_m2_fOR_acce FM_segTB_fOR_a1a2 FM_segTB_fOR_a1p FM_segTB_fOR_a2p  ...
                FM_segTB_dil_fOR_a1a2pL FM_segTB_dil_fOR_a1a2pC ...
                FM_segTB_dil_IMUa_m1_fOR_a1a2pL FM_segTB_dil_IMUa_m1_fOR_a1a2pC ...
                FM_segTB_dil_IMUa_m2_fOR_a1a2pL FM_segTB_dil_IMUa_m2_fOR_a1a2pC ...
                FM_segTB_dil_fOR_a1a2 FM_segTB_dil_fOR_a1p FM_segTB_dil_fOR_a2p

            clear matchTB_FM_Sens matchTB_FM_Sens_num ...
                matchTB_FM_Sens_INV matchTB_FM_Sens_num_INV ...
                matchTB_FM_Sens_IMUa_m1 matchTB_FM_Sens_IMUa_m1_num ...
                matchTB_FM_Sens_IMUa_m1_INV matchTB_FM_Sens_IMUa_m1_num_INV ...
                matchTB_FM_Sens_IMUa_m2 matchTB_FM_Sens_IMUa_m2_num ...
                matchTB_FM_Sens_IMUa_m2_INV matchTB_FM_Sens_IMUa_m2_num_INV ...
                matchTB_FM_Sens_sensi matchTB_FM_Sens_preci ...
                matchTB_FM_Sens_IMUa_m1_sensi matchTB_FM_Sens_IMUa_m1_preci ...
                matchTB_FM_Sens_IMUa_m2_sensi matchTB_FM_Sens_IMUa_m2_preci

        end % portions 1-6

    end % overlap FM 

end % n x std IMUacce




