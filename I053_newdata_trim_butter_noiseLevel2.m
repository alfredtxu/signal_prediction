% remove the lowest 25% signals (FM sensors only), then calculating the adaptive thresholds on the remaining 75% 
clc
clear 
close all

% HOME / OFFICE
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% curr_dir = pwd;
cd(curr_dir);

%% Variables
% Frequency (Hz): sensor / sensation
freq = 400;

% the number of channels / FM sensors
num_channel = 16;
num_FMsensors = 6;

% thresholds: noisy level
noise_level = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];
low_quan = 0.25;

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
Nstd_IMU_TB = 3;
Nstd_acou_TB = 1;
Nstd_piez_TB = 1;

% thresholds: singal-noise ratio
snr_IMU_th = 25;
snr_acou_th = 22;
snr_piez_th = 22;

% percentage of overlap between senstation and maternal movement (IMU)
overlap_percSens = 0.20;
overlap_percFM = 0.10;
% -------------------------------------------------------------------------------------------------------

%% Data loading
% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007', 
% data category: F - focus; D - day; N - night
% HOME / OFFICE
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';
% fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

participant = '002JG';
data_folder = ['c_processed_' participant];
data_category = 'focus';
data_serial = 'filtering_movingAvg';

fdir_p = [fdir '\' data_folder];
preproc_dfile = ['Preproc01_' participant '_' data_category '_' data_serial '.mat'];
load([fdir_p '\' preproc_dfile]);

% the number of data files
num_files = size(sensT, 1);

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    fprintf('Current data file from %d (%d) - %s ...\n', i, num_files, preproc_dfile);

    %% IMUaTB thrsholding
    % SNR: IMUaTB
    % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
    % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
    % SNR = 10 * log10(P_signal / P_noise)
    tmp_SNR_IMUaTB = 0;
    tmp_Nstd_IMU_TB = Nstd_IMU_TB;
    tmp_IMUaTB = abs(IMUaccNetTB{i, :});

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
    
    IMUaTB_th(i).thresh = tmp_IMUaTB_th;
    IMUaTB_th(i).Nstd = tmp_Nstd_IMU_TB;
    IMUaTB_th(i).SNR = tmp_SNR_IMUaTB;
    
    fprintf('---- IMUa Butterworth filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUaTB_th, tmp_Nstd_IMU_TB-1, tmp_SNR_IMUaTB);
    
    % IMUa maps
    IMUaTB_map{i, :} = tmp_IMUaTB >= IMUaTB_th(i).thresh;
    IMUaTB_mapDil{i, :} = imdilate(IMUaTB_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUaTB tmp_Nstd_IMU_TB tmp_IMUaTB tmp_IMUaTB_th tmp_IMUaTB_P tmp_IMUaTB_N
    % -----------------------------------------------------------------------------------------------

    %% IMUgTB thrsholding
    % SNR: IMUgTB
    tmp_SNR_IMUgTB = 0;
    tmp_Nstd_IMU_TB = Nstd_IMU_TB;
    tmp_IMUgTB = abs(IMUgyrNetTB{i, :});

    while tmp_SNR_IMUgTB <= snr_IMU_th
        
        tmp_IMUgTB_th = mean(tmp_IMUgTB) + tmp_Nstd_IMU_TB * std(tmp_IMUgTB);

        tmp_IMUgTB_P = tmp_IMUgTB(tmp_IMUgTB >= tmp_IMUgTB_th);
        tmp_IMUgTB_N = tmp_IMUgTB(tmp_IMUgTB < tmp_IMUgTB_th);
    
        p_signal_IMUgTB = 0;
        for pa = 1 : length(tmp_IMUgTB_P)
            p_signal_IMUgTB = p_signal_IMUgTB + tmp_IMUgTB_P(pa)^2;
        end
        n_signal_IMUgTB = 0;
        for na = 1 : length(tmp_IMUgTB_N)
            n_signal_IMUgTB = n_signal_IMUgTB + tmp_IMUgTB_N(na)^2;
        end
    
        tmp_SNR_IMUgTB = 10 * log10((p_signal_IMUgTB / length(tmp_IMUgTB_P)) / (n_signal_IMUgTB / length(tmp_IMUgTB_N)));

        % increase the threshold weights if the SNR is not sufficient
        tmp_Nstd_IMU_TB = tmp_Nstd_IMU_TB + 1;
        
    end
    
    IMUgTB_th(i).thresh = tmp_IMUgTB_th;
    IMUgTB_th(i).Nstd = tmp_Nstd_IMU_TB;
    IMUgTB_th(i).SNR = tmp_SNR_IMUgTB;
    
    fprintf('---- IMUg Butterworth filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUgTB_th, tmp_Nstd_IMU_TB-1, tmp_SNR_IMUgTB);
    
    % IMUg maps
    IMUgTB_map{i, :} = tmp_IMUgTB >= IMUgTB_th(i).thresh;
    IMUgTB_mapDil{i, :} = imdilate(IMUgTB_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUgTB tmp_Nstd_IMU_TB tmp_IMUgTB tmp_IMUgTB_th tmp_IMUgTB_P tmp_IMUgTB_N
    % -----------------------------------------------------------------------------------------------

    %% Sensation
    % Initializing the map: sensT
    % senstation labels by connected components
    tmp_sensT = sensT{i, :};
    [tmp_sensTL, tmp_sensTC] = bwlabel(tmp_sensT);

    % initialization of sensation maps
    tmp_sensT_map = zeros(length(tmp_sensT), 1);     
    tmp_sensTB_mapIMUa = zeros(length(tmp_sensT), 1); 
    tmp_sensTB_mapIMUg = zeros(length(tmp_sensT), 1); 
    tmp_sensTB_mapIMUag = zeros(length(tmp_sensT), 1); 
    
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
        tmp_sensTB_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTB_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        % Remove maternal sensation coincided with body movement (IMU a/g) - butterworth
        tmp_xIMUaTB = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUaTB_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMUgTB = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUgTB_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTB >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
            tmp_sensTB_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTB_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMUgTB >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
            tmp_sensTB_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTB_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTB tmp_xIMUgTB
    end

    % sensation maps
    sensT_map{i, :} = tmp_sensT_map;
    sensTB_mapIMUa{i, :} = tmp_sensTB_mapIMUa;
    sensTB_mapIMUg{i, :} = tmp_sensTB_mapIMUg;
    sensTB_mapIMUag{i, :} = tmp_sensTB_mapIMUag;
    
    [tmp_sensT_mapL, tmp_sensT_mapC] = bwlabel(tmp_sensT_map);
    [tmp_sensTB_mapIMUaL, tmp_sensTB_mapIMUaC] = bwlabel(tmp_sensTB_mapIMUa);
    [tmp_sensTB_mapIMUgL, tmp_sensTB_mapIMUgC] = bwlabel(tmp_sensTB_mapIMUg);
    [tmp_sensTB_mapIMUagL, tmp_sensTB_mapIMUagC] = bwlabel(tmp_sensTB_mapIMUag);

    sensT_mapL{i, :} = tmp_sensT_mapL; 
    sensT_mapC(i, :) = tmp_sensT_mapC;
    sensTB_mapIMUaL{i, :} = tmp_sensTB_mapIMUaL;
    sensTB_mapIMUaC(i, :) = tmp_sensTB_mapIMUaC;
    sensTB_mapIMUgL{i, :} = tmp_sensTB_mapIMUgL;
    sensTB_mapIMUgC(i, :) = tmp_sensTB_mapIMUgC;
    sensTB_mapIMUagL{i, :} = tmp_sensTB_mapIMUagL;
    sensTB_mapIMUagC(i, :) = tmp_sensTB_mapIMUagC;

    fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d - %d - %d ... \n', ...
            sensT_mapC(i, :), sensTB_mapIMUaC(i, :), sensTB_mapIMUgC(i, :), sensTB_mapIMUagC(i, :));

    clear tmp_sensT tmp_sensTL tmp_sensTC tmp_sensT_map ...
          tmp_sensTB_mapIMUa tmp_sensTB_mapIMUg tmp_sensTB_mapIMUag ...
          tmp_sensT_mapL tmp_sensT_mapC ...
          tmp_sensTB_mapIMUaL tmp_sensTB_mapIMUaC ...
          tmp_sensTB_mapIMUgL tmp_sensTB_mapIMUgC ...
          tmp_sensTB_mapIMUagL tmp_sensTB_mapIMUagC
    % --------------------------------------------------------------------------------

    %% Segmenting FM sensors
    % 1. apply for an adaptive threshold
    % 2. remove body movement
    % 3. dilate the fetal movements at desinated duration
    FM_suiteTB{i, 1} = acouLTB{i,:};
    FM_suiteTB{i, 2} = acouMTB{i,:};
    FM_suiteTB{i, 3} = acouRTB{i,:};
    FM_suiteTB{i, 4} = piezLTB{i,:};
    FM_suiteTB{i, 5} = piezMTB{i,:};
    FM_suiteTB{i, 6} = piezRTB{i,:};

    % FM Sensors Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(FM_suiteTB{i, j2});
        tmp_low_cutoff = quantile(tmp_sensor_indiv, low_quan); 
        tmp_sensor_indiv_cutoff = tmp_sensor_indiv;
        tmp_sensor_indiv_cutoff(tmp_sensor_indiv_cutoff<=tmp_low_cutoff, :) = [];

        tmp_SNR_sensor = 0;

        if j2 <= num_FMsensors / 2
            tmp_std_times_FM = Nstd_acou_TB;
            tmp_snr_FM = snr_acou_th;
        else
            tmp_std_times_FM = Nstd_piez_TB;
            tmp_snr_FM = snr_piez_th;
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

        FM_TB_th(i, j2).thresh = tmp_FM_threshTB;
        FM_TB_th(i, j2).Nstd = tmp_std_times_FM;
        FM_TB_th(i, j2).SNE = tmp_SNR_sensor;

        fprintf('---- FM(%d-th sensor) Butterworth filtering: threshold > %d; SNR (%d*std): %d ... \n', ...
                j2, tmp_FM_threshTB, tmp_std_times_FM-1, tmp_SNR_sensor);

        clear tmp_SNR_sensor tmp_std_times_FM tmp_snr_FM ...
              tmp_FM_removed tmp_FM_segmented tmp_FM_threshTB
        % --------------------------------------------------------------------------------------------------------------

        %% segmentation by thresholding
        FM_segTB{i, j2} = (tmp_sensor_indiv >= FM_TB_th(i, j2).thresh);
        FM_segTB_dil{i, j2} = double(imdilate(FM_segTB{i, j2}, FM_lse)); 

        % labels by connected components
        [tmp_LTB, tmp_CTB] = bwlabel(FM_segTB{i, j2});
        [tmp_LTB_dil, tmp_CTB_dil] = bwlabel(FM_segTB_dil{i, j2});

        % Method 1 - denoise segmentations by removing movements
        FM_segTB_dil_IMUa_m1{i, j2} = FM_segTB_dil{i, j2} .* (1-IMUaTB_mapDil{i,:}); 
        FM_segTB_dil_IMUg_m1{i, j2} = FM_segTB_dil{i, j2} .* (1-IMUgTB_mapDil{i,:}); 
        FM_segTB_dil_IMUag_m1{i, j2} = FM_segTB_dil{i, j2} .* (1-IMUaTB_mapDil{i,:}) .* (1-IMUgTB_mapDil{i,:}); 
        
        % labels by connected components
        [tmp_LTB_IMUa_m1, tmp_CTB_IMUa_m1] = bwlabel(FM_segTB_dil_IMUa_m1{i, j2});
        [tmp_LTB_IMUg_m1, tmp_CTB_IMUg_m1] = bwlabel(FM_segTB_dil_IMUg_m1{i, j2});
        [tmp_LTB_IMUag_m1, tmp_CTB_IMUag_m1] = bwlabel(FM_segTB_dil_IMUag_m1{i, j2});
        % ------------------------------------------------------------------------------------------------------

        % Method 2 - further segmentations by removing movements
        tmp_segTB_dil_IMUa_m2 = zeros(length(FM_segTB_dil{i, j2}), 1); 
        tmp_segTB_dil_IMUg_m2 = zeros(length(FM_segTB_dil{i, j2}), 1);
        tmp_segTB_dil_IMUag_m2 = zeros(length(FM_segTB_dil{i, j2}), 1);
        
        for c = 1: tmp_CTB_dil
            
            tmp_idx = find(tmp_LTB_dil == c);
            
            tmp_segTB_dil_IMUa_m2(tmp_idx, :) = 1;
            tmp_segTB_dil_IMUg_m2(tmp_idx, :) = 1;
            tmp_segTB_dil_IMUag_m2(tmp_idx, :) = 1;

            tmp_xIMUaTB = sum(FM_segTB_dil{i, j2}(tmp_idx) .* IMUaTB_mapDil{i, :}(tmp_idx));
            tmp_xIMUgTB = sum(FM_segTB_dil{i, j2}(tmp_idx) .* IMUgTB_mapDil{i, :}(tmp_idx));

            if tmp_xIMUaTB >= overlap_percFM*length(tmp_idx)
                tmp_segTB_dil_IMUa_m2(tmp_idx) = 0;
                tmp_segTB_dil_IMUag_m2(tmp_idx) = 0;
            end
            
            if tmp_xIMUgTB >= overlap_percFM*length(tmp_idx)
                tmp_segTB_dil_IMUg_m2(tmp_idx) = 0;
                tmp_segTB_dil_IMUag_m2(tmp_idx) = 0;
            end

            clear tmp_idx tmp_xIMUaTB tmp_xIMUgTB
        end

        [tmp_LTB_IMUa_m2, tmp_CTB_IMUa_m2] = bwlabel(tmp_segTB_dil_IMUa_m2);
        [tmp_LTB_IMUg_m2, tmp_CTB_IMUg_m2] = bwlabel(tmp_segTB_dil_IMUg_m2);
        [tmp_LTB_IMUag_m2, tmp_CTB_IMUag_m2] = bwlabel(tmp_segTB_dil_IMUag_m2);

        FM_segTB_dil_IMUa_m2{i, j2} = tmp_segTB_dil_IMUa_m2;
        FM_segTB_dil_IMUg_m2{i, j2} = tmp_segTB_dil_IMUg_m2;
        FM_segTB_dil_IMUag_m2{i, j2} = tmp_segTB_dil_IMUag_m2;
        
        % display instant performance
        fprintf('---- FM(%d-th sensor) Butterworth filtering > Denoising: method 1 vs method 2: IMUa > %d - %d; IMUg > %d - %d; IMUag > %d - %d ...\n', ...
                j2, tmp_CTB_IMUa_m1, tmp_CTB_IMUa_m2, tmp_CTB_IMUg_m1, tmp_CTB_IMUg_m2, tmp_CTB_IMUag_m1, tmp_CTB_IMUag_m2);
        % -------------------------------------------------------------------------------------------------------

        clear tmp_sensor_indiv_ori tmp_sensor_indiv tmp_low_cutoff  ...
              tmp_LTB tmp_CTB tmp_LTB_dil tmp_CTB_dil ...
              tmp_LTB_IMUa_m1 tmp_CTB_IMUa_m1 tmp_LTB_IMUg_m1 tmp_CTB_IMUg_m1 tmp_LTB_IMUag_m1 tmp_CTB_IMUag_m1 ...
              tmp_segTB_dil_IMUa_m2 tmp_segTB_dil_IMUg_m2 tmp_segTB_dil_IMUag_m2 ...
              tmp_LTB_IMUa_m2 tmp_CTB_IMUa_m2 tmp_LTB_IMUg_m2 tmp_CTB_IMUg_m2 tmp_LTB_IMUag_m2 tmp_CTB_IMUag_m2
    end
    % -------------------------------------------------------------------------
    
    %% Sensor fusion
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'OR'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    FM_segTB_fOR_acou{i, :} = double(FM_segTB{i, 1} | FM_segTB{i, 2} | FM_segTB{i, 3});
    FM_segTB_fOR_piez{i, :} = double(FM_segTB{i, 4} | FM_segTB{i, 5} | FM_segTB{i, 6});
    FM_segTB_fOR_acouORpiez{i, :}  = double(FM_segTB_fOR_acou{i,:} | FM_segTB_fOR_piez{i, :});

    FM_segTB_dil_fOR_acou{i, :} = double(FM_segTB{i, 1} | FM_segTB{i, 2} | FM_segTB{i, 3});
    FM_segTB_dil_fOR_piez{i, :} = double(FM_segTB{i, 4} | FM_segTB{i, 5} | FM_segTB{i, 6});
    FM_segTB_dil_fOR_acouORpiez{i, :} = double(FM_segTB_dil_fOR_acou{i,:} | FM_segTB_dil_fOR_piez{i, :});
    
    FM_segTB_dil_IMUag_m1_fOR_acou{i, :} = double(FM_segTB_dil_IMUag_m1{i, 1} | FM_segTB_dil_IMUag_m1{i, 2} | FM_segTB_dil_IMUag_m1{i, 3});
    FM_segTB_dil_IMUag_m1_fOR_piez{i, :} = double(FM_segTB_dil_IMUag_m1{i, 4} | FM_segTB_dil_IMUag_m1{i, 5} | FM_segTB_dil_IMUag_m1{i, 6});
    FM_segTB_dil_IMUag_m1_fOR_acouORpiez{i, :}  = double(FM_segTB_dil_IMUag_m1_fOR_acou{i,:} | FM_segTB_dil_IMUag_m1_fOR_piez{i, :});

    FM_segTB_dil_IMUag_m2_fOR_acou{i, :} = double(FM_segTB_dil_IMUag_m2{i, 1} | FM_segTB_dil_IMUag_m2{i, 2} | FM_segTB_dil_IMUag_m2{i, 3});
    FM_segTB_dil_IMUag_m2_fOR_piez{i, :} = double(FM_segTB_dil_IMUag_m2{i, 4} | FM_segTB_dil_IMUag_m2{i, 5} | FM_segTB_dil_IMUag_m2{i, 6});
    FM_segTB_dil_IMUag_m2_fOR_acouORpiez{i, :}  = double(FM_segTB_dil_IMUag_m2_fOR_acou{i,:} | FM_segTB_dil_IMUag_m2_fOR_piez{i, :});

    [tmp_FM_segTB_dil_fOR_acouORpiezL, tmp_FM_segTB_dil_fOR_acouORpiezC] = bwlabel(FM_segTB_dil_fOR_acouORpiez{i, :});
    [tmp_FM_segTB_dil_IMUag_m1_fOR_acouORpiezL, tmp_FM_segTB_dil_IMUag_m1_fOR_acouORpiezC] = bwlabel(FM_segTB_dil_IMUag_m1_fOR_acouORpiez{i, :});
    [tmp_FM_segTB_dil_IMUag_m2_fOR_acouORpiezL, tmp_FM_segTB_dil_IMUag_m2_fOR_acouORpiezC] = bwlabel(FM_segTB_dil_IMUag_m2_fOR_acouORpiez{i, :});

    FM_segTB_dil_fOR_acouORpiezL{i, :} = tmp_FM_segTB_dil_fOR_acouORpiezL; 
    FM_segTB_dil_fOR_acouORpiezC(i, :) = tmp_FM_segTB_dil_fOR_acouORpiezC;
    FM_segTB_dil_IMUag_m1_fOR_acouORpiezL{i, :} = tmp_FM_segTB_dil_IMUag_m1_fOR_acouORpiezL; 
    FM_segTB_dil_IMUag_m1_fOR_acouORpiezC(i, :) = tmp_FM_segTB_dil_IMUag_m1_fOR_acouORpiezC;
    FM_segTB_dil_IMUag_m2_fOR_acouORpiezL{i, :} = tmp_FM_segTB_dil_IMUag_m2_fOR_acouORpiezL; 
    FM_segTB_dil_IMUag_m2_fOR_acouORpiezC(i, :) = tmp_FM_segTB_dil_IMUag_m2_fOR_acouORpiezC;

    clear tmp_FM_segTB_dil_fOR_acouORpiezL tmp_FM_segTB_dil_fOR_acouORpiezC ...
          tmp_FM_segTB_dil_IMUag_m1_fOR_acouORpiezL tmp_FM_segTB_dil_IMUag_m1_fOR_acouORpiezC ...
          tmp_FM_segTB_dil_IMUag_m2_fOR_acouORpiezL tmp_FM_segTB_dil_IMUag_m2_fOR_acouORpiezC
    % ---------------------------------------------------------------------------------------------------------------------------------

    %% match of FM and sensation
    matchTB_FM_Sens{i, :} = FM_segTB_dil_fOR_acouORpiezL{i, :} .* sensT_map{i, :};
    matchTB_FM_Sens_num(i, :) = length(unique(matchTB_FM_Sens{i, :})) - 1;

    fprintf('---- FM_segTB_dil_fOR_acouORpiez: %d vs sensT_map: %d > match: %d ...\n', ...
            FM_segTB_dil_fOR_acouORpiezC(i, :), sensT_mapC(i, :), matchTB_FM_Sens_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTB_FM_Sens_INV{i, :} = sensT_mapL{i, :} .* FM_segTB_dil_fOR_acouORpiez{i, :};
    matchTB_FM_Sens_num_INV(i, :) = length(unique(matchTB_FM_Sens_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTB_dil_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensT_mapC(i, :), FM_segTB_dil_fOR_acouORpiezC(i, :), matchTB_FM_Sens_num_INV(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTB_FM_Sens_IMUag_m1{i, :} = FM_segTB_dil_IMUag_m1_fOR_acouORpiezL{i, :} .* sensTB_mapIMUag{i, :};
    matchTB_FM_Sens_IMUag_m1_num(i, :) = length(unique(matchTB_FM_Sens_IMUag_m1{i, :})) - 1;

    fprintf('---- FM_segTB_dil_IMUag_m1_fOR_acouORpiez: %d vs sensTB_mapIMUag: %d > match: %d ...\n', ...
            FM_segTB_dil_IMUag_m1_fOR_acouORpiezC(i, :), sensTB_mapIMUagC(i, :), matchTB_FM_Sens_IMUag_m1_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTB_FM_Sens_IMUag_m1_INV{i, :} = sensTB_mapIMUagL{i, :} .* FM_segTB_dil_IMUag_m1_fOR_acouORpiez{i, :};
    matchTB_FM_Sens_IMUag_m1_num_INV(i, :) = length(unique(matchTB_FM_Sens_IMUag_m1_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTB_dil_IMUag_m1_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensTB_mapIMUagC(i, :), FM_segTB_dil_IMUag_m1_fOR_acouORpiezC(i, :), matchTB_FM_Sens_IMUag_m1_num(i, :));
    % -----------------------------------------------------------------------------------------------------

    matchTB_FM_Sens_IMUag_m2{i, :} = FM_segTB_dil_IMUag_m2_fOR_acouORpiezL{i, :} .* sensTB_mapIMUag{i, :};
    matchTB_FM_Sens_IMUag_m2_num(i, :) = length(unique(matchTB_FM_Sens_IMUag_m2{i, :})) - 1;

    fprintf('---- FM_segTB_dil_IMUag_m2_fOR_acouORpiez: %d vs sensTB_mapIMUag: %d > match: %d ...\n', ...
            FM_segTB_dil_IMUag_m2_fOR_acouORpiezC(i, :), sensTB_mapIMUagC(i, :), matchTB_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTB_FM_Sens_IMUag_m2_INV{i, :} = sensTB_mapIMUagL{i, :} .* FM_segTB_dil_IMUag_m2_fOR_acouORpiez{i, :};
    matchTB_FM_Sens_IMUag_m2_num_INV(i, :) = length(unique(matchTB_FM_Sens_IMUag_m2_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensTB_mapIMUag: %d vs FM_segTB_dil_IMUag_m2_fOR_acouORpiez: %d > match: %d ...\n\n\n', ...
            sensTB_mapIMUagC(i, :), FM_segTB_dil_IMUag_m2_fOR_acouORpiezC(i, :), matchTB_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

% end of loop for the files
end

% save segmented and fused data
% cd([fdir '\' data_folder]);
% save(['Seg01_butter_' participant '_' data_category '.mat'], ...
%       'IMUaTB_th', 'IMUaTB_map', 'IMUaTB_mapDil', ...
%       'IMUgTB_th', 'IMUgTB_map', 'IMUgTB_mapDil', ...
%       'sensT_map', 'sensT_mapL', 'sensT_mapC', ...
%       'sensTB_mapIMUa', 'sensTB_mapIMUaL', 'sensTB_mapIMUaC', ...
%       'sensTB_mapIMUg', 'sensTB_mapIMUgL', 'sensTB_mapIMUgC', ...
%       'sensTB_mapIMUag', 'sensTB_mapIMUagL', 'sensTB_mapIMUagC', ...
%       'FM_suiteTB', ...
%       'FM_TB_th', 'FM_segTB', 'FM_segTB_dil', ...
%       'FM_segTB_dil_IMUa_m1', 'FM_segTB_dil_IMUg_m1', 'FM_segTB_dil_IMUag_m1', ...
%       'FM_segTB_dil_IMUa_m2', 'FM_segTB_dil_IMUg_m2', 'FM_segTB_dil_IMUag_m2', ...
%       'FM_segTB_fOR_acou', 'FM_segTB_fOR_piez', 'FM_segTB_fOR_acouORpiez', ...
%       'FM_segTB_dil_fOR_acou', 'FM_segTB_dil_fOR_piez', 'FM_segTB_dil_fOR_acouORpiez', ...
%       'FM_segTB_dil_IMUag_m1_fOR_acou', 'FM_segTB_dil_IMUag_m1_fOR_piez', 'FM_segTB_dil_IMUag_m1_fOR_acouORpiez', ...
%       'FM_segTB_dil_IMUag_m2_fOR_acou', 'FM_segTB_dil_IMUag_m2_fOR_piez', 'FM_segTB_dil_IMUag_m2_fOR_acouORpiez', ...
%       'FM_segTB_dil_fOR_acouORpiezL', 'FM_segTB_dil_fOR_acouORpiezC', ...
%       'FM_segTB_dil_IMUag_m1_fOR_acouORpiezL', 'FM_segTB_dil_IMUag_m1_fOR_acouORpiezC', ...
%       'FM_segTB_dil_IMUag_m2_fOR_acouORpiezL', 'FM_segTB_dil_IMUag_m2_fOR_acouORpiezC', ...
%       'matchTB_FM_Sens', 'matchTB_FM_Sens_num', ...
%       'matchTB_FM_Sens_INV', 'matchTB_FM_Sens_num_INV', ...
%       'matchTB_FM_Sens_IMUag_m1', 'matchTB_FM_Sens_IMUag_m1_num', ...
%       'matchTB_FM_Sens_IMUag_m1_INV', 'matchTB_FM_Sens_IMUag_m1_num_INV', ...
%       'matchTB_FM_Sens_IMUag_m2', 'matchTB_FM_Sens_IMUag_m2_num', ...
%       'matchTB_FM_Sens_IMUag_m2_INV', 'matchTB_FM_Sens_IMUag_m2_num_INV', ...
%       '-v7.3');
% cd(curr_dir)











