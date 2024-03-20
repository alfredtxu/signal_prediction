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
Nstd_IMU_TBMA = 3;
Nstd_acou_TBMA = 1;
Nstd_piez_TBMA = 1;

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

    %% IMUaTBMA thrsholding
    % SNR: IMUaTBMA
    % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
    % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
    % SNR = 10 * log10(P_signal / P_noise)
    tmp_SNR_IMUaTBMA = 0;
    tmp_Nstd_IMU_TBMA = Nstd_IMU_TBMA;
    tmp_IMUaTBMA = abs(IMUaccNetTB_MA{i, :});

    while tmp_SNR_IMUaTBMA <= snr_IMU_th
        
        tmp_IMUaTBMA_th = mean(tmp_IMUaTBMA) + tmp_Nstd_IMU_TBMA * std(tmp_IMUaTBMA);

        tmp_IMUaTBMA_P = tmp_IMUaTBMA(tmp_IMUaTBMA >= tmp_IMUaTBMA_th);
        tmp_IMUaTBMA_N = tmp_IMUaTBMA(tmp_IMUaTBMA < tmp_IMUaTBMA_th);
    
        p_signal_IMUaTBMA = 0;
        for pa = 1 : length(tmp_IMUaTBMA_P)
            p_signal_IMUaTBMA = p_signal_IMUaTBMA + tmp_IMUaTBMA_P(pa)^2;
        end
        n_signal_IMUaTBMA = 0;
        for na = 1 : length(tmp_IMUaTBMA_N)
            n_signal_IMUaTBMA = n_signal_IMUaTBMA + tmp_IMUaTBMA_N(na)^2;
        end
    
        tmp_SNR_IMUaTBMA = 10 * log10((p_signal_IMUaTBMA / length(tmp_IMUaTBMA_P)) / (n_signal_IMUaTBMA / length(tmp_IMUaTBMA_N)));

        % increase the threshold weights if the SNR is not sufficient
        tmp_Nstd_IMU_TBMA = tmp_Nstd_IMU_TBMA + 1;
        
    end
    
    IMUaTBMA_th(i).thresh = tmp_IMUaTBMA_th;
    IMUaTBMA_th(i).Nstd = tmp_Nstd_IMU_TBMA;
    IMUaTBMA_th(i).SNR = tmp_SNR_IMUaTBMA;
    
    fprintf('---- IMUa Butterworth MA filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUaTBMA_th, tmp_Nstd_IMU_TBMA-1, tmp_SNR_IMUaTBMA);
    
    % IMUa maps
    IMUaTBMA_map{i, :} = tmp_IMUaTBMA >= IMUaTBMA_th(i).thresh;
    IMUaTBMA_mapDil{i, :} = imdilate(IMUaTBMA_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUaTBMA tmp_Nstd_IMU_TBMA tmp_IMUaTBMA tmp_IMUaTBMA_th tmp_IMUaTBMA_P tmp_IMUaTBMA_N
    % -----------------------------------------------------------------------------------------------

    %% IMUgTBMA thrsholding
    % SNR: IMUgTBMA
    tmp_SNR_IMUgTBMA = 0;
    tmp_Nstd_IMU_TBMA = Nstd_IMU_TBMA;
    tmp_IMUgTBMA = abs(IMUgyrNetTB_MA{i, :});

    while tmp_SNR_IMUgTBMA <= snr_IMU_th
        
        tmp_IMUgTBMA_th = mean(tmp_IMUgTBMA) + tmp_Nstd_IMU_TBMA * std(tmp_IMUgTBMA);

        tmp_IMUgTBMA_P = tmp_IMUgTBMA(tmp_IMUgTBMA >= tmp_IMUgTBMA_th);
        tmp_IMUgTBMA_N = tmp_IMUgTBMA(tmp_IMUgTBMA < tmp_IMUgTBMA_th);
    
        p_signal_IMUgTBMA = 0;
        for pa = 1 : length(tmp_IMUgTBMA_P)
            p_signal_IMUgTBMA = p_signal_IMUgTBMA + tmp_IMUgTBMA_P(pa)^2;
        end
        n_signal_IMUgTBMA = 0;
        for na = 1 : length(tmp_IMUgTBMA_N)
            n_signal_IMUgTBMA = n_signal_IMUgTBMA + tmp_IMUgTBMA_N(na)^2;
        end
    
        tmp_SNR_IMUgTBMA = 10 * log10((p_signal_IMUgTBMA / length(tmp_IMUgTBMA_P)) / (n_signal_IMUgTBMA / length(tmp_IMUgTBMA_N)));

        % increase the threshold weights if the SNR is not sufficient
        tmp_Nstd_IMU_TBMA = tmp_Nstd_IMU_TBMA + 1;
        
    end
    
    IMUgTBMA_th(i).thresh = tmp_IMUgTBMA_th;
    IMUgTBMA_th(i).Nstd = tmp_Nstd_IMU_TBMA;
    IMUgTBMA_th(i).SNR = tmp_SNR_IMUgTBMA;
    
    fprintf('---- IMUg Butterworth MA filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUgTBMA_th, tmp_Nstd_IMU_TBMA-1, tmp_SNR_IMUgTBMA);
    
     % IMUg maps
    IMUgTBMA_map{i, :} = tmp_IMUgTBMA >= IMUgTBMA_th(i).thresh;
    IMUgTBMA_mapDil{i, :} = imdilate(IMUgTBMA_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUgTBMA tmp_Nstd_IMU_TBMA tmp_IMUgTBMA tmp_IMUgTBMA_th tmp_IMUgTBMA_P tmp_IMUgTBMA_N
    % -----------------------------------------------------------------------------------------------

    %% Sensation
    % Initializing the map: sensT
    % senstation labels by connected components
    tmp_sensT_MA = sensT_MA_B{i, :};
    [tmp_sensT_MA_L, tmp_sensT_MA_C] = bwlabel(tmp_sensT_MA);

    % initialization of sensation maps
    tmp_sensT_MA_map = zeros(length(tmp_sensT_MA), 1);     
    tmp_sensTBMA_mapIMUa = zeros(length(tmp_sensT_MA), 1); 
    tmp_sensTBMA_mapIMUg = zeros(length(tmp_sensT_MA), 1); 
    tmp_sensTBMA_mapIMUag = zeros(length(tmp_sensT_MA), 1); 
    
    for j1 = 1 : tmp_sensT_MA_C

        % the idx range of the current cluster (component)
        tmp_idx = find(tmp_sensT_MA_L == j1);
        
        tmp_idxS = min(tmp_idx) - sens_dilation_sizeB; 
        tmp_idxE = max(tmp_idx) + sens_dilation_sizeF; 

        tmp_idxS = max(tmp_idxS, 1);
        tmp_idxE = min(tmp_idxE, length(tmp_sensT_MA_map));
        
        % sensation map
        tmp_sensT_MA_map(tmp_idxS:tmp_idxE) = 1; 

        tmp_sensTBMA_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTBMA_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTBMA_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        % Remove maternal sensation coincided with body movement (IMU a/g) - butterworth MA
        tmp_xIMUaTBMA = sum(tmp_sensT_MA_map(tmp_idxS:tmp_idxE) .* IMUaTBMA_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMUgTBMA = sum(tmp_sensT_MA_map(tmp_idxS:tmp_idxE) .* IMUgTBMA_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTBMA >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
            tmp_sensTBMA_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTBMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMUgTBMA >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
            tmp_sensTBMA_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTBMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTBMA tmp_xIMUgTBMA
    end

    % sensation maps
    sensT_map{i, :} = tmp_sensT_MA_map;
    sensTBMA_mapIMUa{i, :} = tmp_sensTBMA_mapIMUa;
    sensTBMA_mapIMUg{i, :} = tmp_sensTBMA_mapIMUg;
    sensTBMA_mapIMUag{i, :} = tmp_sensTBMA_mapIMUag;
    
    [tmp_sensT_mapL, tmp_sensT_mapC] = bwlabel(tmp_sensT_MA_map);
    [tmp_sensTBMA_mapIMUaL, tmp_sensTBMA_mapIMUaC] = bwlabel(tmp_sensTBMA_mapIMUa);
    [tmp_sensTBMA_mapIMUgL, tmp_sensTBMA_mapIMUgC] = bwlabel(tmp_sensTBMA_mapIMUg);
    [tmp_sensTBMA_mapIMUagL, tmp_sensTBMA_mapIMUagC] = bwlabel(tmp_sensTBMA_mapIMUag);

    sensT_mapL{i, :} = tmp_sensT_mapL; 
    sensT_mapC(i, :) = tmp_sensT_mapC;
    sensTBMA_mapIMUaL{i, :} = tmp_sensTBMA_mapIMUaL;
    sensTBMA_mapIMUaC(i, :) = tmp_sensTBMA_mapIMUaC;
    sensTBMA_mapIMUgL{i, :} = tmp_sensTBMA_mapIMUgL;
    sensTBMA_mapIMUgC(i, :) = tmp_sensTBMA_mapIMUgC;
    sensTBMA_mapIMUagL{i, :} = tmp_sensTBMA_mapIMUagL;
    sensTBMA_mapIMUagC(i, :) = tmp_sensTBMA_mapIMUagC;

    fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d - %d - %d ... \n', ...
            sensT_mapC(i, :), sensTBMA_mapIMUaC(i, :), sensTBMA_mapIMUgC(i, :), sensTBMA_mapIMUagC(i, :));

    clear tmp_sensT_MA tmp_sensT_MA_L tmp_sensT_MA_C tmp_sensT_MA_map ...
          tmp_sensTBMA_mapIMUa tmp_sensTBMA_mapIMUg tmp_sensTBMA_mapIMUag ...
          tmp_sensT_mapL tmp_sensT_mapC ...
          tmp_sensTBMA_mapIMUaL tmp_sensTBMA_mapIMUaC ...
          tmp_sensTBMA_mapIMUgL tmp_sensTBMA_mapIMUgC ...
          tmp_sensTBMA_mapIMUagL tmp_sensTBMA_mapIMUagC
    % --------------------------------------------------------------------------------

    %% Segmenting FM sensors
    % 1. apply for an adaptive threshold
    % 2. remove body movement
    % 3. dilate the fetal movements at desinated duration
    FM_suiteTBMA{i, 1} = acouLTB_MA{i,:};
    FM_suiteTBMA{i, 2} = acouMTB_MA{i,:};
    FM_suiteTBMA{i, 3} = acouRTB_MA{i,:};
    FM_suiteTBMA{i, 4} = piezLTB_MA{i,:};
    FM_suiteTBMA{i, 5} = piezMTB_MA{i,:};
    FM_suiteTBMA{i, 6} = piezRTB_MA{i,:};

    % FM Sensors Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(FM_suiteTBMA{i, j2});
        tmp_SNR_sensor = 0;

        if j2 <= num_FMsensors / 2
            tmp_std_times_FM = Nstd_acou_TBMA;
            tmp_snr_FM = snr_acou_th;
        else
            tmp_std_times_FM = Nstd_piez_TBMA;
            tmp_snr_FM = snr_piez_th;
        end

        while tmp_SNR_sensor <= tmp_snr_FM

            tmp_FM_threshTBMA = mean(tmp_sensor_indiv) + tmp_std_times_FM * std(tmp_sensor_indiv);
            if isnan(tmp_FM_threshTBMA)
                tmp_FM_threshTBMA = 0;
            end

            tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < tmp_FM_threshTBMA);
            tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= tmp_FM_threshTBMA);
    
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

        FM_TBMA_th(i, j2).thresh = tmp_FM_threshTBMA;
        FM_TBMA_th(i, j2).Nstd = tmp_std_times_FM;
        FM_TBMA_th(i, j2).SNE = tmp_SNR_sensor;

        fprintf('---- FM(%d-th sensor) Butterworth MA filtering: threshold > %d; SNR (%d*std): %d ... \n', ...
                j2, tmp_FM_threshTBMA, tmp_std_times_FM-1, tmp_SNR_sensor);

        clear tmp_SNR_sensor tmp_std_times_FM tmp_snr_FM ...
              tmp_FM_removed tmp_FM_segmented tmp_FM_threshTBMA
        % --------------------------------------------------------------------------------------------------------------

        %% segmentation by thresholding
        FM_segTBMA{i, j2} = (tmp_sensor_indiv >= FM_TBMA_th(i, j2).thresh);
        FM_segTBMA_dil{i, j2} = double(imdilate(FM_segTBMA{i, j2}, FM_lse)); 

        % labels by connected components
        [tmp_LTBMA, tmp_CTBMA] = bwlabel(FM_segTBMA{i, j2});
        [tmp_LTBMA_dil, tmp_CTBMA_dil] = bwlabel(FM_segTBMA_dil{i, j2});

        % Method 1 - denoise segmentations by removing movements
        FM_segTBMA_dil_IMUa_m1{i, j2} = FM_segTBMA_dil{i, j2} .* (1-IMUaTBMA_mapDil{i,:}); 
        FM_segTBMA_dil_IMUg_m1{i, j2} = FM_segTBMA_dil{i, j2} .* (1-IMUgTBMA_mapDil{i,:}); 
        FM_segTBMA_dil_IMUag_m1{i, j2} = FM_segTBMA_dil{i, j2} .* (1-IMUaTBMA_mapDil{i,:}) .* (1-IMUgTBMA_mapDil{i,:}); 
        
        % labels by connected components
        [tmp_LTBMA_IMUa_m1, tmp_CTBMA_IMUa_m1] = bwlabel(FM_segTBMA_dil_IMUa_m1{i, j2});
        [tmp_LTBMA_IMUg_m1, tmp_CTBMA_IMUg_m1] = bwlabel(FM_segTBMA_dil_IMUg_m1{i, j2});
        [tmp_LTBMA_IMUag_m1, tmp_CTBMA_IMUag_m1] = bwlabel(FM_segTBMA_dil_IMUag_m1{i, j2});
        % ------------------------------------------------------------------------------------------------------

        % Method 2 - further segmentations by removing movements
        tmp_segTBMA_dil_IMUa_m2 = zeros(length(FM_segTBMA_dil{i, j2}), 1); 
        tmp_segTBMA_dil_IMUg_m2 = zeros(length(FM_segTBMA_dil{i, j2}), 1);
        tmp_segTBMA_dil_IMUag_m2 = zeros(length(FM_segTBMA_dil{i, j2}), 1);
        
        for c = 1: tmp_CTBMA_dil
            
            tmp_idx = find(tmp_LTBMA_dil == c);
            
            tmp_segTBMA_dil_IMUa_m2(tmp_idx, :) = 1;
            tmp_segTBMA_dil_IMUg_m2(tmp_idx, :) = 1;
            tmp_segTBMA_dil_IMUag_m2(tmp_idx, :) = 1;

            tmp_xIMUaTBMA = sum(FM_segTBMA_dil{i, j2}(tmp_idx) .* IMUaTBMA_mapDil{i, :}(tmp_idx));
            tmp_xIMUgTBMA = sum(FM_segTBMA_dil{i, j2}(tmp_idx) .* IMUgTBMA_mapDil{i, :}(tmp_idx));

            if tmp_xIMUaTBMA >= overlap_percFM*length(tmp_idx)
                tmp_segTBMA_dil_IMUa_m2(tmp_idx) = 0;
                tmp_segTBMA_dil_IMUag_m2(tmp_idx) = 0;
            end
            
            if tmp_xIMUgTBMA >= overlap_percFM*length(tmp_idx)
                tmp_segTBMA_dil_IMUg_m2(tmp_idx) = 0;
                tmp_segTBMA_dil_IMUag_m2(tmp_idx) = 0;
            end

            clear tmp_idx tmp_xIMUaTBMA tmp_xIMUgTBMA
        end

        [tmp_LTBMA_IMUa_m2, tmp_CTBMA_IMUa_m2] = bwlabel(tmp_segTBMA_dil_IMUa_m2);
        [tmp_LTBMA_IMUg_m2, tmp_CTBMA_IMUg_m2] = bwlabel(tmp_segTBMA_dil_IMUg_m2);
        [tmp_LTBMA_IMUag_m2, tmp_CTBMA_IMUag_m2] = bwlabel(tmp_segTBMA_dil_IMUag_m2);

        FM_segTBMA_dil_IMUa_m2{i, j2} = tmp_segTBMA_dil_IMUa_m2;
        FM_segTBMA_dil_IMUg_m2{i, j2} = tmp_segTBMA_dil_IMUg_m2;
        FM_segTBMA_dil_IMUag_m2{i, j2} = tmp_segTBMA_dil_IMUag_m2;
        
        % display instant performance
        fprintf('---- FM(%d-th sensor) Butterworth MA filtering > Denoising: method 1 vs method 2: IMUa > %d - %d; IMUg > %d - %d; IMUag > %d - %d ...\n', ...
                j2, tmp_CTBMA_IMUa_m1, tmp_CTBMA_IMUa_m2, tmp_CTBMA_IMUg_m1, tmp_CTBMA_IMUg_m2, tmp_CTBMA_IMUag_m1, tmp_CTBMA_IMUag_m2);
        % -------------------------------------------------------------------------------------------------------

        clear tmp_sensor_indiv  ...
              tmp_LTBMA tmp_CTBMA tmp_LTBMA_dil tmp_CTBMA_dil ...
              tmp_LTBMA_IMUa_m1 tmp_CTBMA_IMUa_m1 tmp_LTBMA_IMUg_m1 tmp_CTBMA_IMUg_m1 tmp_LTBMA_IMUag_m1 tmp_CTBMA_IMUag_m1 ...
              tmp_segTBMA_dil_IMUa_m2 tmp_segTBMA_dil_IMUg_m2 tmp_segTBMA_dil_IMUag_m2 ...
              tmp_LTBMA_IMUa_m2 tmp_CTBMA_IMUa_m2 tmp_LTBMA_IMUg_m2 tmp_CTBMA_IMUg_m2 tmp_LTBMA_IMUag_m2 tmp_CTBMA_IMUag_m2
    end
    % -------------------------------------------------------------------------
    
    %% Sensor fusion
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'OR'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    FM_segTBMA_fOR_acou{i, :} = double(FM_segTBMA{i, 1} | FM_segTBMA{i, 2} | FM_segTBMA{i, 3});
    FM_segTBMA_fOR_piez{i, :} = double(FM_segTBMA{i, 4} | FM_segTBMA{i, 5} | FM_segTBMA{i, 6});
    FM_segTBMA_fOR_acouORpiez{i, :}  = double(FM_segTBMA_fOR_acou{i,:} | FM_segTBMA_fOR_piez{i, :});

    FM_segTBMA_dil_fOR_acou{i, :} = double(FM_segTBMA{i, 1} | FM_segTBMA{i, 2} | FM_segTBMA{i, 3});
    FM_segTBMA_dil_fOR_piez{i, :} = double(FM_segTBMA{i, 4} | FM_segTBMA{i, 5} | FM_segTBMA{i, 6});
    FM_segTBMA_dil_fOR_acouORpiez{i, :} = double(FM_segTBMA_dil_fOR_acou{i,:} | FM_segTBMA_dil_fOR_piez{i, :});
    
    FM_segTBMA_dil_IMUag_m1_fOR_acou{i, :} = double(FM_segTBMA_dil_IMUag_m1{i, 1} | FM_segTBMA_dil_IMUag_m1{i, 2} | FM_segTBMA_dil_IMUag_m1{i, 3});
    FM_segTBMA_dil_IMUag_m1_fOR_piez{i, :} = double(FM_segTBMA_dil_IMUag_m1{i, 4} | FM_segTBMA_dil_IMUag_m1{i, 5} | FM_segTBMA_dil_IMUag_m1{i, 6});
    FM_segTBMA_dil_IMUag_m1_fOR_acouORpiez{i, :}  = double(FM_segTBMA_dil_IMUag_m1_fOR_acou{i,:} | FM_segTBMA_dil_IMUag_m1_fOR_piez{i, :});

    FM_segTBMA_dil_IMUag_m2_fOR_acou{i, :} = double(FM_segTBMA_dil_IMUag_m2{i, 1} | FM_segTBMA_dil_IMUag_m2{i, 2} | FM_segTBMA_dil_IMUag_m2{i, 3});
    FM_segTBMA_dil_IMUag_m2_fOR_piez{i, :} = double(FM_segTBMA_dil_IMUag_m2{i, 4} | FM_segTBMA_dil_IMUag_m2{i, 5} | FM_segTBMA_dil_IMUag_m2{i, 6});
    FM_segTBMA_dil_IMUag_m2_fOR_acouORpiez{i, :}  = double(FM_segTBMA_dil_IMUag_m2_fOR_acou{i,:} | FM_segTBMA_dil_IMUag_m2_fOR_piez{i, :});

    [tmp_FM_segTBMA_dil_fOR_acouORpiezL, tmp_FM_segTBMA_dil_fOR_acouORpiezC] = bwlabel(FM_segTBMA_dil_fOR_acouORpiez{i, :});
    [tmp_FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezL, tmp_FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC] = bwlabel(FM_segTBMA_dil_IMUag_m1_fOR_acouORpiez{i, :});
    [tmp_FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezL, tmp_FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC] = bwlabel(FM_segTBMA_dil_IMUag_m2_fOR_acouORpiez{i, :});

    FM_segTBMA_dil_fOR_acouORpiezL{i, :} = tmp_FM_segTBMA_dil_fOR_acouORpiezL; 
    FM_segTBMA_dil_fOR_acouORpiezC(i, :) = tmp_FM_segTBMA_dil_fOR_acouORpiezC;
    FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezL{i, :} = tmp_FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezL; 
    FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC(i, :) = tmp_FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC;
    FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezL{i, :} = tmp_FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezL; 
    FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC(i, :) = tmp_FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC;

    clear tmp_FM_segTBMA_dil_fOR_acouORpiezL tmp_FM_segTBMA_dil_fOR_acouORpiezC ...
          tmp_FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezL tmp_FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC ...
          tmp_FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezL tmp_FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC
    % ---------------------------------------------------------------------------------------------------------------------------------

    %% match of FM and sensation
    matchTBMA_FM_Sens{i, :} = FM_segTBMA_dil_fOR_acouORpiezL{i, :} .* sensT_map{i, :};
    matchTBMA_FM_Sens_num(i, :) = length(unique(matchTBMA_FM_Sens{i, :})) - 1;

    fprintf('---- FM_segTBMA_dil_fOR_acouORpiez: %d vs sensT_map: %d > match: %d ...\n', ...
            FM_segTBMA_dil_fOR_acouORpiezC(i, :), sensT_mapC(i, :), matchTBMA_FM_Sens_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTBMA_FM_Sens_INV{i, :} = sensT_mapL{i, :} .* FM_segTBMA_dil_fOR_acouORpiez{i, :};
    matchTBMA_FM_Sens_num_INV(i, :) = length(unique(matchTBMA_FM_Sens_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTBMA_dil_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensT_mapC(i, :), FM_segTBMA_dil_fOR_acouORpiezC(i, :), matchTBMA_FM_Sens_num_INV(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTBMA_FM_Sens_IMUag_m1{i, :} = FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezL{i, :} .* sensTBMA_mapIMUag{i, :};
    matchTBMA_FM_Sens_IMUag_m1_num(i, :) = length(unique(matchTBMA_FM_Sens_IMUag_m1{i, :})) - 1;

    fprintf('---- FM_segTBMA_dil_IMUag_m1_fOR_acouORpiez: %d vs sensTBMA_mapIMUag: %d > match: %d ...\n', ...
            FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC(i, :), sensTBMA_mapIMUagC(i, :), matchTBMA_FM_Sens_IMUag_m1_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTBMA_FM_Sens_IMUag_m1_INV{i, :} = sensTBMA_mapIMUagL{i, :} .* FM_segTBMA_dil_IMUag_m1_fOR_acouORpiez{i, :};
    matchTBMA_FM_Sens_IMUag_m1_num_INV(i, :) = length(unique(matchTBMA_FM_Sens_IMUag_m1_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTBMA_dil_IMUag_m1_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensTBMA_mapIMUagC(i, :), FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC(i, :), matchTBMA_FM_Sens_IMUag_m1_num(i, :));
    % -----------------------------------------------------------------------------------------------------

    matchTBMA_FM_Sens_IMUag_m2{i, :} = FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezL{i, :} .* sensTBMA_mapIMUag{i, :};
    matchTBMA_FM_Sens_IMUag_m2_num(i, :) = length(unique(matchTBMA_FM_Sens_IMUag_m2{i, :})) - 1;

    fprintf('---- FM_segTBMA_dil_IMUag_m2_fOR_acouORpiez: %d vs sensTBMA_mapIMUag: %d > match: %d ...\n', ...
            FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC(i, :), sensTBMA_mapIMUagC(i, :), matchTBMA_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTBMA_FM_Sens_IMUag_m2_INV{i, :} = sensTBMA_mapIMUagL{i, :} .* FM_segTBMA_dil_IMUag_m2_fOR_acouORpiez{i, :};
    matchTBMA_FM_Sens_IMUag_m2_num_INV(i, :) = length(unique(matchTBMA_FM_Sens_IMUag_m2_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensTBMA_mapIMUag: %d vs FM_segTBMA_dil_IMUag_m2_fOR_acouORpiez: %d > match: %d ...\n\n\n', ...
            sensTBMA_mapIMUagC(i, :), FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC(i, :), matchTBMA_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

% end of loop for the files
end

% save segmented and fused data
% cd([fdir '\' data_folder]);
% save(['Seg01_butterMA_' participant '_' data_category '.mat'], ...
%       'IMUaTBMA_th', 'IMUaTBMA_map', 'IMUaTBMA_mapDil', ...
%       'IMUgTBMA_th', 'IMUgTBMA_map', 'IMUgTBMA_mapDil', ...
%       'sensT_map', 'sensT_mapL', 'sensT_mapC', ...
%       'sensTBMA_mapIMUa', 'sensTBMA_mapIMUaL', 'sensTBMA_mapIMUaC', ...
%       'sensTBMA_mapIMUg', 'sensTBMA_mapIMUgL', 'sensTBMA_mapIMUgC', ...
%       'sensTBMA_mapIMUag', 'sensTBMA_mapIMUagL', 'sensTBMA_mapIMUagC', ...
%       'FM_suiteTBMA', ...
%       'FM_TBMA_th', 'FM_segTBMA', 'FM_segTBMA_dil', ...
%       'FM_segTBMA_dil_IMUa_m1', 'FM_segTBMA_dil_IMUg_m1', 'FM_segTBMA_dil_IMUag_m1', ...
%       'FM_segTBMA_dil_IMUa_m2', 'FM_segTBMA_dil_IMUg_m2', 'FM_segTBMA_dil_IMUag_m2', ...
%       'FM_segTBMA_fOR_acou', 'FM_segTBMA_fOR_piez', 'FM_segTBMA_fOR_acouORpiez', ...
%       'FM_segTBMA_dil_fOR_acou', 'FM_segTBMA_dil_fOR_piez', 'FM_segTBMA_dil_fOR_acouORpiez', ...
%       'FM_segTBMA_dil_IMUag_m1_fOR_acou', 'FM_segTBMA_dil_IMUag_m1_fOR_piez', 'FM_segTBMA_dil_IMUag_m1_fOR_acouORpiez', ...
%       'FM_segTBMA_dil_IMUag_m2_fOR_acou', 'FM_segTBMA_dil_IMUag_m2_fOR_piez', 'FM_segTBMA_dil_IMUag_m2_fOR_acouORpiez', ...
%       'FM_segTBMA_dil_fOR_acouORpiezL', 'FM_segTBMA_dil_fOR_acouORpiezC', ...
%       'FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezL', 'FM_segTBMA_dil_IMUag_m1_fOR_acouORpiezC', ...
%       'FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezL', 'FM_segTBMA_dil_IMUag_m2_fOR_acouORpiezC', ...
%       'matchTBMA_FM_Sens', 'matchTBMA_FM_Sens_num', ...
%       'matchTBMA_FM_Sens_INV', 'matchTBMA_FM_Sens_num_INV', ...
%       'matchTBMA_FM_Sens_IMUag_m1', 'matchTBMA_FM_Sens_IMUag_m1_num', ...
%       'matchTBMA_FM_Sens_IMUag_m1_INV', 'matchTBMA_FM_Sens_IMUag_m1_num_INV', ...
%       'matchTBMA_FM_Sens_IMUag_m2', 'matchTBMA_FM_Sens_IMUag_m2_num', ...
%       'matchTBMA_FM_Sens_IMUag_m2_INV', 'matchTBMA_FM_Sens_IMUag_m2_num_INV', ...
%       '-v7.3');
% cd(curr_dir)











