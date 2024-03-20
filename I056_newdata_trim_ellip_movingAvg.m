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
Nstd_IMU_TEMA = 3;
Nstd_acou_TEMA = 1;
Nstd_piez_TEMA = 1;

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

    %% IMUaTEMA thrsholding
    % SNR: IMUaTEMA
    % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
    % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
    % SNR = 10 * log10(P_signal / P_noise)
    tmp_SNR_IMUaTEMA = 0;
    tmp_Nstd_IMU_TEMA = Nstd_IMU_TEMA;
    tmp_IMUaTEMA = abs(IMUaccNetTE_MA{i, :});

    while tmp_SNR_IMUaTEMA <= snr_IMU_th
        
        tmp_IMUaTEMA_th = mean(tmp_IMUaTEMA) + tmp_Nstd_IMU_TEMA * std(tmp_IMUaTEMA);

        tmp_IMUaTEMA_P = tmp_IMUaTEMA(tmp_IMUaTEMA >= tmp_IMUaTEMA_th);
        tmp_IMUaTEMA_N = tmp_IMUaTEMA(tmp_IMUaTEMA < tmp_IMUaTEMA_th);
    
        p_signal_IMUaTEMA = 0;
        for pa = 1 : length(tmp_IMUaTEMA_P)
            p_signal_IMUaTEMA = p_signal_IMUaTEMA + tmp_IMUaTEMA_P(pa)^2;
        end
        n_signal_IMUaTEMA = 0;
        for na = 1 : length(tmp_IMUaTEMA_N)
            n_signal_IMUaTEMA = n_signal_IMUaTEMA + tmp_IMUaTEMA_N(na)^2;
        end
    
        tmp_SNR_IMUaTEMA = 10 * log10((p_signal_IMUaTEMA / length(tmp_IMUaTEMA_P)) / (n_signal_IMUaTEMA / length(tmp_IMUaTEMA_N)));

        % increase the threshold weights if the SNR is not sufficient
        tmp_Nstd_IMU_TEMA = tmp_Nstd_IMU_TEMA + 1;
        
    end
    
    IMUaTEMA_th(i).thresh = tmp_IMUaTEMA_th;
    IMUaTEMA_th(i).Nstd = tmp_Nstd_IMU_TEMA;
    IMUaTEMA_th(i).SNR = tmp_SNR_IMUaTEMA;
    
    fprintf('---- IMUa Ellpic MA filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUaTEMA_th, tmp_Nstd_IMU_TEMA-1, tmp_SNR_IMUaTEMA);
    
    % IMUa maps
    IMUaTEMA_map{i, :} = tmp_IMUaTEMA >= IMUaTEMA_th(i).thresh;
    IMUaTEMA_mapDil{i, :} = imdilate(IMUaTEMA_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUaTEMA tmp_Nstd_IMU_TEMA tmp_IMUaTEMA tmp_IMUaTEMA_th tmp_IMUaTEMA_P tmp_IMUaTEMA_N
    % -----------------------------------------------------------------------------------------------

    %% IMUgTEMA thrsholding
    % SNR: IMUgTEMA
    tmp_SNR_IMUgTEMA = 0;
    tmp_Nstd_IMU_TEMA = Nstd_IMU_TEMA;
    tmp_IMUgTEMA = abs(IMUgyrNetTE_MA{i, :});

    while tmp_SNR_IMUgTEMA <= snr_IMU_th
        
        tmp_IMUgTEMA_th = mean(tmp_IMUgTEMA) + tmp_Nstd_IMU_TEMA * std(tmp_IMUgTEMA);

        tmp_IMUgTEMA_P = tmp_IMUgTEMA(tmp_IMUgTEMA >= tmp_IMUgTEMA_th);
        tmp_IMUgTEMA_N = tmp_IMUgTEMA(tmp_IMUgTEMA < tmp_IMUgTEMA_th);
    
        p_signal_IMUgTEMA = 0;
        for pa = 1 : length(tmp_IMUgTEMA_P)
            p_signal_IMUgTEMA = p_signal_IMUgTEMA + tmp_IMUgTEMA_P(pa)^2;
        end
        n_signal_IMUgTEMA = 0;
        for na = 1 : length(tmp_IMUgTEMA_N)
            n_signal_IMUgTEMA = n_signal_IMUgTEMA + tmp_IMUgTEMA_N(na)^2;
        end
    
        tmp_SNR_IMUgTEMA = 10 * log10((p_signal_IMUgTEMA / length(tmp_IMUgTEMA_P)) / (n_signal_IMUgTEMA / length(tmp_IMUgTEMA_N)));

        % increase the threshold weights if the SNR is not sufficient
        tmp_Nstd_IMU_TEMA = tmp_Nstd_IMU_TEMA + 1;
        
    end
    
    IMUgTEMA_th(i).thresh = tmp_IMUgTEMA_th;
    IMUgTEMA_th(i).Nstd = tmp_Nstd_IMU_TEMA;
    IMUgTEMA_th(i).SNR = tmp_SNR_IMUgTEMA;
    
    fprintf('---- IMUg Ellpic MA filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUgTEMA_th, tmp_Nstd_IMU_TEMA-1, tmp_SNR_IMUgTEMA);
    
     % IMUg maps
    IMUgTEMA_map{i, :} = tmp_IMUgTEMA >= IMUgTEMA_th(i).thresh;
    IMUgTEMA_mapDil{i, :} = imdilate(IMUgTEMA_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUgTEMA tmp_Nstd_IMU_TEMA tmp_IMUgTEMA tmp_IMUgTEMA_th tmp_IMUgTEMA_P tmp_IMUgTEMA_N
    % -----------------------------------------------------------------------------------------------

    %% Sensation
    % Initializing the map: sensT
    % senstation labels by connected components
    tmp_sensT_MA = sensT_MA_B{i, :};
    [tmp_sensT_MA_L, tmp_sensT_MA_C] = bwlabel(tmp_sensT_MA);

    % initialization of sensation maps
    tmp_sensT_MA_map = zeros(length(tmp_sensT_MA), 1);     
    tmp_sensTEMA_mapIMUa = zeros(length(tmp_sensT_MA), 1); 
    tmp_sensTEMA_mapIMUg = zeros(length(tmp_sensT_MA), 1); 
    tmp_sensTEMA_mapIMUag = zeros(length(tmp_sensT_MA), 1); 
    
    for j1 = 1 : tmp_sensT_MA_C

        % the idx range of the current cluster (component)
        tmp_idx = find(tmp_sensT_MA_L == j1);
        
        tmp_idxS = min(tmp_idx) - sens_dilation_sizeB; 
        tmp_idxE = max(tmp_idx) + sens_dilation_sizeF; 

        tmp_idxS = max(tmp_idxS, 1);
        tmp_idxE = min(tmp_idxE, length(tmp_sensT_MA_map));
        
        % sensation map
        tmp_sensT_MA_map(tmp_idxS:tmp_idxE) = 1; 

        tmp_sensTEMA_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTEMA_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTEMA_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        % Remove maternal sensation coincided with body movement (IMU a/g) - ellpic MA
        tmp_xIMUaTEMA = sum(tmp_sensT_MA_map(tmp_idxS:tmp_idxE) .* IMUaTEMA_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMUgTEMA = sum(tmp_sensT_MA_map(tmp_idxS:tmp_idxE) .* IMUgTEMA_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTEMA >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
            tmp_sensTEMA_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTEMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMUgTEMA >= overlap_percSens*(tmp_idxE-tmp_idxS+1))
            tmp_sensTEMA_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTEMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTEMA tmp_xIMUgTEMA
    end

    % sensation maps
    sensT_map{i, :} = tmp_sensT_MA_map;
    sensTEMA_mapIMUa{i, :} = tmp_sensTEMA_mapIMUa;
    sensTEMA_mapIMUg{i, :} = tmp_sensTEMA_mapIMUg;
    sensTEMA_mapIMUag{i, :} = tmp_sensTEMA_mapIMUag;
    
    [tmp_sensT_mapL, tmp_sensT_mapC] = bwlabel(tmp_sensT_MA_map);
    [tmp_sensTEMA_mapIMUaL, tmp_sensTEMA_mapIMUaC] = bwlabel(tmp_sensTEMA_mapIMUa);
    [tmp_sensTEMA_mapIMUgL, tmp_sensTEMA_mapIMUgC] = bwlabel(tmp_sensTEMA_mapIMUg);
    [tmp_sensTEMA_mapIMUagL, tmp_sensTEMA_mapIMUagC] = bwlabel(tmp_sensTEMA_mapIMUag);

    sensT_mapL{i, :} = tmp_sensT_mapL; 
    sensT_mapC(i, :) = tmp_sensT_mapC;
    sensTEMA_mapIMUaL{i, :} = tmp_sensTEMA_mapIMUaL;
    sensTEMA_mapIMUaC(i, :) = tmp_sensTEMA_mapIMUaC;
    sensTEMA_mapIMUgL{i, :} = tmp_sensTEMA_mapIMUgL;
    sensTEMA_mapIMUgC(i, :) = tmp_sensTEMA_mapIMUgC;
    sensTEMA_mapIMUagL{i, :} = tmp_sensTEMA_mapIMUagL;
    sensTEMA_mapIMUagC(i, :) = tmp_sensTEMA_mapIMUagC;

    fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d - %d - %d ... \n', ...
            sensT_mapC(i, :), sensTEMA_mapIMUaC(i, :), sensTEMA_mapIMUgC(i, :), sensTEMA_mapIMUagC(i, :));

    clear tmp_sensT_MA tmp_sensT_MA_L tmp_sensT_MA_C tmp_sensT_MA_map ...
          tmp_sensTEMA_mapIMUa tmp_sensTEMA_mapIMUg tmp_sensTEMA_mapIMUag ...
          tmp_sensT_mapL tmp_sensT_mapC ...
          tmp_sensTEMA_mapIMUaL tmp_sensTEMA_mapIMUaC ...
          tmp_sensTEMA_mapIMUgL tmp_sensTEMA_mapIMUgC ...
          tmp_sensTEMA_mapIMUagL tmp_sensTEMA_mapIMUagC
    % --------------------------------------------------------------------------------

    %% Segmenting FM sensors
    % 1. apply for an adaptive threshold
    % 2. remove body movement
    % 3. dilate the fetal movements at desinated duration
    FM_suiteTEMA{i, 1} = acouLTE_MA{i,:};
    FM_suiteTEMA{i, 2} = acouMTE_MA{i,:};
    FM_suiteTEMA{i, 3} = acouRTE_MA{i,:};
    FM_suiteTEMA{i, 4} = piezLTE_MA{i,:};
    FM_suiteTEMA{i, 5} = piezMTE_MA{i,:};
    FM_suiteTEMA{i, 6} = piezRTE_MA{i,:};

    % FM Sensors Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(FM_suiteTEMA{i, j2});
        tmp_SNR_sensor = 0;

        if j2 <= num_FMsensors / 2
            tmp_std_times_FM = Nstd_acou_TEMA;
            tmp_snr_FM = snr_acou_th;
        else
            tmp_std_times_FM = Nstd_piez_TEMA;
            tmp_snr_FM = snr_piez_th;
        end

        while tmp_SNR_sensor <= tmp_snr_FM

            tmp_FM_threshTEMA = mean(tmp_sensor_indiv) + tmp_std_times_FM * std(tmp_sensor_indiv);
            if isnan(tmp_FM_threshTEMA)
                tmp_FM_threshTEMA = 0;
            end

            tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < tmp_FM_threshTEMA);
            tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= tmp_FM_threshTEMA);
    
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

        FM_TEMA_th(i, j2).thresh = tmp_FM_threshTEMA;
        FM_TEMA_th(i, j2).Nstd = tmp_std_times_FM;
        FM_TEMA_th(i, j2).SNE = tmp_SNR_sensor;

        fprintf('---- FM(%d-th sensor) Ellpic MA filtering: threshold > %d; SNR (%d*std): %d ... \n', ...
                j2, tmp_FM_threshTEMA, tmp_std_times_FM-1, tmp_SNR_sensor);

        clear tmp_SNR_sensor tmp_std_times_FM tmp_snr_FM ...
              tmp_FM_removed tmp_FM_segmented tmp_FM_threshTEMA
        % --------------------------------------------------------------------------------------------------------------

        %% segmentation by thresholding
        FM_segTEMA{i, j2} = (tmp_sensor_indiv >= FM_TEMA_th(i, j2).thresh);
        FM_segTEMA_dil{i, j2} = double(imdilate(FM_segTEMA{i, j2}, FM_lse)); 

        % labels by connected components
        [tmp_LTEMA, tmp_CTEMA] = bwlabel(FM_segTEMA{i, j2});
        [tmp_LTEMA_dil, tmp_CTEMA_dil] = bwlabel(FM_segTEMA_dil{i, j2});

        % Method 1 - denoise segmentations by removing movements
        FM_segTEMA_dil_IMUa_m1{i, j2} = FM_segTEMA_dil{i, j2} .* (1-IMUaTEMA_mapDil{i,:}); 
        FM_segTEMA_dil_IMUg_m1{i, j2} = FM_segTEMA_dil{i, j2} .* (1-IMUgTEMA_mapDil{i,:}); 
        FM_segTEMA_dil_IMUag_m1{i, j2} = FM_segTEMA_dil{i, j2} .* (1-IMUaTEMA_mapDil{i,:}) .* (1-IMUgTEMA_mapDil{i,:}); 
        
        % labels by connected components
        [tmp_LTEMA_IMUa_m1, tmp_CTEMA_IMUa_m1] = bwlabel(FM_segTEMA_dil_IMUa_m1{i, j2});
        [tmp_LTEMA_IMUg_m1, tmp_CTEMA_IMUg_m1] = bwlabel(FM_segTEMA_dil_IMUg_m1{i, j2});
        [tmp_LTEMA_IMUag_m1, tmp_CTEMA_IMUag_m1] = bwlabel(FM_segTEMA_dil_IMUag_m1{i, j2});
        % ------------------------------------------------------------------------------------------------------

        % Method 2 - further segmentations by removing movements
        tmp_segTEMA_dil_IMUa_m2 = zeros(length(FM_segTEMA_dil{i, j2}), 1); 
        tmp_segTEMA_dil_IMUg_m2 = zeros(length(FM_segTEMA_dil{i, j2}), 1);
        tmp_segTEMA_dil_IMUag_m2 = zeros(length(FM_segTEMA_dil{i, j2}), 1);
        
        for c = 1: tmp_CTEMA_dil
            
            tmp_idx = find(tmp_LTEMA_dil == c);
            
            tmp_segTEMA_dil_IMUa_m2(tmp_idx, :) = 1;
            tmp_segTEMA_dil_IMUg_m2(tmp_idx, :) = 1;
            tmp_segTEMA_dil_IMUag_m2(tmp_idx, :) = 1;

            tmp_xIMUaTEMA = sum(FM_segTEMA_dil{i, j2}(tmp_idx) .* IMUaTEMA_mapDil{i, :}(tmp_idx));
            tmp_xIMUgTEMA = sum(FM_segTEMA_dil{i, j2}(tmp_idx) .* IMUgTEMA_mapDil{i, :}(tmp_idx));

            if tmp_xIMUaTEMA >= overlap_percFM*length(tmp_idx)
                tmp_segTEMA_dil_IMUa_m2(tmp_idx) = 0;
                tmp_segTEMA_dil_IMUag_m2(tmp_idx) = 0;
            end
            
            if tmp_xIMUgTEMA >= overlap_percFM*length(tmp_idx)
                tmp_segTEMA_dil_IMUg_m2(tmp_idx) = 0;
                tmp_segTEMA_dil_IMUag_m2(tmp_idx) = 0;
            end

            clear tmp_idx tmp_xIMUaTEMA tmp_xIMUgTEMA
        end

        [tmp_LTEMA_IMUa_m2, tmp_CTEMA_IMUa_m2] = bwlabel(tmp_segTEMA_dil_IMUa_m2);
        [tmp_LTEMA_IMUg_m2, tmp_CTEMA_IMUg_m2] = bwlabel(tmp_segTEMA_dil_IMUg_m2);
        [tmp_LTEMA_IMUag_m2, tmp_CTEMA_IMUag_m2] = bwlabel(tmp_segTEMA_dil_IMUag_m2);

        FM_segTEMA_dil_IMUa_m2{i, j2} = tmp_segTEMA_dil_IMUa_m2;
        FM_segTEMA_dil_IMUg_m2{i, j2} = tmp_segTEMA_dil_IMUg_m2;
        FM_segTEMA_dil_IMUag_m2{i, j2} = tmp_segTEMA_dil_IMUag_m2;
        
        % display instant performance
        fprintf('---- FM(%d-th sensor) Ellpic MA filtering > Denoising: method 1 vs method 2: IMUa > %d - %d; IMUg > %d - %d; IMUag > %d - %d ...\n', ...
                j2, tmp_CTEMA_IMUa_m1, tmp_CTEMA_IMUa_m2, tmp_CTEMA_IMUg_m1, tmp_CTEMA_IMUg_m2, tmp_CTEMA_IMUag_m1, tmp_CTEMA_IMUag_m2);
        % -------------------------------------------------------------------------------------------------------

        clear tmp_sensor_indiv  ...
              tmp_LTEMA tmp_CTEMA tmp_LTEMA_dil tmp_CTEMA_dil ...
              tmp_LTEMA_IMUa_m1 tmp_CTEMA_IMUa_m1 tmp_LTEMA_IMUg_m1 tmp_CTEMA_IMUg_m1 tmp_LTEMA_IMUag_m1 tmp_CTEMA_IMUag_m1 ...
              tmp_segTEMA_dil_IMUa_m2 tmp_segTEMA_dil_IMUg_m2 tmp_segTEMA_dil_IMUag_m2 ...
              tmp_LTEMA_IMUa_m2 tmp_CTEMA_IMUa_m2 tmp_LTEMA_IMUg_m2 tmp_CTEMA_IMUg_m2 tmp_LTEMA_IMUag_m2 tmp_CTEMA_IMUag_m2
    end
    % -------------------------------------------------------------------------
    
    %% Sensor fusion
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'OR'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    FM_segTEMA_fOR_acou{i, :} = double(FM_segTEMA{i, 1} | FM_segTEMA{i, 2} | FM_segTEMA{i, 3});
    FM_segTEMA_fOR_piez{i, :} = double(FM_segTEMA{i, 4} | FM_segTEMA{i, 5} | FM_segTEMA{i, 6});
    FM_segTEMA_fOR_acouORpiez{i, :}  = double(FM_segTEMA_fOR_acou{i,:} | FM_segTEMA_fOR_piez{i, :});

    FM_segTEMA_dil_fOR_acou{i, :} = double(FM_segTEMA{i, 1} | FM_segTEMA{i, 2} | FM_segTEMA{i, 3});
    FM_segTEMA_dil_fOR_piez{i, :} = double(FM_segTEMA{i, 4} | FM_segTEMA{i, 5} | FM_segTEMA{i, 6});
    FM_segTEMA_dil_fOR_acouORpiez{i, :} = double(FM_segTEMA_dil_fOR_acou{i,:} | FM_segTEMA_dil_fOR_piez{i, :});
    
    FM_segTEMA_dil_IMUag_m1_fOR_acou{i, :} = double(FM_segTEMA_dil_IMUag_m1{i, 1} | FM_segTEMA_dil_IMUag_m1{i, 2} | FM_segTEMA_dil_IMUag_m1{i, 3});
    FM_segTEMA_dil_IMUag_m1_fOR_piez{i, :} = double(FM_segTEMA_dil_IMUag_m1{i, 4} | FM_segTEMA_dil_IMUag_m1{i, 5} | FM_segTEMA_dil_IMUag_m1{i, 6});
    FM_segTEMA_dil_IMUag_m1_fOR_acouORpiez{i, :}  = double(FM_segTEMA_dil_IMUag_m1_fOR_acou{i,:} | FM_segTEMA_dil_IMUag_m1_fOR_piez{i, :});

    FM_segTEMA_dil_IMUag_m2_fOR_acou{i, :} = double(FM_segTEMA_dil_IMUag_m2{i, 1} | FM_segTEMA_dil_IMUag_m2{i, 2} | FM_segTEMA_dil_IMUag_m2{i, 3});
    FM_segTEMA_dil_IMUag_m2_fOR_piez{i, :} = double(FM_segTEMA_dil_IMUag_m2{i, 4} | FM_segTEMA_dil_IMUag_m2{i, 5} | FM_segTEMA_dil_IMUag_m2{i, 6});
    FM_segTEMA_dil_IMUag_m2_fOR_acouORpiez{i, :}  = double(FM_segTEMA_dil_IMUag_m2_fOR_acou{i,:} | FM_segTEMA_dil_IMUag_m2_fOR_piez{i, :});

    [tmp_FM_segTEMA_dil_fOR_acouORpiezL, tmp_FM_segTEMA_dil_fOR_acouORpiezC] = bwlabel(FM_segTEMA_dil_fOR_acouORpiez{i, :});
    [tmp_FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezL, tmp_FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC] = bwlabel(FM_segTEMA_dil_IMUag_m1_fOR_acouORpiez{i, :});
    [tmp_FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezL, tmp_FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC] = bwlabel(FM_segTEMA_dil_IMUag_m2_fOR_acouORpiez{i, :});

    FM_segTEMA_dil_fOR_acouORpiezL{i, :} = tmp_FM_segTEMA_dil_fOR_acouORpiezL; 
    FM_segTEMA_dil_fOR_acouORpiezC(i, :) = tmp_FM_segTEMA_dil_fOR_acouORpiezC;
    FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezL{i, :} = tmp_FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezL; 
    FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC(i, :) = tmp_FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC;
    FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezL{i, :} = tmp_FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezL; 
    FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC(i, :) = tmp_FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC;

    clear tmp_FM_segTEMA_dil_fOR_acouORpiezL tmp_FM_segTEMA_dil_fOR_acouORpiezC ...
          tmp_FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezL tmp_FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC ...
          tmp_FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezL tmp_FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC
    % ---------------------------------------------------------------------------------------------------------------------------------

    %% match of FM and sensation
    matchTEMA_FM_Sens{i, :} = FM_segTEMA_dil_fOR_acouORpiezL{i, :} .* sensT_map{i, :};
    matchTEMA_FM_Sens_num(i, :) = length(unique(matchTEMA_FM_Sens{i, :})) - 1;

    fprintf('---- FM_segTEMA_dil_fOR_acouORpiez: %d vs sensT_map: %d > match: %d ...\n', ...
            FM_segTEMA_dil_fOR_acouORpiezC(i, :), sensT_mapC(i, :), matchTEMA_FM_Sens_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTEMA_FM_Sens_INV{i, :} = sensT_mapL{i, :} .* FM_segTEMA_dil_fOR_acouORpiez{i, :};
    matchTEMA_FM_Sens_num_INV(i, :) = length(unique(matchTEMA_FM_Sens_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTEMA_dil_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensT_mapC(i, :), FM_segTEMA_dil_fOR_acouORpiezC(i, :), matchTEMA_FM_Sens_num_INV(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTEMA_FM_Sens_IMUag_m1{i, :} = FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezL{i, :} .* sensTEMA_mapIMUag{i, :};
    matchTEMA_FM_Sens_IMUag_m1_num(i, :) = length(unique(matchTEMA_FM_Sens_IMUag_m1{i, :})) - 1;

    fprintf('---- FM_segTEMA_dil_IMUag_m1_fOR_acouORpiez: %d vs sensTEMA_mapIMUag: %d > match: %d ...\n', ...
            FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC(i, :), sensTEMA_mapIMUagC(i, :), matchTEMA_FM_Sens_IMUag_m1_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTEMA_FM_Sens_IMUag_m1_INV{i, :} = sensTEMA_mapIMUagL{i, :} .* FM_segTEMA_dil_IMUag_m1_fOR_acouORpiez{i, :};
    matchTEMA_FM_Sens_IMUag_m1_num_INV(i, :) = length(unique(matchTEMA_FM_Sens_IMUag_m1_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTEMA_dil_IMUag_m1_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensTEMA_mapIMUagC(i, :), FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC(i, :), matchTEMA_FM_Sens_IMUag_m1_num(i, :));
    % -----------------------------------------------------------------------------------------------------

    matchTEMA_FM_Sens_IMUag_m2{i, :} = FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezL{i, :} .* sensTEMA_mapIMUag{i, :};
    matchTEMA_FM_Sens_IMUag_m2_num(i, :) = length(unique(matchTEMA_FM_Sens_IMUag_m2{i, :})) - 1;

    fprintf('---- FM_segTEMA_dil_IMUag_m2_fOR_acouORpiez: %d vs sensTEMA_mapIMUag: %d > match: %d ...\n', ...
            FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC(i, :), sensTEMA_mapIMUagC(i, :), matchTEMA_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTEMA_FM_Sens_IMUag_m2_INV{i, :} = sensTEMA_mapIMUagL{i, :} .* FM_segTEMA_dil_IMUag_m2_fOR_acouORpiez{i, :};
    matchTEMA_FM_Sens_IMUag_m2_num_INV(i, :) = length(unique(matchTEMA_FM_Sens_IMUag_m2_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensTEMA_mapIMUag: %d vs FM_segTEMA_dil_IMUag_m2_fOR_acouORpiez: %d > match: %d ...\n\n\n', ...
            sensTEMA_mapIMUagC(i, :), FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC(i, :), matchTEMA_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

% end of loop for the files
end

% save segmented and fused data
% cd([fdir '\' data_folder]);
% save(['Seg01_ellpicMA_' participant '_' data_category '.mat'], ...
%       'IMUaTEMA_th', 'IMUaTEMA_map', 'IMUaTEMA_mapDil', ...
%       'IMUgTEMA_th', 'IMUgTEMA_map', 'IMUgTEMA_mapDil', ...
%       'sensT_map', 'sensT_mapL', 'sensT_mapC', ...
%       'sensTEMA_mapIMUa', 'sensTEMA_mapIMUaL', 'sensTEMA_mapIMUaC', ...
%       'sensTEMA_mapIMUg', 'sensTEMA_mapIMUgL', 'sensTEMA_mapIMUgC', ...
%       'sensTEMA_mapIMUag', 'sensTEMA_mapIMUagL', 'sensTEMA_mapIMUagC', ...
%       'FM_suiteTEMA', ...
%       'FM_TEMA_th', 'FM_segTEMA', 'FM_segTEMA_dil', ...
%       'FM_segTEMA_dil_IMUa_m1', 'FM_segTEMA_dil_IMUg_m1', 'FM_segTEMA_dil_IMUag_m1', ...
%       'FM_segTEMA_dil_IMUa_m2', 'FM_segTEMA_dil_IMUg_m2', 'FM_segTEMA_dil_IMUag_m2', ...
%       'FM_segTEMA_fOR_acou', 'FM_segTEMA_fOR_piez', 'FM_segTEMA_fOR_acouORpiez', ...
%       'FM_segTEMA_dil_fOR_acou', 'FM_segTEMA_dil_fOR_piez', 'FM_segTEMA_dil_fOR_acouORpiez', ...
%       'FM_segTEMA_dil_IMUag_m1_fOR_acou', 'FM_segTEMA_dil_IMUag_m1_fOR_piez', 'FM_segTEMA_dil_IMUag_m1_fOR_acouORpiez', ...
%       'FM_segTEMA_dil_IMUag_m2_fOR_acou', 'FM_segTEMA_dil_IMUag_m2_fOR_piez', 'FM_segTEMA_dil_IMUag_m2_fOR_acouORpiez', ...
%       'FM_segTEMA_dil_fOR_acouORpiezL', 'FM_segTEMA_dil_fOR_acouORpiezC', ...
%       'FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezL', 'FM_segTEMA_dil_IMUag_m1_fOR_acouORpiezC', ...
%       'FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezL', 'FM_segTEMA_dil_IMUag_m2_fOR_acouORpiezC', ...
%       'matchTEMA_FM_Sens', 'matchTEMA_FM_Sens_num', ...
%       'matchTEMA_FM_Sens_INV', 'matchTEMA_FM_Sens_num_INV', ...
%       'matchTEMA_FM_Sens_IMUag_m1', 'matchTEMA_FM_Sens_IMUag_m1_num', ...
%       'matchTEMA_FM_Sens_IMUag_m1_INV', 'matchTEMA_FM_Sens_IMUag_m1_num_INV', ...
%       'matchTEMA_FM_Sens_IMUag_m2', 'matchTEMA_FM_Sens_IMUag_m2_num', ...
%       'matchTEMA_FM_Sens_IMUag_m2_INV', 'matchTEMA_FM_Sens_IMUag_m2_num_INV', ...
%       '-v7.3');
% cd(curr_dir)











