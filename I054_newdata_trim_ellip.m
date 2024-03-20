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
Nstd_IMU_TE = 1;
Nstd_acou_TE = 1;
Nstd_piez_TE = 1;

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

    %% IMUaTE thrsholding
    % SNR: IMUaTE
    % P_signal = (1 / N) * Σ(signal[i]^2), where i = 1 to N
    % P_noise = (1 / M) * Σ(noise[i]^2), where i = 1 to M
    % SNR = 10 * log10(P_signal / P_noise)
    tmp_SNR_IMUaTE = 0;
    tmp_Nstd_IMU_TE = Nstd_IMU_TE;
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
    
    IMUaTE_th(i).thresh = tmp_IMUaTE_th;
    IMUaTE_th(i).Nstd = tmp_Nstd_IMU_TE;
    IMUaTE_th(i).SNR = tmp_SNR_IMUaTE;
    
    fprintf('---- IMUa Ellpicworth filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUaTE_th, tmp_Nstd_IMU_TE-1, tmp_SNR_IMUaTE);
    
    % IMUa maps
    IMUaTE_map{i, :} = tmp_IMUaTE >= IMUaTE_th(i).thresh;
    IMUaTE_mapDil{i, :} = imdilate(IMUaTE_map{i, :}, IMU_lse);

    clear tmp_SNR_IMUaTE tmp_Nstd_IMU_TE tmp_IMUaTE tmp_IMUaTE_th tmp_IMUaTE_P tmp_IMUaTE_N
    % -----------------------------------------------------------------------------------------------

    %% IMUgTE thrsholding
    % SNR: IMUgTE
    tmp_SNR_IMUgTE = 0;
    tmp_Nstd_IMU_TE = Nstd_IMU_TE;
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
    
    IMUgTE_th(i).thresh = tmp_IMUgTE_th;
    IMUgTE_th(i).Nstd = tmp_Nstd_IMU_TE;
    IMUgTE_th(i).SNR = tmp_SNR_IMUgTE;
    
    fprintf('---- IMUg Ellpicworth filtering: threshold > %d; SNR (%d*std): %d ... \n', tmp_IMUgTE_th, tmp_Nstd_IMU_TE-1, tmp_SNR_IMUgTE);
    
     % IMUg maps
    IMUgTE_map{i, :} = tmp_IMUgTE >= IMUgTE_th(i).thresh;
    IMUgTE_mapDil{i, :} = imdilate(IMUgTE_map{i, :}, IMU_lse);

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

        % Remove maternal sensation coincided with body movement (IMU a/g) - ellpic
        tmp_xIMUaTE = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUaTE_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMUgTE = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUgTE_mapDil{i, :}(tmp_idxS:tmp_idxE));

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
    sensT_map{i, :} = tmp_sensT_map;
    sensTE_mapIMUa{i, :} = tmp_sensTE_mapIMUa;
    sensTE_mapIMUg{i, :} = tmp_sensTE_mapIMUg;
    sensTE_mapIMUag{i, :} = tmp_sensTE_mapIMUag;
    
    [tmp_sensT_mapL, tmp_sensT_mapC] = bwlabel(tmp_sensT_map);
    [tmp_sensTE_mapIMUaL, tmp_sensTE_mapIMUaC] = bwlabel(tmp_sensTE_mapIMUa);
    [tmp_sensTE_mapIMUgL, tmp_sensTE_mapIMUgC] = bwlabel(tmp_sensTE_mapIMUg);
    [tmp_sensTE_mapIMUagL, tmp_sensTE_mapIMUagC] = bwlabel(tmp_sensTE_mapIMUag);

    sensT_mapL{i, :} = tmp_sensT_mapL; 
    sensT_mapC(i, :) = tmp_sensT_mapC;
    sensTE_mapIMUaL{i, :} = tmp_sensTE_mapIMUaL;
    sensTE_mapIMUaC(i, :) = tmp_sensTE_mapIMUaC;
    sensTE_mapIMUgL{i, :} = tmp_sensTE_mapIMUgL;
    sensTE_mapIMUgC(i, :) = tmp_sensTE_mapIMUgC;
    sensTE_mapIMUagL{i, :} = tmp_sensTE_mapIMUagL;
    sensTE_mapIMUagC(i, :) = tmp_sensTE_mapIMUagC;

    fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d - %d - %d ... \n', ...
            sensT_mapC(i, :), sensTE_mapIMUaC(i, :), sensTE_mapIMUgC(i, :), sensTE_mapIMUagC(i, :));

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

    % FM Sensors Thresholding
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(FM_suiteTE{i, j2});
        tmp_SNR_sensor = 0;

        if j2 <= num_FMsensors / 2
            tmp_std_times_FM = Nstd_acou_TE;
            tmp_snr_FM = snr_acou_th;
        else
            tmp_std_times_FM = Nstd_piez_TE;
            tmp_snr_FM = snr_piez_th;
        end

        while tmp_SNR_sensor <= tmp_snr_FM

            tmp_FM_threshTE = mean(tmp_sensor_indiv) + tmp_std_times_FM * std(tmp_sensor_indiv);
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

        FM_TE_th(i, j2).thresh = tmp_FM_threshTE;
        FM_TE_th(i, j2).Nstd = tmp_std_times_FM;
        FM_TE_th(i, j2).SNE = tmp_SNR_sensor;

        fprintf('---- FM(%d-th sensor) Ellpic filtering: threshold > %d; SNR (%d*std): %d ... \n', ...
                j2, tmp_FM_threshTE, tmp_std_times_FM-1, tmp_SNR_sensor);

        clear tmp_SNR_sensor tmp_std_times_FM tmp_snr_FM ...
              tmp_FM_removed tmp_FM_segmented tmp_FM_threshTE
        % --------------------------------------------------------------------------------------------------------------

        %% segmentation by thresholding
        FM_segTE{i, j2} = (tmp_sensor_indiv >= FM_TE_th(i, j2).thresh);
        FM_segTE_dil{i, j2} = double(imdilate(FM_segTE{i, j2}, FM_lse)); 

        % labels by connected components
        [tmp_LTE, tmp_CTE] = bwlabel(FM_segTE{i, j2});
        [tmp_LTE_dil, tmp_CTE_dil] = bwlabel(FM_segTE_dil{i, j2});

        % Method 1 - denoise segmentations by removing movements
        FM_segTE_dil_IMUa_m1{i, j2} = FM_segTE_dil{i, j2} .* (1-IMUaTE_mapDil{i,:}); 
        FM_segTE_dil_IMUg_m1{i, j2} = FM_segTE_dil{i, j2} .* (1-IMUgTE_mapDil{i,:}); 
        FM_segTE_dil_IMUag_m1{i, j2} = FM_segTE_dil{i, j2} .* (1-IMUaTE_mapDil{i,:}) .* (1-IMUgTE_mapDil{i,:}); 
        
        % labels by connected components
        [tmp_LTE_IMUa_m1, tmp_CTE_IMUa_m1] = bwlabel(FM_segTE_dil_IMUa_m1{i, j2});
        [tmp_LTE_IMUg_m1, tmp_CTE_IMUg_m1] = bwlabel(FM_segTE_dil_IMUg_m1{i, j2});
        [tmp_LTE_IMUag_m1, tmp_CTE_IMUag_m1] = bwlabel(FM_segTE_dil_IMUag_m1{i, j2});
        % ------------------------------------------------------------------------------------------------------

        % Method 2 - further segmentations by removing movements
        tmp_segTE_dil_IMUa_m2 = zeros(length(FM_segTE_dil{i, j2}), 1); 
        tmp_segTE_dil_IMUg_m2 = zeros(length(FM_segTE_dil{i, j2}), 1);
        tmp_segTE_dil_IMUag_m2 = zeros(length(FM_segTE_dil{i, j2}), 1);
        
        for c = 1: tmp_CTE_dil
            
            tmp_idx = find(tmp_LTE_dil == c);
            
            tmp_segTE_dil_IMUa_m2(tmp_idx, :) = 1;
            tmp_segTE_dil_IMUg_m2(tmp_idx, :) = 1;
            tmp_segTE_dil_IMUag_m2(tmp_idx, :) = 1;

            tmp_xIMUaTE = sum(FM_segTE_dil{i, j2}(tmp_idx) .* IMUaTE_mapDil{i, :}(tmp_idx));
            tmp_xIMUgTE = sum(FM_segTE_dil{i, j2}(tmp_idx) .* IMUgTE_mapDil{i, :}(tmp_idx));

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

        FM_segTE_dil_IMUa_m2{i, j2} = tmp_segTE_dil_IMUa_m2;
        FM_segTE_dil_IMUg_m2{i, j2} = tmp_segTE_dil_IMUg_m2;
        FM_segTE_dil_IMUag_m2{i, j2} = tmp_segTE_dil_IMUag_m2;
        
        % display instant performance
        fprintf('---- FM(%d-th sensor) Ellpic filtering > Denoising: method 1 vs method 2: IMUa > %d - %d; IMUg > %d - %d; IMUag > %d - %d ...\n', ...
                j2, tmp_CTE_IMUa_m1, tmp_CTE_IMUa_m2, tmp_CTE_IMUg_m1, tmp_CTE_IMUg_m2, tmp_CTE_IMUag_m1, tmp_CTE_IMUag_m2);
        % -------------------------------------------------------------------------------------------------------

        clear tmp_sensor_indiv  ...
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
    FM_segTE_fOR_acou{i, :} = double(FM_segTE{i, 1} | FM_segTE{i, 2} | FM_segTE{i, 3});
    FM_segTE_fOR_piez{i, :} = double(FM_segTE{i, 4} | FM_segTE{i, 5} | FM_segTE{i, 6});
    FM_segTE_fOR_acouORpiez{i, :}  = double(FM_segTE_fOR_acou{i,:} | FM_segTE_fOR_piez{i, :});

    FM_segTE_dil_fOR_acou{i, :} = double(FM_segTE{i, 1} | FM_segTE{i, 2} | FM_segTE{i, 3});
    FM_segTE_dil_fOR_piez{i, :} = double(FM_segTE{i, 4} | FM_segTE{i, 5} | FM_segTE{i, 6});
    FM_segTE_dil_fOR_acouORpiez{i, :} = double(FM_segTE_dil_fOR_acou{i,:} | FM_segTE_dil_fOR_piez{i, :});
    
    FM_segTE_dil_IMUag_m1_fOR_acou{i, :} = double(FM_segTE_dil_IMUag_m1{i, 1} | FM_segTE_dil_IMUag_m1{i, 2} | FM_segTE_dil_IMUag_m1{i, 3});
    FM_segTE_dil_IMUag_m1_fOR_piez{i, :} = double(FM_segTE_dil_IMUag_m1{i, 4} | FM_segTE_dil_IMUag_m1{i, 5} | FM_segTE_dil_IMUag_m1{i, 6});
    FM_segTE_dil_IMUag_m1_fOR_acouORpiez{i, :}  = double(FM_segTE_dil_IMUag_m1_fOR_acou{i,:} | FM_segTE_dil_IMUag_m1_fOR_piez{i, :});

    FM_segTE_dil_IMUag_m2_fOR_acou{i, :} = double(FM_segTE_dil_IMUag_m2{i, 1} | FM_segTE_dil_IMUag_m2{i, 2} | FM_segTE_dil_IMUag_m2{i, 3});
    FM_segTE_dil_IMUag_m2_fOR_piez{i, :} = double(FM_segTE_dil_IMUag_m2{i, 4} | FM_segTE_dil_IMUag_m2{i, 5} | FM_segTE_dil_IMUag_m2{i, 6});
    FM_segTE_dil_IMUag_m2_fOR_acouORpiez{i, :}  = double(FM_segTE_dil_IMUag_m2_fOR_acou{i,:} | FM_segTE_dil_IMUag_m2_fOR_piez{i, :});

    [tmp_FM_segTE_dil_fOR_acouORpiezL, tmp_FM_segTE_dil_fOR_acouORpiezC] = bwlabel(FM_segTE_dil_fOR_acouORpiez{i, :});
    [tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezL, tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezC] = bwlabel(FM_segTE_dil_IMUag_m1_fOR_acouORpiez{i, :});
    [tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezL, tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezC] = bwlabel(FM_segTE_dil_IMUag_m2_fOR_acouORpiez{i, :});

    FM_segTE_dil_fOR_acouORpiezL{i, :} = tmp_FM_segTE_dil_fOR_acouORpiezL; 
    FM_segTE_dil_fOR_acouORpiezC(i, :) = tmp_FM_segTE_dil_fOR_acouORpiezC;
    FM_segTE_dil_IMUag_m1_fOR_acouORpiezL{i, :} = tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezL; 
    FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, :) = tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezC;
    FM_segTE_dil_IMUag_m2_fOR_acouORpiezL{i, :} = tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezL; 
    FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, :) = tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezC;

    clear tmp_FM_segTE_dil_fOR_acouORpiezL tmp_FM_segTE_dil_fOR_acouORpiezC ...
          tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezL tmp_FM_segTE_dil_IMUag_m1_fOR_acouORpiezC ...
          tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezL tmp_FM_segTE_dil_IMUag_m2_fOR_acouORpiezC
    % ---------------------------------------------------------------------------------------------------------------------------------

    %% match of FM and sensation
    matchTE_FM_Sens{i, :} = FM_segTE_dil_fOR_acouORpiezL{i, :} .* sensT_map{i, :};
    matchTE_FM_Sens_num(i, :) = length(unique(matchTE_FM_Sens{i, :})) - 1;

    fprintf('---- FM_segTE_dil_fOR_acouORpiez: %d vs sensT_map: %d > match: %d ...\n', ...
            FM_segTE_dil_fOR_acouORpiezC(i, :), sensT_mapC(i, :), matchTE_FM_Sens_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTE_FM_Sens_INV{i, :} = sensT_mapL{i, :} .* FM_segTE_dil_fOR_acouORpiez{i, :};
    matchTE_FM_Sens_num_INV(i, :) = length(unique(matchTE_FM_Sens_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTE_dil_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensT_mapC(i, :), FM_segTE_dil_fOR_acouORpiezC(i, :), matchTE_FM_Sens_num_INV(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTE_FM_Sens_IMUag_m1{i, :} = FM_segTE_dil_IMUag_m1_fOR_acouORpiezL{i, :} .* sensTE_mapIMUag{i, :};
    matchTE_FM_Sens_IMUag_m1_num(i, :) = length(unique(matchTE_FM_Sens_IMUag_m1{i, :})) - 1;

    fprintf('---- FM_segTE_dil_IMUag_m1_fOR_acouORpiez: %d vs sensTE_mapIMUag: %d > match: %d ...\n', ...
            FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, :), sensTE_mapIMUagC(i, :), matchTE_FM_Sens_IMUag_m1_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTE_FM_Sens_IMUag_m1_INV{i, :} = sensTE_mapIMUagL{i, :} .* FM_segTE_dil_IMUag_m1_fOR_acouORpiez{i, :};
    matchTE_FM_Sens_IMUag_m1_num_INV(i, :) = length(unique(matchTE_FM_Sens_IMUag_m1_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensT_map: %d vs FM_segTE_dil_IMUag_m1_fOR_acouORpiez: %d > match: %d ...\n', ...
            sensTE_mapIMUagC(i, :), FM_segTE_dil_IMUag_m1_fOR_acouORpiezC(i, :), matchTE_FM_Sens_IMUag_m1_num(i, :));
    % -----------------------------------------------------------------------------------------------------

    matchTE_FM_Sens_IMUag_m2{i, :} = FM_segTE_dil_IMUag_m2_fOR_acouORpiezL{i, :} .* sensTE_mapIMUag{i, :};
    matchTE_FM_Sens_IMUag_m2_num(i, :) = length(unique(matchTE_FM_Sens_IMUag_m2{i, :})) - 1;

    fprintf('---- FM_segTE_dil_IMUag_m2_fOR_acouORpiez: %d vs sensTE_mapIMUag: %d > match: %d ...\n', ...
            FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, :), sensTE_mapIMUagC(i, :), matchTE_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

    matchTE_FM_Sens_IMUag_m2_INV{i, :} = sensTE_mapIMUagL{i, :} .* FM_segTE_dil_IMUag_m2_fOR_acouORpiez{i, :};
    matchTE_FM_Sens_IMUag_m2_num_INV(i, :) = length(unique(matchTE_FM_Sens_IMUag_m2_INV{i, :})) - 1;

    fprintf('---- *Inversed* sensTE_mapIMUag: %d vs FM_segTE_dil_IMUag_m2_fOR_acouORpiez: %d > match: %d ...\n\n\n', ...
            sensTE_mapIMUagC(i, :), FM_segTE_dil_IMUag_m2_fOR_acouORpiezC(i, :), matchTE_FM_Sens_IMUag_m2_num(i, :));
    % ---------------------------------------------------------------------------------------------------

% end of loop for the files
end

% save segmented and fused data
% cd([fdir '\' data_folder]);
% save(['Seg01_ellpic_' participant '_' data_category '.mat'], ...
%       'IMUaTE_th', 'IMUaTE_map', 'IMUaTE_mapDil', ...
%       'IMUgTE_th', 'IMUgTE_map', 'IMUgTE_mapDil', ...
%       'sensT_map', 'sensT_mapL', 'sensT_mapC', ...
%       'sensTE_mapIMUa', 'sensTE_mapIMUaL', 'sensTE_mapIMUaC', ...
%       'sensTE_mapIMUg', 'sensTE_mapIMUgL', 'sensTE_mapIMUgC', ...
%       'sensTE_mapIMUag', 'sensTE_mapIMUagL', 'sensTE_mapIMUagC', ...
%       'FM_suiteTE', ...
%       'FM_TE_th', 'FM_segTE', 'FM_segTE_dil', ...
%       'FM_segTE_dil_IMUa_m1', 'FM_segTE_dil_IMUg_m1', 'FM_segTE_dil_IMUag_m1', ...
%       'FM_segTE_dil_IMUa_m2', 'FM_segTE_dil_IMUg_m2', 'FM_segTE_dil_IMUag_m2', ...
%       'FM_segTE_fOR_acou', 'FM_segTE_fOR_piez', 'FM_segTE_fOR_acouORpiez', ...
%       'FM_segTE_dil_fOR_acou', 'FM_segTE_dil_fOR_piez', 'FM_segTE_dil_fOR_acouORpiez', ...
%       'FM_segTE_dil_IMUag_m1_fOR_acou', 'FM_segTE_dil_IMUag_m1_fOR_piez', 'FM_segTE_dil_IMUag_m1_fOR_acouORpiez', ...
%       'FM_segTE_dil_IMUag_m2_fOR_acou', 'FM_segTE_dil_IMUag_m2_fOR_piez', 'FM_segTE_dil_IMUag_m2_fOR_acouORpiez', ...
%       'FM_segTE_dil_fOR_acouORpiezL', 'FM_segTE_dil_fOR_acouORpiezC', ...
%       'FM_segTE_dil_IMUag_m1_fOR_acouORpiezL', 'FM_segTE_dil_IMUag_m1_fOR_acouORpiezC', ...
%       'FM_segTE_dil_IMUag_m2_fOR_acouORpiezL', 'FM_segTE_dil_IMUag_m2_fOR_acouORpiezC', ...
%       'matchTE_FM_Sens', 'matchTE_FM_Sens_num', ...
%       'matchTE_FM_Sens_INV', 'matchTE_FM_Sens_num_INV', ...
%       'matchTE_FM_Sens_IMUag_m1', 'matchTE_FM_Sens_IMUag_m1_num', ...
%       'matchTE_FM_Sens_IMUag_m1_INV', 'matchTE_FM_Sens_IMUag_m1_num_INV', ...
%       'matchTE_FM_Sens_IMUag_m2', 'matchTE_FM_Sens_IMUag_m2_num', ...
%       'matchTE_FM_Sens_IMUag_m2_INV', 'matchTE_FM_Sens_IMUag_m2_num_INV', ...
%       '-v7.3');
% cd(curr_dir)











