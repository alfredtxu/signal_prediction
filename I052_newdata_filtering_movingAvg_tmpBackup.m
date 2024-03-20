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

% thresholds: noisy level
noise_level = [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001];

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
Nstd_IMU_TBMA = 3;
Nstd_IMU_TE = 3;
Nstd_IMU_TEMA = 3;
Nstd_acou_TB = 3;
Nstd_acou_TBMA = 3;
Nstd_acou_TE = 3;
Nstd_acou_TEMA = 3;
Nstd_piez_TB = 3;
Nstd_piez_TE = 3;
Nstd_piez_TBMA = 3;
Nstd_piez_TEMA = 3;

% percentage of overlap between senstation and maternal movement (IMU)
overlap_perc = 0.15;
% -------------------------------------------------------------------------------------------------------

%% Data loading
% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007', 
% data category: F - focus; D - day; N - night
% HOME / OFFICE
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';
% fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

participant = '003AM';
data_folder = ['c_processed_' participant];
data_category = 'focus';
data_serial = 'filtering_movingAvg';

fdir_p = [fdir '\' data_folder];
preproc_dfile = ['Proc01_' participant '_' data_category '_' data_serial '.mat'];
load([fdir_p '\' preproc_dfile]);

% the number of data files
num_files = size(sensT, 1);

%% SEGMENTING PRE-PROCESSED DATA
for i = 1 : num_files

    fprintf('Current data file from %d (%d) - %s ...\n', i, num_files, preproc_dfile);

    %% IMUa
    tmp_IMUaTB = abs(IMUaccNetTB{i, :});
    tmp_IMUaTBMA = abs(IMUaccNetTB_MA{i, :});
    tmp_IMUaTE = abs(IMUaccNetTE{i, :});
    tmp_IMUaTEMA = abs(IMUaccNetTE_MA{i, :});

    IMUaTB_th(i, :) = mean(tmp_IMUaTB) + Nstd_IMU_TB * std(tmp_IMUaTB);
    IMUaTBMA_th(i, :) = mean(tmp_IMUaTBMA) + Nstd_IMU_TBMA * std(tmp_IMUaTBMA);
    IMUaTE_th(i, :) = mean(tmp_IMUaTE) + Nstd_IMU_TE * std(tmp_IMUaTE);
    IMUaTEMA_th(i, :) = mean(tmp_IMUaTEMA) + Nstd_IMU_TEMA * std(tmp_IMUaTEMA);
    
    % IMUa maps
    IMUaTB_map{i, :} = tmp_IMUaTB >= IMUaTB_th(i, :);
    IMUaTB_mapDil{i, :} = imdilate(IMUaTB_map{i, :}, IMU_lse);

    IMUaTBMA_map{i, :} = tmp_IMUaTBMA >= IMUaTBMA_th(i, :);
    IMUaTBMA_mapDil{i, :} = imdilate(IMUaTBMA_map{i, :}, IMU_lse);

    IMUaTE_map{i, :} = tmp_IMUaTE >= IMUaTE_th(i, :);
    IMUaTE_mapDil{i, :} = imdilate(IMUaTE_map{i, :}, IMU_lse);

    IMUaTEMA_map{i, :} = tmp_IMUaTEMA >= IMUaTEMA_th(i, :);
    IMUaTEMA_mapDil{i, :} = imdilate(IMUaTEMA_map{i, :}, IMU_lse);
    % -----------------------------------------------------------------------------------------------
    %% SNR 
    % IMUaTB
    tmp_IMUaTB_P = tmp_IMUaTB(tmp_IMUaTB >= IMUaTB_th(i, :));
    tmp_IMUaTB_N = tmp_IMUaTB(tmp_IMUaTB < IMUaTB_th(i, :));

    p_signal_IMUaTB = 0;
    for pa = 1 : length(tmp_IMUaTB_P)
        p_signal_IMUaTB = p_signal_IMUaTB + tmp_IMUaTB_P(pa)^2;
    end
    n_signal_IMUaTB = 0;
    for na = 1 : length(tmp_IMUaTB_N)
        n_signal_IMUaTB = n_signal_IMUaTB + tmp_IMUaTB_N(na)^2;
    end

    SNR_IMUaTB(i, :) = 10 * log10((p_signal_IMUaTB / length(tmp_IMUaTB_P)) / (n_signal_IMUaTB / length(tmp_IMUaTB_N)));
    
    clear tmp_IMUaTB_P tmp_IMUaTB_N
    % -----------------------------------------------------------------------------------------------

    % IMUaTBMA
    tmp_IMUaTBMA_P = tmp_IMUaTBMA(tmp_IMUaTBMA >= IMUaTBMA_th(i, :));
    tmp_IMUaTBMA_N = tmp_IMUaTBMA(tmp_IMUaTBMA < IMUaTBMA_th(i, :));

    p_signal_IMUaTBMA = 0;
    for pa = 1 : length(tmp_IMUaTBMA_P)
        p_signal_IMUaTBMA = p_signal_IMUaTBMA + tmp_IMUaTBMA_P(pa)^2;
    end
    n_signal_IMUaTBMA = 0;
    for na = 1 : length(tmp_IMUaTBMA_N)
        n_signal_IMUaTBMA = n_signal_IMUaTBMA + tmp_IMUaTBMA_N(na)^2;
    end

    SNR_IMUaTBMA(i, :) = 10 * log10((p_signal_IMUaTBMA / length(tmp_IMUaTBMA_P)) / (n_signal_IMUaTBMA / length(tmp_IMUaTBMA_N)));
    
    clear tmp_IMUaTBMA_P tmp_IMUaTBMA_N
    % -----------------------------------------------------------------------------------------------

    % IMUaTE
    tmp_IMUaTE_P = tmp_IMUaTE(tmp_IMUaTE >= IMUaTE_th(i, :));
    tmp_IMUaTE_N = tmp_IMUaTE(tmp_IMUaTE < IMUaTE_th(i, :));

    p_signal_IMUaTE = 0;
    for pa = 1 : length(tmp_IMUaTE_P)
        p_signal_IMUaTE = p_signal_IMUaTE + tmp_IMUaTE_P(pa)^2;
    end
    n_signal_IMUaTE = 0;
    for na = 1 : length(tmp_IMUaTE_N)
        n_signal_IMUaTE = n_signal_IMUaTE + tmp_IMUaTE_N(na)^2;
    end

    SNR_IMUaTE(i, :) = 10 * log10((p_signal_IMUaTE / length(tmp_IMUaTE_P)) / (n_signal_IMUaTE / length(tmp_IMUaTE_N)));
    
    clear tmp_IMUaTE_P tmp_IMUaTE_N
    % -----------------------------------------------------------------------------------------------

    % IMUaTEMA
    tmp_IMUaTEMA_P = tmp_IMUaTEMA(tmp_IMUaTEMA >= IMUaTEMA_th(i, :));
    tmp_IMUaTEMA_N = tmp_IMUaTEMA(tmp_IMUaTEMA < IMUaTEMA_th(i, :));

    p_signal_IMUaTEMA = 0;
    for pa = 1 : length(tmp_IMUaTEMA_P)
        p_signal_IMUaTEMA = p_signal_IMUaTEMA + tmp_IMUaTEMA_P(pa)^2;
    end
    n_signal_IMUaTEMA = 0;
    for na = 1 : length(tmp_IMUaTEMA_N)
        n_signal_IMUaTEMA = n_signal_IMUaTEMA + tmp_IMUaTEMA_N(na)^2;
    end

    SNR_IMUaTEMA(i, :) = 10 * log10((p_signal_IMUaTEMA / length(tmp_IMUaTEMA_P)) / (n_signal_IMUaTEMA / length(tmp_IMUaTEMA_N)));
    
    clear tmp_IMUaTEMA_P tmp_IMUaTEMA_N

    clear tmp_IMUaTB tmp_IMUaTBMA tmp_IMUaTE tmp_IMUaTEMA
    % -----------------------------------------------------------------------------------------------

    %% IMUg
    tmp_IMUgTB = abs(IMUgyrNetTB{i, :});
    tmp_IMUgTBMA = abs(IMUgyrNetTB_MA{i, :});
    tmp_IMUgTE = abs(IMUgyrNetTE{i, :});
    tmp_IMUgTEMA = abs(IMUgyrNetTE_MA{i, :});

    IMUgTB_th(i, :) = mean(tmp_IMUgTB) + Nstd_IMU_TB * std(tmp_IMUgTB);
    IMUgTBMA_th(i, :) = mean(tmp_IMUgTBMA) + Nstd_IMU_TBMA * std(tmp_IMUgTBMA);
    IMUgTE_th(i, :) = mean(tmp_IMUgTE) + Nstd_IMU_TE * std(tmp_IMUgTE);
    IMUgTEMA_th(i, :) = mean(tmp_IMUgTEMA) + Nstd_IMU_TEMA * std(tmp_IMUgTEMA);
    
    % IMUg maps
    IMUgTB_map{i, :} = tmp_IMUgTB >= IMUgTB_th(i, :);
    IMUgTB_mapDil{i, :} = imdilate(IMUgTB_map{i, :}, IMU_lse);

    IMUgTBMA_map{i, :} = tmp_IMUgTBMA >= IMUgTBMA_th(i, :);
    IMUgTBMA_mapDil{i, :} = imdilate(IMUgTBMA_map{i, :}, IMU_lse);

    IMUgTE_map{i, :} = tmp_IMUgTE >= IMUgTE_th(i, :);
    IMUgTE_mapDil{i, :} = imdilate(IMUgTE_map{i, :}, IMU_lse);

    IMUgTEMA_map{i, :} = tmp_IMUgTEMA >= IMUgTEMA_th(i, :);
    IMUgTEMA_mapDil{i, :} = imdilate(IMUgTEMA_map{i, :}, IMU_lse);
    % -----------------------------------------------------------------------------------------------

    %% SNR 
    % IMUgTB
    tmp_IMUgTB_P = tmp_IMUgTB(tmp_IMUgTB >= IMUgTB_th(i, :));
    tmp_IMUgTB_N = tmp_IMUgTB(tmp_IMUgTB < IMUgTB_th(i, :));

    p_signal_IMUgTB = 0;
    for pa = 1 : length(tmp_IMUgTB_P)
        p_signal_IMUgTB = p_signal_IMUgTB + tmp_IMUgTB_P(pa)^2;
    end
    n_signal_IMUgTB = 0;
    for na = 1 : length(tmp_IMUgTB_N)
        n_signal_IMUgTB = n_signal_IMUgTB + tmp_IMUgTB_N(na)^2;
    end

    SNR_IMUgTB(i, :) = 10 * log10((p_signal_IMUgTB / length(tmp_IMUgTB_P)) / (n_signal_IMUgTB / length(tmp_IMUgTB_N)));
    
    clear tmp_IMUgTB_P tmp_IMUgTB_N
    % -----------------------------------------------------------------------------------------------

    % IMUgTBMA
    tmp_IMUgTBMA_P = tmp_IMUgTBMA(tmp_IMUgTBMA >= IMUgTBMA_th(i, :));
    tmp_IMUgTBMA_N = tmp_IMUgTBMA(tmp_IMUgTBMA < IMUgTBMA_th(i, :));

    p_signal_IMUgTBMA = 0;
    for pa = 1 : length(tmp_IMUgTBMA_P)
        p_signal_IMUgTBMA = p_signal_IMUgTBMA + tmp_IMUgTBMA_P(pa)^2;
    end
    n_signal_IMUgTBMA = 0;
    for na = 1 : length(tmp_IMUgTBMA_N)
        n_signal_IMUgTBMA = n_signal_IMUgTBMA + tmp_IMUgTBMA_N(na)^2;
    end

    SNR_IMUgTBMA(i, :) = 10 * log10((p_signal_IMUgTBMA / length(tmp_IMUgTBMA_P)) / (n_signal_IMUgTBMA / length(tmp_IMUgTBMA_N)));
    
    clear tmp_IMUgTBMA_P tmp_IMUgTBMA_N
    % -----------------------------------------------------------------------------------------------

    % IMUgTE
    tmp_IMUgTE_P = tmp_IMUgTE(tmp_IMUgTE >= IMUgTE_th(i, :));
    tmp_IMUgTE_N = tmp_IMUgTE(tmp_IMUgTE < IMUgTE_th(i, :));

    p_signal_IMUgTE = 0;
    for pa = 1 : length(tmp_IMUgTE_P)
        p_signal_IMUgTE = p_signal_IMUgTE + tmp_IMUgTE_P(pa)^2;
    end
    n_signal_IMUgTE = 0;
    for na = 1 : length(tmp_IMUgTE_N)
        n_signal_IMUgTE = n_signal_IMUgTE + tmp_IMUgTE_N(na)^2;
    end

    SNR_IMUgTE(i, :) = 10 * log10((p_signal_IMUgTE / length(tmp_IMUgTE_P)) / (n_signal_IMUgTE / length(tmp_IMUgTE_N)));
    
    clear tmp_IMUgTE_P tmp_IMUgTE_N
    % -----------------------------------------------------------------------------------------------

    % IMUgTEMA
    tmp_IMUgTEMA_P = tmp_IMUgTEMA(tmp_IMUgTEMA >= IMUgTEMA_th(i, :));
    tmp_IMUgTEMA_N = tmp_IMUgTEMA(tmp_IMUgTEMA < IMUgTEMA_th(i, :));

    p_signal_IMUgTEMA = 0;
    for pa = 1 : length(tmp_IMUgTEMA_P)
        p_signal_IMUgTEMA = p_signal_IMUgTEMA + tmp_IMUgTEMA_P(pa)^2;
    end
    n_signal_IMUgTEMA = 0;
    for na = 1 : length(tmp_IMUgTEMA_N)
        n_signal_IMUgTEMA = n_signal_IMUgTEMA + tmp_IMUgTEMA_N(na)^2;
    end

    SNR_IMUgTEMA(i, :) = 10 * log10((p_signal_IMUgTEMA / length(tmp_IMUgTEMA_P)) / (n_signal_IMUgTEMA / length(tmp_IMUgTEMA_N)));
    
    clear tmp_IMUgTEMA_P tmp_IMUgTEMA_N

    clear tmp_IMUgTB tmp_IMUgTBMA tmp_IMUgTE tmp_IMUgTEMA
    % -----------------------------------------------------------------------------------------------

    %% Sensation
    %% Initializing the map: sensT
    tmp_sensT = sensT{i, :};
    tmp_sensTP = find(tmp_sensT == 1);

    tmp_sensT_map = zeros(length(tmp_sensT), 1); 
    
    tmp_sensTB_mapIMUa = zeros(length(tmp_sensT), 1); 
    tmp_sensTB_mapIMUg = zeros(length(tmp_sensT), 1); 
    tmp_sensTB_mapIMUag = zeros(length(tmp_sensT), 1); 

    tmp_sensTE_mapIMUa = zeros(length(tmp_sensT), 1); 
    tmp_sensTE_mapIMUg = zeros(length(tmp_sensT), 1); 
    tmp_sensTE_mapIMUag = zeros(length(tmp_sensT), 1);

    for j1 = 1 : length(tmp_sensTP) 

        tmp_idx = tmp_sensTP(j1); 
        
        tmp_idxS = tmp_idx - sens_dilation_sizeB; 
        tmp_idxE = tmp_idx + sens_dilation_sizeF; 

        tmp_idxS = max(tmp_idxS, 1);
        tmp_idxE = min(tmp_idxE, length(tmp_sensT_map));
        
        % sensation map
        tmp_sensT_map(tmp_idxS:tmp_idxE) = 1; 

        tmp_sensTB_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTB_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTB_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        tmp_sensTE_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTE_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTE_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        % Remove maternal sensation coincided with body movement (IMU a/g) - butterworth
        tmp_xIMUaTB = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUaTB_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMugTB = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUgTB_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTB >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTB_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTB_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMugTB >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTB_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTB_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        % Remove maternal sensation coincided with body movement (IMU a/g) - ellipic
        tmp_xIMUaTE = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUaTE_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMugTE = sum(tmp_sensT_map(tmp_idxS:tmp_idxE) .* IMUgTE_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTE >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTE_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTE_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMugTE >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTE_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTE_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTB tmp_xIMugTB tmp_xIMUaTE tmp_xIMugTE
    end

    % sensation maps
    sensT_map{i, :} = tmp_sensT_map;
    sensTB_mapIMUa{i, :} = tmp_sensTB_mapIMUa;
    sensTB_mapIMUg{i, :} = tmp_sensTB_mapIMUg;
    sensTB_mapIMUag{i, :} = tmp_sensTB_mapIMUag;
    sensTE_mapIMUa{i, :} = tmp_sensTE_mapIMUa;
    sensTE_mapIMUg{i, :} = tmp_sensTE_mapIMUg;
    sensTE_mapIMUag{i, :} = tmp_sensTE_mapIMUag;
    
    clear tmp_sensT tmp_sensTP tmp_sensT_map ...
          tmp_sensTB_mapIMUa tmp_sensTB_mapIMUg tmp_sensTB_mapIMUag ...
          tmp_sensTE_mapIMUa tmp_sensTE_mapIMUg tmp_sensTE_mapIMUag
    % --------------------------------------------------------------------------------

    %% Initializing the map: sensT_MA_threshB
    tmp_sensTMA = sensT_MA_threshB{i, :};
    tmp_sensTMAP = find(tmp_sensTMA == 1);

    tmp_sensTMA_map = zeros(length(tmp_sensTMA), 1); 
    
    tmp_sensTBMA_mapIMUa = zeros(length(tmp_sensTMA), 1); 
    tmp_sensTBMA_mapIMUg = zeros(length(tmp_sensTMA), 1); 
    tmp_sensTBMA_mapIMUag = zeros(length(tmp_sensTMA), 1); 

    tmp_sensTEMA_mapIMUa = zeros(length(tmp_sensTMA), 1); 
    tmp_sensTEMA_mapIMUg = zeros(length(tmp_sensTMA), 1); 
    tmp_sensTEMA_mapIMUag = zeros(length(tmp_sensTMA), 1);

    for j1 = 1 : length(tmp_sensTMAP) 

        tmp_idx = tmp_sensTMAP(j1); 
        
        tmp_idxS = tmp_idx - sens_dilation_sizeB; 
        tmp_idxE = tmp_idx + sens_dilation_sizeF; 

        tmp_idxS = max(tmp_idxS, 1);
        tmp_idxE = min(tmp_idxE, length(tmp_sensTMA_map));
        
        % sensation map
        tmp_sensTMA_map(tmp_idxS:tmp_idxE) = 1; 

        tmp_sensTBMA_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTBMA_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTBMA_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        tmp_sensTEMA_mapIMUa(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTEMA_mapIMUg(tmp_idxS:tmp_idxE) = 1;
        tmp_sensTEMA_mapIMUag(tmp_idxS:tmp_idxE) = 1;

        % Remove maternal sensation coincided with body movement (IMU a/g) - butterworth
        tmp_xIMUaTBMA = sum(tmp_sensTMA_map(tmp_idxS:tmp_idxE) .* IMUaTBMA_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMugTBMA = sum(tmp_sensTMA_map(tmp_idxS:tmp_idxE) .* IMUgTBMA_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTBMA >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTBMA_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTBMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMugTBMA >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTBMA_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTBMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        % Remove maternal sensation coincided with body movement (IMU a/g) - ellipic
        tmp_xIMUaTEMA = sum(tmp_sensTMA_map(tmp_idxS:tmp_idxE) .* IMUaTEMA_mapDil{i, :}(tmp_idxS:tmp_idxE));
        tmp_xIMugTEMA = sum(tmp_sensTMA_map(tmp_idxS:tmp_idxE) .* IMUgTEMA_mapDil{i, :}(tmp_idxS:tmp_idxE));

        if (tmp_xIMUaTEMA >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTEMA_mapIMUa(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTEMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end
        
        if (tmp_xIMugTEMA >= overlap_perc*(tmp_idxE-tmp_idxS+1))
            tmp_sensTEMA_mapIMUg(tmp_idxS:tmp_idxE) = 0;
            tmp_sensTEMA_mapIMUag(tmp_idxS:tmp_idxE) = 0;
        end

        clear tmp_idx tmp_idxS tmp_idxE tmp_xIMUaTBMA tmp_xIMugTBMA tmp_xIMUaTEMA tmp_xIMugTEMA
    end

    % sensation maps
    sensTMA_map{i, :} = tmp_sensTMA_map;
    sensTBMA_mapIMUa{i, :} = tmp_sensTBMA_mapIMUa;
    sensTBMA_mapIMUg{i, :} = tmp_sensTBMA_mapIMUg;
    sensTBMA_mapIMUag{i, :} = tmp_sensTBMA_mapIMUag;
    sensTEMA_mapIMUa{i, :} = tmp_sensTEMA_mapIMUa;
    sensTEMA_mapIMUg{i, :} = tmp_sensTEMA_mapIMUg;
    sensTEMA_mapIMUag{i, :} = tmp_sensTEMA_mapIMUag;
    
    clear tmp_sensTMA tmp_sensTMAP tmp_sensTMA_map ...
          tmp_sensTBMA_mapIMUa tmp_sensTBMA_mapIMUg tmp_sensTBMA_mapIMUag ...
          tmp_sensTEMA_mapIMUa tmp_sensTEMA_mapIMUg tmp_sensTEMA_mapIMUag          
    % ------------------------------------------------------------------------------------------------

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

    FM_suiteTE{i, 1} = acouLTE{i,:};
    FM_suiteTE{i, 2} = acouMTE{i,:};
    FM_suiteTE{i, 3} = acouRTE{i,:};
    FM_suiteTE{i, 4} = piezLTE{i,:};
    FM_suiteTE{i, 5} = piezMTE{i,:};
    FM_suiteTE{i, 6} = piezRTE{i,:};

    FM_suiteTBMA{i, 1} = acouLTB_MA{i,:};
    FM_suiteTBMA{i, 2} = acouMTB_MA{i,:};
    FM_suiteTBMA{i, 3} = acouRTB_MA{i,:};
    FM_suiteTBMA{i, 4} = piezLTB_MA{i,:};
    FM_suiteTBMA{i, 5} = piezMTB_MA{i,:};
    FM_suiteTBMA{i, 6} = piezRTB_MA{i,:};

    FM_suiteTEMA{i, 1} = acouLTE_MA{i,:};
    FM_suiteTEMA{i, 2} = acouMTE_MA{i,:};
    FM_suiteTEMA{i, 3} = acouRTE_MA{i,:};
    FM_suiteTEMA{i, 4} = piezLTE_MA{i,:};
    FM_suiteTEMA{i, 5} = piezMTE_MA{i,:};
    FM_suiteTEMA{i, 6} = piezRTE_MA{i,:};

    %% FM Sensors Thresholding (butterworth)
    for j2 = 1 : num_FMsensors

        tmp_sensor_indiv = abs(FM_suiteTB{i, j2});
        
        %% Segmentation by a designated threshold
        % determine the individual/adaptive threshold
        if j2 <= num_FMsensors / 2
            tmp_std_times_FM = Nstd_acou_TB;
        else
            tmp_std_times_FM = Nstd_piez_TB;
        end
        FM_threshTB(i, j2) = mean(tmp_sensor_indiv) + tmp_std_times_FM * std(tmp_sensor_indiv);

        if isnan(FM_threshTB(i, j2))
            FM_threshTB(i, j2) = Inf;
        elseif FM_threshTB(i, j2) < noise_level
            FM_threshTB(i, j2) = Inf;
        end

        FM_segTB{i, j2} = (tmp_sensor_indiv >= FM_threshTB(i, j2));
        FM_segTB_dil{i, j2} = double(imdilate(FM_segTB{i, j2}, FM_lse)); 

        % method 1: Exclusion of body movement data
        FM_segTB_dil_IMUa{i, j2} = FM_segTB_dil{i, j2} .* (1-IMUaTB_mapDil{i,:}); 
        FM_segTB_dil_IMUg{i, j2} = FM_segTB_dil{i, j2} .* (1-IMUgTB_mapDil{i,:}); 
        FM_segTB_dil_IMUag{i, j2} = FM_segTB_dil{i, j2} .* (1-IMUaTB_mapDil{i,:}) .* (1-IMUgTB_mapDil{i,:}); 

        % labels and components
        [tmp_LTB, tmp_CTB] = bwlabel(FM_segTB{i, j2});
        [tmp_LTB_dil, tmp_CTB_dil] = bwlabel(FM_segTB_dil{i, j2});
        [tmp_LTB_IMUa, tmp_CTB_IMUa] = bwlabel(FM_segTB_dil_IMUa{i, j2});
        [tmp_LTB_IMUg, tmp_CTB_IMUg] = bwlabel(FM_segTB_dil_IMUg{i, j2});
        [tmp_LTB_IMUag, tmp_CTB_IMUag] = bwlabel(FM_segTB_dil_IMUag{i, j2});

        % method 2: Exclusion of body movement data
        tmp_FM_segTB_dil = zeros(length(FM_segTB_dil{i, j2}), 1);
        for c = 1: tmp_CTB_dil
            tmp_idxC = find(tmp_LTB_dil == c);
            tmp_FM_segTB_dil(tmp_idxC, :) = 1;

            tmp_xIMUa = sum(tmp_FM_segTB_dil .* IMUaTB_mapDil{i, :}(tmp_idxC, :));
            tmp_xIMUg = sum(tmp_FM_segTB_dil .* IMUgTB_mapDil{i, :}(tmp_idxC, :));

        end
        


        % -------------------------------------------------------------------------------------------------------

        %% SNR 
        % binarising: above threshold - 1 otherwise - 0
        tmp_FM_removed = tmp_sensor_indiv(tmp_sensor_indiv < FM_threshTB(i, j2));
        tmp_FM_segmented = tmp_sensor_indiv(tmp_sensor_indiv >= FM_threshTB(i, j2));

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
        % --------------------------------------------------------------------------------------------------------
        
        
        %% match of FM and sensation
        [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(sensT_map{i, :});
        [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(sensTB_mapIMUag{i, :});
    
        [tmp_sensor_label, tmp_sensor_comp] = bwlabel(FM_segTB{i, j2});
        [tmp_sensor_IMUag_label, tmp_sensor_IMUag_comp] = bwlabel(FM_segTB_dil_IMUag{i, j2});
    
        match_indiv_sensation{i, j2} = tmp_sensor_label .* sensT_map{i, :};
        match_indiv_sensation_num(i, j2) = length(unique(match_indiv_sensation{i, j2})) - 1;
        
        match_indiv_sensation_INV{i, j2} = tmp_sens_map_label .* FM_segTB{i, j2};
        match_indiv_sensation_num_INV(i, j2) = length(unique(match_indiv_sensation_INV{i, j2})) - 1;
    
        match_indiv_IMUag_sensation{i, j2} = tmp_sensor_IMUag_label .* sensTB_mapIMUag{i, :};
        match_indiv_IMUag_sensation_num(i, j2) = length(unique(match_indiv_IMUag_sensation{i, j2})) - 1;
    
        match_indiv_IMUag_sensation_INV{i, j2} = tmp_sens_mapIMUag_label .* FM_segTB_dil_IMUag{i, j2};
        match_indiv_IMUag_sensation_num_INV(i, j2) = length(unique(match_indiv_IMUag_sensation_INV{i, j2})) - 1;
    
%         fprintf('Signal segmentation file %d - sensor %d >> comp (sens / FM): %d / %d; match: %d ...\n', ...
%                 i, j2, tmp_sens_map_comp, tmp_sensor_comp, match_indiv_sensation_num(i, j2));
    
        fprintf('Signal segmentation file %d - sensor %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
                i, j2, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_indiv_IMUag_sensation_num(i, j2));

%         fprintf('Signal segmentation file %d - sensor %d >> comp (sens / FM): %d / %d; match: %d ...\n', ...
%                 i, j2, tmp_sens_map_comp, tmp_sensor_comp, match_indiv_sensation_num_INV(i, j2));
    
        fprintf('Signal segmentation file %d - sensor %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
                i, j2, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_indiv_IMUag_sensation_num_INV(i, j2));

        clear tmp_sensor_indiv tmp_low_cutoff tmp_sensor_suite_low ...
              tmp_sens_map_label tmp_sens_map_comp tmp_sens_mapIMUag_label tmp_sens_mapIMUag_comp ...
              tmp_sensor_label tmp_sensor_comp tmp_sensor_IMUag_label tmp_sensor_IMUag_comp ...
              tmp_std_times_FM
    end
    % -------------------------------------------------------------------------
    
    % Sensor fusion group I:
    % * Data fusion performed after dilation.
    % For sensor type, they are combined with logical 'OR'
    % Between the types of sensors, they are combined with logical 'OR' / 'AND'
    sensor_fusionOR_acou{i, :} = double(FM_segTB{i, 1} | FM_segTB{i, 2} | FM_segTB{i, 3});
    sensor_fusionOR_piez{i, :} = double(FM_segTB{i, 4} | FM_segTB{i, 5} | FM_segTB{i, 6});
    sensor_fusionOR_acouORpiez{i, :}  = double(sensor_fusionOR_acou{i,:} | sensor_fusionOR_piez{i, :});
    
    sensor_IMUag_fusionOR_acou{i, :} = double(FM_segTB_dil_IMUag{i, 1} | FM_segTB_dil_IMUag{i, 2} | FM_segTB_dil_IMUag{i, 3});
    sensor_IMUag_fusionOR_piez{i, :} = double(FM_segTB_dil_IMUag{i, 4} | FM_segTB_dil_IMUag{i, 5} | FM_segTB_dil_IMUag{i, 6});
    sensor_IMUag_fusionOR_acouORpiez{i, :}  = double(sensor_IMUag_fusionOR_acou{i,:} | sensor_IMUag_fusionOR_piez{i, :});
    % -------------------------------------------------------------------------

    %% match of FM and sensation
    [tmp_sens_map_label, tmp_sens_map_comp] = bwlabel(sensT_map{i, :});
    [tmp_sens_mapIMUag_label, tmp_sens_mapIMUag_comp] = bwlabel(sensTB_mapIMUag{i, :});

    [tmp_sensor_label, tmp_sensor_comp] = bwlabel(sensor_fusionOR_acouORpiez{i, :});
    [tmp_sensor_IMUag_label, tmp_sensor_IMUag_comp] = bwlabel(sensor_IMUag_fusionOR_acouORpiez{i, :});

    match_sensor_sensation{i, :} = tmp_sensor_label .* sensT_map{i, :};
    match_sensor_sensation_num(i, :) = length(unique(match_sensor_sensation{i, :})) - 1;

    match_sensor_sensation_INV{i, :} = tmp_sens_map_label .* sensor_fusionOR_acouORpiez{i, :};
    match_sensor_sensation_num_INV(i, :) = length(unique(match_sensor_sensation_INV{i, :})) - 1;
    % ---------------------------------------------------------------------------------------------------

    match_sensor_IMUag_sensation{i, :} = tmp_sensor_IMUag_label .* sensTB_mapIMUag{i, :};
    match_sensor_IMUag_sensation_num(i, :) = length(unique(match_sensor_IMUag_sensation{i, :})) - 1;

    match_sensor_IMUag_sensation_INV{i, :} = tmp_sens_mapIMUag_label .* sensor_IMUag_fusionOR_acouORpiez{i, :};
    match_sensor_IMUag_sensation_num_INV(i, :) = length(unique(match_sensor_IMUag_sensation_INV{i, :})) - 1;
    % ---------------------------------------------------------------------------------------------------

%     fprintf('Signal segmentation file %d >> comp (sens / FM): %d / %d; match: %d ...\n', ...
%             i, tmp_sens_map_comp, tmp_sensor_comp, match_sensor_sensation_num(i, :));

    fprintf('Fusion > Signal segmentation file %d >> comp (sens_IMUag / FM_IMUag): %d / %d; match: %d ...\n', ...
            i, tmp_sens_mapIMUag_comp, tmp_sensor_IMUag_comp, match_sensor_IMUag_sensation_num(i, :));

%     fprintf('Signal segmentation file %d >> comp (sens / FM): %d / %d; match: %d ...\n', ...
%             i, tmp_sens_map_comp, tmp_sensor_comp, match_sensor_sensation_num_INV(i, :));

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
%       '-v7.3');
% cd(curr_dir)











