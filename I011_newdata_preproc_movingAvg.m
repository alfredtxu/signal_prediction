%% Moving Average Filter:
% Purpose: Calculates the average of a window of data points to reduce noise.
% Applications: Smoothing noisy signals, simple implementation.
% Characteristics: Attenuates high-frequency noise, may introduce lag.

%% Cascading Filters:
% Apply multiple filters in sequence, with the output of one filter serving as the input to the next. 
% Cascading filters can help achieve a multi-stage signal conditioning process.
% 
% Firstly, apply a band- / low-pass filter to remove high-frequency noise; 
% Then follow it with a moving average filter to further smooth the signal. 

clc
clear
close all

% HOME / OFFICE
% curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
curr_dir = pwd;
cd(curr_dir);

%% Pre-setting
% Define known parameters
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
% old data: 1024 Hz, new yellow belt: 512 Hz, new pink belt: 400 Hz
freq = 400;
num_channel = 16;
num_FMsensors = 6;

% ADC resolution is 12 bit
ADC_resolution = 12;
maxADC_value = 2^ADC_resolution-1;

% 3.3 volt corresponds to 4095 ADC value (12-bit)
volt = 3.3;
slope = volt / maxADC_value;

% Maximum value of the test data
maxTestData_value = 256;

% Trim settings
% Remove the beginning and end period in second
trimS = 60;
trimE = 60;

%% Filter design
% filter order
filter_order = 10;

% Bandpass filter: 1-30Hz disigned as fetal movement, and Bandwidth (Hz)
bpL_FM = 1;
bpH_FM = 30;
bpBW_FM = bpH_FM - bpL_FM; 

% Bandpass filter: 1-10Hz designed as the IMU data, and Bandwidth (Hz)
bpL_IMU = 1;
bpH_IMU = 10;
bpBW_IMU = bpH_IMU - bpL_IMU; 

% Lowpass filter: 10Hz designed as the force sensor
lpL_force = 10;
lpH_force = 30;

% Method: Butterworth 
% filter order: twice the value of 1st parameter
[b_FM, a_FM] = butter(filter_order/2, [bpL_FM bpH_FM] / (freq/2), 'bandpass');
[z_FM, p_FM, k_FM] = butter(filter_order/2, [bpL_FM bpH_FM] / (freq/2), 'bandpass');
[sos_FM, g_FM] = zp2sos(z_FM, p_FM, k_FM);

% Method: Butterworth 
% filter order: twice the value of 1st parameter
[b_IMU,a_IMU] = butter(filter_order/2,[bpL_IMU bpH_IMU]/(freq/2),'bandpass');
[z_IMU,p_IMU,k_IMU] = butter(filter_order/2,[bpL_IMU bpH_IMU]/(freq/2),'bandpass');
[sos_IMU,g_IMU] = zp2sos(z_IMU,p_IMU,k_IMU);

% Method: Butterworth 
% filter order: the value of 1st parameter
[z_force,p_force,k_force] = butter(filter_order,lpL_force/(freq/2),'low');
[sos_force,g_force] = zp2sos(z_force,p_force,k_force);
% --------------------------------------------------------------------------------------------------

% Method: Ellip
% [n,Wn] = ellipord(Wp,Ws,Rp,Rs) - * Wp, Ws are [0 1]
% FM sensors
Wp_FM = [bpL_FM, bpH_FM] / (freq/2);

if (bpL_FM - bpBW_FM/2) < 0
    Ws_FM = [bpL_FM/2, (bpH_FM + bpBW_FM/2)] / (freq/2);
else
    Ws_FM = [bpL_FM - bpBW_FM/2, (bpH_FM + bpBW_FM/2)] / (freq/2);
end

Rp_FM = 1;
Rs_FM = 40;

% IMU
Wp_IMU = [bpL_IMU, bpH_IMU] / (freq/2);

if (bpL_IMU - bpBW_IMU/2) < 0
    Ws_IMU = [bpL_IMU/2, (bpH_IMU + bpBW_IMU/2)] / (freq/2);
else
    Ws_IMU = [(bpL_IMU - bpBW_IMU/2), (bpH_IMU + bpBW_IMU/2)] / (freq/2);
end

Rp_IMU = 1;
Rs_IMU = 40;

% Force sensor
Wp_force = lpL_force / (freq/2);
Ws_force = lpH_force / (freq/2);
Rp_force = 1;
Rs_force = 40;

% Design the elliptic bandpass filter: FM
[N_FM, Wn_FM] = ellipord(Wp_FM, Ws_FM, Rp_FM, Rs_FM);
[b_ellip_FM, a_ellip_FM] = ellip(N_FM, Rp_FM, Rs_FM, Wn_FM, 'bandpass');
[z_ellip_FM, p_ellip_IMU, k_ellip_FM] = ellip(N_FM, Rp_FM, Rs_FM, Wn_FM, 'bandpass');
[sos_ellip_FM, g_ellip_FM] = zp2sos(z_ellip_FM, p_ellip_IMU, k_ellip_FM);

% Design the elliptic bandpass filter: IMU
[N_IMU, Wn_IMU] = ellipord(Wp_IMU, Ws_IMU, Rp_IMU, Rs_IMU);
[b_ellip_IMU, a_ellip_IMU] = ellip(N_IMU, Rp_IMU, Rs_IMU, Wn_IMU, 'bandpass');
[z_ellip_IMU, p_ellip_IMU, k_ellip_IMU] = ellip(N_IMU, Rp_IMU, Rs_IMU, Wn_IMU, 'bandpass');
[sos_ellip_IMU, g_ellip_IMU] = zp2sos(z_ellip_IMU, p_ellip_IMU, k_ellip_IMU);

% Design the elliptic lowpass filter: force
[N_force, Wn_force] = ellipord(Wp_force, Ws_force, Rp_force, Rs_force);
[b_ellip_force, a_ellip_force] = ellip(N_force, Rp_force, Rs_force, Wn_force, 'low');
[z_ellip_force, p_ellip_force, k_ellip_force] = ellip(N_force, Rp_force, Rs_force, Wn_force, 'low');
[sos_ellip_force, g_ellip_force] = zp2sos(z_ellip_force, p_ellip_force, k_ellip_force);
% --------------------------------------------------------------------------------------------------

%% Moving average
% moving size window in second
shift_size = 3.0;
shift_thresh = 0.01;

%% Plotting parameters
ylim_multiplier = 1.2;

sntn_multiplier = 1;
IMU_multiplier = 1;
fm_multiplier = 0.9;

line_width = 1;
box_width = 1; 

Font_size_labels = 8;
Font_size_legend = 8;

col_code1 = [0 0 0];
col_code2 = [0.4660 0.6740 0.1880];
col_code3 = [0.4940 0.1840 0.5560];
col_code4 = [0.8500 0.3250 0.0980];
col_code5 = [0.9290 0.6940 0.1250];

%% DATA LOADING
% data file directory
% HOME / OFFICE
% fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';
fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007', 
% data category: F - focus; D - day; N - night
participant = '003AM';
data_folder = ['b_mat_' participant];
data_processed = ['c_processed_' participant];
data_category = 'focus';
% --------------------------------------------------------------------------------------------------
switch data_category
    case 'focus'
        fdir_p = [fdir '\' data_folder '_F'];
        files = dir([fdir_p '\*.mat']);
    case 'day'
        fdir_p = [fdir '\' data_folder '_D'];
        files = dir([fdir_p '\*.mat']);
    case 'night'
        fdir_p = [fdir '\' data_folder '_N'];
        files = dir([fdir_p '\*.mat']);
    otherwise
        disp('Error: the designated particicpant is beyond pre-setting...')
end
% --------------------------------------------------------------------------------------------------

% Loading the data files
for i = 1 : length(files)

    tmp_file = files(i).name;
    load([fdir_p '\' tmp_file]);

    fprintf('Current data file < %s (%s): %d - %s ...\n', data_folder, data_category, i, tmp_file);

    % assort the sensor data - orignal signal
    forc{i, :} = tmp_mtx(:, 1) * slope;
    acouL{i, :} = tmp_mtx(:, 5) * slope;
    acouM{i, :} = tmp_mtx(:, 4) * slope;
    acouR{i, :} = tmp_mtx(:, 6) * slope;
    piezL{i, :} = tmp_mtx(:, 7) * slope;
    piezM{i, :} = tmp_mtx(:, 3) * slope;
    piezR{i, :} = tmp_mtx(:, 2) * slope;
    IMUacc{i, :} = tmp_mtx(:, 9:11) * slope;
    IMUgyr{i, :} = tmp_mtx(:, 12:14) * slope;
    IMUaccNet{i, :} = sqrt(sum(IMUacc{i, :}.^2,2));
    IMUgyrNet{i, :} = sqrt(sum(IMUgyr{i, :}.^2,2));
    sens{i, :} = bitshift(tmp_mtx(:, 8), -8);

    % Time vector  - orginal signal
    tmp_len = length(forc{i, :});
    tmp_dura = ((0 : (tmp_len-1)) / freq)';

    % assort the sensor data - trimmed data from both ends
    forcT{i, :} = forc{i, :}((trimS*freq + 1):(end-trimE*freq));
    acouLT{i, :} = acouL{i, :}((trimS*freq + 1):(end-trimE*freq));
    acouMT{i, :} = acouM{i, :}((trimS*freq + 1):(end-trimE*freq));
    acouRT{i, :} = acouR{i, :}((trimS*freq + 1):(end-trimE*freq));
    piezLT{i, :} = piezL{i, :}((trimS*freq + 1):(end-trimE*freq));
    piezMT{i, :} = piezM{i, :}((trimS*freq + 1):(end-trimE*freq));
    piezRT{i, :} = piezR{i, :}((trimS*freq + 1):(end-trimE*freq));
    IMUaccT{i, :} = IMUacc{i, :}((trimS*freq + 1):(end-trimE*freq), :);
    IMUgyrT{i, :} = IMUgyr{i, :}((trimS*freq + 1):(end-trimE*freq), :);
    IMUaccNetT{i, :} = IMUaccNet{i, :}((trimS*freq + 1):(end-trimE*freq));
    IMUgyrNetT{i, :} = IMUgyrNet{i, :}((trimS*freq + 1):(end-trimE*freq));
    sensT{i, :} = sens{i, :}((trimS*freq + 1):(end-trimE*freq));

    % Bandpass & Low-pass filtering: butterworth
    forcTB{i, :} = filtfilt(sos_force, g_force, forcT{i, :});
    acouLTB{i, :} = filtfilt(sos_FM, g_FM, acouLT{i, :});
    acouMTB{i, :} = filtfilt(sos_FM, g_FM, acouMT{i, :});
    acouRTB{i, :} = filtfilt(sos_FM, g_FM, acouRT{i, :});
    piezLTB{i, :} = filtfilt(sos_FM, g_FM, piezLT{i, :});
    piezMTB{i, :} = filtfilt(sos_FM, g_FM, piezMT{i, :});
    piezRTB{i, :} = filtfilt(sos_FM, g_FM, piezRT{i, :});
    IMUaccNetTB{i, :} = filtfilt(sos_IMU, g_IMU, IMUaccNetT{i, :});
    IMUgyrNetTB{i, :} = filtfilt(sos_IMU, g_IMU, IMUgyrNetT{i, :});

    % Bandpass & Low-pass filtering: elliptic
    % filtered_signal = filtfilt(b_ellip, a_ellip, signal);
    forcTE{i, :} = filtfilt(sos_ellip_force, g_ellip_force, forcT{i, :});
    acouLTE{i, :} = filtfilt(sos_ellip_FM, g_ellip_FM, acouLT{i, :});
    acouMTE{i, :} = filtfilt(sos_ellip_FM, g_ellip_FM, acouMT{i, :});
    acouRTE{i, :} = filtfilt(sos_ellip_FM, g_ellip_FM, acouRT{i, :});
    piezLTE{i, :} = filtfilt(sos_ellip_FM, g_ellip_FM, piezLT{i, :});
    piezMTE{i, :} = filtfilt(sos_ellip_FM, g_ellip_FM, piezMT{i, :});
    piezRTE{i, :} = filtfilt(sos_ellip_FM, g_ellip_FM, piezRT{i, :});
    IMUaccNetTE{i, :} = filtfilt(sos_ellip_IMU, g_ellip_IMU, IMUaccNetT{i, :});
    IMUgyrNetTE{i, :} = filtfilt(sos_ellip_IMU, g_ellip_IMU, IMUgyrNetT{i, :});

    % time vector - trimmed
    tmp_lenT = length(forcT{i, :});
    tmp_duraT = ((0 : (tmp_lenT-1)) / freq)';
    
    % the whole sensation and its subset of button presses
    sensT_mtx{i, :} = [tmp_duraT, sensT{i, :}];
    sensT_idxP{i, :} = sensT{i, :}==1;
    sensT_mtxP{i, :} = sensT_mtx{i, :}(sensT_idxP{i, :}, :);

    % Moving average
    tmp_lenT_MA = tmp_lenT - shift_size * freq + 1;
    tmp_duraT_MA = ((0 : (tmp_lenT_MA-1)) / freq)';

    for w = 1 : tmp_lenT_MA
        forcTB_MA{i, :}(w, :) = mean(forcTB{i, :}(w : w+shift_size * freq-1, :));
        acouLTB_MA{i, :}(w, :) = mean(acouLTB{i, :}(w : w+shift_size * freq-1, :));
        acouMTB_MA{i, :}(w, :) = mean(acouMTB{i, :}(w : w+shift_size * freq-1, :));
        acouRTB_MA{i, :}(w, :) = mean(acouRTB{i, :}(w : w+shift_size * freq-1, :));
        piezLTB_MA{i, :}(w, :) = mean(piezLTB{i, :}(w : w+shift_size * freq-1, :));
        piezMTB_MA{i, :}(w, :) = mean(piezMTB{i, :}(w : w+shift_size * freq-1, :));
        piezRTB_MA{i, :}(w, :) = mean(piezRTB{i, :}(w : w+shift_size * freq-1, :));
        IMUaccNetTB_MA{i, :}(w, :) = mean(IMUaccNetTB{i, :}(w : w+shift_size * freq-1, :));
        IMUgyrNetTB_MA{i, :}(w, :) = mean(IMUgyrNetTB{i, :}(w : w+shift_size * freq-1, :));
        
        forcTE_MA{i, :}(w, :) = mean(forcTE{i, :}(w : w+shift_size * freq-1, :));
        acouLTE_MA{i, :}(w, :) = mean(acouLTE{i, :}(w : w+shift_size * freq-1, :));
        acouMTE_MA{i, :}(w, :) = mean(acouMTE{i, :}(w : w+shift_size * freq-1, :));
        acouRTE_MA{i, :}(w, :) = mean(acouRTE{i, :}(w : w+shift_size * freq-1, :));
        piezLTE_MA{i, :}(w, :) = mean(piezLTE{i, :}(w : w+shift_size * freq-1, :));
        piezMTE_MA{i, :}(w, :) = mean(piezMTE{i, :}(w : w+shift_size * freq-1, :));
        piezRTE_MA{i, :}(w, :) = mean(piezRTE{i, :}(w : w+shift_size * freq-1, :));
        IMUaccNetTE_MA{i, :}(w, :) = mean(IMUaccNetTE{i, :}(w : w+shift_size * freq-1, :));
        IMUgyrNetTE_MA{i, :}(w, :) = mean(IMUgyrNetTE{i, :}(w : w+shift_size * freq-1, :));

        sensT_MA{i, :}(w, :) = mean(sensT{i, :}(w : w+shift_size * freq-1, :));        
    end

    % sensation thresholding
    sensT_MA_thresh{i, :} = sensT_MA{i, :};
    sensT_MA_thresh{i, :}(sensT_MA_thresh{i, :} < shift_thresh) = 0;

    % binary masks
    sensT_MA_B{i, :} = double(logical(sensT_MA{i, :}));
    sensT_MA_threshB{i, :} = double(logical(sensT_MA_thresh{i, :}));

    sensT_MALoss(i, 1) = sum(sensT{i, :});
    sensT_MALoss(i, 2) = sum(sensT_MA{i, :});
    sensT_MALoss(i, 3) = sum(sensT_MA_thresh{i, :});
    sensT_MALoss(i, 4) = sum(sensT_MA_B{i, :});
    sensT_MALoss(i, 5) = sum(sensT_MA_threshB{i, :});
    
    % the whole sensation and its subset of button presses
    sensT_MA_mtx{i, :} = [tmp_duraT_MA, sensT_MA{i, :}];
    sensT_MA_idxP{i, :} = sensT_MA{i, :}==1;
    sensT_MA_mtxP{i, :} = sensT_MA_mtx{i, :}(sensT_MA_idxP{i, :}, :);

    sensT_MA_thresh_mtx{i, :} = [tmp_duraT_MA, sensT_MA_thresh{i, :}];
    sensT_MA_thresh_idxP{i, :} = sensT_MA_thresh{i, :}==1;
    sensT_MA_thresh_mtxP{i, :} = sensT_MA_thresh_mtx{i, :}(sensT_MA_thresh_idxP{i, :}, :);

    sensT_MA_threshB_mtx{i, :} = [tmp_duraT_MA, sensT_MA_threshB{i, :}];
    sensT_MA_threshB_idxP{i, :} = sensT_MA_threshB{i, :}==1;
    sensT_MA_threshB_mtxP{i, :} = sensT_MA_threshB_mtx{i, :}(sensT_MA_threshB_idxP{i, :}, :);

    % -------------------------------------------------------------------
    % Plotting trimmed, filtered signals and moving average signals
    % -------------------------------------------------------------------
    % F1 - force senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, forcT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(forcT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, forcTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(forcTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, forcTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(forcTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (forcTE{i, :} - forcTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(forcTE{i, :} - forcTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, forcTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(forcTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, forcTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(forcTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (forcTE_MA{i, :} - forcTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(forcTE_MA{i, :} - forcTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('forcT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F2 - acouL senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, acouLT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouLT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, acouLTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouLTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, acouLTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouLTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (acouLTE{i, :} - acouLTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(acouLTE{i, :} - acouLTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, acouLTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouLTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, acouLTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouLTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (acouLTE_MA{i, :} - acouLTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(acouLTE_MA{i, :} - acouLTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouLT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F3 - acouM senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, acouMT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouMT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, acouMTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouMTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, acouMTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouMTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (acouMTE{i, :} - acouMTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(acouMTE{i, :} - acouMTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, acouMTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouMTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, acouMTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouMTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (acouMTE_MA{i, :} - acouMTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(acouMTE_MA{i, :} - acouMTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouMT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F4 - acouR senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, acouRT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouRT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, acouRTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouRTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, acouRTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouRTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (acouRTE{i, :} - acouRTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(acouRTE{i, :} - acouRTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, acouRTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouRTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, acouRTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(acouRTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (acouRTE_MA{i, :} - acouRTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(acouRTE_MA{i, :} - acouRTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('acouRT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F5 - piezL senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, piezLT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezLT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, piezLTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezLTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, piezLTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezLTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (piezLTE{i, :} - piezLTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(piezLTE{i, :} - piezLTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, piezLTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezLTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, piezLTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezLTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (piezLTE_MA{i, :} - piezLTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(piezLTE_MA{i, :} - piezLTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezLT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F6 - piezM senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, piezMT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezMT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, piezMTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezMTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, piezMTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezMTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (piezMTE{i, :} - piezMTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(piezMTE{i, :} - piezMTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, piezMTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezMTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, piezMTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezMTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (piezMTE_MA{i, :} - piezMTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(piezMTE_MA{i, :} - piezMTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezMT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F7 - piezR senor 
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, piezRT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezRT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, piezRTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezRTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, piezRTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezRTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (piezRTE{i, :} - piezRTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(piezRTE{i, :} - piezRTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, piezRTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezRTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, piezRTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(piezRTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (piezRTE_MA{i, :} - piezRTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(piezRTE_MA{i, :} - piezRTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('piezRT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F8 - IMU acc
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, IMUaccNetT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, IMUaccNetTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, IMUaccNetTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (IMUaccNetTE{i, :} - IMUaccNetTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetTE{i, :} - IMUaccNetTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, IMUaccNetTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, IMUaccNetTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (IMUaccNetTE_MA{i, :} - IMUaccNetTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(IMUaccNetTE_MA{i, :} - IMUaccNetTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUaccNetT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F9 - IMU gyr
    figure
    tiledlayout(7, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, IMUgyrNetT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, IMUgyrNetTB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT butterworth');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, IMUgyrNetTE{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetTE{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT, (IMUgyrNetTE{i, :} - IMUgyrNetTB{i, :}), 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetTE{i, :} - IMUgyrNetTB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT diff - butterworth vs ellipic');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, IMUgyrNetTB_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT butterworth moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, IMUgyrNetTE_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetTE_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT ellipic moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, (IMUgyrNetTE_MA{i, :} - IMUgyrNetTB_MA{i, :}), 'LineWidth', line_width); 
    ylim([0, max(IMUgyrNetTE_MA{i, :} - IMUgyrNetTB_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('IMUgyrNetT diff - butterworth MA vs ellipic MA');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    % F10 - sensation
    figure
    tiledlayout(5, 1, 'Padding', 'tight', 'TileSpacing', 'none');

    nexttile
    hold on;
    plot(tmp_duraT, sensT{i, :}, 'LineWidth', line_width); 
    ylim([0, max(sensT{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('sensT');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, sensT_MA{i, :}, 'LineWidth', line_width); 
    ylim([0, max(sensT_MA{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('sensT moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, sensT_MA_thresh{i, :}, 'LineWidth', line_width); 
    ylim([0, max(sensT_MA_thresh{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('sensT moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, sensT_MA_B{i, :}, 'LineWidth', line_width); 
    ylim([0, max(sensT_MA_B{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('sensT moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_duraT_MA, sensT_MA_threshB{i, :}, 'LineWidth', line_width); 
    ylim([0, max(sensT_MA_threshB{i, :})*ylim_multiplier]);
    hold off;

    lgd = legend('sensT moving avg');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','east')
    legend boxoff;
    % -------------------------------------------------------------------------

    clear tmp_mtx tmp_file tmp_len tmp_lenT tmp_lenT_MA tmp_dura tmp_duraT tmp_duraT_MA

% end of loop for data files
end

% save organized data matrices
cd([fdir '\' data_processed]);
save(['Preproc01_' participant '_' data_category '_filtering_movingAvg.mat'], ...
      'forc', 'acouL', 'acouM', 'acouR', 'piezL', 'piezM', 'piezR', 'sens', 'IMUacc', 'IMUgyr', 'IMUaccNet', 'IMUgyrNet', ...
      'forcT', 'acouLT', 'acouMT', 'acouRT', 'piezLT', 'piezMT', 'piezRT', 'sensT', 'IMUaccT', 'IMUgyrT', 'IMUaccNetT', 'IMUgyrNetT', ...
      'forcTB', 'acouLTB', 'acouMTB', 'acouRTB', 'piezLTB', 'piezMTB', 'piezRTB', 'IMUaccNetTB', 'IMUgyrNetTB', ...
      'forcTE', 'acouLTE', 'acouMTE', 'acouRTE', 'piezLTE', 'piezMTE', 'piezRTE', 'IMUaccNetTE', 'IMUgyrNetTE', ...
      'forcTB_MA', 'acouLTB_MA', 'acouMTB_MA', 'acouRTB_MA', 'piezLTB_MA', 'piezMTB_MA', 'piezRTB_MA', 'IMUaccNetTB_MA', 'IMUgyrNetTB_MA', ...
      'forcTE_MA', 'acouLTE_MA', 'acouMTE_MA', 'acouRTE_MA', 'piezLTE_MA', 'piezMTE_MA', 'piezRTE_MA', 'IMUaccNetTE_MA', 'IMUgyrNetTE_MA', ...
      'sensT_MA', 'sensT_MA_B', 'sensT_MA_thresh', 'sensT_MA_threshB', 'sensT_MALoss', ...
      'sensT_mtx', 'sensT_idxP', 'sensT_mtxP', ...
      'sensT_MA_mtx', 'sensT_MA_idxP', 'sensT_MA_mtxP', ...
      'sensT_MA_thresh_mtx', 'sensT_MA_thresh_idxP', 'sensT_MA_thresh_mtxP', ...
      'sensT_MA_threshB_mtx', 'sensT_MA_threshB_idxP', 'sensT_MA_threshB_mtxP', ...
      '-v7.3' );
cd(curr_dir)













