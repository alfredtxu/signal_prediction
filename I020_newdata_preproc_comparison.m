% Channel sequence
% 1 A0---Force sensor
% 2 A1---piezo sensor(right)
% 3 A2---piezo sensor(middle)
% 4 A3---Acoustic sensor (middle)
% 5 A4---Acoustic sensor (left)
% 6 A5---Acoustic sensor (right)
% 7 A6---piezo sensor(left)
% 8-Sensation
% 9-16/17 IMU
% ---------------------------------------------------------------------------------------

clc
clear
close all

% HOME / OFFICE
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% curr_dir = pwd;
cd(curr_dir);

%% Pre-setting
% Frequency of sensor / sensation sampling rate (Hz)
% old yellow belt: 1024 Hz
% new yellow belt: 512 Hz 
% new pink belt: 400 Hz
freq = 400;

% the number of channels and FM sensors
num_channel = 16;
num_FMsensors = 6;

% Using the concept of slope, convert a digital signal back to an analog value. 
% ADC resolution is 12 bit
% ref voltage: 3.3 volt 
% ADC value (12-bit)
maxADC = 2^12-1;
volt = 3.3;
slope = volt / maxADC;

% Trim settings
% * Removal period in second
trimS = 30;
trimE = 30;

%% Filter design
% 1. lowpass filter
% 2. bandpass filter
% 3. Moving Average Filter
% 4. Savitzky-Golay Filter 
% 5. Wavelet Transform
% 6. Empirical Mode Decomposition (EMD):
%% Filter design - bandpass / lowpass filter
% filter order 
% * for bandpass filter is twice the value of 1st parameter
filter_order = 10;

% > band-pass filter with a passband of 1-30Hz disigned as fetal movement
boundL_FM = 1;
boundH_FM = 30;

% > band-pass filter with a passband of 1-10Hz designed as the IMU data
boundL_IMU = 1;
boundH_IMU = 10;

% > band-pass filter with a passband of 10Hz designed as the force sensor
boundH_force = 10;

% Transfer function-based desing
% Zero-Pole-Gain-based design
% Convert zero-pole-gain filter parameters to second-order sections form
[b_FM,a_FM] = butter(filter_order/2,[boundL_FM boundH_FM]/(freq/2),'bandpass');
[z_FM,p_FM,k_FM] = butter(filter_order/2,[boundL_FM boundH_FM]/(freq/2),'bandpass');
[sos_FM,g_FM] = zp2sos(z_FM,p_FM,k_FM);

[b_IMU,a_IMU] = butter(filter_order/2,[boundL_IMU boundH_IMU]/(freq/2),'bandpass');
[z_IMU,p_IMU,k_IMU] = butter(filter_order/2,[boundL_IMU boundH_IMU]/(freq/2),'bandpass');
[sos_IMU,g_IMU] = zp2sos(z_IMU,p_IMU,k_IMU);

% Low-pass filter
% * This filter is used for the force sensor data only
% Convert zero-pole-gain filter parameters to second-order sections form
[z_force,p_force,k_force] = butter(filter_order,boundH_force/(freq/2),'low');
[sos_force,g_force] = zp2sos(z_force,p_force,k_force);
% --------------------------------------------------------------------------------------------------

% * This notch filter is used for removing noise due to SD card writing (base frequency 32HZ)
% IIR notch filter: normalized location of the notch
% Q factor
% bandwidth at -3dB level
bound_notch = 32;
w0 = bound_notch/(freq/2);
q = 35;
notch_bw = w0/q;
[b_notch,a_notch] = iirnotch(w0,notch_bw); 


%% DATA LOADING
fprintf('Loading data files ...\n');

% data file directory
% HOME
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

% OFFICE
% fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

% ** set current loading data portion
% group the data matrice by individuals
% participant_list: 'rb', 'jg', 'am', 'cr', '005', '006', '007', 
% data category: F - focus; D - day; N - night
data_folder = 'b_mat_003AM';
data_processed = 'c_processed_003AM';
data_category = 'night';
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
    sens{i, :} = bitshift(tmp_mtx(:, 8), -8);
    IMUacc{i, :} = tmp_mtx(:, 9:11) * slope;
    IMUgyr{i, :} = tmp_mtx(:, 12:14) * slope;
    IMUaccNet{i, :} = sqrt(sum(IMUacc{i, :}.^2,2));
    IMUgyrNet{i, :} = sqrt(sum(IMUgyr{i, :}.^2,2));

    % assort the sensor data - trimmed data from both ends
    forc_T{i, :} = forc{i, :}((trimS*freq + 1):(end-trimE*freq));
    acouL_T{i, :} = acouL{i, :}((trimS*freq + 1):(end-trimE*freq));
    acouM_T{i, :} = acouM{i, :}((trimS*freq + 1):(end-trimE*freq));
    acouR_T{i, :} = acouR{i, :}((trimS*freq + 1):(end-trimE*freq));
    piezL_T{i, :} = piezL{i, :}((trimS*freq + 1):(end-trimE*freq));
    piezM_T{i, :} = piezM{i, :}((trimS*freq + 1):(end-trimE*freq));
    piezR_T{i, :} = piezR{i, :}((trimS*freq + 1):(end-trimE*freq));
    sens_T{i, :} = sens{i, :}((trimS*freq + 1):(end-trimE*freq));
    IMUacc_T{i, :} = IMUacc{i, :}((trimS*freq + 1):(end-trimE*freq), :);
    IMUgyr_T{i, :} = IMUgyr{i, :}((trimS*freq + 1):(end-trimE*freq), :);
    IMUaccNet_T{i, :} = IMUaccNet{i, :}((trimS*freq + 1):(end-trimE*freq));
    IMUgyrNet_T{i, :} = IMUgyrNet{i, :}((trimS*freq + 1):(end-trimE*freq));

    % Bandpass & Low-pass filtering applied on trimmed data
    forc_TF{i, :} = filtfilt(sos_force, g_force, forc_T{i, :});
    acouL_TF{i, :} = filtfilt(sos_FM, g_FM, acouL_T{i, :});
    acouM_TF{i, :} = filtfilt(sos_FM, g_FM, acouM_T{i, :});
    acouR_TF{i, :} = filtfilt(sos_FM, g_FM, acouR_T{i, :});
    piezL_TF{i, :} = filtfilt(sos_FM, g_FM, piezL_T{i, :});
    piezM_TF{i, :} = filtfilt(sos_FM, g_FM, piezM_T{i, :});
    piezR_TF{i, :} = filtfilt(sos_FM, g_FM, piezR_T{i, :});
    IMUaccNet_TF{i, :} = filtfilt(sos_IMU, g_IMU, IMUaccNet_T{i, :});
    IMUgyrNet_TF{i, :} = filtfilt(sos_IMU, g_IMU, IMUgyrNet_T{i, :});

    clear tmp_mtx tmp_file
    
end

% save organized data matrices
cd([fdir '\' data_processed]);
save(['FM01_' data_folder '_' data_category '_preproc.mat'], ...
      'forc', 'acouL', 'acouM', 'acouR', 'piezL', 'piezM', 'piezR', 'sens', 'IMUacc', 'IMUgyr', 'IMUaccNet', 'IMUgyrNet', ...
      'forc_T', 'acouL_T', 'acouM_T', 'acouR_T', 'piezL_T', 'piezM_T', 'piezR_T', 'sens_T', 'IMUacc_T', 'IMUgyr_T', 'IMUaccNet_T', 'IMUgyrNet_T', ...
      'forc_TF', 'acouL_TF', 'acouM_TF', 'acouR_TF', 'piezL_TF', 'piezM_TF', 'piezR_TF', 'IMUaccNet_TF', 'IMUgyrNet_TF', ...
      '-v7.3' );
cd(curr_dir)













