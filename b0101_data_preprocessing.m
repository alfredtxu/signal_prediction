% -------------------------------------------------------------------------
% SD1:
% Channel 1: Flexi force sensor
% Channel 2: Piezoelectric plate sensor 1 (Left belly)
% Channel 3: Piezoelectric plate sensor 2 (Right belly)
% Channel 4: Acoustic sensor 1 (Left belly)
% Channel 5-7: IMU data (Accelerometer)
% Channel 8: Maternal senstation
% 
% SD2:
% Channel 1: Accoustic sensor 2 (Right belly)
% Channel 2-4: Accelerometer 2 (Right belly)
% Channel 5-7: Accelerometer 1 (Left belly)
% Channel 8: Maternal sensation
% -------------------------------------------------------------------------
clc
clear
close all

curr_dir = pwd;
cd(curr_dir);

% Add paths of the folders with all necessary function files
addpath(genpath('SP_function_files')) % Holds the function related to signal processing algorithms
addpath(genpath('ML_function_files')) % Holds the function related to machine learning algorithms
addpath(genpath('Learned_models')) % Holds the saved learned models
addpath(genpath('matplotlib_function_files')) % Holds the custom functions for matplotlib colormaps

% Define known parameters
Fs_sensor    = 1024; % Frequency of sensor data sampling in Hz
Fs_sensation = 1024; % Frequency of sensation data sampling in Hz
n_sensor     = 8;    % Total number of sensors
n_FM_sensors = 6; % number of sensors detecting FM

% ** set current loading data portion
participant = 'S1';

% loading data
cd([curr_dir '\z1_data_mat_raw'])
load(['sensor_data_suite_' participant '.mat']);
cd(curr_dir)

%% DATA  PREPROCESSING -- FILTER & TRIMMING settings
% Filter design
filter_order = 10;

% Bandpass filter
% > band-pass filter with a passband of 1-30Hz disigned as fetal movement
boundL_FM = 1;
boundH_FM = 30;

% > band-pass filter with a passband of 1-10Hz designed as the IMU data
boundL_IMU = 1;
boundH_IMU = 10;

% > band-pass filter with a passband of 10Hz designed as the force sensor
boundH_force = 10;

% * This notch filter is used for removing noise due to SD card writing (base frequency 32HZ)
bound_notch = 32; 

% * filter order for bandpass filter is twice the value of 1st parameter
% Transfer function-based desing
% Zero-Pole-Gain-based design
% Convert zero-pole-gain filter parameters to second-order sections form
[b_FM,a_FM] = butter(filter_order/2,[boundL_FM boundH_FM]/(Fs_sensor/2),'bandpass');
[z_FM,p_FM,k_FM] = butter(filter_order/2,[boundL_FM boundH_FM]/(Fs_sensor/2),'bandpass'); 
[sos_FM,g_FM] = zp2sos(z_FM,p_FM,k_FM); 

[b_IMU,a_IMU] = butter(filter_order/2,[boundL_IMU boundH_IMU]/(Fs_sensor/2),'bandpass');
[z_IMU,p_IMU,k_IMU] = butter(filter_order/2,[boundL_IMU boundH_IMU]/(Fs_sensor/2),'bandpass');
[sos_IMU,g_IMU] = zp2sos(z_IMU,p_IMU,k_IMU);

% Low-pass filter
% * This filter is used for the force sensor data only
% Convert zero-pole-gain filter parameters to second-order sections form
[z_force,p_force,k_force] = butter(filter_order,boundH_force/(Fs_sensor/2),'low');
[sos_force,g_force] = zp2sos(z_force,p_force,k_force);

% IIR notch filter
% normalized location of the notch
% Q factor
% bandwidth at -3dB level
w0 = bound_notch/(Fs_sensor/2); 
q = 35;
notch_bw = w0/q; 
[b_notch,a_notch] = iirnotch(w0,notch_bw);

% Trim settings
% * Removal period in second
trim_start = 30; 
trim_end = 30;
% -----------------------------------------------------------------------------------------

%% DATA PREPROCESSING -- TRIMMING

%% DATA PREPROCESSING -- FILTERING & TRIMMING
fprintf('Preprocessing started: the section of filtering & trimming  ...\n');

for i = 1 : num_data_file

    fprintf('>>> Current data file in trimming: %d/%d ... \n', i, num_data_file)
    
    accelero_left_ed_filtered{i} = accelero_left_ed{i};
    accelero_right_ed_filtered{i} = accelero_right_ed{i};
    acoustic_left_filtered{i} = acoustic_left{i};
    acoustic_right_filtered{i} = acoustic_right{i};
    piezo_left_filtered{i} = piezo_left{i};
    piezo_right_filtered{i} = piezo_right{i};
    IMU_accelero_ed_filtered{i} = IMU_accelero_ed{i};
    force_data_ed_filtered{i} = force_data_ed{i};
    force_data_filtered{i} = force_data{i};

    % IIR notch filter for removing noise due to power supply glitch
    accelero_left_ed_filteredN{i} = filtfilt(b_notch, a_notch, accelero_left_ed_filtered{i});
    accelero_right_ed_filteredN{i} = filtfilt(b_notch, a_notch, accelero_right_ed_filtered{i});
    acoustic_left_filteredN{i}  = filtfilt(b_notch, a_notch, acoustic_left_filtered{i});
    acoustic_right_filteredN{i}  = filtfilt(b_notch, a_notch, acoustic_right_filtered{i});
    piezo_left_filteredN{i} = filtfilt(b_notch, a_notch, piezo_left_filtered{i});
    piezo_right_filteredN{i} = filtfilt(b_notch, a_notch, piezo_right_filtered{i});
    IMU_accelero_ed_filteredN{i}  = filtfilt(b_notch, a_notch, IMU_accelero_ed_filtered{i});
    force_data_ed_filteredN{i}    = filtfilt(b_notch, a_notch, force_data_ed_filtered{i});
    force_data_filteredN{i}    = filtfilt(b_notch, a_notch, force_data_filtered{i});
    
    % Bandpass & Low-pass filtering
    accelero_left_ed_filtered{i} = filtfilt(sos_FM, g_FM, accelero_left_ed_filtered{i});
    accelero_right_ed_filtered{i} = filtfilt(sos_FM, g_FM, accelero_right_ed_filtered{i});
    acoustic_left_filtered{i}  = filtfilt(sos_FM, g_FM, acoustic_left_filtered{i});
    acoustic_right_filtered{i}  = filtfilt(sos_FM, g_FM, acoustic_right_filtered{i});
    piezo_left_filtered{i} = filtfilt(sos_FM, g_FM, piezo_left_filtered{i});
    piezo_right_filtered{i} = filtfilt(sos_FM, g_FM, piezo_right_filtered{i});
    IMU_accelero_ed_filtered{i}    = filtfilt(sos_IMU,g_IMU,IMU_accelero_ed_filtered{i});
    force_data_ed_filtered{i}  = filtfilt(sos_force, g_force, force_data_ed_filtered{i});
    force_data_filtered{i}  = filtfilt(sos_force, g_force, force_data_filtered{i});

    accelero_left_ed_filteredN{i} = filtfilt(sos_FM, g_FM, accelero_left_ed_filteredN{i});
    accelero_right_ed_filteredN{i} = filtfilt(sos_FM, g_FM, accelero_right_ed_filteredN{i});
    acoustic_left_filteredN{i} = filtfilt(sos_FM, g_FM, acoustic_left_filteredN{i});
    acoustic_right_filteredN{i} = filtfilt(sos_FM, g_FM, acoustic_right_filteredN{i});
    piezo_left_filteredN{i} = filtfilt(sos_FM, g_FM, piezo_left_filteredN{i});
    piezo_right_filteredN{i} = filtfilt(sos_FM, g_FM, piezo_right_filteredN{i});
    IMU_accelero_ed_filteredN{i} = filtfilt(sos_IMU,g_IMU,IMU_accelero_ed_filteredN{i});
    force_data_ed_filteredN{i} = filtfilt(sos_force, g_force, force_data_ed_filteredN{i});
    force_data_filteredN{i} = filtfilt(sos_force, g_force, force_data_filteredN{i});

    % Trimming of raw data
    sensation_SD1_trimmed{i} = sensation_SD1{i}((trim_start*Fs_sensation + 1):(end-trim_end*Fs_sensation));
    sensation_SD2_trimmed{i} = sensation_SD2{i}((trim_start*Fs_sensation + 1):(end-trim_end*Fs_sensation));
    
    accelero_left_ed_trimmed{i} = accelero_left_ed{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    accelero_right_ed_trimmed{i} = accelero_right_ed{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    acoustic_left_trimmed{i} = acoustic_left{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    acoustic_right_trimmed{i} = acoustic_right{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    piezo_left_trimmed{i} = piezo_left{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    piezo_right_trimmed{i} = piezo_right{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    IMU_accelero_ed_trimmed{i} = IMU_accelero_ed{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    force_data_ed_trimmed{i} = force_data_ed{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    force_data_trimmed{i} = force_data{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));

    % Trimming of filtered data
    accelero_left_ed_filtered_trimmed{i} = accelero_left_ed_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    accelero_right_ed_filtered_trimmed{i} = accelero_right_ed_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    acoustic_left_filtered_trimmed{i} = acoustic_left_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    acoustic_right_filtered_trimmed{i} = acoustic_right_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    piezo_left_filtered_trimmed{i} = piezo_left_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    piezo_right_filtered_trimmed{i} = piezo_right_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    IMU_accelero_ed_filtered_trimmed{i} = IMU_accelero_ed_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    force_data_ed_filtered_trimmed{i} = force_data_ed_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    force_data_filtered_trimmed{i} = force_data_filtered{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    
    % Trimming of notch filtered data
    accelero_left_ed_filteredN_trimmed{i} = accelero_left_ed_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    accelero_right_ed_filteredN_trimmed{i} = accelero_right_ed_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    acoustic_left_filteredN_trimmed{i} = acoustic_left_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    acoustic_right_filteredN_trimmed{i} = acoustic_right_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    piezo_left_filteredN_trimmed{i} = piezo_left_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    piezo_right_filteredN_trimmed{i} = piezo_right_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    IMU_accelero_ed_filteredN_trimmed{i} = IMU_accelero_ed_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    force_data_ed_filteredN_trimmed{i} = force_data_ed_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));
    force_data_filteredN_trimmed{i} = force_data_filteredN{i}((trim_start*Fs_sensor + 1):(end-trim_end*Fs_sensor));

    % Equalizing the length of SD1 and SD2 data sets
    % *min_len_sensation & min_len_sensor are equal except for the data taken with DAQ v1.0
    min_len_sensation = min(length(sensation_SD1_trimmed{i}),length(sensation_SD2_trimmed{i}));
    min_len_sensor = min(length(acoustic_left_filtered{i}),length(acoustic_right_filtered{i}));
    
    sensation_SD1_trimmed{i} = sensation_SD1_trimmed{i}(1:min_len_sensation);
    sensation_SD2_trimmed{i} = sensation_SD2_trimmed{i}(1:min_len_sensation);

    accelero_left_ed_trimmed{i} = accelero_left_ed_trimmed{i}(1:min_len_sensor);
    accelero_right_ed_trimmed{i} = accelero_right_ed_trimmed{i}(1:min_len_sensor);
    acoustic_left_trimmed{i} = acoustic_left_trimmed{i}(1:min_len_sensor);
    acoustic_right_trimmed{i} = acoustic_right_trimmed{i}(1:min_len_sensor);
    piezo_left_trimmed{i} = piezo_left_trimmed{i}(1:min_len_sensor);
    piezo_right_trimmed{i} = piezo_right_trimmed{i}(1:min_len_sensor);
    IMU_accelero_ed_trimmed{i} = IMU_accelero_ed_trimmed{i}(1:min_len_sensor);
    force_data_ed_trimmed{i} = force_data_ed_trimmed{i}(1:min_len_sensor);
    force_data_trimmed{i} = force_data_trimmed{i}(1:min_len_sensor);

    accelero_left_ed_filtered_trimmed{i} = accelero_left_ed_filtered_trimmed{i}(1:min_len_sensor);
    accelero_right_ed_filtered_trimmed{i} = accelero_right_ed_filtered_trimmed{i}(1:min_len_sensor);
    acoustic_left_filtered_trimmed{i} = acoustic_left_filtered_trimmed{i}(1:min_len_sensor);
    acoustic_right_filtered_trimmed{i} = acoustic_right_filtered_trimmed{i}(1:min_len_sensor);
    piezo_left_filtered_trimmed{i} = piezo_left_filtered_trimmed{i}(1:min_len_sensor);
    piezo_right_filtered_trimmed{i} = piezo_right_filtered_trimmed{i}(1:min_len_sensor);
    IMU_accelero_ed_filtered_trimmed{i} = IMU_accelero_ed_filtered_trimmed{i}(1:min_len_sensor);
    force_data_ed_filtered_trimmed{i} = force_data_ed_filtered_trimmed{i}(1:min_len_sensor);
    force_data_filtered_trimmed{i} = force_data_filtered_trimmed{i}(1:min_len_sensor);

    accelero_left_ed_filteredN_trimmed{i} = accelero_left_ed_filteredN_trimmed{i}(1:min_len_sensor);
    accelero_right_ed_filteredN_trimmed{i} = accelero_right_ed_filteredN_trimmed{i}(1:min_len_sensor);
    acoustic_left_filteredN_trimmed{i} = acoustic_left_filteredN_trimmed{i}(1:min_len_sensor);
    acoustic_right_filteredN_trimmed{i} = acoustic_right_filteredN_trimmed{i}(1:min_len_sensor);
    piezo_left_filteredN_trimmed{i} = piezo_left_filteredN_trimmed{i}(1:min_len_sensor);
    piezo_right_filteredN_trimmed{i} = piezo_right_filteredN_trimmed{i}(1:min_len_sensor);
    IMU_accelero_ed_filteredN_trimmed{i} = IMU_accelero_ed_filteredN_trimmed{i}(1:min_len_sensor);
    force_data_ed_filteredN_trimmed{i} = force_data_ed_filteredN_trimmed{i}(1:min_len_sensor);
    force_data_filteredN_trimmed{i} = force_data_filteredN_trimmed{i}(1:min_len_sensor);
end

% General information extraction
% Extracting force sensor data during each recording session
force_data_abs = cell(1,num_data_file);
force_data_mean = zeros(num_data_file, 1);
force_signal_power = zeros(1, num_data_file);

% sample size in seconds
% sample_size = 30; 

for i = 1 : n_data_files
    % force_data_abs{i} = force_data_filtered_trimmed{i}(1:sample_size*Fs_sensor);
    force_data_abs{i} = abs(force_data_filtered_trimmed{i});
    force_data_mean(i) = mean(force_data_abs{i});
    force_signal_power(i) = sum(force_data_abs{i}.^2)/length(force_data_abs{i});
end

% Duration of each data file (in seconds)
data_period = zeros(num_data_file,1);
for i = 1:num_data_file
    data_period(i)= length(acoustic_left{i}) / Fs_sensor;
end

% Ending notification
fprintf('Preprocessing completed: the section of filtering & trimming  ...\n\n');

% save processed data
save('sensor_suite_data_preprocessed.mat', ...
    'accelero_left_ed_filtered',  ... 
    'accelero_right_ed_filtered',  ... 
    'acoustic_left_filtered',  ... 
    'acoustic_right_filtered',  ... 
    'piezo_left_filtered',  ... 
    'piezo_right_filtered',  ... 
    'IMU_accelero_ed_filtered',  ...   
    'force_data_ed_filtered',  ... 
    'force_data_filtered',  ... 
    'accelero_left_ed_filteredN',  ... 
    'accelero_right_ed_filteredN',  ... 
    'acoustic_left_filteredN',  ... 
    'acoustic_right_filteredN',  ... 
    'piezo_left_filteredN',  ... 
    'piezo_right_filteredN',  ... 
    'IMU_accelero_ed_filteredN',  ...   
    'force_data_ed_filteredN',  ... 
    'force_data_filteredN',  ... 
    'sensation_SD1_trimmed',  ... 
    'sensation_SD2_trimmed',  ... 
    'accelero_left_ed_trimmed',  ... 
    'accelero_right_ed_trimmed',  ... 
    'acoustic_left_trimmed',  ... 
    'acoustic_right_trimmed',  ... 
    'piezo_left_trimmed',  ... 
    'piezo_right_trimmed',  ... 
    'IMU_accelero_ed_trimmed',  ...   
    'force_data_ed_trimmed',  ... 
    'force_data_trimmed',  ... 
    'accelero_left_ed_filtered_trimmed',  ... 
    'accelero_right_ed_filtered_trimmed',  ... 
    'acoustic_left_filtered_trimmed',  ... 
    'acoustic_right_filtered_trimmed',  ... 
    'piezo_left_filtered_trimmed',  ... 
    'piezo_right_filtered_trimmed',  ... 
    'IMU_accelero_ed_filtered_trimmed',  ...   
    'force_data_ed_filtered_trimmed',  ... 
    'force_data_filtered_trimmed',  ... 
    'accelero_left_ed_filteredN_trimmed',  ... 
    'accelero_right_ed_filteredN_trimmed',  ... 
    'acoustic_left_filteredN_trimmed',  ... 
    'acoustic_right_filteredN_trimmed',  ... 
    'piezo_left_filteredN_trimmed',  ... 
    'piezo_right_filteredN_trimmed',  ... 
    'IMU_accelero_ed_filteredN_trimmed',  ...   
    'force_data_ed_filteredN_trimmed',  ... 
    'force_data_filteredN_trimmed',  ... 
    'force_data_abs', ...
    'force_data_mean', ...
    'forse_singal_power', ...
    'v7.3');

