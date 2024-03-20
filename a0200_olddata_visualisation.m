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
% ------------------------------------------------------------------------------------------------

time_vecT=(0 : (length(forcT)-1)) / Fs_sensor;

sensationT_mtx=[time_vecT', sensation_dataT];
sensationT_idxP=sensation_dataT > 0;
sensation_dataTP=sensationT_mtx(sensationT_idxP,:);

%% PLOTTING PREPROCESSED DATA
% sensor and sensation data
% * for plotting use: x-axis is 10% longer than the actual wearing time
duraT=ceil(max(time_vecT))*1.1;
maxSenVT_filt=max(max(sensor_dataT_filt_reorder))*1.1;
maxSensaVT=1.5;

% percetile
percT_filt=99.995;

% pre-processing operations
pproc='Trimmed & Filtered';

% the accumulated time period
% sensor and sensation data
figure
for v=1:num_channels

    if v==1
        subplot(num_channels,1,v)
        plot(sensation_dataTP(:,1), sensation_dataTP(:,2), 'r.');
        title('Sensation (Trimmed)')
        xlim([0 duraT])
        ylim([0 maxSensaVT])
    else
        subplot(num_channels,1,v)
        plot(time_vecT, sensor_dataT_filt_reorder(:,v-1));
        
        tmp_p=prctile(sensor_dataT_filt_reorder(:,v-1),percT_filt);
        tmp_pidx=find(sensor_dataT_filt_reorder(:,v-1)>=tmp_p);
        tmp_yp=sensor_dataT_filt_reorder(tmp_pidx,v-1);
        
        hold on;
        plot(time_vecT(tmp_pidx),tmp_yp,'r.');
        hold off;

        title([sensor_list{v-1,:} ' (' pproc ')'])
        xlim([0 duraT])
        ylim([0 maxSensaVT])

        clear tmp_p tmp_pidx tmp_yp
    end
end