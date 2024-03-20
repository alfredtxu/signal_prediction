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

curr_dir=pwd;
cd(curr_dir);

% Add paths of the folders with all necessary function files
% SP_function_files: signal processing algorithms
% ML_function_files: machine learning algorithms
% Learned_models: learned models
% matplotlib colormaps: matplotlib_function_files
addpath(genpath('SP_function_files')) 
addpath(genpath('ML_function_files')) 
addpath(genpath('Learned_models')) 
addpath(genpath('matplotlib_function_files')) 
addpath(genpath('z10_olddata_mat_raw')) 
addpath(genpath('z11_olddata_mat_preproc')) 
addpath(genpath('z12_olddata_mat_proc'))
addpath(genpath('z13_olddata_mat_proc_revision'))
% -------------------------------------------------------------------------

% Define known parameters
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
fres_sensor=1024; 
fres_sensation=1024; 
num_channel=8;    
num_FMsensors=6; 

% ADC resolution is 12 bit
ADC_resolution=12;
maxADC_value=2^ADC_resolution-1;

% 3.3 volt corresponds to 4095 ADC value (12-bit)
volt=3.3;
slope=volt / maxADC_value;

% Maximum value of the test data
maxTestData_value=256;

% Trim settings
% * Removal period in second
trimS=30;
trimE=30;
% -------------------------------------------------------------------------

% Filter design
filter_order=10;

% Bandpass filter
% > band-pass filter with a passband of 1-30Hz disigned as fetal movement
boundL_FM=1;
boundH_FM=30;

% > band-pass filter with a passband of 1-10Hz designed as the IMU data
boundL_IMU=1;
boundH_IMU=10;

% > band-pass filter with a passband of 10Hz designed as the force sensor
boundH_force=10;

% * This notch filter is used for removing noise due to SD card writing (base frequency 32HZ)
% IIR notch filter: normalized location of the notch
% Q factor
% bandwidth at -3dB level
bound_notch=32;
w0=bound_notch/(fres_sensor/2);
q=35;
notch_bw=w0/q;
[b_notch,a_notch]=iirnotch(w0,notch_bw);

% * filter order for bandpass filter is twice the value of 1st parameter
% Transfer function-based desing
% Zero-Pole-Gain-based design
% Convert zero-pole-gain filter parameters to second-order sections form
[b_FM,a_FM]=butter(filter_order/2,[boundL_FM boundH_FM]/(fres_sensor/2),'bandpass');
[z_FM,p_FM,k_FM]=butter(filter_order/2,[boundL_FM boundH_FM]/(fres_sensor/2),'bandpass');
[sos_FM,g_FM]=zp2sos(z_FM,p_FM,k_FM);

[b_IMU,a_IMU]=butter(filter_order/2,[boundL_IMU boundH_IMU]/(fres_sensor/2),'bandpass');
[z_IMU,p_IMU,k_IMU]=butter(filter_order/2,[boundL_IMU boundH_IMU]/(fres_sensor/2),'bandpass');
[sos_IMU,g_IMU]=zp2sos(z_IMU,p_IMU,k_IMU);

% Low-pass filter
% * This filter is used for the force sensor data only
% Convert zero-pole-gain filter parameters to second-order sections form
[z_force,p_force,k_force]=butter(filter_order,boundH_force/(fres_sensor/2),'low');
[sos_force,g_force]=zp2sos(z_force,p_force,k_force);
% --------------------------------------------------------------------------------------------------

%% DATA LOADING
fprintf('Loading data files ...\n');

% ** set current loading data portion
% group the data matrice by individuals
participant='S5';
switch participant
    case 'S1'
        [data_fname, data_pname]=uigetfile('S1*.mat','Select the data files','MultiSelect','on'); 
        num_dfile=length(data_fname);
        addpath(data_pname);
    case 'S2'
        [data_fname, data_pname]=uigetfile('S2*.mat','Select the data files','MultiSelect','on'); 
        num_dfile=length(data_fname);
        addpath(data_pname);
    case 'S3'
        [data_fname, data_pname]=uigetfile('S3*.mat','Select the data files','MultiSelect','on'); 
        num_dfile=length(data_fname);
        addpath(data_pname);
    case 'S4'
        [data_fname, data_pname]=uigetfile('S4*.mat','Select the data files','MultiSelect','on'); 
        num_dfile=length(data_fname);
        addpath(data_pname);
    case 'S5'
        [data_fname, data_pname]=uigetfile('S5*.mat','Select the data files','MultiSelect','on'); 
        num_dfile=length(data_fname);
        addpath(data_pname);
    otherwise
        disp('Error: the number of sensor types is beyond pre-setting...')
end
% --------------------------------------------------------------------------------------------------

% Loading the data files
for i=1 : num_dfile

    fprintf('Current data file <%s>: %d - %s ...\n', participant, i, fullfile(data_pname, data_fname{i}));

    tmp_data_suite=load(data_fname{i},'data_var');
    tmp_sensors_SD1=tmp_data_suite.data_var{1};
    tmp_sensation_SD1=tmp_data_suite.data_var{2};
    tmp_sensors_SD2=tmp_data_suite.data_var{3};
    tmp_sensation_SD2=tmp_data_suite.data_var{4};

    % Extracting sensor data
    % Force sensor data to measure tightness of the belt
    % Accelerometer data to measure maternal body movement
    % Left / Right Piezo-plate data to measure abdomenal vibration
    % Left / Right Acoustic sensor data to measure abdomenal vibration
    % Left / Right Accelerometer data to measure abdomenal vibration
    tmp_forc=tmp_sensors_SD1(:, 1);
    tmp_IMUacceNet=sqrt(sum(tmp_sensors_SD1(:, 5:7).^2,2));  
    tmp_acceLNet=sqrt(sum(tmp_sensors_SD2(:, 5:7).^2,2)); 
    tmp_acceRNet=sqrt(sum(tmp_sensors_SD2(:, 2:4).^2,2)); 
    tmp_acouL=tmp_sensors_SD1(:, 4); 
    tmp_acouR=tmp_sensors_SD2(:, 1); 
    tmp_piezL=tmp_sensors_SD1(:, 2); 
    tmp_piezR=tmp_sensors_SD1(:, 3); 

    % trim each raw section of data (individual data file)
    forcT{i, :} = tmp_forc((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    IMUacceNetT{i, :} = tmp_IMUacceNet((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acceLNetT{i, :} = tmp_acceLNet((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acceRNetT{i, :} = tmp_acceRNet((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acouLT{i, :} = tmp_acouL((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acouRT{i, :} = tmp_acouR((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    piezLT{i, :} = tmp_piezL((trimS*fres_sensor + 1):(end-trimE*fres_sensor)); 
    piezRT{i, :} = tmp_piezR((trimS*fres_sensor + 1):(end-trimE*fres_sensor));

    sens1T{i,:} = tmp_sensation_SD1((trimS*fres_sensation + 1):(end-trimE*fres_sensation));
    sens2T{i,:} = tmp_sensation_SD2((trimS*fres_sensation + 1):(end-trimE*fres_sensation));
    
    % --------------------------------------------------------------------------------------------------
    % Bandpass & Low-pass filtering
    forcFil{i,:}=filtfilt(sos_force, g_force, tmp_forc);
    IMUacceNetFil{i,:}=filtfilt(sos_IMU, g_IMU, tmp_IMUacceNet);
    acceLNetFil{i,:}=filtfilt(sos_FM, g_FM, tmp_acceLNet);
    acceRNetFil{i,:}=filtfilt(sos_FM, g_FM, tmp_acceRNet);
    acouLFil{i,:}=filtfilt(sos_FM, g_FM, tmp_acouL);
    acouRFil{i,:}=filtfilt(sos_FM, g_FM, tmp_acouR);
    piezLFil{i,:}=filtfilt(sos_FM, g_FM, tmp_piezL);
    piezRFil{i,:}=filtfilt(sos_FM, g_FM, tmp_piezR);
    
    % trim each raw section of data (individual data file)
    forcFilT{i,:}=forcFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    IMUacceNetFilT{i,:}=IMUacceNetFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acceLNetFilT{i,:}=acceLNetFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acceRNetFilT{i,:}=acceRNetFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acouLFilT{i,:}=acouLFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    acouRFilT{i,:}=acouRFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));
    piezLFilT{i,:}=piezLFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor)); 
    piezRFilT{i,:}=piezRFil{i,:}((trimS*fres_sensor + 1):(end-trimE*fres_sensor));

    % Equalizing the length of SD1 and SD2 data sets
    % * min_length_sensation & min_length_sensor are equal except for the case of data sets taken with DAQ version 1.0
    minL_sens = min(length(sens1T{i,:}),length(sens2T{i,:}));
    minL_FM = min(length(acouLFilT{i,:}),length(acouRFilT{i,:}));

    forcTEq{i,:}=forcT{i,:}(1:minL_FM);
    IMUacceNetTEq{i,:}=IMUacceNetT{i,:}(1:minL_FM);
    acceLNetTEq{i,:}=acceLNetT{i,:}(1:minL_FM);
    acceRNetTEq{i,:}=acceRNetT{i,:}(1:minL_FM);
    acouLTEq{i,:}=acouLT{i,:}(1:minL_FM);
    acouRTEq{i,:}=acouRT{i,:}(1:minL_FM);
    piezLTEq{i,:}=piezLT{i,:}(1:minL_FM);
    piezRTEq{i,:}=piezRT{i,:}(1:minL_FM);

    sens1TEq{i,:}=sens1T{i,:}(1:minL_sens);
    sens2TEq{i,:}=sens2T{i,:}(1:minL_sens);

    forcFilTEq{i,:}=forcFilT{i,:}(1:minL_FM);
    IMUacceNetFilTEq{i,:}=IMUacceNetFilT{i,:}(1:minL_FM);
    acceLNetFilTEq{i,:}=acceLNetFilT{i,:}(1:minL_FM);
    acceRNetFilTEq{i,:}=acceRNetFilT{i,:}(1:minL_FM);
    acouLFilTEq{i,:}=acouLFilT{i,:}(1:minL_FM);
    acouRFilTEq{i,:}=acouRFilT{i,:}(1:minL_FM);
    piezLFilTEq{i,:}=piezLFilT{i,:}(1:minL_FM);
    piezRFilTEq{i,:}=piezRFilT{i,:}(1:minL_FM);

    % left them out because the length from two SD cards were varied.
    % all_dataT_filt(i,1,:)=forcT_filt{i,:};
    % all_dataT_filt(i,2,:)=IMUacceNetT_filt{i,:};
    % all_dataT_filt(i,3,:)=acceLNetT_filt{i,:};
    % all_dataT_filt(i,4,:)=acceRNetT_filt{i,:};
    % all_dataT_filt(i,5,:)=acouLT_filt{i,:};
    % all_dataT_filt(i,6,:)=acouRT_filt{i,:};
    % all_dataT_filt(i,7,:)=piezLT_filt{i,:};
    % all_dataT_filt(i,8,:)=piezRT_filt{i,:};
    % all_dataT_filt(i,9,:)=sens1T{i,:};
    % all_dataT_filt(i,10,:)=sens2T{i,:};

    % time vector for individual data files
    time_vecTEq{i,:}=((0 : (length(forcFilTEq{i,:})-1)) / fres_sensor)';
    
    tmp_sens1TEq_mtx=[time_vecTEq{i,:}, sens1TEq{i,:}];
    tmp_sens1TEq_idxP=sens1TEq{i,:}>0;
    tmp_sens1T_mtxP=tmp_sens1TEq_mtx(tmp_sens1TEq_idxP,:);

    sens1TEq_mtxP{i,:}=tmp_sens1T_mtxP;

    clear tmp_data_suite tmp_sensors_SD1 tmp_sensation_SD1 tmp_sensors_SD2 tmp_sensation_SD2 ...
          tmp_forc tmp_IMUacceNet tmp_acceLNet tmp_acceRNet tmp_acouL tmp_acouR tmp_piezL tmp_piezR ...
          tmp_sens1TEq_mtx tmp_sens1TEq_idxP tmp_sens1T_mtxP

end

% concatenate all the data files into one
forcFilTEq_cat=cat(1,forcFilTEq{:});
IMUacceNetFilTEq_cat=cat(1,IMUacceNetFilTEq{:});
acceLNetFilTEq_cat=cat(1,acceLNetFilTEq{:});
acceRNetFilTEq_cat=cat(1,acceRNetFilTEq{:});
acouLFilTEq_cat=cat(1,acouLFilTEq{:});
acouRFilTEq_cat=cat(1,acouRFilTEq{:});
piezLFilTEq_cat=cat(1,piezLFilTEq{:});
piezRFilTEq_cat=cat(1,piezRFilTEq{:});
sens1TEq_cat=cat(1,sens1TEq{:});
sens2TEq_cat=cat(1,sens2TEq{:});

% save organized data matrices
cd([curr_dir '/z12_olddata_mat_preproc'])
save(['sensor_data_suite_' participant '_preproc.mat'], ...
      'forcT', 'IMUacceNetT', 'acceLNetT', 'acceRNetT', 'acouLT', 'acouRT', 'piezLT', 'piezRT', ...
      'forcTEq', 'IMUacceNetTEq', 'acceLNetTEq', 'acceRNetTEq', 'acouLTEq', 'acouRTEq', 'piezLTEq', 'piezRTEq', ...
      'forcFil', 'IMUacceNetFil', 'acceLNetFil', 'acceRNetFil', 'acouLFil', 'acouRFil', 'piezLFil', 'piezRFil', ...
      'forcFilT', 'IMUacceNetFilT', 'acceLNetFilT', 'acceRNetFilT', 'acouLFilT', 'acouRFilT', 'piezLFilT', 'piezRFilT', ...
      'forcFilTEq', 'IMUacceNetFilTEq', 'acceLNetFilTEq', 'acceRNetFilTEq', 'acouLFilTEq', 'acouRFilTEq', 'piezLFilTEq', 'piezRFilTEq', ...
      'forcFilTEq_cat', 'IMUacceNetFilTEq_cat', 'acceLNetFilTEq_cat', 'acceRNetFilTEq_cat', 'acouLFilTEq_cat', 'acouRFilTEq_cat', 'piezLFilTEq_cat', 'piezRFilTEq_cat',  ...
      'sens1T', 'sens2T', 'sens1TEq', 'sens2TEq', 'sens1TEq_cat','sens2TEq_cat', 'time_vecTEq', 'sens1TEq_mtxP', ...
     '-v7.3' );
cd(curr_dir)













