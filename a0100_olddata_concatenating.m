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
% SP_function_files: signal processing algorithms
% ML_function_files: machine learning algorithms
% Learned_models: learned models
% matplotlib colormaps: matplotlib_function_files
addpath(genpath('SP_function_files')) 
addpath(genpath('ML_function_files')) 
addpath(genpath('Learned_models')) 
addpath(genpath('matplotlib_function_files')) 
addpath(genpath('z11_olddata_mat_raw')) 
addpath(genpath('z12_olddata_mat_preproc')) 
addpath(genpath('z13_olddata_mat_proc'))

% Define known parameters
% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
fres_sensor = 1024; 
fres_sensation = 1024; 
num_channel = 8;    
num_FMsensors = 6; 

%% DATA LOADING
fprintf('Loading data files ...\n');

% Locate and load selected data files
% Returns file names with the extension in a cell, and path name
[data_fname_p1, data_pname_p1] = uigetfile('S1*.mat','Select the data files','MultiSelect','on'); 
[data_fname_p2, data_pname_p2] = uigetfile('S2*.mat','Select the data files','MultiSelect','on'); 
[data_fname_p3, data_pname_p3] = uigetfile('S3*.mat','Select the data files','MultiSelect','on'); 
[data_fname_p4, data_pname_p4] = uigetfile('S4*.mat','Select the data files','MultiSelect','on'); 
[data_fname_p5, data_pname_p5] = uigetfile('S5*.mat','Select the data files','MultiSelect','on'); 

num_dfile_p1 = length(data_fname_p1);
num_dfile_p2 = length(data_fname_p2);
num_dfile_p3 = length(data_fname_p3);
num_dfile_p4 = length(data_fname_p4);
num_dfile_p5 = length(data_fname_p5);

% current settings: data_pname_p1-p5 are the same directory
data_pname = data_pname_p1;
addpath(data_pname);
% ------------------------------------------------------------------------------------------------

% ** set current loading data portion
% group the data matrice by individuals
participant = 'S5';
switch participant
    case 'S1'
        data_fname = data_fname_p1;
        num_dfile = num_dfile_p1;
    case 'S2'
        data_fname = data_fname_p2;
        num_dfile = num_dfile_p2;
    case 'S3'
        data_fname = data_fname_p3;
        num_dfile = num_dfile_p3;
    case 'S4'
        data_fname = data_fname_p4;
        num_dfile = num_dfile_p4;
    case 'S5'
        data_fname = data_fname_p5;
        num_dfile = num_dfile_p5;
    otherwise
        disp('Error: the number of sensor types is beyond pre-setting...')
end

% Loading the data files
for i = 1 : num_dfile

    fprintf('Current data file <%s>: %d - %s ...\n', participant, i, fullfile(data_pname, data_fname{i}));

    tmp_data_suite = load(data_fname{i},'data_var');
    tmp_sensors_SD1 = tmp_data_suite.data_var{1};
    tmp_sensation_SD1 = tmp_data_suite.data_var{2};
    tmp_sensors_SD2 = tmp_data_suite.data_var{3};
    tmp_sensation_SD2 = tmp_data_suite.data_var{4};

    % Extracting sensor data
    % Force sensor data to measure tightness of the belt
    % Accelerometer data to measure maternal body movement
    % Left / Right Piezo-plate data to measure abdomenal vibration
    % Left / Right Acoustic sensor data to measure abdomenal vibration
    % Left / Right Accelerometer data to measure abdomenal vibration
    forc{i, :} = tmp_sensors_SD1(:, 1);  
    piezL{i, :} = tmp_sensors_SD1(:, 2); 
    piezR{i, :} = tmp_sensors_SD1(:, 3); 
    acouL{i, :} = tmp_sensors_SD1(:, 4); 
    IMUacce{i, :} = tmp_sensors_SD1(:, 5:7);  
    acouR{i, :} = tmp_sensors_SD2(:, 1); 
    acceL{i, :} = tmp_sensors_SD2(:, 5:7); 
    acceR{i, :} = tmp_sensors_SD2(:, 2:4); 

    % the total acceleration from 3-axis data
    IMUacceNet{i, :} = sqrt(sum(IMUacce{i, :}.^2,2));
    acceLNet{i, :} = sqrt(sum(acceL{i, :}.^2,2));
    acceRNet{i, :} = sqrt(sum(acceR{i, :}.^2,2));

    % normalize data: y = (x - min) / (max - min)
    acceLNet_norm{i, :} = (acceLNet{i, :} - min(acceLNet{i, :}))./(max(acceLNet{i, :}) - min(acceLNet{i, :}));
    acceRNet_norm{i, :} = (acceRNet{i, :} - min(acceRNet{i, :}))./(max(acceRNet{i, :}) - min(acceRNet{i, :}));
    acouLNet_norm{i, :} = (acouL{i, :} - min(acouL{i, :}))./(max(acouL{i, :}) - min(acouL{i, :}));
    acouR_norm{i, :} = (acouR{i, :} - min(acouR{i, :}))./(max(acouR{i, :}) - min(acouR{i, :})); 
    piezL_norm{i, :} = (piezL{i, :} - min(piezL{i, :}))./(max(piezL{i, :}) - min(piezL{i, :}));
    piezR_norm{i, :} = (piezR{i, :} - min(piezR{i, :}))./(max(piezR{i, :}) - min(piezR{i, :}));
    forc_norm{i, :} = (forc{i, :} - min(forc{i, :}))./(max(forc{i, :}) - min(forc{i, :}));
    IMUacceNet_norm{i, :} = (IMUacceNet{i, :} - min(IMUacceNet{i, :}))./(max(IMUacceNet{i, :}) - min(IMUacceNet{i, :}));
    
    % standardlize data: y = (x - mean) / standard_deviation
    acceLNet_std{i, :} = (acceLNet{i, :} - mean(acceLNet{i, :})) ./ std(acceLNet{i, :});
    acceRNet_std{i, :} = (acceRNet{i, :} - mean(acceRNet{i, :})) ./ std(acceRNet{i, :});
    acouL_std{i, :} = (acouL{i, :} - mean(acouL{i, :})) ./ std(acouL{i, :});
    acouR_std{i, :} = (acouR{i, :} - mean(acouR{i, :})) ./ std(acouR{i, :}); 
    piezL_std{i, :} = (piezL{i, :} - mean(piezL{i, :})) ./ std(piezL{i, :});
    piezR_std{i, :} = (piezR{i, :} - mean(piezR{i, :})) ./ std(piezR{i, :});
    forc_std{i, :} = (forc{i, :} - mean(forc{i, :})) ./ std(forc{i, :});
    IMUacceNet_std{i, :} = (IMUacceNet{i, :} - mean(IMUacceNet{i, :})) ./ std(IMUacceNet{i, :});

    % sensation data
    sens1{i, :} = tmp_sensation_SD1;
    sens2{i, :} = tmp_sensation_SD2;

    clear tmp_data_suite tmp_sensors_SD1 tmp_sensation_SD1 tmp_sensors_SD2 tmp_sensation_SD2
end

% concatenate all the data files into one
acceLNet_cat = cat(1,acceLNet{:});
acceRNet_cat = cat(1,acceRNet{:});
acouL_cat = cat(1,acouL{:});
acouR_cat = cat(1,acouR{:});
piezL_cat = cat(1,piezL{:});
piezR_cat = cat(1,piezR{:});
forc_cat = cat(1,forc{:});
IMUacceNet_cat = cat(1,IMUacceNet{:});

acceLNet_norm_cat = cat(1,acceLNet_norm{:});
acceRNet_norm_cat = cat(1,acceRNet_norm{:});
acouL_norm_cat = cat(1,acouLNet_norm{:});
acouR_norm_cat = cat(1,acouR_norm{:});
piezL_norm_cat = cat(1,piezL_norm{:});
piezR_norm_cat = cat(1,piezR_norm{:});
forc_norm_cat = cat(1,forc_norm{:});
IMUacceNet_norm_cat = cat(1,IMUacceNet_norm{:});

acceLNet_std_cat = cat(1,acceLNet_std{:});
acceRNet_std_cat = cat(1,acceRNet_std{:});
acouL_std_cat = cat(1,acouL_std{:});
acouR_std_cat = cat(1,acouR_std{:});
piezL_std_cat = cat(1,piezL_std{:});
piezR_std_cat = cat(1,piezR_std{:});
forc_std_cat = cat(1,forc_std{:});
IMUacceNet_std_cat = cat(1,IMUacceNet_std{:});

sens1_cat = cat(1,sens1{:});
sens2_cat = cat(1,sens2{:});

% save organized data matrices
cd([curr_dir '/z11_olddata_mat_raw'])
save(['sensor_data_suite_' participant '.mat'], ...
     'acceL', 'acceR', 'acouL', 'acouR', 'piezL', 'piezR', 'forc', 'IMUacce', ...
     'acceLNet', 'acceRNet', 'IMUacceNet', ...
     'acceLNet_cat', 'acceRNet_cat', 'acouL_cat', 'acouR_cat', 'piezL_cat', 'piezR_cat', 'forc_cat', 'IMUacceNet_cat', ...
     'acceLNet_norm_cat', 'acceRNet_norm_cat', 'acouL_norm_cat', 'acouR_norm_cat', ...
     'piezL_norm_cat', 'piezR_norm_cat', 'forc_norm_cat', 'IMUacceNet_norm_cat', ...
     'acceLNet_std_cat', 'acceRNet_std_cat', 'acouL_std_cat', 'acouR_std_cat', ...
     'piezL_std_cat', 'piezR_std_cat', 'forc_std_cat', 'IMUacceNet_std_cat', ...
     'sens1','sens2', 'sens1_cat','sens2_cat', ...
     '-v7.3' );
cd(curr_dir)













