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

% LOADING PRE-PROCESSED DATA
% ** set current loading data portion
% group the data matrice by individuals
participant = 'S5';
load(['sensor_data_suite_' participant '.mat']);

% the number of data file
nfile = size(sens1, 1);
%-------------------------------------------------------------------------

for i = 1 : nfile

    tmp_sd1 = sens1{i,:};
    tmp_sd2 = sens2{i,:};

    tmp_u1 = unique(tmp_sd1);
    tmp_u2 = unique(tmp_sd2);

%     fprintf('Unique value range from SD1 and SD2: %d - %d [%d %d] vs %d - %d [%d %d] ... \n', ...
%         length(tmp_sd1), length(tmp_u1), min(tmp_u1), max(tmp_u1), ...
%         length(tmp_sd2), length(tmp_u2), min(tmp_u2), max(tmp_u2));

    tmp_forc = forc{i,:};
    tmp_IMUacce = IMUacce{i,:};
    tmp_acceL = acceL{i,:};
    tmp_acceR = acceR{i,:};
    tmp_acouL = acouL{i,:};
    tmp_acouR = acouR{i,:};
    tmp_piezL = piezL{i,:};
    tmp_piezR = piezR{i,:};

    tmp_len1 = unique([length(tmp_sd1) length(tmp_forc) length(tmp_piezL) length(tmp_piezR) length(acouL) length(IMUacce)]);
    tmp_len2 = unique([length(tmp_sd2) length(acouR) length(acceL) length(acceR)]);

    fprintf('Validating data length SD1 vs SD2: [%d %d %d %d %d %d] vs [%d %d %d %d] ... \n', ...
        length(tmp_sd1), length(tmp_forc), length(tmp_piezL), length(tmp_piezR), length(tmp_acouL), length(tmp_IMUacce), ...
        length(tmp_sd2), length(tmp_acouR), length(tmp_acceL), length(tmp_acceR));

    clear tmp_*

end

