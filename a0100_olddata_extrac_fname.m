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

data_fnameS5 = data_fname';
clear data_fname data_pname num_dfile

all_data_fname = cat(1, data_fnameS1, data_fnameS2, data_fnameS3, data_fnameS4, data_fnameS5);

save('data_fname.mat', 'all_data_fname', 'data_fnameS1', 'data_fnameS2', 'data_fnameS3', 'data_fnameS4', 'data_fnameS5', '-v7.3');