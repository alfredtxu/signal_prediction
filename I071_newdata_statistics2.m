% This function is designated for the participants who have already
% deliverd the birth.
% 
% 001RB: 20230525 - 20230717 w28 - w35+5
% 002JG: 20230706 - 20230829 w33+2 - w40+5

clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

participant = '003AM';
data_folder = ['c_processed_' participant];
data_seq = '02';
data_stage = 'proc';
data_category = 'focus';

data_file = [fdir '\' data_folder '\FM' data_seq '_b_mat_' participant '_' data_category '_' data_stage '.mat'];

% load data
load(data_file);

% sampling rate
freq = 400;

cd([fdir '\' data_folder])
save(['FMactivity_' participant '_' data_category '.mat'], ...
     'sens_map_activity', 'sens_mapIMUa_activity', 'sens_mapIMUag_activity', 'sens_mapIMUg_activity', '-v7.3');
cd(curr_dir)