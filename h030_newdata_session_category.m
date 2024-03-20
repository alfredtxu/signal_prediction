% This is a semi-automic function to distinguish the data generated for day
% or night
% 
% 2023.09.16
% T.Xu
clc
clear 
close all

% HOME
% curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';

% OFFICE
curr_dir = pwd;
cd(curr_dir)

participant = 'b_mat_003AM2';

% HOME
fdir = ['D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\' participant];

% OFFICE
% fdir = ['G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data\' participant];

files = dir([fdir '\*.mat']);

sess_cate = {'focus', 'general_day', 'general_night'};
fdir_focus = [fdir '_F'];
fdir_day = [fdir '_D'];
fdir_night = [fdir '_N'];
% --------------------------------------------------------------------------------

% sampling rate
freq = 400;

% trim data to remove the '1s' at the beginning and end of the time series data
trimS = 15;
trimE = 15;

% threshold for deciding a night session (in minutes)
thresh_night = 300;

for i = 1 : length(files)
    
    % load data file
    load([fdir '\' files(i).name]);
    
    % session duration
    file_dura(i, :) = size(tmp_mtx, 1) / 400 / 60; 
    
    % the number of sensation press
    tmp_sens = bitshift(tmp_mtx(:, 8), -8); 
    tmp_sens_T = tmp_sens((trimS*freq + 1) : (end-trimE*freq));
    num_press(i, :) = sum(tmp_sens_T);

    if num_press(i, :) > 0
        system(['copy ' fdir '\' files(i).name ' ' fdir_focus]);
    else
        system(['copy ' fdir '\' files(i).name ' ' fdir_day]);
    end

    fprintf('File > %s: duration - %d & number of press - %d ... \n', files(i).name, file_dura(i, :), num_press(i, :));

    clear tmp_sens tmp_sens_T
end



