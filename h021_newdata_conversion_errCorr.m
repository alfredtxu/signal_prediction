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

% This is the current used version
% 2023.09.15
% T.Xu
clc
clear 
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir);

% files with defects
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\b_mat_003AM2_z2failed';
files = dir([fdir '\*.mat']);

% the number of channels in each file
num_chann_all = 16;
num_chann_imu = 8;
seq_sens = 8;
imu_thresh = 4000;
fm_thresh = 2000;

for f = 1 : length(files)

    tmp_file = files(f).name;
    load([fdir '\' tmp_file]);

    % convert to a vector
    tmp_mtx2 = tmp_mtx';
    tmp_mtx2_v = reshape(tmp_mtx2, [], 1);

    tmp_signal_validation = length(find(tmp_mtx2_v < imu_thresh & tmp_mtx2_v > fm_thresh));

    tmp_idx_zero = find(tmp_mtx2_v == 0);
    tmp_idx_start = tmp_idx_zero(2) - seq_sens + 1; % sensation is 8th channel in a row

    tmp_last_line = tmp_mtx2_v(end-(num_chann_all-1) : end, :);

    % decide if the 8-channel IMU is a sequential
    tmp_imu_position = find(tmp_last_line >= imu_thresh);
    tmp_differences = diff(tmp_imu_position);
    isSequential = all(tmp_differences == tmp_differences(1)); % Check if the differences are constant

    if isSequential
        tmp_idx_end = num_chann_all - max(tmp_imu_position);
    else
        tmp_imu_position2 = tmp_imu_position;
        tmp_imu_position2(tmp_imu_position2>=num_chann_imu) = [];
        tmp_idx_end = num_chann_all - max(tmp_imu_position2);
    end

    % cut off the irrelevant elements
    tmp_mtx2_v_trim = tmp_mtx2_v(tmp_idx_start : length(tmp_mtx2_v)-tmp_idx_end);

    % ceil: half row will be cut off from first and last line, respectively
    tmp_row_cut = ceil(tmp_idx_start / num_chann_all);
    tmp_row_trim = size(tmp_mtx, 1) - tmp_row_cut;

    % reshape to the matrix
    tmp_mtx_corr = reshape(tmp_mtx2_v_trim, 16, tmp_row_trim);

    % unify the matrix name to 'tmp_mtx'
    clear tmp_mtx
    tmp_mtx = tmp_mtx_corr';

    cd(fdir)
    save([tmp_file(1 : end-4) '_corr.mat'], 'tmp_mtx', '-v7.3');
    cd(curr_dir);

    fprintf('File %d > %s was corrected. The number of high signals is %d ... \n', f, tmp_file, tmp_signal_validation);

    clear tmp_*
end
    
  
