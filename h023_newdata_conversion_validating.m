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

curr_dir=pwd;
cd(curr_dir);

pname = 'b_mat_003AM';
fdir = ['D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\' pname];
files = dir([fdir '\*.mat']);

% raw data format
Data_precision = 'uint16';

num_chann = 16;
num_FMsensors = 6;
freq = 400;

% Maximum value of the test data
maxTestData_value = 256;

for i = 1 : length(files)

    % load the binary data file (*.dat)
    tmp_fname = files(i).name;
    load([fdir '\' tmp_fname]);

    % test
    tmp_seq = 1;
    tmp_cycle = 0;
    tmp_test_flag = 1;
    
    tmp_test = bitand(tmp_mtx(:,8), 255);
    
    while tmp_seq <= length(tmp_test)
    
        if tmp_test(tmp_seq) ~= (tmp_seq-1 - tmp_cycle*maxTestData_value)
            tmp_test_flag = 0;
            break;
        end
    
        % condition to update the cycle_no
        if mod(tmp_seq, maxTestData_value) == 0
            tmp_cycle = tmp_cycle + 1;
        end
    
        tmp_seq = tmp_seq + 1;
    end
    
    % if the flag has not changed, that means data is not corrupted
    if (tmp_test_flag == 1)
        fprintf('> PASSED: Data file %d - %s has no data corruption.\n', i, tmp_fname);
        hometrial_info(i).test = 'passed';
    else
        fprintf('* FAILED: Data file %d - %s: has data corruption.\n', i, tmp_fname);
        hometrial_info(i).test = 'failed';
    end

    clear tmp_*
end

