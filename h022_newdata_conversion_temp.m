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

rdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\z_demo_20230919_raw';
mdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\z_demo_20230919_mat';

% raw data format
Data_precision = 'uint16';

num_chann = 16;
num_channVis = 10;
num_FMsensors = 6;

% for plotting titles 
sensor_list = {'Force sensor'; ...
               'Acoustic sensor Left'; 'Acoustic sensor Middle'; 'Acoustic sensor Right'; ...
               'Piezo-Plate Left'; 'Piezo-Plate Middle'; 'Piezo-Plate Right';...
               'IMU Accelerometers';'IMU Gyroscopes'};

% Frequency of sensor / sensation data sampling in 1024 / 512 HZ
freq = 400;

% ADC resolution is 12 bit
ADC_resolution = 12;
maxADC_value = 2^ADC_resolution - 1;

% 3.3 volt corresponds to 4095 ADC value (12-bit)
volt = 3.3;
slope = volt / maxADC_value;

% Maximum value of the test data
maxTestData_value = 256;

%% Convert raw data (*.dat files) to Matlab data file (*.mat)
% Locate and load selected data files
% Returns file names with the extension in a cell, and path name
% participant = 'jg';
% [data_fname, data_pname] = uigetfile([participant '*.dat'],'Select the data files','MultiSelect','on');
% addpath(data_pname);
files = dir([rdir '\*.dat']);

hometrial_info = [];

for i = 1 : length(files)

    % load the binary data file (*.dat)
    cd(rdir);
    tmp_fname = files(i).name;
    tmp_fid = fopen(tmp_fname);
    tmp_dfile = fread(tmp_fid, Data_precision);
    fclose(tmp_fid);
    cd(curr_dir)

    % number of rows in the converted data variable
    tmp_rows = length(tmp_dfile) / num_chann;

    % form the data vector into 2-D matrix
    tmp_elem = 1;
    tmp_mtx = zeros(tmp_rows, num_chann);
    for r = 1 : tmp_rows
        for c = 1 : num_chann
            tmp_mtx(r, c) = tmp_dfile(tmp_elem);
            tmp_elem = tmp_elem + 1;
        end
    end

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

    hometrial_info(i).fname = tmp_fname;
    hometrial_info(i).duration = tmp_rows / freq / 60;

    hometrial_data{i, :} = tmp_mtx;

    cd(mdir)
    save([tmp_fname(1:end-4) '.mat'], 'tmp_mtx', '-v7.3');
    cd(curr_dir)

    fprintf('Raw data is converted ... %d - %s (trial duration: %d) ... \n', i, tmp_fname, hometrial_info(i).duration);

    clear tmp_fname tmp_fid tmp_dfile tmp_rows tmp_elem tmp_mtx ...
          tmp_seq tmp_cycle tmp_test tmp_test_flag
end

% ts = datestr(datenum(datetime), 'yyyymmdd');
% save(['hometrial_dfile_' num2str(num_file) '_' participant '_' ts '.mat'], 'hometrial_data', '-v7.3');
% save(['hometrial_info_' participant '.mat'], 'hometrial_info', '-v7.3');
% ---------------------------------------------------------------------------------------

%% visualization 
% data organization
forc = tmp_mtx(:, 1); 

IMUacc = tmp_mtx(:, 9:11);
IMUaccNet = sqrt(sum(IMUacc.^2,2));

IMUgyr = tmp_mtx(:, 12:14); 
IMUgyrNet = sqrt(sum(IMUgyr.^2,2));

acouL = tmp_mtx(:, 7);
acouM = tmp_mtx(:, 4);
acouR = tmp_mtx(:, 6);
piezL = tmp_mtx(:, 1);
piezM = tmp_mtx(:, 3);
piezR = tmp_mtx(:, 2);
sens = tmp_mtx(:, 8);

% Trimming
% * Removal period in second
trimS = 15;
trimE = 15;

forcT = forc((trimS*freq + 1):(end-trimE*freq));

IMUaccT = IMUacc((trimS*freq + 1):(end-trimE*freq),:);
IMUgyrT = IMUgyr((trimS*freq + 1):(end-trimE*freq),:);

IMUaccNetT = IMUaccNet((trimS*freq + 1):(end-trimE*freq));
IMUgyrNetT = IMUgyrNet((trimS*freq + 1):(end-trimE*freq));

acouLT = acouL((trimS*freq + 1):(end-trimE*freq));
acouMT = acouM((trimS*freq + 1):(end-trimE*freq));
acouRT = acouR((trimS*freq + 1):(end-trimE*freq));
piezLT = piezL((trimS*freq + 1):(end-trimE*freq));
piezMT = piezM((trimS*freq + 1):(end-trimE*freq));
piezRT = piezR((trimS*freq + 1):(end-trimE*freq));

sensT = sens((trimS*freq + 1):(end-trimE*freq));
% ----------------------------------------------------------------------------------------

%% Visualization > plotting raw data (Full & Trimmed)
% multiply the amplitude of sensation data to plot properly
amp_multiplier_raw = 1; 

title_raw = {'Raw Sensor response', 'Raw Sensor response (Trimmed)', ...
             '', '', '', '', '', '', '', '', '', '', ...
             '', '', '', '', '', '', '', '', '', ''};
legend_raw = {'Flexi force', ...
              'IMU accelerometer', 'IMU gyroscope', ...
              'IMU accelerometer (net)', 'IMU gyroscope  (net)', ...
              'Acoustic sensor L', 'Acoustic sensor M', 'Acoustic sensor R', ...
              'Piezoelectric plate L', 'Piezoelectric plate M', 'Piezoelectric plate R'};
xlabel_raw = {'', '', '', '', '', '', '', '', '', '', ...
              '', '', '', '', '', '', '', '', '', '', ...
              'Time (s)', 'Frequency (Hz)'};
    
% time vector for the sensor and sensation data in SD1 / SD2
sensor_timeV = (0 : (length(acouL)-1)) / freq;
sensor_timeTV = (0 : (length(acouLT)-1)) / freq;
sensor_all_raw = {forc, IMUacc, IMUgyr, IMUaccNet, IMUgyrNet, acouL, acouM, acouR, piezL, piezM, piezR};
sensor_all_rawT = {forcT, IMUaccT, IMUgyrT, IMUaccNetT, IMUgyrNetT, acouLT, acouMT, acouRT, piezLT, piezMT, piezRT};

% Plotting using subplot
figure
for j = 1 : num_FMsensors + 5

    subplot(num_FMsensors+5, 2, 2*j-1);
    plot(sensor_timeV', sensor_all_raw{j}, 'LineWidth', 2);
    title(title_raw{2*j-1});
    legend (legend_raw{j});
    legend boxoff;
    ylabel('Response (V)');
    xlabel(xlabel_raw{2*j-1});

    subplot(num_FMsensors+5, 2, 2*j);
    plot(sensor_timeTV', sensor_all_rawT{j}, 'LineWidth', 2);
    title(title_raw{2*j});
    legend (legend_raw{j});
    legend boxoff;
    ylabel('Response (V)');
    xlabel(xlabel_raw{2*j});
end
% -----------------------------------------------------------------------------------------




