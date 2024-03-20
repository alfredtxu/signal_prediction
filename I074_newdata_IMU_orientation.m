clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';

participant = '003AM';
data_folder = ['c_processed_' participant];
data_seq = '01';
data_stage = 'preproc';
data_category = 'day';

data_file = [fdir '\' data_folder '\FM' data_seq '_b_mat_' participant '_' data_category '_' data_stage '.mat'];

% load data
load(data_file);

% sampling rate
freq = 400;

% day: 1
% focus: 7
% night: 4
for i = 1 : size(IMUacc_T, 1)

    tmp_acc = IMUacc_T{i};
    tmp_gyr = IMUgyr_T{i};
    tmp_ax = tmp_acc(:,1);
    tmp_ay = tmp_acc(:,2);
    tmp_az = tmp_acc(:,3);
    tmp_gx = tmp_gyr(:,1);
    tmp_gy = tmp_gyr(:,2);
    tmp_gz = tmp_gyr(:,3);

    tmp_time_vec = (0 : length(tmp_ax) - 1) / freq;

    figure
    hold on
    subplot(2,1,1)
    plot(tmp_time_vec, tmp_acc);
    xlabel('Time (s)')
    ylabel('Rotation (degrees)')
    title('IMU Raw data - Accelerometer')
    
    subplot(2,1,2)
    plot(tmp_time_vec, tmp_gyr);
    xlabel('Time (s)')
    ylabel('Rotation (degrees)')
    title('IMU Raw data - Gyroscope')

%     figure
%     hold on
%     subplot(6,1,1)
%     plot(tmp_time_vec, tmp_ax);
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     title('IMU Raw data - Accelerometer dim 1')
%     
%     subplot(6,1,2)
%     plot(tmp_time_vec, tmp_ay);
%     legend('Y-axis')
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     title('IMU Raw data - Accelerometer dim 2')
%     
%     subplot(6,1,3)
%     plot(tmp_time_vec, tmp_az);
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     title('IMU Raw data - Accelerometer dim 3')
%     
%     subplot(6,1,4)
%     plot(tmp_time_vec, tmp_gx);
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     title('IMU Raw data - Gyroscope dim 1')
%     
%     subplot(6,1,5)
%     plot(tmp_time_vec, tmp_gy);
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     title('IMU Raw data - Gyroscope dim 2')
%     
%     subplot(6,1,6)
%     plot(tmp_time_vec, tmp_gz);
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     title('IMU Raw data - Gyroscope dim 3')

    % Pitch is rotation on the Y-axis, which means an object is tilted up or down.
    % Roll is rotation on the X-axis, which means an object is tilted right or left.
    % pitchFromAccel = atan(-accelerometer.x / sqrt(pow(accelerometer.y, 2) + pow(accelerometer.z, 2)));	
    % rollFromAccel = atan(accelerometer.y / sqrt(pow(accelerometer.x, 2) + pow(accelerometer.z, 2)));
    % return value in the angle in RADIANS
    tmp_pitchFromAccel = atan(-tmp_ax ./ sqrt(tmp_ay.^2 + tmp_az.^2));	
    tmp_rollFromAccel = atan(tmp_ay ./ sqrt(tmp_ax.^2 + tmp_az.^2));
    
    tmp_pitchFromAccel_deg = rad2deg(tmp_pitchFromAccel);
    tmp_rollFromAccel_deg = rad2deg(tmp_rollFromAccel);
    
    figure
    hold on
    subplot(2,1,1)
    plot(tmp_time_vec, tmp_pitchFromAccel_deg);
    xlabel('Time (s)')
    ylabel('Rotation (degrees)')
    title('Accelerometer tilt (in degrees) for pitch')
    
    subplot(2,1,2)
    plot(tmp_time_vec, tmp_rollFromAccel_deg);
    xlabel('Time (s)')
    ylabel('Rotation (degrees)')
    title('Accelerometer tilt (in degrees) for roll')

%     % imufilter
%     decim = 1;
%     fuse_ag = imufilter('SampleRate',freq,'DecimationFactor',decim);
%     q_ag = fuse_ag(tmp_acc,tmp_gyr);
%     e_ag = eulerd(q_ag,'ZYX','frame');
%     time_ag = (0:decim:size(tmp_acc,1)-1)/freq;
%     
%     figure
%     plot(time_ag,e_ag)
%     title('Orientation Estimate - Accelerometer & Gyroscope')
%     legend('Z-axis', 'Y-axis', 'X-axis')
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
%     % -----------------------------
% 
%     % plot the difference
%     figure
%     plot(tmp_time_vec,tmp_pitchFromAccel_deg, time_ag, e_ag(:,2))
%     title('Comparison of Orientation Estimate')
%     legend('Accelerometer', 'Fusion of Acc & Gyr')
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')
% 
%     figure
%     plot(tmp_time_vec,tmp_rollFromAccel_deg, time_ag, e_ag(:,3))
%     title('Comparison of Orientation Estimate')
%     legend('Accelerometer', 'Fusion of Acc & Gyr')
%     xlabel('Time (s)')
%     ylabel('Rotation (degrees)')

end



