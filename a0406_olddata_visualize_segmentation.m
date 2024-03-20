% IMUx	Ols	Olf	LN	LL	WN	WL
%  2	20	10	5	25	100	100
% 						
% 	Sensi	Preci	Sensi m1	Preci m1	Sensi m2	Preci m2
% FM SNR: 20	0.9338	0.3039	0.9338	0.3034	0.9338	0.304
% FM SNR: 21	0.8959	0.3435	0.8959	0.3429	0.8959	0.3441
% FM SNR: 22	0.8353	0.3746	0.8353	0.3743	0.8353	0.376
% FM SNR: 23	0.7778	0.3974	0.7727	0.3975	0.7727	0.3996
clc
clear 
close all

%% project directory (Home / Office)
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% curr_dir = 'G:\My Drive\ic_welcom_leap_fm\b_src_matlab';
rdir = [curr_dir '\z15_olddata_mat_proc_new_summary'];
cd(curr_dir);

% load result file(s) - new method
% RA302_butter_all_olddata_FMsnr23_IMUx2_OLs20_OLf10_LN5_LL25_WN100_WL100_summaryRes.mat
cd(rdir)
snr_th = num2str(23);
Nstd_IMU = num2str(2);
ols_perc = num2str(0.20*100);
olf_perc = num2str(0.10*100);
noise_l = num2str(0.05*100);
low_l = num2str(0.25*100);
noise_w = num2str(1.00*100);
low_w = num2str(1.00*100);

res_file = ['RA302_butter_all_olddata_FMsnr' snr_th '_IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_LN' noise_l '_LL' low_l '_WN' noise_w '_WL' low_w '_summaryRes.mat'];
fprintf('Loading files: %s ... \n', res_file);
load(res_file);

% load results file - old method
load('detection_mtx_oldway.mat');
load('FM_thresh_oldway.mat');

cd(curr_dir);

%% Plotting
% thresholds - IMU
% IMU acce - old method
% Subject 1 & 2: 0.003
% Subject 3, 4 & 5: 0.002
% Loaded processed data ... 1 (12) - sensor_data_suite_S1_proc.mat ... 
% Loaded processed data ... 2 (21) - sensor_data_suite_S2_proc.mat ... 
% Loaded processed data ... 3 (29) - sensor_data_suite_S3_proc.mat ... 
% Loaded processed data ... 4 (22) - sensor_data_suite_S4_proc.mat ... 
% Loaded processed data ... 5 (47) - sensor_data_suite_S5_proc.mat ... 
th_imu_old = ones(1,131);
th_imu_old(:,1:(12+21)) = 0.003;
th_imu_old(:,(12+21+1):end) = 0.002;

for i = 1 : length(all_th_imu)
    th_imu_new(:, i) = all_th_imu(i).thresh;
end

figure
hold on
X_th_imu = 1 : 131;
plot(X_th_imu, th_imu_old, 'r.');
plot(X_th_imu, th_imu_new, 'b.');
legend('OLD', 'NEW');
title('The comprison of IMU fixed and adaptive thresholds through 131 focused sessions');

% thresholds - FM sensors
th_fm_old = cat(1, all_FM_thresh{:});

all_th_fm_selected = squeeze(all_th_fm(:,3,:));
for i = 1 : size(all_th_fm_selected, 1)
    th_fm_new_acouL(i, :) = all_th_fm_selected(i,1).thresh;
    th_fm_new_acouM(i, :) = all_th_fm_selected(i,2).thresh;
    th_fm_new_acouR(i, :) = all_th_fm_selected(i,3).thresh;
    th_fm_new_piezL(i, :) = all_th_fm_selected(i,4).thresh;
    th_fm_new_piezM(i, :) = all_th_fm_selected(i,5).thresh;
    th_fm_new_piezR(i, :) = all_th_fm_selected(i,6).thresh;
end

figure
subplot(2,3,1)
hold on
x_acouL = 1:131;
y_acouL_old = th_fm_old(:, 1);
y_acouL_new = th_fm_new_acouL;
plot(x_acouL, y_acouL_old, 'r.');
plot(x_acouL, y_acouL_new, 'b.');
legend('OLD', 'NEW');
title('Threshold left acoustic (*OLD vs NEW)');

subplot(2,3,2)
hold on
x_acouM = 1:131;
y_acouM_old = th_fm_old(:, 2);
y_acouM_new = th_fm_new_acouM;
plot(x_acouM, y_acouM_old, 'r.')
plot(x_acouM, y_acouM_new, 'b.')
legend('OLD', 'NEW');
title('Threshold middle acoustic (*OLD vs NEW)');

subplot(2,3,3)
hold on
x_acouR = 1:131;
y_acouR_old = th_fm_old(:, 3);
y_acouR_new = th_fm_new_acouR;
plot(x_acouR, y_acouR_old, 'r.')
plot(x_acouR, y_acouR_new, 'b.')
legend('OLD', 'NEW');
title('Threshold right acoustic (*OLD vs NEW)');

subplot(2,3,4)
hold on
x_piezL = 1:131;
y_piezL_old = th_fm_old(:, 4);
y_piezL_new = th_fm_new_piezL;
plot(x_piezL, y_piezL_old, 'r.')
plot(x_piezL, y_piezL_new, 'b.')
legend('OLD', 'NEW');
title('Threshold left piezstic (*OLD vs NEW)');

subplot(2,3,5)
hold on
x_piezM = 1:131;
y_piezM_old = th_fm_old(:, 5);
y_piezM_new = th_fm_new_piezM;
plot(x_piezM, y_piezM_old, 'r.')
plot(x_piezM, y_piezM_new, 'b.')
legend('OLD', 'NEW');
title('Threshold middle piezstic (*OLD vs NEW)');

subplot(2,3,6)
hold on
x_piezR = 1:131;
y_piezR_old = th_fm_old(:, 6);
y_piezR_new = th_fm_new_piezR;
plot(x_piezR, y_piezR_old, 'r.')
plot(x_piezR, y_piezR_new, 'b.')
legend('OLD', 'NEW');
title('Threshold right piezstic (*OLD vs NEW)');

% sensitivity & precision
sp_old = [mean(detection_mtx(:,1) ./ detection_mtx(:,2)); mean(detection_mtx(:,1) ./ detection_mtx(:,3))];
sp_new = [all_sensi_m2N_mu(3); all_preci_m2R_mu(3)];

figure
X = categorical({'Sensitivity', 'Precision'});
Y = [sp_old(1) sp_new(1); sp_old(2) sp_new(2)];
b = bar(X,Y);

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center','VerticalAlignment','bottom')

ylim([0 1.1])
legend('OLD','NEW')
title("FMM Sensitivity & Precision (*OLD vs NEW)")


