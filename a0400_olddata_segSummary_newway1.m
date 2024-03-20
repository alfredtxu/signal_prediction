clc
clear 
close all

% project directory (Home / Office)
curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
% curr_dir = 'G:\My Drive\ic_welcom_leap_fm\b_src_matlab';
fdir = [curr_dir '\z14_olddata_mat_proc_new'];
cd(curr_dir);

%% Loading data (Home / Office)
% example 1: Seg01_butter_all_olddata_001_022_FMsnr23_IMUx3_OLs20_OLf10_LN1_LL10_WN100_WL100_FM.mat
% example 2: Seg01_butter_all_olddata_001_022_FMsnr23_IMUx3_OLs20_OLf10_WN100_WL100_FM.mat
cd(fdir)
portions = '001_022';
snr_th = num2str(23);
Nstd_IMU = num2str(3);
ols_perc = num2str(0.20*100);
olf_perc = num2str(0.10*100);
noise_l = num2str(0.05*100);
low_l = num2str(0.25*100);
noise_w = num2str(1.10*100);
low_w = num2str(1.10*100);

match_file = ['Seg01_butter_all_olddata_' portions '_FMsnr' snr_th '_IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_LN' noise_l '_LL' low_l '_WN' noise_w '_WL' low_w '_Match.mat'];
sens_file = ['Seg01_butter_all_olddata_' portions '_FMsnr' snr_th 'IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_LN' noise_l '_LL' low_l '_WN' noise_w '_WL' low_w '_IMUSens.mat'];
fmLabel_file = ['Seg01_butter_all_olddata_' portions '_FMsnr' snr_th 'IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_LN' noise_l '_LL' low_l '_WN' noise_w '_WL' low_w '_FM_label.mat'];

fprintf('Load result file (Type 2): %s ... \n', match_file);
load(match_file);
cd(curr_dir);

%% Sensitivity & Precision
% Layers: i, n1, n2, w1, w2
% matchTB_FM_Sens_sensi = matchTB_FM_Sens_num / sensT_mapC;
% matchTB_FM_Sens_preci = matchTB_FM_Sens_num / FM_segTB_dil_fOR_acouORpiezC;
% 
% matchTB_FM_Sens_sensi, matchTB_FM_Sens_preci
% matchTB_FM_Sens_IMUa_m1_sensi, matchTB_FM_Sens_IMUa_m1_preci, 
% matchTB_FM_Sens_IMUa_m2_sensi, matchTB_FM_Sens_IMUa_m2_preci, 
matchTB_FM_Sens_sensiV = matchTB_FM_Sens_sensi(:);
matchTB_FM_Sens_sensiV(matchTB_FM_Sens_sensiV>1) = 1;
matchTB_FM_Sens_sensi_norm = reshape(matchTB_FM_Sens_sensiV, size(matchTB_FM_Sens_sensi));
matchTB_FM_Sens_sensi_norm_mu = squeeze(mean(matchTB_FM_Sens_sensi_norm, 1));

matchTB_FM_Sens_preciV = matchTB_FM_Sens_preci(:);
matchTB_FM_Sens_preciV(matchTB_FM_Sens_preciV>1) = 1;
matchTB_FM_Sens_preci_norm = reshape(matchTB_FM_Sens_preciV, size(matchTB_FM_Sens_preci));
matchTB_FM_Sens_preci_norm_mu = squeeze(mean(matchTB_FM_Sens_preci_norm, 1));

[matchTB_FM_Sens_sensi_norm_mu matchTB_FM_Sens_preci_norm_mu]
% ---------------------------------------------------------------------------------------------

matchTB_FM_Sens_IMUa_m1_sensiV = matchTB_FM_Sens_IMUa_m1_sensi(:);
matchTB_FM_Sens_IMUa_m1_sensiV(matchTB_FM_Sens_IMUa_m1_sensiV>1) = 1;
matchTB_FM_Sens_IMUa_m1_sensi_norm = reshape(matchTB_FM_Sens_IMUa_m1_sensiV, size(matchTB_FM_Sens_IMUa_m1_sensi));
matchTB_FM_Sens_IMUa_m1_sensi_norm_mu = squeeze(mean(matchTB_FM_Sens_IMUa_m1_sensi_norm, 1));

matchTB_FM_Sens_IMUa_m1_preciV = matchTB_FM_Sens_IMUa_m1_preci(:);
matchTB_FM_Sens_IMUa_m1_preciV(matchTB_FM_Sens_IMUa_m1_preciV>1) = 1;
matchTB_FM_Sens_IMUa_m1_preci_norm = reshape(matchTB_FM_Sens_IMUa_m1_preciV, size(matchTB_FM_Sens_IMUa_m1_preci));
matchTB_FM_Sens_IMUa_m1_preci_norm_mu = squeeze(mean(matchTB_FM_Sens_IMUa_m1_preci_norm, 1));

[matchTB_FM_Sens_IMUa_m1_sensi_norm_mu matchTB_FM_Sens_IMUa_m1_preci_norm_mu]
% ---------------------------------------------------------------------------------------------

% matchTB_FM_Sens_IMUa_m2_sensiV = matchTB_FM_Sens_IMUa_m2_sensi(:);
% matchTB_FM_Sens_IMUa_m2_sensiV(matchTB_FM_Sens_IMUa_m2_sensiV>1) = 1;
% matchTB_FM_Sens_IMUa_m2_sensi_norm = reshape(matchTB_FM_Sens_IMUa_m2_sensiV, size(matchTB_FM_Sens_IMUa_m2_sensi));
% matchTB_FM_Sens_IMUa_m2_sensi_norm_mu = squeeze(mean(matchTB_FM_Sens_IMUa_m2_sensi_norm, 1));
% 
% matchTB_FM_Sens_IMUa_m2_preciV = matchTB_FM_Sens_IMUa_m2_preci(:);
% matchTB_FM_Sens_IMUa_m2_preciV(matchTB_FM_Sens_IMUa_m2_preciV>1) = 1;
% matchTB_FM_Sens_IMUa_m2_preci_norm = reshape(matchTB_FM_Sens_IMUa_m2_preciV, size(matchTB_FM_Sens_IMUa_m2_preci));
% matchTB_FM_Sens_IMUa_m2_preci_norm_mu = squeeze(mean(matchTB_FM_Sens_IMUa_m2_preci_norm, 1));
% 
% [matchTB_FM_Sens_IMUa_m2_sensi_norm_mu matchTB_FM_Sens_IMUa_m2_preci_norm_mu]
% ---------------------------------------------------------------------------------------------

%% Plotting
% choose the best results
% figure
% X = ["P1" "P2" "P3"];
% Y = [0.8881 0.0984; 0.8871 0.1297; 1.0000 0.2206];
% X = [1 2 3];
% Y = [0.8881 0.8871 1.0000; 0.0984 0.1297 0.2206];
% b = bar(X, Y);
% 
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
% 
% xtips2 = b(2).XEndPoints;
% ytips2 = b(2).YEndPoints;
% labels2 = string(b(2).YData);
% text(xtips2,ytips2,labels2,'HorizontalAlignment','center','VerticalAlignment','bottom')
% 
% ylim([0 1.2])
% title("FMM Sensitivity & Precision (filter: butter)")


