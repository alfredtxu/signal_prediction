clc
clear 
close all

% project directory
% curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
curr_dir = 'G:\My Drive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir);

%% Loading data
% example: Seg01_ellip_all_focus_FMsnr25_IMUx2_OL10_WN100_WL100_FM
% fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_all_F_hr10';
fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_all_F_hr10';
cd(fdir)

% Seg01_butter | Seg02_ellip
fserial = 'Preproc012';
fseq = 'Seg01'; 
ftype = 'butter';
snr_th = num2str(23);
Nstd_IMU = num2str(3);
ols_perc = num2str(0.20*100);
olf_perc = num2str(0.05*100);
noise_w = num2str(1.10*100);
low_w = num2str(1.10*100);

preproc_file = [fserial '_all_focus_' ftype '.mat'];
fprintf('Load preprocessed data files: %s ... \n', preproc_file);
load(preproc_file);

match_fileT1 = [fseq '_' ftype '_all_focus_FMsnr' snr_th '_IMUx' Nstd_IMU '_OL' ols_perc '_WN' noise_w '_WL' low_w '_Match.mat'];
sens_fileT1 = [fseq '_' ftype '_all_focus_FMsnr' snr_th '_IMUx' Nstd_IMU '_OL' ols_perc '_WN' noise_w '_WL' low_w '_IMUSens.mat'];
fmLabel_fileT1 = [fseq '_' ftype '_all_focus_FMsnr' snr_th '_IMUx' Nstd_IMU '_OL' ols_perc '_WN' noise_w '_WL' low_w '_FM.mat'];

match_fileT2 = [fseq '_' ftype '_all_focus_FMsnr' snr_th '_IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_WN' noise_w '_WL' low_w '_Match.mat'];
sens_fileT2 = [fseq '_' ftype '_all_focus_FMsnr' snr_th 'IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_WN' noise_w '_WL' low_w '_IMUSens.mat'];
fmLabel_fileT2 = [fseq '_' ftype '_all_focus_FMsnr' snr_th 'IMUx' Nstd_IMU '_OLs' ols_perc '_OLf' olf_perc '_WN' noise_w '_WL' low_w '_FM.mat'];

if isfile(match_fileT1)
    fprintf('Load result file (Type 1): %s ... \n', match_fileT1);
    load(match_fileT1);
    load(sens_fileT1);
    load(fmLabel_fileT1);
elseif isfile(match_fileT2)
    fprintf('Load result file (Type 2): %s ... \n', match_fileT2);
    load(match_fileT2);
    load(sens_fileT2);
    load(fmLabel_fileT2);
else
    fprintf('Load result file - None ... \n');
    return
end
cd(curr_dir)

%% Potting
% the number of channels / FM sensors
freq = 400; 
num_channel = 16;    
num_FMsensors = 6; 

% N-sessions x M-snr x P-sensors
FM_segTB_dil_IMUa_m2s = squeeze(FM_segTB_dil_IMUa_m2);

for f = 1 : size(FM_segTB_dil_IMUa_m2s, 1)

    tmp_acouL = acouLTB_hr10{f, :}; % acou L
    tmp_timeV = ((1:length(tmp_acouL)) / freq)'; % time vector

    for r = 1 : size(FM_segTB_dil_IMUa_m2s, 2)

        for s = 1 %: size(FM_segTB_dil_IMUa_m2s, 3)
            tmp_seg = FM_segTB_dil_IMUa_m2s{f, r, 1};


        end % end of loop: sensor array
    end % end of loop: signal-noise ratios
end % end of loop: sessions









