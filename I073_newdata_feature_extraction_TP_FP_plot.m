clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

%% Loading results
% Load participant: 001RB
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_001RB');
load('TP_FP_sens_001RB.mat');

detection_TPc_001RB = detection_TPc; 
detection_FPc_001RB = detection_FPc; 
num_TP_001RB = num_TP;
num_FP_001RB = num_FP;
num_sensP_001RB = num_sensP;
num_sens_001RB = num_sens;

precision_001RB = num_TP_001RB / (num_TP_001RB + num_FP_001RB);
sensitivity_001RB = num_TP_001RB / num_sensP_001RB;

% Manual: remove 5th data file
precision_001RB_M = (num_TP_001RB-1) / (num_TP_001RB + num_FP_001RB-40);
sensitivity_001RB_M = (num_TP_001RB-1) / (num_sensP_001RB-1);

clear detection_FPc detection_TPc num_FP num_sensP num_TP num_sens

cd(curr_dir)
% ------------------------------------------------------------------------------------------
 
% Load participant: 002JG
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_002JG');
load('TP_FP_sens_002JG.mat');

detection_TPc_002JG = detection_TPc; 
detection_FPc_002JG = detection_FPc; 
num_TP_002JG = num_TP;
num_FP_002JG = num_FP;
num_sensP_002JG = num_sensP;
num_sens_002JG = num_sens;

precision_002JG = num_TP_002JG / (num_TP_002JG + num_FP_002JG);
sensitivity_002JG = num_TP_002JG / num_sensP_002JG;

% Manual: remove ?th data file
precision_002JG_M = num_TP_002JG / (num_TP_002JG + num_FP_002JG);
sensitivity_002JG_M = num_TP_002JG / num_sensP_002JG;

clear detection_FPc detection_TPc num_FP num_sensP num_TP num_sens
cd(curr_dir)
% ---------------------------------------------------------------------------------------------

% Load participant: 003AM
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_003AM');
load('TP_FP_sens_003AM.mat');

detection_TPc_003AM = detection_TPc; 
detection_FPc_003AM = detection_FPc; 
num_TP_003AM = num_TP;
num_FP_003AM = num_FP;
num_sensP_003AM = num_sensP;
num_sens_003AM = num_sens;

precision_003AM = num_TP_003AM / (num_TP_003AM + num_FP_003AM);
sensitivity_003AM = num_TP_003AM / num_sensP_003AM;

% Manual: remove ?th data file
precision_003AM_M = num_TP_003AM / (num_TP_003AM + num_FP_003AM);
sensitivity_003AM_M = num_TP_003AM / num_sensP_003AM;

clear detection_FPc detection_TPc num_FP num_sensP num_TP num_sens
cd(curr_dir)
% ---------------------------------------------------------------------------------------------

%% Plot
% bar chart: the number focus vs day vs night sessions
figure
X = [1 2 3];
Y = [precision_001RB sensitivity_001RB; 
     precision_002JG sensitivity_002JG; 
     precision_003AM sensitivity_003AM];
b = bar(X, Y);

xticklabels({'P01', 'P02' , 'P03'}); ylim([0 1.2])
legend({'precision', 'sensitivity'});

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% -----------------------------------------------------------------------------------------

figure
X = [1 2 3];
Y = [precision_001RB_M sensitivity_001RB_M; 
     precision_002JG_M sensitivity_002JG_M; 
     precision_003AM_M sensitivity_003AM_M];
b = bar(X, Y);

xticklabels({'P01', 'P02' , 'P03'}); ylim([0 1.2])
legend({'precision', 'sensitivity'});

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% -----------------------------------------------------------------------------------------








