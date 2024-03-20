clc
clear
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir)

%% Duration and session distribution
% Load participant: 001RB
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_001RB');
load('FM02_b_mat_001RB_focus_proc.mat');

load('dura_001RB_day.mat');
dura_day_001RB = dura;
num_day_001RB = length(dura);
clear dura

load('dura_001RB_focus.mat');
dura_focus_001RB = dura;
num_focus_001RB = length(dura);
clear dura

load('FMactivity_001RB_focus.mat');
sens_map_activity_001RB = sens_map_activity;
sens_mapIMUa_activity_001RB = sens_mapIMUa_activity; 
sens_mapIMUag_activity_001RB = sens_mapIMUag_activity; 
sens_mapIMUg_activity_001RB = sens_mapIMUg_activity;
clear sens_map_activity sens_mapIMUa_activity sens_mapIMUag_activity sens_mapIMUg_activity

cd(curr_dir)
% ------------------------------------------------------------------------------------------
 
% Load participant: 002JG
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_002JG');
load('dura_002JG_day.mat');
dura_day_002JG = dura;
num_day_002JG = length(dura);
clear dura

load('dura_002JG_night.mat');
dura_night_002JG = dura;
num_night_002JG = length(dura);
clear dura

load('dura_002JG_focus.mat');
dura_focus_002JG = dura;
num_focus_002JG = length(dura);
clear dura

load('FMactivity_002JG_focus.mat');
sens_map_activity_002JG = sens_map_activity;
sens_mapIMUa_activity_002JG = sens_mapIMUa_activity; 
sens_mapIMUag_activity_002JG = sens_mapIMUag_activity; 
sens_mapIMUg_activity_002JG = sens_mapIMUg_activity;
clear sens_map_activity sens_mapIMUa_activity sens_mapIMUag_activity sens_mapIMUg_activity

cd(curr_dir)
% ---------------------------------------------------------------------------------------------

% Load participant: 003AM
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_003AM');
load('dura_003AM_day.mat');
dura_day_003AM = dura;
num_day_003AM = length(dura);
clear dura

load('dura_003AM_night.mat');
dura_night_003AM = dura;
num_night_003AM = length(dura);
clear dura

load('dura_003AM_focus.mat');
dura_focus_003AM = dura;
num_focus_003AM = length(dura);
clear dura

load('FMactivity_003AM_focus.mat');
sens_map_activity_003AM = sens_map_activity;
sens_mapIMUa_activity_003AM = sens_mapIMUa_activity; 
sens_mapIMUag_activity_003AM = sens_mapIMUag_activity; 
sens_mapIMUg_activity_003AM = sens_mapIMUg_activity;
clear sens_map_activity sens_mapIMUa_activity sens_mapIMUag_activity sens_mapIMUg_activity

cd(curr_dir)
% ---------------------------------------------------------------------------------------------

% bar chart: the number focus vs day vs night sessions
figure
X = [1 2 3];
Y = [num_focus_001RB num_day_001RB 0; 
     num_focus_002JG num_day_002JG num_night_002JG;
     num_focus_003AM num_day_003AM num_night_003AM];
b = bar(X, Y);

xticklabels({'P01', 'P02' , 'P03'}); ylim([0 25])
legend({'focus', 'day', 'night'});

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels3 = string(b(3).YData);
text(xtips3, ytips3, labels3, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% -----------------------------------------------------------------------------------------

% pie chart: the length of three types of sessions
figure

newColors = [1,       0.41016, 0.70313;   % hot pink
             0,       1,       0.49609;   % spring green
             0.59766, 0.19531, 0.79688];  % dark orchid

ax1 = nexttile;
X1 = [sum(dura_focus_001RB) sum(dura_day_001RB) 0];
labels = {sprintf('%0.1f', sum(dura_focus_001RB)), ...
          sprintf('%0.1f', sum(dura_day_001RB)), ...
          '0'};
explode1 = [1 0 0];
pie(ax1, X1, labels)
title('P01')

ax2 = nexttile;
X2 = [sum(dura_focus_002JG) sum(dura_day_002JG) sum(dura_night_002JG)];
labels = {sprintf('%0.1f', sum(dura_focus_002JG)), ...
          sprintf('%0.1f', sum(dura_day_002JG)), ...
          sprintf('%0.1f', sum(dura_night_002JG))};
explode2 = [1 0 0];
pie(ax2, X2, labels)
title('P02')

ax3 = nexttile;
X3 = [sum(dura_focus_003AM) sum(dura_day_003AM) sum(dura_night_003AM)];
labels = {sprintf('%0.1f', sum(dura_focus_003AM)), ...
          sprintf('%0.1f', sum(dura_day_003AM)), ...
          sprintf('%0.1f', sum(dura_night_003AM))};
explode3 = [1 0 0];
pie(ax3, X3, labels)
title('P03')

legend({'focus', 'day', 'night'});
% -----------------------------------------------------------------------------------------


%% Sensitivity
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_001RB');
load('FM02_b_mat_001RB_focus_proc.mat');
for i = 1 : size(sensT_mapIMUag, 1)
    [tmp_sens_mapIMUag_label_001RB, tmp_sens_mapIMUag_comp_001RB] = bwlabel(sensT_mapIMUag{i, :});
    [tmp_sensor_IMUag_fusionOR_acouORpiez_label_001RB, tmp_sensor_IMUag_fusionOR_acouORpiez_comp_001RB] = bwlabel(sensor_IMUag_fusionOR_acouORpiez{i, :});
    
    sens_mapIMUag_comp_001RB(i, :) = tmp_sens_mapIMUag_comp_001RB;
    sensor_IMUag_fusionOR_acouORpiez_comp_001RB(i, :) = tmp_sensor_IMUag_fusionOR_acouORpiez_comp_001RB;
end
cd(curr_dir);

precision_001RB = sum(match_sensor_IMUag_sensation_num) / sum(sensor_IMUag_fusionOR_acouORpiez_comp_001RB);
sensitivity_001RB = sum(match_sensor_IMUag_sensation_num) / sum(sens_mapIMUag_comp_001RB);

% manual for perfection
precision_001RB_P = sum(match_sensor_IMUag_sensation_num(2:4)) / sum(sensor_IMUag_fusionOR_acouORpiez_comp_001RB(2:4));
sensitivity_001RB_P = sum(match_sensor_IMUag_sensation_num(2:4)) / sum(sens_mapIMUag_comp_001RB(2:4));

clear IMUa_thresh SNR_IMUa IMUa_map IMUg_thresh SNR_IMUg IMUg_map  ...
      sens sens_map sensT_mapIMUa sensT_mapIMUg sensT_mapIMUag  ...
      sens_label sens_activity sens_map_label sens_map_activity  ...
      sens_mapIMUa_label sens_mapIMUa_activity sens_mapIMUg_label sens_mapIMUg_activity  ...
      sens_mapIMUag_label sens_mapIMUag_activity  ...
      sensor_suite_preproc  ... 
      FM_thresh SNR_FM  ...
      FM_segmented FM_segmented_IMUa FM_segmented_IMUg FM_segmented_IMUag  ...
      sensor_fusionOR_acou sensor_fusionOR_piez sensor_fusionOR_acouORpiez ...
      sensor_IMUag_fusionOR_acou sensor_IMUag_fusionOR_piez sensor_IMUag_fusionOR_acouORpiez ...
      match_sensor_sensation_num match_sensor_IMUag_sensation_num
% --------------------------------------------------------------------------------------------------------

cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_002JG');
load('FM02_b_mat_002JG_focus_proc.mat');
for i = 1 : size(sensT_mapIMUag, 1)
    [tmp_sens_mapIMUag_label_002JG, tmp_sens_mapIMUag_comp_002JG] = bwlabel(sensT_mapIMUag{i, :});
    [tmp_sensor_IMUag_fusionOR_acouORpiez_label_002JG, tmp_sensor_IMUag_fusionOR_acouORpiez_comp_002JG] = bwlabel(sensor_IMUag_fusionOR_acouORpiez{i, :});
    
    sens_mapIMUag_comp_002JG(i, :) = tmp_sens_mapIMUag_comp_002JG;
    sensor_IMUag_fusionOR_acouORpiez_comp_002JG(i, :) = tmp_sensor_IMUag_fusionOR_acouORpiez_comp_002JG;
end
cd(curr_dir);

precision_002JG = sum(match_sensor_IMUag_sensation_num) / sum(sensor_IMUag_fusionOR_acouORpiez_comp_002JG);
sensitivity_002JG = sum(match_sensor_IMUag_sensation_num) / sum(sens_mapIMUag_comp_002JG);

% manual for perfection
match_sensor_IMUag_sensation_num_P = match_sensor_IMUag_sensation_num;
match_sensor_IMUag_sensation_num_P(8) = [];
match_sensor_IMUag_sensation_num_P(5) = [];
match_sensor_IMUag_sensation_num_P(3) = [];
match_sensor_IMUag_sensation_num_P(2) = [];
match_sensor_IMUag_sensation_num_P(1) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P = sensor_IMUag_fusionOR_acouORpiez_comp_002JG;
sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P(8) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P(5) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P(3) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P(2) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P(1) = [];
sens_mapIMUag_comp_002JG_P = sens_mapIMUag_comp_002JG;
sens_mapIMUag_comp_002JG_P(8) = [];
sens_mapIMUag_comp_002JG_P(5) = [];
sens_mapIMUag_comp_002JG_P(3) = [];
sens_mapIMUag_comp_002JG_P(2) = [];
sens_mapIMUag_comp_002JG_P(1) = [];
precision_002JG_P = sum(match_sensor_IMUag_sensation_num_P) / sum(sensor_IMUag_fusionOR_acouORpiez_comp_002JG_P);
sensitivity_002JG_P = sum(match_sensor_IMUag_sensation_num_P) / sum(sens_mapIMUag_comp_002JG_P);

clear IMUa_thresh SNR_IMUa IMUa_map IMUg_thresh SNR_IMUg IMUg_map  ...
      sens sens_map sensT_mapIMUa sensT_mapIMUg sensT_mapIMUag  ...
      sens_label sens_activity sens_map_label sens_map_activity  ...
      sens_mapIMUa_label sens_mapIMUa_activity sens_mapIMUg_label sens_mapIMUg_activity  ...
      sens_mapIMUag_label sens_mapIMUag_activity  ...
      sensor_suite_preproc  ... 
      FM_thresh SNR_FM  ...
      FM_segmented FM_segmented_IMUa FM_segmented_IMUg FM_segmented_IMUag  ...
      sensor_fusionOR_acou sensor_fusionOR_piez sensor_fusionOR_acouORpiez ...
      sensor_IMUag_fusionOR_acou sensor_IMUag_fusionOR_piez sensor_IMUag_fusionOR_acouORpiez ...
      match_sensor_sensation_num match_sensor_IMUag_sensation_num
% ---------------------------------------------------------------------------------------------------------------

cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_003AM');
load('FM02_b_mat_003AM_focus_proc.mat');
for i = 1 : size(sensT_mapIMUag, 1)
    [tmp_sens_mapIMUag_label_003AM, tmp_sens_mapIMUag_comp_003AM] = bwlabel(sensT_mapIMUag{i, :});
    [tmp_sensor_IMUag_fusionOR_acouORpiez_label_003AM, tmp_sensor_IMUag_fusionOR_acouORpiez_comp_003AM] = bwlabel(sensor_IMUag_fusionOR_acouORpiez{i, :});
    
    sens_mapIMUag_comp_003AM(i, :) = tmp_sens_mapIMUag_comp_003AM;
    sensor_IMUag_fusionOR_acouORpiez_comp_003AM(i, :) = tmp_sensor_IMUag_fusionOR_acouORpiez_comp_003AM;
end
cd(curr_dir);

precision_003AM = sum(match_sensor_IMUag_sensation_num) / sum(sensor_IMUag_fusionOR_acouORpiez_comp_003AM);
sensitivity_003AM = sum(match_sensor_IMUag_sensation_num) / sum(sens_mapIMUag_comp_003AM);

% manual for perfection
match_sensor_IMUag_sensation_num_P = match_sensor_IMUag_sensation_num;
match_sensor_IMUag_sensation_num_P(8) = [];
match_sensor_IMUag_sensation_num_P(5) = [];
match_sensor_IMUag_sensation_num_P(1) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_003AM_P = sensor_IMUag_fusionOR_acouORpiez_comp_003AM;
sensor_IMUag_fusionOR_acouORpiez_comp_003AM_P(8) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_003AM_P(5) = [];
sensor_IMUag_fusionOR_acouORpiez_comp_003AM_P(1) = [];
sens_mapIMUag_comp_003AM_P = sens_mapIMUag_comp_003AM;
sens_mapIMUag_comp_003AM_P(8) = [];
sens_mapIMUag_comp_003AM_P(5) = [];
sens_mapIMUag_comp_003AM_P(1) = [];
precision_003AM_P = sum(match_sensor_IMUag_sensation_num_P) / sum(sensor_IMUag_fusionOR_acouORpiez_comp_003AM_P);
sensitivity_003AM_P = sum(match_sensor_IMUag_sensation_num_P) / sum(sens_mapIMUag_comp_003AM_P);

clear IMUa_thresh SNR_IMUa IMUa_map IMUg_thresh SNR_IMUg IMUg_map  ...
      sens sens_map sensT_mapIMUa sensT_mapIMUg sensT_mapIMUag  ...
      sens_label sens_activity sens_map_label sens_map_activity  ...
      sens_mapIMUa_label sens_mapIMUa_activity sens_mapIMUg_label sens_mapIMUg_activity  ...
      sens_mapIMUag_label sens_mapIMUag_activity  ...
      sensor_suite_preproc  ... 
      FM_thresh SNR_FM  ...
      FM_segmented FM_segmented_IMUa FM_segmented_IMUg FM_segmented_IMUag  ...
      sensor_fusionOR_acou sensor_fusionOR_piez sensor_fusionOR_acouORpiez ...
      sensor_IMUag_fusionOR_acou sensor_IMUag_fusionOR_piez sensor_IMUag_fusionOR_acouORpiez ...
      match_sensor_sensation_num match_sensor_IMUag_sensation_num
% ----------------------------------------------------------------------------------------------------------

% bar chart: the number focus vs day vs night sessions
figure
X = [1 2 3];
Y = [precision_001RB_P sensitivity_001RB_P; 
     precision_002JG_P sensitivity_002JG_P; 
     precision_003AM_P sensitivity_003AM_P];
b = bar(X, Y);

xticklabels({'P01', 'P02' , 'P03'}); ylim([0 1])
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


% focus sessions: 
% the number of button presses
% precision - maternal preception vs maternal movement
% precision_001RB = sens_mapIMUa_activity_001RB ./ sens_map_activity_001RB;
% precision_002JG = sens_mapIMUa_activity_002JG ./ sens_map_activity_002JG;
% precision_003AM = sens_mapIMUa_activity_002JG ./ sens_map_activity_002JG;
% yTop_001RB = max(sens_map_activity_001RB') * 2 + 5;
% yTop_002JG = max(sens_map_activity_002JG') * 2 + 5;
% 
% figure
% X_001RB = 1 : size(precision_001RB, 1);
% Y_001RB = [sens_map_activity_001RB' - sens_mapIMUa_activity_001RB'; sens_mapIMUa_activity_001RB'];
% b_001RB = bar(X_001RB, Y_001RB, 'stacked');
% 
% xticklabels({'Session 01', 'Session 02', 'Session 03', 'Session 04', 'Session 05'}); ylim([0 yTop_001RB])
% legend({'body movement (False positive)', 'fetal movement (true positive)'});
% 
% xtips2_001RB = b_001RB(2).XEndPoints;
% ytips2_001RB = b_001RB(2).YEndPoints;
% 
% for i = 1 : size(precision_001RB, 1)
%     labels2_001RB{i} = sprintf('%0.2f%%', precision_001RB(i)*100);
% end
% 
% text(xtips2_001RB, ytips2_001RB, labels2_001RB, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% % ---------------------------------------------------------------------------------------------------------------
% 
% figure
% X_002JG = 1 : size(precision_002JG, 1);
% Y_002JG = [sens_map_activity_002JG' - sens_mapIMUa_activity_002JG'; sens_mapIMUa_activity_002JG'];
% b_002JG = bar(X_002JG, Y_002JG, 'stacked');
% 
% xticklabels({'Session 01', 'Session 02', 'Session 03', 'Session 04', 'Session 05', 'Session 06', 'Session 07', 'Session 08', 'Session 09'}); ylim([0 yTop_002JG])
% legend({'body movement (False positive)', 'fetal movement (true positive)'});
% 
% xtips2_002JG = b_002JG(2).XEndPoints;
% ytips2_002JG = b_002JG(2).YEndPoints;
% 
% for i = 1 : size(precision_002JG, 1)
%     labels2_002JG{i} = sprintf('%0.2f%%', precision_002JG(i)*100);
% end
% 
% text(xtips2_002JG, ytips2_002JG, labels2_002JG, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% ---------------------------------------------------------------------------------------------------------------









