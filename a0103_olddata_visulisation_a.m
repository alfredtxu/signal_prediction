% -------------------------------------------------------------------------
% SD1:
% Channel 1: Flexi force sensor
% Channel 2: Piezoelectric plate sensor 1 (Left belly)
% Channel 3: Piezoelectric plate sensor 2 (Right belly)
% Channel 4: Acoustic sensor 1 (Left belly)
% Channel 5-7: IMU data (Accelerometer)
% Channel 8: Maternal senstation
% 
% SD2:
% Channel 1: Accoustic sensor 2 (Right belly)
% Channel 2-4: Accelerometer 2 (Right belly)
% Channel 5-7: Accelerometer 1 (Left belly)
% Channel 8: Maternal sensation
% -------------------------------------------------------------------------
clc
clear
close all

curr_dir = pwd;
cd(curr_dir);

% Add paths of the folders with all necessary function files
% SP_function_files: signal processing algorithms
% ML_function_files: machine learning algorithms
% Learned_models: learned models
% matplotlib colormaps: matplotlib_function_files
addpath(genpath('SP_function_files')) 
addpath(genpath('ML_function_files')) 
addpath(genpath('Learned_models')) 
addpath(genpath('matplotlib_function_files')) 
addpath(genpath('z11_olddata_mat_raw')) 
addpath(genpath('z12_olddata_mat_preproc')) 
addpath(genpath('z13_olddata_mat_proc')) 

% LOADING PRE-PROCESSED DATA
% ** set current loading data portion
participant = 'S1';
load(['sensor_data_suite_' participant '_preproc.mat']);
load(['sensor_data_suite_' participant '_proc.mat']);

% the number of data file
nfile = size(sensation_map, 1);
% -------------------------------------------------------------------------

% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
freq_sensor = 1024; 
freq_sensation = 1024; 
num_channel = 8;    
num_FMsensors = 6; 

% L or R of each type of sensor
sensor_suite_segmented_allOR = [sensor_fusion1_accl, sensor_fusion1_acou, sensor_fusion1_piez];

for i = 1 : nfile
    
    sensor_suite{i, 1} = acceLNetFilTEq{i,:};
    sensor_suite{i, 2} = acceRNetFilTEq{i,:};
    sensor_suite{i, 3} = acouLFilTEq{i,:};
    sensor_suite{i, 4} = acouRFilTEq{i,:};
    sensor_suite{i, 5} = piezLFilTEq{i,:};
    sensor_suite{i, 6} = piezRFilTEq{i,:};

    % Initialization of variables
    sensor_suite_segmented_numS1{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numS2{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numS3{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numS4{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numS5{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numS6{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);

    sensor_suite_segmented_numT1{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numT2{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);
    sensor_suite_segmented_numT3{i,:} = zeros(length(sensor_suite_segmented{i,1}),1);

    for l = 1 : sensor_fusion1_all_comp{i, :}

        % indices of each detection
        tmp_idxS = find(sensor_fusion1_all_bw{i,:} == l, 1 ); 
        tmp_idxE = find(sensor_fusion1_all_bw{i,:} == l, 1, 'last' );

        % binary map to indicate each detection
        tmp_map = zeros(length(sensor_suite_segmented{i,1}),1); 
        tmp_map(tmp_idxS:tmp_idxE) = 1; 
    
        % the number of involved sensors for each detection
        tmp_numS = 0; 
        for j = 1 : num_FMsensors
            if sum(tmp_map.*sensor_suite_segmented{i,j}) > 0
                tmp_numS = tmp_numS + 1;
            end
        end
       
        switch tmp_numS
            case 1
                sensor_suite_segmented_numS1{i, :}(tmp_idxS:tmp_idxE) = 1;
            case 2
                sensor_suite_segmented_numS1{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS2{i, :}(tmp_idxS:tmp_idxE) = 1;
            case 3
                sensor_suite_segmented_numS1{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS2{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS3{i, :}(tmp_idxS:tmp_idxE) = 1;
            case 4
                sensor_suite_segmented_numS1{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS2{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS3{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS4{i, :}(tmp_idxS:tmp_idxE) = 1;
            case 5
                sensor_suite_segmented_numS1{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS2{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS3{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS4{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS5{i, :}(tmp_idxS:tmp_idxE) = 1;
            case 6
                sensor_suite_segmented_numS1{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS2{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS3{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS4{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS5{i, :}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numS6{i, :}(tmp_idxS:tmp_idxE) = 1;
            otherwise
                disp('This would never print.')
        end

        % the number of involved type of sensors for each detection
        % * num_FMsensors/2: the number of each type of sensor is two
        tmp_numT = 0;
        for j = 1 : num_FMsensors/2
            if (sum(tmp_map.*sensor_suite_segmented_allOR{i, j})) % Non-zero value indicates intersection
                tmp_numT = tmp_numT + 1;
            end
        end

        switch tmp_numT
            case 1
                sensor_suite_segmented_numT1{i,:}(tmp_idxS:tmp_idxE) = 1;
            case 2
                sensor_suite_segmented_numT1{i,:}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numT2{i,:}(tmp_idxS:tmp_idxE) = 1;
            case 3
                sensor_suite_segmented_numT1{i,:}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numT2{i,:}(tmp_idxS:tmp_idxE) = 1;
                sensor_suite_segmented_numT3{i,:}(tmp_idxS:tmp_idxE) = 1;
            otherwise
                disp('This would never print.')
        end

        clear tmp_idxS tmp_idxE tmp_map tmp_numS
    end
end
% -------------------------------------------------------------------------

%% PLOTTING
% Parameters for plotting
sntn_multiplier = 1;
IMU_multiplier = 1;
fm_multiplier = 0.9;
ylim_multiplier = 1.1;

IMUacce_thresh = [0.003 0.002];

col_code1 = [0 0 0];
col_code2 = [0.4660 0.6740 0.1880];
col_code3 = [0.4940 0.1840 0.5560];
col_code4 = [0.8500 0.3250 0.0980];
col_code5 = [0.9290 0.6940 0.1250];

% sampled period (seconds) for poltting (an example range: [550 650])
% 336 for the example used in the paper (data set no. 31)
% 536 for the example used in the paper (data set no. 31)
% i=8. maximum indix is 380640
% xlim_sampS = 550; 
% xlim_sampE = 650; 
xlim_sampS = 150; 
xlim_sampE = 350; 
Font_size_labels = 16;
Font_size_legend = 14;

% width of the box / line
box_width = 1; 
line_width = 1;

% 1 is added because Matlab index starts from 1. If xlim_1 = 0, index = 1
xlim_index1 = xlim_sampS*freq_sensor + 1; 
xlim_index2 = xlim_sampE*freq_sensor + 1;

% time vector index starts from 0s
sensor_timeV = (xlim_index1-1 : xlim_index2-1) / freq_sensor; 
sensation_timeV = (xlim_index1-1 : xlim_index2-1) / freq_sensation; 

% shift the time vector to 0
sensor_timeV = sensor_timeV - xlim_sampS; 
sensation_timeV = sensation_timeV-xlim_sampS;

% sensation extraction
sensation_mtx = [sensation_timeV', sens1TEq{1,:}(xlim_index1 : xlim_index2)]; 
sensation_idx = find(sensation_mtx(:,2)==1); 
sensation_pos = sensation_mtx(sensation_idx,:); 

legend_all = {'IMU accelerometer', ...
              'L-accelerometer', 'R-accelerometer', ...
              'L-acoustic', 'R-acoustic', ...
              'L-piezoelectric', 'R-piezoelectric',};
xlabel_all = {'', '', '', '', '', '', 'time (s)', 'time (s)'};

for i = 1 : nfile

    % Plotting using tiles
    figure
    tiledlayout(11, 1, 'Padding', 'tight', 'TileSpacing', 'none');
    
    % Tile 1: maternal senstaiton
    nexttile
    plot(sensation_timeV, sens2TEq{i,:}(xlim_index1:xlim_index2), 'go','LineWidth', line_width);
    plot(sensation_timeV, sensation_map{i,:}(xlim_index1:xlim_index2), 'color', col_code2, 'LineWidth', line_width)
    ylim([-1, 2*y_lim_multiplier]);
    hold on
    axis off;

    lgd = legend('Maternal Sensation');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;

    % Tile 2: IMU accelerometer
    nexttile
    hold on;
    plot(sensor_timeV, IMUacceNetFilTEq{i,:}(xlim_index1:xlim_index2), 'LineWidth', line_width); 
    plot(sensor_timeV, IMUacce_map{i,:}(xlim_index1:xlim_index2)*IMUacce_thresh(1), 'Color', col_code1, 'LineWidth', line_width);
    ylim([min(IMUacceNetFilTEq{i,:}(xlim_index1:xlim_index2))*ylim_multiplier, max(IMUacceNetFilTEq{i,:}(xlim_index1:xlim_index2))*ylim_multiplier]);
    hold off;
    axis off; 
    
    lgd= legend(legend_all{1}, 'Body movement map');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;

    % Tile 3-8: FM sensor suite
    for j = 1 : num_FMsensors
        nexttile
        hold on;
        plot(sensor_timeV, sensor_suite{i, j}(xlim_index1:xlim_index2),'LineWidth', line_width)
        plot(sensor_timeV, sensor_suite_segmented{i, j}(xlim_index1:xlim_index2)*max(sensor_suite{i, j}(xlim_index1:xlim_index2))*fm_multiplier,'color', [0.4660 0.6740 0.1880], 'LineWidth', line_width)
        plot(sensation_pos(:,1), sensation_pos(:,2)*max(sensor_suite{i, j}(xlim_index1:xlim_index2))*fm_multiplier/2, 'r*', 'LineWidth', line_width);
        ylim([min(sensor_suite{i, j}(xlim_index1:xlim_index2))*ylim_multiplier, max(sensor_suite{i, j}(xlim_index1:xlim_index2))*ylim_multiplier]);
        hold off;
        axis off;

        lgd= legend (legend_all{j+1}, 'FM detection');
        lgd.FontName = 'Times New Roman';
        lgd.FontSize = Font_size_legend;
        lgd.NumColumns = 1;
        legend('Location','northeastoutside')
        legend boxoff;
    end

    % Tile 9: sensor fusion - acceL | acceR | acouL | acouR | piezL | piezR
    nexttile
    hold on
    plot(sensor_timeV, sensor_fusion1_all{i,:}(xlim_index1:xlim_index2) * 1.5, 'color', col_code3, 'LineWidth', line_width)
    plot(sensation_pos(:,1), sensation_pos(:,2) * 0.75, 'r*', 'LineWidth', line_width);
    ylim([-1, 2*y_lim_multiplier]);
    hold off;
    axis off;

    lgd = legend('Sensor fusion: at least one sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;

    % Tile 10: FM detection by at least one type of sensors 
    nexttile
    hold on
    plot(sensor_timeV, sensor_suite_segmented_numT1{i,:}(xlim_index1:xlim_index2) * 1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(sensation_pos(:,1), sensation_pos(:,2) * 0.75, 'r*', 'LineWidth', line_width);
    ylim([-1, 2*y_lim_multiplier]);
    hold off;
    axis off;

    lgd = legend('Sensor fusion: at least one type of sensors');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;

    % Tile 11: FM detection by at least two types of sensors 
    nexttile
    hold on
    plot(sensor_timeV, sensor_suite_segmented_numT2{i,:}(xlim_index1:xlim_index2) * 1.5, 'color', col_code5, 'LineWidth', line_width)
    plot(sensation_pos(:,1), sensation_pos(:,2) * 0.75, 'r*', 'LineWidth', line_width);
    ylim([-1, 2*y_lim_multiplier]);
    hold off;
    axis off;

    lgd = legend('Sensor fusion: at least two types of sensors');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;

    set(gca, 'FontName', 'Times New Roman', 'FontSize', Font_size_labels)
    set(gca,'linewidth', box_width)
    set(gca,'TickLength',[0.005, 0.01])
end
















