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

% HOME
% curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';

% OFFICE
curr_dir = pwd;

cd(curr_dir)

% HOME
% fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data';
fdir = 'G:\My Drive\ic_welcom_leap_fm\a98_hometrial_data';

participant = '003AM';
data_folder = ['c_processed_' participant];

% load preprocessed and processed data
data_file1 = [fdir '\' data_folder '\FM01_b_mat_003AM_focus_preproc.mat'];
data_file2 = [fdir '\' data_folder '\FM02_b_mat_003AM_focus_proc.mat'];
load(data_file1);
load(data_file2);

% the number of data file
nfile = size(sens_map, 1);
% -------------------------------------------------------------------------

% Frequency of sensor / sensation data sampling in Hz
% the number of channels / FM sensors
freq = 400; 
num_channel = 16;    
num_FMsensors = 6; 

% L or R of each type of sensor
sensor_suite_segmented_allOR = [sensor_fusionOR_acou, sensor_fusionOR_piez];

% selected date files:
% focus: 7
% day: 1
% night: 4
selected_file = 2;

for i = selected_file
    
    % Initialization of variables
    sensor_suite_segmented_numS1{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numS2{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numS3{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numS4{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numS5{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numS6{i,:} = zeros(length(FM_segmented{i,1}),1);

    sensor_suite_segmented_numT1{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numT2{i,:} = zeros(length(FM_segmented{i,1}),1);
    sensor_suite_segmented_numT3{i,:} = zeros(length(FM_segmented{i,1}),1);

    for l = 1 : sensor_fusionOR_acouORpiez_comp{i, :}

        % indices of each detection
        tmp_idxS = find(sensor_fusionOR_acouORpiez_label{i,:} == l, 1 ); 
        tmp_idxE = find(sensor_fusionOR_acouORpiez_label{i,:} == l, 1, 'last' );

        % binary map to indicate each detection
        tmp_map = zeros(length(FM_segmented{i,1}),1); 
        tmp_map(tmp_idxS:tmp_idxE) = 1; 
    
        % the number of involved sensors for each detection
        tmp_numS = 0; 
        for j = 1 : num_FMsensors
            if sum(tmp_map.*FM_segmented{i,j}) > 0
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
        for j = 1 : num_FMsensors/3
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
            otherwise
                disp('This would never print.')
        end

        clear tmp_idxS tmp_idxE tmp_map tmp_numS
    end
end
% -------------------------------------------------------------------------

%% PLOTTING -- time series
% Parameters for plotting
IMUa_thresh = 0.003;
IMUg_thresh = 0.003;

sntn_multiplier = 1;
IMU_multiplier = 1;
fm_multiplier = 0.9;
ylim_multiplier = 1.1;

col_code1 = [0 0 0];
col_code2 = [0.4660 0.6740 0.1880];
col_code3 = [0.4940 0.1840 0.5560];
col_code4 = [0.8500 0.3250 0.0980];
col_code5 = [0.9290 0.6940 0.1250];

Font_size_labels = 16;
Font_size_legend = 14;
box_width = 1; 
line_width = 1;

legend_pre = {'IMU accelerometer', 'IMU gyroscope', ...
              'L-acoustic', 'M-acoustic', 'R-acoustic', ...
              'L-piezoelectric', 'M-piezoelectric', 'R-piezoelectric',};

for i = selected_file

    tmp_sensor_timeV = (0 : length(IMUaccNet_TF{i,:}) - 1) / freq;
    tmp_sensation_timeV = (0 : length(sens_T{i,:}) - 1) / freq;

    tmp_sensation_mtx = [tmp_sensation_timeV', sens_T{i,:}]; 
    tmp_sensation_idx = find(tmp_sensation_mtx(:,2)==1); 
    tmp_sensation_posi = tmp_sensation_mtx(tmp_sensation_idx,:); 

    % Plotting using tiles
    figure
    tiledlayout(16, 1, 'Padding', 'tight', 'TileSpacing', 'none');
    
    % Tile 1-2: IMU accelerometer
    nexttile
    hold on;
    plot(tmp_sensor_timeV, IMUaccNet_TF{i,:}, 'LineWidth', line_width); 
    plot(tmp_sensor_timeV, IMUa_map{i,:}*IMUa_thresh, 'Color', col_code1, 'LineWidth', line_width);
    ylim([min(IMUaccNet_TF{i,:})*ylim_multiplier, max(IMUaccNet_TF{i,:})*ylim_multiplier]);
    hold off;
    axis off; 
    
    lgd = legend(legend_pre{1}, 'Body movement map (accelerometer)');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;

    nexttile
    hold on;
    plot(tmp_sensor_timeV, IMUgyrNet_TF{i,:}, 'LineWidth', line_width); 
    plot(tmp_sensor_timeV, IMUg_map{i,:}*IMUg_thresh, 'Color', col_code1, 'LineWidth', line_width);
    ylim([min(IMUgyrNet_TF{i,:})*ylim_multiplier, max(IMUgyrNet_TF{i,:})*ylim_multiplier]);
    hold off;
    axis off; 
    
    lgd = legend(legend_pre{2}, 'Body movement map (gyroscope)');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    % Tile 3-8: FM sensor suite
    for j = 1 : num_FMsensors
        nexttile
        hold on;
        plot(tmp_sensor_timeV, sensor_suite{i, j},'LineWidth', line_width)
        plot(tmp_sensor_timeV, FM_segmented{i, j}*max(sensor_suite{i, j})*fm_multiplier,'color', col_code2, 'LineWidth', line_width)
        plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*max(sensor_suite{i, j})*fm_multiplier/2, 'r.', 'LineWidth', line_width);
        ylim([min(sensor_suite{i, j})*ylim_multiplier, max(sensor_suite{i, j})*ylim_multiplier]);
        hold off;
        axis off;

        lgd = legend (legend_pre{j+1}, 'FM detection');
        lgd.FontName = 'Times New Roman';
        lgd.FontSize = Font_size_legend;
        lgd.NumColumns = 1;
        legend('Location','northeastoutside')
        legend boxoff;
    end
    % -----------------------------------------------------

    % Tile 9-14: sensor fusion - acouL | acouM |acouR | piezL | piezM |piezR
    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numS1{i,:}*1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r.', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least one sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numS2{i,:}*1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r.', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least two sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numS3{i,:}*1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r.', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least three sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numS4{i,:}*1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r.', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least four sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numS5{i,:}*1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r.', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least five sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numS6{i,:}*1.5, 'color', col_code4, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r.', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least six sensor');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    % Tile 15-16: acou | piez
    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numT1{i,:}*1.5, 'color', col_code5, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r*', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least one sensor type');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------

    nexttile
    hold on
    plot(tmp_sensor_timeV, sensor_suite_segmented_numT2{i,:}*1.5, 'color', col_code5, 'LineWidth', line_width)
    plot(tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*0.75, 'r*', 'LineWidth', line_width);
    ylim([-1, 2*ylim_multiplier]);
    hold off;
    axis off;

    lgd = legend('FM detection by at least two sensor type');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = Font_size_legend;
    lgd.NumColumns = 1;
    legend('Location','northeastoutside')
    legend boxoff;
    % -----------------------------------------------------
    
    set(gca, 'FontName', 'Times New Roman', 'FontSize', Font_size_labels)
    set(gca,'linewidth', box_width)
    set(gca,'TickLength',[0.005, 0.01])

    clear tmp_sensor_timeV tmp_sensation_timeV tmp_sensation_mtx tmp_sensation_idx tmp_sensation_posi
end

%% PLOTTING: raw data
% Trim length of frequency in Hz;
freq_trim_raw = 0;

% X axis starting / end limit in Hz
xlimS_raw = 1; 
xlimE_raw = 50;

% multiply the amplitude of sensation data to plot properly
amp_multiplier_raw = 1; 

title_raw = {'Raw Sensor response', 'Single-Sided Amplitude Spectrum', ...
             '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''};
legend_raw = {'Flexi force', ...
              'IMU accelerometer', 'IMU gyroscope', ...
              'Acoustic sensor L', 'Acoustic sensor M', 'Acoustic sensor R', ...
              'Piezoelectric plate sensor L', 'Piezoelectric plate sensor M', 'Piezoelectric plate sensor R'};
xlabel_raw = {'', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ...
              'Time (s)', 'Frequency (Hz)'};

for i = selected_file
    
    % get the spectrum data
    [tmp_forc_spectrum,~,~] = get_frequency_mode(forc_T{i,:},freq,freq_trim_raw);
    [tmp_IMUacc_spectrum,~,~] = get_frequency_mode(IMUaccNet_T{i,:},freq,freq_trim_raw);
    [tmp_IMUgyr_spectrum,~,~] = get_frequency_mode(IMUgyrNet_T{i,:},freq,freq_trim_raw);
    [tmp_acouL_spectrum,tmp_freq_vSD1,~] = get_frequency_mode(acouL_T{i,:},freq,freq_trim_raw);
    [tmp_acouM_spectrum,tmp_freq_vSD1,~] = get_frequency_mode(acouM_T{i,:},freq,freq_trim_raw);
    [tmp_acouR_spectrum,tmp_freq_vSD1,~] = get_frequency_mode(acouR_T{i,:},freq,freq_trim_raw);
    [tmp_piezL_spectrum,~,~] = get_frequency_mode(piezL_T{i,:},freq,freq_trim_raw);
    [tmp_piezM_spectrum,~,~] = get_frequency_mode(piezM_T{i,:},freq,freq_trim_raw);
    [tmp_piezR_spectrum,~,~] = get_frequency_mode(piezR_T{i,:},freq,freq_trim_raw);

    % time vector for the sensor and sensation data in SD1 / SD2
    tmp_sensor_timeV_SD = (0:(length(acouL_T{i,:})-1))/freq;
    tmp_sensation_timeV = (0:(length(sens_T{i,:})-1))/freq;
    tmp_sensation_mtx = [tmp_sensation_timeV', sens_T{i,:}];
    tmp_sensation_idx = find(tmp_sensation_mtx(:,2)==1); 
    tmp_sensation_posi = tmp_sensation_mtx(tmp_sensation_idx,:); 

    tmp_sensor_suite = {forc_T{i,:}, IMUaccNet_T{i,:}, IMUgyrNet_T{i,:}, ...
                        acouL_T{i,:}, acouM_T{i,:}, acouR_T{i,:}, ...
                        piezL_T{i,:}, piezM_T{i,:}, piezR_T{i,:}};
    tmp_sensor_suite_timeV = {tmp_sensor_timeV_SD', tmp_sensor_timeV_SD', tmp_sensor_timeV_SD', ...
                              tmp_sensor_timeV_SD', tmp_sensor_timeV_SD', tmp_sensor_timeV_SD', ...
                              tmp_sensor_timeV_SD', tmp_sensor_timeV_SD', tmp_sensor_timeV_SD'};
    tmp_spectrum_suite = {tmp_forc_spectrum, tmp_IMUacc_spectrum, tmp_IMUgyr_spectrum, ...
                          tmp_acouL_spectrum, tmp_acouM_spectrum, tmp_acouR_spectrum, ...
                          tmp_piezL_spectrum, tmp_piezM_spectrum, tmp_piezR_spectrum};
    tmp_freq_vec_suite = {tmp_freq_vSD1, tmp_freq_vSD1, tmp_freq_vSD1, ...
                          tmp_freq_vSD1, tmp_freq_vSD1, tmp_freq_vSD1, ...
                          tmp_freq_vSD1, tmp_freq_vSD1, tmp_freq_vSD1};

    % Plotting using subplot
    figure
    for j = 1 : num_FMsensors+3

        subplot(9, 2, 2*j-1);
        plot(tmp_sensor_suite_timeV{j}, tmp_sensor_suite{j}, ...
             tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*max(tmp_sensor_suite{j})*amp_multiplier_raw, 'go', 'LineWidth', 1);
        title(title_raw{2*j-1});
        legend (legend_raw{j}, 'Sensation data');
        legend boxoff;
        ylabel('Response (V)');
        xlabel(xlabel_raw{2*j -1});

        subplot(9, 2, 2*j);
        plot(tmp_freq_vec_suite{j}, tmp_spectrum_suite{j});
        %xlim([xlimS_raw xlimE_raw]);
        title(title_raw{2*j})
        legend (legend_raw{j});
        legend boxoff;
        ylabel('|F coefficient|');
        xlabel(xlabel_raw{2*j});
    end

    clear tmp_forc_spectrum tmp_IMUacc_spectrum tmp_IMUgyr_spectrum ...
          tmp_acouL_spectrum tmp_acouM_spectrum tmp_acouR_spectrum ...
          tmp_piezL_spectrum tmp_piezM_spectrum tmp_piezR_spectrum tmp_freq_vec ...
          tmp_sensor_timeV_SD tmp_sensor_timeV_SD tmp_sensation_timeV ...
          tmp_sensation_mtx tmp_sensation_idx tmp_sensation_posi ...
          tmp_sensor_suite tmp_sensor_suite_timeV tmp_spectrum_suite tmp_freq_vec_suite
end

%% PLOTTING frequency mode: preprocessed data
% Trim length of frequency in Hz;
freq_trim_pre = 0;

% X axis starting / end limit in Hz
xlimS_pre = 0.0; 
xlimE_pre = 30;

% multiply the amplitude of sensation data to plot properly
amp_multiplier_pre = 1; 

title_pre = {'Raw Sensor response', 'Filtered response' , 'Single-Sided Amplitude Spectrum', ...
             '', '', '', '', '', '', '', '', '', '',  '', '', '', ...
             '', '', '', '', '', '', '', '', '', '', ''};
legend_pre = {'Flexi force', 'IMU accelerometer', 'IMU gyroscope', ...
              'Acoustic sensor L', 'Acoustic sensor M', 'Acoustic sensor R', ...
              'Piezoelectric plate sensor L', 'Piezoelectric plate sensor M', 'Piezoelectric plate sensor R'};
xlabel_pre = {'', '', '', '', '', '', '','', '', '', ...
              '', '', '', '', '', '', '', '', '', '', '', '', '', '', ...
              'Time (s)', 'Time (s)', 'Frequency (Hz)'};

for i = selected_file

    % Get the spectrum data
    [tmp_forc_spectrum,tmp_freq_vec,~] = get_frequency_mode(forc_TF{i,:},freq,freq_trim_pre);
    [tmp_IMUacc_spectrum,~,~] = get_frequency_mode(IMUaccNet_TF{i,:},freq,freq_trim_pre);
    [tmp_IMUgyr_spectrum,~,~] = get_frequency_mode(IMUgyrNet_TF{i,:},freq,freq_trim_pre);
    
    [tmp_acouL_spectrum,~,~] = get_frequency_mode(acouL_TF{i,:},freq,freq_trim_pre);
    [tmp_acouM_spectrum,~,~] = get_frequency_mode(acouM_TF{i,:},freq,freq_trim_pre);
    [tmp_acouR_spectrum,~,~] = get_frequency_mode(acouR_TF{i,:},freq,freq_trim_pre);

    [tmp_piezL_spectrum,~,~] = get_frequency_mode(piezL_TF{i,:},freq,freq_trim_pre);
    [tmp_piezM_spectrum,~,~] = get_frequency_mode(piezM_TF{i,:},freq,freq_trim_pre);
    [tmp_piezR_spectrum,~,~] = get_frequency_mode(piezR_TF{i,:},freq,freq_trim_pre);
    
    % time coverage
    tmp_sensor_timeV = (0:(length(forc_TF{i,:})-1)) / freq; 
    tmp_sensation_timeV = (0:(length(sens_T{i,:})-1)) / freq;
    tmp_sensation_mtx = [tmp_sensation_timeV', sens_T{i}];
    tmp_sensation_idx = find(tmp_sensation_mtx(:,2)==1);
    tmp_sensation_posi = tmp_sensation_mtx(tmp_sensation_idx,:);

    tmp_sensor_T = {forc_T{i,:}, IMUaccNet_T{i,:}, IMUgyrNet_T{i,:}, ...
                      acouL_T{i,:}, acouM_T{i,:}, acouR_T{i,:}, ...
                      piezL_T{i,:}, piezM_T{i,:}, piezR_T{i,:}};

    tmp_sensor_TF = {forc_TF{i,:}, IMUaccNet_TF{i,:}, IMUgyrNet_TF{i,:}, ...
                         acouL_TF{i,:}, acouM_TF{i,:}, acouR_TF{i,:}, ...
                         piezL_TF{i,:}, piezM_TF{i,:}, piezR_TF{i,:}};

    tmp_spectrum = {tmp_forc_spectrum, tmp_IMUacc_spectrum, tmp_IMUgyr_spectrum, ...
                    tmp_acouL_spectrum, tmp_acouM_spectrum, tmp_acouR_spectrum, ... 
                    tmp_piezL_spectrum, tmp_piezM_spectrum, tmp_piezR_spectrum};

    % plot sensor suite with subplots
    % 8-row: force sensor, IMU accelerometer and 6 FM sensors     
    % 3-column: raw, filter&trimmed, spectrum
    figure
    for j = 1 : num_FMsensors + 3

        subplot(9,3,3*j-2);
        plot(tmp_sensor_timeV, tmp_sensor_T{j}, ...
             tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*max(tmp_sensor_T{j})*amp_multiplier_pre, ...
             'go', 'LineWidth', 1);
        title(title_pre{3*j-2});
        legend (legend_pre{j}, 'Sensation data (no filter)');
        legend boxoff;
        xlabel(xlabel_pre{3*j-2});
        ylabel('Response (V)');
        
        subplot(9,3,3*j-1); 
        plot(tmp_sensor_timeV, tmp_sensor_TF{j}, ...
             tmp_sensation_posi(:,1), tmp_sensation_posi(:,2)*max(tmp_sensor_TF{j})*amp_multiplier_pre, ...
             'go', 'LineWidth', 1);
        title(title_pre{3*j-1});
        legend (legend_pre{j}, 'Sensation data (filtered)');
        legend boxoff;
        xlabel(xlabel_pre{3*j-1});
        ylabel('AU');
        %

        subplot(9,3,3*j);
        plot(tmp_freq_vec, tmp_spectrum{j}, 'LineWidth', 1);
        title(title_pre{3*j});
        legend (legend_pre{j});
        legend boxoff;
        xlabel(xlabel_pre{3*j});
        ylabel('|F coefficient|');
        %
    end

    clear tmp_forc_spectrum tmp_IMUacc_spectrum tmp_IMUgyr_spectrum ...
          tmp_acouL_spectrum tmp_acouM_spectrum tmp_acouR_spectrum ...
          tmp_piezL_spectrum tmp_piezM_spectrum tmp_piezR_spectrum tmp_freq_vec ...
          tmp_sensor_timeV tmp_sensation_timeV ...
          tmp_sensation_mtx tmp_sensation_idx tmp_sensation_posi ...
          tmp_sensor_T tmp_sensor_TF tmp_spectrum
end





% %% PLOTTING: filter characteristic
% % Butterworth digital filter: [B,A] = butter(n,Wn)
% % n: the order of filter
% % Wn = thresh*2 / frequency
% % y = filter(B,A,x)
% 
% % filter order
% fil_order = 10;
% 
% % Bandpass filter
% % IMU accelerometer: a passband of 1-10 Hz
% % FM sensor: a passband of 1-30Hz
% cutL_IMUacce = 1;
% cutH_IMUacce = 10;
% cutL_FM = 1;
% cutH_FM = 30;
% 
% % lowpass filter
% cutH_forc = 10;
% 
% % Transfer function-based desing
% [b_FM,a_FM] = butter(fil_order/2,[cutL_FM cutH_FM]/(freq/2),'bandpass');
% [b_IMUacce,a_IMUacce] = butter(fil_order/2,[cutL_IMUacce cutH_IMUacce]/(freq/2),'bandpass');
% 
% % Zero-Pole-Gain-based design
% % - filter order for bandpass filter is twice the value of 1st parameter
% % - Convert zero-pole-gain filter parameters to second-order sections form
% [z_FM,p_FM,k_FM] = butter(fil_order/2,[cutL_FM cutH_FM]/(freq/2),'bandpass'); 
% [sos_FM,g_FM] = zp2sos(z_FM,p_FM,k_FM); 
% 
% [z_IMUacce,p_IMUacce,k_IMUacce] = butter(fil_order/2,[cutL_IMUacce cutH_IMUacce]/(freq/2),'bandpass');
% [sos_IMUacce,g_IMUacce] = zp2sos(z_IMUacce,p_IMUacce,k_IMUacce);
% 
% % Low-pass filter: * for the force sensor data only
% % - Convert zero-pole-gain filter parameters to second-order sections form
% [z_forc,p_forc,k_forc] = butter(fil_order,cutH_forc/(freq/2),'low');
% [sos_forc,g_forc] = zp2sos(z_forc,p_forc,k_forc); 
% 
% % IR notch filter: removing noise due to SD card writing
% % base frequency
% notch_fb = 32; 
% 
% % normalized location of the notch
% w0 = notch_fb/(freq/2); 
% 
% % Q factor
% q = 35; 
% 
% % bandwidth at -3dB level
% notch_bw = w0/q; 
% [b_notch,a_notch] = iirnotch(w0,notch_bw);
% 
% % 4 x 1024
% n_point = 8192; 
% 
% % n_point frequency response in h and corresponding physical frequency in w
% [h_FM, w] = freqz(zp2sos(z_FM,p_FM,k_FM), n_point, freq); 
% [h_IMUacce, ~] = freqz(zp2sos(z_IMUacce,p_IMUacce,k_IMUacce), n_point, freq);
% [h_forc, ~] = freqz(zp2sos(z_forc,p_forc,k_forc), n_point, freq);
% [h_notch, ~] = freqz(b_notch, a_notch, n_point, freq);
% 
% % direct current
% % [h_DC, ~] = freqz(b_DC, a_DC, n_point, freq_sensor);
% 
% % Atteneution at the cutoff frequencies
% indx_01Hz = find(w == 1);
% indx_10Hz = find(w == 10);
% indx_30Hz = find(w == 30);
% indx_32Hz = find(w == 32);
% 
% % Atntn_1Hz = [db(h_FM(indx_1Hz)), db(h_IMUacce(indx_1Hz)), db(h_forc(indx_1Hz)), db(h_DC(indx_1Hz)), db(h_notch(indx_1Hz))];
% % Atntn_10Hz = [db(h_FM(indx_10Hz)), db(h_IMUacce(indx_10Hz)), db(h_forc(indx_10Hz)), db(h_DC(indx_10Hz)), db(h_notch(indx_10Hz))];
% % Atntn_30Hz = [db(h_FM(indx_20Hz)), db(h_IMUacce(indx_20Hz)), db(h_forc(indx_20Hz)), db(h_DC(indx_20Hz)), db(h_notch(indx_20Hz))];
% % Atntn_32Hz = [db(h_FM(indx_32Hz)), db(h_IMUacce(indx_32Hz)), db(h_forc(indx_32Hz)), db(h_DC(indx_32Hz)), db(h_notch(indx_32Hz))];
% 
% Atntn_01Hz = [db(h_FM(indx_01Hz)), db(h_IMUacce(indx_01Hz)), db(h_forc(indx_01Hz)), db(h_notch(indx_01Hz))];
% Atntn_10Hz = [db(h_FM(indx_10Hz)), db(h_IMUacce(indx_10Hz)), db(h_forc(indx_10Hz)), db(h_notch(indx_10Hz))];
% Atntn_30Hz = [db(h_FM(indx_30Hz)), db(h_IMUacce(indx_30Hz)), db(h_forc(indx_30Hz)), db(h_notch(indx_30Hz))];
% Atntn_32Hz = [db(h_FM(indx_32Hz)), db(h_IMUacce(indx_32Hz)), db(h_forc(indx_32Hz)), db(h_notch(indx_32Hz))];
% 
% % Plotting the magnitude response
% % semilogx(w, db(h_IMUacce), w, db(h_FM), w, db(h_forc), w, db(h_DC), w, db(h_notch), 'LineWidth', 1)
% semilogx(w, db(h_IMUacce), w, db(h_FM), w, db(h_forc), w, db(h_notch), 'LineWidth', 1)
% xlim([0.1 100]);
% ylim([-150 50]);
% title('Magnitude response of different filters');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% % legend('Band-pass filter: 1-10 Hz','Band-pass filter: 1-30 Hz', 'Low-pass filter: 10 Hz', 'DC removal filter', 'IIR notch filter: notch at 32 Hz', 'Location', 'best');
% legend('Band-pass filter: 1-10 Hz','Band-pass filter: 1-30 Hz', 'Low-pass filter: 10 Hz', 'IIR notch filter: notch at 32 Hz', 'Location', 'best');
% legend boxoff;
% 
% grid on;
% ax = gca;
% ax.FontSize = 12;
% 
% % Display and compare the filters using fvtool()
% % hfvt = fvtool(b_FM,a_FM, ...
% %               zp2sos(z_FM,p_FM,k_FM), ...
% %               zp2sos(z_FM_cheby,p_FM_cheby,k_FM_cheby),...
% %               b_DC,a_DC, ...
% %               b_notch, a_notch, ...
% %               'FrequencyScale','log');
% % legend(hfvt,'Butterworth: TF-based design', ...
% %             'Butterworth: Zero-Pole-Gain-based design',...
% %             'Chebyshev2: Zero-Pole-Gain-based design', ...
% %             'DC removal filter', ...
% %             'Notch filter')
% 
% hfvt = fvtool(b_FM,a_FM, ...
%               zp2sos(z_FM,p_FM,k_FM), ...
%               b_notch, a_notch, ...
%               'FrequencyScale','log');
% legend(hfvt,'Butterworth: TF-based design', ...
%             'Butterworth: Zero-Pole-Gain-based design',...
%             'Notch filter')
% legend boxoff;
% % -------------------------------------------------------------------------------------------------
% 
%% FUNCTION: get_frequency_mode
%  Retun the main frequency mode of the input data
%  Input variables: 
%  data_in - time series data
%  freq - sensor/sensation sampling rate
%  trim_hz - length of triming in Hz to retain mode above a certain frequncy
%   
%  Output variable: 
%  data_spectrum - A vector of Fourier coefficient values
%  freq_vector - Corresponding frequency vector
%  freq_mode - the main frequency mode
%  --------------------------------------------------------------------------------
function [data_spectrum,freq_vector,freq_mode] = get_frequency_mode(data_in, freq, trim_hz)

% Generating the frequency vector of the spectrum
L = length(data_in); 

% L/Fs: the number of points/Hz in the frequency spectrum.
freq_trim_indx = ceil(trim_hz*L/freq+1);

% There are L/2 points in the single-sided spectrum.
% Each point will be Fs/L apart.
freq_vector = freq*(0:floor(L/2))/L; 
freq_vector_trimed = freq_vector(freq_trim_indx:end);

% Fourier transform
% Two-sided Fourier spectrum. 
% Normalizing by L is generally performed during fft so that it is not neede for inverse fft
ft_2sided = abs(fft(data_in))/L; 
ft_1sided = ft_2sided(1:floor(L/2)+1); 

% multiplication by 2 is used to maintain the conservation of energy
ft_1sided(2:end-1) = 2 * ft_1sided(2:end-1); 
ft_1sided_trimed = ft_1sided(freq_trim_indx:end);

% Main frequency mode
[~,I] = max(ft_1sided_trimed); 
freq_mode_data = freq_vector_trimed(I);

% Finalizing the output variables
freq_vector = freq_vector_trimed;
data_spectrum = ft_1sided_trimed;
freq_mode = freq_mode_data;

end










