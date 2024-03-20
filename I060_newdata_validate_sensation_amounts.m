clc
clear 
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir);

%% Loading data
% fprintf('---- Sensations: the number of segmented button presses in a line: %d - %d - %d - %d ... \n', ...
%         sensT_mapC(i, n1, :), sensTB_mapIMUaC(i, n1, :), sensTB_mapIMUgC(i, n1, :), sensTB_mapIMUagC(i, n1, :));
fdir = 'D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_all';
cd(fdir)

% hit rate: the number of button presses - maternal FM recognition are at least 10 times.
hr = 10;

fname = 'Seg01_butter_all_focus_Nstd-IMU4_W100_res.txt';
fid = fopen(fname);

L = 0;
key_lines = 'Sensations: the number of segmented button presses in a line';

while ~feof(fid)
    
    tmp_line = fgetl(fid);
    if ~isempty(strfind(tmp_line, key_lines))
        L = L + 1;
        lines{L, :} = tmp_line;
    end

    clear tmp_line
end

fclose(fid);
cd(curr_dir)


% Selected idx
idx_hr10 = int32([4,5,6,7,8,9,10,12,13,14,16,17,18,19,20,23,24]);

%% Appendix - Nstd IMU: 1
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 6 - 1 - 3 - 1 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 18 - 12 - 9 - 9 ... '  }
%     {'---- Sensations: the number of segmented button presses in a line: 21 - 21 - 20 - 20 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 14 - 8 - 11 - 8 ... '  }
%     {'---- Sensations: the number of segmented button presses in a line: 28 - 28 - 25 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 16 - 16 - 15 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 11 - 6 - 11 - 6 ... '  }
%     {'---- Sensations: the number of segmented button presses in a line: 48 - 43 - 48 - 43 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 6 - 6 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 24 - 25 - 24 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 34 - 28 - 26 - 26 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 31 - 24 - 24 - 22 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 17 - 17 - 17 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 64 - 60 - 60 - 60 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 32 - 25 - 26 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 20 - 20 - 19 - 19 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 16 - 13 - 11 - 11 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 8 - 1 - 5 - 1 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 55 - 55 - 55 - 55 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 22 - 21 - 22 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 0 - 0 - 0 - 0 ... '    }

%% Appendix - Nstd IMU: 2
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 6 - 3 - 3 - 3 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 18 - 14 - 12 - 11 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 21 - 21 - 21 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 14 - 13 - 11 - 11 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 28 - 28 - 25 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 19 - 16 - 16 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 11 - 11 - 11 - 11 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 48 - 47 - 48 - 47 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 24 - 25 - 24 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 34 - 30 - 31 - 30 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 31 - 28 - 28 - 28 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 19 - 18 - 18 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 64 - 60 - 60 - 60 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 32 - 25 - 26 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 20 - 20 - 19 - 19 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 16 - 14 - 14 - 13 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 8 - 1 - 5 - 1 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 55 - 55 - 55 - 55 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 22 - 21 - 22 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 0 - 0 - 0 - 0 ... '    }

%% Appendix - Nstd IMU: 3
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 6 - 4 - 3 - 3 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 18 - 16 - 13 - 12 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 21 - 21 - 21 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 14 - 14 - 13 - 13 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 28 - 28 - 25 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 21 - 21 - 20 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 11 - 11 - 11 - 11 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 48 - 48 - 48 - 48 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 24 - 25 - 24 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 34 - 31 - 32 - 31 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 31 - 28 - 28 - 28 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 19 - 21 - 19 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 64 - 60 - 60 - 60 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 32 - 25 - 26 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 20 - 20 - 19 - 19 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 16 - 15 - 15 - 15 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 8 - 2 - 5 - 2 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 55 - 55 - 55 - 55 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 22 - 21 - 22 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 0 - 0 - 0 - 0 ... '    }

%% Appendix - Nstd IMU: 4
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 6 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 18 - 17 - 15 - 14 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 21 - 21 - 21 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 14 - 14 - 13 - 13 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 28 - 28 - 25 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 23 - 24 - 22 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 11 - 11 - 11 - 11 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 48 - 48 - 48 - 48 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 24 - 25 - 24 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 34 - 32 - 32 - 31 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 31 - 29 - 29 - 28 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 4 - 4 - 4 - 4 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 25 - 21 - 21 - 20 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 64 - 60 - 60 - 60 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 32 - 25 - 26 - 25 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 20 - 20 - 19 - 19 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 16 - 15 - 15 - 15 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 7 - 7 - 7 - 7 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 8 - 4 - 5 - 3 ... '    }
%     {'---- Sensations: the number of segmented button presses in a line: 55 - 55 - 55 - 55 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 22 - 21 - 22 - 21 ... '}
%     {'---- Sensations: the number of segmented button presses in a line: 0 - 0 - 0 - 0 ... '    }

%% Refine the data set by hitting rate
acouL_hr10 = acouL(idx_hr10); 
acouLT_hr10 = acouLT(idx_hr10); 
acouLTB_hr10 = acouLTB(idx_hr10);  
acouLTB_MA_hr10 = acouLTB_MA(idx_hr10); 
acouLTE_hr10 = acouLTE(idx_hr10); 
acouLTE_MA_hr10 = acouLTE_MA(idx_hr10); 

acouM_hr10 = acouM(idx_hr10); 
acouMT_hr10 = acouMT(idx_hr10); 
acouMTB_hr10 = acouMTB(idx_hr10); 
acouMTB_MA_hr10 = acouMTB_MA(idx_hr10); 
acouMTE_hr10 = acouMTE(idx_hr10); 
acouMTE_MA_hr10 = acouMTE_MA(idx_hr10); 

acouR_hr10 = acouR(idx_hr10); 
acouRT_hr10 = acouRT(idx_hr10); 
acouRTB_hr10 = acouRTB(idx_hr10); 
acouRTB_MA_hr10 = acouRTB_MA(idx_hr10); 
acouRTE_hr10 = acouRTE(idx_hr10); 
acouRTE_MA_hr10 = acouRTE_MA(idx_hr10); 

forc_hr10 = forc(idx_hr10);
forcT_hr10 = forcT(idx_hr10); 
forcTB_hr10 = forcTB(idx_hr10); 
forcTB_MA_hr10 = forcTB_MA(idx_hr10);
forcTE_hr10 = forcTE(idx_hr10);
forcTE_MA_hr10 = forcTE_MA(idx_hr10);

IMUacc_hr10 = IMUacc(idx_hr10);
IMUaccNet_hr10 = IMUaccNet(idx_hr10);
IMUaccNetT_hr10 = IMUaccNetT(idx_hr10);
IMUaccNetTB_hr10 = IMUaccNetTB(idx_hr10);
IMUaccNetTB_MA_hr10 = IMUaccNetTB_MA(idx_hr10);
IMUaccNetTE_hr10 = IMUaccNetTE(idx_hr10);
IMUaccNetTE_MA_hr10 = IMUaccNetTE_MA(idx_hr10);
IMUaccT_hr10 = IMUaccT(idx_hr10);

IMUgyr_hr10 = IMUgyr(idx_hr10);
IMUgyrNet_hr10 = IMUgyrNet(idx_hr10);
IMUgyrNetT_hr10 = IMUgyrNetT(idx_hr10);
IMUgyrNetTB_hr10 = IMUgyrNetTB(idx_hr10);
IMUgyrNetTB_MA_hr10 = IMUgyrNetTB_MA(idx_hr10);
IMUgyrNetTE_hr10 = IMUgyrNetTE(idx_hr10);
IMUgyrNetTE_MA_hr10 = IMUgyrNetTE_MA(idx_hr10);
IMUgyrT_hr10 = IMUgyrT(idx_hr10);

piezL_hr10 = piezL(idx_hr10);
piezLT_hr10 = piezLT(idx_hr10);
piezLTB_hr10 = piezLTB(idx_hr10);
piezLTB_MA_hr10 = piezLTB_MA(idx_hr10);
piezLTE_hr10 = piezLTE(idx_hr10);
piezLTE_MA_hr10 = piezLTE_MA(idx_hr10);

piezM_hr10 = piezM(idx_hr10);
piezMT_hr10 = piezMT(idx_hr10);
piezMTB_hr10 = piezMTB(idx_hr10);
piezMTB_MA_hr10 = piezMTB_MA(idx_hr10);
piezMTE_hr10 = piezMTE(idx_hr10);
piezMTE_MA_hr10 = piezMTE_MA(idx_hr10);

piezR_hr10 = piezR(idx_hr10);
piezRT_hr10 = piezRT(idx_hr10);
piezRTB_hr10 = piezRTB(idx_hr10);
piezRTB_MA_hr10 = piezRTB_MA(idx_hr10);
piezRTE_hr10 = piezRTE(idx_hr10);
piezRTE_MA_hr10 = piezRTE_MA(idx_hr10);

sens_hr10 = sens(idx_hr10);
sensT_hr10 = sensT(idx_hr10);
sensT_idxP_hr10 = sensT_idxP(idx_hr10);
sensT_MA_hr10 = sensT_MA(idx_hr10);
sensT_MA_B_hr10 = sensT_MA_B(idx_hr10);
sensT_MA_idxP_hr10 = sensT_MA_idxP(idx_hr10);
sensT_MA_mtx_hr10 = sensT_MA_mtx(idx_hr10);
sensT_MA_mtxP_hr10 = sensT_MA_mtxP(idx_hr10);
sensT_MA_thresh_hr10 = sensT_MA_thresh(idx_hr10);
sensT_MA_thresh_idxP_hr10 = sensT_MA_thresh_idxP(idx_hr10);
sensT_MA_thresh_mtx_hr10 = sensT_MA_thresh_mtx(idx_hr10);
sensT_MA_thresh_mtxP_hr10 = sensT_MA_thresh_mtxP(idx_hr10);
sensT_MA_threshB_hr10 = sensT_MA_threshB(idx_hr10);
sensT_MA_threshB_idxP_hr10 = sensT_MA_threshB_idxP(idx_hr10);
sensT_MA_threshB_mtx_hr10 = sensT_MA_threshB_mtx(idx_hr10);
sensT_MA_threshB_mtxP_hr10 = sensT_MA_threshB_mtxP(idx_hr10);
sensT_MALoss_hr10 = sensT_MALoss(idx_hr10);
sensT_mtx_hr10 = sensT_mtx(idx_hr10);
sensT_mtxP_hr10 = sensT_mtxP(idx_hr10);

%% save organized data matrices
participant = 'all';
data_category = 'focus';

% All-in-one
cd('D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\c_processed_all_F_hr10');
save(['Preproc010_' participant '_' data_category '_AllInOne.mat'], ...
      'forc_hr10', 'acouL_hr10', 'acouM_hr10', 'acouR_hr10', 'piezL_hr10', 'piezM_hr10', 'piezR_hr10', 'sens_hr10', 'IMUacc_hr10', 'IMUgyr_hr10', 'IMUaccNet_hr10', 'IMUgyrNet_hr10', ...
      'forcT_hr10', 'acouLT_hr10', 'acouMT_hr10', 'acouRT_hr10', 'piezLT_hr10', 'piezMT_hr10', 'piezRT_hr10', 'sensT_hr10', 'IMUaccT_hr10', 'IMUgyrT_hr10', 'IMUaccNetT_hr10', 'IMUgyrNetT_hr10', ...
      'forcTB_hr10', 'acouLTB_hr10', 'acouMTB_hr10', 'acouRTB_hr10', 'piezLTB_hr10', 'piezMTB_hr10', 'piezRTB_hr10', 'IMUaccNetTB_hr10', 'IMUgyrNetTB_hr10', ...
      'forcTE_hr10', 'acouLTE_hr10', 'acouMTE_hr10', 'acouRTE_hr10', 'piezLTE_hr10', 'piezMTE_hr10', 'piezRTE_hr10', 'IMUaccNetTE_hr10', 'IMUgyrNetTE_hr10', ...
      'forcTB_MA_hr10', 'acouLTB_MA_hr10', 'acouMTB_MA_hr10', 'acouRTB_MA_hr10', 'piezLTB_MA_hr10', 'piezMTB_MA_hr10', 'piezRTB_MA_hr10', 'IMUaccNetTB_MA_hr10', 'IMUgyrNetTB_MA_hr10', ...
      'forcTE_MA_hr10', 'acouLTE_MA_hr10', 'acouMTE_MA_hr10', 'acouRTE_MA_hr10', 'piezLTE_MA_hr10', 'piezMTE_MA_hr10', 'piezRTE_MA_hr10', 'IMUaccNetTE_MA_hr10', 'IMUgyrNetTE_MA_hr10', ...
      'sensT_MA_hr10', 'sensT_MA_B_hr10', 'sensT_MA_thresh_hr10', 'sensT_MA_threshB_hr10', 'sensT_MALoss_hr10', ...
      'sensT_mtx_hr10', 'sensT_idxP_hr10', 'sensT_mtxP_hr10', ...
      'sensT_MA_mtx_hr10', 'sensT_MA_idxP_hr10', 'sensT_MA_mtxP_hr10', ...
      'sensT_MA_thresh_mtx_hr10', 'sensT_MA_thresh_idxP_hr10', 'sensT_MA_thresh_mtxP_hr10', ...
      'sensT_MA_threshB_mtx_hr10', 'sensT_MA_threshB_idxP_hr10', 'sensT_MA_threshB_mtxP_hr10', ...
      '-v7.3' );

%% save organized data matrices by individual methods
% common variables
save(['Preproc011_' participant '_' data_category '_common.mat'], ...
      'forc_hr10', 'acouL_hr10', 'acouM_hr10', 'acouR_hr10', 'piezL_hr10', 'piezM_hr10', 'piezR_hr10', 'sens_hr10', 'IMUacc_hr10', 'IMUgyr_hr10', 'IMUaccNet_hr10', 'IMUgyrNet_hr10', ...
      'forcT_hr10', 'acouLT_hr10', 'acouMT_hr10', 'acouRT_hr10', 'piezLT_hr10', 'piezMT_hr10', 'piezRT_hr10', 'sensT_hr10', 'IMUaccT_hr10', 'IMUgyrT_hr10', 'IMUaccNetT_hr10', 'IMUgyrNetT_hr10', ...
      'sensT_mtx_hr10', 'sensT_idxP_hr10', 'sensT_mtxP_hr10', ...
      'sensT_MA_hr10', 'sensT_MA_B_hr10', 'sensT_MA_thresh_hr10', 'sensT_MA_threshB_hr10', 'sensT_MALoss_hr10', ...
      'sensT_MA_mtx_hr10', 'sensT_MA_idxP_hr10', 'sensT_MA_mtxP_hr10', ...
      'sensT_MA_thresh_mtx_hr10', 'sensT_MA_thresh_idxP_hr10', 'sensT_MA_thresh_mtxP_hr10', ...
      'sensT_MA_threshB_mtx_hr10', 'sensT_MA_threshB_idxP_hr10', 'sensT_MA_threshB_mtxP_hr10', ...
      '-v7.3' );


% butterworth
save(['Preproc012_' participant '_' data_category '_butter.mat'], ...
     'forcTB_hr10', 'acouLTB_hr10', 'acouMTB_hr10', 'acouRTB_hr10', 'piezLTB_hr10', 'piezMTB_hr10', 'piezRTB_hr10', 'IMUaccNetTB_hr10', 'IMUgyrNetTB_hr10', ...
      '-v7.3' );

% ellip
save(['Preproc013_' participant '_' data_category '_ellip.mat'], ...
      'forcTE_hr10', 'acouLTE_hr10', 'acouMTE_hr10', 'acouRTE_hr10', 'piezLTE_hr10', 'piezMTE_hr10', 'piezRTE_hr10', 'IMUaccNetTE_hr10', 'IMUgyrNetTE_hr10', ...
      '-v7.3' );

% butterworth + moving average
save(['Preproc014_' participant '_' data_category '_butter_movingAvg.mat'], ...
      'forcTB_MA_hr10', 'acouLTB_MA_hr10', 'acouMTB_MA_hr10', 'acouRTB_MA_hr10', 'piezLTB_MA_hr10', 'piezMTB_MA_hr10', 'piezRTB_MA_hr10', 'IMUaccNetTB_MA_hr10', 'IMUgyrNetTB_MA_hr10', ...
      '-v7.3' );

% ellpic + moving average
save(['Preproc015_' participant '_' data_category '_ellip_movingAvg.mat'], ...
      'forcTE_MA_hr10', 'acouLTE_MA_hr10', 'acouMTE_MA_hr10', 'acouRTE_MA_hr10', 'piezLTE_MA_hr10', 'piezMTE_MA_hr10', 'piezRTE_MA_hr10', 'IMUaccNetTE_MA_hr10', 'IMUgyrNetTE_MA_hr10', ...
      '-v7.3' );
cd(curr_dir)




















