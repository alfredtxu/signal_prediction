clc
clear 
close all

curr_dir = 'D:\x_gdrive\ic_welcom_leap_fm\b_src_matlab';
cd(curr_dir);

participant = 'a_raw_data_003AM';
fdir = ['D:\x_gdrive\ic_welcom_leap_fm\a98_hometrial_data\' participant];
files = dir([fdir '\*.dat']);

cd(fdir)
for i = 1 : length(files)

    tmp_fname = files(i).name;
    tmp_new = ['am_data_' sprintf('%03d', i) '.dat'];
    movefile(tmp_fname, tmp_new);

    clear tmp_fname tmp_new
end
cd(curr_dir)