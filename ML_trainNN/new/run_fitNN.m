clear all;clc;

% Input parameters
istart = 0;
extract_min_data = true;

% setup paths and libraries
addpath('matlib','pydir');
print_header();

% new run ?
if istart == 0 % remove old files
    delete('combined_data.csv');
end

% Combine data files
num_data_file = pyrunfile("join_data.py 'data' 'combined_data.csv'",'num_data_file');
fprintf('Found %d files in the database.\n',num_data_file);

% Filter struture having minimum total energy
if extract_min_data == true
    pyrunfile("extract_min_data.py 'combined_data.csv' 'combined_data.csv' 'tot_en' 'formula'");
end


