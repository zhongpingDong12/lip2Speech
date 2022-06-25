% This script serves as the batch origin for PESE
% Created 18th may for noisy comparion experiments
% Need to adjust snr, get noise, and source folder here

% clear variables first
clear all; clc;

%get source folders
%source_dir = uigetdir(pwd,'Select the speech source folder');
clean_dir = 'Z:\Main\PHD\Matlab\workspace\noise_distort_tests\clean';
noisy_dir = 'Z:\Main\PHD\Matlab\workspace\noise_distort_tests\noisy';
spect_dir = 'Z:\Main\PHD\Matlab\workspace\noise_distort_tests\subtract';
twofilt_dir = 'Z:\Main\PHD\Matlab\workspace\noise_distort_tests\2filt';

% number of comparisons to run
nmbcomp = 3;

cle_list = dir(fullfile(clean_dir, '*.wav'));
numSpe = length(cle_list);


% % for each clean file, get file, define folders, and run comparisons
for c = 1:numSpe
    % Get name of first speaker
    sourcefolders = cle_list(c).name(1:end-13);
    
    % Create file access for noisy and enhanced comparison
    noisFoldName = [noisy_dir,'\',sourcefolders];
    spectFoldName = [spect_dir,'\',sourcefolders];
    twoFoldName = [twofilt_dir,'\',sourcefolders];

    % Go into noisy path first 
    % get list of files
    nois_list = dir(fullfile(noisFoldName, '*.wav'));
    numNois = length(nois_list);
    
    for noicount = 1:numNoise
        noisres(noicountsourcefolders = nois_list(c).name;
    end
    
  
      
%      
%     trgnam = [source_list(c).name(1:end-12),'_spect_enh.wav'];
%     trgfile = fullfile(dest_dir, trgnam);
%     
%     GA_SE(srcfile,trgfile);
end