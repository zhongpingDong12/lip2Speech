%% SETUP
% make sure the nsltools is in the path list
% load colormap, filterbank and parameters
% WAVFORM
% sequential format:	see LOADFILE
% .au format:		see AUREAD
% .aiff format:		see AIFFREAD
%x = loadfile('_come.sho');


loadload;
wav_folder = dir('?C:\Users\sally\Desktop\test\*.wav');
mat_folder = '?C:\Users\sally\Desktop\test\'; % Or wherever you want.
print(length(wav_folder))

i=0;
for k = 1 : length(wav_folder)  % For every blob.....
    x = audioread(wav_folder(k).name);
    x = unitseq(x);
    y = wav2aud(x, paras);
    
    % Create a filename.
    baseFileName = sprintf('%03d.mat', i);
    fullFileName = fullfile(mat_folder, baseFileName);
    i=i+1;
    %print(fullFileName);
    % save mat file in the folder
    save(fullFileName,'y');
end
