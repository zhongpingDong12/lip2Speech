%% SETUP
% make sure the nsltools is in the path list
% load colormap, filterbank and parameters
% WAVFORM
% sequential format:	see LOADFILE
% .au format:		see AUREAD
% .aiff format:		see AIFFREAD
%x = loadfile('_come.sho');


% loadload;
% wav_folder = dir('D:\Mrs_backup\speech_test\audio\s3\*.wav');
% mat_folder = 'D:\Mrs_backup\speech_test\AudSpecs\s3\'; % Or wherever you want.
% disp(length(wav_folder))
% 
% i=0;
% for k = 1 : length(wav_folder)  % For every blob.....
%     x = audioread(wav_folder(k).name);
%     x = unitseq(x);
%     y = wav2aud(x, paras);
%     
%     % Create a filename.
%     baseFileName = sprintf('%03d.mat', i);
%     fullFileName = fullfile(mat_folder, baseFileName);
%     i=i+1;
%     disp(fullFileName);
%     % save mat file in the folder
%     save(fullFileName,'y');
% end

%-----------------------------Load from all folder -----------
loadload;
wavPath ='D:\Mrs_backup\speech_test\audio\s9\';
specPath = 'D:\Mrs_backup\speech_test\AudSpecs\s9\'; 
if ~exist(specPath, 'dir')
    mkdir(specPath)
end

wavFolderInfo = dir(wavPath);

for i=3: length(wavFolderInfo)
    % Here is the wavform path
    fileName= wavFolderInfo(i).name;  
    wavFile = fullfile(wavPath,fileName);
    display(wavFile);
    
    % Here is the spec path      
    [folder, baseFileNameNoExt, extension] = fileparts(fileName);
    specFile = fullfile(specPath, [baseFileNameNoExt '.mat']);
    display(specFile);
    
    x = audioread(wavFile);
    x = unitseq(x);
    y = wav2aud(x, paras);
    save(specFile,'y');
    
end 