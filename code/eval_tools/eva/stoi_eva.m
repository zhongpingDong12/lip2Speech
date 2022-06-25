% ori_path  = 'D:\Mrs_backup\speech_test\evaluation\test_all_sample\multi_test\male_gf\waveform\oriTraining\';
% pre_path = 'D:\Mrs_backup\speech_test\evaluation\test_all_sample\multi_test\male_gf\waveform\training\';
% save_path = 'D:\Mrs_backup\speech_test\evaluation\test_all_sample\multi_test\male_gf\waveform\intell_training.mat';
% 
% list=dir(fullfile(ori_path));
% fileNum=size(list,1); 
% disp(fileNum);
% 
% stoi_value = zeros(1,fileNum-2);

% [oriAudio,Fs] = audioread(oriPath);
% [preAudio,Fs] = audioread(prePath);
% l1= length(oriAudio);
% l2 = length(preAudio);
% disp(l1);
% disp(l2);
% 
% stoiValue  = stoi(oriAudio, preAudio, Fs);
% disp(stoiValue);

% for n=3:fileNum
%     oriPath = fullfile(ori_path,list(n).name);
%     disp(oriPath)
%     
%     prePath = fullfile(pre_path,list(n).name);
%     disp(prePath)
%     
%     try
%         [oriAudio,Fs] = audioread(oriPath);
%         [preAudio,Fs] = audioread(prePath);
%         stoi_value(1,(n-2)) = stoi(oriAudio, preAudio, Fs) ;
%     catch
%         stoi_value(1,(n-2))=0;
%     end 
% end
% 
% disp(stoi_value);
% save(save_path,'stoi_value')
% 
% oriPath = 'D:\Mrs_backup\speech_test\audio_ori\s1\018.wav';
% prePath = 'C:\Users\sally\Desktop\anaysis_result\mutiple_speaker\audio_remove_nosie\1s1_gf2.wav';
% [oriAudio,Fs] = audioread(oriPath);
% [preAudio,Fs] = audioread(prePath);
% l1= length(oriAudio);
% 
% preAudio=preAudio(1:l1);
% l2 = length(preAudio);
% disp(l1);
% disp(l2);
% value = stoi(oriAudio, preAudio, Fs) ;
% display(value );





ori_path  = 'C:\Users\Andrew\Desktop\word\muti_ori_wav\';
pre_path = 'C:\Users\Andrew\Desktop\word\muti_pre_wav\cnn\';
output_path = 'C:\Users\Andrew\Desktop\word\muti_pesq\cnn\';


listModel=dir(fullfile(pre_path));
docNum=size(listModel,1); 
disp(docNum);

for n=3:docNum
    preSpec = fullfile(pre_path,listModel(n).name);
%     disp(preSpec);
    
    [filepath,name,ext] = fileparts(preSpec);
    outputSpec = fullfile(output_path,strcat(name));
%     disp(outputSpec);
    if ~exist(outputSpec, 'dir')
       mkdir(outputSpec)
    end
    
    list=dir(fullfile(preSpec));
    fileNum=size(list,1); 
    disp(fileNum);
    

    for n=3:fileNum
        preDir = fullfile(preSpec,list(n).name);
%         disp(preDir);

        [filepath,name,ext] = fileparts(preDir);

        oriDir = fullfile(ori_path,strcat(name));
%         disp(oriDir);

        outputDir = fullfile(outputSpec,strcat(name));
%         disp(outputDir);
        if ~exist(outputDir, 'dir')
           mkdir(outputDir)
        end

        preFile = fullfile(preDir,'test');
        disp(preFile)

        outputFile = fullfile(outputDir,'test');
        disp(outputFile)
        if ~exist(outputFile, 'dir')
          mkdir(outputFile)
        end

        listPre =dir(fullfile(preFile));
        NumFile=size(listPre,1); 
        disp(NumFile);
 
        stoi_value = zeros(size(3:NumFile));
    
        for n=3:NumFile
            preWav = fullfile(preFile,listPre(n).name);
            disp(preWav)

            [filepath,name,ext] = fileparts(preWav);
            oriWav = fullfile(oriDir,strcat(name,'.wav'));
            disp(oriWav);
            
            [oriAudio,Fs] = audioread(oriWav);
            [preAudio,Fs] = audioread(preWav);
            l1= length(oriAudio);

            preAudio=preAudio(1:l1);
            l2 = length(preAudio);
          
            stoi_value(n-2) = stoi(oriAudio, preAudio, Fs) ;
        end
        save(fullfile(outputFile,'stoi.mat'),'stoi_value');
    end

end