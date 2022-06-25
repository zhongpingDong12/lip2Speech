loadload;
folderPath = 'D:\Mrs_backup\speech_test\all_vocabulary\13_gf_corr2D\';
outputPath =  'D:\Mrs_backup\speech_test\all_vocabulary\15_gf_pesq\';

list=dir(fullfile(folderPath));
fileNum=size(list,1); 
% disp(fileNum);

for n=3:fileNum
    inputPath = fullfile(folderPath,list(n).name);
%     disp(inputPath)
 
    [filepath,name,ext] = fileparts(inputPath);
    outputName = fullfile(outputPath,strcat(name));
%     disp(outputName);
    if ~exist(outputName, 'dir')
       mkdir(outputName)
    end

    
    
    
    
    
    inputTrain = fullfile(inputPath,'train');
%     disp(inputTrain)
    
    outputTrain = fullfile(outputName,'train');
%     disp(outputTrain)
    if ~exist(outputTrain, 'dir')
      mkdir(outputTrain)
    end
    
    
    trainList =dir(fullfile(inputTrain));
    trainNum=size(trainList,1); 
%     disp(trainNum);
    
    for n=3:trainNum
        inputFile = fullfile(inputTrain,trainList(n).name);
        disp(inputFile)
        
        [filepath,name,ext] = fileparts(inputFile);
        outputFile = fullfile(outputTrain,strcat(name,'.wav'));
        disp(outputFile);
        
        load(inputFile)
        res_wav = aud2wav(audio_input, [], [paras 10 1 1]);
        audiowrite(outputFile,res_wav,8000);
    end 
        
    
    
    
    inputTest = fullfile(inputPath,'test');
%     disp(inputTest)
    
    outputTest = fullfile(outputName,'test');
%     disp(outputTest)
    if ~exist(outputTest, 'dir')
      mkdir(outputTest)
    end
    
    
    testList =dir(fullfile(inputTest));
    testNum=size(testList,1); 
%     disp(trainNum);
    
    for n=3:testNum
        inputFile = fullfile(inputTest,testList(n).name);
        disp(inputFile)
        
        [filepath,name,ext] = fileparts(inputFile);
        outputFile = fullfile(outputTest,strcat(name,'.wav'));
        disp(outputFile);
        
        load(inputFile)
        res_wav = aud2wav(audio_input, [], [paras 10 1 1]);
        audiowrite(outputFile,res_wav,8000);
    end 

end