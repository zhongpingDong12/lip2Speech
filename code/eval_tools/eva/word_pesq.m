diary speech_quality_result_AVG.txt
diary on


ori_path  = 'C:\Users\Andrew\Desktop\word\pesq_eva\02_audio\';
pre_path = 'C:\Users\Andrew\Desktop\word\pesq_eva\15_gf_specToWav\';
output_path = 'C:\Users\Andrew\Desktop\word\pesq_eva\17_gf_pesq\';

list=dir(fullfile(ori_path));
fileNum=size(list,1); 
% disp(fileNum);



for n=3:fileNum
    oriFile = fullfile(ori_path,list(n).name);
%     disp(oriFile)
    
    preFile = fullfile(pre_path,list(n).name);
%     disp(preFile);
    
    [filepath,name,ext] = fileparts(oriFile);
    outputFile = fullfile(output_path,strcat(name));
%     disp(outputFile);
    if ~exist(outputFile, 'dir')
       mkdir(outputFile)
    end
    
   
    preTrain = fullfile(preFile,'train');
%     disp(preTrain)
    
    outputTrain = fullfile(outputFile,'train');
%     disp(outputTrain)
    if ~exist(outputTrain, 'dir')
      mkdir(outputTrain)
    end
    
    trainList =dir(fullfile(preTrain));
    trainNum=size(trainList,1); 
%     disp(trainNum);
    
    pesq_train = zeros(size(3:trainNum));
    speechDistortion_train = zeros(size(3:trainNum));
    backgroundDistortion_train = zeros(size(3:trainNum));
    overallQuality_train = zeros(size(3:trainNum));

    for i=3:trainNum
        preWav = fullfile(preTrain,trainList(i).name);
%         disp(preWav)
        
        [filepath,name,ext] = fileparts(preWav);
        oriWav = fullfile(oriFile,strcat(name,'.wav'));
%         disp(oriWav);
   
        try
            pesq_train(i-2) = pesq(8000,oriWav ,preWav )
            [speechDistortion_train, backgroundDistortion_train(i-2), overallQuality_train(i-2)]=composite(oriWav ,preWav)
            
            
            pesq_train(pesq_train==0)= NaN; 
            mean_pesq_train = mean(pesq_train,'omitnan');
            
            overallQuality_train(overallQuality_train==0)= NaN; 
            mean_overallQuality_train = mean(overallQuality_train,'omitnan');
            
            fprintf(' Average PESQ train is \n %f \n', mean_pesq_train);
            fprintf(' Average overallQuality train is \n %f \n', mean_overallQuality_train);
            
        catch
            pesq_train(i-2)=0;
            speechDistortion_train(i-2)=0;
            backgroundDistortion_train(i-2)=0;
            overallQuality_train(i-2)=0;
        end
           
    end

%     save(fullfile(outputTrain,'peaq.mat'),'pesq_train');
%     save(fullfile(outputTrain,'overallQuality.mat'),'overallQuality_train');
    
    
    
    
    
    
    
    preTest = fullfile(preFile,'test');
%     disp(preTest)
    
    outputTest = fullfile(outputFile,'test');
%     disp(outputTrain)
    if ~exist(outputTest, 'dir')
      mkdir(outputTest)
    end
    
    testList =dir(fullfile(preTest));
    testNum=size(testList,1); 
%     disp(testNum);
    
    pesq_test = zeros(size(3:testNum));
    speechDistortion_test = zeros(size(3:testNum));
    backgroundDistortion_test = zeros(size(3:testNum));
    overallQuality_test = zeros(size(3:testNum));

    for t=3:testNum
        preWav = fullfile(preTest,testList(t).name);
%         disp(preWav)
        
        [filepath,name,ext] = fileparts(preWav);
        oriWav = fullfile(oriFile,strcat(name,'.wav'));
%         disp(oriWav);
        
        try
            pesq_test(t-2) = pesq(8000,oriWav ,preWav )
            [speechDistortion_test(t-2), backgroundDistortion_test(t-2), overallQuality_test(t-2)]=composite(oriWav ,preWav)
           
            
            pesq_test(pesq_test==0)= NaN; 
            mean_pesq_test = mean(pesq_test,'omitnan');
            
            overallQuality_test(overallQuality_test==0)= NaN; 
            mean_overallQuality_test = mean(overallQuality_test,'omitnan');
            
            fprintf(' Average PESQ test is \n %f \n', mean_pesq_test);
            fprintf(' Average overallQuality test is \n %f \n', overallQuality_test);
        catch
            pesq_test(t-2)=0;
            speechDistortion_test(t-2)=0;
            backgroundDistortion_test(t-2)=0;
            overallQuality_test(t-2)=0;
        end
        
    end
   
%     save(fullfile(outputTest,'peaq.mat'),'pesq_test');
%     save(fullfile(outputTest,'overallQuality.mat'),'overallQuality_test');
    
    
    
end

diary off
