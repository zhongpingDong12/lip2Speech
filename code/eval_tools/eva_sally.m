ori_path  = 'C:\Users\Andrew\sally_file\eval_tools\eva_sally\test_all_sample\male_gf\waveform\oriTraining\';
pre_path = 'C:\Users\Andrew\sally_file\eval_tools\eva_sally\test_all_sample\male_gf\waveform\training\';
save_path = 'C:\Users\Andrew\sally_file\eval_tools\eva_sally\test_all_sample\male_gf\waveform\eva_training.mat';

list=dir(fullfile(ori_path));
fileNum=size(list,1); 
disp(fileNum);

pesq_value = zeros(1,fileNum-2);
speechDistortion = zeros(1,fileNum-2);
backgroundDistortion = zeros(1,fileNum-2);
overallQuality = zeros(1,fileNum-2);


for n=3:fileNum
    oriPath = fullfile(ori_path,list(n).name);
    disp(oriPath)
    
    prePath = fullfile(pre_path,list(n).name);
    disp(prePath)
    
    try
        pesq_value(1,(n-2)) = pesq(8000,oriPath ,prePath );
        [speechDistortion(1,(n-2)), backgroundDistortion(1,(n-2)), overallQuality(1,(n-2))]=composite(oriPath ,prePath);
    catch
        pesq_value(1,(n-2))=0;
        speechDistortion(1,(n-2))=0;
        backgroundDistortion(1,(n-2))=0;
        overallQuality(1,(n-2))=0;
    end 
end

disp(pesq_value);
disp(overallQuality);
save(save_path,'pesq_value','speechDistortion','backgroundDistortion','overallQuality')