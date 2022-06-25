ori_path  = 'C:\Users\Andrew\Desktop\word\muti_ori_wav\';
pre_path = 'C:\Users\Andrew\Desktop\word\muti_pre_wav\';
output_path = 'C:\Users\Andrew\Desktop\word\muti_pesq\';

list=dir(fullfile(ori_path));
fileNum=size(list,1); 
disp(fileNum);


for n=3:fileNum
    oriDir = fullfile(ori_path,list(n).name);
    disp(oriDir)
    preDir = fullfile(pre_path,list(n).name);
    disp(preFile);
    
%     [filepath,name,ext] = fileparts(oriDir);
%     outputDir = fullfile(output_path,strcat(name));
%     disp(outputDir);
%     if ~exist(outputDir, 'dir')
%        mkdir(outputDir)
%     end
    
%     preFile = fullfile(preDir,'train');
% %     disp(preFile)
%     
%     outputFile = fullfile(outputDir,'train');
% %     disp(outputFile)
%     if ~exist(outputFile, 'dir')
%       mkdir(outputFile)
%     end
%     
%     listPre =dir(fullfile(preFile));
%     NumFile=size(listPre,1); 
%     disp(NumFile);
    
% %     pesq_value = zeros(NumFile-2);
% %     sd = zeros((NumFile-2);
% %     bd = zeros(NumFile-2);
% %     oo = zeros(NumFile-2);
%     
%     for n=3:NumFile
%         preWav = fullfile(preFile,listPre(n).name);
%         disp(preWav)
%         
%         [filepath,name,ext] = fileparts(preWav);
%         oriWav = fullfile(oriDir,strcat(name,'.wav'));
%         disp(oriWav);
%         
%         try
%             pesq_value(n-2) = pesq(8000,oriWav ,preWav );
%             [sd(n-2), bd(n-2), oo(n-2)]=composite(oriWav ,preWav);
%         catch
%              pesq_value(n-2)=0;
%              oo(n-2)=0;
% 
%         end 
%             
%     end
%     save(fullfile(outputFile,'pesq.mat'),'pesq_value');
%     save(fullfile(outputFile,'oo.mat'),'oo');
end

