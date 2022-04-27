% close all
clc
clear ;
%% preallocation and file introduction
SubjList = setxor(24:43,[26,32]); %all 4diff subjects
% setxor(24:43,[25,26,32,36,39]); %all 4diff subjects (excluding rather bad subs)
% 259:269; %quad data 
% setxor(24:43,[26,32]); %all 4diff subjects
% 245:258; %experimental error
% setxor(24:43,[26,32,42,39]); %4difflev data (RAW poster data)
AllData=cell(numel(SubjList),1);
meanScore=zeros(1,numel(SubjList));

cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMS4DiffLevel/')
ColNames = {'Subject_code','trial_code', 'foreperiod', 'predictability', 'difficulty', ...
    'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
    'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
    'confidence_RT', 'TCI','score','RTCorrect','ConfCorrect','TP','TN','FP','FN','RT_TP','pos_rad','xpos','ypos','perfGrp'};
VarsOfInterest = [1,2,3,4,5,6,7,8,14,15,16,17]; %for 4difflev

% cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants/')
% ColNames = {'Subject_code','trial_code', 'foreperiod', 'predictability', 'difficulty', ...
%     'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
%     'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
%     'confidence_RT','TCI','Quad','score','RTCorrect','ConfCorrect','TP','TN','FP','FN','RT_TP','pos_rad','xpos','ypos','perfGrp'};
% VarsOfInterest = [1,2,3,4,5,6,7,8,14,15,16,17,19]; %for quads
 
% cd('/Users/Tara/Documents/MATLAB/MATLAB Programs/PhD Thesis Programs/DMS Project/Results/Trainings/') %for trainings
% ColNames = {'code', 'foreperiod', 'predictability', 'difficulty', ...
%     'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
%     'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
%     'confidence_RT', 'TCI','Quad','score','RTCorrect','ConfCorrect','TP','TN','FP','FN','pos_rad','xpos','ypos','perfGrp'};
% VarsOfInterest = [2, 3, 4, 5, 6, 7, 13, 14, 15, 16]; %for training

%% Preprocessing the data
for sId = 1:numel(SubjList)
    fname = fullfile(sprintf('%d-1.mat',SubjList(sId))); %-2-1 for trainings
    load(fname);
    conditionsMat(:,2:end+1) = double(conditionsMat);
    conditionsMat(:,1) = sId;
    conditionsMat(conditionsMat(:,6) == 100,6) = nan; % this nans all the cells without any answers

    tmp = unique(conditionsMat(:,3));
    conditionsMat(conditionsMat(:,3) == tmp(2), :) = []; %Omits data regarding unpredictable foreperiods = 650 & 1400 ms
    conditionsMat(conditionsMat(:,3) == tmp(3), :) = [];
    conditionsMat(conditionsMat(:,3) == max(conditionsMat(:,3)),3) = 1.4;
    
    %Calculate performance
    Data = array2table(conditionsMat(:,VarsOfInterest),'VariableNames',ColNames(VarsOfInterest)); %for other than overtime you need to put conditionsMat to 20:end
    Data.score = Data.response == Data.correct_response;
    Data.RTCorrect=Data.reaction_time; Data.RTCorrect(Data.score==0)=nan;
    Data.ConfCorrect=Data.confidence; Data.ConfCorrect(Data.score==0)=nan;
    Data.TP = Data.response == 1 & Data.correct_response == 1;
    Data.TN = Data.response == 2 & Data.correct_response == 2;
    Data.FP = Data.response == 1 & Data.correct_response == 2;
    Data.FN = Data.response == 2 & Data.correct_response == 1;
    Data.RT_TP = Data.RTCorrect; Data.RT_TP(Data.TP~=1)=nan; %RT for true positives (used for position analysis mainly)
    
    %Calculate position
%     Data.pos_rad = 2*pi*(Data.question-1)./Data.difficulty + pi./Data.difficulty;  %for quad
%     Data.xpos=cos(2*pi*(Data.question-1)./Data.difficulty + pi./Data.difficulty); %for quad
%     Data.ypos=sin(2*pi*(Data.question-1)./Data.difficulty + pi./Data.difficulty); %for quad
    Data.pos_rad = 2*pi*(Data.question-1)./Data.difficulty;  %for 4diff
    Data.xpos=cos(2*pi*(Data.question-1)./Data.difficulty); %for 4diff
    Data.ypos=sin(2*pi*(Data.question-1)./Data.difficulty); %for 4diff
   
    Data.pos_rad(Data.correct_response==2)=nan;
    Data.xpos(Data.correct_response==2)=nan;
    Data.ypos(Data.correct_response==2)=nan;
        
    %Label subjects regarding their performances
    Data.perfGrp = nan(height(Data),1);
    difficulty   = unique(Data.difficulty);    
    %Decide in respect to which variable you want to group performance
    meanScore(:,sId) = mean(Data.score(Data.difficulty==8)); %in 4difflev 8 is a better difficulty than 10
    
    %label performance groups -- not su very useful now
%     for diffLev  = 1:length(difficulty)            
%         prfGrpLl = mean(Data.score(Data.difficulty == diffLev));
%         if prfGrpLl>=0.85    
%             Data.perfGrp(Data.difficulty==difficulty(diffLev))=1; %perfect
%         elseif prfGrpLl>=0.7
%             Data.perfGrp(Data.difficulty==difficulty(diffLev))=2; %good
%         elseif prfGrpLl>0.55
%             Data.perfGrp(Data.difficulty==difficulty(diffLev))=3;
%         else
%             Data.perfGrp(Data.difficulty==difficulty(diffLev))=4; %bad
%         end

%     end

    AllData{sId} = Data;
    %Omit subjects with score <50% in the min diff level (if there is any)   
    ll = mean(Data.score(Data.difficulty == min(Data.difficulty)));
    if ll>0.5
        AllData{sId} = Data;
        disp(ll)
    else
        fprintf('%d is a bad performer subject \n',SubjList(sId));
    end
end

%rank subjects by scores
[meanSort,r]=sort(meanScore,2,'descend');
rank=1:length(meanScore);
rank(r)=rank;
meanScore(:,rank<=(max(rank)/2))=1; meanScore(:,rank>(max(rank)/2))=2; %1=good performer 2=bad performer

for sId=1:numel(SubjList)
    AllData{sId}.perfGrp=meanScore(sId)*ones(height(AllData{sId}),1);
end       
% save('AllData4Diff_2','AllData');       
% save('AllDataQuad_2','AllData');

LengthCond=height(Data);
allCondMat=nan((numel(SubjList)*LengthCond),width(Data));

for sub=1:numel(SubjList)
    allCondMat(LengthCond*(sub-1)+1:sub*LengthCond,:)=table2array(AllData{sub});
end
PooledData=array2table(allCondMat,'VariableNames',ColNames([VarsOfInterest,19:end])); %19:end for 4difflev 20:end for quad
% save('PooledData4Diff_2','PooledData');  writetable(PooledData,'PooledData4Diff_2.csv');         
% save('PooledDataQuad_2','PooledData');  writetable(PooledData,'PooledDataQuad_2.csv')

%6=>subject's response,
%7=>reaction time(RT),
%8=>correct Response,
%9=>timing check of ForePeriod
%10&11=>ITI and recheck,
%12=>delay phase time check
%14=>ii,
%15=>jj
%16&17=>confidence scale and RT,
%18=>Test to confidence interval (TCI)
