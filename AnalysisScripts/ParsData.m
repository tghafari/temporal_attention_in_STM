% close all
clc
clear ;
%% preallocation and file introduction
SubjList = setxor(24:43,[26,32]); %all 4diff subjects
% 259:269; %quad data
% [259]; %very good performance in all quadrants
% 245:258; %experimental error
% setxor(24:43,[26,32,42,39]); %4difflev data (RAW poster data)
AllData=cell(numel(SubjList),1);
meanScore=zeros(1,numel(SubjList));

cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMS4DiffLevel/')
ColNames = {'code', 'foreperiod', 'predictability', 'difficulty', ...
    'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
    'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
    'confidence_RT', 'TCI','score','RTCorrect','ConfCorrect','TP','TN','FP','FN','xpos','ypos','perfGrp'};
VarsOfInterest = [1,2,3,4,5,6,7,13,14,15,16]; %for 4difflev

% cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants/')
% ColNames = {'code', 'foreperiod', 'predictability', 'difficulty', ...
%     'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
%     'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
%     'confidence_RT', 'TCI','Quad','score','RTCorrect','ConfCorrect','TP','TN','FP','FN','xpos','ypos','perfGrp'};
% VarsOfInterest = [1,2,3,4,5,6,7,13,14,15,16,18]; %for quads
 
% cd('/Users/Tara/Documents/MATLAB/MATLAB Programs/PhD Thesis Programs/DMS Project/Results/Trainings/') %for trainings
% ColNames = {'code', 'foreperiod', 'predictability', 'difficulty', ...
%     'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
%     'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
%     'confidence_RT', 'TCI','Quad','score','RTCorrect','ConfCorrect','TP','TN','FP','FN','xpos','ypos','perfGrp'};
% VarsOfInterest = [2, 3, 4, 5, 6, 7, 13, 14, 15, 16]; %for training

%% Preprocessing the data
for sId = 1:numel(SubjList)
    fname = fullfile(sprintf('%d-1.mat',SubjList(sId))); %-2-1 for trainings
    load(fname);
    conditionsMat = double(conditionsMat);
    conditionsMat(conditionsMat == 100) = nan; % this nans all the cells without any answers

    tmp = unique(conditionsMat(:,2));
    conditionsMat(conditionsMat(:,2) == tmp(2), :) = []; %Omits data regarding unpredictable foreperiods = 650 & 1400 ms
    conditionsMat(conditionsMat(:,2) == tmp(3), :) = [];
    conditionsMat(conditionsMat(:,2) == max(conditionsMat(:,2)),2) = 1.4;
    
    %Calculate performance
    Data = array2table(conditionsMat(:,VarsOfInterest),'VariableNames',ColNames(VarsOfInterest)); %for other than overtime you need to put conditionsMat to 20:end
    Data.score = Data.response == Data.correct_response;
    Data.RTCorrect=Data.reaction_time; Data.RTCorrect(Data.score==0)=nan;
    Data.ConfCorrect=Data.confidence; Data.ConfCorrect(Data.score==0)=nan;
    Data.TP = Data.response == 1 & Data.correct_response == 1;
    Data.TN = Data.response == 2 & Data.correct_response == 2;
    Data.FP = Data.response == 1 & Data.correct_response == 2;
    Data.FN = Data.response == 2 & Data.correct_response == 1;
    
    %Calculate position
    Data.xpos=cos(2*pi*(Data.question-1)./Data.difficulty+pi./Data.difficulty);
    Data.ypos=sin(2*pi*(Data.question-1)./Data.difficulty+pi./Data.difficulty);
    %Data.xpos=cos(2*pi*(Data.question-1)./Data.difficulty) %for 4diff
    %Data.ypos=sin(2*pi*(Data.question-1)./Data.difficulty) %for 4diff
    Data.xpos(Data.correct_response==2)=nan;
    Data.ypos(Data.correct_response==2)=nan;
    
    %Label subjects regarding their performances
    Data.perfGrp = nan(height(Data),1);
    difficulty   = unique(Data.difficulty);    
    %Decide in respect to which variable you want to group performance
    meanScore(:,sId) = median(Data.score(Data.difficulty==4));
%     & Data.foreperiod==.65 & Data.predictability==1));
    %mean(Data.score(Data.difficulty==max(difficulty)) %in 4difflev 8 is a
    %better difficulty than 10
%     %lable subjects according to their performance in each difflevel--not so very useful atm
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
        fprintf('%d is a bad performer subject',SubjList(sId));
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
    
    
save('AllData4Diff','AllData');


LengthCond=height(Data);
allCondMat=nan((numel(SubjList)*LengthCond),width(Data));

for sub=1:numel(SubjList)
     allCondMat(LengthCond*(sub-1)+1:sub*LengthCond,:)=table2array(AllData{sub});
end
PooledData=array2table(allCondMat,'VariableNames',ColNames([VarsOfInterest,18:end])); %18:end for 4difflev 19:end for quad
save('PooledData4Diff','PooledData');

%5=>subject's response,
%6=>reaction time(RT),
%7=>correct Response,
%8=>timing check of ForePeriod
%9&10=>ITI and recheck,
%11=>delay phase time check
%13=>ii,
%14=>jj
%15&16=>confidence scale and RT,
%17=>Test to confidence interval (TCI)

%% Pool all trials of all subjects together -- Not needed-- only a back up program for pooled trials
LengthCond=216;
SubjList =setxor(24:43,[26,32,42,39]);
allCondMat=nan(numel(SubjList)*LengthCond,17);
ColNames = {'code', 'foreperiod', 'predictability', 'difficulty', ...
    'response', 'reaction_time', 'correct_response', 'fp_check', 'ITI',...
    'ITI_check', 'delay_phase_check', 'repetition', 'stimulus','question', 'confidence', ...
    'confidence_RT', 'TCI'};
cd('/Users/Tara/Documents/MATLAB/MATLAB Programs/PhD Thesis Programs/DMS Project/Results/Analyse/')
VarsOfInterest = [2, 3, 4, 5, 6, 7, 13, 14, 15, 16];

for sId = 1:numel(SubjList)
    fname = fullfile(sprintf('%d-1.mat',SubjList(sId)));
    load(fname);
    conditionsMat = double(conditionsMat);
    conditionsMat(conditionsMat == 100) = nan; 
    tmp = unique(conditionsMat(:,2));
    conditionsMat(conditionsMat(:,2) == tmp(2), :) = []; %Omits data regarding unpredictable foreperiods = 900 & 1150 ms
    conditionsMat(conditionsMat(:,2) == tmp(3), :) = [];
    conditionsMat(conditionsMat(:,2) == max(conditionsMat(:,2)),2) = 1.4;
    
    allCondMat(LengthCond*(sId-1)+1:sId*LengthCond,:)=conditionsMat;  
    
end

PooledData = array2table(allCondMat(:,VarsOfInterest),'VariableNames',ColNames(VarsOfInterest)); %for other than overtime you need to put conditionsMat to 20:end
PooledData.score = PooledData.response == PooledData.correct_response;
PooledData.RTCorrect=PooledData.reaction_time; PooledData.RTCorrect(PooledData.score==0)=nan;
PooledData.ConfCorrect=PooledData.confidence; PooledData.ConfCorrect(PooledData.score==0)=nan;
PooledData.TP = PooledData.response == 1 & PooledData.correct_response == 1;
PooledData.TN = PooledData.response == 2 & PooledData.correct_response == 2;
PooledData.FP = PooledData.response == 1 & PooledData.correct_response == 2;
PooledData.FN = PooledData.response == 2 & PooledData.correct_response == 1;
PooledData.xpos = cos(2 * pi * (PooledData.question - 1) ./ PooledData.difficulty);
PooledData.ypos = sin(2 * pi * (PooledData.question - 1) ./ PooledData.difficulty);
PooledData.xpos(PooledData.correct_response == 2) = nan;
PooledData.ypos(PooledData.correct_response == 2) = nan;

save('PooledData','PooledData');
