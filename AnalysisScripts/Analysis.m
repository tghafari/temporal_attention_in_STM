clear;
% close all
clc;

%% Making dataframes for ANOVA
% clear;
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMS4DiffLevel')
load AllData4Diff.mat 
nSubj=numel(AllData);
pg=zeros(1,nSubj); %vector of subject's performance group
% mean_TP=[]; %position-can't preallocate because the length of MMs are not equal
% mean_RT=[];

length_MM = 16;
mean_TP=nan(length_MM*nSubj,6); % no pos
mean_RT=nan(length_MM*nSubj,5); % no pos no perf group
mean_IES=nan(length_MM*nSubj,5);

for i=1:nSubj
    Data=AllData{i};
    pg(i)=Data.perfGrp(1); %define performance group for each subject
    Data(Data.correct_response == 2,:)=[];
    
%     Data(abs(Data.ypos)<0.0001,:)=[];
%     Data.Top=Data.ypos>0;
%     
%     Data(abs(Data.xpos)<0.0001,:)=[];
%     Data.Right=Data.xpos>0;
    
%     MM = grpstats(Data, {'pos_rad','foreperiod','predictability','difficulty'}, {@nanmean,'std'}); %all conditions separated
%     mean_TP = [mean_TP;i*ones(height(MM),1),MM.pos_rad,MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_TP];
%     mean_RT = [mean_RT;i*ones(height(MM),1),MM.pos_rad,MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_RT_TP];

    MM = grpstats(Data, {'foreperiod','predictability','difficulty'}, {@nanmean,'std'}); % without position
    mean_TP(length_MM*(i-1)+1:i*length_MM,:) = [i*ones(length_MM,1),MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_score,pg(i)*ones(length_MM,1)];
    mean_RT(length_MM*(i-1)+1:i*length_MM,:) = [i*ones(length_MM,1),MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_RTCorrect]; %,pg(i)*ones(length_MM,1)];
    mean_IES(length_MM*(i-1)+1:i*length_MM,:) = [i*ones(length_MM,1),MM.foreperiod,MM.predictability,MM.difficulty,(MM.nanmean_reaction_time./MM.nanmean_score)]; %,pg(i)*ones(length_MM,1)];

end
% %with position
% mean_TP_table = array2table(mean_TP,'VariableNames',{'Subject_code','pos_rad','foreperiod','predictability','difficulty','nanmean_TP'});
% save('DataFrame4Diff_TP_2','mean_TP_table');  writetable(mean_TP_table,'DataFrame4Diff_TP_2.csv');         
% mean_RT_table = array2table(mean_RT,'VariableNames',{'Subject_code','pos_rad','foreperiod','predictability','difficulty','nanmean_RT_TP'});
% save('DataFrame4Diff_RT_2','mean_RT_table');  writetable(mean_RT_table,'DataFrame4Diff_RT_2.csv');         
% %without position
% mean_score_table = array2table(mean_TP,'VariableNames',{'Subject_code','foreperiod','predictability','difficulty','nanmean_score','perf_grp'});
% save('DataFrame4Diff_score_perf_2','mean_score_table');  writetable(mean_score_table,'DataFrame4Diff_score_perf_2.csv');         
% mean_RT(isnan(mean_RT)) = nanmean(mean_RT(:,5)); %remove nans (where subject's score was 0 and hence RTCorr is nan)
% mean_RTCor_table = array2table(mean_RT,'VariableNames',{'Subject_code','foreperiod','predictability','difficulty','nanmean_RTCorrect'});
% save('DataFrame4Diff_RTCor_2','mean_RTCor_table');  writetable(mean_RTCor_table,'DataFrame4Diff_RTCor_2.csv');         
% mean_IES(isinf(mean_IES)) = nanmean(mean_IES(~isinf(mean_IES(:,5)),5)); %remove nans (where subject's score was 0 and hence RTCorr is nan)
% mean_IES_table = array2table(mean_IES,'VariableNames',{'Subject_code','foreperiod','predictability','difficulty','IES'});
% save('DataFrame4Diff_IES_2','mean_IES_table');  writetable(mean_IES_table,'DataFrame4Diff_IES_2.csv');         

%% Main analysis
% cd('/Users/Tara/Documents/MATLAB/MATLAB Programs/PhD Thesis Programs/DMS Project/Results/Trainings/')
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMS4DiffLevel')
load AllData4Diff.mat 
nSubj=numel(AllData);
pg=zeros(1,nSubj); %vector of subject's performance group
MeanRTCorrect=nan(4,4,nSubj);
MeanRT=nan(4,4,nSubj);
MeanScore=nan(4,4,nSubj);
InverseEfficiencyScore=nan(4,4,nSubj);
STDScore=nan(4,4,nSubj);
GroupCnt=nan(4,4,nSubj);
CVScore=nan(4,4,nSubj);
STEScore=nan(4,4,nSubj);
MeanConf=nan(4,4,nSubj);
MeanConfCorrect=nan(4,4,nSubj);
Pred=nan(4,4,nSubj);
ForeP=nan(4,4,nSubj);
ConfusMat=nan(4,4,nSubj,2,2);
MeanResponse=nan(4,4,nSubj);
PashlerK=nan(4,4,nSubj);

for i=1:nSubj
    Data = AllData{i};
    pg(i)=Data.perfGrp(1); %define performance group for each subject
   
    M = grpstats(Data, {'foreperiod','predictability','difficulty'}, {@nanmean,'std'});
    MeanScore(:,:,i)=reshape(M.nanmean_score,4,[]);
    MeanRTCorrect(:,:,i)=reshape(M.nanmean_RTCorrect,4,[]);
    MeanRT(:,:,i)=reshape(M.nanmean_reaction_time,4,[]);
    InverseEfficiencyScore(:,:,i)=MeanRT(:,:,i)./MeanScore(:,:,i);
    GroupCnt(:,:,i)=reshape(M.GroupCount,4,[]);
    STDScore(:,:,i)=reshape(M.std_score,4,[]);
    STEScore(:,:,i)=STDScore(:,:,i)./sqrt(GroupCnt(:,:,i));
    CVScore(:,:,i)=STDScore(:,:,i)./MeanScore(:,:,i);
    MeanConf(:,:,i)=reshape(M.nanmean_confidence,4,[]);
    MeanConfCorrect(:,:,i)=reshape(M.nanmean_ConfCorrect,4,[]);
    Pred(:,:,i)=reshape(M.predictability,4,[]);
    ForeP(:,:,i)=reshape(M.foreperiod,4,[]);
    MeanResponse(:,:,i)=reshape(M.nanmean_response,4,[]);    
    PashlerK(:,:,i)=reshape(M.difficulty.*((M.nanmean_TP-M.nanmean_FP)./(1-M.nanmean_FP)),4,[]);

    ConfusMat(:,:,i,1,1) = reshape(M.nanmean_TP .* M.GroupCount, 4, []);
    ConfusMat(:,:,i,1,2) = reshape(M.nanmean_FN .* M.GroupCount, 4, []);
    ConfusMat(:,:,i,2,1) = reshape(M.nanmean_FP .* M.GroupCount, 4, []);
    ConfusMat(:,:,i,2,2) = reshape(M.nanmean_TN .* M.GroupCount, 4, []);
end
TempAttEff_short = squeeze(InverseEfficiencyScore(:,1,:) - InverseEfficiencyScore(:,2,:));
TempAttEff_long = squeeze(InverseEfficiencyScore(:,3,:) - InverseEfficiencyScore(:,4,:));

MeanScoreMean=mean(MeanScore,3);
MeanRTMean=mean(MeanRTCorrect,3);
InverseEfficiencyMean=mean(InverseEfficiencyScore,3);
CVScoreMean=mean(CVScore,3);
STDScoreMean=mean(STDScore,3);


DiffLvl = unique(Data.difficulty);
GroupNames ={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};

%% Omit Subjects with high STDs

ll=mean(MeanRTCorrect,3)+(3* std(MeanRTCorrect,[],3));
for nSub=1:nSubj
    if  sum(sum(MeanRTCorrect(:,:,nSub)>ll))>0
        sprintf('%d is a variable subject',nSub)
    end
end

%% Inverse Efficiency vs Difficulty

figure()
subplot(1,2,1)
errorbar(repmat(DiffLvl,1,2), mean(InverseEfficiencyScore(:,[1,2],:),3), ...
    std(InverseEfficiencyScore(:,[1,2],:),[],3)/sqrt(nSubj),'LineWidth',3);
xlabel('Difficulty');
ylabel('Inverse Efficiency');
ylim([0.6 2.5])
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
grid on

subplot(1,2,2)
errorbar(repmat(DiffLvl,1,2), mean(InverseEfficiencyScore(:,[3,4],:),3), ...
    std(InverseEfficiencyScore(:,[3,4],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty');
ylabel('Inverse Efficiency');
ylim([0.6 2.5])
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
grid on

%% Reaction Time vs Difficulty

figure()
subplot(1,2,1)
errorbar(repmat(DiffLvl,1,2), mean(MeanRTCorrect(:,[1 2],:),3), ...
    std(MeanRTCorrect(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty');
ylabel('Reaction time');
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
grid on

subplot(1,2,2)
errorbar(repmat(DiffLvl,1,2), mean(MeanRTCorrect(:,[3 4],:),3), ...
    std(MeanRTCorrect(:,[3 4],:),[],3)/sqrt(nSubj),'LineWidth',3)

xlabel('Difficulty');
ylabel('Reaction time');
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
grid on

%% Performance vs Difficulty for Individual data
figure();

% plot(repmat(DiffLvl,1,4), mean(MeanScore,3),'LineWidth',3,'Marker','o')

errorbar(repmat(DiffLvl,1,4), mean(MeanScore,3), ...
    std(MeanScore,[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty Level');
ylabel('% Correct');
legend(GroupNames)
set(gca,'XTick',DiffLvl);
xlim([3,11]);
ylim([.3 1])
grid on
%% Performance vs Difficulty
figure()
subplot(1,2,1)
hold on

errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[1 2],:),3), ...
   std(MeanScore(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty Level');
ylabel('% Correct');
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
xlim([3,11]);
ylim([.3 1])
grid on


subplot(1,2,2)
hold on

errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[3 4],:),3), ...
   std(MeanScore(:,[3 4],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty Level');
ylabel('% Correct');
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
xlim([3,11]);
ylim([.3 1])
grid on

%% barplots: performance/RT/IES vs difficulty
% MeanRTCorrect=MeanRTCorrect(:,:,setxor(1:18,[5,11,12]));
% InverseEfficiencyScore(:,:,[5,11,12])=[];
errorNeg = [0;0;0;0];
figure()
subplot(1,2,1)
hold on

bar(repmat(DiffLvl,1,2), mean(InverseEfficiencyScore(:,[1 2],:),3), 'BaseValue',0.5)
errorbar(DiffLvl-.3, mean(InverseEfficiencyScore(:,1,:),3),errorNeg,std(InverseEfficiencyScore(:,1,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);
errorbar(DiffLvl+.3, mean(InverseEfficiencyScore(:,2,:),3),errorNeg,std(InverseEfficiencyScore(:,2,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);

xlabel('Difficulty Level');
% ylabel('Proportion Correct');
ylabel('Inverse Efficiency Score (IES)');
% legend({'Predictable','Unpredictable'})
set(gca,'XTick',DiffLvl,'XTickLabel',4:2:10,'YTick',.5:.5:3,'FontSize',15);
xlim([3,11]);
ylim([.5 3])
% grid on


subplot(1,2,2)
hold on

bar(repmat(DiffLvl,1,2), mean(InverseEfficiencyScore(:,[3 4],:),3), 'BaseValue',0.5)
errorbar(DiffLvl-.3, mean(InverseEfficiencyScore(:,3,:),3),errorNeg,std(InverseEfficiencyScore(:,3,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);
errorbar(DiffLvl+.3, mean(InverseEfficiencyScore(:,4,:),3),errorNeg,std(InverseEfficiencyScore(:,4,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);

xlabel('Difficulty Level');
% ylabel('Proportion Correct');
ylabel('Inverse Efficiency Score (IES)');
% legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl,'XTickLabel',4:2:10,'YTick',.5:.5:3,'FontSize',15);
xlim([3,11]);
ylim([.5 3])
% grid on

%% tempatt effect on Score/IES
% TempAttEff_short = TempAttEff_long';
figure(); hold on
errorbar([2,4,6,8], mean(TempAttEff_short), std(TempAttEff_short)./sqrt(nSubj),'k.','MarkerSize',10)
yline(0, 'r--', 'LineWidth', 4);
set(gca,'XTick',2:2:8, 'XTickLabel',4:2:10);
xlim([1 9])
xlabel('Difficulty'); 
ylabel('Temporal attention effect');

%Violin plot
figure()
vs = violinplot(TempAttEff_short);
set(gca,'XTick',1:4, 'XTickLabel',4:2:10);
xlim([.5 4.5])
xlabel('Difficulty'); 
ylabel('Temporal attention effect');

%% Running ANOVAs just to screen
% InverseEfficiencyScore1=reshape(InverseEfficiencyScore,4,[]); %make a 18*4 by 4 matrix of pred+FP * nSub by diff
% InverseEfficiencyScore1=InverseEfficiencyScore1';
% InverseEfficiencyScore1=reshape(InverseEfficiencyScore1,[],4);
% InverseEfficiencyScore1=reshape(InverseEfficiencyScore1,4,[]);
% InverseEfficiencyScore1=InverseEfficiencyScore1';
%% Separate performance groups - %correct vs. diff level in short/long/good perf/bad perf

for perfGrp = 1:2
    figure(perfGrp)

subplot(1,2,1)
hold on

errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[1 2],(pg == perfGrp)),3), ...
   std(MeanScore(:,[1 2],(pg == perfGrp)),[],3)/sqrt(nSubj),'LineWidth',3);


xlabel('Difficulty Level');
ylabel('% Correct-Mean');
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
xlim([3,11]);
ylim([.3 1])
grid on

subplot(1,2,2)
hold on
errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[3 4],(pg == perfGrp)),3), ...
    std(MeanScore(:,[3 4],(pg == perfGrp)),[],3)/sqrt(nSubj),'LineWidth',3);


xlabel('Difficulty Level');
ylabel('% Correct-Mean');
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
xlim([3,11]);
ylim([.3 1])
grid on

sgtitle(['Performance Group: ' num2str(perfGrp)])
end

%% std vs diff -- bar
figure()
% bar(repmat(DiffLvl,1,4), std(MeanScore,[],3)/sqrt(nSubj));
% bar(repmat(DiffLvl,1,4), mean(CVScore,3));

xlabel('Difficulty');
% ylabel('STE Subject');
% ylabel('STEM');
ylabel('CV Subject');
legend(GroupNames)
set(gca,'XTick',DiffLvl);
xlim([2,12]);

%% Score/CV/STE vs Diff level/Condition- subjects separately

figure
colmp=colormap('colorcube');
SubNames=cell(numel(nSubj));
% GroupShortNames={'SP','SUp','LP','LUp'};
GroupNames ={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};

for nSub=1:nSubj+1
    for ii=1:4
        hold on
        subplot(2,2,ii)
        if nSub==nSubj+1
            plot(DiffLvl, MeanScoreMean(:,ii),'r','Marker','.','MarkerSize',20,'MarkerFaceColor','r','MarkerEdgeColor','r')
        else
            %     scatter(DiffLvl, STEScore(:,ii,nSub),'MarkerFaceColor',colmp(nSub,:),'MarkerEdgeColor',colmp(nSub,:));
            %     scatter(DiffLvl, CVScore(:,ii,nSub),'MarkerFaceColor',colmp(nSub,:),'MarkerEdgeColor',colmp(nSub,:));
            
            scatter(DiffLvl+rand/2, MeanScore(:,ii,nSub),'MarkerFaceColor',colmp(nSub,:),'MarkerEdgeColor',colmp(nSub,:));
        end
        %     plot(DiffLvl, MeanScore(:,ii,nSub),'MarkerFaceColor',colmp(nSub,:),'MarkerEdgeColor',colmp(nSub,:),'LineWidth',3);
        xlabel('Difficulty');
        %     xlabel('Condition')
        ylabel('Score');
        set(gca,'XTick',DiffLvl)
        %     ,'XTickLabel',GroupShortNames);
        title(GroupNames{ii})
        %     title('DiffLevel = ' num2str(ii))
        xlim([2,12]);
        ylim([0 1.2]);
        
    end
    
    if nSub==nSubj+1; SubNames(nSub)={'Average'};
    else; SubNames(nSub)={['subject-' num2str(nSub)]};end
    legend(SubNames)
    
end

%% Dot plot --  conditions separate, difficulties separate

figure
colmp=colormap('colorcube');
SubNames=cell(numel(nSubj));
GroupNames ={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};

% for difflev=1:length(DiffLvl)
    for nSub=1:nSubj+1
        
        hold on
        subplot(1,2,1)
        
        if nSub==nSubj+1
            %             plot([1 3],MeanScoreMean(difflev,[1 2]),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
                        plot([1 3],mean(MeanScoreMean(:,[1 2]),1),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            %             plot([1 3],STDScoreMean(difflev,[1 2]),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            %             plot([1 3],InverseEfficiencyMean(difflev,[1 2]),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            
        else
            %             plot([1 3],MeanScore(difflev,[1 2],nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
                        plot([1 3],mean(MeanScore(:,[1 2],nSub),1),'LineWidth',1,'Color',colmp(5*nSub,:),'Marker','o','MarkerFaceColor',colmp(5*nSub,:),'MarkerSize',5);
            %             plot([1 3],STDScore(difflev,[1 2],nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            %             plot([1 3],InverseEfficiencyScore(difflev,[1 2],nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            disp(mean(MeanScore(:,[1 2],nSub),1))
            %             xlabel('Condition')
%             if difflev==1
                            ylabel('% Correct');
                %             ylabel('STD');
                %             ylabel('Inverse Efficiency Score');

%             end
            set(gca,'XTick',[1 3],'XTickLabel',{'Short Pred.','Short Unpred.'},'XTickLabelRotation',45);
            title('Short Foreperiod')
%             title(['Difficulty Level = ' num2str(difflev)])
            xlim([0,4]);
             ylim([.3 1.1]);
%             ylim([.5 2.5]);
        end
        
        hold on
        subplot(1,2,2)
        
        if nSub==nSubj+1
            %             plot([1 3],MeanScoreMean(difflev,[3 4]),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            plot([1 3],mean(MeanScoreMean(:,[3 4]),1),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            %             plot([1 3],STDScoreMean(difflev,[3 4]),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            %             plot([1 3],InverseEfficiencyMean(difflev,[3 4]),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
        else
            %             plot([1 3],MeanScore(difflev,[3 4],nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            plot([1 3],mean(MeanScore(:,[3 4],nSub),1),'LineWidth',1,'Color',colmp(5*nSub,:),'Marker','o','MarkerFaceColor',colmp(5*nSub,:),'MarkerSize',5);
            %             plot([1 3],STDScore(difflev,[3 4],nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            %             plot([1 3],InverseEfficiencyScore(difflev,[3 4],nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            %             xlabel('Condition')
%             if difflev==1
                ylabel('% Correct');
                %             ylabel('STD');
                %             ylabel('Inverse Efficiency Score');
%             end
            set(gca,'XTick',[1 3],'XTickLabel',{'Long Pred.','Long Unpred.'},'XTickLabelRotation',45);
            title('Long Foreperiod')
            %             title(['DiffLevel = ' num2str(difflev)])
            xlim([0,4]);
            ylim([.3 1.1]);
            %             ylim([.5 2.5]);
        end
        if  nSub==nSubj+1; SubNames(nSub)={'Average'}; else; SubNames(nSub)={['subject-' num2str(nSub)]}; end
    end
% end
legend(SubNames)
%% t-test score,STE

hScore =zeros(4,1); pScore = zeros(4,1);
hIE=zeros(4,1); pIE=zeros(4,1);
hSTE=zeros(4,1); pSTE=zeros(4,1);
hCV=zeros(4,1); pCV=zeros(4,1);
hBias=zeros(4,1); pBias=zeros(4,1);
for i=1:4
    [hScore(i),pScore(i)]=ttest(MeanScore(i,1,:),MeanScore(i,2,:));
%     [hScore(i),pScore(i)]=ranksum(median(MeanScore(i,1,:)),median(MeanScore(i,2,:)));
    [hIE(i),pIE(i)]=ttest(InverseEfficiencyScore(i,1,:),InverseEfficiencyScore(i,2,:));
%     [pIE(i),hIE(i)]=ranksum(InverseEfficiencyMean(i,1),InverseEfficiencyMean(i,2));
%     [pIE(i),hIE(i)]=ranksum(median(InverseEfficiencyScore(i,1,:)),median(InverseEfficiencyScore(i,2,:)));
    [hSTE(i),pSTE(i)]=ttest(STEScore(i,1,:),STEScore(i,2,:));
    [hCV(i),pCV(i)]=ttest(CVScore(i,1,:),CVScore(i,2,:));
%     [hBias(i),pBias(i)]=ttest(MeanResponse(i,1,:),ones(1,1,10)*1.5);
end

%% Overtime performance different conditions separately

Data = AllData{1};
heightDataTable=height(Data);
timeBins=10;
OverTimeLabel=nan(heightDataTable,1); %create  latency label for each trial -- 216 is the number of trials for each participant after removal of unnecessary ones
while mod(heightDataTable,timeBins)~=0; timeBins=timeBins+1; end

binTrials=heightDataTable/timeBins;
for i=1:timeBins; OverTimeLabel(binTrials*(i-1)+1:binTrials*i)=ones(binTrials,1)*i; end

% meanOverTime = nan(max(timeBins),length(DiffLvl),2);
% steOverTime = nan(max(timeBins),length(DiffLvl),2);

meanOverTime = nan(max(timeBins),2);
steOverTime = nan(max(timeBins),2);


for timeLbl = 1:max(timeBins)   
%     for diflvl = 1:length(DiffLvl)
   
        scoreOverTime1 = [];
        scoreOverTime2 = [];

        for i = 1:nSubj
            Data = AllData{i};
            
            Data.OverTimeLabel=OverTimeLabel;
            Data(Data.foreperiod == 1.4,:) = [];
            
%             mskOverTime1 = Data.OverTimeLabel == timeLbl & Data.predictability == 1 & Data.difficulty == DiffLvl(diflvl);
%             mskOverTime2 = Data.OverTimeLabel == timeLbl & Data.predictability == 3 & Data.difficulty == DiffLvl(diflvl);
            
            mskOverTime1 = Data.OverTimeLabel == timeLbl & Data.predictability == 1;
            mskOverTime2 = Data.OverTimeLabel == timeLbl & Data.predictability == 3;

            scoreOverTime1 = [scoreOverTime1;Data.score(mskOverTime1)];
            scoreOverTime2 = [scoreOverTime2;Data.score(mskOverTime2)];
            
        end
        
%         meanOverTime(timeLbl,diflvl,1) = mean(scoreOverTime1); %Predictable
%         meanOverTime(timeLbl,diflvl,2) = mean(scoreOverTime2); %Unpredictable
%         steOverTime(timeLbl,diflvl,1) = std(scoreOverTime1)/sqrt(nSubj);%Predictable
%         steOverTime(timeLbl,diflvl,2) = std(scoreOverTime2)/sqrt(nSubj);%Unpredictable
        
        meanOverTime(timeLbl,1) = mean(scoreOverTime1); %Predictable
        meanOverTime(timeLbl,2) = mean(scoreOverTime2); %Unpredictable
        steOverTime(timeLbl,1) = std(scoreOverTime1)/sqrt(nSubj);%Predictable
        steOverTime(timeLbl,2) = std(scoreOverTime2)/sqrt(nSubj);%Unpredictable

       
%     end    
end
%% Overtime plot--short delays

timeBin=unique(OverTimeLabel);
dif=1; %Temporary for when not having difLevels
figure()

% for dif=1:length(DiffLvl)
%     subplot(2,2,dif)
    hold on

%      bar(repmat(timeBin,1,2),squeeze(meanOverTime(:,dif,:)));
    bar(repmat(timeBin,1,2),meanOverTime);
    
%     errorbar(timeBin-.15,meanOverTime(:,dif,1),steOverTime(:,dif,1),'b.')
%     errorbar(timeBin+.15,meanOverTime(:,dif,2),steOverTime(:,dif,2),'r.')    
    errorbar(timeBin-.15,meanOverTime(:,1),steOverTime(:,1),'b.')
    errorbar(timeBin+.15,meanOverTime(:,2),steOverTime(:,2),'r.')
    
    xlim([0 max(timeBin)+1]);
    ylim([.1 1.1]);
    set(gca,'XTick',timeBin,'XTickLabel',timeBin)
    xlabel('Time points');
    ylabel('% Correct');
    grid on
%     title(['Difficulty = ', num2str(dif)])
%         if dif==length(DiffLvl); 
            legend(GroupNames{1:2}); 
%         end;
% end

%% STD vs performance

figure;

msk1 = Pred(:) == 1 & ForeP(:) == 0.65;
msk2 = Pred(:) == 3 & ForeP(:) == 0.65;

hold on
scatter(MeanScore(msk1),STDScore(msk1),'r','filled');
scatter(MeanScore(msk2),STDScore(msk2),'b','filled');

% scatter(MeanRTCorrect(Pred(:) == 1),CVScore(Pred(:) == 1),'m','filled');
% scatter(MeanRTCorrect(Pred(:) == 3),CVScore(Pred(:) == 3),'c','filled');

ylabel('STD');
xlabel('Performance');
% xlabel('RT');

%% Confidence vs Difficulty
figure()
hold on
errorbar(repmat(DiffLvl,1,2), mean(MeanConfCorrect(:,[1 2],:),3), ...
    std(MeanConfCorrect(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3);
xlabel('Difficulty');
ylabel('Confidence');
ylim([1 4]);
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
grid on

%% RT vs Conf
figure;
hold on
scatter(MeanRTCorrect(Pred(:) == 1),MeanConfCorrect(Pred(:) == 1),'r','filled');
scatter(MeanRTCorrect(Pred(:) == 3),MeanConfCorrect(Pred(:) == 3),'b','filled');
xlabel('Reaction Time');
ylabel('Confidence');

%% FP vs Conf
figure;
hold on
msk1 = Pred(:) == 1 & ForeP(:) == 0.65;
msk2 = Pred(:) == 3 & ForeP(:) == 0.65;

scatter(MeanScore(msk1),MeanConfCorrect(msk1),'r','filled');
scatter(MeanScore(msk2),MeanConfCorrect(msk2),'b','filled');
xlabel('Performance');
ylabel('Confidence');

%% ForeP vs RT/IE
figure
msk = ForeP(:) == min(ForeP(:));
[Avg,St] = grpstats(InverseEfficiencyScore(msk), Pred(msk),{'mean','sem'});
% [Avg,St] = grpstats(MeanRTCorrect(msk), Pred(msk),{'mean','sem'});
hold on
bar([1,3],Avg)
errorbar([1,3], Avg, St,'.k');
xlabel('Predictability'); 
ylabel('IE');
% ylabel('Performance');
%% Bias vs difficulty group by pred/
figure()
errorbar(repmat(DiffLvl,1,4), mean(MeanResponse,3), ...
    std(MeanResponse,[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty');
ylabel('Response (Left/Right)');
legend(GroupNames)
set(gca,'XTick',DiffLvl);

%% performance vs difficulty - fit
s=1;
% wbl = 'a + exp(-b*x) + c/b *(1 - exp(-b*x))'; %Wilsch 2018
% wbl = '1/(1+exp(-b*(x-a)))'; %Logit
% wbl= 'a+(b/1+exp(-(x-c)/d))';
 wbl='1-exp(-(x/a)^b)'; %classical Weibull : which is implemented using the combination of an exponential-sigmoid and a poly-core. 
% wbl='1-exp(-exp(a*log(x)+b))'; %a Weibull can also be obtained using a gumbel sigmoid and the log-core 

%the first parameter in Weibull gives the midpoint (threshold) the
%second is the slop at midpoint.

x = 1 - (DiffLvl - 4)/6;
a = zeros(4, nSubj);
b = zeros(4, nSubj);
c = zeros(4, nSubj);

ta = zeros(4, 4); 
tb = zeros(4, 4); 
tc = zeros(4, 4);

figure()
cmap=colormap('colorcube');
hold on
GroupNamesForFit={'PS','PS','UpS','UpS','PL','PL','UpL','UpL'};

for i = 1:4
%     for s = 1:nSubj
        y = mean(MeanScore(:,i,:),3);
        ptp = fit(x,y,wbl,'Lower',[0.2,0.2], 'Upper',[1.5, 1.5],'Start',[0 .5]);
        a(i,s) = ptp.a;
        b(i,s) = ptp.b;
%         c(i,s) = p.c;
        p=plot(ptp,x,y);
        set(p,'Color',cmap(i,:),'LineWidth',3);
        xlim([-.5 1.5]);ylim([-.5 1.5]);
        legend(GroupNamesForFit{1:4})
%     end
%     for j=1:i-1 %is used when subjects are being fitted separately
%         [~,ta(i,j)] = ttest(a(i,:),a(j,:));
%         [~,tb(i,j)] = ttest(b(i,:),b(j,:));
%         [~,tc(i,j)] = ttest(c(i,:),c(j,:));
%     end
end

%% roc curve analysis
TPR = ConfusMat(:,:,:,1,1)./(ConfusMat(:,:,:,1,1) + ConfusMat(:,:,:,1,2));
FPR = ConfusMat(:,:,:,2,1)./(ConfusMat(:,:,:,2,1) + ConfusMat(:,:,:,2,2));

AvgTPR = mean(TPR,3);
AvgFPR = mean(FPR,3);
figure()
hold on;
for i = 1:4
    plot(AvgFPR(:,i), AvgTPR(:,i));
end
legend(GroupNames)
xlabel('FPR'); ylabel('TPR');
plot([0,1],[0,1], 'k--');

%% TPR
TPR = ConfusMat(:,:,:,1,1)./(ConfusMat(:,:,:,1,1) + ConfusMat(:,:,:,1,2));
zTransTPR=zscore(TPR);

figure();
hold on
for i = 1:nSubj
    plot(DiffLvl,TPR(:,1,i),'b');
    plot(DiffLvl,TPR(:,2,i),'r');
end
legend(GroupNames{1:2})
xlabel('Difficulty'); ylabel('TPR');
errorbar(DiffLvl, nanmean(TPR(:,1,:),3), nanstd(TPR(:,1,:),[],3)/sqrt(nSubj),'b--','LineWidth',3)
errorbar(DiffLvl, nanmean(TPR(:,2,:),3), nanstd(TPR(:,2,:),[],3)/sqrt(nSubj),'r--','LineWidth',3)

%% TNR
FPR = ConfusMat(:,:,:,2,1)./(ConfusMat(:,:,:,2,1) + ConfusMat(:,:,:,2,2));
zTransFPR=zscore(FPR);
TNR = 1- FPR;

figure;
hold on
for i = 1:nSubj
    plot(DiffLvl,TNR(:,1,i),'b');
    plot(DiffLvl,TNR(:,2,i),'r');
end

legend(GroupNames{1:2})
xlabel('Difficulty'); ylabel('TNR');
errorbar(DiffLvl, nanmean(TNR(:,1,:),3), nanstd(TNR(:,1,:),[],3)/sqrt(nSubj),'b--','LineWidth',3)
errorbar(DiffLvl, nanmean(TNR(:,2,:),3), nanstd(TNR(:,2,:),[],3)/sqrt(nSubj),'r--','LineWidth',3)

%% d prime bar plots
TPR = ConfusMat(:,:,:,1,1)./(ConfusMat(:,:,:,1,1) + ConfusMat(:,:,:,1,2)); zTransTPR=zscore(TPR,[],'all');
FPR = ConfusMat(:,:,:,2,1)./(ConfusMat(:,:,:,2,1) + ConfusMat(:,:,:,2,2)); FPR(isnan(FPR)) = 0; zTransFPR=zscore(FPR,[],'all');

dPrime=zTransTPR-zTransFPR;

figure; %individual data
hold on
for i = 1:nSubj
    plot(DiffLvl,dPrime(:,1,i),'b');
    plot(DiffLvl,dPrime(:,2,i),'r');
end
legend(GroupNames{1:2})
xlabel('Difficulty'); ylabel('d prime');
errorbar(DiffLvl, nanmean(dPrime(:,1,:),3), nanstd(dPrime(:,1,:),[],3)/sqrt(nSubj),'b--','LineWidth',3)
errorbar(DiffLvl, nanmean(dPrime(:,2,:),3), nanstd(dPrime(:,2,:),[],3)/sqrt(nSubj),'r--','LineWidth',3)

errorNeg = [0;0;0;0];

figure() 

%short
subplot(1,2,1);hold on;
bar(repmat(DiffLvl,1,2), mean(dPrime(:,[1 2],:),3),'BaseValue',-3);
errorbar(DiffLvl-.3, mean(dPrime(:,1,:),3),errorNeg, std(dPrime(:,1,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)
errorbar(DiffLvl+.3, mean(dPrime(:,2,:),3),errorNeg, std(dPrime(:,2,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)

ylim([-3 3]); 
legend(GroupNames{1:2})
xlabel('Difficulty'); ylabel('d prime');
grid on

%long
subplot(1,2,2);hold on;
bar(repmat(DiffLvl,1,2), mean(dPrime(:,[3 4],:),3),'BaseValue',-3);
errorbar(DiffLvl-.3, nanmean(dPrime(:,3,:),3),errorNeg, nanstd(dPrime(:,3,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)
errorbar(DiffLvl+.3, nanmean(dPrime(:,4,:),3),errorNeg, nanstd(dPrime(:,4,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)

ylim([-3 3]); 
legend(GroupNames{3:4})
xlabel('Difficulty'); ylabel('d prime');
grid on

%% transform to zscores
% TP_diff zscore
clear;
load('DataFramePooled_pos.mat', 'indiv_TP'); %Data will be unbalanced because positions are not equal in two experiments for each sub (exp 1-> 20, exp2-> 12)
TP_pos_strd = sortrows(indiv_TP(1:486,:),2); 

diff = unique(TP_pos_strd.difficulty); 
zTrans_pooledTP_diff=[];

for diffLev = 1:length(diff)
    mean_score_diffLev(diffLev) = mean(indiv_TP.TP_mean(indiv_TP.difficulty == diff(diffLev))); %for both MPooled and MPooledPred
    std_score_diffLev(diffLev) = std(indiv_TP.TP_mean(indiv_TP.difficulty == diff(diffLev)));   %for both MPooled and MPooledPred
    
    %z-score standardization on true positive rate 
    zTrans_pooledTP_diff = [zTrans_pooledTP_diff;(TP_pos_strd.TP_mean(TP_pos_strd.difficulty == diff(diffLev)) - mean_score_diffLev(diffLev)) ./  std_score_diffLev(diffLev)];  
end

TP_pos_strd.difficulty = zTrans_pooledTP_diff;
TP_pos_strd.Properties.VariableNames{2} = 'z_score';
TP_pos_strd.TP_mean = [];
%interpolate in each position with z-score = 0 for subjects that don't have any positions
%1->289,,2->290,434,,3->X,,4->310,418,436,,5->185,383,,6->X,,7->43,493,,8->440,,9->99,243
%10->496,,11->X,,12->282,,13->X,,14->X,15->321,,16->X,,17->395,,18->431
TP_pos_strd_intrplt.pos_rad = round(TP_pos_strd_intrplt.pos_rad,6);

%average to put all similar positions togeather
TP_pos_intrplt_avg = grpstats(TP_pos_strd_intrplt,{'Subject_code','pos_rad'},{'mean','std'});
TP_pos_intrplt = TP_pos_intrplt_avg(:,[1,2,4]);
    
% save('DataFrame4Diff_TP_diff_zScore','TP_pos_intrplt');  writetable(TP_pos_intrplt,'DataFrame4Diff_TP_diff_zScore.csv');

%% performance groups separately on dPrime
TPR = ConfusMat(:,:,:,1,1)./(ConfusMat(:,:,:,1,1) + ConfusMat(:,:,:,1,2)); zTransTPR=zscore(TPR,[],'all');
FPR = ConfusMat(:,:,:,2,1)./(ConfusMat(:,:,:,2,1) + ConfusMat(:,:,:,2,2)); FPR(isnan(FPR)) = 0; zTransFPR=zscore(FPR,[],'all');

dPrime=zTransTPR-zTransFPR;

for perGrp = 1:2
    
figure();

%short
subplot(1,2,1);hold on;
errorbar(repmat(DiffLvl,1,2), mean(dPrime(:,[1 2],(pg == perGrp)),3), std(dPrime(:,[1 2],(pg == perGrp)),[],3)/sqrt(nSubj),'LineWidth',2)

ylim([-3 3]); 
legend(GroupNames{1:2})
xlabel('Difficulty'); ylabel('d prime');
grid on

%long
subplot(1,2,2);hold on;
errorbar(repmat(DiffLvl,1,2), nanmean(dPrime(:,[3 4],(pg == perGrp)),3), nanstd(dPrime(:,[3 4],(pg == perGrp)),[],3)/sqrt(nSubj),'LineWidth',2)

ylim([-3 3]); 
legend(GroupNames{3:4})
xlabel('Difficulty'); ylabel('d prime');
grid on
sgtitle(['Performance Group: ' num2str(perGrp)]);
end

%% diff vs dPrime: scatter on individual data
TPR = ConfusMat(:,:,:,1,1)./(ConfusMat(:,:,:,1,1) + ConfusMat(:,:,:,1,2)); zTransTPR=zscore(TPR,[],'all');
FPR = ConfusMat(:,:,:,2,1)./(ConfusMat(:,:,:,2,1) + ConfusMat(:,:,:,2,2)); FPR(isnan(FPR)) = 0; zTransFPR=zscore(FPR,[],'all');

dPrime=zTransTPR-zTransFPR;
diff = unique(M.difficulty); 

figure;
cmap=colormap('colorcube');
for dif = 1:length(diff)
    subplot(2,2,dif)
    title(GroupNames{dif})
for nS = 1:size(dPrime,3)+1
    hold on
    if nS == size(dPrime,3)+1; scatter(diff,mean(dPrime(:,dif,:),3),100,'filled','MarkerFaceColor','red','Marker','d');
    else;scatter(diff,dPrime(:,dif,nS),'filled','MarkerFaceColor',cmap(nS*5,:,:));
    end
    xlim([3,11]); xlabel('difficulty level')
    ylabel('dPrime')
end
end

%% t-test tpr tnr
htp =zeros(4,1); ptp = zeros(4,1);
htn =zeros(4,1); ptn = zeros(4,1);
hdp =zeros(4,1); pdp = zeros(4,1);

for i=1:4
    [htp(i),ptp(i)] = ttest(TPR(i,1,:),TPR(i,2,:));
    %     [ptp(i),htp(i)] = ranksum(squeeze(TPR(i,1,:)),squeeze(TPR(i,2,:)));
    [htn(i),ptn(i)] = ttest(TNR(i,1,:),TNR(i,2,:));
    [hdp(i),pdp(i)] = ttest(dPrime(i,1,:),dPrime(i,2,:));
end

%%%%%%%%%%%%%%%%%%%%%%%% POS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% position -- all conditions
clear
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMS4DiffLevel')
load AllData4Diff.mat
nSubj=numel(AllData);
% PosTP = zeros(4,2,nSubj); % rows: 'Bottom-Left','Bottom-Right','Top-Left','Top-Right' columns: 'predictable','unpredictable' 3rd dim: subjects
% stdPosTP = zeros(4,2,nSubj);
% PosRT = zeros(4,2,nSubj);
% stdPosRT = zeros(4,2,nSubj);
% PosIES = zeros(4,2,nSubj);
% PosConf = zeros(4,2,nSubj);
% pg=zeros(1,nSubj); %vector of subject's performance group


for i = setxor(1:nSubj,9)
    Data = AllData{i};
    pg(i) = Data.perfGrp(1); %performance group of each subject
    
    Data(Data.correct_response == 2,:) = [];
%     Data(Data.foreperiod == 1.4,:) = [];
%     Data(Data.difficulty == 4 | Data.difficulty == 10,:)=[];
    
    Data.ypos(abs(Data.ypos)< 0.001) = nan;
    Data.Top = Data.ypos > 0;
    
    Data.xpos(abs(Data.xpos)< 0.001) = nan;   
    Data.Right = Data.xpos > 0;
    
    Data(isnan(Data.ypos),:) =[];
    Data(isnan(Data.xpos),:) =[];
    
    
    M  = grpstats(Data, {'difficulty','Top','Right'}, {'mean','std'});
    PosTP(i,:,:) = reshape(M.mean_score,4,3); % 4,2 for predictability 4,3 for difficulty
    stdPosTP(:,:,i) = reshape(M.std_score,4,3);
%     PosRT(:,:,i) = reshape(M.mean_reaction_time,4,2); 
%     stdPosRT(:,:,i) = reshape(M.std_reaction_time,4,2);
%     PosIES(:,:,i) = PosRT(:,:,i)./PosTP(:,:,i);
%     PosConf(:,:,i) = reshape(M.mean_confidence,4,2);
end
% PosIESTempEff(:,:,:) = PosTP(:,2,:) - PosTP(:,1,:); %the effect of temporal attention on IES
% GroupNames ={'Predictable', 'Unpredictable'};

PosTP_anova = reshape(PosTP,[],3);
[p,tbl] = anova1(PosTP_anova);

%% Pos performance
figure()
hold on

TPmeanMat = mean(PosTP,3); %collaps over predictabilities as well
TPstdMat = std(PosTP,[],3);
bar([2,4,6,8],TPmeanMat)

errorbar([2,4,6,8]-.3, TPmeanMat(:,1), TPstdMat(:,1)/ sqrt(nSubj),'k.');
errorbar([2,4,6,8]+.3, TPmeanMat(:,2), TPstdMat(:,2)/ sqrt(nSubj),'k.');

set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlabel('Position'); 
ylabel('TPR');

%% Pos reaction time
figure()
hold on

RTmeanMat = mean(PosRT,3);
RTstdMat = std(PosRT,[],3);
bar([2,4,6,8],RTmeanMat)

errorbar([2,4,6,8]-.25, RTmeanMat(:,1), RTstdMat(:,1)/ sqrt(nSubj),'k.');
errorbar([2,4,6,8]+.25, RTmeanMat(:,2), RTstdMat(:,2)/ sqrt(nSubj),'k.');

set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlabel('Position'); 
ylabel('RT');

legend(GroupNames)

%% Pos IES
figure()
hold on

PosIES(PosIES == inf) = nan; %some participants TPR = 0 for some quadrants: inf->nan
IESmeanMat = nanmean(PosIES,3);
IESstdMat = nanstd(PosIES,[],3);
bar([2,4,6,8],IESmeanMat)

errorbar([2,4,6,8]-.3, IESmeanMat(:,1), IESstdMat(:,1)/ sqrt(nSubj),'k.');
errorbar([2,4,6,8]+.3, IESmeanMat(:,2), IESstdMat(:,2)/ sqrt(nSubj),'k.');

set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlabel('Position'); 
ylabel('IES');

legend(GroupNames)

%% tempatt effect on IES
PosIESTempEff = squeeze(PosIESTempEff); PosIESTempEff = PosIESTempEff'; %rows->nSubj, columns->positions 

figure(); hold on
errorbar([2,4,6,8], mean(PosIESTempEff), std(PosIESTempEff)./sqrt(nSubj),'k.','MarkerSize',10)
yline(0, 'r--', 'LineWidth', 4);
set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlim([1 9])
xlabel('Position'); 
ylabel('Temporal attention effect');

%Violin plot
figure()
vs = violinplot(PosIESTempEff);
ylabel('Temporal attention enhancement');
set(gca,'XTick',1:4, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlim([.5, 4.5]);

%% Pos confidence
figure()
hold on
bar(mean(PosConf))
errorbar(1:4, mean(PosConf), std(PosConf)/ sqrt(nSubj),'k.');
set(gca,'XTick',1:4, 'XTickLabel',PosGroups);

%% Pooled data for quadrant analysis
clear
load PooledData4Diff.mat
% PosTP = zeros(4,2,2); % rows: 'Bottom-Left','Bottom-Right','Top-Left','Top-Right' columns: 'predictable','unpredictable' 3rd dim: 'short', 'long'
% PosTP_std = zeros(4,2,2);
% PosRT_TP = zeros(4,2,2);
% PosIES = zeros(4,2,2);
% PosConf = zeros(4,2,2);

PooledData(PooledData.correct_response == 2,:) = [];
% PooledData(PooledData.foreperiod == 1.4,:) = [];
% PooledData(PooledData.difficulty == 4 | PooledData.difficulty == 10,:)=[];

PooledData(abs(PooledData.ypos)< 0.0001,:) = [];
PooledData.Top = PooledData.ypos > 0;

PooledData(abs(PooledData.xpos)< 0.0001,:) = [];
PooledData.Right = PooledData.xpos > 0;

% 'foreperiod','predictability'
M  = grpstats(PooledData, {'difficulty','Top','Right'}, {@nanmean,'std'});
PosTP(:,:,:) = reshape(M.nanmean_TP,4,3,[]); 
PosTP_std(:,:,:) = reshape(M.std_TP,4,3,[]);
PosRT_TP(:,:,:) = reshape(M.nanmean_RT_TP,4,3,[]);
PosIES(:,:,:) = PosRT_TP(:,:,:)./PosTP(:,:,:);
PosConf(:,:,:) = reshape(M.nanmean_ConfCorrect,4,3,[]);

posGroups={'Bottom Left','Bottom Right','Top Left','Top Right'};
GroupNames={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};
%% errorbar for pooled data on TP vs quad (pred and unpred)
figure()
subplot(1,2,1)
errorbar(repmat(1:4,2,1)',PosTP(:,:,1),PosTP_std(:,:,1)./sqrt(18),'LineWidth',3);

grid on
ylabel('TPR'); ylim([.2 1.1]);
set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
legend(GroupNames{1:2})

subplot(1,2,2)
errorbar(repmat(1:4,2,1)',PosTP(:,:,2),PosTP_std(:,:,2)./sqrt(18),'LineWidth',3);

grid on
ylabel('TPR'); ylim([.2 1.1]);
set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
legend(GroupNames{3:4})


%% difference between difficulty levels and positions
nSubj=18;
figure()
errorbar(repmat([1,2,3,4],3,1)',PosTP,PosTP_std./sqrt(nSubj),'LineWidth',3)
grid on
ylabel('TPR'); ylim([.2 1.1]);
set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
legend({'diff = 2','diff = 3','diff = 4'})


%% Memory decay index -- not possible due to low data points for each difficulty level
load AllData4Diff.mat
nSubj=numel(AllData);
% pg=zeros(1,nSubj); %vector of subject's performance group
% PosTP = nan(4,2,nSubj); % rows: 'Bottom-Left','Bottom-Right','Top-Left','Top-Right' columns: 'predictable','unpredictable' OR difficulty, 3rd dim: subjects
% stdPosTP = nan(4,2,nSubj);
% PosRT = nan(4,2,nSubj);
% stdPosRT = nan(4,2,nSubj);
% PosIES = nan(4,2,nSubj);
% PosConf = nan(4,2,nSubj);
% memDecIdx = nan(4,nSubj);

for i = 1:nSubj
    Data = AllData{i};
    pg(i) = Data.perfGrp(1);
    
    Data(Data.correct_response == 2,:) = [];
%     Data(Data.foreperiod == 1.4,:) = [];
    
    Data.Top = Data.ypos > 0;
    
    Data.Right = Data.xpos > 0;
    
    %quadrants
    M  = grpstats(Data, {'difficulty','Top','Right'}, {'mean','std'}); 
    PosTP(:,:,i) = reshape(M.mean_score,4,2);
    stdPosTP(:,:,i) = reshape(M.std_score,4,2);
    PosRT(:,:,i) = reshape(M.mean_reaction_time,4,2);
    stdPosRT(:,:,i) = reshape(M.std_reaction_time,4,2);
    PosIES(:,:,i) = PosRT(:,:,i)./PosTP(:,:,i);
    PosConf(:,:,i) = reshape(M.mean_confidence,4,2);
    memDecIdx(:,i) = (PosTP(:,1,i) - PosTP(:,2,i)) ./ (PosTP(:,1,i) + PosTP(:,2,i)); %subtract TP in 8 from 4
    
%     %predictability and memory decay
%     M  = grpstats(Data, {'predictability','difficulty','Top','Right'}, {'mean','std'}); 
%     PosTP(:,:,:,i) = reshape(M.mean_score,4,2,2);
%     memDecIdx(:,:,i) = (PosTP(:,1,:,i) - PosTP(:,2,:,i)) ./ (PosTP(:,1,:,i) + PosTP(:,2,:,i)); %subtract TP in 8 from 4 - rows->quads cols->pred unpred

end
memDecIdx_4diff = [flip(memDecIdx([1,2],:));memDecIdx([3,4],:)]'; % if you want to put rows in order of quadrants (4,3,2,1) for only difficulty
% memDecIdx = [flip(memDecIdx([1,2],:,:));memDecIdx([3,4],:,:)]; %put rows in order of quadrants (4,3,2,1) for diff and predictability

