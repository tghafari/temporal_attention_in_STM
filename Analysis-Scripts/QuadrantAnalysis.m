clear;
% close all
clc;

%% Overall analyses
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants/')
load AllData.mat
nSubj=numel(AllData);
MeanRTCorrect=nan(2,4,nSubj);
MeanScore=nan(2,4,nSubj);
InverseEfficiencyScore=nan(2,4,nSubj);
STDScore=nan(2,4,nSubj);
GroupCnt=nan(2,4,nSubj);
CVScore=nan(2,4,nSubj);
STEScore=nan(2,4,nSubj);
MeanConf=nan(2,4,nSubj);
MeanConfCorrect=nan(2,4,nSubj);
Pred=nan(2,4,nSubj);
ForeP=nan(2,4,nSubj);
ConfusMat=nan(2,4,nSubj,2,2);
MeanResponse=nan(2,4,nSubj);


for i=1:nSubj
    Data = AllData{i};
    
    M = grpstats(Data, {'foreperiod','predictability','difficulty'}, {@nanmean,'std'});
    MeanScore(:,:,i)=reshape(M.nanmean_score,2,[]);
    MeanRTCorrect(:,:,i)=reshape(M.nanmean_RTCorrect,2,[]);
    InverseEfficiencyScore(:,:,i)=MeanRTCorrect(:,:,i)./MeanScore(:,:,i);
    GroupCnt(:,:,i)=reshape(M.GroupCount,2,[]);
    STDScore(:,:,i)=reshape(M.std_score,2,[]);
    STEScore(:,:,i)=STDScore(:,:,i)./sqrt(GroupCnt(:,:,i));
    CVScore(:,:,i)=STDScore(:,:,i)./MeanScore(:,:,i);
    MeanConf(:,:,i)=reshape(M.nanmean_confidence,2,[]);
    MeanConfCorrect(:,:,i)=reshape(M.nanmean_ConfCorrect,2,[]);
    Pred(:,:,i)=reshape(M.predictability,2,[]);
    ForeP(:,:,i)=reshape(M.foreperiod,2,[]);
    MeanResponse(:,:,i)=reshape(M.nanmean_response,2,[]);
    
    ConfusMat(:,:,i,1,1) = reshape(M.nanmean_TP .* M.GroupCount, 2, []);
    ConfusMat(:,:,i,1,2) = reshape(M.nanmean_FN .* M.GroupCount,2, []);
    ConfusMat(:,:,i,2,1) = reshape(M.nanmean_FP .* M.GroupCount, 2, []);
    ConfusMat(:,:,i,2,2) = reshape(M.nanmean_TN .* M.GroupCount, 2, []);
end

MeanScoreMean=mean(MeanScore,3);
InverseEfficiencyMean=mean(InverseEfficiencyScore,3);
CVScoreMean=mean(CVScore,3);
STDScoreMean=mean(STDScore,3);


DiffLvl = unique(Data.difficulty);
GroupNames ={'Predictable-Short', 'Unpredictable-Short',...
    'Predictable-Long', 'Unpredictable-Long'};

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
errorbar(repmat(DiffLvl,1,4), mean(MeanRTCorrect,3), ...
    std(MeanRTCorrect,[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty');
ylabel('Reaction time');
legend(GroupNames)
set(gca,'XTick',DiffLvl);
grid on

%% Performance and RT vs Difficulty
figure()
subplot(1,2,1)
hold on

% errorbar(repmat(DiffLvl,1,2), mean(MeanRT(:,[1 2],:),3), ...
%     std(MeanRT(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3)

errorbar(repmat(DiffLvl,1,2), median(MeanScore(:,[1 2],:),3), ...
    std(MeanScore(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3)

% errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[1 2],:),3), ...
%    std(MeanScore(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3);


xlabel('Difficulty Level');
% ylabel('STE');
% ylabel('RT');
ylabel('% Correct-Median');
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
xlim([3,9]);
ylim([.3 1])
grid on


subplot(1,2,2)
hold on
% errorbar(repmat(DiffLvl,1,2), mean(MeanRT(:,[3 4],:),3), ...
%     std(MeanRT(:,[3 4],:),[],3)/sqrt(nSubj),'LineWidth',3)

% errorbar(repmat(DiffLvl,1,2), median(MeanScore(:,[3 4],:),3), ...
%    std(MeanScore(:,[3 4],:),[],3)/sqrt(nSubj),'LineWidth',3);

errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[3 4],:),3), ...
    std(MeanScore(:,[3 4],:),[],3)/sqrt(nSubj),'LineWidth',3);


xlabel('Difficulty Level');
% ylabel('RT');
ylabel('% Correct-Mean');
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
xlim([3,9]);
ylim([.3 1])
grid on
%% calculate performance for each subject in each diff level in each quadrant
clear;
cd('/Users/Tara/Documents/MATLAB/MATLAB Programs/PhD Thesis Programs/DMS Project/Results/Analyse/DMSQuadrants')
load AllData.mat
nSubj=numel(AllData);
% MeanScore=nan(2,4,4,nSubj); %with foreperiod
MeanScore=nan(2,4,2,nSubj); %without foreperiod OR two positions
% NrmlzMeanScore=nan(1,4,4,nSubj); % performance in 8 normalized by 4
% STDScore=nan(2,4,4,nSubj); %with foreperiod
STDScore=nan(2,4,2,nSubj); %without foreperiod OR two positions
MeanRTCorrect=nan(2,4,4,nSubj);
IEScore=nan(2,4,4,nSubj);
pg=zeros(1,nSubj); %vector of subject's performance group

for i=1:nSubj
    Data=AllData{i};
    pg(i)=Data.perfGrp(1); %define performance group for each subject
    Data(Data.correct_response ==2,:)=[];
    
    Data(abs(Data.ypos)<0.0001,:)=[];
    Data.Top=Data.ypos>0;
    
    Data(abs(Data.xpos)<0.0001,:)=[];
    Data.Right=Data.xpos>0;
    
    %     MM = grpstats(Data, {'Top','Right','foreperiod','predictability','difficulty'}, {@nanmean,'std'}); %all conditions separated
    MM = grpstats(Data, {'Top','foreperiod','predictability','difficulty'}, {@nanmean,'std'}); %only two positions
    %     MM=grpstats(Data,{'Top','Right','predictability','difficulty'},{@nanmean,'std'}); %pool foreperiods
    %     MeanScore(:,:,:,i)=reshape(MM.nanmean_score,2,4,[]); %with foreperiod
    %     NrmlzMeanScore(:,:,:,i)=MeanScore(2,:,:,i)./MeanScore(1,:,:,i);
    MeanScore(:,:,:,i)=reshape(MM.nanmean_score,2,4,[]); %without foreperiod OR two positions
    %     STDScore(:,:,:,i)=reshape(MM.std_score,2,4,[]);
    STDScore(:,:,:,i)=reshape(MM.std_score,2,4,[]); %without foreperiod OR two positions
    %     MeanRTCorrect(:,:,:,i)=reshape(MM.nanmean_RTCorrect,2,4,[]);
    %     InverseEfficiencyScore(:,:,:,i)=MeanRTCorrect(:,:,:,i)./MeanScore(:,:,:,i);
end

% MeanScoreMean=mean(MeanScore,3);
% InverseEfficiencyMean=mean(InverseEfficiencyScore,3);
% CVScoreMean=mean(CVScore,3);
% STDScoreMean=mean(STDScore,3);


DiffLvl = unique(Data.difficulty);
GroupNames ={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};
posGroups={'Bottom Left','Bottom Right','Top Left','Top Right'};

%% errorbars for each quadrant without bargraphs

for pos=1:4
    for prf=1:2
    figure();
    
    subplot(1,2,1)
    hold on
    errorbar(DiffLvl,median(MeanScore(:,1,pos,(pg==prf)),4),std(MeanScore(:,1,pos,(pg==prf)),[],4)./sqrt(nSubj),'LineWidth',3) %adding perfGr
    errorbar(DiffLvl,median(MeanScore(:,2,pos,(pg==prf)),4),std(MeanScore(:,2,pos,(pg==prf)),[],4)./sqrt(nSubj),'LineWidth',3) %adding perfGr
    %     errorbar(DiffLvl,median(MeanScore(:,1,pos,:),4),std(MeanScore(:,1,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    %     errorbar(DiffLvl,median(MeanScore(:,2,pos,:),4),std(MeanScore(:,2,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    %     errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,1,pos,:),4),std(InverseEfficiencyScore(:,1,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    %     errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,2,pos,:),4),std(InverseEfficiencyScore(:,2,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    grid on
    ylabel('TPR');
    %     ylabel('IE');
    xlabel('Difficulty Level');
    %     ylim([.2 1.1]);
    ylim([.2 1.1]);
    set(gca,'XTick',[4 8]);
    legend(GroupNames([1,2]))
    title([posGroups(pos) 'perf grp:' prf])

    subplot(1,2,2)
    hold on
    errorbar(DiffLvl,median(MeanScore(:,3,pos,(pg==prf)),4),std(MeanScore(:,3,pos,(pg==prf)),[],4)./sqrt(nSubj),'LineWidth',3) %adding perfGr
    errorbar(DiffLvl,median(MeanScore(:,4,pos,(pg==prf)),4),std(MeanScore(:,4,pos,(pg==prf)),[],4)./sqrt(nSubj),'LineWidth',3) %adding perfGr
    %     errorbar(DiffLvl,median(MeanScore(:,3,pos,:),4),std(MeanScore(:,3,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    %     errorbar(DiffLvl,median(MeanScore(:,4,pos,:),4),std(MeanScore(:,4,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    %     errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,3,pos,:),4),std(InverseEfficiencyScore(:,3,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    %     errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,4,pos,:),4),std(InverseEfficiencyScore(:,4,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
    grid on
    ylabel('TPR');
    %     ylabel('IE');
    xlabel('Difficulty Level');
    ylim([.2 1.1]);
    %     ylim([.2 2]);
    set(gca,'XTick',[4 8]);
    legend(GroupNames([3,4]))
    
    title([posGroups(pos) 'perf grp:' prf])
    end
end

%% errorbars in all quadrants-foreperiod and predictability variable
%     figure();

for Dif=1:2
%     for prf=1:2
% for pred=1:2
    figure();
    
    subplot(1,2,1)
    hold on
    %     errorbar([1 2 3 4],squeeze(mean(mean(MeanScore(Dif,:,:,:),4))),squeeze(mean(std(MeanScore(Dif,:,:,:),[],4))./sqrt(nSubj)),'LineWidth',3) %Averaging all conditions only to show differences in visual quadrants
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,1,[1,2],:),4)),squeeze(std(MeanScore(Dif,1,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,2,[1,2],:),4)),squeeze(std(MeanScore(Dif,2,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    errorbar([1 2],squeeze(median(MeanScore(Dif,1,[1,2],:),4)),squeeze(std(MeanScore(Dif,1,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    errorbar([1 2],squeeze(median(MeanScore(Dif,2,[1,2],:),4)),squeeze(std(MeanScore(Dif,2,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,1,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,1,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,2,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,2,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,1,:,:),4)),squeeze(std(MeanScore(Dif,1,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,2,:,:),4)),squeeze(std(MeanScore(Dif,2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,1,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,1,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,2,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(median(NrmlzMeanScore(:,pred,:,:),4)),squeeze(std(NrmlzMeanScore(:,pred,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) %normalized scores

    grid on
        ylabel('TPR');
    %     ylabel('IE');
%     ylabel('TPR difficult/easy');
    ylim([.2 1.1]);
%     set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
        set(gca,'XTick',[1 2],'XTickLabel',{'Bottom','Top'},'XTickLabelRotation',45);
    legend(GroupNames([1,2]))
    %     legend('Predictable','Unpredictable')
        title(['Difficulty Level : ',num2str(DiffLvl(Dif))])
    %     ' perf grp: ' num2str(prf)])
% title('Normalized data- Short foreperiod');

    subplot(1,2,2)
    hold on
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,1,[3,4],:),4)),squeeze(std(MeanScore(Dif,1,[3,4],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,2,[3,4],:),4)),squeeze(std(MeanScore(Dif,2,[3,4],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    errorbar([1 2],squeeze(median(MeanScore(Dif,3,[1,2],:),4)),squeeze(std(MeanScore(Dif,3,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    errorbar([1 2],squeeze(median(MeanScore(Dif,4,[1,2],:),4)),squeeze(std(MeanScore(Dif,4,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,3,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,3,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,4,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,4,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,3,:,:),4)),squeeze(std(MeanScore(Dif,3,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(median(MeanScore(Dif,4,:,:),4)),squeeze(std(MeanScore(Dif,4,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,3,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,3,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,4,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,4,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
%     errorbar([1 2 3 4],squeeze(median(NrmlzMeanScore(:,pred+2,:,:),4)),squeeze(std(NrmlzMeanScore(:,pred+2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) %normalized scores
    grid on
        ylabel('TPR');
    %     ylabel('IE');
%     ylabel('TPR difficult/easy');
    ylim([.2 1.1]);
%     set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
        set(gca,'XTick',[1 2],'XTickLabel',{'Bottom','Top'},'XTickLabelRotation',45); %for two positions
    legend(GroupNames([3,4]))
        title(['Difficulty Level : ',num2str(DiffLvl(Dif))])
    %     ' perf grp: ' num2str(prf)])
% title('Normalized data- Long foreperiod');
% end
%     end
end

%% Dot plot --  conditions separate, difficulties separate

colmp=colormap('colorcube');
SubNames=cell(numel(nSubj));

for pos=1:4
    figure;
    for nSub=1:nSubj+1
        
        subplot(2,1,1)
        hold on
        
        if nSub==nSubj+1
            plot([1 3],mean(MeanScore(2,[1 2],pos,:),4),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            
        else
            plot([1 3],MeanScore(2,[1 2],pos,nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            
            ylabel('TPR');
            set(gca,'XTick',[1 3],'XTickLabel',{'Short Pred.','Short Unpred.'},'XTickLabelRotation',45);
            title(posGroups{pos})
            xlim([0,4]);
            ylim([0 1]);
            
        end
        
        subplot(2,1,2)
        hold on
        
        if nSub==nSubj+1
            plot([1 3],mean(MeanScore(2,[3 4],pos,:),4),'LineWidth',3,'Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
            
        else
            plot([1 3],MeanScore(2,[3 4],pos,nSub),'LineWidth',1,'Color',colmp(nSub,:),'Marker','o','MarkerFaceColor',colmp(nSub,:),'MarkerSize',5);
            
            ylabel('TPR');
            set(gca,'XTick',[1 3],'XTickLabel',{'Long Pred.','Long Unpred.'},'XTickLabelRotation',45);
            xlim([0,4]);
            ylim([0 1]);
            
        end
        
        if  nSub==nSubj+1; SubNames(nSub)={'Average'}; else; SubNames(nSub)={['subject-' num2str(nSub)]}; end
        
    end
    legend(SubNames)
end

%% errorbar + bargraphs for subjs individually for each quadrant
for pos=1:4
    figure();
    hold on
    
    bar([4 8],mean(MeanScore(:,:,pos,:),4))
    
    errorbar(DiffLvl-1.1,mean(MeanScore(:,1,pos,:),4),std(MeanScore(:,1,pos,:),[],4)./sqrt(nSubj),'b.')
    errorbar(DiffLvl-.35,mean(MeanScore(:,2,pos,:),4),std(MeanScore(:,2,pos,:),[],4)./sqrt(nSubj),'c.')
    errorbar(DiffLvl+.35,mean(MeanScore(:,3,pos,:),4),std(MeanScore(:,3,pos,:),[],4)./sqrt(nSubj),'g.')
    errorbar(DiffLvl+1.1,mean(MeanScore(:,4,pos,:),4),std(MeanScore(:,4,pos,:),[],4)./sqrt(nSubj),'k.')
    
    set(gca,'XTick',[4 8],'XTickLabel',{'4','8'});
    ylim([0 1.2]);
    xlabel('Difficulty Level');
    ylabel('TPR');
    legend(GroupNames)
end

%% significancy check for position

hPosTP=nan(2,2,4,1); pPosTP=nan(2,2,4,1);
hrnksum=nan(2,2,4,1); prnksum=nan(2,2,4,1);
for posi=1:4
    for predic=[1 3]
        for diffPerf=1:2
            [hPosTP(diffPerf,predic,posi,:),pPosTP(diffPerf,predic,posi,:)] = ttest(MeanScore(diffPerf,predic,posi,:),MeanScore(diffPerf,predic+1,posi,:));
            meanForRnkSum1=MeanScore(diffPerf,predic,posi,:); meanForRnkSum2=MeanScore(diffPerf,predic+1,posi,:);
            [prnksum(diffPerf,predic,posi,:),hrnksum(diffPerf,predic,posi,:)] = ranksum(meanForRnkSum1(:),meanForRnkSum2(:));
        end
    end
end

%% Prepare Data for R--statistical analysis

% Compare pred vs unpred in different quads in short-difficutl condition
MeanScoreShortDif=MeanScore(2,[1,2],:,:);

%%%%%%%%%%%%%%%%%%%%%%%When not enough trials%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Position-- short delays

meanPos = nan(2,length(DiffLvl),2);
stePos = nan(2,length(DiffLvl),2);

for topLbl = 1:2
    
    for diflvl = 1:length(DiffLvl)
        scorePos1 = [];
        scorePos2 = [];
        
        for i = 1:nSubj
            Data = AllData{i};
            
            Data(Data.correct_response == 2,:) = [];
            Data(Data.foreperiod == 1.4,:) = [];
            
            Data(abs(Data.ypos)< 0.0001,:) = [];
            Data.Top = Data.ypos > 0;
            
            mskPos1 = Data.Top== topLbl-1 & Data.predictability == 1 & Data.difficulty == DiffLvl(diflvl);
            mskPos2 = Data.Top== topLbl-1 & Data.predictability == 3 & Data.difficulty == DiffLvl(diflvl);
            scorePos1 = [scorePos1;Data.score(mskPos1)];
            scorePos2 = [scorePos2;Data.score(mskPos2)];
            
        end
        meanPos(topLbl,diflvl,1) = mean(scorePos1); %Predictable
        meanPos(topLbl,diflvl,2) = mean(scorePos2); %Unpredictable
        stePos(topLbl,diflvl,1) = std(scorePos1)/sqrt(nSubj);%Predictable
        stePos(topLbl,diflvl,2) = std(scorePos2)/sqrt(nSubj);%Unpredictable
    end
    
end
%% Position- short delays plots
figure()
PosTop=unique(Data.Top);

for dif=1:length(DiffLvl)
    subplot(2,2,dif)
    hold on
    bar(repmat(PosTop,1,2),squeeze(meanPos(:,dif,:)));
    
    errorbar(PosTop-.15,meanPos(:,dif,1),stePos(:,dif,1),'b.')
    errorbar(PosTop+.15,meanPos(:,dif,2),stePos(:,dif,2),'r.')
    
    xlim([-1 max(PosTop)+1]);
    ylim([.2 1.1]);
    set(gca,'XTick',0:1,'XTickLabel',{'Bottom','Top'})
    xlabel('Position');
    ylabel('% Correct');
    grid on
    title(['Difficulty = ', num2str(dif)])
    if dif==length(DiffLvl); legend(GroupNames{1:2}); end;
    % {'Predictable','Unpredictable'}
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%% POOLED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% position -- all trials pooled
clc;clear;

cd('/Users/Tara/Documents/MATLAB/MATLAB Programs/PhD Thesis Programs/DMS Project/Results/Analyse/')
load('PooledData');

PooledData(PooledData.correct_response == 2,:) = [];
% PooledData(PooledData.foreperiod == 1.4,:) = []; %Only for 4 quadrifields in the last position plots

PooledData(abs(PooledData.ypos)< 0.0001,:) = [];
PooledData.Top = PooledData.ypos > 0;

PooledData(abs(PooledData.xpos)< 0.0001,:) = [];
PooledData.Right = PooledData.xpos > 0;

MPooled = grpstats(PooledData, {'predictability','Top','Right'}, {'mean','std','sem'});  %For 4 quadrifields in the last position plots
PooledScore=reshape(MPooled.mean_score,2,2,[]);
PooledSTEScore=reshape(MPooled.sem_score,2,2,[]);
PooledRT=reshape(MPooled.mean_RTCorrect,2,2,[]);
PooledIEScore=PooledRT./PooledScore;

% MPooled = grpstats(PooledData, {'Top','Right','foreperiod','predictability','difficulty'}, {'mean','std','sem'}); %For top/bottom AND right/left Only
% PooledScore=reshape(MPooled.mean_score,3,4,[]);
% PooledSTDScore=reshape(MPooled.std_score,3,4,[]);
% PooledSTEScore=reshape(MPooled.sem_score,3,4,[]);
% PooledRT=reshape(MPooled.mean_RTCorrect,3,4,[]);
% PooledSTDRT=reshape(MPooled.std_RTCorrect,3,4,[]);
% PooledSTERT=reshape(MPooled.sem_RTCorrect,3,4,[]);
% PooledIEScore=PooledRT./PooledScore;
% PooledFN=reshape(MPooled.mean_FN,3,4,[]);
% PooledSTDFN=reshape(MPooled.std_FN,3,4,[]);
% PooledSTEFN=reshape(MPooled.sem_FN,3,4,[]);
% PooledConf=reshape(MPooled.mean_confidence,3,4,[]);
% PooledSTDConf=reshape(MPooled.std_confidence,3,4,[]);
% PooledSTEConf=reshape(MPooled.sem_confidence,3,4,[]);

% MPooled = grpstats(PooledData, {'Top','foreperiod','predictability','difficulty'}, {'mean','std','sem'}); %For top/bottom OR right/left Only
% PooledScore=reshape(MPooled.mean_score,4,4,[]);
% PooledSTDScore=reshape(MPooled.std_score,4,4,[]);
% PooledSTEScore=reshape(MPooled.sem_score,4,4,[]);
% PooledRT=reshape(MPooled.mean_RTCorrect,4,4,[]);
% PooledSTDRT=reshape(MPooled.std_RTCorrect,4,4,[]);
% PooledSTERT=reshape(MPooled.sem_RTCorrect,4,4,[]);
% PooledIEScore=PooledRT./PooledScore;
% PooledFN=reshape(MPooled.mean_FN,4,4,[]);
% PooledSTDFN=reshape(MPooled.std_FN,4,4,[]);
% PooledSTEFN=reshape(MPooled.sem_FN,4,4,[]);
% PooledConf=reshape(MPooled.mean_confidence,4,4,[]);
% PooledSTDConf=reshape(MPooled.std_confidence,4,4,[]);
% PooledSTEConf=reshape(MPooled.sem_confidence,4,4,[]);


GroupNames ={'Predictable-Short', 'Unpredictable-Short',...
    'Predictable-Long', 'Unpredictable-Long'};
posGroups={'Bottom','Top','Left','Right'};
pDiffLev=unique(PooledData.difficulty);
pPerfGrp=unique(PooledData.perfGrp);

%% Plot pooled position by perfgroup/ diff level

for pos=1:4
    % subplot(2,1,pos)
    figure()
    hold on
    % bar(repmat(pDiffLev,1,4),PooledScore(:,:,pos))
    % bar(repmat(pDiffLev,1,4),PooledFN(:,:,pos))
    % bar(repmat(pPerfGrp,1,4),PooledConf(:,:,pos))
    
    errorbar(pDiffLev-.55,PooledScore(:,1,pos),PooledSTEScore(:,1,pos),'b.')
    errorbar(pDiffLev-.2,PooledScore(:,2,pos),PooledSTEScore(:,2,pos),'c.')
    errorbar(pDiffLev+.2,PooledScore(:,3,pos),PooledSTEScore(:,3,pos),'g.')
    errorbar(pDiffLev+.55,PooledScore(:,4,pos),PooledSTEScore(:,4,pos),'k.')
    
    % errorbar(pDiffLev-.55,PooledFN(:,1,pos),PooledSTEFN(:,1,pos),'b.')
    % errorbar(pDiffLev-.2,PooledFN(:,2,pos),PooledSTEFN(:,2,pos),'c.')
    % errorbar(pDiffLev+.2,PooledFN(:,3,pos),PooledSTEFN(:,3,pos),'g.')
    % errorbar(pDiffLev+.55,PooledFN(:,4,pos),PooledSTEFN(:,4,pos),'k.')
    
    % errorbar(pPerfGrp-.55,PooledConf(:,1,pos),PooledSTEConf(:,1,pos),'b.')
    % errorbar(pPerfGrp-.2,PooledConf(:,2,pos),PooledSTEConf(:,2,pos),'c.')
    % errorbar(pPerfGrp+.2,PooledConf(:,3,pos),PooledSTEConf(:,3,pos),'g.')
    % errorbar(pPerfGrp+.55,PooledConf(:,4,pos),PooledSTEConf(:,4,pos),'k.')
    
    % errorbar(pDiffLev,PooledScore(:,1,pos),PooledSTEScore(:,1,pos),'LineWidth',3)
    % errorbar(pDiffLev,PooledScore(:,2,pos),PooledSTEScore(:,2,pos),'LineWidth',3)
    % errorbar(pDiffLev,PooledScore(:,3,pos),PooledSTEScore(:,3,pos),'LineWidth',3)
    % errorbar(pDiffLev,PooledScore(:,4,pos),PooledSTEScore(:,4,pos),'LineWidth',3)
    
    
    
    ylim([0 1.2]);
    % ylabel('FNR');
    ylabel('TPR');
    % xlabel('Performance Group');
    xlabel('Difficulty Level');
    set(gca,'XTick',[6 8 10]);
    % ,'XTickLabel',{'Best','Medium','Worst'})
    grid on
    legend(GroupNames)
    
end

