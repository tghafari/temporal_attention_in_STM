clear;
% close all
clc;

%% Making dataframes for ANOVA
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants')
load AllDataQuad.mat
nSubj=numel(AllData);
height_MM = 8; %12 for only pos and diff, 48 for pos,8 for no pos
mean_TP=nan(height_MM*nSubj,5); %4 for only pos and diff, 6 for pos 5 for no pos
mean_RT=nan(height_MM*nSubj,5);
mean_IES=nan(height_MM*nSubj,5);
pg=zeros(1,nSubj); %vector of subject's performance group

for i=1:nSubj
    Data=AllData{i};
    pg(i)=Data.perfGrp(1); %define performance group for each subject
    Data(Data.correct_response == 2,:)=[]; %only for TP
    
    %     MM = grpstats(Data, {'pos_rad','difficulty'}, {@nanmean,'std'}); %all conditions separated
    %     mean_TP(height(MM)*(i-1)+1:i*height(MM),:) = [i*ones(height(MM),1),MM.pos_rad,MM.difficulty,MM.nanmean_TP];
    %     mean_RT(height(MM)*(i-1)+1:i*height(MM),:) = [i*ones(height(MM),1),MM.pos_rad,MM.difficulty,MM.nanmean_RT_TP];
    %     MM = grpstats(Data, {'pos_rad','foreperiod','predictability','difficulty'}, {@nanmean,'std'}); %all conditions separated
    %     mean_TP(height(MM)*(i-1)+1:i*height(MM),:) = [i*ones(height(MM),1),MM.pos_rad,MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_TP];
    %     mean_RT(height(MM)*(i-1)+1:i*height(MM),:) = [i*ones(height(MM),1),MM.pos_rad,MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_RT_TP];
        MM = grpstats(Data, {'foreperiod','predictability','difficulty'}, {@nanmean,'std'}); % without position
        mean_TP(height_MM*(i-1)+1:i*height_MM,:) = [i*ones(height_MM,1),MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_score];
        mean_RT(height_MM*(i-1)+1:i*height_MM,:) = [i*ones(height_MM,1),MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_RTCorrect];
        mean_IES(height_MM*(i-1)+1:i*height_MM,:) = [i*ones(height_MM,1),MM.foreperiod,MM.predictability,MM.difficulty,(MM.nanmean_reaction_time./MM.nanmean_score)];

end
% %only position and diff
% mean_TP_table = array2table(mean_TP,'VariableNames',{'Subject_code','pos_rad','difficulty','nanmean_TP'});
% save('DataFrameQuad_TP_diff_2','mean_TP_table');  writetable(mean_TP_table,'DataFrameQuad_TP_2.csv');
% mean_RT_TP_table = array2table(mean_RT,'VariableNames',{'Subject_code','pos_rad','difficulty','nanmean_RT_TP'});
% save('DataFrameQuad_RT_TP_diff_2','mean_RT_TP_table');  writetable(mean_RT_TP_table,'DataFrameQuad_RT_TP_2.csv');
% %with position
% mean_TP_table = array2table(mean_TP,'VariableNames',{'Subject_code','pos_rad','foreperiod','predictability','difficulty','nanmean_TP'});
% save('DataFrameQuad_TP_2','mean_TP_table');  writetable(mean_TP_table,'DataFrameQuad_TP_2.csv');
% mean_RT_TP_table = array2table(mean_RT,'VariableNames',{'Subject_code','pos_rad','foreperiod','predictability','difficulty','nanmean_RT_TP'});
% save('DataFrameQuad_RT_TP_2','mean_RT_TP_table');  writetable(mean_RT_TP_table,'DataFrameQuad_RT_TP_2.csv');
% %without position
% mean_score_table = array2table(mean_TP,'VariableNames',{'Subject_code','foreperiod','predictability','difficulty','nanmean_score'});
% % save('DataFrameQuad_score_2','mean_score_table');  writetable(mean_score_table,'DataFrameQuad_score_2.csv');
% mean_RTCor_table = array2table(mean_RT,'VariableNames',{'Subject_code','foreperiod','predictability','difficulty','nanmean_RTCorrect'});
% save('DataFrameQuad_RTCor_2','mean_RTCor_table');  writetable(mean_RTCor_table,'DataFrameQuad_RTCor_2.csv');
% mean_IES_table = array2table(mean_IES,'VariableNames',{'Subject_code','foreperiod','predictability','difficulty','IES'});
% save('DataFrameQuad_IES_2','mean_IES_table');  writetable(mean_IES_table,'DataFrameQuad_IES_2.csv');

%% Overall analyses
clear
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants/')
load AllDataQuad.mat
nSubj=numel(AllData);
pg=zeros(1,nSubj); %vector of subject's performance group
MeanScore=nan(2,4,nSubj); %rows->diff lev columns->short pred, short unpred, long pred, long unpred
MeanRTCorrect=nan(2,4,nSubj);
MeanRT=nan(2,4,nSubj);
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
PashlerK=nan(2,4,nSubj);


for i=1:nSubj
    Data = AllData{i};
    pg(i)=Data.perfGrp(1); %define performance group for each subject
    
    M = grpstats(Data, {'foreperiod','predictability','difficulty'}, {@nanmean,'std'});    
    MeanScore(:,:,i)=reshape(M.nanmean_score,2,[]);
    MeanRTCorrect(:,:,i)=reshape(M.nanmean_RTCorrect,2,[]);
    MeanRT(:,:,i)=reshape(M.nanmean_reaction_time,2,[]);
    InverseEfficiencyScore(:,:,i)=MeanRT(:,:,i)./MeanScore(:,:,i);
    GroupCnt(:,:,i)=reshape(M.GroupCount,2,[]);
    STDScore(:,:,i)=reshape(M.std_score,2,[]);
    STEScore(:,:,i)=STDScore(:,:,i)./sqrt(GroupCnt(:,:,i));
    CVScore(:,:,i)=STDScore(:,:,i)./MeanScore(:,:,i);
    MeanConf(:,:,i)=reshape(M.nanmean_confidence,2,[]);
    MeanConfCorrect(:,:,i)=reshape(M.nanmean_ConfCorrect,2,[]);
    Pred(:,:,i)=reshape(M.predictability,2,[]);
    ForeP(:,:,i)=reshape(M.foreperiod,2,[]);
    MeanResponse(:,:,i)=reshape(M.nanmean_response,2,[]);
    PashlerK(:,:,i)=reshape(M.difficulty.*((M.nanmean_TP-M.nanmean_FP)./(1-M.nanmean_FP)),2,[]);
    
    ConfusMat(:,:,i,1,1) = reshape(M.nanmean_TP .* M.GroupCount, 2, []);
    ConfusMat(:,:,i,1,2) = reshape(M.nanmean_FN .* M.GroupCount,2, []);
    ConfusMat(:,:,i,2,1) = reshape(M.nanmean_FP .* M.GroupCount, 2, []);
    ConfusMat(:,:,i,2,2) = reshape(M.nanmean_TN .* M.GroupCount, 2, []);
end
% TempAttEff = squeeze(mean(InverseEfficiencyScore(:,[1,3],:)) - mean(InverseEfficiencyScore(:,[2,4],:)));
TempAttEff = squeeze([mean(mean(InverseEfficiencyScore(:,[1,3],:))) ;mean(mean(InverseEfficiencyScore(:,[2,4],:)))]);

MeanScoreMean=mean(MeanScore,3);
IESMean=mean(InverseEfficiencyScore,3);
CVScoreMean=mean(CVScore,3);
STDScoreMean=mean(STDScore,3);


DiffLvl = unique(Data.difficulty);
GroupNames ={'Predictable-Short', 'Unpredictable-Short',...
    'Predictable-Long', 'Unpredictable-Long'};

%% plot each subject separately

for i = 1:nSubj
    figure(i)
    plot([1,3],MeanScore(:,:,i),'LineWidth',3,'Marker','o')
    
    xlabel('Difficulty Level'); xlim([.5,3.5]); xticks([1,3]); xticklabels([4,8]);
    ylabel('% Correct'); ylim([.3 1]);
    grid on
    legend(GroupNames)
end

%% Omit Subjects with high STDs

ll=mean(MeanScore,3)+(2.5* std(MeanScore,[],3));
for nSub=1:nSubj
    if  sum(sum(MeanScore(:,:,nSub)>ll))>0
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
errorbar(repmat(DiffLvl,1,2), mean(MeanRTCorrect(:,[1,2],:),3), ...
    std(MeanRTCorrect(:,[1,2],:),[],3)/sqrt(nSubj),'LineWidth',3);
xlabel('Difficulty');
ylabel('Inverse Efficiency');
ylim([0.6 2.5])
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
grid on

subplot(1,2,2)
errorbar(repmat(DiffLvl,1,2), mean(MeanRTCorrect(:,[3,4],:),3), ...
    std(MeanRTCorrect(:,[3,4],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty');
ylabel('Reaction Time ');
ylim([0.6 2.5])
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
grid on

%all conds in one plot
figure()
errorbar(repmat(DiffLvl,1,4), mean(MeanRTCorrect,3), ...
    std(MeanRTCorrect,[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty');
ylabel('Reaction time');
legend(GroupNames)
set(gca,'XTick',DiffLvl);
grid on

%% Performance vs Difficulty
figure()
subplot(1,2,1)
hold on

errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[1 2],:),3), ...
    std(MeanScore(:,[1 2],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty Level');
ylabel('% Correct-Mean');
legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl);
xlim([3,9]);
ylim([.3 1])
grid on


subplot(1,2,2)
hold on

errorbar(repmat(DiffLvl,1,2), mean(MeanScore(:,[3 4],:),3), ...
    std(MeanScore(:,[3 4],:),[],3)/sqrt(nSubj),'LineWidth',3);

xlabel('Difficulty Level');
ylabel('% Correct-Mean');
legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl);
xlim([3,9]);
ylim([.3 1])
grid on

%% barplots: performance/RT/IES vs difficulty
errorNeg = [0;0];
figure()
subplot(1,2,1)
hold on
%InverseEfficiencyScore    'Inverse Efficiency Score (IES)'
bar(repmat(DiffLvl,1,2), mean(MeanScore(:,[1 2],:),3),'BaseValue',0.5)
errorbar(DiffLvl-.6, mean(MeanScore(:,1,:),3),errorNeg,std(MeanScore(:,1,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);
errorbar(DiffLvl+.6, mean(MeanScore(:,2,:),3),errorNeg,std(MeanScore(:,2,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);

xlabel('Difficulty Level','FontSize',7);
ylabel('Proportion Correct','FontSize',7);
% legend(GroupNames{1:2})
set(gca,'XTick',DiffLvl,'XTickLabel',[4,8],'YTick',.5:.1:1,'FontSize',20); %MeanScore
% set(gca,'XTick',DiffLvl,'XTickLabel',[4,8],'YTick',.5:.5:2,'FontSize',20); %IES
ylim([.5 1]) %MeanScore
% ylim([.5 2.2]) %IES
xlim([2.5,9.5]);
grid on


subplot(1,2,2)
hold on

bar(repmat(DiffLvl,1,2), mean(MeanScore(:,[3 4],:),3),'BaseValue',0.5)
errorbar(DiffLvl-.6, mean(MeanScore(:,3,:),3),errorNeg,std(MeanScore(:,3,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);
errorbar(DiffLvl+.6, mean(MeanScore(:,4,:),3),errorNeg,std(MeanScore(:,4,:),[],3)/sqrt(nSubj),'k.','LineWidth',2);

xlabel('Difficulty Level','FontSize',7);
ylabel('Proportion Correct','FontSize',7);
% legend(GroupNames{3:4})
set(gca,'XTick',DiffLvl,'XTickLabel',[4,8],'YTick',.5:.1:1,'FontSize',20);
% set(gca,'XTick',DiffLvl,'XTickLabel',[4,8],'YTick',.5:.5:2,'FontSize',20);
ylim([.5 1]) %MeanScore
% ylim([.5 2.2]) %IES
xlim([2.5,9.5]);
grid on

%% tempatt effect on Score/IES
% TempAttEff = TempAttEff';
figure(); hold on
errorbar([4,8], mean(TempAttEff), std(TempAttEff)./sqrt(nSubj),'k.','MarkerSize',10)
yline(0, 'r--', 'LineWidth', 4);
set(gca,'XTick',[4 8], 'XTickLabel',[4 8]);
xlim([3 9])
xlabel('Difficulty'); 
ylabel('Temporal attention effect');

%Violin plot
figure()
vs = violinplot(TempAttEff);
set(gca,'XTick',[1 2], 'XTickLabel',[4 8]);
xlim([.5 2.5])
xlabel('Difficulty'); 
ylabel('Temporal attention effect');


%% d-prime bar plots
TPR = ConfusMat(:,:,:,1,1)./(ConfusMat(:,:,:,1,1) + ConfusMat(:,:,:,1,2)); zTransTPR=zscore(TPR,[],'all');
FPR = ConfusMat(:,:,:,2,1)./(ConfusMat(:,:,:,2,1) + ConfusMat(:,:,:,2,2)); zTransFPR=zscore(FPR,[],'all');

dPrime=zTransTPR-zTransFPR;

% figure; %plot subjects separately
% hold on
% for i = 1:nSubj
%     plot(DiffLvl,dPrime(:,1,i),'b');
%     plot(DiffLvl,dPrime(:,2,i),'r');
% end

errorNeg = [0;0];
figure;
%short
subplot(1,2,1);hold on;
bar(repmat(DiffLvl,1,2), mean(dPrime(:,[1 2],:),3),'BaseValue',-3);
errorbar(DiffLvl-.6, nanmean(dPrime(:,1,:),3),errorNeg, nanstd(dPrime(:,1,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)
errorbar(DiffLvl+.6, nanmean(dPrime(:,2,:),3),errorNeg, nanstd(dPrime(:,2,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)

ylim([-3 3]);
legend(GroupNames{1:2})
xlabel('Difficulty'); ylabel('d prime');
grid on
%long
subplot(1,2,2);hold on;
bar(repmat(DiffLvl,1,2), mean(dPrime(:,[3 4],:),3),'BaseValue',-3);
errorbar(DiffLvl-.6, nanmean(dPrime(:,3,:),3),errorNeg, nanstd(dPrime(:,3,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)
errorbar(DiffLvl+.6, nanmean(dPrime(:,4,:),3),errorNeg, nanstd(dPrime(:,4,:),[],3)/sqrt(nSubj),'k.','LineWidth',2)

ylim([-3 3]);
legend(GroupNames{3:4})
xlabel('Difficulty'); ylabel('d prime');
grid on

%% transform to zscores
% TP_diff zscore
load('DataFrameQuad_TP_diff.mat', 'mean_TP_table');
TP_pos_strd = sortrows(mean_TP_table,3);

diff = unique(mean_TP_table.difficulty); 
zTrans_meanTP_diff=[];

for diffLev = 1:length(diff)
    mean_score_diffLev(diffLev) = mean(mean_TP_table.nanmean_TP(mean_TP_table.difficulty == diff(diffLev))); %for both MPooled and MPooledPred
    std_score_diffLev(diffLev) = std(mean_TP_table.nanmean_TP(mean_TP_table.difficulty == diff(diffLev)));   %for both MPooled and MPooledPred
    
    %z-score standardization on true positive rate 
    zTrans_meanTP_diff = [zTrans_meanTP_diff;(TP_pos_strd.nanmean_TP(TP_pos_strd.difficulty == diff(diffLev)) - mean_score_diffLev(diffLev)) ./  std_score_diffLev(diffLev)];  
end
TP_pos_strd.difficulty = zTrans_meanTP_diff;
TP_pos_strd.Properties.VariableNames{3} = 'z_score';
TP_pos_strd.nanmean_TP = [];

% save('DataFrameQuad_TP_diff_zScore','TP_pos_strd');  writetable(TP_pos_strd,'DataFrameQuad_TP_diff_zScore.csv');

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
    xlim([3,9]);
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
    xlim([3,9]);
    ylim([.3 1])
    grid on
    
    title(['Performance Group: ' num2str(perfGrp)])
end

%% Interaction plots-average over FP (due to no significant difference)
mean_over_FP = reshape(mean([MeanScore(:,[1,3],:);MeanScore(:,[2,4],:)],2),2,2,[]); %rows -> diff lev, columns -> pred,unpred 3rd dim->subj
subplot(1,2,1)
errorbar(repmat([1;2],1,2),mean(mean_over_FP,3),std(mean_over_FP,[],3)./sqrt(size(mean_over_FP,3)),'LineWidth',4);
xlim([0 3]); grid on
subplot(1,2,2) %just the short FP, to compare with avg of FPs
errorbar(repmat([1;2],1,2),mean(MeanScore(:,[1,2],:),3),std(MeanScore(:,[1,2],:),[],3)./sqrt(size(MeanScore(:,[1,2],:),3)),'LineWidth',4);
xlim([0 3]); grid on

%% calculate performance for each subject in each diff level in each quadrant
clear;
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants')
load AllDataQuad.mat
nSubj=numel(AllData);
% MeanScore=nan(4,2,2,nSubj); %with foreperiod
% MeanScore=nan(2,2,12,nSubj); %with pos
% MeanScore=nan(2,4,2,nSubj); %without foreperiod OR two positions
% MeanRT_TP=nan(2,4,2,nSubj); %without foreperiod OR two positions
% height_MM = 16;
% NrmlzMeanScore=nan(1,4,4,nSubj); % performance in 8 normalized by 4
% STDScore=nan(2,4,4,nSubj); %with foreperiod
% STDScore=nan(2,4,2,nSubj); %without foreperiod OR two positions
% MeanRTCorrect=nan(2,4,4,nSubj);
% IEScore=nan(2,4,4,nSubj);
% PosTP = zeros(4,2,nSubj); % rows: 'Bottom-Left','Bottom-Right','Top-Left','Top-Right' columns: 'predictable','unpredictable' 3rd dim: subjects
% PosRT = zeros(4,2,nSubj);
% PosIES = zeros(4,2,nSubj);
 
pg=zeros(1,nSubj); %vector of subject's performance group

for i=1:nSubj
    Data=AllData{i};
    pg(i)=Data.perfGrp(1); %define performance group for each subject
    Data(Data.correct_response == 2,:)=[];
%     Data(Data.difficulty == 4,:)=[]; %only for pos_rad analysis
  
    Data.ypos(abs(Data.ypos)< 0.0001) = 0;
    Data.Top=Data.ypos>0;
    Data.xpos(abs(Data.xpos)< 0.0001) = 0;
    Data.Right=Data.xpos>0;
    
%         MM = grpstats(Data, {'foreperiod','predictability','Top','Right'}, {@nanmean,'std'}); %all conditions separated
    %     MM = grpstats(Data, {'pos_rad','foreperiod','predictability'}, {@nanmean,'std'}); %all conditions separated
        MM = grpstats(Data, {'Top','Right','foreperiod','predictability','difficulty'}, {@nanmean,'std'}); %only two positions
%         MM = grpstats(Data, {'predictability','Top','Right'},{@nanmean,'std'}); %pool foreperiods
    
        MeanScore(:,:,:,i)=reshape(MM.nanmean_TP,2,4,4,[]); % 4,2,2,[] with quad OR without foreperiod OR two positions 4,2,[] pred and quadrants
        MeanSTD(:,:,:,i)=reshape(MM.std_TP,2,4,4,[]);
    %     MeanRT(:,:,:,i)=reshape(MM.nanmean_reaction_time,4,2,[]); %with quad OR without foreperiod OR two positions
    %     MeanRT_TP(:,:,:,i)=reshape(MM.nanmean_RT_TP,2,4,[]); %with quad OR without foreperiod OR two positions
    %     mean_TP(height(MM)*(i-1)+1:i*height(MM),:) = [i*ones(height(MM),1),MM.Top,MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_TP,pg(i)*ones(height(MM),1)];
    %     mean_RT(height(MM)*(i-1)+1:i*height(MM),:) = [i*ones(height(MM),1),MM.Top,MM.foreperiod,MM.predictability,MM.difficulty,MM.nanmean_RT_TP,pg(i)*ones(height(MM),1)];
    %     MeanScore(:,:,:,i)=reshape(MM.nanmean_TP,2,2,[]); %with foreperiod and diff = 8 OR with pos
    %     NrmlzMeanScore(:,:,:,i)=MeanScore(2,:,:,i)./MeanScore(1,:,:,i);
    %     STDScore(:,:,:,i)=reshape(MM.std_score,2,4,[]); %without foreperiod OR two positions
    %     MeanRTCorrect(:,:,:,i)=reshape(MM.nanmean_RTCorrect,2,4,[]); %with quad
    %     InverseEfficiencyScore(:,:,:,i)=MeanRT(:,:,:,i)./MeanScore(:,:,:,i);
    %     PosTP(:,:,i) = reshape(MM.nanmean_TP,4,2);
    %     PosSTD(:,:,i) = reshape(MM.std_TP,4,2);
    %     PosRT(:,:,i) = reshape(MM.nanmean_reaction_time,4,2);
    %     PosIES(:,:,i) = PosRT(:,:,i)./PosTP(:,:,i);
end
% PosTempEff(:,:,:) = PosTP(:,2,:) - PosTP(:,1,:); %the effect of temporal attention on IES

% DiffLvl = unique(Data.difficulty);
GroupNames ={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};
posGroups={'Bottom Left','Bottom Right','Top Left','Top Right'};

% %prepare for ANOVA
% mean_TP_table = array2table(mean_TP,'VariableNames',{'Subject_code','Top','foreperiod','predictability','difficulty','nanmean_TP','performanceGroup'});
% save('DataFrameQuad_TP_top_2','mean_TP_table');  writetable(mean_TP_table,'DataFrameQuad_TP_top_2.csv');
% mean_RT_TP_table = array2table(mean_RT,'VariableNames',{'Subject_code','Top','foreperiod','predictability','difficulty','nanmean_RT_TP','performanceGroup'});
% save('DataFrameQuad_RT_TP_top_2','mean_RT_TP_table');  writetable(mean_RT_TP_table,'DataFrameQuad_RT_TP_top_2.csv');

%% Plot each subject individually: pred/unpred vs quad
colmp=colormap('colorcube');
SubNames=cell(numel(nSubj));

for nSub = 1:nSubj
    hold on
    subplot(3,4,nSub)
    plot(repmat([1 2 3 4],2,1)',PosTP(:,:,nSub),'LineWidth',3,'Marker','o','MarkerSize',5)
    xticks([1 2 3 4]); xticklabels(posGroups); xtickangle(45)
    ylabel('TPR'); ylim([0 1])
    grid on
    title(['Sub #' num2str(nSub)])
    if nSub == nSubj; legend({'predictable','unpredictable'}); end
end

% quad vs pred/unpred

for nSub = 1:nSubj
    hold on
    subplot(3,4,nSub)
    plot(repmat([1 2 3 4],2,1)',PosTP(:,:,nSub),'LineWidth',3,'Marker','o','MarkerSize',5)
    xticks([1 2 3 4]); xticklabels(posGroups); xtickangle(45)
    ylabel('TPR'); ylim([0 1])
    grid on
    title(['Sub #' num2str(nSub)])
    if nSub == nSubj; legend({'predictable','unpredictable'}); end
end

%% violin plots for quadrants and predictability
MeanScoreQuad=permute(InverseEfficiencyScore,[4,2,3,1]);
ForePer={'Short','Long'};
MeanScoreQuad = squeeze(mean(MeanScoreQuad,4)); 
MeanScoreQuad(isinf(MeanScoreQuad))=mean(MeanScoreQuad(~isinf(MeanScoreQuad)));

for FP = 1:2
for quad = 1:4
    figure()
    hold on
    
    v=violinplot(MeanScoreQuad(:,:,FP,quad));
    v(1).ViolinColor = [0 0 1]; v(2).ViolinColor = [1 0 0];
    plot([1 2],[mean(MeanScoreQuad(:,1,FP,quad)),mean(MeanScoreQuad(:,2,FP,quad))],'LineWidth',1,'Color','k')
    set(gca,'XTick',[1 2],'XTickLabel',{'predictable','unpredictable'},'XTickLabelRotation',45);
    xlim([.5 2.5]); %ylim([.2 1.1]);
    % xlabel('Predictability');
    ylabel('TPR');
    title([ForePer{FP},' ', posGroups{quad}])
    [p(quad,FP),h(quad,FP)] = ttest2(MeanScoreQuad(:,1,FP,quad),MeanScoreQuad(:,2,FP,quad));
    p_txt = sprintf('p-value = %.3f',h(quad,FP));
    text(.65,1.05,p_txt)

end
end      


%% performance groups for position -- very messy plots
for prf=1:2
    figure();
%     for diff=2
        
        subplot(1,2,1)
        hold on
        errorbar(1:size(MeanScore,3),squeeze(mean(MeanScore(1,1,:,(pg==prf)),4)),squeeze(std(MeanScore(1,1,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',1) %adding perfGr
        errorbar(1:size(MeanScore,3),squeeze(mean(MeanScore(2,1,:,(pg==prf)),4)),squeeze(std(MeanScore(2,1,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',1) %adding perfGr
        grid on
        ylabel('TPR'); ylim([.2 1.1]);
        xlabel('Position'); set(gca,'XTick',1:12);
        
        legend(GroupNames([1,2]))
        sgtitle(['Performance Group:' prf])
        
        subplot(1,2,2)
        hold on
        errorbar(1:size(MeanScore,3),squeeze(mean(MeanScore(1,2,:,(pg==prf)),4)),squeeze(std(MeanScore(1,2,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',1) %adding perfGr
        errorbar(1:size(MeanScore,3),squeeze(mean(MeanScore(2,2,:,(pg==prf)),4)),squeeze(std(MeanScore(2,2,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',1) %adding perfGr
        grid on
        ylabel('TPR'); ylim([.2 1.1]);
        xlabel('Position'); set(gca,'XTick',1:12);
        
        legend(GroupNames([3,4]))
        sgtitle(['Performance Group:' num2str(prf)])
%     end
end

%% errorbars for each quadrant without bargraphs
quad=[1,2,3,4];
top_bot = [1,2];
for prf=1:2
    figure();
    for diff=2
        
        subplot(1,2,1)
        hold on
        errorbar(top_bot,squeeze(mean(MeanScore(diff,1,:,(pg==prf)),4)),squeeze(std(MeanScore(diff,1,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr: only top-bottom
        errorbar(top_bot,squeeze(mean(MeanScore(diff,2,:,(pg==prf)),4)),squeeze(std(MeanScore(diff,2,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr: only top-bottom
        %         errorbar(quad,squeeze(mean(MeanScore(diff,1,:,(pg==prf)),4)),squeeze(std(MeanScore(diff,1,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
        %         errorbar(quad,squeeze(mean(MeanScore(diff,2,:,(pg==prf)),4)),squeeze(std(MeanScore(diff,2,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
        %         errorbar(DiffLvl,mean(MeanScore(:,1,pos,:),4),std(MeanScore(:,1,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        %         errorbar(DiffLvl,mean(MeanScore(:,2,pos,:),4),std(MeanScore(:,2,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        %         errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,1,pos,:),4),std(InverseEfficiencyScore(:,1,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        %         errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,2,pos,:),4),std(InverseEfficiencyScore(:,2,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        grid on
        ylabel('TPR'); ylim([.2 1.1]);
        %     ylabel('IE'); ylim([.2 1.1]);
        %         set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
        set(gca,'XTick',[1 2 3 4],'XTickLabel',{'Bottom','Top'},'XTickLabelRotation',45);
        %     xlabel('Difficulty Level'); set(gca,'XTick',[4 8]);
        
        legend(GroupNames([1,2]))
        sgtitle(['Performance Group:' prf])
        
        subplot(1,2,2)
        hold on
        errorbar(top_bot,squeeze(mean(MeanScore(diff,3,:,(pg==prf)),4)),squeeze(std(MeanScore(diff,3,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr: only top-bottom
        errorbar(top_bot,squeeze(mean(MeanScore(diff,4,:,(pg==prf)),4)),squeeze(std(MeanScore(diff,4,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr: only top-bottom
        %         errorbar(DiffLvl,mean(MeanScore(:,3,pos,:),4),std(MeanScore(:,3,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        %         errorbar(DiffLvl,mean(MeanScore(:,4,pos,:),4),std(MeanScore(:,4,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        %         errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,3,pos,:),4),std(InverseEfficiencyScore(:,3,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        %         errorbar(DiffLvl,nanmean(InverseEfficiencyScore(:,4,pos,:),4),std(InverseEfficiencyScore(:,4,pos,:),[],4)./sqrt(nSubj),'LineWidth',3)
        grid on
        ylabel('TPR'); ylim([.2 1.1]);
        %     ylabel('IE');  ylim([.2 2]);
        %         set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
        set(gca,'XTick',[1 2 3 4],'XTickLabel',{'Bottom','Top'},'XTickLabelRotation',45);
        %     xlabel('Difficulty Level'); set(gca,'XTick',[4 8]);
        
        legend(GroupNames([3,4]))
        sgtitle(['Performance Group:' num2str(prf)])
    end
end

%% errorbars in all quadrants-foreperiod and predictability variable
%     figure();

for Dif=2
    %     for prf=1:2
    % for pred=1:2
    figure();
    
    subplot(1,2,1)
    hold on
    %     errorbar([1 2 3 4],squeeze(mean(mean(MeanScore(Dif,:,:,:),4))),squeeze(mean(std(MeanScore(Dif,:,:,:),[],4))./sqrt(nSubj)),'LineWidth',3) %Averaging all conditions only to show differences in visual quadrants
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,1,[1,2],:),4)),squeeze(std(MeanScore(Dif,1,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,2,[1,2],:),4)),squeeze(std(MeanScore(Dif,2,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,1,[1,2],:),4)),squeeze(std(MeanScore(Dif,1,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,2,[1,2],:),4)),squeeze(std(MeanScore(Dif,2,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    %     errorbar([1 2 3 4],squeeze(mean(MeanScore(Dif,1,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,1,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar([1 2 3 4],squeeze(mean(MeanScore(Dif,2,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,2,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar(1:8,squeeze(mean(MeanScore(1,1,:,:),4)),squeeze(std(MeanScore(1,1,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) % pos_rad & perfGr
    %     errorbar(1:8,squeeze(mean(MeanScore(2,1,:,:),4)),squeeze(std(MeanScore(2,1,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) % pos_rad & perfGr
    %     errorbar([1 2 3 4],squeeze(nanmean(MeanRTCorrect(Dif,1,:,:),4)),squeeze(std(MeanRTCorrect(Dif,1,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(MeanRTCorrect(Dif,2,:,:),4)),squeeze(std(MeanRTCorrect(Dif,2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,1,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,1,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,2,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(median(NrmlzMeanScore(:,pred,:,:),4)),squeeze(std(NrmlzMeanScore(:,pred,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) %normalized scores
        errorbar([1 2 3 4],squeeze(mean(MeanScore(:,1,1,:),4)),squeeze(std(MeanScore(:,1,1,:),[],4)./sqrt(nSubj)),'LineWidth',3) %diff=8 predictablity, foreperiod + pos SHORT
        errorbar([1 2 3 4],squeeze(mean(MeanScore(:,2,1,:),4)),squeeze(std(MeanScore(:,2,1,:),[],4)./sqrt(nSubj)),'LineWidth',3) %diff=8 predictablity, foreperiod + pos SHORT
    
    grid on
        ylabel('TPR');
%     ylabel('RT');
    %     ylabel('IES');
    %     ylabel('TPR difficult/easy');
    ylim([0.3 1]);
    set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
    %     set(gca,'XTick',[1 2],'XTickLabel',{'Bottom','Top'},'XTickLabelRotation',45);
    legend(GroupNames([1,2]))
    % legend('Predictable','Unpredictable')
    % title(['Difficulty Level : ',num2str(DiffLvl(Dif)) ' perf grp: ' num2str(prf)])
    % title('Normalized data- Short foreperiod');
    
    subplot(1,2,2)
    hold on
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,1,[3,4],:),4)),squeeze(std(MeanScore(Dif,1,[3,4],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,2,[3,4],:),4)),squeeze(std(MeanScore(Dif,2,[3,4],:),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr-in two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,3,[1,2],:),4)),squeeze(std(MeanScore(Dif,3,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    %     errorbar([1 2],squeeze(median(MeanScore(Dif,4,[1,2],:),4)),squeeze(std(MeanScore(Dif,4,[1,2],:),[],4)./sqrt(nSubj)),'LineWidth',3) %only two positions
    %     errorbar([1 2 3 4],squeeze(mean(MeanScore(Dif,3,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,3,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar([1 2 3 4],squeeze(mean(MeanScore(Dif,4,:,(pg==prf)),4)),squeeze(std(MeanScore(Dif,4,:,(pg==prf)),[],4)./sqrt(nSubj)),'LineWidth',3) %adding perfGr
    %     errorbar(1:8,squeeze(mean(MeanScore(1,2,:,:),4)),squeeze(std(MeanScore(1,2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) % pos_rad & perfGr
    %     errorbar(1:8,squeeze(mean(MeanScore(2,2,:,:),4)),squeeze(std(MeanScore(2,2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) % pos_rad & perfGr
%     errorbar([1 2 3 4],squeeze(nanmean(MeanRTCorrect(Dif,3,:,:),4)),squeeze(std(MeanRTCorrect(Dif,3,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
%     errorbar([1 2 3 4],squeeze(nanmean(MeanRTCorrect(Dif,4,:,:),4)),squeeze(std(MeanRTCorrect(Dif,4,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,3,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,3,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(nanmean(InverseEfficiencyScore(Dif,4,:,:),4)),squeeze(nanstd(InverseEfficiencyScore(Dif,4,:,:),[],4)./sqrt(nSubj)),'LineWidth',3)
    %     errorbar([1 2 3 4],squeeze(median(NrmlzMeanScore(:,pred+2,:,:),4)),squeeze(std(NrmlzMeanScore(:,pred+2,:,:),[],4)./sqrt(nSubj)),'LineWidth',3) %normalized scores
        errorbar([1 2 3 4],squeeze(mean(MeanScore(:,1,2,:),4)),squeeze(std(MeanScore(:,1,2,:),[],4)./sqrt(nSubj)),'LineWidth',3) %diff=8 predictablity, foreperiod + pos LONG
        errorbar([1 2 3 4],squeeze(mean(MeanScore(:,2,2,:),4)),squeeze(std(MeanScore(:,2,2,:),[],4)./sqrt(nSubj)),'LineWidth',3) %diff=8 predictablity, foreperiod + pos LONG
    
    grid on
        ylabel('TPR');
%     ylabel('RT');
    %     ylabel('IES');
    %     ylabel('TPR difficult/easy');
    ylim([0.3 1]);
    set(gca,'XTick',[1 2 3 4],'XTickLabel',posGroups,'XTickLabelRotation',45);
    %         set(gca,'XTick',[1 2],'XTickLabel',{'Bottom','Top'},'XTickLabelRotation',45); %for two positions
    legend(GroupNames([3,4]))
    % title(['Difficulty Level : ',num2str(DiffLvl(Dif)) ' perf grp: ' num2str(prf)])
    % title('Normalized data- Long foreperiod');
    % sgtitle('predictability vs. quadrant in difficulty = 8')
    % end
    %     end
end

%% plot positions for difficulty = 8 in pred and unpred
figure()
subplot(1,2,1)
hold on
plot(1:8,squeeze(mean(MeanScore(:,1,:,:),4)),'LineWidth',3,'Color','b','Marker','.','MarkerSize',40) % short-pred
plot(1:8,squeeze(mean(MeanScore(:,2,:,:),4)),'LineWidth',3,'Color','r','Marker','.','MarkerSize',40) % short-unpred
legend(GroupNames([1,2]))
ylabel('TPR'); xlabel('Position')
ylim([.2 1])
grid on
subplot(1,2,2)
hold on
plot(1:8,squeeze(mean(MeanScore(1,2,:,:),4)),'LineWidth',3,'Color','b','Marker','.','MarkerSize',40) % long-pred
plot(1:8,squeeze(mean(MeanScore(2,2,:,:),4)),'LineWidth',3,'Color','r','Marker','.','MarkerSize',40) % long-unpred
legend(GroupNames([3,4]))
ylabel('TPR'); xlabel('Position')
ylim([.2 1])
grid on
sgtitle('predictability vs. position in difficulty = 8')

%% tempatt effect on IES
% PosTempEff = squeeze(PosTempEff); PosTempEff = PosTempEff'; %rows->nSubj, columns->positions 

figure(); hold on
errorbar([2,4,6,8], mean(PosTempEff), std(PosTempEff)./sqrt(nSubj),'k.','MarkerSize',10)
yline(0, 'r--', 'LineWidth', 4);
set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlim([1 9])
xlabel('Position'); 
ylabel('Temporal attention effect');

%Violin plot
figure()
vs = violinplot(PosTempEff);
ylabel('Temporal attention enhancement');
set(gca,'XTick',1:4, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
xlim([.5, 4.5]);

%% Dot plot --  conditions separate, difficulties separate

colmp=colormap('jet');
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

%TPR vs. dificulty for both pred and unpred in quads- individuals

for pos=1:4
    figure;
    for nSub=1:nSubj+1
        subplot(1,2,1)
        hold on
        if nSub==nSubj+1
            plot([1 3],mean(MeanScore([1 2],1,pos,:),4),'LineWidth',3,'Color','b','Marker','o','MarkerFaceColor','b','MarkerSize',7);
            plot([1 3],mean(MeanScore([1 2],2,pos,:),4),'LineWidth',3,'LineStyle','--','Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
        else
            plot([1 3],MeanScore([1 2],1,pos,nSub),'LineWidth',1,'Color',colmp(nSub*20,:),'Marker','o','MarkerFaceColor',colmp(nSub*20,:),'MarkerSize',5);
            plot([1 3],MeanScore([1 2],2,pos,nSub),'LineStyle','--','LineWidth',1,'Color',colmp(nSub*20,:),'Marker','o','MarkerFaceColor',colmp(nSub*20,:),'MarkerSize',5);
            ylabel('TPR');
            set(gca,'XTick',[1 3],'XTickLabel',{'4','8'},'XTickLabelRotation',45); xlabel('Difficulty');
            title(posGroups{pos})
            xlim([0,4]);
            ylim([0 1.1]); grid on

        end
        
        subplot(1,2,2)
        hold on
        if nSub==nSubj+1
            plot([1 3],mean(MeanScore([1 2],3,pos,:),4),'LineWidth',3,'Color','b','Marker','o','MarkerFaceColor','b','MarkerSize',7);
            plot([1 3],mean(MeanScore([1 2],4,pos,:),4),'LineWidth',3,'LineStyle','--','Color','r','Marker','o','MarkerFaceColor','r','MarkerSize',7);
        else
            plot([1 3],MeanScore([1 2],3,pos,nSub),'LineWidth',1,'Color',colmp(nSub*20,:),'Marker','o','MarkerFaceColor',colmp(nSub*20,:),'MarkerSize',5);
            plot([1 3],MeanScore([1 2],4,pos,nSub),'LineStyle','--','LineWidth',1,'Color',colmp(nSub*20,:),'Marker','o','MarkerFaceColor',colmp(nSub*20,:),'MarkerSize',5);
            
            ylabel('TPR');
            set(gca,'XTick',[1 3],'XTickLabel',{'4','8'},'XTickLabelRotation',45); xlabel('Difficulty');
            title(posGroups{pos})
            xlim([0,4]);
            ylim([0 1.1]); grid on
        end
    end
    %     if  nSub==nSubj+1; SubNames(nSub)={'Average'}; else; SubNames(nSub)={['subject-' num2str(nSub)]}; end
    %     legend(SubNames)
    
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

%%%%%%%%%%%%%%%%%%%%%%% When not enough trials %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    if dif==length(DiffLvl); legend(GroupNames{1:2}); end
    % {'Predictable','Unpredictable'}
end

%% position -- all conditions collapsed
clear
load AllDataQuad.mat
nSubj=numel(AllData);
pg=zeros(1,nSubj); %vector of subject's performance group
PosTP = nan(4,2,nSubj); % rows: 'Bottom-Left','Bottom-Right','Top-Left','Top-Right' columns: 'predictable','unpredictable' OR difficulty, 3rd dim: subjects
stdPosTP = nan(4,2,nSubj);
PosRT = nan(4,2,nSubj);
stdPosRT = nan(4,2,nSubj);
PosIES = nan(4,2,nSubj);
PosConf = nan(4,2,nSubj);
memDecIdx = nan(4,nSubj);

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
memDecIdx_Quad = [flip(memDecIdx([1,2],:));memDecIdx([3,4],:)]'; % if you want to put rows in order of quadrants (4,3,2,1) for only difficulty
% memDecIdx = [flip(memDecIdx([1,2],:,:));memDecIdx([3,4],:,:)]; %put rows in order of quadrants (4,3,2,1) for diff and predictability

% PosGroups = M.Row;
GroupNames ={'Short-Predictable', 'Short-Unpredictable'};

%% memory decay in quad -- correlation ->not precise
figure()
scatter(repmat([4,3,2,1],1,11),memDecIdx(:)','filled')
hl = lsline;
[rho,pVal]=corrcoef(repmat([4,3,2,1],1,11),memDecIdx(:)'); %not precise for nominal variables (quadrants)
%Calculate slope
% [p,s] = polyfit(get(hl,'xdata'),get(hl,'ydata'),1);
% slp = p(1,1);

xlim([0 5]); xticks([1,2,3,4]); xticklabels({'Top-Right','Top-Left','Bottom-Right','Bottom-Left'}); xtickangle(45);
ylim([-.1,.7]); ylabel('Performance Decay Index (MDI)'); 
set(hl,'LineWidth', 2)
% title('Correlation between visual quadrant and memory decay from easy to difficult')
rho_txt = sprintf('Rho = %.3f',rho(2,1));
text(.5,.65,rho_txt)
p_txt = sprintf('P-value = %.3f',pVal(2,1));
text(.5,.6,p_txt)
grid on

%% memory decay in quad -- ANOVA

%Violin plot
figure('Units','pixels')
violin_PDI = violinplot(flip(memDecIdx_Quad,2));
set(gca,'XTick',[1,2,3,4], 'XTickLabel',{'Top-Right','Top-Left','Bottom-Left','Bottom-Right'},'XTick',1:4,'FontSize',20); 
xlim([0 5]); xtickangle(45);
ylim([-.1,.7]); ylabel('Performance Decay Index (PDI)','FontSize',20); 
% legend({' ',' ',' ','Mean',' ','Median'})

[p,tbl,stats] = anova1(flip(memDecIdx_Quad,2)); %run one way ANOVA with  plot
figure
[c,m,h,nms] = multcompare(stats);   %multiple comparison of variables

%% memory decay in quad- predictability
% pred_memDec_1 = squeeze(memDecIdx(:,1,pg==1)); pred_memDec_2 = squeeze(memDecIdx(:,1,pg==2)); %for perf groups
% unpred_memDec_1 = squeeze(memDecIdx(:,2,pg==1)); unpred_memDec_2 = squeeze(memDecIdx(:,2,pg==2));

pred_memDec = squeeze(memDecIdx(:,1,:));
unpred_memDec = squeeze(memDecIdx(:,2,:));

figure()
hold on
h1=scatter(repmat([4,3,2,1],1,11),pred_memDec(:)','filled','MarkerFaceColor','b');
[rho_p,pVal_p]=corrcoef(repmat([4,3,2,1],1,11),pred_memDec(:)');
h2=scatter(repmat([4,3,2,1],1,11),unpred_memDec(:)','filled','MarkerFaceColor','r');
l = lsline; l(1).Color=[0 0 1]; l(2).Color=[1 0 0];
set(l,'LineWidth', 2)
[rho_u,pVal_u]=corrcoef(repmat([4,3,2,1],1,11),unpred_memDec(:)');


rho_txt = sprintf('Rho(pred) = %.3f',rho_p(2,1));
text(.5,.65,rho_txt)
p_txt = sprintf('P-value(pred) = %.3f',pVal_p(2,1));
text(.5,.6,p_txt)
rho_txt = sprintf('Rho(unpred) = %.3f',rho_u(2,1));
text(.5,.55,rho_txt)
p_txt = sprintf('P-value(unpred) = %.3f',pVal_u(2,1));
text(.5,.5,p_txt)

xlim([0 5]); xticks([1,2,3,4]); xticklabels({'Top-Right','Top-Left','Bottom-Left','Bottom-Right'}); xtickangle(45);
ylim([-.2,.7]); ylabel('Memory Decay Index (4-8/4+8)'); 
title('Correlation between visual quadrant and memory decay from easy to difficult')
legend(GroupNames)
grid on

%% Pos performance
figure()
hold on

TPmeanMat = mean(mean(PosTP,3),2); %collaps over predictabilities as well
TPstdMat = mean(std(PosTP,[],3),2);
bar([2,4,6,8],TPmeanMat)

errorbar([2,4,6,8], TPmeanMat(:,1), TPstdMat(:,1)/ sqrt(nSubj),'k.');

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

errorbar([2,4,6,8]-.25, IESmeanMat(:,1), IESstdMat(:,1)/ sqrt(nSubj),'k.');
errorbar([2,4,6,8]+.25, IESmeanMat(:,2), IESstdMat(:,2)/ sqrt(nSubj),'k.');

set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});


xlabel('Position');
ylabel('IES');

legend(GroupNames)

%% Quad performance - performance groups
for prf = 1:2
TPmeanMat(:,prf) = mean(mean(PosTP(:,:,pg==prf),3),2); %collaps over predictabilities as well
TPstdMat(:,prf) = mean(std(PosTP(:,:,pg==prf),[],3),2);
end

figure()
hold on

bar(repmat([2,4,6,8]',1,2),TPmeanMat)

errorbar([2,4,6,8]-.25, TPmeanMat(:,1), TPstdMat(:,1)/ sqrt(nSubj),'k.');
errorbar([2,4,6,8]+.25, TPmeanMat(:,2), TPstdMat(:,2)/ sqrt(nSubj),'k.');

set(gca,'XTick',2:2:8, 'XTickLabel',{'Bottom-Left','Bottom-Right','Top-Left','Top-Right'});
legend('Perf = 1','Perf = 2')

xlabel('Position');
ylabel('TPR');
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


GroupNames ={'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};
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

