%% calculate performance for each subject in each diff level in each position - 4Diff
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMS4DiffLevel')
save_folder='/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse';
load PooledData4Diff.mat   % 1_ filename

cd(save_folder)
DiffLvl = [4,6,8,10];

for diff = 1:length(DiffLvl)
    clearvars -except PooledData nSubj save_folder DiffLvl diff
    
    GroupNames = {'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};
    %pos = {'pos_1','pos_2','pos_3','pos_4','pos_5','pos_6','pos_7','pos_8','pos_9','pos_10'}; % Row names
    
    Data=PooledData;
    Data(Data.correct_response ==2,:)=[];
    Data(Data.difficulty ~= DiffLvl(diff),:)=[];
    
    MM = grpstats(Data, {'foreperiod','predictability','question'}, {@nanmean,'std'}); %all conditions separated
    MeanScore=reshape(MM.nanmean_score,DiffLvl(diff),[]);
    MeanRTCorrect=reshape(MM.nanmean_RTCorrect,DiffLvl(diff),[]);
    MeanRT=reshape(MM.nanmean_reaction_time,DiffLvl(diff),[]);
    IES=MeanRT./MeanScore;
        
    writetable(array2table(MeanScore,'VariableNames',GroupNames),['1_mean_allSubs_TPR_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanRTCorrect,'VariableNames',GroupNames),['1_mean_allSubs_RT_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(IES,'VariableNames',GroupNames),['1_mean_allSubs_IES_diff_' num2str(DiffLvl(diff)) '.csv'])
    
end

%% calculate performance for each subject in each diff level in each position - Quad
cd('/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse/DMSQuadrants')
save_folder='/Users/Tara/Documents/MATLAB/MATLAB-Programs/PhD-Thesis-Programs/DMS-Project/Results/Analyse';
load AllDataQuadrants.mat  % 2_ filename

nSubj=numel(AllData);
cd(save_folder)

DiffLvl = [4,8];

for diff = 1:length(DiffLvl)
    clearvars -except AllData nSubj save_folder DiffLvl diff
    
    GroupNames = {'Predictable-Short', 'Unpredictable-Short','Predictable-Long', 'Unpredictable-Long'};
    %pos = {'pos_1','pos_2','pos_3','pos_4','pos_5','pos_6','pos_7','pos_8','pos_9','pos_10'}; % Row names
    cols = {'sub_1','sub_2','sub_3','sub_4','sub_5','sub_6','sub_7','sub_8','sub_9','sub_10','sub_11'};
    
    for i=1:nSubj
        Data=AllData{i};
        Data(Data.correct_response ==2,:)=[];
        Data(Data.difficulty ~= DiffLvl(diff),:)=[];
        
        MM = grpstats(Data, {'foreperiod','predictability','question'}, {@nanmean,'std'}); %all conditions separated
        MeanScore(:,i,:)=reshape(MM.nanmean_score,DiffLvl(diff),[]);
        MeanRTCorrect(:,i,:)=reshape(MM.nanmean_RTCorrect,DiffLvl(diff),[]);
        MeanRT(:,i,:)=reshape(MM.nanmean_reaction_time,DiffLvl(diff),[]);
        IES(:,i,:)=MeanRT(:,i,:)./MeanScore(:,i,:);
    end
  
    writetable(array2table(MeanScore(:,:,1),'VariableNames',cols),['2_TPR_short_pred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanScore(:,:,2),'VariableNames',cols),['2_TPR_short_unpred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanScore(:,:,3),'VariableNames',cols),['2_TPR_long_pred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanScore(:,:,4),'VariableNames',cols),['2_TPR_long_unpred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanRTCorrect(:,:,1),'VariableNames',cols),['2_RT_short_pred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanRTCorrect(:,:,2),'VariableNames',cols),['2_RT_short_unpred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanRTCorrect(:,:,3),'VariableNames',cols),['2_RT_long_pred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanRTCorrect(:,:,4),'VariableNames',cols),['2_RT_long_unpred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(IES(:,:,1),'VariableNames',cols),['2_IES_short_pred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(IES(:,:,2),'VariableNames',cols),['2_IES_short_unpred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(IES(:,:,3),'VariableNames',cols),['2_IES_long_pred_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(IES(:,:,4),'VariableNames',cols),['2_IES_long_unpred_diff_' num2str(DiffLvl(diff)) '.csv'])
    
    MeanScoreMean = squeeze(nanmean(MeanScore,2));
    MeanRTCorrectMean = squeeze(nanmean(MeanRTCorrect,2));
    IESMean = squeeze(nanmean(IES,2));
    
    writetable(array2table(MeanScoreMean,'VariableNames',GroupNames),['2_mean_allSubs_TPR_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(MeanRTCorrectMean,'VariableNames',GroupNames),['2_mean_allSubs_RT_diff_' num2str(DiffLvl(diff)) '.csv'])
    writetable(array2table(IESMean,'VariableNames',GroupNames),['2_mean_allSubs_IES_diff_' num2str(DiffLvl(diff)) '.csv'])
    
end
