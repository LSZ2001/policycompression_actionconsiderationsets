clear all; close all;
% base_folder = 'C:\Users\liu_s\policycompression_actionconsiderationsets';
% cd(base_folder)
% addpath(genpath(base_folder))

% Figure and font default setting
set(0,'units','inches');
Inch_SS = get(0,'screensize');
set(0,'units','pixels');
figsize = get(0, 'ScreenSize');
Res = figsize(3)./Inch_SS(3);
set(groot, 'DefaultAxesTickDir', 'out', 'DefaultAxesTickDirMode', 'manual');
fontsize=12;
set(groot,'DefaultAxesFontName','Arial','DefaultAxesFontSize',fontsize);
set(groot,'DefaultLegendFontSize',fontsize-2,'DefaultLegendFontSizeMode','manual')

% paths
figpath = "figures\"; %"newplots\"
datapath = "data\";
save_path = "saved_files\";
png_dpi = 500;

% Color palettes
cmap = brewermap(3, 'Set1');
cmap = cmap([1,3,2],:);
cmap_subj = brewermap(200, 'Set1');
cmap_exp3 = brewermap(9, 'Set2');
cmap_exp3 = cmap_exp3([2,1,3,4,5,6,9],:);
cmap_gray = flipud(gray(256));
cmap2 = brewermap(7, 'Spectral');
load(datapath+"data_exp1.mat");

datas.train = remove_nonresponsive_trials(datas.train);
datas.test = remove_nonresponsive_trials(datas.test);
data_exps.exp1 = datas.test;
data_exps_train.exp1 = datas.train;

% genders = [];
% for subj=1:75
%    genders = [genders, datas.survey(subj).gender];
% end


%% Parse data
% % Needed for the first time to save the data files
% experiment = "exp1";
% hutter_alpha = 0.01;
% exps_cutoffs.exp1 = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, true, 1, -Inf);
% exps_cutoffs.exp1_rewardcutoff = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, true, 1, 0.15);
% exps_cutoffs.exp1_thres2 = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, true, 2, -Inf);
% exps_cutoffs.exp1_thres2_rewardcutoff = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, true, 2, 0.15);
% exps_cutoffs.exp1_thres3 = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, true, 3, -Inf);
% exps_cutoffs.exp1_thres3_rewardcutoff = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, true, 3, 0.15);
% save(save_path+"exps_cutoffs.mat", "exps_cutoffs")

load(save_path+"exps_cutoffs.mat")
exp1 = exps_cutoffs.exp1;
exp1_rewardcutoff = exps_cutoffs.exp1_rewardcutoff;

%% Figure 2: Example suboptimality analysis
Figure2(cmap2, [10,10,1200,400])
saveas(gca, figpath+'Fig2.fig')
exportgraphics(gcf,figpath+'Fig2.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig2.pdf',"ContentType","vector");

%% Figure 3: Simulation example.
% fit_simulations()  % Fit simulations and same the resultant .mat files. Takes very long.
visualize_simulation_wrapper(cmap_exp3, [1,1,1280,600], "exp0_randomQ", "difference");
saveas(gca, figpath+'Fig3.fig')
exportgraphics(gcf,figpath+'Fig3.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig3.pdf',"ContentType","vector");

%% Figure 4 Partial: Task setup
Figure4Partial1(exp1, cmap2, [10,10,1200,400]);
saveas(gca, figpath+'Fig4_partial1.fig')
exportgraphics(gcf,figpath+'Fig4_partial1.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig4_partial1.pdf',"ContentType","vector");

Figure4Partial2(exp1,[50,50,700*2/3,600*2/3])
saveas(gca, figpath+'Fig4_partial2.fig')
exportgraphics(gcf,figpath+'Fig4_partial2.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig4_partial2.pdf',"ContentType","vector");

%% Figure 5: Human behavioral main results.
Figure5(exp1, cmap, cmap2, [0,0,1280,600])
saveas(gca, figpath+'Fig5.fig')
exportgraphics(gcf,figpath+'Fig5.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig5.pdf',"ContentType","vector");


%% Figure S1-S5: More simulations.
task_names = ["exp0_randomQ","exp0_sparseQ","exp0_sparseQ","exp1_Ns6","exp1_Ns6"];
plot_types = ["original","original","difference","original","difference"];
for fig_idx=1:length(task_names)
    visualize_simulation_wrapper(cmap_exp3, [1,1,1280,600], task_names(fig_idx), plot_types(fig_idx));
    saveas(gca, figpath+"FigS"+fig_idx+".fig")
    exportgraphics(gcf,figpath+"FigS"+fig_idx+".png",'Resolution',png_dpi);
    exportgraphics(gcf,figpath+"FigS"+fig_idx+".pdf","ContentType","vector");
end

%% Figure S6: I and Na versus other behavioral variables.
FigureS6(exp1_rewardcutoff, cmap, [50,50,1000,600])
saveas(gca, figpath+'FigS6.fig')
exportgraphics(gcf,figpath+'FigS6.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS6.pdf',"ContentType","vector");

%% Figure S7: I(S;A) distribution for each Na
ks_test_results = FigureS7(exp1, [50,50,900,600]);
saveas(gca, figpath+'FigS7.fig')
exportgraphics(gcf,figpath+'FigS7.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS7.pdf',"ContentType","vector");

%% Figure S8: P(a) for each participant
FigureS8(exp1, cmap, [50,50,1000,600])
saveas(gca, figpath+'FigS8.fig')
exportgraphics(gcf,figpath+'FigS8.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS8.pdf',"ContentType","vector");

%% Figure S9: Noisy Q values simulations
% noisyQ_simulations(exp1) % Save the simulation files. Takes some time.
FigureS9(cmap2, [1,1,1200,600])
saveas(gca, figpath+'FigS9.fig')
exportgraphics(gcf,figpath+'FigS9.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS9.pdf',"ContentType","vector");


%% Helper functions
function [parsed_experiment_data] = parse_data(data_exps, data_exps_train, experiment, hutter_alpha, fit_lme, count_action_thres, reward_cutoff)
    if(nargin==3)
        hutter_alpha=0.01; fit_lme = true; count_action_thres = 1; reward_cutoff = -Inf;
    elseif(nargin==4)
        fit_lme = true; count_action_thres=1; reward_cutoff=-Inf;
    elseif(nargin==5)
         count_action_thres=1; reward_cutoff=-Inf;
    elseif(nargin==6)
         reward_cutoff=-Inf;
    end


    % Test data;
    data = data_exps.(experiment);
    n_subj = length(data);
    switch experiment
        case "exp1"
            n_states = 6;
            n_actions = n_states+1;
            p_state = ones(1,n_states)./n_states;
            Q = eye(n_states);
            q = 0.2.*ones(n_states,1);
            Q = normalize(Q, "range", [-0.18,1]);
            Q = [q,Q];
            conds = unique(data(1).cond); % ITI conditions
    end

    % Train data--and accuracy.
    data_train = data_exps_train.(experiment);
    for s = 1:n_subj
        for block = 1:4
            idx = data_train(s).block==block;
            acc = data_train(s).acc(idx);
            acc(acc<1)=0; % Safety action counts as inaccurate
            r = data_train(s).r(idx);
            accuracy_bycond_train(s,block) = nanmean(acc);
        end
    end
    BehavioralStats.train_accuracy_bycond = accuracy_bycond_train;


    % Back to test data
    for s = 1:n_subj
        for c = 1:length(conds)
            idx = data(s).cond == conds(c);
            state = data(s).s(idx);
            action = data(s).a(idx);
            acc = data(s).acc(idx);
            r = data(s).r(idx);
            rt = data(s).rt(idx);
            tt = data(s).tt(idx); % total time of the block
            n_trials_bycond(s,c) = length(state);
            accuracy_bycond(s,c) = nanmean(acc);
            reward_bycond(s,c) = nanmean(r);
            reward_count(s,c) = sum(r);
            complexity_bycond(s,c) = mutual_information(round(state),round(action),hutter_alpha)./log(2);
            rt_bycond(s,c) = nanmean(rt./1000); % RT in seconds
        end
    end

    BehavioralStats.rt_deadline_conds = conds'./1000;
    BehavioralStats.n_trials_bycond = n_trials_bycond;
    BehavioralStats.accuracy_bycond=accuracy_bycond;
    BehavioralStats.reward_bycond=reward_bycond;
    BehavioralStats.complexity_bycond=complexity_bycond;
    BehavioralStats.rt_bycond=rt_bycond; 
   
    
    % Theoretical curves assuming linear RTH
    n_tot = 300;
    beta_set = linspace(0.1,15,n_tot);
    optimal_sol.beta_set = beta_set;
    optimal_sol.Q = Q;
    optimal_sol.p_state = p_state;
    [optimal_sol.R, optimal_sol.V, optimal_sol.Pa, optimal_sol.optimal_policy] = blahut_arimoto(p_state,Q,beta_set);
    R = optimal_sol.R; V=optimal_sol.V;
    V_interp_bycond = interp1(R,V,complexity_bycond,'linear');
    V_interp_bycond(isnan(V_interp_bycond)) = interp1(R,V,complexity_bycond(isnan(V_interp_bycond)),'pchip'); % This just deals with V when I=0.
    BehavioralStats.V_interp_bycond = V_interp_bycond;
    reward_diff_bycond = reward_bycond-V_interp_bycond;
    BehavioralStats.reward_diff_bycond = reward_diff_bycond;

    % Optimal policies with certain subsets of actions missing
    n_tot = length(beta_set);
    Rs_partial = zeros(n_actions,2,n_tot);
    Vs_partial = zeros(n_actions,2,n_tot);
    R_grid = 0:0.01:log2(6);
    V_grid = interp1(R,V,R_grid,'linear');
    Rs_partial_grid = zeros(n_actions,2,length(R_grid));
    Vs_partial_grid = zeros(n_actions,2,length(R_grid));
    n_available_actions = 1:n_actions;
    for na = n_available_actions
        Q_temp1 = Q(:,1:na);
        [R_temp1, V_temp1, Pa_temp1, optimal_policy_temp1] = blahut_arimoto(p_state,Q_temp1,beta_set);
        Rs_partial(na, 1, :) = R_temp1;
        Vs_partial(na, 1, :) = V_temp1;
    
        Q_temp2 = Q(:,(end-na+1):(end));
        [R_temp2, V_temp2, Pa_temp2, optimal_policy_temp2] = blahut_arimoto(p_state,Q_temp2,beta_set);
        Rs_partial(na, 2, :) = R_temp2;
        Vs_partial(na, 2, :) = V_temp2;
    
        if(na==1) % When N_a=1, policy complexity must be 0 bits.
            Rs_partial_grid(na,:,:) = 0;
            Vs_partial_grid(na,1,:) = V_temp1(1);
            Vs_partial_grid(na,2,:) = V_temp2(1);
        else
            Rs_partial_grid(na,1,:) = R_grid;
            Vs_partial_grid(na,1,:) = interp1(R_temp1,V_temp1,R_grid,'linear');
            Rs_partial_grid(na,2,:) = R_grid;
            Vs_partial_grid(na,2,:)= interp1(R_temp2,V_temp2,R_grid,'linear');
        end
    end
    optimal_sol.R_grid = R_grid;
    optimal_sol.V_grid = V_grid;
    optimal_sol.Rs_partial_grid = Rs_partial_grid;
    optimal_sol.Vs_partial_grid = Vs_partial_grid;


    % P(A|S)
    P_a_given_s = zeros(n_subj,n_states,length(conds),n_actions); % subj, states, conds, actions
    count_a_given_s = zeros(n_subj,n_states,length(conds),n_actions); % subj, states, conds, actions
    for subj=1:n_subj
        for state=1:n_states
            s_idx = find(data(subj).s==state);
            subj_thisstateoccurrences = data(subj).s(s_idx);
            subj_actions_giventhisstate = data(subj).a(s_idx);
            for c = 1:length(conds)
                cond_idx = find(data(subj).cond(s_idx) == conds(c));
                states = subj_thisstateoccurrences(cond_idx);
                actions = subj_actions_giventhisstate(cond_idx);
                [N,~] = histcounts(actions,-0.5:1:(n_actions-1+0.5));
                count_a_given_s(subj, state, c, :) = N;
                P_a_given_s(subj, state, c, :) = N./sum(N);
            end
        end
    end
    BehavioralStats.P_a_given_s = P_a_given_s;
    BehavioralStats.count_a_given_s = count_a_given_s;
    P_a = squeeze(nanmean(nanmean(P_a_given_s,3),2)); %[subject, actions]; the inner mean averages over ITI conditions.
    P_a_bycond = squeeze(nanmean(P_a_given_s,2));
    [P_a_bycond_ranked, actions_bycond_ranked] = sort(P_a_bycond,3,'descend');
    [P_a_ranked, actions_ranked] = sort(P_a,2,'descend');
    % n_actions_chosen_bycond = zeros(n_subj,length(conds));
    % for c=1:length(conds)
    %     n_actions_chosen_bycond(:,c) = sum(squeeze(P_a_bycond(:,c,:))>0,2);
    % end
    BehavioralStats.P_a = P_a;
    BehavioralStats.P_a_bycond = P_a_bycond;
    BehavioralStats.P_asafe_bycond = P_a_bycond(:,:,1);
    BehavioralStats.P_a_ranked = P_a_ranked;
    BehavioralStats.P_a_bycond_ranked = P_a_bycond_ranked;

    % Compute number of actions chosen based on the action-counting threshold
    count_a_bycond = squeeze(sum(count_a_given_s,2));
    n_actions_chosen_bycond = sum(count_a_bycond>=count_action_thres, 3);
    BehavioraStats.count_action_thres = count_action_thres;
    BehavioralStats.count_a_bycond = count_a_bycond;
    BehavioralStats.n_actions_chosen_bycond = n_actions_chosen_bycond;


    % Exclude subjects with trial-averaged reward below threshold.
    included_subjs = find(min(reward_bycond,[],2) > reward_cutoff);
    excluded_subjs = find(min(reward_bycond,[],2) <= reward_cutoff);
    Analyses.included_subjs = included_subjs;
    Analyses.excluded_subjs = excluded_subjs;

    % Do LMEs on only the nonrand participant group.
    subjs_bycond = repmat(included_subjs, 1,3);
    rtdeadlines_bycond = repmat(conds'./1000, length(included_subjs),1);
    P_asafe_bycond_flat = P_a_bycond(included_subjs,:,1);
    complexity_bycond_flat = complexity_bycond(included_subjs,:);
    rt_bycond_flat = rt_bycond(included_subjs,:);
    reward_bycond_flat = reward_bycond(included_subjs,:);
    reward_diff_bycond_flat = reward_diff_bycond(included_subjs,:);
    n_actions_chosen_bycond_flat = n_actions_chosen_bycond(included_subjs,:);
    tbl = table(subjs_bycond(:), rt_bycond_flat(:), rtdeadlines_bycond(:), complexity_bycond_flat(:), reward_bycond_flat(:), reward_diff_bycond_flat(:), n_actions_chosen_bycond_flat(:), P_asafe_bycond_flat(:), 'VariableNames',{'Subject','RT','RTDeadline', 'Complexity', 'Reward', 'ReductionReward', 'NActionsChosen','Pasafe'});
    Analyses.tbl = tbl;

    lme_eqs_tbl = [];
    lme_fixedeffectname_tbl = [];
    lme_fixedeffects_coeff_tbl = [];
    lme_fixedeffects_se_tbl = [];
    lme_pval_tbl = [];
    lme_randomeffects_sd_tbl = [];
    if(fit_lme)
        lme_equations = ["Complexity ~ RTDeadline+(RTDeadline|Subject)";
                          "Reward ~ RTDeadline+(RTDeadline|Subject)";
                          "Pasafe ~ RTDeadline+(RTDeadline|Subject)";
                          "RT ~ Complexity+(Complexity|Subject)";
                          "NActionsChosen ~ RTDeadline+(RTDeadline|Subject)";
                          "RT ~ NActionsChosen+(NActionsChosen|Subject)";
                          "Reward ~ Complexity*NActionsChosen+(Complexity*NActionsChosen|Subject)";
                          "ReductionReward ~ Complexity*NActionsChosen+(Complexity*NActionsChosen|Subject)";
                          "RT ~ Complexity*NActionsChosen+(Complexity*NActionsChosen|Subject)";
                          ];
        lmes = cell(length(lme_equations),1);
        for lme_idx =1:length(lme_equations)
            lmes{lme_idx}.lme_equation = lme_equations(lme_idx);
            lme = fitlme(tbl,lmes{lme_idx}.lme_equation);
            lmes{lme_idx}.lme = lme;
            [fixed_effects, fixed_effect_names, fixed_effect_stats] = fixedEffects(lme);
            [random_effects, random_effect_names, random_effect_stats] = randomEffects(lme);
            if(lme_idx<7) % LMEs with one predictor
                lme_eqs_tbl = [lme_eqs_tbl; lme_equations(lme_idx)];
                lme_fixedeffectname_tbl = [lme_fixedeffectname_tbl; convertCharsToStrings(fixed_effect_stats{2,1})];
                lme_fixedeffects_coeff_tbl = [lme_fixedeffects_coeff_tbl; fixed_effect_stats{2,2}];
                lme_fixedeffects_se_tbl = [lme_fixedeffects_se_tbl; fixed_effect_stats{2,3}];
                lme_pval_tbl = [lme_pval_tbl; fixed_effect_stats{2,6}];
                lme_randomeffects_sd_tbl = [lme_randomeffects_sd_tbl; std(random_effects(2:2:end))];
            elseif(lme_idx<9) %LMEs: interested in interaction term
                for predictor_idx = [2,4]
                    lme_eqs_tbl = [lme_eqs_tbl; lme_equations(lme_idx)];
                    lme_fixedeffectname_tbl = [lme_fixedeffectname_tbl; convertCharsToStrings(fixed_effect_stats{predictor_idx,1})];
                    lme_fixedeffects_coeff_tbl = [lme_fixedeffects_coeff_tbl; fixed_effect_stats{predictor_idx,2}];
                    lme_fixedeffects_se_tbl = [lme_fixedeffects_se_tbl; fixed_effect_stats{predictor_idx,3}];
                    lme_pval_tbl = [lme_pval_tbl; fixed_effect_stats{predictor_idx,6}];
                    lme_randomeffects_sd_tbl = [lme_randomeffects_sd_tbl; std(random_effects(predictor_idx:4:end))];
                end
            else %LMEs: not interested in interaction term
                for predictor_idx = [2,3]
                    lme_eqs_tbl = [lme_eqs_tbl; lme_equations(lme_idx)];
                    lme_fixedeffectname_tbl = [lme_fixedeffectname_tbl; convertCharsToStrings(fixed_effect_stats{predictor_idx,1})];
                    lme_fixedeffects_coeff_tbl = [lme_fixedeffects_coeff_tbl; fixed_effect_stats{predictor_idx,2}];
                    lme_fixedeffects_se_tbl = [lme_fixedeffects_se_tbl; fixed_effect_stats{predictor_idx,3}];
                    lme_pval_tbl = [lme_pval_tbl; fixed_effect_stats{predictor_idx,6}];
                    lme_randomeffects_sd_tbl = [lme_randomeffects_sd_tbl; std(random_effects(predictor_idx:4:end))];
                end
            end
            
        end
        Analyses.LME.lme_equations = lme_equations;
        Analyses.LME.lmes = lmes;
        Analyses.LME.lme_tbl = table(lme_eqs_tbl, lme_fixedeffectname_tbl, lme_fixedeffects_coeff_tbl,lme_fixedeffects_se_tbl, lme_pval_tbl, lme_randomeffects_sd_tbl, 'VariableNames',{'LME','Variable','FixedEffectCoeff', 'FixedEffectSE', 'FixedEffectPVal', 'RandomEffectSD'});
    end
    R0 = corrcoef([complexity_bycond_flat(:), n_actions_chosen_bycond_flat(:)]); % correlation matrix
    Analyses.LME.VIF=diag(inv(R0))';

    % Pearson correlations
    BehavioralStats.mean_complexity = mean(complexity_bycond,2);
    BehavioralStats.mean_n_actions_chosen = mean(n_actions_chosen_bycond,2);
    BehavioralStats.mean_train_accuracy = mean(accuracy_bycond_train,2);
    [a,b]=corr(complexity_bycond_flat(:), n_actions_chosen_bycond_flat(:));
    [c,d]=corr(BehavioralStats.mean_complexity(included_subjs), BehavioralStats.mean_train_accuracy(included_subjs));
    [e,f]=corr(BehavioralStats.mean_n_actions_chosen(included_subjs), BehavioralStats.mean_train_accuracy(included_subjs));
    if(isempty(excluded_subjs))
        Analyses.Correlations = [a,b;c,d;e,f];
    else
        [g,h]=corr(BehavioralStats.mean_complexity(excluded_subjs), BehavioralStats.mean_train_accuracy(excluded_subjs));
        [i,j]=corr(BehavioralStats.mean_n_actions_chosen(excluded_subjs), BehavioralStats.mean_train_accuracy(excluded_subjs));
        Analyses.Correlations = [a,b;c,d;e,f;g,h;i,j];
    end




    parsed_experiment_data.optimal_sol = optimal_sol;
    parsed_experiment_data.BehavioralStats = BehavioralStats;
    parsed_experiment_data.Analyses = Analyses;
    
    

end


%% Figure 2: Example suboptimality analysis
function [] = Figure2(cmap2, figsize)
    n_states = 6;
    n_actions = n_states; %n_states+1;
    p_state = ones(1,n_states)./n_states;
    Q = eye(n_states);
    Q = normalize(Q, "range", [0,1]);   
    p_s = zeros(n_states,1);
    p_s(:,1) = p_state;
    V_a = p_s' * Q; % The average Q value of an action: V(a) = \sum_s p(s)*Q(s,a)

    % Exact policy compression
    n_tot = 100;
    beta_set = linspace(0.1,15,n_tot);
    [R, V, Pa, optimal_policy] = blahut_arimoto(p_state,Q,beta_set);
    R_grid = 0:0.001:max(R);
    V_grid = interp1(R,V,R_grid,'pchip');
    
    % Optimal policies with certain subsets of actions missing
    alpha = 0.5; line_alpha = 0.05; ttl_fontsize = 12; ttl_position_xshift = -0.21; ttl_position_yshift = 0.99;
    
    Rs_partial = zeros(n_actions,2,n_tot);
    Vs_partial = zeros(n_actions,2,n_tot);
    n_available_actions = 1:n_actions;
    for na = n_available_actions
        Q_temp1 = Q(:,1:na);
        [R_temp1, V_temp1, Pa_temp1, optimal_policy_temp1] = blahut_arimoto(p_state,Q_temp1,beta_set);
        Rs_partial(na, 1, :) = R_temp1;
        Vs_partial(na, 1, :) = V_temp1;
    end
    
    figure("Position",figsize);
    t=tiledlayout(1,3, 'Padding', 'compact', 'TileSpacing', 'compact'); 
    nexttile(1); 
    heatmap(Q);
    ylabel("States")
    xlabel("Actions")
    colorbar off
    % Add a "title" using a text annotation
    annotation('textbox', [0.03 0.95 0.05 0.05], ...
        'String', 'A', ...
        'FontSize', ttl_fontsize, ...
        'FontWeight', 'bold', ...
        'LineStyle', 'none', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
    
    % ttl = title("A", "Fontsize", ttl_fontsize);
    % ttl.Units = 'Normalize'; 
    % ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    % ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    % ttl.HorizontalAlignment = 'left'; 
    % set(gca,'box','off')
    
    for na = n_available_actions
        nexttile(2); hold on;
        if(na==1)
            plot(R,V,"k-","LineWidth",2.5)
        end
        if(na>1)
            plot(squeeze(Rs_partial(na, 1, :)), squeeze(Vs_partial(na, 1, :)), "-","Color", cmap2(na,:), "LineWidth",2)
        else
            plot(squeeze(Rs_partial(na, 1, :)), squeeze(Vs_partial(na, 1, :)), ".","Color", cmap2(na,:), "MarkerSize",20)
        end
        xlim([0, log2(min(n_states, n_actions))])
        ylim([0,1])
        yticks([0:0.25:1])
        legend(["Theoretical","Na = "+n_available_actions],"location","southeast")
        xlabel("Policy complexity (bits)")
        ylabel("Trial-averaged reward")
        ttl = title("B", "Fontsize", ttl_fontsize);
        ttl.Units = 'Normalize'; 
        ttl.Position(1) = ttl_position_xshift+0.02; % use negative values (ie, -0.1) to move further left
        ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
        ttl.HorizontalAlignment = 'left'; 
        set(gca,'box','off')
    
    
        nexttile(3); hold on;
        plot([0,log2(min(n_states, n_actions))],[0,0],"k-","LineWidth",2.5,"HandleVisibility","off");
        if(na>1)
            v_corresponding_optimal = interp1(R,V,squeeze(Rs_partial(na, 1, :)),"pchip");
            plot(squeeze(Rs_partial(na, 1, :)), squeeze(Vs_partial(na, 1, :))-v_corresponding_optimal, "-","Color", cmap2(na,:), "LineWidth",2)
        else
            plot(squeeze(Rs_partial(na, 1, :)), squeeze(Vs_partial(na, 1, :))-squeeze(Vs_partial(na, 1, :)), ".","Color", cmap2(na,:), "MarkerSize",20)
        end
        xlim([0, log2(min(n_states, n_actions))])
        ylim([-0.2,0])
        yticks([-0.2:0.05:0])
        xlabel("Policy complexity (bits)")
        ylabel("Reduction in trial-averaged reward")
        ttl = title("C", "Fontsize", ttl_fontsize);
        ttl.Units = 'Normalize'; 
        ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
        ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
        ttl.HorizontalAlignment = 'left'; 
        set(gca,'box','off')
    end
end


%% Figure 3: Simulation results and plots
function [] = fit_simulations(save_path, task_names, use_IS_approxs, sample_with_replacements)
    if(nargin==0)
        save_path = "saved_files\";
        use_IS_approxs = [false,true,false];
        sample_with_replacements = [false,true,true];
        task_names = ["exp1_Ns6","exp0_sparseQ","exp0_randomQ"];
    elseif(nargin==1)
        use_IS_approxs = [false,true,false];
        sample_with_replacements = [false,true,true];
        task_names = ["exp1_Ns6","exp0_sparseQ","exp0_randomQ"];

    end
    for task_name = task_names
        for fit_type_idx = 1:length(use_IS_approxs)
            use_IS_approx = use_IS_approxs(fit_type_idx);
            sample_with_replacement = sample_with_replacements(fit_type_idx);
            [task_name, use_IS_approx, sample_with_replacement]
            approximate_policy_compression_wrapper(task_name, use_IS_approx, sample_with_replacement)
            
        end
    end
end

function [] = visualize_simulation_wrapper(cmap,figsize,task_names, plot_rules,save_path)
    if(nargin==1)
        save_path = "saved_files\";
        task_names = ["exp0_randomQ"];
        plot_rules = ["difference"];
        figsize = [1,1,1280,600];
    elseif(nargin==2)
        task_names = ["exp0_randomQ"];
        plot_rules = ["difference"];
        save_path = "saved_files\";
    elseif(nargin==4)
        save_path = "saved_files\";
    end

    for task_name = task_names
        switch task_name
            case "exp1_Ns6"
                sample_size_idxs_visualized_withoutreplacement = [1,2,3,4,6,7];
                sample_size_idxs_visualized_withreplacement = [1,3,4,6,7,8];
            case "exp0_randomQ"
                sample_size_idxs_visualized_withoutreplacement = [1,2,3,5,7,8];
                sample_size_idxs_visualized_withreplacement = [1,3,4,6,7,8];
            case "exp0_sparseQ"
                sample_size_idxs_visualized_withoutreplacement = [1,2,3,5,7,8];
                sample_size_idxs_visualized_withreplacement = [1,3,4,6,7,8];
        end

        for plot_rule = plot_rules
            visualize_simulations(save_path, plot_rule, task_name, sample_size_idxs_visualized_withoutreplacement, sample_size_idxs_visualized_withreplacement, cmap, figsize)
        end
    end
    
end


%% Figure 4: Task setup
function [] = Figure4Partial1(exp, cmap2, figsize)
    beta_set = exp.optimal_sol.beta_set;
    p_state = exp.optimal_sol.p_state;
    R = exp.optimal_sol.R;
    V = exp.optimal_sol.V;
    Q = exp.optimal_sol.Q;
    n_actions = size(Q,2);
    Vs_partial_grid = exp.optimal_sol.Vs_partial_grid;
    Rs_partial_grid = exp.optimal_sol.Rs_partial_grid;
    
    
    fig=figure("Position", figsize);
    tiledlayout(1,3, 'Padding', 'none', 'TileSpacing', 'tight'); 
    for na=1:n_actions
        if(na==1)
            symbol = ".";
        else
            symbol = "-";
        end
        nexttile(1); hold on;
        plot(squeeze(Rs_partial_grid(na,1,:)), squeeze(Vs_partial_grid(na,1,:)),symbol,"Color",cmap2(na,:),'MarkerSize',20,'LineWidth',2)
        ylim([-0.1,1])
        xlim([0,log2(6)])
        title("Keep safety action in consideration set")
        ylabel("Trial-averaged reward")
        xlabel("Policy complexity (bits)")
        legend("Na="+[1:n_actions],"location","northwest")
    
        nexttile(2); hold on;
        plot(squeeze(Rs_partial_grid(na,2,:)), squeeze(Vs_partial_grid(na,2,:)),symbol,"Color",cmap2(na,:),'MarkerSize',20,'LineWidth',2)
        title("Remove safety action from consideration set")
        ylim([-0.1,1])
        xlim([0,log2(6)])
        ylabel("Trial-averaged reward")
        xlabel("Policy complexity (bits)")
    
        nexttile(3); hold on;
        plot(squeeze(Rs_partial_grid(na,1,:)), max(squeeze(Vs_partial_grid(na,:,:)),[],1),symbol,"Color",cmap2(na,:),'MarkerSize',20,'LineWidth',2)
        ylim([-0.1,1])
        xlim([0,log2(6)])
        title("Better of both, at each I(S;A) level")
        ylabel("Trial-averaged reward")
        xlabel("Policy complexity (bits)")
    end
end

function [] = Figure4Partial2(exp, figsize)
    Q = exp.optimal_sol.Q;

    figure("Position", figsize)
    heatmap(Q);
    ylabel("States")
    xlabel("Actions")
    title("Q(s,a)")
    colorbar;
end
%% Figure 5: Human behavioral main results
function [] = Figure5(exp, cmap, cmap2, figsize)
    Q = exp.optimal_sol.Q;
    n_states = size(Q,1); n_actions = size(Q,2);
    n_available_actions = 1:n_actions;
    n_actions_chosen_bycond = exp.BehavioralStats.n_actions_chosen_bycond;
    n_actions_chosen_flat = n_actions_chosen_bycond(:);
    complexity_bycond = exp.BehavioralStats.complexity_bycond;
    reward_bycond = exp.BehavioralStats.reward_bycond;
    reward_diff_bycond = exp.BehavioralStats.reward_diff_bycond;
    rt_bycond = exp.BehavioralStats.rt_bycond;
    P_asafe_bycond = exp.BehavioralStats.P_asafe_bycond;
    n_subj = size(complexity_bycond,1);
    conds = exp.BehavioralStats.rt_deadline_conds;
    complexity = complexity_bycond(:);
    reward = reward_bycond(:);
    reward_diff = reward_diff_bycond(:);
    R = exp.optimal_sol.R;
    V = exp.optimal_sol.V;
    V_grid = exp.optimal_sol.V_grid;
    Rs_partial_grid = exp.optimal_sol.Rs_partial_grid;
    Vs_partial_grid = exp.optimal_sol.Vs_partial_grid;

    markersize = 10; linewidth=1.5;
    alpha = 0.5; line_alpha = 0.05; ttl_fontsize = 12; ttl_position_xshift = -0.12; ttl_position_yshift = 1.02;
    ttl_position_xshift2 = -0.29; ttl_position_yshift2 = 0.96;

    figure("Position", figsize)
    tiledlayout(8,6, 'Padding', 'compact', 'TileSpacing', 'compact'); 

    % Rate distortion curve
    nexttile([5,2]); hold on;
    plot(R,V,"k-","LineWidth",1)
    for subj=1:n_subj
        plot(complexity_bycond(subj,:), reward_bycond(subj,:),"-", "Color",[0,0,0,line_alpha], "HandleVisibility","off")
    end
    for cond=1:length(conds)
        scatter(complexity_bycond(:,cond),reward_bycond(:,cond),10,'MarkerFaceColor',cmap(cond,:),'MarkerEdgeColor',cmap(cond,:),'MarkerFaceAlpha',alpha+0.2,'MarkerEdgeAlpha',alpha+0.2)
    end
    %scatter(complexity,V_interp,1,'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',1,'MarkerEdgeAlpha',1)
    ylim([-0.1,1])
    yticks([0:0.25:1])
    xlim([0,log2(6)])
    xlabel("Policy complexity (bits)")
    ylabel("Trial-averaged reward")
    legend("Theoretical","RT < 0.5s","RT < 1s","RT < 2s", "location","northwest")
    ttl = title("A", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    
    % Subject deviation from rate-distortion curve
    nexttile([5,2]); hold on;
    for na = n_available_actions(1:(end-1))
        if(na==1)
            plot(squeeze(Rs_partial_grid(na,1,:)), max(squeeze(Vs_partial_grid(na,:,:)),[],1),"-","Color",cmap2(na,:),'LineWidth',2, "HandleVisibility","off")
        else
            plot(squeeze(Rs_partial_grid(na,1,:)), max(squeeze(Vs_partial_grid(na,:,:)),[],1),"-","Color",cmap2(na,:),'LineWidth',2, "HandleVisibility","off")
        end
        xlim([0, log2(min(n_states, n_actions))])
        ylim([0,1])
    end
    plot(R,V,"k-","LineWidth",1)
    for subj=1:n_subj
        plot(complexity_bycond(subj,:), reward_bycond(subj,:),"-", "Color",[0,0,0,line_alpha], "HandleVisibility","off")
    end
    for Na=1:n_actions
        relevant_subjcond = find(n_actions_chosen_flat==Na);
        scatter(complexity(relevant_subjcond), reward(relevant_subjcond),12,'MarkerFaceColor',cmap2(Na,:),'MarkerEdgeColor',"k",'MarkerFaceAlpha',alpha+0.4,'MarkerEdgeAlpha',alpha)
    end
    ylim([-0.1,1])
    yticks([0:0.25:1])
    xlim([0,log2(6)])
    xlabel("Policy complexity (bits)")
    xlabel("Policy complexity (bits)")
    ylabel("Trial-averaged reward")
    legend(["Theoretical", "Na = "+[1:n_actions]], "location","southeast")
    ttl = title("B", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    nexttile([5,2]); hold on;
    for na = n_available_actions(1:(end-1))
        if(na==1)
            plot(squeeze(Rs_partial_grid(na,1,:)), max(squeeze(Vs_partial_grid(na,:,:)),[],1)-V_grid(1),"-","Color",cmap2(na,:),'LineWidth',2, "HandleVisibility","off")
        else
            plot(squeeze(Rs_partial_grid(na,1,:)), max(squeeze(Vs_partial_grid(na,:,:)),[],1)-V_grid,"-","Color",cmap2(na,:),'LineWidth',2, "HandleVisibility","off")
        end
        xlim([0, log2(min(n_states, n_actions))])
        ylim([0,1])
    end
    plot(R,zeros(length(R),1),"k-","LineWidth",1)
    for subj=1:n_subj
        plot(complexity_bycond(subj,:), reward_diff_bycond(subj,:),"-", "Color",[0,0,0,line_alpha],"LineWidth",1,"HandleVisibility","off")
    end
    for Na=1:n_actions
        relevant_subjcond = find(n_actions_chosen_flat==Na);
        scatter(complexity(relevant_subjcond), reward_diff(relevant_subjcond),12,'MarkerFaceColor',cmap2(Na,:),'MarkerEdgeColor',"k",'MarkerFaceAlpha',alpha+0.4,'MarkerEdgeAlpha',alpha)
    end
    ylim([-0.5, 0.02])
    yticks([-0.5:0.1:0])
    xlim([0,log2(6)])
    ylabel("Reduction in trial-avg. reward")
    xlabel("Policy complexity (bits)")
    ttl = title("C", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    % Policy complexity, by RT deadline cond
    nexttile([3,1]); hold on;
    [se,m] = wse(complexity_bycond,2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    ylim([0,2])
    xlim([min(conds)-0.5,max(conds)+0.5])
    xticks(conds)
    xlabel("RT deadline (s)")
    ylabel("Policy complexity (bits)")
    ttl = title("D", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift2; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    % Reward, by RT deadline cond
    nexttile([3,1]); hold on;
    [se,m] = wse(reward_bycond,2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    ylim([0,1])
    xlim([min(conds)-0.5,max(conds)+0.5])
    xticks(conds)
    xlabel("RT deadline (s)")
    ylabel("Trial-averaged reward")
    ttl = title("E", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift2; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    
    % P(a_safe) as a function of RT deadline
    nexttile([3,1]); hold on;
    [se,m] = wse(P_asafe_bycond, 2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    plot([-0.5, n_actions+0.5],[1/n_actions, 1/n_actions],"k:")
    ylim([0,1])
    xlim([min(conds)-0.5,max(conds)+0.5])
    xticks(conds)
    xlabel("RT deadline (s)")
    ylabel("P(choose safety action)")
    ttl = title("F", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift2; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    
    % RT as a function of RT deadline
    nexttile([3,1]); hold on;
    [se,m] = wse(rt_bycond, 2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    ylim([0,1])
    xlim([min(conds)-0.5,max(conds)+0.5])
    xticks(conds)
    xlabel("RT deadline (s)")
    ylabel("Response time (s)")
    ttl = title("G", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift2; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    
    % n_actions_chosen as a function of RT deadline
    nexttile([3,1]); hold on;
    [se,m] = wse(n_actions_chosen_bycond, 2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    ylim([1,7])
    yticks(1:7)
    xlim([min(conds)-0.5,max(conds)+0.5])
    xticks(conds)
    xlabel("RT deadline (s)")
    ylabel("Number of actions chosen")
    ttl = title("H", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift2; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
    
    
    % Correlation between I(S;A) and Na
    nexttile([3,1]); hold on;
    for subj=1:n_subj
        plot(n_actions_chosen_bycond(subj,:),complexity_bycond(subj,:),"-","Color",[0,0,0,line_alpha],"LineWidth",1,"HandleVisibility","off");
    end
    for c=1:length(conds)
        scatter(n_actions_chosen_bycond(:,c),complexity_bycond(:,c),10,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',alpha+0.2,'MarkerEdgeAlpha',alpha+0.2)
    end
    % [r, r_p_val] = corr(n_actions_chosen_bycond(:),complexity_bycond(:), "rows","pairwise");
    % [rho, rho_p_val] = corr(n_actions_chosen_bycond(:),complexity_bycond(:), "rows","pairwise","type","Spearman");
    % title(sprintf("Pearson\'s r $$="+round(r,3)+"$$, $$p="+round(r_p_val,3)+"$$ \n Spearman\'s rho $$="+round(rho,3)+"$$, $$p="+round(rho_p_val,3)+"$$"),"interpreter","latex")
    xlim([0.5,7.5])
    xticks(1:7)
    ylim([0, log2(6)])
    xlabel("Number of actions chosen")
    ylabel("Policy complexity (bits)")
    ttl = title("I", "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift2; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    set(gca,'box','off')
end



%% Fig S6: Human behavioral results cont'd

function [] = FigureS6(exp1_rewardcutoff, cmap, figsize)
    Q = exp1_rewardcutoff.optimal_sol.Q;
    n_states = size(Q,1); n_actions = size(Q,2);
    n_actions_chosen_bycond = exp1_rewardcutoff.BehavioralStats.n_actions_chosen_bycond;
    complexity_bycond = exp1_rewardcutoff.BehavioralStats.complexity_bycond;
    rt_bycond = exp1_rewardcutoff.BehavioralStats.rt_bycond;
    mean_train_accuracy = exp1_rewardcutoff.BehavioralStats.mean_train_accuracy;
    P_asafe_bycond = exp1_rewardcutoff.BehavioralStats.P_asafe_bycond;
    included_subjs = exp1_rewardcutoff.Analyses.included_subjs;
    excluded_subjs = exp1_rewardcutoff.Analyses.excluded_subjs;
    n_subj = size(complexity_bycond,1);
    conds = exp1_rewardcutoff.BehavioralStats.rt_deadline_conds;
    
    alpha = 0.5; line_alpha = 0.05; ttl_fontsize = 12; ttl_position_xshift = -0.22; ttl_position_yshift = 1.02;
    
    figure("Position", figsize)
    tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'tight'); 
    titles = ["A","B","C","D","E","F"];
    for tile=1:6
        nexttile(tile); hold on;
        ttl = title(titles(tile), "Fontsize", ttl_fontsize);
        ttl.Units = 'Normalize'; 
        ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
        ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
        ttl.HorizontalAlignment = 'left'; 
        set(gca,'box','off')
    end
    
    nexttile(1); hold on;
    plot([1:n_actions], log2([1:n_states,n_states]), "k.-");
    
    for subj=1:n_subj
        nexttile(1); hold on;
        plot(n_actions_chosen_bycond(subj,:),complexity_bycond(subj,:),"-", "Color",[0,0,0,line_alpha],"HandleVisibility","off");
        nexttile(2); hold on;
        plot(complexity_bycond(subj,:),rt_bycond(subj,:), "-", "Color",[0,0,0,line_alpha],"HandleVisibility","off")
        nexttile(3); hold on;
        plot(n_actions_chosen_bycond(subj,:),rt_bycond(subj,:),"-", "Color",[0,0,0,line_alpha],"HandleVisibility","off")
        nexttile(4); hold on;
        plot(complexity_bycond(subj,:),P_asafe_bycond(subj,:), "-", "Color",[0,0,0,line_alpha],"HandleVisibility","off")
    end
    for c=1:length(conds)
        nexttile(1); hold on;
        scatter(n_actions_chosen_bycond(:,c),complexity_bycond(:,c),10,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',alpha+0.1,'MarkerEdgeAlpha',alpha+0.1)
        xlabel("Number of actions chosen")
        ylabel("Policy complexity (bits)")
        xlim([0.5,7.5])
        ylim([0,log2(6)])
        
        nexttile(2); hold on;
        scatter(complexity_bycond(:,c),rt_bycond(:,c),10,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',alpha+0.1,'MarkerEdgeAlpha',alpha+0.1)
        xlabel("Policy complexity (bits)")
        ylabel("Response time (sec)")
        xlim([0,log2(6)])
    
        nexttile(3); hold on;
        scatter(n_actions_chosen_bycond(:,c),rt_bycond(:,c),10,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',alpha+0.1,'MarkerEdgeAlpha',alpha+0.1)
        xlabel("Number of actions chosen")
        ylabel("Response time (sec)")
        xlim([0.5,7.5])
    
        nexttile(4); hold on;
        scatter(complexity_bycond(:,c),P_asafe_bycond(:,c),10,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',alpha+0.1,'MarkerEdgeAlpha',alpha+0.1)
        xlabel("Policy complexity (bits)")
        ylabel("P(choose safety action)")
        xlim([0,log2(6)])
        ylim([0,1])
    end
    nexttile(1);
    leg=legend("Maximum","RT < 0.5s", "RT < 1s", "RT < 2s", "location","northwest");
    set(leg.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
    
    
    nexttile(5); hold on;
    scatter(mean_train_accuracy(included_subjs), mean(complexity_bycond(included_subjs,:),2), 10,'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
    scatter(mean_train_accuracy(excluded_subjs), mean(complexity_bycond(excluded_subjs,:),2), 10,'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
    xlabel("Training blocks mean accuracy")
    ylabel({"Test blocks,", "Mean policy complexity (bits)"})
    leg=legend("Included","Excluded", "location","northwest");
    set(leg.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
    xlim([0,1])
    ylim([0,log2(6)])
    
    nexttile(6); hold on;
    scatter(mean_train_accuracy(included_subjs), mean(n_actions_chosen_bycond(included_subjs,:),2), 10,'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
    scatter(mean_train_accuracy(excluded_subjs), mean(n_actions_chosen_bycond(excluded_subjs,:),2), 10,'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
    xlim([0,1])
    ylim([0.5,7.5])
    xlabel("Training blocks mean accuracy")
    ylabel({"Test blocks,", "Mean number of actions chosen"})

end

%% Fig S7: I(S;A) distribution for each Na
function [ks_test_results] = FigureS7(exp, figsize)
    n_actions_chosen = exp.BehavioralStats.n_actions_chosen_bycond;
    n_actions_max = max(n_actions_chosen(:));
    [n_subjects,n_conds] = size(n_actions_chosen);
    N_a_histcounts = histcounts(n_actions_chosen(:));
    N_a_histcounts_cumsum = [0,cumsum(N_a_histcounts)];
    
    % Test for uniformity of I(S;A) within each N_a. 
    complexity = exp.BehavioralStats.complexity_bycond;
    complexity_flat = complexity(:);
    n_actions_chosen_flat = n_actions_chosen(:);
    ks_test_results = zeros(n_actions_max,1);
    ks_test_results(:)= nan; 
    n_dpts = zeros(n_actions_max,1);
    for n_a = 2:n_actions_max
        relevant_subjconds = (n_actions_chosen_flat==n_a);
        relevant_complexity = complexity_flat(relevant_subjconds);
        test_cdf = makedist('Uniform','Lower',0,'Upper',log2(n_a));
        %test_cdf = makedist('Uniform','Lower',min(relevant_complexity),'Upper',max(relevant_complexity));
        [h,p] = kstest(relevant_complexity,'CDF',test_cdf);
        n_dpts(n_a) = length(relevant_complexity);
        ks_test_results(n_a) = p;
    end
    
    % Figure
    figure("Position",figsize);
    tiledlayout(2,3,'Padding', 'compact', 'TileSpacing', 'compact');
    for n_a = 2:n_actions_max
        relevant_subjconds = (n_actions_chosen_flat==n_a);
        relevant_complexity = complexity_flat(relevant_subjconds);
        nexttile; hold on;
        histogram(relevant_complexity, 0:(log2(n_a)/20):log2(n_a), 'FaceColor', 'k', 'FaceAlpha',0.2)
        title("Na = "+n_a);
        %title("$N_a = "+n_a+"$","interpreter","latex");
        xlim([0, log2(n_a)]);
        if(n_a==2 || n_a==5)
            ylabel("Frequency")
        end
        if(n_a>=5)
            xlabel("Policy complexity (bits)")
        end
    end
end

%% Fig S8: P(a) of all subjects

function [] = FigureS8(exp, cmap, figsize)
    markersize = 10; linewidth=1.5;

    Q = exp.optimal_sol.Q;
    n_states = size(Q,1); n_actions = size(Q,2);
    P_a = exp.BehavioralStats.P_a;
    P_a_ranked = exp.BehavioralStats.P_a_ranked;
    P_a_bycond = exp.BehavioralStats.P_a_bycond;
    P_a_bycond_ranked = exp.BehavioralStats.P_a_bycond_ranked;
    n_subj = size(P_a,1);
    conds = exp.BehavioralStats.rt_deadline_conds;

    figure("Position",figsize)
    tiledlayout(2,length(conds)+1, 'Padding', 'compact', 'TileSpacing', 'tight'); 
    
    nexttile(1); hold on;
    plot(0:(n_actions-1),P_a', 'Color',[0,0,0,0.1])
    scatter(0:(n_actions-1),P_a',1,'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1)
    errorbar(0:(n_actions-1),mean(P_a,1),std(P_a,[],1)./sqrt(n_subj),'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    plot([-0.5, n_actions+0.5],[1/n_actions, 1/n_actions],"k:")
    ylim([0,1])
    xticks(0:(n_actions-1))
    xticklabels(["E",1:6])
    xlim([-0.5,n_actions-1+0.5])
    xlabel("Actions (keys; E is safety)")
    ylabel("P(a)")
    title("Aggregate over conditions")
    
    
    % Empirical P(a), ordered by freq
    nexttile(length(conds)+2); hold on;
    plot(1:n_actions,P_a_ranked', 'Color',[0,0,0,0.1])
    scatter(1:n_actions,P_a_ranked',1,'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1)
    errorbar(1:n_actions,mean(P_a_ranked,1),std(P_a_ranked,[],1)./sqrt(n_subj),'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    plot([0.5, n_actions+0.5],[1/n_actions, 1/n_actions],"k:")
    ylim([0,1])
    xticks(1:n_actions)
    xlim([0.5,n_actions+0.5])
    xlabel("Actions, by order of chosen freq")
    ylabel("P(a)")
    
    for c=1:length(conds)
        % Empirical P(a)
        nexttile(1+c); hold on;
        plot(0:(n_actions-1),squeeze(P_a_bycond(:,c,:))', 'Color',[cmap(c,:),0.1])
        scatter(0:(n_actions-1),squeeze(P_a_bycond(:,c,:))',1,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1)
        errorbar(0:(n_actions-1),mean(squeeze(P_a_bycond(:,c,:)),1),std(squeeze(P_a_bycond(:,c,:)),[],1)./sqrt(n_subj),'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
        plot([-0.5, n_actions+0.5],[1/n_actions, 1/n_actions],"k:")
        ylim([0,1])
        xticks(0:(n_actions-1))
        xticklabels(["E",1:6])
        xlim([-0.5,n_actions-1+0.5])
        xlabel("Actions (keys; E is safety)")
        title("Condition RT<"+conds(c)+"s")
        %ylabel("P(a)")
        
        % % Empirical P(a), ordered by freq
        nexttile(5+c); hold on;
        plot(1:n_actions,squeeze(P_a_bycond_ranked(:,c,:))', 'Color',[cmap(c,:),0.1])
        scatter(1:n_actions,squeeze(P_a_bycond_ranked(:,c,:))',1,'MarkerFaceColor',cmap(c,:),'MarkerEdgeColor',cmap(c,:),'MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1)
        errorbar(1:n_actions,mean(squeeze(P_a_bycond_ranked(:,c,:)),1),std(squeeze(P_a_bycond_ranked(:,c,:)),[],1)./sqrt(n_subj),'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
        plot([0.5, n_actions+0.5],[1/n_actions, 1/n_actions],"k:")
        ylim([0,1])
        xticks(1:n_actions)
        xlim([0.5,n_actions+0.5])
        xlabel("Actions, by order of chosen freq")
        %ylabel("P(a)")
    end

end

%% Figure S8: Noisy Q-value simulations
function [] = noisyQ_simulations(exp1, gaussian_stds, n_sims, beta_set, rng_seed, save_path)
    if(nargin==1)
        gaussian_stds = [0.2, 0.5, 1.5, 3];
        n_sims = 200;
        n_tot = 50;
        beta_set = linspace(0.1,15,n_tot);
        save_path = "saved_files\";
        rng_seed = 0;
    end
    R = exp1.optimal_sol.R;
    V = exp1.optimal_sol.V;
    p_state = exp1.optimal_sol.p_state;
    Q = exp1.optimal_sol.Q;
    n_actions = size(Q,2);
    
    R_noisys = zeros(length(gaussian_stds), 2, n_actions, n_tot, n_sims);
    V_noisy_trues = zeros(length(gaussian_stds), 2, n_actions, n_tot, n_sims);
    
    for noise_level=1:length(gaussian_stds)
        noise_level
        noise = gaussian_stds(noise_level);
    
        % Retaining a_safe
        rng(rng_seed); % Set see for reproducibility
        for n_a =1:n_actions
            Q_partialNa = Q(:,1:n_a);
            for iter = 1:n_sims
                Q_noisy = Q_partialNa + randn(size(Q_partialNa)).*noise;
                [R_noisy, V_noisy, ~,optimal_policy_noisy] = blahut_arimoto(p_state,Q_noisy,beta_set);
                V_noisy_true = zeros(n_tot,1);
                for beta_idx=1:n_tot
                    V_noisy_true(beta_idx) = sum(p_state*(optimal_policy_noisy{beta_idx}.*Q_partialNa));
                end
                R_noisys(noise_level,1,n_a, :, iter) = R_noisy;
                V_noisy_trues(noise_level,1,n_a, :, iter) = V_noisy_true;
            end
            R_noisys_mean = mean(squeeze(R_noisys(noise_level,1,n_a,:,:)),2);
            R_noisys_sem = std(squeeze(R_noisys(noise_level,1,n_a,:,:)),[],2)./sqrt(n_sims);
            V_noisy_trues_mean = mean(squeeze(V_noisy_trues(noise_level,1,n_a,:,:)),2);
            V_noisy_trues_sem = std(squeeze(V_noisy_trues(noise_level,1,n_a,:,:)),[],2)./sqrt(n_sims);
        end
    
        % Not retaining a_safe
        nexttile(noise_level+4); hold on;
        rng(rng_seed); % Set see for reproducibility
        for n_a =1:n_actions
            Q_partialNa = Q(:,(end-n_a+1):end);
            for iter = 1:n_sims
                Q_noisy = Q_partialNa + randn(size(Q_partialNa)).*noise;
                [R_noisy, V_noisy, ~,optimal_policy_noisy] = blahut_arimoto(p_state,Q_noisy,beta_set);
                V_noisy_true = zeros(n_tot,1);
                for beta_idx=1:n_tot
                    V_noisy_true(beta_idx) = sum(p_state*(optimal_policy_noisy{beta_idx}.*Q_partialNa));
                end
                R_noisys(noise_level,2,n_a, :, iter) = R_noisy;
                V_noisy_trues(noise_level,2,n_a, :, iter) = V_noisy_true;
            end
            R_noisys_mean = mean(squeeze(R_noisys(noise_level,2,n_a,:,:)),2);
            R_noisys_sem = std(squeeze(R_noisys(noise_level,2,n_a,:,:)),[],2)./sqrt(n_sims);
            V_noisy_trues_mean = mean(squeeze(V_noisy_trues(noise_level,2,n_a,:,:)),2);
            V_noisy_trues_sem = std(squeeze(V_noisy_trues(noise_level,2,n_a,:,:)),[],2)./sqrt(n_sims);
        end
    end
    save(save_path+"noisyQ_simulation_results.mat", "p_state","Q","beta_set","R_noisys","V_noisy_trues","gaussian_stds","R","V","n_sims","n_tot")

end

function [] = FigureS9(cmap2, figsize,save_path)
    if(nargin==2)
        save_path="saved_files\";
    end
    figure("Position", figsize)
    tiledlayout(2,4, 'Padding', 'compact', 'TileSpacing', 'tight'); 
    load(save_path+"noisyQ_simulation_results.mat");
    n_actions = size(Q,2);
    for noise_level=1:length(gaussian_stds)
        noise_level
        noise = gaussian_stds(noise_level);
    
        % Retaining a_safe
        nexttile(noise_level); hold on;
        plot(R,V,"k-")
        for n_a =1:n_actions
            R_noisys_mean = mean(squeeze(R_noisys(noise_level,1,n_a,:,:)),2);
            R_noisys_sem = std(squeeze(R_noisys(noise_level,1,n_a,:,:)),[],2)./sqrt(n_sims);
            V_noisy_trues_mean = mean(squeeze(V_noisy_trues(noise_level,1,n_a,:,:)),2);
            V_noisy_trues_sem = std(squeeze(V_noisy_trues(noise_level,1,n_a,:,:)),[],2)./sqrt(n_sims);
            errorbar(R_noisys_mean, V_noisy_trues_mean, -V_noisy_trues_sem, V_noisy_trues_sem, -R_noisys_sem, R_noisys_sem,".:",'Color',cmap2(n_a,:),'Capsize',0,"MarkerSize",5)
        end
        xlim([0, log2(6)])
        ylim([-0.2,1])
        xlabel("Policy complexity (bits)")
        title("$\sigma="+noise+"$", "interpreter","latex")
        if(noise_level==1)
            leg = legend(["Theoretical","Na = "+[1:n_actions]], "location","southeast","fontsize",8);
            set(leg.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]))
            ylabel({"Keep safety action", "Trial-averaged reward"})
        else
            ylabel("Trial-averaged reward")
        end
    
        % Not retaining a_safe
        nexttile(noise_level+4); hold on;
        plot(R,V,"k-","HandleVisibility","off")
        for n_a =1:n_actions
            R_noisys_mean = mean(squeeze(R_noisys(noise_level,2,n_a,:,:)),2);
            R_noisys_sem = std(squeeze(R_noisys(noise_level,2,n_a,:,:)),[],2)./sqrt(n_sims);
            V_noisy_trues_mean = mean(squeeze(V_noisy_trues(noise_level,2,n_a,:,:)),2);
            V_noisy_trues_sem = std(squeeze(V_noisy_trues(noise_level,2,n_a,:,:)),[],2)./sqrt(n_sims);
            errorbar(R_noisys_mean, V_noisy_trues_mean, -V_noisy_trues_sem, V_noisy_trues_sem, -R_noisys_sem, R_noisys_sem,".:",'Color',cmap2(n_a,:),'Capsize',0,"MarkerSize",5)
        end
        xlim([0, log2(6)])
        ylim([-0.2,1])
        xlabel("Policy complexity (bits)")
        if(noise_level==1)
            ylabel({"Remove safety action", "Trial-averaged reward"})
        else
            ylabel("Trial-averaged reward")
        end
    end
end

