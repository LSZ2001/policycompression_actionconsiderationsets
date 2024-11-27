function [] = approximate_policy_compression_wrapper(task_name, use_IS_approx, sample_with_replacement, save_path, num_samples, rng_seed)
    if(nargin==3)
        save_path = "saved_files\"; num_samples=200; rng_seed=0;
    end

    % Sampling methods
    if(use_IS_approx)
        save_suffix_IS = "_IS";
    else
        save_suffix_IS = "_BA";
    end
    if(sample_with_replacement)
        save_suffix_IS = save_suffix_IS + "samplewithreplace";
    else
        save_suffix_IS = save_suffix_IS + "samplewithoutreplace";
    end

    % Load task
    switch task_name
        case "exp1_Ns6"
            n_states = 6;
            n_actions = 7;
            p_state = ones(1,n_states)./n_states;
            Q = eye(n_states);
            q = 0.2.*ones(n_states,1);
            Q = normalize(Q, "range", [-0.18,1]);
            Q = [q,Q];
            Num_action_samples_withoutreplacement = 1:7;
            Num_action_samples_withreplacement = [1,2,3,5,7,10,20,1000];
        case "exp0_randomQ"
            n_states = 16;
            n_actions = 32;
            p_state = ones(1,n_states)./n_states;
            inv_temp = 5;
            rng(1);
            Q = exprnd(0.5,n_states,n_actions);
            Q = exp(inv_temp.*Q)./sum(inv_temp.*exp(Q),"all");
            Q = normalize(Q, "range", [0,1]);
            Num_action_samples_withoutreplacement = [1,3,5,7,10,15,20,32];
            Num_action_samples_withreplacement = [1,2,3,5,7,10,20,1000];
        case "exp0_sparseQ"
            n_states = 16;
            n_actions = 32;
            p_state = ones(1,n_states)./n_states;
            inv_temp = 5;
            rng(1);
            Q = exprnd(0.1,n_states,n_actions);
            Q = exp(inv_temp.*Q)./sum(inv_temp.*exp(Q),"all");
            Q = normalize(Q, "range", [0,0.1]);
            %Q = zeros(n_states, n_actions);
            Q(1:10,1)=1;
            Q(11:14,2)=1;
            Q(15:16,3)=1;
            Num_action_samples_withoutreplacement = [1,3,5,7,10,15,20,32];
            Num_action_samples_withreplacement = [1,2,3,5,7,10,20,1000];
    end

    % Optimal policy compression
    n_tot = 100;
    beta_set = linspace(0.1,15,n_tot);
    [R, V, ~,~] = blahut_arimoto(p_state,Q,beta_set);

    % Approximate policy compression
    beta_vals(1,1,:) = beta_set;
    if(sample_with_replacement)
        Num_action_samples = Num_action_samples_withreplacement;
    else
        Num_action_samples = Num_action_samples_withoutreplacement;
    end
    proposal_distribution_names = [1,2,3]; % can be 1-uniform, 2-P*(a), or 3-V(a).

    
    for proposal_distribution_name = proposal_distribution_names
   
        [R_approxes,V_approxes,P_a_approxes, ~, P_a_approx_init, resample_every_beta, ~] = approximate_policy_compression(p_state, Q, beta_vals, Num_action_samples, num_samples, proposal_distribution_name, rng_seed, use_IS_approx, sample_with_replacement);
        
        % Show approximately optimal policy and the reward-complexity frontier.
        R_approxes_avg = zeros(length(Num_action_samples), n_tot);
        R_approxes_sem = zeros(length(Num_action_samples), n_tot);
        V_approxes_avg = zeros(length(Num_action_samples), n_tot);
        V_approxes_sem = zeros(length(Num_action_samples), n_tot);
        for samplesize_idx = 1:length(Num_action_samples)
            Num_action_samples(samplesize_idx)
            R_approxes(~isfinite(R_approxes)) = nan; % Prevent Infs from showing up in the R. 
            R_approxes_avg(samplesize_idx,:) = nanmean(squeeze(R_approxes(samplesize_idx, :,:)),1);
            R_approxes_sem(samplesize_idx,:) = nanstd(squeeze(R_approxes(samplesize_idx, :,:)),[],1)./sqrt(num_samples);
            V_approxes_avg(samplesize_idx,:) = nanmean(squeeze(V_approxes(samplesize_idx, :,:)),1);
            V_approxes_sem(samplesize_idx,:) = nanstd(squeeze(V_approxes(samplesize_idx, :,:)),[],1)./sqrt(num_samples);
        end

        save(save_path + "model"+proposal_distribution_name+"_suboptimality_"+task_name+save_suffix_IS,"Num_action_samples_withoutreplacement","Num_action_samples_withreplacement","R_approxes_avg","V_approxes_avg","R_approxes_sem","V_approxes_sem","P_a_approxes","P_a_approx_init","R","V","Q","p_state","beta_set","proposal_distribution_name","resample_every_beta")
    end
end