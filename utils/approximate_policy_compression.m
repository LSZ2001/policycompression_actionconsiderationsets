function [R_approxes,V_approxes,P_a_approxes, optimal_policy_approx, P_a_approx_init, resample_every_beta, proposal_distribution_latex] = approximate_policy_compression(p_state, Q, beta_vals, Num_action_samples, num_samples, proposal_distribution_name, rng_seed, use_IS_approx, sample_with_replacement)
% Given task specification and approximation model specification, return
% the predictions of approximate policy compression.
% p_state: p(s) vector. 1 x |S|
% Q: Q(s,a) matrix. |S| x |A|
% beta_vals: Langrage multipler values. 1 x n_tot
% Num_action_samples: set of sample sizes used in importance sampling. 1 x ?
% num_samples: for each sample size, the number of samples to use in simulation. Scalar. 
% proposal_distribution_name: a integer specifying the proposal distribution P_0(a) used for importance sampling.
% rng_seed: seed used to generate samples for importance sampling.

    % Proposal distribution to sample actions.
    [n_states,n_actions] = size(Q);
    n_tot = length(beta_vals);

    % For some proposal distributions, need to get its form from the task
    % info. 
    switch proposal_distribution_name
        case 2
            [~, ~, Pa, ~] = blahut_arimoto(p_state,Q,squeeze(beta_vals));
        case 3
            p_s = zeros(n_states,1);
            p_s(:,1) = p_state;
            V_a = p_s' * Q; % The average Q value of an action: V(a) = \sum_s p(s)*Q(s,a)
        case 5
            p_s = zeros(n_states,1);
            p_s(:,1) = p_state;
            V_a = p_s' * Q; % The average Q value of an action: V(a) = \sum_s p(s)*Q(s,a)
    end

    P_a_approx_init = zeros(n_tot, n_actions);
    for beta_idx=1:n_tot
        switch proposal_distribution_name
            case 1
                P_a_approx_init(beta_idx,:) = ones(1,n_actions)./n_actions; 
                resample_every_beta = false;
                proposal_distribution_latex = "=\frac{1}{|A|}";
            case 2
                P_a_approx_init(beta_idx,:) = Pa(beta_idx,:);
                resample_every_beta = true;
                proposal_distribution_latex = "=P^*(a)";
            case 3
                P_a_approx_init(beta_idx,:) = V_a./sum(V_a);
                resample_every_beta = false;
                proposal_distribution_latex = "\propto \sum_s p(s) \ Q(s,a)";
            case 4
                resample_every_beta = true;
                proposal_distribution_latex = "\propto \sum_{s \in S} p'(s \in S) \ \pi^*(a|s \in S)";
            case 5
                P_a_approx_init(beta_idx,:) = exp(V_a)./sum(exp(V_a));
                resample_every_beta = false;
                proposal_distribution_latex = "\propto \exp \left(\sum_s p(s) \ Q(s,a) \right)";
        end
    end
    
    
    % Dimensions: [sample size of any sample drawn, the ith sample drawn, the
    % V/R functions over beta].
    rng(rng_seed)
    V_approxes = zeros(length(Num_action_samples), num_samples, n_tot);
    R_approxes = zeros(length(Num_action_samples), num_samples, n_tot);
    P_a_approxes = zeros(length(Num_action_samples), num_samples, n_actions, n_tot);
    for samplesize_idx = 1:length(Num_action_samples)
        Num_action_samples(samplesize_idx)
        num_action_samples = Num_action_samples(samplesize_idx);
        switch proposal_distribution_name
            case 4 % For every sample, resample the state subset being considered.
                state_subset = sort(randsample(n_states,n_states./2,false))';
                state_excluded = setdiff(1:n_states,state_subset);
                p_state_subset = p_state;
                p_state_subset(state_excluded)=0;
                p_state_subset = p_state_subset ./ sum(p_state_subset);
                [~, ~, Pa_subset, ~] = blahut_arimoto(p_state_subset,Q,squeeze(beta_vals));
                for beta_idx=1:n_tot
                    P_a_approx_init(beta_idx, :) = Pa_subset(beta_idx,:);
                end
        end
        for iter = 1:num_samples
            % Dimensions: [state, action, beta]
            if(use_IS_approx)
                [R_approx,V_approx,P_a_approx, optimal_policy_approx] = approx_blahut_arimoto(p_state, Q, beta_vals, num_action_samples, P_a_approx_init, sample_with_replacement, resample_every_beta);
            else
                [R_approx,V_approx,P_a_approx, optimal_policy_approx] = approx_blahut_arimoto_mask(p_state, Q, beta_vals, num_action_samples, P_a_approx_init, sample_with_replacement, resample_every_beta);
            end
            V_approxes(samplesize_idx, iter,:) = V_approx;
            R_approxes(samplesize_idx, iter,:) = R_approx;
            P_a_approxes(samplesize_idx, iter,:,:) = squeeze(P_a_approx);
        end
    end
end