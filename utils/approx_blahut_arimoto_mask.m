function [R_approx,V_approx,P_a_approx, Optimal_policy_approx] = approx_blahut_arimoto_mask(Ps, Q, beta_vals, num_action_samples, P_a_approx_init, sample_with_replacement, resample_every_beta, N_counts)
    % Ps: P(s); |S| x 1 vector.
    % Q: Q(s,a); |S| x |A| matrix.
    % beta_val: grid of Lagrangian multipler values; 1 x 1 x n_tot. 
    % num_action_samples: number of actions in the sample; 1 x |A|. 
    % P_a_approx_init: the theoretical optimal policy's marginal action distribution across all beta values; n_tot x |A|.  
    % resample_every_beta: whether to resample a new set of actions for each beta value.
    % N_counts: only used for select approximation models. When not nan, it provides
    %   directly the action sampled for importance sampling. If it is not nan, then need to provide a
    %   corresponding P_a_approx_init from which N_counts is truly sampled from.
    if(nargin==6)
        resample_every_beta=false;
        N_counts = nan;
    elseif(nargin==7)
        N_counts = nan;
    end

    N_counts_not_provided = isnan(N_counts);

    [n_states, n_actions] = size(Q);
    n_tot = length(squeeze(beta_vals));
    nIter = 50000;
    Ps = Ps(:); % Ensure that P(s) is a column vector
    V_approx = zeros(1,n_tot);
    R_approx = zeros(1,n_tot);
    P_a_approx = zeros(1, n_actions, n_tot);
    Optimal_policy_approx = zeros(n_states, n_actions, n_tot);
    if(~resample_every_beta && N_counts_not_provided)
        p_a_approx_init = P_a_approx_init(1,:);
        if(sample_with_replacement)
            action_samples = randsample(n_actions,num_action_samples,true,p_a_approx_init)';
        else
            action_samples = zeros(1,num_action_samples);
            possible_actions = 1:n_actions;
            p_unnorm = p_a_approx_init;
            if(num_action_samples == n_actions)
                action_samples = possible_actions;
            else
                for i = 1:num_action_samples
                    % Normalize the probability distribution
                    p_norm = p_unnorm ./ sum(p_unnorm);
                    
                    % Sample an integer based on the current probability distribution
                    action_samples(i) = randsample(possible_actions,1,true,p_norm);
                    
                    % Remove the sampled integer from the set and its probability from p
                    sampled_idx = (possible_actions == action_samples(i));
                    possible_actions(sampled_idx) = [];
                    p_unnorm(sampled_idx) = [];
                end
            end
        end
        [N_counts, ~] = histcounts(action_samples, 0.5:1:(n_actions+0.5));
        clear action_samples
    end

    for beta_idx = 1:n_tot
        if(resample_every_beta && N_counts_not_provided)
            p_a_approx_init = P_a_approx_init(beta_idx,:);
            if(sample_with_replacement)
                action_samples = randsample(n_actions,num_action_samples,true,p_a_approx_init)';
            else
                action_samples = zeros(1,num_action_samples);
                possible_actions = 1:n_actions;
                p_unnorm = p_a_approx_init;
                if(num_action_samples == n_actions)
                    action_samples = possible_actions;
                else
                    for i = 1:num_action_samples
                        % Normalize the probability distribution
                        p_norm = p_unnorm ./ sum(p_unnorm);
                        
                        % Sample an integer based on the current probability distribution
                        action_samples(i) = randsample(possible_actions,1,true,p_norm);
                        
                        % Remove the sampled integer from the set and its probability from p
                        sampled_idx = (possible_actions == action_samples(i));
                        possible_actions(sampled_idx) = [];
                        p_unnorm(sampled_idx) = [];
                    end
                end
            end
            [N_counts, ~] = histcounts(action_samples, 0.5:1:(n_actions+0.5));
            clear action_samples
        end

        beta = beta_vals(beta_idx);

        p_a_approx = p_a_approx_init; % Initialize P(a) at P*(a)
        %p_a_approx = ones(1,length(p_a_approx_init)); % Initialize P(a) at P*(a)

        v0 = mean(Q(:));
        F = exp(beta .* Q);
        for i=1:nIter

            %optimal_policy_approx = F .* N_counts .* p_a_approx./(p_a_approx_init);
            %optimal_policy_approx = F .* N_counts .* p_a_approx;
            optimal_policy_approx = F .* (N_counts>0) .* p_a_approx;

            optimal_policy_approx = optimal_policy_approx ./ sum(optimal_policy_approx,2);
            p_a_approx = Ps' * optimal_policy_approx;
            v = sum(Ps' * (optimal_policy_approx .* Q)); % v is the average reward under the current policy
            if abs(v-v0) < 1e-8; break; else v0 = v; end %
        end
        if(i==nIter)
           warning="Warning: non-convergence for beta="+beta_vals(beta_idx)
        end
        V_approx(beta_idx) = v; 

        % Compute mutual information I(s, a) on log2 scale
        P_sa = optimal_policy_approx .* Ps; % P(s, a) = P(a|s) * P(s)
        P_a = sum(P_sa, 1);
        R_approx(beta_idx) = nansum(P_sa .* log2(P_sa ./ (Ps .* P_a)), "all");

        P_a_approx(:,:, beta_idx) = p_a_approx;
        Optimal_policy_approx(:,:, beta_idx) = optimal_policy_approx;

    end
    
end