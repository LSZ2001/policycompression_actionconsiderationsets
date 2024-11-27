function [] = visualize_simulations(save_path, plot_rule, task_name, sample_size_idxs_visualized_withoutreplacement, sample_size_idxs_visualized_withreplacement, cmap, figsize)
    alpha = 0.5; line_alpha = 0.05; ttl_fontsize = 15; ttl_position_xshift = -0.21; ttl_position_yshift = 0.99;
    proposal_distribution_latexes = {"$$P_0(a)=\frac{1}{|A|}$$", "$$P_0(a)=P^*(a)$$", "$$P_0(a) \propto \sum_s p(s) \ Q(s,a)$$", "$$P_0(a)\propto \sum_{s \in S} p'(s \in S) \ \pi^*(a|s \in S)$$", "$$P_0(a) \propto \exp \left(\sum_s p(s) \ Q(s,a) \right)$$"};
    
    % Get matrix Q
    load(save_path + "model1_suboptimality_"+task_name+"_BAsamplewithoutreplace")

    models_plotted = [1,3,2];
    figure("Position",figsize);

    T=tiledlayout(1,5, 'Padding', 'tight', 'TileSpacing', 'tight'); %Outer layout

    t=tiledlayout(T,1,1); % Inner layout
    t.Layout.Tile = 1;
    t.Layout.TileSpan = [1,2];
    nexttile(t,1,[1,1]);
    h=heatmap(Q);
    ylabel("States")
    xlabel("Actions")
    if(task_name=="exp1_Ns6")
        h.XDisplayLabels = ["E",1:6];
    end
    annotation('textbox', [0 0.96 0.05 0.05], ...
    'String', 'A', ...
    'FontSize', ttl_fontsize, ...
    'FontWeight', 'bold', ...
    'LineStyle', 'none', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle');
    
    t=tiledlayout(T,3,6, 'Padding', 'none', 'TileSpacing', 'compact'); 
    t.Layout.Tile = 3;
    t.Layout.TileSpan = [1,3];
    annotation('textbox', [0.39 0.96 0.05 0.05], ...
    'String', 'B', ...
    'FontSize', ttl_fontsize, ...
    'FontWeight', 'bold', ...
    'LineStyle', 'none', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle');
    
    use_IS_approxs = [false,true,false];
    sample_with_replacements = [false,true,true];
    
    for approximation_method =1:length(use_IS_approxs)
        use_IS_approx = use_IS_approxs(approximation_method);
        sample_with_replacement = sample_with_replacements(approximation_method);
        if(use_IS_approx)
            save_suffix_IS = "_IS";
            plot_title = "SNIS";
        else
            save_suffix_IS = "_BA";
            plot_title = "BA on retained actions";
        end
        if(sample_with_replacement)
            save_suffix_IS = save_suffix_IS + "samplewithreplace";
            plot_title = {plot_title,"sample with replacement"};
            Num_action_samples = Num_action_samples_withreplacement;
            sample_size_idxs_visualized = sample_size_idxs_visualized_withreplacement;
        else
            save_suffix_IS = save_suffix_IS + "samplewithoutreplace";
            plot_title = {plot_title,"sample without replacement"};
            Num_action_samples = Num_action_samples_withoutreplacement;
            sample_size_idxs_visualized = sample_size_idxs_visualized_withoutreplacement;
        end
        color_idx = 1;
    
        for model_idx=1:length(models_plotted)
            model = models_plotted(model_idx);
            load(save_path + "model"+model+"_suboptimality_"+task_name+save_suffix_IS)
            tile_count = 0;
            for samplesize_idx = 1:length(Num_action_samples)
                if(~ismember(samplesize_idx, sample_size_idxs_visualized))
                    continue;
                else
                    tile_count = tile_count+1;
                end
                [approximation_method,model,samplesize_idx]
                nexttile(t, tile_count + (approximation_method-1)*length(sample_size_idxs_visualized), [1,1]); hold on;
                r_approxes_avg = R_approxes_avg(samplesize_idx,:);
                v_approxes_avg = V_approxes_avg(samplesize_idx,:);
                v_corresponding_optimal = interp1(R,V,r_approxes_avg,"linear");
                switch plot_rule
                    case "original"
                        if(Num_action_samples(samplesize_idx)==1)
                            errorbar(r_approxes_avg(1),v_approxes_avg(1), -V_approxes_sem(samplesize_idx,1), V_approxes_sem(samplesize_idx,1), -R_approxes_sem(samplesize_idx,1), R_approxes_sem(samplesize_idx,1),".",'Color', cmap(color_idx,:), 'Capsize',0,"MarkerSize",15)
                        else
                            errorbar(r_approxes_avg,(v_approxes_avg), -V_approxes_sem(samplesize_idx,:), V_approxes_sem(samplesize_idx,:), -R_approxes_sem(samplesize_idx,:), R_approxes_sem(samplesize_idx,:),".:",'Color', cmap(color_idx,:), 'Capsize',0,"MarkerSize",5)
                        end                    
                        ymin=0;
                        ymax=1;
                        ylab = "Trial-averaged reward";
                        title_lab = "T";
                        plot(R,V,"k:","HandleVisibility","off")
                        yticks([0:0.25:1])
                    case "difference"
                        if(Num_action_samples(samplesize_idx)==1)
                            errorbar(r_approxes_avg(1),v_approxes_avg(1)-V(1), -V_approxes_sem(samplesize_idx,1), V_approxes_sem(samplesize_idx,1), -R_approxes_sem(samplesize_idx,1), R_approxes_sem(samplesize_idx,1),".",'Color', cmap(color_idx,:), 'Capsize',0,"MarkerSize",15)
                        else
                            errorbar(r_approxes_avg,(v_approxes_avg-v_corresponding_optimal), -V_approxes_sem(samplesize_idx,:), V_approxes_sem(samplesize_idx,:), -R_approxes_sem(samplesize_idx,:), R_approxes_sem(samplesize_idx,:),".:",'Color', cmap(color_idx,:), 'Capsize',0,"MarkerSize",5)
                        end
                        ylab = "Reduction in trial-averaged reward";
                        title_lab = "Loss in t";
                        plot([min(R),max(R)],[0,0],"k:","HandleVisibility","off")
                        if(task_name=="exp0_sparseQ")
                            ymin=-0.65;
                            ymax = 0.02;
                            yticks([-0.6:0.2:0])
                        else
                            ymin=-0.3;
                            ymax = 0.02;
                            yticks([-0.3:0.1:0])
                        end
                end
                if(sample_with_replacement)
                    title("$n = "+Num_action_samples(samplesize_idx)+"$", "interpreter","latex")
                else
                    title("$N_a = "+Num_action_samples(samplesize_idx)+"$", "interpreter","latex")
                end
                ylim([ymin,ymax])
                xlim([0,Inf])
        
                if(samplesize_idx==1 && model==models_plotted(end))
                    ylabel(plot_title, "fontsize",9)
                    if(approximation_method==1)
                        %legend(proposal_distribution_latexes(models_plotted), 'interpreter','latex', 'fontsize',8)
                        leg=legend({"Flat","V(a)","Oracle"}, 'fontsize',8, "location","north");
                        title(leg,'Proposal distr.')
                        set(leg.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[1;1;1;.5]));
                    end
                end
                if(samplesize_idx>1)
                    yticklabels([])
                end
            end
            color_idx = color_idx+1;
        end
    end
    ylabel(t,ylab)
    xlabel(t,"Policy complexity (bits)")
    end