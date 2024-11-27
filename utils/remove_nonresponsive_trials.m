function [data] = remove_nonresponsive_trials(data, remove_nonresponsive)
% If remove_nonresponsive = true, remove these trials.
% If = false, rewrite NaNs into -1.
    if(nargin==1)
        remove_nonresponsive=true;
    end
    
    n_subj = length(data);
    for s=1:n_subj
        if(remove_nonresponsive)
        valid_trials = find(~isnan(data(s).a));
        data(s).cond = data(s).cond(valid_trials);
        data(s).s = data(s).s(valid_trials);
        data(s).a = data(s).a(valid_trials);
        data(s).corrchoice = data(s).corrchoice(valid_trials);
        data(s).acc = data(s).acc(valid_trials);
        data(s).r = data(s).r(valid_trials);
        data(s).rt = data(s).rt(valid_trials);
        data(s).tt = data(s).tt(valid_trials);
        data(s).block = data(s).block(valid_trials);
        if(isfield(data(s),"universally_correct_action"))
            data(s).universally_correct_action = data(s).universally_correct_action(valid_trials);
        end
        else
            invalid_trials = find(isnan(data(s).a));
            data(s).a(invalid_trials) = -1;
        end

    end

end