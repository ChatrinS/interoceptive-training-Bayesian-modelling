function [DCM] = HDT_invert(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)

% If simulating - comment out section on line 196
% If not simulating - specify subject data file in this section 

%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = 2^-1;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if strcmp(field,'IP1')
            pE.(field) = log(DCM.MDP.IP1/(1-DCM.MDP.IP1 + eps) + eps);  % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;          
        elseif strcmp(field,'IP2')          
            pE.(field) = log(DCM.MDP.IP2/(1-DCM.MDP.IP2 + eps) + eps);                   
            pC{i,i}    = prior_variance;          
        elseif strcmp(field,'pS')          
            pE.(field) = log(DCM.MDP.pS/(1-DCM.MDP.pS + eps) + eps);                     
            pC{i,i}    = prior_variance;               
        elseif strcmp(field,'etaA')                                            
            pE.(field) = log(DCM.MDP.etaA/(1-DCM.MDP.etaA + eps) + eps);               
            pC{i,i}    = prior_variance;        
        elseif strcmp(field,'etaD')                                                    
            pE.(field) = log(DCM.MDP.etaD/(1-DCM.MDP.etaD + eps) + eps);              
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'omega')          
            pE.(field) = log(DCM.MDP.omega/(1-DCM.MDP.omega + eps) + eps);              
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'omegaBlock')
            pE.(field) = log(DCM.MDP.omegaBlock/(1-DCM.MDP.omegaBlock + eps) + eps);    
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'IP1Diff')
            pE.(field) = log(DCM.MDP.IP1Diff/(1-DCM.MDP.IP1Diff + eps) + eps);        
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'zeta')
            pE.(field) = log(DCM.MDP.zeta/(1-DCM.MDP.zeta + eps) + eps);               
            pC{i,i}    = prior_variance;
        else
            pE.(field) = 0;      
            pC{i,i}    = prior_variance;
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.mdp   = DCM.MDP;                       % MDP structure

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidnce (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return

function L = spm_mdp_L(P,M,U,Y)
%% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% reconvert parameters in MDP to original value space
%--------------------------------------------------------------------------
mdp   = M.mdp;
field = fieldnames(M.pE);
for i = 1:length(field)
    if strcmp(field{i},'IP1')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'IP2')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'pS')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'etaA')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i}))); 
    elseif strcmp(field{i},'etaD')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));  
    elseif strcmp(field{i},'omega')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'omegaBlock')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'IP1Diff')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'zeta')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    else        
        mdp.(field{i}) = exp(P.(field{i}));
    end
end


% discern whether learning is enabled - and identify unique trials if not
%--------------------------------------------------------------------------
if any(ismember(fieldnames(mdp),{'a','b','d','c','d','e'}))
    j = 1:numel(U);
    k = 1:numel(U);
else
    % find unique trials (up until the last outcome)
    %----------------------------------------------------------------------
    u       = spm_cat(U');
    [i,j,k] = unique(u(:,1:(end - 1)),'rows');
end

L = 0;

mdp_temp = HDT_model(mdp, mdp.fit_options); 

mdp_temp.fit_options = mdp.fit_options; % VB_X function needs to check fit_options.A_learning

mdp_temp.is_exercise = mdp.is_exercise;
mdp_temp.block = mdp.block;

[MDP(1:mdp.N_trials)]   = deal(mdp_temp);

for idx_trial = 1:size(MDP,2)
    MDP(idx_trial).o = [U{idx_trial}];
    MDP(idx_trial).u = [1;
                        1];        
    MDP(idx_trial).action = mdp.action(idx_trial);      
    MDP(idx_trial).block = mdp.block(idx_trial);        % add session/block number
    
end

%% Implement IP1Diff (increased lower level precision during exercise trials)
%----------------------------------------------------------------------
if isfield(mdp.fit_options,'IP1Diff')
    if mdp.fit_options.IP1Diff == 1
        ex_precision = mdp.IP1 + mdp.IP1Diff;
        ex_precision = min(ex_precision,1); % Cap the exercise precision to 1
        for idx_trial = 1:size(MDP,2)
            if MDP(1).is_exercise(idx_trial) == 1
                                            % Start  Async              Sync 
                MDP(idx_trial).MDP.a{1} =   [1      0                   0;                % Start
                                             0      ex_precision        1 - ex_precision;   % Async
                                             0      1 - ex_precision    ex_precision] * 1000 ;     % Sync      
            end
        end
    end
end

%% solve MDP and accumulate log-likelihood
%--------------------------------------------------------------------------

MDP  = spm_MDP_VB_X_HDT(MDP);                                                                                    

for j = 1:mdp.N_trials
        L = L + log(MDP(j).xn{1,1}(16, MDP(j).action, 2, 2) + eps);
end

% Calculate model percentage accuracy and average probability assigned to participant's action 
%--------------------------------------------------------------------------

actions = ones(mdp.N_trials,1);
for j = 1:mdp.N_trials
    actions(j) = MDP(j).action;
end

% Generate simulated responses using posteriors
sim_actions = ones(mdp.N_trials,1);

for trial = 1:length(MDP)
    sim_posteriors(trial, :) = MDP(trial).xn{1,1}(16,:,2,2);
    sim_actions(trial) = find(rand < cumsum(sim_posteriors(trial,:)),1);
end

percent_correct = mean(actions == sim_actions);

observations = cell2mat(U)';
observations = observations(2:3:end);

% Also note subject's model-free accuracy
subj_correct = mean(observations == actions+1);

% print iteration, model accuracy, subject's model-free accuracy,
%--------------------------------------------------------------------------
out_string = '';
for idx = 1:length(mdp.params)
    field = mdp.params(idx);
    value = getfield(mdp, char(field));
    string = sprintf('%s: %.4f', char(field), value);
    if idx == 1
        out_string = [out_string string];
    else
        out_string = [out_string ', ' string];
    end
end

fprintf(out_string)
fprintf(sprintf(' || %.2f model accuracy', percent_correct))
fprintf(', ')

fprintf(sprintf('%.2f subject accuracy', subj_correct))
fprintf(' || ')

clear('MDP')
clear('actions')
clear('sim_actions')
clear('percent_correct')

fprintf('LL: %f \n',L)

