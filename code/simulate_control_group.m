clear all

%% Specify models to use to generate synthetic datasets:
models = {
        % % Non-hierachical
        % {'IP2' 'no_learning'};
        % {'IP2' 'pS' 'no_learning'};
        % {'IP2' 'etaA'};
        % {'IP2' 'pS' 'etaA'};       

        % Hierarchical
        % {'IP1' 'IP2' 'no_learning'};
        % {'IP1' 'IP2' 'pS' 'no_learning'};
        % {'IP1' 'IP2'};
        {'IP1' 'IP2' 'pS'};                 % model 8

        % Extensions of model 8
        % {'IP1' 'IP2' 'pS' 'omega'};
        % {'IP1' 'IP2' 'pS' 'omegaBlock'};
        % {'IP1' 'IP2' 'pS' 'IP1Diff'};
        % {'IP1' 'IP2' 'pS' 'etaD'};
        % {'IP1' 'IP2' 'pS' 'zeta'};
        };


for j = 1:length(models)

    rng("default")
    
    %% Prepare directories for best-fit parameter input and synthetic dataset output
    model_name = strjoin(models{j},'_');
    
    % Task data folder
    data_dir = fullfile('..', 'data', 'assessment', 'control');
    
    for i = 1:26   
        subjects{i} = ['sub' sprintf('%01d', i)];
    end
    
    % Directories
    model_fits_dir = fullfile('..', 'results', 'model_fits_assessment', model_name, 'control', 'all'); % Parameter estimates using concatenated assessment trials from control group
    output_dir = fullfile('..', 'results', 'identifiability', model_name, 'sim_data_control_group');

    % Make output directory if needed
    if (~exist(output_dir)); mkdir(output_dir); end
    
    %% Simulate data using specified model
    %------------------------
    outputs = cell(length(subjects),1);
    
    for i = 1:length(subjects)
      
        % Simulate data for this participant
        MDP_sim = simulate_MDP(subjects{i}, data_dir, model_fits_dir);
        
        % Save output in .mat file
        save([output_dir '/' subjects{i} '.mat'],'MDP_sim')

    end

end

function MDP_sim = simulate_MDP(subject, data_dir, model_fits_dir)
% This function takes an MDP object with best-fit parameter values from
% each participant and their trial-by-trial observations, to simulate
% posteriors on each trial, which are then used to sample task responses.
% Output is a new MDP object with solved posteriors and simulated responses.

%% Import data, get observations & actions

file = [data_dir '/' subject '.csv'];
disp(file)
rawdat = readtable(file); %subject data

ntrials = size(rawdat,1); % up to 320 trials, depends on participant. 
observations = rawdat.inSync(1:ntrials) + 2;   % Recode from 0-1 to 2-3
actions      = rawdat.response(1:ntrials) + 1; % Recode from 0-1 to 1-2

is_exercise = rawdat.isFeedback;
correct = rawdat.response == rawdat.inSync;
response_acc = mean(correct);

%% Load in the fitted DCM.MDP, collect best-fit param values

load([model_fits_dir '/' subject '.out.mat']) % Best fits are stored in a variable called output; we overwrite this variable at the bottom

posteriors = output{1, 2};
params = output{1, 2}.Properties.VariableNames;
disp('Parameters estimated:')
disp(params)

mdp = output{1, 3}.MDP;


% For loop to set the value of each estimated param to the posterior value
% for this participant:
for i = 1:width(params)
    %disp(mdp.(char(params(i))))
    mdp.(char(params(i))) = posteriors.(char(params(i)));
    %disp(mdp.(char(params(i)))) 
    % Uncomment to check that values are copied over
end

%% Specify a new MDP, feeding in estimated parameter values

MDP = HDT_model(mdp, mdp.fit_options); % Specify new model using posterior parameter values

MDP.fit_options = mdp.fit_options;
MDP.is_exercise = rawdat.isFeedback; % Column was mis-named in data files.

% Simulation flag for VB_X script
MDP.sim = 1;

% Feed in observations on each trial into MDP.o
[MDP(1:mdp.N_trials)] = deal(MDP);
o_all = ones(mdp.N_trials, 3); % 3 timesteps
                           
for tr = 1:mdp.N_trials
    o_all(tr, 2) = observations(tr);
    if mdp.fit_options.is_feedback
        o_all(tr, 3) = observations(tr);
    end
end

for tr = 1:mdp.N_trials
    MDP(tr).o = o_all(tr,:);
end

%% Collect session number (for model assuming forgetting between sessions)
%----------------------------------------------------------------------
for trial = 1:length(MDP)
    MDP(trial).block = rawdat.day(trial);
end

%% Implement IP1Diff (increased lower-level precision during exercise trials)
%----------------------------------------------------------------------
if isfield(mdp.fit_options,'IP1Diff')
    if mdp.fit_options.IP1Diff == 1
        ex_precision = mdp.IP1 + mdp.IP1Diff;
        ex_precision = min(ex_precision,1); % Cap the exercise precision to 1
        for idx_trial = 1:size(MDP,2)
            if MDP(1).is_exercise(idx_trial) == 1
                                            % Start  Async              Sync 
                MDP(idx_trial).MDP.a{1} =   [1      0                   0;                      % Start
                                             0      ex_precision        1 - ex_precision;       % Async
                                             0      1 - ex_precision    ex_precision] * 1000 ;  % Sync      
            end
        end
    end
end

%% Solve for simulated posteriors

MDP_sim = spm_MDP_VB_X_HDT(MDP);

%% Generate simulated responses using posteriors
% Using simple response model of sampling randomly from posterior at t = 2.

for trial = 1:length(MDP_sim)
    sim_posteriors(trial, :) = MDP_sim(trial).xn{1,1}(16,:,2,2);
    sim_actions(trial) = find(rand < cumsum(sim_posteriors(trial,:)),1);
    MDP_sim(trial).sim_action = sim_actions(trial);
end

end
