% This script calculates, for each participant, the proportion of trials in
% which the simulated participant response ('in sync' or 'out of sync')
% corresponded to the participant's actual response during the task. The
% average probability (within the model) for emitting the participant's
% actual response across trials is also calculated.

clear variables

%% Directories

model = 'IP1_IP2_pS';

% Task data folder
data_dir = fullfile('..', 'data', 'processed');

% Use the synthetic dataset:
sim_dir = fullfile('..', 'results', 'identifiability', model, 'sim_data');

% Subject data filenames
for i = 1:30
    subjects{i} = ['sub' sprintf('%02d', i)];
end

% Skip sub13 and sub23, who have no data (withdrawn)
subjects(contains(subjects, 'sub13')) = [];
subjects(contains(subjects, 'sub23')) = [];

accuracies = zeros(1,length(subjects));
mean_action_probs = zeros(1,length(subjects));

for i = 1:length(subjects)
    file = [data_dir '/' subjects{i} '.csv'];
    disp(file)

    % Participant responses
    rawdat = readtable(file); %subject data; 0 = 'out of sync', 1 = 'in sync'
    real_actions = rawdat.response + 1;

    % Simulated responses from the model
    load([sim_dir '/' subjects{i} '.mat']) % up to 320 trials of a solved MDP given param values and observations on each trial
    sim_actions = [MDP_sim.sim_action]'; % 1 = 'out of sync', 2 = 'out in sync'
    
    % Append model's accuracy in reproducing participant responses
    accuracies(i) = mean(real_actions == sim_actions);

    % Append average probability of emitting the participant response on
    % each trial (taken to be the posterior over states following 
    % stimulus presentation)
    for trial = 1:length(MDP_sim)
        sim_posteriors(trial, :) = MDP_sim(trial).xn{1,1}(16,:,2,2);
        action = rawdat.response(trial) + 1;
        action_prob(trial) = sim_posteriors(trial,action);
    end
    mean_action_probs(i) = mean(action_prob);

    
    session = str2double(string(rawdat.session));

end

T = table(subjects', accuracies', mean_action_probs', ...
    'VariableNames', {'participant', 'accuracy', 'mean_action_prob'});

disp(T);

disp(mean(accuracies));
disp(mean(mean_action_probs));


