clear all

%% Specify models to perform identifiability analysis for:
field = {
        % % Non-hierachical
        {'IP2' 'no_learning'};
        {'IP2' 'pS' 'no_learning'};
        {'IP2' 'etaA'};
        {'IP2' 'pS' 'etaA'};       

        % Hierarchical
        {'IP1' 'IP2' 'no_learning'};
        {'IP1' 'IP2' 'pS' 'no_learning'};
        {'IP1' 'IP2'};
        {'IP1' 'IP2' 'pS'};                 % model 8

        % Extensions of model 8
        {'IP1' 'IP2' 'pS' 'omega'};
        {'IP1' 'IP2' 'pS' 'omegaBlock'};
        {'IP1' 'IP2' 'pS' 'IP1Diff'};
        {'IP1' 'IP2' 'pS' 'etaD'};
        {'IP1' 'IP2' 'pS' 'zeta'};
        };


%% Set up directories

% Task data folder
data_dir = fullfile('..', filesep, 'data', 'processed');

% Subject data filenames
for i = 1:30
    subjects{i} = ['sub' sprintf('%02d', i)];
end

% Skip sub13 and sub23, who have no data (withdrawn)
subjects(contains(subjects, 'sub13')) = [];
subjects(contains(subjects, 'sub23')) = [];


for j = 1:length(field) % For each synthetic dataset

    model_names{j} = strjoin(field{j},'_');

    % Synthetic dataset to use for second-order model fitting   
    sim_dataset = model_names{j};
    sim_dir{j} = fullfile('..', 'results', 'identifiability', sim_dataset, 'sim_data');
    
    % Results folders 
    results_dir{j} = [fullfile('..', filesep, 'results', 'identifiability') '\' sim_dataset '\model_fits'];
        
end
    
%% Set up each model's fit options

for model = 1:length(field)
    
    % Default settings
    fit_options(model).is_feedback = 1; % Feedback is given at timestep 3
    fit_options(model).is_hier = 0;     
    fit_options(model).A_learning = 1;  % 0 = no learning in 'A' matrix, 1 = learning in 'A' matrix
    fit_options(model).D_learning = 0;  % 0 = no learning in 'D' matrix, 1 = learning in 'D' matrix

    % Hierarchical model and IP1
    if any(strcmp(field{model}, 'IP1')); fit_options(model).is_hier = 1; end

    % Supplemental models:
    if any(strcmp(field{model}, 'etaD')); fit_options(model).D_learning = 1; end
    if any(strcmp(field{model}, 'omegaBlock')); fit_options(model).forgettingBlock = 1; end
    if any(strcmp(field{model}, 'zeta')); fit_options(model).zeta = 1; end
    if any(strcmp(field{model}, 'IP1Diff')); fit_options(model).IP1Diff = 1; end

    % Models assuming no learning:
    if any(strcmp(field{model}, 'no_learning')) 
        fit_options(model).A_learning = 0; 
        field{model}(strcmp(field{model}, 'no_learning')) = []; % delete the 'no_learning' element
    end

end
    
    
%% Fit models
for j = 1:length(field) % Outer loop: for each synthetic dataset
    for model = 1:length(field)  % Inner loop: For each model, index for field, fit_options, and results_dir

        % Set up results sub-folder for this model
        out_dir = fullfile(results_dir{j}, model_names{model});
        % Make folder if it doesn't already exist
        if (~exist(out_dir)); mkdir(out_dir); end
        
        for i = 1:length(subjects) % To use parallel processing, change to parfor
            tic
            rng("default")
            HDT_batch_fit(subjects{i}, field{model}, fit_options(model), data_dir, out_dir, sim_dir{j});
            toc
        end    

        % Save parameter estimates in combine_fits.csv for each model
        combine_fits(out_dir);

    end       
end

function output = HDT_batch_fit(subject, field, fit_options, data_dir, results_dir, sim_dir)

%% Import data, get observations & actions

file = [data_dir '\' subject '.csv'];
disp(file)

% Skip if data file does not exist
try
    rawdat = readtable(file); % subject data
catch
    return
end

% Skip if results file already exists:
if isfile([results_dir '\' subject '.out.mat'])
    return
end

ntrials = size(rawdat,1); % up to 320 trials, depends on participant. 

% Recode observations (2 = Async, 3 = Sync)
observations = rawdat.inSync(1:ntrials) + 2;   % Recode from 0-1 to 2-3

% Recode responses/actions (1 = Sync, 2 = Async)
actions      = rawdat.response(1:ntrials) + 1; % Recode from 0-1 to 1-2

% Rename mis-named column
is_exercise = rawdat.isFeedback;

correct = rawdat.inSync == rawdat.response;

% Actions are coded 1-2 because the relevant state factor (Async vs. Sync)
% has two possible states. This variable is eventually used to index for
% the posterior belief (.xn) over this state factor to calculate log
% evidence for this model in HDT_invert.m.

%% Load in the simulated data MDP, collect simulated actions

load([sim_dir '/' subject '.mat']) % up to 320 trials of a solved MDP given param values and observations on each trial

sim_actions = [MDP_sim.sim_action]';

%% Parameter estimation priors

% Default values
priors = struct(...
    'IP1',        .750 ,... % Lower level precision: set to middle of expected range
    'IP2',        .750 ,... % Higher level precision: set to middle of expected range
    'pS',         .50,  ... % Initial state bias: set to middle of expected range
    'etaA',       1,    ... % Single learning rate for A matrix
    'etaD',       1,    ... % single learning rate for D matrix
    'omega',      1,    ... % Forgetting by trial: fixed to 'off'
    'omegaBlock', 1,    ... % Forgetting by block: fixed to 'off'
    'IP1Diff',    0,    ... % Increased lower-level precision during exercise: fixed to 'off'
    'zeta',       1     ... % 'Faulty memory' mechanism: fixed 'off'
);


% Learning
if any(strcmp(field,'etaA'))  
    priors.etaA = .5; else end 
if any(strcmp(field,'etaD'))
    priors.etaD = .5; else end 

% Forgetting and 'faulty memory'
if any(strcmp(field,'omega')) 
    priors.omega = .75; else end 
if any(strcmp(field,'omegaBlock')) 
    priors.omegaBlock = .5; else end 
if any(strcmp(field,'zeta')) 
    priors.zeta = .5; else end

% Exercise effects
if any(strcmp(field,'IP1Diff')) 
    priors.IP1Diff = .25; else end

disp(priors)

%% Define mdp for inversion 

mdp = HDT_model(priors, fit_options); % lowercase mdp for model inversion
mdp.fit_options = fit_options;
mdp.params = field; % needed for log-likelihood function
mdp.N_trials = ntrials; % trials per block

mdp.is_exercise = is_exercise;

% Collect participant observations (i.e., trial condition)
o_all = ones(mdp.N_trials, 3); % 3 if including feedback timestep                           
for tr = 1:mdp.N_trials
    o_all(tr, 2) = observations(tr);
    if fit_options.is_feedback 
        o_all(tr, 3) = observations(tr);
    end
end

%% Use simulated task responses from synthetic dataset

% mdp.action   = actions; % actual participant's responses 

mdp.action   = sim_actions; % Simulated responses

% Collect session (for forgetting by block model)
mdp.block = rawdat.day;

%% Define DCM for inversion
%--------------------------------------------------------------------------

DCM.MDP = mdp; % Lowercase 1x1 struct for inversion
DCM.field = field;

DCM.U     = num2cell(o_all, 2)';        % Invert script requires 1xN_trial cell (with 1x2 array inside each cell)
                                 
DCM.Y     = repmat({1}, 1, ntrials);    % Invert script requires 1xN_trial cell (1 integer inside each cell)

%% Model inversion (variational inference)
%--------------------------------------------------------------------------

DCM_fit = HDT_invert(DCM); % 

%--------------------------------------------------------------------------
% re-transform values and compare prior with posterior estimates
%--------------------------------------------------------------------------
field = fieldnames(DCM_fit.M.pE);

prior = zeros(1, size(field, 1));
posterior = zeros(1, size(field, 1));

for i = 1:length(field)
    if strcmp(field{i},'IP1')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i}))); 
    elseif strcmp(field{i},'IP2')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i}))); 
    elseif strcmp(field{i},'pS')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i})));
    elseif strcmp(field{i},'etaA')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i}))); 
    elseif strcmp(field{i},'etaD')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i}))); 
    elseif strcmp(field{i},'omega')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i}))); 
    elseif strcmp(field{i},'omegaBlock')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i})));      
    elseif strcmp(field{i},'IP1Diff')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i}))); 
    elseif strcmp(field{i},'zeta')
        prior(i) = 1/(1+exp(-DCM_fit.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM_fit.Ep.(field{i})));
    else
        prior(i) = exp(DCM_fit.M.pE.(field{i}));
        posterior(i) = exp(DCM_fit.Ep.(field{i}));
    end
end
     

prior       = array2table(prior,        'VariableNames', field);
posterior   = array2table(posterior,    'VariableNames', field);

output = {prior posterior DCM_fit DCM};

%% Save output
save([results_dir '/' subject '.out.mat'], 'output');

end

function tab = combine_fits(DIR)

files = dir([DIR '/*.mat']);

tab = [];

for i = 1:numel(files)
    data = load([DIR '/' files(i).name]);
    
    params = data.output{1,2};    
    sub = files(i).name(1:end-4);
    F = data.output{1,3}.F;
    
    temp_tab = [table({sub}) params table(F)];
    
    tab = [tab; temp_tab];
end

writetable(tab,[DIR '/' 'combine_fits.csv']) % Write to csv

clear temp_tab files DIR data params sub LL i
end
