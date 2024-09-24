clear all

%% Specify group and session to fit
% This script is not yet automated to fit all the assessment session data
% at once. For now please specify which group and which session you wish to
% fit models onto.

group = 'control';  % 'training' or 'control'
session = 'all';   % 'bl' for baseline timepoint
                    % 'MA' for middle timepoint
                    % 'exit' for final timepoint
                    % 'all' for all trials concatenated across timepoints
                    % (control group only)

%% Specify models to fit
% In Suksasilp et al. (2024), only model 8 was fit to the assessment data.

field = {
        % % Non-hierachical
        % {'IP2' 'no_learning'};
        % {'IP2' 'pS' 'no_learning'};
        % {'IP2' 'etaA'};
        % {'IP2' 'pS' 'etaA'};       

        % Hierarchical
        % {'IP1' 'IP2' 'no_learning'};
        % {'IP1' 'IP2' 'pS' 'no_learning'};
        % {'IP1' 'IP2'};
        {'IP1' 'IP2' 'pS'}; % model 8

        % Extensions of model 8
        % {'IP1' 'IP2' 'pS' 'omega'};
        % {'IP1' 'IP2' 'pS' 'omegaBlock'};
        % {'IP1' 'IP2' 'pS' 'IP1Diff'};
        % {'IP1' 'IP2' 'pS' 'etaD'};
        % {'IP1' 'IP2' 'pS' 'zeta'};
        };

%% Set up directories

% Task data folder
data_dir = fullfile('..', filesep, 'data', 'assessment');

% Results folders 
temp_dir = fullfile('..', filesep, 'results', 'model_fits_assessment');

% Sub-folders for each model to be fit
for model = 1:length(field)
    suffix = strjoin(field{model},'_');
    results_dir{model} = [temp_dir '\' suffix '\' group '\' session ];
    % Make results directory if needed
    if (~exist(results_dir{model})); mkdir(results_dir{model}); end
end

clear batch_dir suffix

% Subject data filenames
if strcmp(group,'training')
    for i = 1:30
        subjects{i} = ['sub' sprintf('%02d', i)];
    end
elseif strcmp(group,'control')
    for i = 1:26
        subjects{i} = ['sub' sprintf('%01d', i)];
    end
else
    disp('group not correctly specified')
end


%% Fit options corresponding to model

for model = 1:length(field)
    
    % Default settings
    fit_options(model).is_feedback = 0; % Feedback is off for assessment trials
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
% For each model, index for field, fit_options, and results_dir

for model = 1:length(field)
    parfor i = 1:length(subjects) % To use parallel processing, change to parfor
        tic
        rng("default")
        HDT_batch_fit(subjects{i}, field{model}, fit_options(model), data_dir, results_dir{model}, group, session);
        toc
    end
    % Save parameter estimates in combine_fits.csv for each model
    combine_fits(results_dir{model});
end



function output = HDT_batch_fit(subject, field, fit_options, data_dir, results_dir, group, session)

%% Import data, get observations & actions

file = [data_dir '\' group '\' subject '.csv'];
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

% Filter trials for the specified assessment session:
if strcmp(session, 'all')
else
    rawdat = rawdat(strcmp(rawdat.session, session),:);
end

% Skip if subject has no data for the specified assessment session
if height(rawdat) == 0 
    output = 0;
    return
end

ntrials = size(rawdat,1); % up to 320 trials, depends on participant. 

% Recode observations (2 = Async, 3 = Sync)
observations = rawdat.inSync(1:ntrials) + 2;   % Recode from 0-1 to 2-3

% Recode responses/actions (1 = Sync, 2 = Async)
% Actions are coded 1-2 because the relevant state factor (Async vs. Sync)
% has two possible states. This variable is eventually used to index for
% the posterior belief (.xn) over this state factor to calculate log
% evidence for this model in HDT_invert.m.
actions      = rawdat.response(1:ntrials) + 1; % Recode from 0-1 to 1-2

% Rename mis-named column in datafile
is_exercise = rawdat.isFeedback;

% Score correct/incorrect responses on each trial
correct = rawdat.inSync == rawdat.response;



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

% Collect participant actions (i.e., responses on each trial)
mdp.action   = actions;

% Collect session (for forgetting by block)
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
