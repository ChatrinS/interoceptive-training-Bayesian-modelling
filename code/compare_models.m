clear all

%% Specify model comparison or identifiability analysis

task = 1; % 1 = model comparison; 2 = identifiability

% If identifiability analysis, specify which model is providing the
% synthetic dataset:
if task == 2
    ident_model = 8;   
end

%% List of models to compare
field = {
        % Non-hierachical
        % {'IP2' 'no_learning'};
        % {'IP2' 'pS' 'no_learning'};
        % {'IP2' 'etaA'};
        % {'IP2' 'pS' 'etaA'};       

        % Hierarchical
        {'IP1' 'IP2' 'no_learning'};
        {'IP1' 'IP2' 'pS' 'no_learning'};
        {'IP1' 'IP2'};
        {'IP1' 'IP2' 'pS'};                 % model 8

        % Extensions of model 8
        % {'IP1' 'IP2' 'pS' 'omega'};
        % {'IP1' 'IP2' 'pS' 'omegaBlock'};    % model 10
        {'IP1' 'IP2' 'pS' 'IP1Diff'};
        {'IP1' 'IP2' 'pS' 'etaD'};
        % {'IP1' 'IP2' 'pS' 'zeta'};
        };

if task == 1
    fprintf('Bayesian model comparison including %01d models.  \n', length(field));
elseif task == 2
    ident_model = strjoin(field{ident_model},'_');
    fprintf('Identifiability for model %s.  \n', ident_model);
end

%% Set up directories

% Results folders 
if task == 1 % Model comparison

    temp_dir = fullfile('..', 'results', 'model_fits');
    
    % Sub-folders for each model fitted
    for model = 1:length(field)
        model_names{model} = strjoin(field{model},'_');
        results_dir{model} = [temp_dir '\' model_names{model}];
    end

elseif task == 2 % Identifiability

    temp_dir = fullfile('..', 'results', 'identifiability', ident_model, 'model_fits'); % Change back to 'identifiability'
    
    % Sub-folders for each model fitted
    for model = 1:length(field)
        model_names{model} = strjoin(field{model},'_');
        results_dir{model} = [temp_dir '\' model_names{model}];
    end
end

clear temp_dir

%% Read model fits from .csv files

% Initialise array to collect results across models
A = zeros(28, length(results_dir));

for i = 1:length(results_dir) % Loop over results folders for all models included
    
    filename = fullfile(results_dir{i}, 'combine_fits.csv');
    tab_temp = readtable(filename);
    A(:, i) = tab_temp.F;

    clear tab_temp
end

tab = array2table(A, 'VariableNames', model_names);

[alpha,exp_r,xp,pxp,bor] = spm_BMS(table2array(tab));


% Random effects Bayesian model selection
[~,REWinner] = max(pxp); 
fprintf('Model %s wins random-effects Bayesian model selection.\npxp = %.3f \n', model_names{REWinner}, pxp(REWinner));

% Fixed effects Bayesian model selection
columnMeans = mean(table2array(tab), 1);
[~,FEWinner] = max(columnMeans); % fixed effects winning model
fprintf('Model %s wins fixed-effects Bayesian model selection.\nMean F = %.3f \n', model_names{FEWinner}, columnMeans(FEWinner));
