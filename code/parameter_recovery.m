clear all

% Specify which model to assess parameter recovery for
model = 8;

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
        {'IP1' 'IP2' 'pS'}; % model 8

        % Extensions of model 8
        {'IP1' 'IP2' 'pS' 'omega'};
        {'IP1' 'IP2' 'pS' 'omegaBlock'};
        {'IP1' 'IP2' 'pS' 'IP1Diff'};
        {'IP1' 'IP2' 'pS' 'etaD'};
        {'IP1' 'IP2' 'pS' 'zeta'};
        };

model_name = strjoin(field{model},'_');

fprintf('Parameter recovery for model %01d,  %s', model, model_name);

% Directories
generative_dir   = fullfile('..', 'results', 'model_fits', model_name, 'combine_fits.csv');
estimated_dir    = fullfile('..', 'results', 'identifiability', model_name, 'model_fits', model_name, 'combine_fits.csv');


%% Read in generative (first-order) and estimated (second-order) parameter estimates

generative_params   = readtable(generative_dir);
estimated_params    = readtable(estimated_dir);

generative_params   = removevars(generative_params, {'Var1', 'F'});
estimated_params    = removevars(estimated_params, {'Var1', 'F'});



results = zeros(width(generative_params),3);

for i = 1:width(generative_params)
    % Extract the last column of each table
    column1 = generative_params{:, i};
    column2 = estimated_params{:, i};
    
    % Calculate the Pearson correlation coefficient
    [R,P] = corrcoef(column1, column2);
    
    % Calculate the degrees of freedom
    n = numel(column1);
    degrees_of_freedom = n - 2;
    results(i, :) = [R(2,1), P(2,1), degrees_of_freedom];

end

results = array2table(results, 'VariableNames', {'Pearson r', 'p', 'df'});

% Append parameter names
paramNames = table(generative_params.Properties.VariableNames(:), 'VariableNames', {'Parameter'});

% Concatenate the new column with the existing table
results = [paramNames results];

results