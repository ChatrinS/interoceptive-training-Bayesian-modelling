clear variables

model_name = 'IP1_IP2_pS';

group = 1; % 1 = Training group, 2 = Control group

if group == 1
    outFileName = 'end_precision_TRAINING.csv';
    sim_dir = fullfile('..', 'results', 'identifiability', model_name, 'sim_data');
elseif group == 2
    outFileName = 'end_precision_CONTROL.csv';
    sim_dir = fullfile('..', 'results', 'identifiability', model_name, 'sim_data_control_group');;
end

% Get a list of all .mat files in the directory
matFiles = dir(fullfile(sim_dir, '*.mat'));

% Initialize a cell array to store results files from model fitting
results = cell(1, numel(matFiles));

% Loop through each .mat file and load it into the cell array
for i = 1:numel(matFiles)
    % Construct the full file path
    filePath = fullfile(sim_dir, matFiles(i).name);
    
    % Load the .mat file into a variable
    results{i} = load(filePath);
end

% Display a message indicating the completion of the loading process
disp('All .mat files loaded successfully.');

subjects = {matFiles.name}.';
subjects = strrep(subjects, '.mat', '');


% Initialize cell arrays for each table column
participantCell = cell(numel(results), 1);
IP1Cell = cell(numel(results), 1);
IP2Cell = cell(numel(results), 1);
IP2EndCell = cell(numel(results), 1);
IP2ChangeCell = cell(numel(results), 1);

% Loop over each simulated MDP
for i = 1:numel(results)   
    IP1 = results{i}.MDP_sim(1).IP1; 
    IP2 = results{i}.MDP_sim(1).IP2; 
    
    a_end = results{i}.MDP_sim(end).a{1}(:,:,2);
    a_end = spm_norm(a_end);
    IP2_end = mean([a_end(2,1), a_end(3,2)]);

    % Fill in the corresponding cells in each cell array
    participantCell{i} = subjects{i};
    IP1Cell{i} = IP1;
    IP2Cell{i} = IP2;
    IP2EndCell{i} = IP2_end;
    IP2ChangeCell{i} = IP2_end - IP2; 
end


% Append cell arrays horizontally
dataTable = [participantCell, IP1Cell, IP2Cell, IP2EndCell, IP2ChangeCell];

% Convert the cell array to a table
dataTable = cell2table(dataTable, 'VariableNames', {'Participant', 'IP1', 'IP2', 'IP2_end', 'IP2_change'});

% Create a table using the cell arrays
% dataTable = table(participantCell, IP1Cell, IP2Cell, IP2EndCell, IP2ChangeCell, 'VariableNames', {'Participant', 'IP1', 'IP2', 'IP2_end', 'IP2_change'});

% Display the resulting table
disp(dataTable);

% Save the table to a CSV file
outPath = fullfile('..', 'results', outFileName);
writetable(dataTable, outPath);


% auxillary functions
%==========================================================================

function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);
end