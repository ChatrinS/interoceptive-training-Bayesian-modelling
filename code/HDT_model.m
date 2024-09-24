function mdp = HDT_model(params, fit_options)

% Function for setting up a hierarchical generative model of a heartbeat
% discrimination task (HDT). Allows for noisy cardiac sensory signals by
% including a hierarchical lower level with variable likelihood precision.

% Learning can be enabled for the (higher-level) A matrix and D matrix.


%% Level 1: Perception of individual stimuli
%==========================================================================

if fit_options.is_hier == 1 % If using hierarchical levels

    % prior beliefs about initial states
    %--------------------------------------------------------------------------
    
    D{1} = [1 1 1]'; % trial condition {start, async, sync}
    
    % probabilistic (likelihood) mapping from hidden states to outcomes: A
    %--------------------------------------------------------------------------
    
    % outcome modality 1: start/async/sync
    
    A{1} = [1 0 0;
            0 1 0;
            0 0 1];
    
         % Start  Async           Sync 
    a{1}= [1      0               0;                % Start
           0      params.IP1      1 - params.IP1;   % Async
           0      1 - params.IP1  params.IP1] ;     % Sync
    
    % No lower-level learning: multiply a{1} by arbitrarily large number
    a{1} = a{1} * 1000;

    % Transitions between states: B
    %--------------------------------------------------------------------------
        % Start    Async    Sync 
    B{1}= [1       0        0;   % Start
           0       1        0;   % Async
           0       0        1];  % Sync
    
    % MDP Structure
    %--------------------------------------------------------------------------
    mdp_1.T = 1;                      % number of updates
    mdp_1.A = A;                      % likelihood mapping
    mdp_1.B = B;                      % transition probabilities
    mdp_1.D = D;                      % prior over initial states

    mdp_1.erp = 1;
    
    mdp_1.Aname = {'Stimulus'};
    mdp_1.Bname = {'Stimulus'};
    
    mdp_1.a = a;
    
    clear a A B D
    
    MDP_1 = spm_MDP_check(mdp_1);
    
    clear mdp_1

end

%% Level 2: Slower-timescale representations of perceived stimulus sequences
%==========================================================================

% Prior beliefs about initial states in generative process (D) and
% generative model (d)
%--------------------------------------------------------------------------

if fit_options.D_learning == 1
    D{1} = [1-0.5           0.5]';          % ('async' 'sync')
    D{2} = [1 0 0]';                        % ('T1' 'T2' 'T3')

    d{1} = [1-params.pS     params.pS]';    % ('async' 'sync')
    d{2} = D{2}*1000; 
else
    D{1} = [1-params.pS     params.pS]';    % ('async' 'sync')
    D{2} = [1 0 0]';                        % ('T1' 'T2' 'T3')
end

%% set up mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D); % number of state factors (2)
for f = 1:Nf
    Ns(f) = numel(D{f}); % number of states (2,3)
end

No    = [3];         % number of outcomes: (start, async  , sync)
                              
Ng    = numel(No);   % number of outcome modalities (1)

for g = 1:Ng
    A{g} = zeros([No(g),Ns]); % set up likelihood matrix (A)
end

%% A  matrix (likelihood mapping)
%--------------------------------------------------------------------------
% 3x2x3, third dimension is timestep
% columns: {async, sync} state
% rows:                  observations

A{1}(:,:,1) =   [1 1;  % start
                 0 0;  % async
                 0 0]; % sync

A{1}(:,:,2) =   [0 0;  % start
                 1 0;  % async
                 0 1]; % sync

if fit_options.is_feedback == 1
    A{1}(:,:,3) =   [0 0;  % start
                     1 0;  % async
                     0 1]; % sync
else           
    A{1}(:,:,3) =   [1 1;  % start
                     0 0;  % async
                     0 0]; % sync
end
       
for g = 1:Ng
    A{g} = double(A{g});
end

%% Learning A (and assigning IP)
%--------------------------------------------------------------------------

if fit_options.A_learning == 1
    
    a{1} = A{1}*1000;

    % columns:      {async,       sync} state
    a{1}(:,:,2) =   [ 0            0;                        % start
                     params.IP2    1 - params.IP2;             % async
                     1 - params.IP2  params.IP2];  % sync
else 
    A{1}(:,:,2) =   [ 0            0;                        % start
                     params.IP2    1 - params.IP2;             % async
                     1 - params.IP2  params.IP2];              % sync
end

%% B  matrix (state transition priors)
%--------------------------------------------------------------------------

% Zeta controls the precision of state transitions in the async/sync factor
if isfield(params, 'zeta')
           % Async         sync at T
    B{1} =  [params.zeta   1-params.zeta; % async at T+1
            1-params.zeta  params.zeta];  %  sync at T+1
else
    B{1} =  [1 0;  % async at T+1
             0 1]; %  sync at T+1
end

B{2} =  [0 0 0; % T1
         1 0 0; % T2
         0 1 1];% T3

% Time points in a trial (3) 
%--------------------------------------------------------------------------
T=3;

% priors over outcomes C
%--------------------------------------------------------------------------
% flat outcome priors (this plays no role in the model)
C{1}     = zeros(No(1),T);

%% MDP Structure
%==========================================================================
if fit_options.is_hier == 1
    mdp.MDP  = MDP_1; % Insert lower level process
    mdp.link = [1];   % identifies lower level state factors (rows) with higher  
                      % level observation modalities (columns). 
end

mdp.T = T;                      % number of moves
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % priors over outcomes
mdp.D = D;                      % priors over initial states

mdp.IP1 = params.IP1;
mdp.IP2 = params.IP2;

%% Learning a: etas, SI

% only if learning is enabled for a
if fit_options.A_learning == 1
    mdp.a = a; 
    mdp.etaA            = params.etaA;            % single learning rate for A matrix
end

%% Forgetting
% omega, omegaBlock are optional in params input

if isfield(params, 'omega')
    mdp.omega = params.omega;
end

if isfield(params, 'omegaBlock')
    mdp.omegaBlock = params.omegaBlock;
end

%% Learning d: pS and etaD

if isfield(params, 'pS') 
    mdp.pS = params.pS; 
else 
end 

if fit_options.D_learning == 1
    mdp.d = d;                   % initial state bias    
    mdp.etaD = params.etaD;      % learning rate for bias
end

%% Faulty memory mechanism: zeta

if isfield(fit_options, 'zeta')
    mdp.zeta = params.zeta;
end

%% Exercise difference parameter
% IP1Diff optional in params input:

if isfield(params, 'IP1Diff') mdp.IP1Diff = params.IP1Diff; else end

%%
label.factor{1}   = 'Sync (State)';   label.name{1}    = {'out of sync','in sync'};
label.factor{2}   = 'Time (State)';   label.name{2}    = {'T1','T2','T3'};
label.modality{1} = 'Sync (Outcome)'; label.outcome{1} = {'start','out of sync','in sync'};

mdp.label = label;

mdp.erp=1;

mdp       = spm_MDP_check(mdp);


end
             