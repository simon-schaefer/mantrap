function run_coupled_sys()
 
%% Grid
grid_min = [-5; -5; -2.0; -2.0]; % Lower corner of computation domain
grid_max = [5; 5; 2.0; 2.0];     % Upper corner of computation domain
N = [21; 21; 21; 21];            % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);

%% target set
R = 1;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, [3, 4], [0; 0; 0; 0], R);

%% time vector
t0 = 0;
tMax = 2.0;
dt = 0.05;
tau = t0:dt:tMax;

%% problem parameters

% input and disturbance bounds
aMax = 1;
dMax = [0.1, 0.1];

% control trying to min or max value function?
uMode = 'max';
dMode = 'min';
minWidth = 'minVOverTime';

%% Pack problem parameters

% Define dynamic system
dSys = CoupledSys([3, 3, 0, 0], aMax, dMax);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dSys;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
schemeData.dMode = dMode;

%% Compute value function

%HJIextraArgs.visualize = true; %show plot
HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.initialValueSet = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

%[data, tau, extraOuts] = ...c
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, minWidth, HJIextraArgs);

end