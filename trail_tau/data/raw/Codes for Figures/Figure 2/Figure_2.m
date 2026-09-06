%% =====================================================================
% Script: Figure_2.m
% Purpose: Produce the different panels of Figure 2 from the paper
%
% Description:
%   This script loads the required data, runs the simulations,
%   and generates the figures included in the manuscript.

%  Specifically, this script generates:
%    (1) The simulated vs experimental trajectories for 4 selected
%    Resistant cells (R) following a TRAIL dose administration of 25ng
%    (2) The simulated vs experimental trajectories for 4 selected
%    Sensitive cells (S) following a TRAIL dose administration of 25ng
%    (3) The simulated vs experimental trajectories for 4 selected
%    Resistant cells (R) following a TRAIL dose administration of 50ng
%    (4) The simulated vs experimental trajectories for 4 selected
%    Sensitive cells (S) following a TRAIL dose administration of 50ng
%    (5) The simulated vs experimental trajectories for 4 selected
%    Resistant cells (R) following a TRAIL dose administration of 100ng
%    (6) The simulated vs experimental trajectories for 4 selected
%    Sensitive cells (S) following a TRAIL dose administration of 100ng
% ======================================================================

clear all
close all
clc

% Setting the colormap - 4 cell trajectories are selected for each subgroup
% R cells 25ng, S cells 25ng, R cells 50ng, S cells 50ng, R cells 100ng, S cells 100ng
colormap = plasma(4); 

%% ------------------- General setting -------------------------------

% Variables
% y(1) = T - TRAIL
% y(2) = R - Receptors
% y(3) = Z0 - Complex TRAIL+Receptors
% y(4) = Z3 - Complex FLIP+Receptors
% y(5) = pC8 - proCaspase8
% y(6) = Z1 - Complex TRAIL+Receptors+pC8
% y(7) = Z2 - Complex TRAIL+Receptors+pC8+FLIP
% y(8) = FLIP 
% y(9) = C8 - Caspase8
% y(10) = FRET - FRET signal

% Population parameters
% common_parameters = [rK1bK1, rK2bK2, rK3bK3, rK2K1, rK3K1, alphaR_3, alphaC8, K_FRET]

% Individual parameters
% params = [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] 

% Observation time window
% T_window = [0 600] %unit measure = [min] - data collected every 5 minutes
t_obs = linspace(5,600,120);

% Rescaling time parameter - see Eq. (2) and (3) in the paper
K1 = 0.007325300696406; %unit measure = 1/[min]

% Option for the resolution of the ODE system
opt = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);

%% --------------- Model definition ----------------------

% ********* Load estimated population parameters for all the cells ***********

% common_parameters = [rK1bK1, rK2bK2, rK3bK3, rK2K1, rK3K1, alphaR_3, alphaC8, K_FRET]
load("common_parameters.mat")
rK1bK1 = common_parameters(1); %In the paper: K1_hat
rK2bK2 = common_parameters(2); %In the paper: K2_hat
rK3bK3 = common_parameters(3); %In the paper: K3_hat
rK2K1 = common_parameters(4);  %In the paper: K4_hat
rK3K1 = common_parameters(5);  %In the paper: K5_hat
alphaR_3 = common_parameters(6);
alphaC8 = common_parameters(7);
K_FRET = common_parameters(8);

% ********** Define the ODE model ******************

% Variables
% y(1) = T - TRAIL
% y(2) = R - Receptors
% y(3) = Z0 - Complex TRAIL+Receptors
% y(4) = Z3 - Complex FLIP+Receptors
% y(5) = pC8 - proCaspase8
% y(6) = Z1 - Complex TRAIL+Receptors+pC8
% y(7) = Z2 - Complex TRAIL+Receptors+pC8+FLIP
% y(8) = FLIP 
% y(9) = C8 - Caspase8
% y(10) = FRET - FRET signal

% Individual parameters
% params = [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] 

odeSystem = @(t, y, params) [-((y(1)*y(2)^3)/(y(2)^3+alphaR_3)) + rK1bK1*y(3);
                             - 3*((y(1)*y(2)^3)/(y(2)^3+alphaR_3)) + 3*rK1bK1*y(3);
                             + ((y(1)*y(2)^3)/(y(2)^3+alphaR_3)) - rK1bK1*y(3) - rK3K1*y(3)*y(8)^3 + rK3bK3*rK3K1*y(4) - rK2K1*y(3)*y(5)^2 + rK2bK2*rK2K1*y(6) + params(3)*y(6);
                             + rK3K1*y(3)*y(8)^3 - rK3bK3*rK3K1*y(4);
                             - 2*rK2K1*y(3)*y(5)^2 + 2*rK2bK2*rK2K1*y(6);
                             + rK2K1*y(3)*y(5)^2 - rK2bK2*rK2K1*y(6) - rK2K1*y(6)*y(8) + rK2bK2*rK2K1*y(7) - params(3)*y(6);
                             + rK2K1*y(6)*y(8) - rK2bK2*rK2K1*y(7) - params(4)*y(7);
                             - 3*rK3K1*y(3)*y(8)^3 + 3*rK3bK3*rK3K1*y(4) - rK2K1*y(6)*y(8) + rK2bK2*rK2K1*y(7);
                             + params(3)*y(6) + params(4)*y(7) - params(5)*(y(9)/(alphaC8+y(9))) - K_FRET*y(9);
                             + K_FRET*y(9)];

%% ---------- Initial Conditions - 25ng TRAIL ---------------

%TRAIL dose - 25 ng are equivalent to 750 molecules - see Table 1
TRAIL0 = 750; 
%Initial receptors number
R0 = 32000;
%Initial condition for C8
C80 = 30; 
%Initial conditions
IC_0 = @(params) [TRAIL0;R0;0;0;params(1);0;0;params(2);C80;0];

% ******* Selected R cells for the dose 25ng - ind curves R subgroup = [2
% 30 86 96] *****

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 2
% Second line: FRET signal of cell 30
% Third line: FRET signal of cell 86
% Fourth line: FRET signal of cell 86
load("FRET_R_25ng.mat")

% Load the Death time for the cells - for R cells it's the end of the
% observation window
% First line: Tend of cell 2
% Second line: Tend of cell 30
% Third line: Tend of cell 86
% Fourth line: Tend of cell 86
load("Tend_R_25ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 2
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 30
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 86
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 86
load("R_25ng_ind_par.mat")

for k=1:4

    % Define the cell lifespan
    Tend_R_25ng_r = Tend_R_25ng(k)*K1;
    tspan = [0 Tend_R_25ng_r];

    % Set the cell indivdual parameters
    ind_par = R_25ng_ind_par(k,:);

    % Solve the ODE system in the specific cell state in terms of
    % individual parameters and lifespan
    ode_solution_optimal = ode15s(@(t, y) odeSystem(t, y, ind_par), tspan, IC_0(ind_par),opt);

    % Set the color of the curve
    cellcol=colormap(k,:);

    % Plot of simulated vs experimental cell trajectories for R cells 
    figure(1)
    p1=plot(ode_solution_optimal.x/K1,ode_solution_optimal.y(end,:));
    p1.LineStyle='-';
    p1.Marker='none';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=2;
    hold on
    p1=plot(t_obs,FRET_R_25ng(k,:));
    p1.LineStyle='none';
    p1.Marker='*';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=1.5;
    hold on
    xlabel('Time (minutes)')
    ylabel('FRET')
    title('Dose 25 ng/ml -- Resistant cells')
    ylim([0 0.9])
    xlim([0 600])
    legend('cell 2','', 'cell 30', '', 'cell 86','', 'cell 96','')
    legend boxoff
end

% ******* Selected S cells for the dose 25ng - ind curves S subgroup = [1
% 16 40 87] *****

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 1
% Second line: FRET signal of cell 16
% Third line: FRET signal of cell 40
% Fourth line: FRET signal of cell 87
load("FRET_S_25ng.mat")

% Load the Death time for the cells 
% First line: Tend of cell 1
% Second line: Tend of cell 16
% Third line: Tend of cell 40
% Fourth line: Tend of cell 87
load("Tend_S_25ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 1
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 16
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 40
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 87
load("S_25ng_ind_par.mat")

for k=1:4

    % Define the cell lifespan
    Tend_S_25ng_r = Tend_S_25ng(k)*K1;
    tspan = [0 Tend_S_25ng_r];

    % Set the cell indivdual parameters
    ind_par = S_25ng_ind_par(k,:);

    % Solve the ODE system in the specific cell state in terms of
    % individual parameters and lifespan
    ode_solution_optimal = ode15s(@(t, y) odeSystem(t, y, ind_par), tspan, IC_0(ind_par),opt);

    % Set the color of the curve
    cellcol=colormap(k,:);

    % Plot of simulated vs experimental cell trajectories for S cells 
    figure(2)
    p1=plot(ode_solution_optimal.x/K1,ode_solution_optimal.y(end,:));
    p1.LineStyle='-';
    p1.Marker='none';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=2;
    hold on
    p1=plot(t_obs,FRET_S_25ng(k,:));
    p1.LineStyle='none';
    p1.Marker='*';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=1.5;
    hold on
    xlabel('Time (minutes)')
    ylabel('FRET')
    title('Dose 25 ng/ml -- Sensitive cells')
    ylim([0 0.9])
    xlim([0 600])
    legend('cell 1','', 'cell 16', '', 'cell 40','', 'cell 87','')
    legend boxoff
end

%% ---------- Initial Conditions - 50ng TRAIL ---------------

%TRAIL dose - 50 ng are equivalent to 1500 molecules - see Table 1
TRAIL0 = 1500;
%Initial receptors number
R0 = 32000;
%Initial condition for C8
C80 = 30; 
%Initial conditions
IC_0 = @(params) [TRAIL0;R0;0;0;params(1);0;0;params(2);C80;0];

% ******* Selected R cells for the dose 50ng - ind curves R subgroup = [7
% 47 87 89] *****

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 7
% Second line: FRET signal of cell 47
% Third line: FRET signal of cell 87
% Fourth line: FRET signal of cell 89
load("FRET_R_50ng.mat")

% Load the Death time for the cells - for R cells it's the end of the
% observation window
% First line: Tend of cell 7
% Second line: Tend of cell 47
% Third line: Tend of cell 87
% Fourth line: Tend of cell 89
load("Tend_R_50ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 7
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 47
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 87
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 89
load("R_50ng_ind_par.mat")

for k=1:4

    % Define the cell lifespan
    Tend_R_50ng_r = Tend_R_50ng(k)*K1;
    tspan = [0 Tend_R_50ng_r];

    % Set the cell indivdual parameters
    ind_par = R_50ng_ind_par(k,:);

    % Solve the ODE system in the specific cell state in terms of
    % individual parameters and lifespan
    ode_solution_optimal = ode15s(@(t, y) odeSystem(t, y, ind_par), tspan, IC_0(ind_par),opt);

    % Set the color of the curve
    cellcol=colormap(k,:);

    % Plot of simulated vs experimental cell trajectories for R cells 
    figure(3)
    p1=plot(ode_solution_optimal.x/K1,ode_solution_optimal.y(end,:));
    p1.LineStyle='-';
    p1.Marker='none';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=2;
    hold on
    p1=plot(t_obs,FRET_R_50ng(k,:));
    p1.LineStyle='none';
    p1.Marker='*';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=1.5;
    hold on
    xlabel('Time (minutes)')
    ylabel('FRET')
    title('Dose 50 ng/ml -- Resistant cells')
    ylim([0 0.9])
    xlim([0 600])
    legend('cell 7','', 'cell 47', '', 'cell 87','', 'cell 89','')
    legend boxoff
end

% ******* Selected S cells for the dose 50ng - ind curves S subgroup = [14
% 69 81 85] *****

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 14
% Second line: FRET signal of cell 69
% Third line: FRET signal of cell 81
% Fourth line: FRET signal of cell 85
load("FRET_S_50ng.mat")

% Load the Death time for the cells 
% First line: Tend of cell 14
% Second line: Tend of cell 69
% Third line: Tend of cell 81
% Fourth line: Tend of cell 85
load("Tend_S_50ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 14
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 69
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 81
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 85
load("S_50ng_ind_par.mat")

for k=1:4

    % Define the cell lifespan
    Tend_S_50ng_r = Tend_S_50ng(k)*K1;
    tspan = [0 Tend_S_50ng_r];

    % Set the cell indivdual parameters
    ind_par = S_50ng_ind_par(k,:);

    % Solve the ODE system in the specific cell state in terms of
    % individual parameters and lifespan
    ode_solution_optimal = ode15s(@(t, y) odeSystem(t, y, ind_par), tspan, IC_0(ind_par),opt);

    % Set the color of the curve
    cellcol=colormap(k,:);

    % Plot of simulated vs experimental cell trajectories for S cells 
    figure(4)
    p1=plot(ode_solution_optimal.x/K1,ode_solution_optimal.y(end,:));
    p1.LineStyle='-';
    p1.Marker='none';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=2;
    hold on
    p1=plot(t_obs,FRET_S_50ng(k,:));
    p1.LineStyle='none';
    p1.Marker='*';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=1.5;
    hold on
    xlabel('Time (minutes)')
    ylabel('FRET')
    title('Dose 50 ng/ml -- Sensitive cells')
    ylim([0 0.9])
    xlim([0 600])
    legend('cell 14','', 'cell 69', '', 'cell 81','', 'cell 85','')
    legend boxoff
end

%% ---------- Initial Conditions - 100ng TRAIL ---------------

%TRAIL dose - 100 ng are equivalent to 3000 molecules - see Table 1
TRAIL0 = 3000;
%Initial receptors number
R0 = 32000;
%Initial condition for C8
C80 = 30; 
%Initial conditions
IC_0 = @(params) [TRAIL0;R0;0;0;params(1);0;0;params(2);C80;0];

% ******* Selected R cells for the dose 100ng - ind curves R subgroup = [1 22 48 51] *****

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 1
% Second line: FRET signal of cell 22
% Third line: FRET signal of cell 48
% Fourth line: FRET signal of cell 51
load("FRET_R_100ng.mat")

% Load the Death time for the cells - for R cells it's the end of the
% observation window
% First line: Tend of cell 1
% Second line: Tend of cell 22
% Third line: Tend of cell 48
% Fourth line: Tend of cell 51
load("Tend_R_100ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 1
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 22
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 48
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 51
load("R_100ng_ind_par.mat")

for k=1:4

    % Define the cell lifespan
    Tend_R_100ng_r = Tend_R_100ng(k)*K1;
    tspan = [0 Tend_R_100ng_r];

    % Set the cell indivdual parameters
    ind_par = R_100ng_ind_par(k,:);

    % Solve the ODE system in the specific cell state in terms of
    % individual parameters and lifespan
    ode_solution_optimal = ode15s(@(t, y) odeSystem(t, y, ind_par), tspan, IC_0(ind_par),opt);

    % Set the color of the curve
    cellcol=colormap(k,:);

    % Plot of simulated vs experimental cell trajectories for R cells 
    figure(5)
    p1=plot(ode_solution_optimal.x/K1,ode_solution_optimal.y(end,:));
    p1.LineStyle='-';
    p1.Marker='none';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=2;
    hold on
    p1=plot(t_obs,FRET_R_100ng(k,:));
    p1.LineStyle='none';
    p1.Marker='*';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=1.5;
    hold on
    xlabel('Time (minutes)')
    ylabel('FRET')
    title('Dose 100 ng/ml -- Resistant cells')
    ylim([0 0.9])
    xlim([0 600])
    legend('cell 1','', 'cell 22', '', 'cell 48','', 'cell 51','')
    legend boxoff
end

% ******* Selected S cells for the dose 100ng - ind curves S subgroup = [150 252 261 333] *****

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 150
% Second line: FRET signal of cell 252
% Third line: FRET signal of cell 261
% Fourth line: FRET signal of cell 333
load("FRET_S_100ng.mat")

% Load the Death time for the cells 
% First line: Tend of cell 150
% Second line: Tend of cell 252
% Third line: Tend of cell 261
% Fourth line: Tend of cell 333
load("Tend_S_100ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 150
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 252
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 261
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 333
load("S_100ng_ind_par.mat")

for k=1:4

    % Define the cell lifespan
    Tend_S_100ng_r = Tend_S_100ng(k)*K1;
    tspan = [0 Tend_S_100ng_r];

    % Set the cell indivdual parameters
    ind_par = S_100ng_ind_par(k,:);

    % Solve the ODE system in the specific cell state in terms of
    % individual parameters and lifespan
    ode_solution_optimal = ode15s(@(t, y) odeSystem(t, y, ind_par), tspan, IC_0(ind_par),opt);

    % Set the color of the curve
    cellcol=colormap(k,:);

    % Plot of simulated vs experimental cell trajectories for S cells 
    figure(6)
    p1=plot(ode_solution_optimal.x/K1,ode_solution_optimal.y(end,:));
    p1.LineStyle='-';
    p1.Marker='none';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=2;
    hold on
    p1=plot(t_obs,FRET_S_100ng(k,:));
    p1.LineStyle='none';
    p1.Marker='*';
    p1.MarkerSize=2.5;
    p1.MarkerFaceColor=cellcol;
    p1.Color=cellcol;
    p1.LineWidth=1.5;
    hold on
    xlabel('Time (minutes)')
    ylabel('FRET')
    title('Dose 100 ng/ml -- Sensitive cells')
    ylim([0 0.9])
    xlim([0 600])
    legend('cell 150','', 'cell 252', '', 'cell 261','', 'cell 333','')
    legend boxoff
end

%% ----------- Uniform figures for the paper -----------------

% Find figures
figs = findall(0, 'Type', 'figure');

% Define desired size
fig_width = 500;
fig_height = 500;

% Resize all figures
for i = 1:length(figs)
    set(figs(i), 'Position', [100*i, 100*i, fig_width, fig_height]);
end

% Apply to all scatter objects in all figures
scatter_objects = findall(0, 'Type', 'scatter');
set(scatter_objects, 'MarkerFaceAlpha', 0.4);
set(scatter_objects, 'Marker', 'o');
set(scatter_objects, 'SizeData', 80);
set(scatter_objects, 'MarkerEdgeColor', 'flat');  