%% =====================================================================
% Script: Figure_3.m
% Purpose: Produce Figure 3 from the paper
%
% Description:
%   This script loads the required data, runs the simulations,
%   and generates the figures included in the manuscript.
%
%  Specifically, this script generates:
%    (1) Boxplots of pC80 values across conditions
%    (2) Boxplots of FLIP0 values overlaid with scatter points colored by
%        pC80 intensities
%    (3) The final composite figure (Figure 3 in the paper)
% ======================================================================

clear all
close all
clc

%% ------------ Load the estimated values for pC80 and FLIP0 --------------

% Load pC8(0) estimated for each single cell (S (Sensitive) or R (Resistant), 25/50/100 ng TRAIL)
load('pC80_values_R_25ng.mat')
load('pC80_values_S_25ng.mat')
load('pC80_values_R_50ng.mat')
load('pC80_values_S_50ng.mat')
load('pC80_values_R_100ng.mat')
load('pC80_values_S_100ng.mat')

% Load FLIP(0) estimated for each single cell (S (Sensitive) or R (Resistant), 25/50/100 ng TRAIL)
load('FLIP0_values_R_25ng.mat')
load('FLIP0_values_S_25ng.mat')
load('FLIP0_values_R_50ng.mat')
load('FLIP0_values_S_50ng.mat')
load('FLIP0_values_R_100ng.mat')
load('FLIP0_values_S_100ng.mat')

% Label the estimated parameters (S/R × 3 TRAIL concentrations)
xtick_labels = {'S - 25ng','R - 25ng','S - 50ng','R - 50ng','S - 100ng','R - 100ng'};

%% ----------------- Boxplot pC8(0) values ---------------

% Group the estimated parameters (S/R × 3 TRAIL concentrations)
pC80_groups = {
    pC80_values_S_25ng, pC80_values_R_25ng, ...
    pC80_values_S_50ng, pC80_values_R_50ng, ...
    pC80_values_S_100ng, pC80_values_R_100ng
};

% Create the group IDs for the boxplot
pC80_groupIDs = cellfun(@(v,i) repmat(i, numel(v), 1), ...
                        pC80_groups, num2cell(1:6), 'UniformOutput', false);

% Technical step: flatten into matrices
pC80_matrix = vertcat(pC80_groups{:});
pC80_labels = vertcat(pC80_groupIDs{:});

figure(1)
h1 = boxplot(pC80_matrix, pC80_labels, 'Notch', 'on');
set(h1, 'LineWidth', 1.5);
title('pC80 Distribution Across Experimental Conditions');
xticks(1:6);
xticklabels(xtick_labels);
ylabel('pC80 (number of molecules)');
axis square


%% ----------------- Boxplot FLIP(0) values with pC8(0) encoded single cell intensities ---------------

% Group the estimated parameters (S/R × 3 TRAIL concentrations)
FLIP0_groups = {
    FLIP0_values_S_25ng, FLIP0_values_R_25ng, ...
    FLIP0_values_S_50ng, FLIP0_values_R_50ng, ...
    FLIP0_values_S_100ng, FLIP0_values_R_100ng
};

% Create the group IDs for the boxplot
FLIP0_groupIDs = cellfun(@(v,i) repmat(i, numel(v), 1), ...
                         FLIP0_groups, num2cell(1:6), 'UniformOutput', false);

% Technical step: flatten into matrices
FLIP0_matrix = vertcat(FLIP0_groups{:});
FLIP0_labels = vertcat(FLIP0_groupIDs{:});

% Consider pC80 values globally across all conditions
all_pC80 = vertcat(pC80_groups{:});
global_min = min(all_pC80);
global_max = max(all_pC80);

figure(2)
% ------ Boxplot FLIP(0) --------

h2 = boxplot(FLIP0_matrix, FLIP0_labels, 'Notch', 'on');
set(h2, 'LineWidth', 2);
hold on;

% ------ Add dots for each individual of the FLIP(0) estimated value colored in function of
% the corresponding pC8(0) estimated value ------------------------

% Colormap for pC80-based scatter colors
colormap('plasma');
colors = plasma(256);

% Scatter with pC80-derived colors
for i = 1:6
    y_values   = FLIP0_groups{i};
    pC80_local = pC80_groups{i};

    % Normalize the pC8(0) values to have the correspondence with the
    % colorbar
    norm_pC80 = (pC80_local - global_min) / (global_max - global_min);
    color_idx = max(1, floor(norm_pC80 * 255) + 1);
    point_colors = colors(color_idx, :);
    
    % Point positions
    scatter(i * ones(size(y_values)), y_values, ...
            35, point_colors, 'filled', ...
            'jitter', 'on', 'jitterAmount', 0.12);
    
    % Scatterplot setting
    scatter_objects = findall(gcf, 'Type', 'Scatter');
    set(scatter_objects, ...
    'Marker', 'o', ...
    'MarkerEdgeColor', 'flat', ...
    'MarkerFaceAlpha', 0.30, ...
    'SizeData', 40);

end

% Add the colorbar and regulate the axis
colorbar;
caxis([global_min, global_max]);
xticks(1:6);
xticklabels(xtick_labels);
ylabel('FLIP0 (number of molecules)');
title('FLIP0 Boxplot with pC80-Encoded Single-Cell Intensities');
axis square

%% ------------ Figure 3 of the paper ----------

fig_3 = figure(3); 

% Main axes of the composite figure
mainAx = axes('Parent', fig_3, 'Position', [0.13, 0.15, 0.78, 0.78]);

% Copy content of figure boxplot FLIP(0)
fig_2 = figure(2);
ax_2  = get(fig_2, 'CurrentAxes');
copyobj(allchild(ax_2), mainAx);
mainAx.XTick      = ax_2.XTick;
mainAx.XTickLabel = ax_2.XTickLabel;
mainAx.XLim       = ax_2.XLim;
mainAx.YLim       = ax_2.YLim;
title(mainAx, ax_2.Title.String);
xlabel(mainAx, ax_2.XLabel.String);
ylabel(mainAx, ax_2.YLabel.String);
colormap(mainAx, colormap(fig_2));
mainAx.CLim = ax_2.CLim;
mainAx.Layer = 'top';
drawnow;

% Copy the colorbar
cb_2 = findobj(fig_2, 'Type', 'ColorBar');
if ~isempty(cb_2)
    cb_new = colorbar(mainAx, cb_2.Location);
    cb_new.Limits = cb_2.Limits;
    cb_new.Label.String = cb_2.Label.String;
end

% Prepare the inset - boxplot of PC8(0) values
fig_1 = figure(1);
ax_1  = get(fig_1, 'CurrentAxes');
insetAx = axes('Parent', fig_3, 'Position', [0.18, 0.55, 0.32, 0.32]);
copyobj(allchild(ax_1), insetAx);
insetAx.XLim       = ax_1.XLim;
insetAx.YLim       = ax_1.YLim;
insetAx.XTick      = ax_1.XTick;
insetAx.XTickLabel = ax_1.XTickLabel;
insetAx.YTick      = ax_1.YTick;
insetAx.YTickLabel = ax_1.YTickLabel;
ylabel(insetAx, 'pC80 (number of molecules)', 'FontSize', 11);

% Insert the inset
insetAx.Box   = 'on';
insetAx.Layer = 'top';
drawnow;

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

