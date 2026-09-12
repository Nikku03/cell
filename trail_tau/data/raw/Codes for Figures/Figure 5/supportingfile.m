%% =====================================================================
% Script: Figure_S5.m
% Purpose: (a) Produce the different panels of Figure S5 from the paper
%          (b) Using the results derived from the insilico population
%          classification, show how cell states of drug-sensitivity are determined by a
%          dose-dependent hyperplane in parameter landscapes specifically
%          focusing on (pC80, FLIP0, K_deg)
%  The classification threshold is inferred from the 25 ng TRAIL dataset
%  and then applied to classify the 50 ng and 100 ng datasets.

clear all
close all
clc 

% load the sampling
load('sampling.mat')

% load the in-silico data already classified using the procedure described
% in Section 4.1 of the manuscript
% In-silico classified data are matrices (1000,8) - 1000 in silico cells, 8
% attributess
% Specifically the columns represent [cell name, max_der_FRET, pC80, FLIP0,
% K_deg, phenotype]

load('data_classification_25ng.mat')
load('data_classification_50ng.mat')
load('data_classification_100ng.mat')

% ******* Define color for each dose **************

col_dose_25ng = [1 0.8 0.4];
col_dose_50ng = [1 0.9 0.5];
col_dose_100ng = [1 1 0.7 ];

% Number of simulated cells
N_sim = 1000;

% Set the color for the cells fate
color_sen = [0.6350 0.0780 0.1840];
color_res = [0 0.4470 0.7410];

% Set the boundaries for the graphic representation of the sampling
min_pC80_val = min(sampling(:,1));
max_pC80_val = max(sampling(:,1));
min_FLIP0_val = min(sampling(:,2));
max_FLIP0_val = max(sampling(:,2));
min_alpha0_val = min(sampling(:,3));
max_alpha0_val = max(sampling(:,3));
min_alpha1_val = min(sampling(:,4));
max_alpha1_val = max(sampling(:,4));
min_Kdeg_val = min(sampling(:,5));
max_Kdeg_val = max(sampling(:,5));

% Grid points for the surface 
surf_grid_points = 200;

% Combine the data for all doses
pC80_values_25ng = [data_classified_25ng(:,3)];
FLIP0_values_25ng = [data_classified_25ng(:,4)];
alpha0_values_25ng = [data_classified_25ng(:,5)];
alpha1_values_25ng = [data_classified_25ng(:,6)];
Kdeg_values_25ng = [data_classified_25ng(:,7)];
dose_values_25ng = 25 * ones(N_sim, 1);
class_labels_25ng = [data_classified_25ng(:,8)];
class_labels_25ng(class_labels_25ng == -1) = 0;

find_N_S_25ng = find(class_labels_25ng==0);
N_S_25ng = find_N_S_25ng(end);

% Final in-silico dataset
data_combined_25ng = [pC80_values_25ng, FLIP0_values_25ng, alpha0_values_25ng, alpha1_values_25ng, Kdeg_values_25ng, dose_values_25ng, class_labels_25ng];

% Extract features and labels
X_25ng = data_combined_25ng(:, [1,2,5]); % pC80, FLIP0, Kdeg, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(1)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);

% Extract features and labels
X_25ng = data_combined_25ng(:, [1,2,3]); % pC80, FLIP0, alpha0, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(2)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('alpha0 (a.u.)','FontSize',8);

% Extract features and labels
X_25ng = data_combined_25ng(:, [1,2,4]); % pC80, FLIP0, alpha1, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(3)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('alpha1 (a.u.)','FontSize',8);

% Extract features and labels
X_25ng = data_combined_25ng(:, [3,4,5]); % alpha0, alpha1, Kdeg, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(4)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('alpha0 (a.u.)','FontSize',8);
ylabel('alpha1 (a.u.)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);

% Selecting data to find the hyperplanes expression 
% Remove the value for Kdeg to apply the svm methid to find the hyperplane
find_N_S_25ng = find(data_classified_25ng(:,end)==0);
Kdeg_values_R_25ng = Kdeg_values_25ng((N_S_25ng+1):end);
min_Kdeg_25ng = min(Kdeg_values_R_25ng);
data_combined_25ng = data_combined_25ng(data_combined_25ng(:,5) >= min_Kdeg_25ng, :);

% Extract features and labels
X_25ng = data_combined_25ng(:, [1,2,5]); % pC80, FLIP0, Kdeg, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data


figure(5)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);

% Extract features and labels
X_25ng = data_combined_25ng(:, [1,2,3]); % pC80, FLIP0, Kdeg, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(6)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('alpha0 (a.u.)','FontSize',8);

% Extract features and labels
X_25ng = data_combined_25ng(:, [1,2,4]); % pC80, FLIP0, Kdeg, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(7)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('alpha1 (a.u.)','FontSize',8);

% Extract features and labels
X_25ng = data_combined_25ng(:, [3,4,5]); % pC80, FLIP0, Kdeg, and Dose - input data 
y_25ng = data_combined_25ng(:, 7);  % Labels (0: sensitive, 1: resistant) - output data

figure(8)
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
hold on
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
xlabel('alpha0 (a.u.)','FontSize',8);
ylabel('alpha1 (a.u.)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
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

% Apply axis square to all subplots (axes)
all_axes = findall(figs, 'Type', 'axes');
for ax = all_axes'
    axis(ax, 'square');
end