%% =====================================================================
% Script: Figure_5.m
% Purpose: (a) Produce the different panels of Figure 5 from the paper
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

%% ******************* Polynomial surf - 25ng **************************

% Combine the data for all doses
pC80_values_25ng = [data_classified_25ng(:,3)];
FLIP0_values_25ng = [data_classified_25ng(:,4)];
Kdeg_values_25ng = [data_classified_25ng(:,7)];
dose_values_25ng = 25 * ones(N_sim, 1);
class_labels_25ng = [data_classified_25ng(:,8)];
class_labels_25ng(class_labels_25ng == -1) = 0;

find_N_S_25ng = find(class_labels_25ng==0);
N_S_25ng = find_N_S_25ng(end);

% Final in-silico dataset
data_combined_25ng = [pC80_values_25ng, FLIP0_values_25ng, Kdeg_values_25ng, dose_values_25ng, class_labels_25ng];

% Extract features and labels
X_25ng = data_combined_25ng(:, 1:4); % pC80, FLIP0, KFRET, and Dose - input data 
y_25ng = data_combined_25ng(:, 5);  % Labels (0: sensitive, 1: resistant) - output data

% Fit a nonlinear SVM model to classify the points
svm_model_25ng = fitcsvm(X_25ng, y_25ng, 'KernelFunction', 'polynomial', 'Standardize', true);

% Create a 3D grid for a specific dose (e.g., dose = 25ng)
specific_dose_25ng = 25; % Choose the dose level
[pC80_grid_25ng, FLIP0_grid_25ng, Kdeg_grid_25ng] = ndgrid( ...
    linspace(min(X_25ng(:, 1)), max(X_25ng(:, 1)), surf_grid_points ), ... % pC80 range
    linspace(min(X_25ng(:, 2)), max(X_25ng(:, 2)), surf_grid_points ), ... % FLIP0 range
    linspace(min(X_25ng(:, 3)), max(X_25ng(:, 3)), surf_grid_points ));    % KFRET range

% Combine grid points with the fixed dose for prediction
grid_points_25ng = [pC80_grid_25ng(:), FLIP0_grid_25ng(:), Kdeg_grid_25ng(:), specific_dose_25ng * ones(numel(pC80_grid_25ng), 1)];

% Predict on the grid
decision_values_25ng = predict(svm_model_25ng, grid_points_25ng);

% Reshape decision values to match the 3D grid
decision_grid_25ng = reshape(decision_values_25ng, size(pC80_grid_25ng));

%% ***************** Searching for hyperplanes ***************************

% Selecting data to find the hyperplanes expression 
% Remove the value for Kdeg to apply the svm methid to find the hyperplane
find_N_S_25ng = find(data_classified_25ng(:,end)==0);
Kdeg_values_R_25ng = Kdeg_values_25ng((N_S_25ng+1):end);
min_Kdeg_25ng = min(Kdeg_values_R_25ng);
data_combined_25ng_p = data_combined_25ng(data_combined_25ng(:,3) >= min_Kdeg_25ng, :);

%% ******** Hyperplane 25ng ***************************

% Extract features and labels for the hyperplane
X_25ng_p = data_combined_25ng_p(:, 1:4); % pC80, FLIP0, KFRET, and Dose
y_25ng_p = data_combined_25ng_p(:, 5);  % Labels (0: sensitive, 1: resistant)

% Fit a linear SVM model (RBF kernel) to classify the points
svm_model_25ng_p = fitcsvm(X_25ng_p, y_25ng_p, 'KernelFunction', 'linear', 'Standardize', true);

% Create a 3D grid for a specific dose (e.g., dose = 25 ng)
specific_dose_25ng_p = 25; % Choose the dose level
[pC80_grid_25ng_p, FLIP0_grid_25ng_p, Kdeg_grid_25ng_p] = ndgrid( ...
    linspace(min(X_25ng_p(:, 1)), max(X_25ng_p(:, 1)), surf_grid_points ), ... % pC80 range
    linspace(min(X_25ng_p(:, 2)), max(X_25ng_p(:, 2)), surf_grid_points ), ... % FLIP0 range
    linspace(min(X_25ng_p(:, 3)), max(X_25ng_p(:, 3)), surf_grid_points ));    % KFRET range

% Combine grid points with the fixed dose for prediction
grid_points_25ng_p = [pC80_grid_25ng_p(:), FLIP0_grid_25ng_p(:), Kdeg_grid_25ng_p(:), specific_dose_25ng_p * ones(numel(pC80_grid_25ng_p), 1)];

% Predict on the grid
decision_values_25ng_p = predict(svm_model_25ng_p, grid_points_25ng_p);

% Reshape decision values to match the 3D grid
decision_grid_25ng_p = reshape(decision_values_25ng_p, size(pC80_grid_25ng_p));

%Recover faces and vertices separately from the predicted set
[surf_25ng_faces_p, surf_25ng_vertex_p]= isosurface(pC80_grid_25ng_p, FLIP0_grid_25ng_p, Kdeg_grid_25ng_p, decision_grid_25ng_p, 0); % Extract the isosurface

%% ******** Fit a plane to the predicted vertices  ***************************

%Recover the plane found by SVM:  beta1*x1 + beta2*x2 + beta3*x3 +bias=0
%this is however not a good choice, since the plane given by SVM considers all sensitive outliers on one side of the plane
sv = svm_model_25ng_p.SupportVectors;
beta = svm_model_25ng_p.Beta;
b0 = svm_model_25ng_p.Bias;

%Use the above beta as initial guess to fit a similar plane to the  
par0 = [beta(1:3);b0];
par3_25ng = fminsearch(@mcfit3,par0,[],surf_25ng_vertex_p(:,1),surf_25ng_vertex_p(:,2),surf_25ng_vertex_p(:,3));

%Visualize on figure
xx3_25ng=linspace(min(surf_25ng_vertex_p(:,1)),max(surf_25ng_vertex_p(:,1)),40);
yy3_25ng=linspace(min(surf_25ng_vertex_p(:,2)),3.6e4,40); %max FLIP closer to 50 and 100 ng 

  for ii=1:length(xx3_25ng)  %pC80
	   for jj=1:length(yy3_25ng) %FLIP0 as a function of pC8
		    
	    zz3_25ng(jj,ii)=( -par3_25ng(1)*xx3_25ng(ii) - par3_25ng(2)*yy3_25ng(jj) - par3_25ng(4))/par3_25ng(3); 
        zz4_25ng(jj,ii)=min_Kdeg_25ng;   %the constant plane

       end
  end

  %% ******************* Polynomial surf - 50ng **************************

% Combine the data for all doses
pC80_values_50ng = [data_classified_50ng(:,3)];
FLIP0_values_50ng = [data_classified_50ng(:,4)];
Kdeg_values_50ng = [data_classified_50ng(:,7)];
dose_values_50ng = 50 * ones(N_sim, 1);
class_labels_50ng = [data_classified_50ng(:,8)];
class_labels_50ng(class_labels_50ng == -1) = 0;

find_N_S_50ng = find(class_labels_50ng==0);
N_S_50ng = find_N_S_50ng(end);

% Final in-silico dataset
data_combined_50ng = [pC80_values_50ng, FLIP0_values_50ng, Kdeg_values_50ng, dose_values_50ng, class_labels_50ng];

% Extract features and labels
X_50ng = data_combined_50ng(:, 1:4); % pC80, FLIP0, KFRET, and Dose - input data 
y_50ng = data_combined_50ng(:, 5);  % Labels (0: sensitive, 1: resistant) - output data

% Fit a nonlinear SVM model to classify the points
svm_model_50ng = fitcsvm(X_50ng, y_50ng, 'KernelFunction', 'polynomial', 'Standardize', true);

% Create a 3D grid for a specific dose (e.g., dose = 50ng)
specific_dose_50ng = 50; % Choose the dose level
[pC80_grid_50ng, FLIP0_grid_50ng, Kdeg_grid_50ng] = ndgrid( ...
    linspace(min(X_50ng(:, 1)), max(X_50ng(:, 1)), surf_grid_points ), ... % pC80 range
    linspace(min(X_50ng(:, 2)), max(X_50ng(:, 2)), surf_grid_points ), ... % FLIP0 range
    linspace(min(X_50ng(:, 3)), max(X_50ng(:, 3)), surf_grid_points ));    % KFRET range

% Combine grid points with the fixed dose for prediction
grid_points_50ng = [pC80_grid_50ng(:), FLIP0_grid_50ng(:), Kdeg_grid_50ng(:), specific_dose_50ng * ones(numel(pC80_grid_50ng), 1)];

% Predict on the grid
decision_values_50ng = predict(svm_model_50ng, grid_points_50ng);

% Reshape decision values to match the 3D grid
decision_grid_50ng = reshape(decision_values_50ng, size(pC80_grid_50ng));

%% ***************** Searching for hyperplanes ***************************

% Selecting data to find the hyperplanes expression 
% Remove the value for Kdeg to apply the svm methid to find the hyperplane
find_N_S_50ng = find(data_classified_50ng(:,end)==0);
Kdeg_values_R_50ng = Kdeg_values_50ng((N_S_50ng+1):end);
min_Kdeg_50ng = min(Kdeg_values_R_50ng);
data_combined_50ng_p = data_combined_50ng(data_combined_50ng(:,3) >= min_Kdeg_50ng, :);

%% ******** Hyperplane 50ng ***************************

% Extract features and labels for the hyperplane
X_50ng_p = data_combined_50ng_p(:, 1:4); % pC80, FLIP0, KFRET, and Dose
y_50ng_p = data_combined_50ng_p(:, 5);  % Labels (0: sensitive, 1: resistant)

% Fit a linear SVM model (RBF kernel) to classify the points
svm_model_50ng_p = fitcsvm(X_50ng_p, y_50ng_p, 'KernelFunction', 'linear', 'Standardize', true);

% Create a 3D grid for a specific dose (e.g., dose = 50 ng)
specific_dose_50ng_p = 50; % Choose the dose level
[pC80_grid_50ng_p, FLIP0_grid_50ng_p, Kdeg_grid_50ng_p] = ndgrid( ...
    linspace(min(X_50ng_p(:, 1)), max(X_50ng_p(:, 1)), surf_grid_points ), ... % pC80 range
    linspace(min(X_50ng_p(:, 2)), max(X_50ng_p(:, 2)), surf_grid_points ), ... % FLIP0 range
    linspace(min(X_50ng_p(:, 3)), max(X_50ng_p(:, 3)), surf_grid_points ));    % KFRET range

% Combine grid points with the fixed dose for prediction
grid_points_50ng_p = [pC80_grid_50ng_p(:), FLIP0_grid_50ng_p(:), Kdeg_grid_50ng_p(:), specific_dose_50ng_p * ones(numel(pC80_grid_50ng_p), 1)];

% Predict on the grid
decision_values_50ng_p = predict(svm_model_50ng_p, grid_points_50ng_p);

% Reshape decision values to match the 3D grid
decision_grid_50ng_p = reshape(decision_values_50ng_p, size(pC80_grid_50ng_p));

%Recover faces and vertices separately from the predicted set
[surf_50ng_faces_p, surf_50ng_vertex_p]= isosurface(pC80_grid_50ng_p, FLIP0_grid_50ng_p, Kdeg_grid_50ng_p, decision_grid_50ng_p, 0); % Extract the isosurface

%% ******** Fit a plane to the predicted vertices  ***************************

%Recover the plane found by SVM:  beta1*x1 + beta2*x2 + beta3*x3 +bias=0
%this is however not a good choice, since the plane given by SVM considers all sensitive outliers on one side of the plane
sv = svm_model_50ng_p.SupportVectors;
beta = svm_model_50ng_p.Beta;
b0 = svm_model_50ng_p.Bias;

%Use the above beta as initial guess to fit a similar plane to the  
par0 = [beta(1:3);b0];
par3_50ng = fminsearch(@mcfit3,par0,[],surf_50ng_vertex_p(:,1),surf_50ng_vertex_p(:,2),surf_50ng_vertex_p(:,3));

%Visualize on figure
xx3_50ng=linspace(min(surf_50ng_vertex_p(:,1)),max(surf_50ng_vertex_p(:,1)),40);
yy3_50ng=linspace(min(surf_50ng_vertex_p(:,2)),3.6e4,40); %max FLIP closer to 50 and 100 ng 

  for ii=1:length(xx3_50ng)  %pC80
	   for jj=1:length(yy3_50ng) %FLIP0 as a function of pC8
		    
	    zz3_50ng(jj,ii)=( -par3_50ng(1)*xx3_50ng(ii) - par3_50ng(2)*yy3_50ng(jj) - par3_50ng(4))/par3_50ng(3); 
        zz4_50ng(jj,ii)=min_Kdeg_50ng;   %the constant plane

       end
  end

    %% ******************* Polynomial surf - 100ng **************************

% Combine the data for all doses
pC80_values_100ng = [data_classified_100ng(:,3)];
FLIP0_values_100ng = [data_classified_100ng(:,4)];
Kdeg_values_100ng = [data_classified_100ng(:,7)];
dose_values_100ng = 100 * ones(N_sim, 1);
class_labels_100ng = [data_classified_100ng(:,8)];
class_labels_100ng(class_labels_100ng == -1) = 0;

find_N_S_100ng = find(class_labels_100ng==0);
N_S_100ng = find_N_S_100ng(end);

% Final in-silico dataset
data_combined_100ng = [pC80_values_100ng, FLIP0_values_100ng, Kdeg_values_100ng, dose_values_100ng, class_labels_100ng];

% Extract features and labels
X_100ng = data_combined_100ng(:, 1:4); % pC80, FLIP0, KFRET, and Dose - input data 
y_100ng = data_combined_100ng(:, 5);  % Labels (0: sensitive, 1: resistant) - output data

% Fit a nonlinear SVM model to classify the points
svm_model_100ng = fitcsvm(X_100ng, y_100ng, 'KernelFunction', 'polynomial', 'Standardize', true);

% Create a 3D grid for a specific dose (e.g., dose = 100ng)
specific_dose_100ng = 100; % Choose the dose level
[pC80_grid_100ng, FLIP0_grid_100ng, Kdeg_grid_100ng] = ndgrid( ...
    linspace(min(X_100ng(:, 1)), max(X_100ng(:, 1)), surf_grid_points ), ... % pC80 range
    linspace(min(X_100ng(:, 2)), max(X_100ng(:, 2)), surf_grid_points ), ... % FLIP0 range
    linspace(min(X_100ng(:, 3)), max(X_100ng(:, 3)), surf_grid_points ));    % KFRET range

% Combine grid points with the fixed dose for prediction
grid_points_100ng = [pC80_grid_100ng(:), FLIP0_grid_100ng(:), Kdeg_grid_100ng(:), specific_dose_100ng * ones(numel(pC80_grid_100ng), 1)];

% Predict on the grid
decision_values_100ng = predict(svm_model_100ng, grid_points_100ng);

% Reshape decision values to match the 3D grid
decision_grid_100ng = reshape(decision_values_100ng, size(pC80_grid_100ng));

%% ***************** Searching for hyperplanes ***************************

% Selecting data to find the hyperplanes expression 
% Remove the value for Kdeg to apply the svm methid to find the hyperplane
find_N_S_100ng = find(data_classified_100ng(:,end)==0);
Kdeg_values_R_100ng = Kdeg_values_100ng((N_S_100ng+1):end);
min_Kdeg_100ng = min(Kdeg_values_R_100ng);
data_combined_100ng_p = data_combined_100ng(data_combined_100ng(:,3) >= min_Kdeg_100ng, :);

%% ******** Hyperplane 100ng ***************************

% Extract features and labels for the hyperplane
X_100ng_p = data_combined_100ng_p(:, 1:4); % pC80, FLIP0, KFRET, and Dose
y_100ng_p = data_combined_100ng_p(:, 5);  % Labels (0: sensitive, 1: resistant)

% Fit a linear SVM model (RBF kernel) to classify the points
svm_model_100ng_p = fitcsvm(X_100ng_p, y_100ng_p, 'KernelFunction', 'linear', 'Standardize', true);

% Create a 3D grid for a specific dose (e.g., dose = 100 ng)
specific_dose_100ng_p = 100; % Choose the dose level
[pC80_grid_100ng_p, FLIP0_grid_100ng_p, Kdeg_grid_100ng_p] = ndgrid( ...
    linspace(min(X_100ng_p(:, 1)), max(X_100ng_p(:, 1)), surf_grid_points ), ... % pC80 range
    linspace(min(X_100ng_p(:, 2)), max(X_100ng_p(:, 2)), surf_grid_points ), ... % FLIP0 range
    linspace(min(X_100ng_p(:, 3)), max(X_100ng_p(:, 3)), surf_grid_points ));    % KFRET range

% Combine grid points with the fixed dose for prediction
grid_points_100ng_p = [pC80_grid_100ng_p(:), FLIP0_grid_100ng_p(:), Kdeg_grid_100ng_p(:), specific_dose_100ng_p * ones(numel(pC80_grid_100ng_p), 1)];

% Predict on the grid
decision_values_100ng_p = predict(svm_model_100ng_p, grid_points_100ng_p);

% Reshape decision values to match the 3D grid
decision_grid_100ng_p = reshape(decision_values_100ng_p, size(pC80_grid_100ng_p));

%Recover faces and vertices separately from the predicted set
[surf_100ng_faces_p, surf_100ng_vertex_p]= isosurface(pC80_grid_100ng_p, FLIP0_grid_100ng_p, Kdeg_grid_100ng_p, decision_grid_100ng_p, 0); % Extract the isosurface

%% ******** Fit a plane to the predicted vertices  ***************************

%Recover the plane found by SVM:  beta1*x1 + beta2*x2 + beta3*x3 +bias=0
%this is however not a good choice, since the plane given by SVM considers all sensitive outliers on one side of the plane
sv = svm_model_100ng_p.SupportVectors;
beta = svm_model_100ng_p.Beta;
b0 = svm_model_100ng_p.Bias;

%Use the above beta as initial guess to fit a similar plane to the  
par0 = [beta(1:3);b0];
par3_100ng = fminsearch(@mcfit3,par0,[],surf_100ng_vertex_p(:,1),surf_100ng_vertex_p(:,2),surf_100ng_vertex_p(:,3));

%Visualize on figure
xx3_100ng=linspace(min(surf_100ng_vertex_p(:,1)),max(surf_100ng_vertex_p(:,1)),40);
yy3_100ng=linspace(min(surf_100ng_vertex_p(:,2)),3.6e4,40); %max FLIP closer to 100 and 100 ng 

  for ii=1:length(xx3_100ng)  %pC80
	   for jj=1:length(yy3_100ng) %FLIP0 as a function of pC8
		    
	    zz3_100ng(jj,ii)=( -par3_100ng(1)*xx3_100ng(ii) - par3_100ng(2)*yy3_100ng(jj) - par3_100ng(4))/par3_100ng(3); 
        zz4_100ng(jj,ii)=min_Kdeg_100ng;   %the constant plane

       end
  end

 %% **************** Figure 5 - panels ***************
 figure(1)
hold on;
% Plot the data
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
% Plot the 3D separating surface (isosurface for decision boundary)
surf_struct_25ng = isosurface(pC80_grid_25ng, FLIP0_grid_25ng, Kdeg_grid_25ng, decision_grid_25ng, 0); % Extract the isosurface
h_25ng = patch(surf_struct_25ng); % Create a patch object for visualization
set(h_25ng, 'FaceColor', col_dose_25ng, 'EdgeColor', 'none'); % Style the surface
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
grid on;
legend('Sensitive', 'Resistant');
view(3);
camlight; 
lighting phong;
hold off;
box on;
xlim([min_pC80_val max_pC80_val])
ylim([min_FLIP0_val max_FLIP0_val])
zlim([min_Kdeg_val max_Kdeg_val])
title('TRAIL - 25ng','FontSize',12);
% ********* separtion hyperplanes - dose 25ng **************
 figure(2)
hold on; 
box on; 
grid on;
% Plot the data
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p2=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p2.Marker='o';
p2.MarkerFaceColor=color_sen;
p2.MarkerFaceAlpha = 0.7;
p2.MarkerEdgeColor=color_sen;
legend([p1 p2], {'Sensitive', 'Resistant'});
hold on;
% Plot the hyperplanes
p3=surf(xx3_25ng,yy3_25ng,zz3_25ng,'HandleVisibility','off');
hold on;
p4=surf(xx3_25ng,yy3_25ng,zz4_25ng,'HandleVisibility','off');
view([-37.5,30]);
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
zlim([1250 1500]);
title('TRAIL - 25ng','FontSize',12);
% ********* separtion surfaces - dose 50ng **************
 figure(3)
hold on;
% Plot the data
p1=scatter3(X_50ng(y_50ng == 1, 1), X_50ng(y_50ng == 1, 2), X_50ng(y_50ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p1=scatter3(X_50ng(y_50ng == 0, 1), X_50ng(y_50ng == 0, 2), X_50ng(y_50ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
% Plot the 3D separating surface (isosurface for decision boundary)
surf_struct_50ng = isosurface(pC80_grid_50ng, FLIP0_grid_50ng, Kdeg_grid_50ng, decision_grid_50ng, 0); % Extract the isosurface
h_50ng = patch(surf_struct_50ng); % Create a patch object for visualization
set(h_50ng, 'FaceColor', col_dose_50ng, 'EdgeColor', 'none'); % Style the surface
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
grid on;
legend('Sensitive', 'Resistant');
view(3);
camlight; 
lighting phong;
hold off;
box on;
xlim([min_pC80_val max_pC80_val])
ylim([min_FLIP0_val max_FLIP0_val])
zlim([min_Kdeg_val max_Kdeg_val])
title('TRAIL - 50ng','FontSize',12);
% ********* separtion hyperplanes - dose 50ng **************
 figure(4)
hold on; 
box on; 
grid on;
% Plot the data
p1=scatter3(X_50ng(y_50ng == 1, 1), X_50ng(y_50ng == 1, 2), X_50ng(y_50ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p2=scatter3(X_50ng(y_50ng == 0, 1), X_50ng(y_50ng == 0, 2), X_50ng(y_50ng == 0, 3)); % Sensitive
p2.Marker='o';
p2.MarkerFaceColor=color_sen;
p2.MarkerFaceAlpha = 0.7;
p2.MarkerEdgeColor=color_sen;
legend([p1 p2], {'Sensitive', 'Resistant'});
hold on;
% Plot the hyperplanes
p3=surf(xx3_50ng,yy3_50ng,zz3_50ng,'HandleVisibility','off');
hold on;
p4=surf(xx3_50ng,yy3_50ng,zz4_50ng,'HandleVisibility','off');
view([-37.5,30]);
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
zlim([1250 1500]);
title('TRAIL - 50ng','FontSize',12);
% ********* separtion surfaces - dose 100ng **************
 figure(5)
hold on;
% Plot the data
p1=scatter3(X_100ng(y_100ng == 1, 1), X_100ng(y_100ng == 1, 2), X_100ng(y_100ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p1=scatter3(X_100ng(y_100ng == 0, 1), X_100ng(y_100ng == 0, 2), X_100ng(y_100ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
% Plot the 3D separating surface (isosurface for decision boundary)
surf_struct_100ng = isosurface(pC80_grid_100ng, FLIP0_grid_100ng, Kdeg_grid_100ng, decision_grid_100ng, 0); % Extract the isosurface
h_100ng = patch(surf_struct_100ng); % Create a patch object for visualization
set(h_100ng, 'FaceColor', col_dose_100ng, 'EdgeColor', 'none'); % Style the surface
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
grid on;
legend('Sensitive', 'Resistant');
view(3);
camlight; 
lighting phong;
hold off;
box on;
xlim([min_pC80_val max_pC80_val])
ylim([min_FLIP0_val max_FLIP0_val])
zlim([min_Kdeg_val max_Kdeg_val])
title('TRAIL - 100ng','FontSize',12);
% ********* separtion hyperplanes - dose 100ng **************
 figure(6)
hold on; 
box on; 
grid on;
% Plot the data
p1=scatter3(X_100ng(y_100ng == 1, 1), X_100ng(y_100ng == 1, 2), X_100ng(y_100ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p2=scatter3(X_100ng(y_100ng == 0, 1), X_100ng(y_100ng == 0, 2), X_100ng(y_100ng == 0, 3)); % Sensitive
p2.Marker='o';
p2.MarkerFaceColor=color_sen;
p2.MarkerFaceAlpha = 0.7;
p2.MarkerEdgeColor=color_sen;
legend([p1 p2], {'Sensitive', 'Resistant'});
hold on;
% Plot the hyperplanes
p3=surf(xx3_100ng,yy3_100ng,zz3_100ng,'HandleVisibility','off');
hold on;
p4=surf(xx3_100ng,yy3_100ng,zz4_100ng,'HandleVisibility','off');
view([-37.5,30]);
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
zlim([1250 1500]);
title('TRAIL - 100ng','FontSize',12);


%% ************** Figure 5 ************************

figure(7)
t = tiledlayout(3,2, 'TileSpacing', 'none', 'Padding', 'none');
% ********* separtion surface - dose 25ng ***************
nexttile 
hold on;
% Plot the data
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p1=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
% Plot the 3D separating surface (isosurface for decision boundary)
surf_struct_25ng = isosurface(pC80_grid_25ng, FLIP0_grid_25ng, Kdeg_grid_25ng, decision_grid_25ng, 0); % Extract the isosurface
h_25ng = patch(surf_struct_25ng); % Create a patch object for visualization
set(h_25ng, 'FaceColor', col_dose_25ng, 'EdgeColor', 'none'); % Style the surface
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
grid on;
legend('Sensitive', 'Resistant');
view(3);
camlight; 
lighting phong;
hold off;
box on;
xlim([min_pC80_val max_pC80_val])
ylim([min_FLIP0_val max_FLIP0_val])
zlim([min_Kdeg_val max_Kdeg_val])
% ********* separtion hyperplanes - dose 25ng **************
nexttile 
hold on; 
box on; 
grid on;
% Plot the data
p1=scatter3(X_25ng(y_25ng == 1, 1), X_25ng(y_25ng == 1, 2), X_25ng(y_25ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p2=scatter3(X_25ng(y_25ng == 0, 1), X_25ng(y_25ng == 0, 2), X_25ng(y_25ng == 0, 3)); % Sensitive
p2.Marker='o';
p2.MarkerFaceColor=color_sen;
p2.MarkerFaceAlpha = 0.7;
p2.MarkerEdgeColor=color_sen;
legend([p1 p2], {'Sensitive', 'Resistant'});
hold on;
% Plot the hyperplanes
p3=surf(xx3_25ng,yy3_25ng,zz3_25ng,'HandleVisibility','off');
hold on;
p4=surf(xx3_25ng,yy3_25ng,zz4_25ng,'HandleVisibility','off');
view([-37.5,30]);
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
zlim([1250 1500]);
% ********* separtion surfaces - dose 50ng **************
nexttile 
hold on;
% Plot the data
p1=scatter3(X_50ng(y_50ng == 1, 1), X_50ng(y_50ng == 1, 2), X_50ng(y_50ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p1=scatter3(X_50ng(y_50ng == 0, 1), X_50ng(y_50ng == 0, 2), X_50ng(y_50ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
% Plot the 3D separating surface (isosurface for decision boundary)
surf_struct_50ng = isosurface(pC80_grid_50ng, FLIP0_grid_50ng, Kdeg_grid_50ng, decision_grid_50ng, 0); % Extract the isosurface
h_50ng = patch(surf_struct_50ng); % Create a patch object for visualization
set(h_50ng, 'FaceColor', col_dose_50ng, 'EdgeColor', 'none'); % Style the surface
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
grid on;
legend('Sensitive', 'Resistant');
view(3);
camlight; 
lighting phong;
hold off;
box on;
xlim([min_pC80_val max_pC80_val])
ylim([min_FLIP0_val max_FLIP0_val])
zlim([min_Kdeg_val max_Kdeg_val])
% ********* separtion hyperplanes - dose 50ng **************
nexttile 
hold on; 
box on; 
grid on;
% Plot the data
p1=scatter3(X_50ng(y_50ng == 1, 1), X_50ng(y_50ng == 1, 2), X_50ng(y_50ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p2=scatter3(X_50ng(y_50ng == 0, 1), X_50ng(y_50ng == 0, 2), X_50ng(y_50ng == 0, 3)); % Sensitive
p2.Marker='o';
p2.MarkerFaceColor=color_sen;
p2.MarkerFaceAlpha = 0.7;
p2.MarkerEdgeColor=color_sen;
legend([p1 p2], {'Sensitive', 'Resistant'});
hold on;
% Plot the hyperplanes
p3=surf(xx3_50ng,yy3_50ng,zz3_50ng,'HandleVisibility','off');
hold on;
p4=surf(xx3_50ng,yy3_50ng,zz4_50ng,'HandleVisibility','off');
view([-37.5,30]);
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
zlim([1250 1500]);
% ********* separtion surfaces - dose 100ng **************
nexttile 
hold on;
% Plot the data
p1=scatter3(X_100ng(y_100ng == 1, 1), X_100ng(y_100ng == 1, 2), X_100ng(y_100ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p1=scatter3(X_100ng(y_100ng == 0, 1), X_100ng(y_100ng == 0, 2), X_100ng(y_100ng == 0, 3)); % Sensitive
p1.Marker='o';
p1.MarkerFaceColor=color_sen;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_sen;
% Plot the 3D separating surface (isosurface for decision boundary)
surf_struct_100ng = isosurface(pC80_grid_100ng, FLIP0_grid_100ng, Kdeg_grid_100ng, decision_grid_100ng, 0); % Extract the isosurface
h_100ng = patch(surf_struct_100ng); % Create a patch object for visualization
set(h_100ng, 'FaceColor', col_dose_100ng, 'EdgeColor', 'none'); % Style the surface
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
grid on;
legend('Sensitive', 'Resistant');
view(3);
camlight; 
lighting phong;
hold off;
box on;
xlim([min_pC80_val max_pC80_val])
ylim([min_FLIP0_val max_FLIP0_val])
zlim([min_Kdeg_val max_Kdeg_val])
% ********* separtion hyperplanes - dose 100ng **************
nexttile 
hold on; 
box on; 
grid on;
% Plot the data
p1=scatter3(X_100ng(y_100ng == 1, 1), X_100ng(y_100ng == 1, 2), X_100ng(y_100ng == 1, 3)); % Resistant
p1.Marker='o';
p1.MarkerFaceColor=color_res;
p1.MarkerFaceAlpha = 0.7;
p1.MarkerEdgeColor=color_res;
p2=scatter3(X_100ng(y_100ng == 0, 1), X_100ng(y_100ng == 0, 2), X_100ng(y_100ng == 0, 3)); % Sensitive
p2.Marker='o';
p2.MarkerFaceColor=color_sen;
p2.MarkerFaceAlpha = 0.7;
p2.MarkerEdgeColor=color_sen;
legend([p1 p2], {'Sensitive', 'Resistant'});
hold on;
% Plot the hyperplanes
p3=surf(xx3_100ng,yy3_100ng,zz3_100ng,'HandleVisibility','off');
hold on;
p4=surf(xx3_100ng,yy3_100ng,zz4_100ng,'HandleVisibility','off');
view([-37.5,30]);
xlabel('pC80 (number of molecules)','FontSize',8);
ylabel('FLIP0 (number of molecules)','FontSize',8);
zlabel('Kdeg (a.u.)','FontSize',8);
zlim([1250 1500]);

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