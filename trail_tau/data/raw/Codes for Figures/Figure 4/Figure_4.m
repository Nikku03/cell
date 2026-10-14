%% =====================================================================
% Script: Figure_4.m
% Purpose: (a) Produce the different panels of Figure 4 from the paper
%          (b) Illustrate the classification procedure of in-silico single-cell trajectories
%              into Sensitive (S) or Resistant (R) phenotypes based on the maximal derivative 
%              of the FRET signal, following the experimental methodology described in ROUX 2015 
%              and adopted in the manuscript.
%  The classification threshold is inferred from the 25 ng TRAIL dataset
%  and then applied to classify the 50 ng and 100 ng datasets.
%
%  The script specifically produces:
%     - the predicted vs experimental proportions of Sensitive and
%     Resistant cells
%     - 3D histograms of slope distributions across doses
% =========================================================================

clear all
close all
clc

%% ---------------- In Silico virtual cells population -----------------

% We generated a set of state-space sampled cell states by uniformly sampling 
% the five key parameters (pC80, FLIP0, α0, α1, Kdeg) within the minimum and 
% maximum values obtained from fitting our experimental dataset across the 
% three TRAIL doses. This corresponds to drawing N_sim = 1000 points uniformly 
% within the 5-dimensional hypercube defining the admissible initial cell-state 
% space. Each sampled point represents an individual cell.
% See file sampling.m 

% Load the set of state-space sampled cell states (N_sim = 1000, 5 params, 1000 X 5 matrix)
% Each line represents an individual (i=1,...,N_sim)
load('sampling.mat')

% We then simulated the FRET signal for each of these state-space–sampled 
% cell states (i.e., for each individual cell) at the three TRAIL doses 
% included in our dataset. 
% The files insilico_25ng.mat, insilico_50ng.mat, and insilico_100ng.mat 
% contain, in row i, the simulated FRET time-course corresponding to 
% individual i for an administered dose of 25 ng, 50 ng, and 100 ng, 
% respectively.
% See files insilico_25ng.m, insilico_50ng.m, insilico_100ng.m  

% Load in silico single-cell trajectories for 25 ng dose
load('insilico_25ng.mat')

%% ------------- Inferring the classification threshold from 25ng dose ---------------

% We inferred the slope-based classification threshold from the in silico 
% dataset insilico_25ng.mat as the value for which 55% of the cells lie above 
% it, consistent with the experimentally observed proportion of sensitive 
% cells at the 25ng dose.

% Total number of simulated cells at 25 ng
N_sim_25 = size(sampling,1);

% Compute the maximum derivative of the FRET signal for each cell
% (each row corresponds to one simulated cell)
max_der_FRET_25ng = max(der_FRET_signal, [], 2);

% Vector of cell indices
labels_vector_25ng = (1:N_sim_25)';

% Assemble dataset: [cell_id, max_FRET_derivative, sampled_parameters]
data_to_classify_25ng = [labels_vector_25ng, max_der_FRET_25ng, sampling];

% Sort cells by decreasing maximal derivative
data_to_classify_25ng_ordered = sortrows(data_to_classify_25ng, 2, 'descend');

% Fraction of Sensitive cells observed experimentally at 25 ng (177
% Sensitive cells, 150 Resistant cells in the experimental dataset)
perc_of_S_cells = round(177/(150+177), 2);

% Corresponding number of Sensitive cells in silico
N_S_cells_25ng = round(perc_of_S_cells * N_sim_25);

% Classification vector:
%   -1 = Sensitive    (highest slopes)
%   +1 = Resistant    (lowest slopes)
class_vector_25ng = [(-1)*ones(N_S_cells_25ng,1); ...
                      ones(N_sim_25 - N_S_cells_25ng, 1)];

% Full dataset with class labels
data_classified_25ng = [data_to_classify_25ng_ordered, class_vector_25ng];

% Classification threshold θ_ref = smallest slope assigned to a Sensitive cell
theta_ref = data_classified_25ng(N_S_cells_25ng, 2);


%% ---------------- Insilico population classification -------------------------

% We next applied the reference threshold inferred above to classify the 
% virtual cells in insilico_50ng.mat and insilico_100ng.mat. For each cell, 
% the maximal derivative of the FRET signal was computed and compared to 
% the threshold: cells with values above the threshold were classified as 
% Sensitive.

% ---------------- Dose 50ng ----------------------------
% Load in silico single-cell trajectories for 50ng dose
load('insilico_50ng.mat')

% Total number of simulations
N_sim_50 = size(sampling,1);

% Maximal derivative of FRET signal
max_der_FRET_50ng = max(der_FRET_signal, [], 2);

% Cell indices
labels_vector_50ng = (1:N_sim_50)';

% Assemble dataset
data_to_classify_50ng = [labels_vector_50ng, max_der_FRET_50ng, sampling];

% Sort cells by decreasing maximal derivative
data_to_classify_50ng_ordered = sortrows(data_to_classify_50ng, 2, 'descend');

% Cell classification is performed by comparing each cell's slope to the 
% reference threshold θ_ref. Slopes above θ_ref lead to classification 
% as Sensitive.
class_vector_50ng = NaN(N_sim_50,1);
for i = 1:N_sim_50
    if data_to_classify_50ng_ordered(i,2) > theta_ref
        class_vector_50ng(i) = -1;   % Sensitive
    else
        class_vector_50ng(i) = 1;    % Resistant
    end
end

% Full dataset with labels
data_classified_50ng = [data_to_classify_50ng_ordered, class_vector_50ng];

% Number of predicted Sensitive cells
N_S_cells_50ng = sum(class_vector_50ng == -1);

% ---------------- Dose 100ng ----------------------------
% Load in silico single-cell trajectories for 100ng dose
load('insilico_100ng.mat')

% Total number of simulations
N_sim_100 = size(sampling,1);

% Maximal derivative of FRET signal
max_der_FRET_100ng = max(der_FRET_signal, [], 2);

% Cell indices
labels_vector_100ng = (1:N_sim_100)';

% Assemble dataset
data_to_classify_100ng = [labels_vector_100ng, max_der_FRET_100ng, sampling];

% Sort cells by decreasing maximal derivative
data_to_classify_100ng_ordered = sortrows(data_to_classify_100ng, 2, 'descend');

% Cell classification is performed by comparing each cell's slope to the 
% reference threshold θ_ref. Slopes above θ_ref lead to classification 
% as Sensitive.
class_vector_100ng = NaN(N_sim_100,1);
for i = 1:N_sim_100
    if data_to_classify_100ng_ordered(i,2) > theta_ref
        class_vector_100ng(i) = -1;   % Sensitive
    else
        class_vector_100ng(i) = 1;    % Resistant
    end
end

% Full dataset with labels
data_classified_100ng = [data_to_classify_100ng_ordered, class_vector_100ng];

% Number of predicted Sensitive cells
N_S_cells_100ng = sum(class_vector_100ng == -1);

%% ---------------- Model validation ------------------------

% Experimental fractions of Sensitive cells (ROUX 2015)
% Experimental dataset 25ng: 177 S, 150 R
% Experimental  dataset 50ng: 300 S, 114 R
% Experimental  dataset 50ng: 518 S, 65 R
perc_S_25ng_exp  = 177 / (150 + 177);
perc_S_50ng_exp  = 300 / (114 + 300);
perc_S_100ng_exp = 518 / (65 + 518);

% Predicted fractions using the classification criterion theta_ref
perc_S_25ng_comp  = N_S_cells_25ng  / N_sim_25;
perc_S_50ng_comp  = N_S_cells_50ng  / N_sim_50;
perc_S_100ng_comp = N_S_cells_100ng / N_sim_100;

% Convert to percentages
experimental = [perc_S_25ng_exp,  perc_S_50ng_exp,  perc_S_100ng_exp]  * 100;
computed     = [perc_S_25ng_comp, perc_S_50ng_comp, perc_S_100ng_comp] * 100;

% Comparing the experimental and computed percentages of sensitive cells in
% the case of 50ng and 100ng TRAIL administrated
figure(1)
bar_data = [experimental(1), 100-experimental(1); ...
            computed(1),     100-computed(1); ...
            0, 0; ...
            experimental(2), 100-experimental(2); ...
            computed(2),     100-computed(2); ...
            0, 0; ...
            experimental(3), 100-experimental(3); ...
            computed(3),     100-computed(3)];

bhand = bar(bar_data, 'stacked', 'FaceAlpha', 0.5, 'FaceColor', 'flat');

% Colors for experimental bars
for i = 1:3:8
    bhand(1).CData(i,:) = [1 0 0];   % Sensitive (red)
    bhand(2).CData(i,:) = [0 0 1];   % Resistant (blue)
end

% Colors for in silico bars (darker shades)
for i = 2:3:8
    bhand(1).CData(i,:) = [0.6 0 0];
    bhand(2).CData(i,:) = [0 0 0.5];
end

% Adjust width
bhand(1).BarWidth = 0.6;
bhand(2).BarWidth = 0.6;

% Correct X labels (8 bars → 8 labels)
set(gca, 'XTick', 1:8);
set(gca, 'XTickLabel', { ...
    'Exp 25 ng', 'Pred 25 ng', '', ...
    'Exp 50 ng', 'Pred 50 ng', '', ...
    'Exp 100 ng','Pred 100 ng'});

ylabel('Cell-response phenotype (% of the population)');


%% -------- 3D histograms of slopes for the free doses ------

% Techinical step: Bin centers for slope histogram
center_hist = [0.008855:0.000004:0.008887, ...
               0.008887:0.05:0.45, ...
               0.45:0.8:4.45, ...
               4.45:3:10];

% Compute histograms for each dose
bhist_25  = hist(data_to_classify_25ng(:,2),  center_hist);
bhist_50  = hist(data_to_classify_50ng(:,2),  center_hist);
bhist_100 = hist(data_to_classify_100ng(:,2), center_hist);

bhist3D = [bhist_25; bhist_50; bhist_100];
dXticks = floor(length(center_hist)/10);

% Index of θ_ref in histogram bins
indkthres = find(center_hist - theta_ref > 0, 1, 'first');

% ------- Original 3D plot  (kept invisible but used for consistency)----
% 3D histograms of slope distributions across doses
fig21 = figure(21);
set(fig21,'Visible','off');
bobj1 = bar3(bhist3D); 
% Axis formatting
whereXticks = 1:dXticks:length(center_hist);
set(gca, 'XTick', whereXticks, ...
         'XTickLabel', num2str(center_hist(whereXticks)'));
set(gca, 'YTick', 1:3, 'YTickLabel', num2str([25;50;100]));
zlabel('Slope Distribution','fontsize',14)
xlabel('FRET Slopes','fontsize',14)
ylabel('Doses')
set(gca,'PlotBoxAspectRatio',[8 2 4])


% ------- Custom 3D plot with patch-based bars ------------------

% Reorder dose rows for visualization
bhist3Didose(1,:) = bhist3D(3,:);
bhist3Didose(2,:) = bhist3D(2,:);
bhist3Didose(3,:) = bhist3D(1,:);

figure(2)
hold on;
% Colormap and range
cmap = colormap(hot(10));
shading interp;
data_range = [min(bhist3D(:)), max(bhist3D(:))];
[m, n] = size(bhist3D);
% Draw each bar manually using PATCH for full control
for i = 1:m
    for j = 1:n
        if bhist3D(i,j) > 0

            % Map height → color index
            color_idx = round((bhist3D(i,j) - data_range(1)) / ...
                             (data_range(2) - data_range(1)) * (size(cmap,1)-1)) + 1;
            color_idx = max(1, min(size(cmap,1), color_idx));

            % Bar height
            h = bhist3D(i,j);

            % X/Y coordinates of bar footprint
            x = j + [-0.4, 0.4, 0.4, -0.4];
            y = (m+1-i) + [-0.4, -0.4, 0.4, 0.4];  % reverse y ordering

            % Bottom, top and side faces
            patch(x, y, [0 0 0 0],   cmap(color_idx,:), 'EdgeColor','black','LineWidth',0.1);
            patch(x, y, [h h h h],   cmap(color_idx,:), 'EdgeColor','black','LineWidth',0.1);

            patch([x(1) x(2) x(2) x(1)], [y(1) y(2) y(2) y(1)], [0 0 h h], cmap(color_idx,:), 'EdgeColor','black','LineWidth',0.1);
            patch([x(2) x(3) x(3) x(2)], [y(2) y(3) y(3) y(2)], [0 0 h h], cmap(color_idx,:), 'EdgeColor','black','LineWidth',0.1);
            patch([x(3) x(4) x(4) x(3)], [y(3) y(4) y(4) y(3)], [0 0 h h], cmap(color_idx,:), 'EdgeColor','black','LineWidth',0.1);
            patch([x(4) x(1) x(1) x(4)], [y(4) y(1) y(1) y(4)], [0 0 h h], cmap(color_idx,:), 'EdgeColor','black','LineWidth',0.1);
        end
    end
end

% 3D view setup
view(3);
axis tight;
xlim([0.5, n+0.5]);
ylim([0.5, m+0.5]);
% Axis formatting
whereXticks = 1:dXticks:length(center_hist);
set(gca, 'XTick',whereXticks, ...
         'XTickLabel',num2str(center_hist(whereXticks)'));
set(gca, 'YTick', 1:m, 'YTickLabel', num2str([25,50,100]'));
zlabel('Slope Distribution','fontsize',14)
xlabel('FRET Slopes','fontsize',14)
ylabel('TRAIL (ng/ml)')
set(gca,'PlotBoxAspectRatio',[8 2 4])
colormap(hot(64))
colorbar;

% Plot classification threshold θ_ref
plot3(indkthres*[1,1], (m+1-1.5)*[1,1], [0,140], 'r-','linewidth',4);
text(indkthres+1, (m+1-1.5), 140, '\theta^{ref}','fontsize',16,'Color', [1 0 0]);
% Horizontal line marking % sensitive fraction
plot3([indkthres,length(center_hist)], ...
      [(m+1-1.5),(m+1-1.5)], ...
      [120,120], 'r-','linewidth',4)
text(indkthres+6, (m+1-1.5), 140, '% Sensitive at TRAIL 25ng/ml', ...
     'fontsize',16,'Rotation',20,'Color', [1 0 0]);


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