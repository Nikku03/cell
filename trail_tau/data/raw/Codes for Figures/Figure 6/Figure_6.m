%% =====================================================================
% Script: Figure_6.m
% Purpose: Produce the different panels of Figure 6 from the paper
%
% Description:
%   This script loads the required data, runs the simulations,
%   and generates the figures included in the manuscript.

%  Specifically, this script generates:
   %  (1) The theoretical surfaces relating parameters K_deg, FLIP(0), and pC8(0)
   %  (2) The theoretical curves FLIP(0) vs pC8(0) and compares to the projection obtained from data (see these data obtained curves in Figure S5)
% ======================================================================

clear all
close all
clc

%% ------------------- General setting -------------------------------

% Population parameters
% common_parameters = [rK1bK1, rK2bK2, rK3bK3, rK2K1, rK3K1, alphaR_3, alphaC8, K_FRET]

% Individual parameters
% params = [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] 

% Rescaling time parameter - see Eq. (2) and (3) in the paper
K1 = 0.007325300696406; %unit measure = 1/[min]


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
K_FRET = common_parameters(8); %In the paper: alpha_2

%Initial receptors number
R0 = 32000; %Unit:
%Initial condition for C8
C80 = 30; %Unit:


% ************ Load estimated cell-specific parameters for some cells
% ************ (pC80,FLIP0,alpha0, alpha1, K_deg)

% Load the longitudinal experimental data for the selected cells 
% First line: FRET signal of cell 7
% Second line: FRET signal of cell 47
% Third line: FRET signal of cell 87
% Fourth line: FRET signal of cell 89
load("FRET_R_50ng.mat")

% Load the Individual parameters for each cell 
% First line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 7
% Second line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 47
% Third line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 87
% Fourth line: [pC8(0),FLIP(0),alpha_0,alpha_1,K_deg] of cell 89
load("R_50ng_ind_par.mat")


% ******** Choose one cell to plot

ipar = 3;
ind_par = R_50ng_ind_par(ipar,:);

%RE-DEFINE PARAMETERS
pC80 = ind_par(1);
FLIP0 = ind_par(2);
alpha0 = ind_par(3);
alpha1 = ind_par(4);
K_deg = ind_par(5);


%% ******* COMPUTING AND PLOTTING THEORETICAL SURFACE 

%% ******* Following equations (15),(16)
  
       %%Some assumptions:
       %%Approximation: assuming all exponentials are zero, ie., variables are at their highest max
  %%Approximation: Linear degradation, approximating the new saturated function

       
%Very dense grid
%xx = [0.2:0.01:3.5]*10^4 ;   %FLIP0
%yy = [0.2:0.01:4.4]*10^5 ;   %pC80

%Less dense grid
xx = [0.2:0.1:4]*10^4 ;   %FLIP0
yy = [1:0.1:4.4]*10^5 ;   %pC80

Lx = length(xx);
Ly = length(yy);


beta0TR25 = rK2K1*750*yy.^2;
beta0TR50 = rK2K1*1500*yy.^2;
beta0TR100 = rK2K1*3000*yy.^2;

lambda1 = alpha0 + rK2bK2*rK2K1 +rK2K1*xx;


  for ix = 1:Lx
     for iy = 1:Ly
		lambda0(ix,iy) = rK1bK1 +rK2K1*yy(iy)^2 +rK3K1*xx(ix)^3;

  
 C8M=2;

  auxA1=4*alphaC8;
  auxA2= alpha0+alpha1*rK2K1*xx(ix)/(alpha1+rK2bK2*rK2K1);
zzA25(iy,ix) = auxA2.*beta0TR25(iy)./( lambda0(ix,iy)*lambda1(ix) )/C8M - K_FRET;
zzA25(iy,ix) = auxA1*zzA25(iy,ix) ;

zzA50(iy,ix) = auxA2.*beta0TR50(iy)./( lambda0(ix,iy)*lambda1(ix) )/C8M - K_FRET;
zzA50(iy,ix) = auxA1*zzA50(iy,ix) ;

zzA100(iy,ix) = auxA2.*beta0TR100(iy)./( lambda0(ix,iy)*lambda1(ix) )/C8M - K_FRET;
zzA100(iy,ix) = auxA1*zzA100(iy,ix);


  end
  end

  %% ******* COMPUTING AND PLOTTING FLIP(0) VS. pC8(0)
  %% ******* Following equation (16)
  
  Kdegc=1345;
  xxf = [0.5:0.1:4]*1e4; %flip
  C8M=  [0.2,0.22,0.25];  0.1225;  
  

auxA1=4*alphaC8./(C8M*Kdegc);
auxA2= alpha0+alpha1*rK2K1*xxf/(alpha1+rK2bK2*rK2K1);
auxA3=rK3K1*(xxf.^3)/rK2K1;
auxA425=auxA1(1)*auxA2*750./(rK2K1*xxf)-1;
auxA450=auxA1(2)*auxA2*1500./(rK2K1*xxf)-1;
auxA4100=auxA1(3)*auxA2*3000./(rK2K1*xxf)-1;
xxp25 = sqrt(auxA3./auxA425);
xxp50 = sqrt(auxA3./auxA450);
xxp100 =sqrt(auxA3./auxA4100);

%Keep only real branch
for i=1:length(xxp25)
	if isreal(xxp25(i))==1
	lastreal=i;
        end
end

% ******* ADD EXPERIMENTAL DATA CURVES
% ******* EXPERIMENTAL STRAIGHT LINES OBTAINED  FROM DATA (see code for Figure 5)
% ******* 2D projections,  Flip0 as function of pC80

%Use equation of hyperplane with mean Kdeg=1420 to plot projection of hyperplane

  %Parameters, as estimated from previous code, for Figure 5
  par3_25ng =[1.0302e-05,  -7.1323e-04, -0.0096, 20.9403];
  par3_50ng =[-4.5924e-06,  8.5309e-05,  0.0010, -2.4290];
  par3_100ng =[-1.5660e-05,   1.1793e-04,  0.0011, -2.5685];

  %xxp3=linspace(0.5*10^5,10*10^5,40);
  xxp3=linspace(0.5*10^5,4.5*10^5,40);

  yy2d_25ng = (-par3_25ng(1)*xxp3 -par3_25ng(3)*1420 -par3_25ng(4))/par3_25ng(2);
  yy2d_50ng = (-par3_50ng(1)*xxp3 -par3_50ng(3)*1420 -par3_50ng(4))/par3_50ng(2);
  yy2d_100ng = (-par3_100ng(1)*xxp3 -par3_100ng(3)*1420 -par3_100ng(4))/par3_100ng(2);


%% ******* TO CONSTRUCT FIGURE 6A    
  
figure(1);
surf(yy,xx,zzA25');
hold on;
surf(yy,xx,zzA50');
hold on;
surf(yy,xx,zzA100');
view([-37.5,30]);  %view(200,17);
ylabel('FLIP0 (number of molecules)','fontsize',16);
xlabel('pC80 (number of molecules)','fontsize',16);
zlabel('K_{deg} a.u.','fontsize',16);
zlim([0 1500]);

figure(2); 
hold on;
plot(xxp25(1:lastreal),xxf(1:lastreal),'Color',[0.6 0 0],'LineStyle','--','linewidth',3);
plot(xxp50,xxf,'Color',[0.8 0 0],'LineStyle','--','linewidth',3);
plot(xxp100,xxf,'Color',[1 0 0],'LineStyle','--','linewidth',3);
hold on;
plot(xxp3,yy2d_25ng,'Color',[0.6 0 0],'linewidth',3);
plot(xxp3,yy2d_50ng,'Color',[0.8 0 0],'linewidth',3);
plot(xxp3,yy2d_100ng,'Color',[1 0 0],'linewidth',3);
legend(' 25 ng/ml, theor',' 50 ng/ml, theor','100 ng/ml, theor',' 25 ng/ml, exp',' 50 ng/ml, exp','100 ng/ml, exp','fontsize',10,'Location','southeast');
box on
ylabel('FLIP0 (number of molecules)','fontsize',16);
xlabel('pC80 (number of molecules)','fontsize',16);
zlabel('K_{deg} a.u.','fontsize',16);
xlim([0.5 4.5]*1e5);
ylim([1 4]*1e4);

figure(3);
subplot(1,2,1)
surf(yy,xx,zzA25');
hold on;
surf(yy,xx,zzA50');
hold on;
surf(yy,xx,zzA100');
view([-37.5,30]);  %view(200,17);
ylabel('FLIP0 (number of molecules)','fontsize',16);
xlabel('pC80 (number of molecules)','fontsize',16);
zlabel('K_{deg} a.u.','fontsize',16);
zlim([0 1500]);

figure(3); 
subplot(1,2,2)
hold on;
plot(xxp25(1:lastreal),xxf(1:lastreal),'Color',[0.6 0 0],'LineStyle','--','linewidth',3);
plot(xxp50,xxf,'Color',[0.8 0 0],'LineStyle','--','linewidth',3);
plot(xxp100,xxf,'Color',[1 0 0],'LineStyle','--','linewidth',3);
hold on;
plot(xxp3,yy2d_25ng,'Color',[0.6 0 0],'linewidth',3);
plot(xxp3,yy2d_50ng,'Color',[0.8 0 0],'linewidth',3);
plot(xxp3,yy2d_100ng,'Color',[1 0 0],'linewidth',3);
legend(' 25 ng/ml, theor',' 50 ng/ml, theor','100 ng/ml, theor',' 25 ng/ml, exp',' 50 ng/ml, exp','100 ng/ml, exp','fontsize',10,'Location','southeast');
box on
ylabel('FLIP0 (number of molecules)','fontsize',16);
xlabel('pC80 (number of molecules)','fontsize',16);
zlabel('K_{deg} a.u.','fontsize',16);
xlim([0.5 4.5]*1e5);
ylim([1 4]*1e4);

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

