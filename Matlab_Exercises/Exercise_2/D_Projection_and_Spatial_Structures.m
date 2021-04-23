

%% D. Compute Projection.

% Given the temporal Basis for the POD and the mPOD we compute their
% spatial structures

clear all; clc; close all

% Load Data
load('Data.mat')
[n_x,n_y]=size(Xg);
n_s=size(D,1);
% Load mPOD basis
load('Psis_mPOD.mat','PSI_M')

% Load POD basis
load('Psis_POD.mat','PSI_P','Sigma_P')

 
%% Compute the spatial basis for mPOD and POD
% Projection from Eq. 2.6
R=size(PSI_M,2);
PHI_M_SIGMA_M=D*PSI_M;
% Initialize the output
PHI_M=zeros([n_s,R]);
SIGMA_M=zeros([R,R]);

 for i=1:1:R
     
     disp(['Completing mPOD Mode ',num2str(i)]);
     % Normalize the columns of C to get spatial modes
     PHI_M(:,i) = PHI_M_SIGMA_M(:,i)/norm(PHI_M_SIGMA_M(:,i));
     % Assign the norm as amplitude
     SIGMA_M(i,i) = norm(PHI_M_SIGMA_M(:,i));
     
end

% Sort the amplitudes in decreasing order
[Sort_SM,Perm]=sort(diag(SIGMA_M),'descend');

Phi_M = PHI_M(:,Perm); % Spatial Basis mPOD
Psi_M = PSI_M(:,Perm); % Temporal Basis mPOD
Sigma_VV = diag(SIGMA_M);
Sigma_VV = Sigma_VV(Perm);
Sigma_M = diag(Sigma_VV); % Amplitude for mPOD Basis



% Show some exemplary modes for mPOD. We take from 1 to 3.
HFIG=figure(1);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results mPOD Analysis';

for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the Spatial structure
    S=reshape(Phi_M(:,j),[n_y,n_x]);
    pcolor(Xg',Yg',S);
    shading interp
    daspect([1 1 1])
    ylim([-19 19])  
    xlim([-19 19])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[-]$','Interpreter','Latex')
    ylabel('$y[-]$','Interpreter','Latex')
    title(['Spatial Structure ',num2str(j)],'Interpreter','Latex')
    set(gcf,'color','white')

end



for j=1:3
    
    subplot(3,2,2*j)
    plot(t,Psi_M(:,j),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 5])
    % Label Information
    xlabel('$t[-]$','Interpreter','Latex')
    ylabel('$\widehat{\psi}_{\mathcal{M}}$','Interpreter','Latex')
    title(['T. Structure ',num2str(j)],'Interpreter','Latex')
    set(gcf,'color','white')
 
end

print(HFIG,'Results_mPOD.png','-dpng')


% Compute the spatial basis for the POD
load('Psis_POD.mat','PSI_P','Sigma_P')
R=size(PSI_P,2);
PHI_P_SIGMA_P=D*PSI_P;
% Initialize the output
Phi_P=zeros([n_s,R]);

for i=1:1:R
    
    disp(['Completing POD Mode ',num2str(i)]);
    % Normalize the columns of C to get spatial modes
    Phi_P(:,i) = PHI_P_SIGMA_P(:,i)/Sigma_P(i,i);
    % The POD does not need a normalization for the amplitude!
    
end


% Show some exemplary modes for POD. We take from 1 to 3.
HFIG=figure(3);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results POD Analysis';
for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the Spatial structure
    S=reshape(Phi_P(:,j),[n_y,n_x]);
    pcolor(Xg',Yg',S);
    shading interp
    daspect([1 1 1])
    ylim([-19 19])  
    xlim([-19 19])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[-]$','Interpreter','Latex')
    ylabel('$y[-]$','Interpreter','Latex')
    title(['Spatial Structure ',num2str(j)],'Interpreter','Latex')
    set(gcf,'color','white')

end



for j=1:3
    
    subplot(3,2,2*j)
    plot(t,PSI_P(:,j),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 5])
    % Label Information
    xlabel('$t[-]$','Interpreter','Latex')
    ylabel('$\widehat{\psi}_{\mathcal{P}}$','Interpreter','Latex')
    title(['T. Structure ',num2str(j)],'Interpreter','Latex')
    set(gcf,'color','white')

end

print(HFIG,'Results_POD.png','-dpng')


% compare this with the introduced modes... Do they look the same ??


