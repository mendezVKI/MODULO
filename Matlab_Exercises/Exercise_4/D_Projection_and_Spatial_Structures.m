

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



% First 3 Modes Results mPOD
HFIG=figure(1);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results mPOD Analysis- Part 1';

for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_M(1:n_s/2,j+1),[n_x,n_y]);
    V_Y=reshape(Phi_M(n_s/2+1:end,j+1),[n_x,n_y]);
    pcolor(Xg,Yg,sqrt(V_X.^2+V_Y.^2));
    shading interp
    set(gca,'YDir','reverse');
    STR=streamslice(Xg,Yg,V_X,V_Y,5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([10 30])  
    xlim([-2 35.5])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[mm]$','Interpreter','Latex')
    ylabel('$y[mm]$','Interpreter','Latex')
    title(['Spatial Structure ',num2str(j+1)],'Interpreter','Latex')
    set(gcf,'color','white')

end



Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis


for j=1:3
    
    subplot(3,2,2*j)
    plot(Freq,abs(fftshift(fft(Psi_M(:,j+1)))),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 600])
    % Label Information
    xlabel('$f[Hz]$','Interpreter','Latex')
    ylabel('$\widehat{\psi}_{\mathcal{M}}$','Interpreter','Latex')
    title(['Frequency in the T. Structure ',num2str(j+1)],'Interpreter','Latex')
    set(gcf,'color','white')
    
end

print(HFIG,'Results_mPOD_1.png','-dpng')



% First 3 Modes Results mPOD
HFIG=figure(2);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results mPOD Analysis- Part 2';

for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_M(1:n_s/2,j+4),[n_x,n_y]);
    V_Y=reshape(Phi_M(n_s/2+1:end,j+4),[n_x,n_y]);
    pcolor(Xg,Yg,sqrt(V_X.^2+V_Y.^2));
    shading interp
    set(gca,'YDir','reverse');
    STR=streamslice(Xg,Yg,V_X,V_Y,5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([10 30])  
    xlim([-2 35.5])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[mm]$','Interpreter','Latex')
    ylabel('$y[mm]$','Interpreter','Latex')
    title(['Spatial Structure ',num2str(j+4)],'Interpreter','Latex')
    set(gcf,'color','white')
    
end



Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis


for j=1:3
    
    subplot(3,2,2*j)
    plot(Freq,abs(fftshift(fft(Psi_M(:,j+4)))),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 600])
    % Label Information
    xlabel('$f[Hz]$','Interpreter','Latex')
    ylabel('$\widehat{\psi}_{\mathcal{M}}$','Interpreter','Latex')
    title(['Frequency in the T. Structure ',num2str(j+4)],'Interpreter','Latex')
    set(gcf,'color','white')

end

print(HFIG,'Results_mPOD_2.png','-dpng')


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


% First 3 Modes Results POD
HFIG=figure(3);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results POD Analysis- Part 1';

for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_P(1:n_s/2,j+1),[n_x,n_y]);
    V_Y=reshape(Phi_P(n_s/2+1:end,j+1),[n_x,n_y]);
    pcolor(Xg,Yg,sqrt(V_X.^2+V_Y.^2));
    shading interp
    set(gca,'YDir','reverse');
    STR=streamslice(Xg,Yg,V_X,V_Y,5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([10 30])  
    xlim([-2 35.5])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[mm]$','Interpreter','Latex')
    ylabel('$y[mm]$','Interpreter','Latex')
    title(['Spatial Structure ',num2str(j+1)],'Interpreter','Latex')
    set(gcf,'color','white')
 
end



Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis


for j=1:3
    
    subplot(3,2,2*j)
    plot(Freq,abs(fftshift(fft(PSI_P(:,j+1)))),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 600])
    % Label Information
    xlabel('$f[Hz]$','Interpreter','Latex')
    ylabel('$\widehat{\psi}_{\mathcal{P}}$','Interpreter','Latex')
    title(['Frequency in the T. Structure ',num2str(j+1)],'Interpreter','Latex')
    set(gcf,'color','white')
    
end

print(HFIG,'Results_POD_1.png','-dpng')

% First 3 Modes Results POD
HFIG=figure(4);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results POD Analysis- Part 2';

for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_P(1:n_s/2,j+4),[n_x,n_y]);
    V_Y=reshape(Phi_P(n_s/2+1:end,j+4),[n_x,n_y]);
    pcolor(Xg,Yg,sqrt(V_X.^2+V_Y.^2));
    shading interp
    set(gca,'YDir','reverse');
    STR=streamslice(Xg,Yg,V_X,V_Y,5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([10 30])  
    xlim([-2 35.5])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[mm]$','Interpreter','Latex')
    ylabel('$y[mm]$','Interpreter','Latex')
    title(['Spatial Structure ',num2str(j+4)],'Interpreter','Latex')
    set(gcf,'color','white')

end



Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis


for j=1:3
    
    subplot(3,2,2*j)
    plot(Freq,abs(fftshift(fft(PSI_P(:,j+4)))),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 600])
    % Label Information
    xlabel('$f[Hz]$','Interpreter','Latex')
    ylabel('$\widehat{\psi}_{\mathcal{P}}$','Interpreter','Latex')
    title(['Frequency in the T. Structure ',num2str(j+4)],'Interpreter','Latex')
    set(gcf,'color','white')

end

print(HFIG,'Results_POD_2.png','-dpng')


