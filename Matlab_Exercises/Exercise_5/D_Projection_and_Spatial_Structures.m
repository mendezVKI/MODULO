

%% D. Compute Projection.

% Given the temporal Basis for the POD and the mPOD we compute their
% spatial structures

clear all; clc; close all

% Load Data
load('Data.mat')

% Load mPOD basis
load('Psis_mPOD.mat','PSI_M')

% Load POD basis
load('Psis_POD.mat','PSI_P','Sigma_P')

R=size(PSI_M,2);



%% START THE PARTITIONING
% Define dimension of column partitions and row partitions
dim_col = floor(n_t/partitions);
dim_row = floor(n_s/partitions);
% Define row partitions of D
dr = zeros(dim_row,n_t);


%% 1. Convert partitions dC to dR

% Compute number of blocks
if rem(n_s,partitions)>0
    tot_blocks_row = partitions + 1;
else
    tot_blocks_row = partitions;
end


if rem(n_t,partitions)>0
    tot_blocks_col = partitions + 1;
else
    tot_blocks_col = partitions;
end

% Load PSI_M
disp('Preparing projections by blocks, please wait...')

fixed = 0;

for i=1:tot_blocks_row
    
    % Eventually fix dim_row
    if i==tot_blocks_row && (n_s-dim_row*partitions>0)
        dim_row2 = n_s-dim_row*partitions;
        dr = zeros(dim_row2,n_t);
    end
    
    for b=1:tot_blocks_col
        
        load(['temp',filesep,'dc_',num2str(b),'.mat'],'di')
        
        % Compute auto-fixing indices for the blocks: the variable fixed is
        % used not to recompute them if we have to fix them in the last
        % block
        if i==tot_blocks_row && (n_s-dim_row*partitions>0) && fixed==0
            R1 = R2 + 1;
            R2 = R1 + (n_s-dim_row*partitions) - 1;
            fixed = 1;
        elseif fixed==0
            R1 = (i-1)*dim_row+1;
            R2 = i*dim_row;
        end
        
        % Same as before, but we don't need the variable fixed because if
        % the code runs this loop, it will be the last time
        if b==tot_blocks_col && (n_t-dim_col*partitions>0)
            C1 = C2 + 1;
            C2 = C1 + (n_t-dim_col*partitions) - 1;
        else
            C1 = (b-1)*dim_col+1;
            C2 = b*dim_col;
        end
        
        dr(:,C1:C2) = di(R1:R2,:);
        
    end
    %% 2. Compute partitions R of PHI_SIGMA
    PHI_SIGMA_BLOCK = dr*PSI_M;
    save(['temp',filesep,'phi_sigma_',num2str(i),'.mat'],'PHI_SIGMA_BLOCK')
end

%delete(['temp',filesep','dc_*.mat'])

%% 3. Convert partitions R to partitions C and get the SIGMA

% Ora devo convertire da righe a colonne.
% Now dim_col has to be based on R and not on n_t
dim_col = floor(R/partitions);      % Number of modes computed
dim_row = floor(n_s/partitions);
dps = zeros(n_s,dim_col);           % Vector containing phi_sigma
SIGMA_M = [];
save(['temp',filesep,'SIGMA_M.mat'],'SIGMA_M')
PHI_M = [];

% Recompute the number of column partitions (R based)
if rem(R,partitions)>0
    tot_blocks_col = partitions + 1;
else
    tot_blocks_col = partitions;
end

fixed = 0;

% The logic of this loop os the same as before
for i=1:tot_blocks_col
    
    if i==tot_blocks_col && (R-dim_col*partitions>0)
        dim_col2 = R-dim_col*partitions;
        dps = zeros(n_s,dim_col2);
    end
    
    for b=1:tot_blocks_row
        
        load(['temp',filesep,'phi_sigma_',num2str(b),'.mat'],'PHI_SIGMA_BLOCK')
        
        if i==tot_blocks_col && (R-dim_col*partitions>0) && fixed==0
            R1 = R2 + 1;
            R2 = R1 + (R-dim_col*partitions) - 1;
            fixed = 1;
        elseif fixed==0
            R1 = (i-1)*dim_col+1;
            R2 = i*dim_col;
        end
        
        if b==tot_blocks_row && (n_s-dim_row*partitions>0)
            C1 = C2 + 1;
            C2 = C1 + (n_s-dim_row*partitions) - 1;
        else
            C1 = (b-1)*dim_row+1;
            C2 = b*dim_row;
        end
        
        dps(C1:C2,:) = PHI_SIGMA_BLOCK(:,R1:R2);
        
    end
    
    
    % Get the Sigmas and the Phi
    for j=R1:R2     
        jj = j - R1 + 1;
        disp(['Completing mPOD Mode ',num2str(j),' of ',num2str(R)])
        SIGMA_M(j) = norm(dps(:,jj));
        Phi_M = dps(:,jj)/SIGMA_M(j);
        save(['temp',filesep,'phi_',num2str(j),'.mat'],'Phi_M')
        %save(['temp',filesep,'SIGMA_M.mat'],'SIGMA_M','-append')
    end
    
end


% Sort the amplitudes in decreasing order
[Sort_SM,Perm]=sort(SIGMA_M,'descend');
Psi_M = PSI_M(:,Perm); % Temporal Basis mPOD
Sigma_VV = diag(SIGMA_M);
Sigma_VV = Sigma_VV(Perm);
Sigma_M = diag(Sigma_VV); % Amplitude for mPOD Basis

% I do not need to permutate Phi_M because I can load a certain mode I want
% to export. Its index will be given by the Permutation vector.

%Phi_M = PHI_M(:,Perm); % Spatial Basis mPOD



% Show some exemplary modes for mPOD. We take from 2 to 7.
HFIG=figure(1);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results mPOD Analysis- Part 1';

for j=1:3
    load(['temp',filesep,'phi_',num2str(Perm(j+1)),'.mat'],'Phi_M')
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_M(1:n_s/2),[n_x,n_y]);
    V_Y=reshape(Phi_M(n_s/2+1:end),[n_x,n_y]);
    pcolor(Xg',Yg',sqrt(V_X'.^2+V_Y'.^2));
    shading interp
    %set(gca,'YDir','reverse');
    STR=streamslice(Xg',Yg',V_X',V_Y',5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([-13 13])  
    xlim([4 66])
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




% Show some exemplary modes for mPOD. We take from 2 to 7.
HFIG=figure(2);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results mPOD Analysis- Part 2';

for j=1:3
    load(['temp',filesep,'phi_',num2str(Perm(j+4)),'.mat'],'Phi_M')
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_M(1:n_s/2),[n_x,n_y]);
    V_Y=reshape(Phi_M(n_s/2+1:end),[n_x,n_y]);
    pcolor(Xg',Yg',sqrt(V_X'.^2+V_Y'.^2));
    shading interp
    %set(gca,'YDir','reverse');
    STR=streamslice(Xg',Yg',V_X',V_Y',5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([-13 13])  
    xlim([4 66])
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



%% START THE PARTITIONING
% Define dimension of column partitions and row partitions
dim_col = floor(n_t/partitions);
dim_row = floor(n_s/partitions);
% Define row partitions of D
dr = zeros(dim_row,n_t);


%% 1. Convert partitions dC to dR

% Compute number of blocks
if rem(n_s,partitions)>0
    tot_blocks_row = partitions + 1;
else
    tot_blocks_row = partitions;
end


if rem(n_t,partitions)>0
    tot_blocks_col = partitions + 1;
else
    tot_blocks_col = partitions;
end

% Load Psi_P
disp('Preparing projections by blocks, please wait...')

fixed = 0;

for i=1:tot_blocks_row
    
    % Eventually fix dim_row
    if i==tot_blocks_row && (n_s-dim_row*partitions>0)
        dim_row2 = n_s-dim_row*partitions;
        dr = zeros(dim_row2,n_t);
    end
    
    for b=1:tot_blocks_col
        
        load(['temp',filesep,'dc_',num2str(b),'.mat'],'di')
        
        % Compute auto-fixing indices for the blocks: the variable fixed is
        % used not to recompute them if we have to fix them in the last
        % block
        if i==tot_blocks_row && (n_s-dim_row*partitions>0) && fixed==0
            R1 = R2 + 1;
            R2 = R1 + (n_s-dim_row*partitions) - 1;
            fixed = 1;
        elseif fixed==0
            R1 = (i-1)*dim_row+1;
            R2 = i*dim_row;
        end
        
        % Same as before, but we don't need the variable fixed because if
        % the code runs this loop, it will be the last time
        if b==tot_blocks_col && (n_t-dim_col*partitions>0)
            C1 = C2 + 1;
            C2 = C1 + (n_t-dim_col*partitions) - 1;
        else
            C1 = (b-1)*dim_col+1;
            C2 = b*dim_col;
        end
        
        dr(:,C1:C2) = di(R1:R2,:);
        
    end
    %% 2. Compute partitions R of PHI_SIGMA
    PHI_SIGMA_BLOCK = dr*PSI_P;
    save(['temp',filesep,'phi_sigma_',num2str(i),'.mat'],'PHI_SIGMA_BLOCK')
end

%delete(['temp',filesep','dc_*.mat'])

%% 3. Convert partitions R to partitions C and get the SIGMA

% Ora devo convertire da righe a colonne.
dim_col = floor(R/partitions);      % Number of modes computed
dim_row = floor(n_s/partitions);
dps = zeros(n_s,dim_col);           % Vector containing phi_sigma
Phi_P = [];

% Recompute the number of column partitions (R based)
if rem(R,partitions)>0
    tot_blocks_col = partitions + 1;
else
    tot_blocks_col = partitions;
end

fixed = 0;

% The logic of this loop os the same as before
for i=1:tot_blocks_col
    
    if i==tot_blocks_col && (R-dim_col*partitions>0)
        dim_col2 = R-dim_col*partitions;
        dps = zeros(n_s,dim_col2);
    end
    
    for b=1:tot_blocks_row
        
        load(['temp',filesep,'phi_sigma_',num2str(b),'.mat'],'PHI_SIGMA_BLOCK')
        
        if i==tot_blocks_col && (R-dim_col*partitions>0) && fixed==0
            R1 = R2 + 1;
            R2 = R1 + (R-dim_col*partitions) - 1;
            fixed = 1;
        elseif fixed==0
            R1 = (i-1)*dim_col+1;
            R2 = i*dim_col;
        end
        
        if b==tot_blocks_row && (n_s-dim_row*partitions>0)
            C1 = C2 + 1;
            C2 = C1 + (n_s-dim_row*partitions) - 1;
        else
            C1 = (b-1)*dim_row+1;
            C2 = b*dim_row;
        end
        
        dps(C1:C2,:) = PHI_SIGMA_BLOCK(:,R1:R2);
        
    end
    
    
    % Get the Sigmas and the Phi
    for j=R1:R2
        jj = j - R1 + 1;
        disp(['Completing POD Mode ',num2str(j),' of ',num2str(R)])
        Phi_P = dps(:,jj)/norm(dps(:,jj));
        save(['temp',filesep,'phi_',num2str(j),'.mat'],'Phi_P')
    end
    
end

% Show some exemplary modes for POD. We take from 2 to 7.
HFIG=figure(3);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results POD Analysis- Part 1';

for j=1:3
    load(['temp',filesep,'phi_',num2str(j+1),'.mat'],'Phi_P')
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_P(1:n_s/2),[n_x,n_y]);
    V_Y=reshape(Phi_P(n_s/2+1:end),[n_x,n_y]);
    pcolor(Xg',Yg',sqrt(V_X'.^2+V_Y'.^2));
    shading interp
    %set(gca,'YDir','reverse');
    STR=streamslice(Xg',Yg',V_X',V_Y',5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([-13 13])  
    xlim([4 66])
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


% Show some exemplary modes for POD. We take from 2 to 7.

HFIG=figure(4);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Results POD Analysis- Part 2';

for j=1:3
    load(['temp',filesep,'phi_',num2str(j+4),'.mat'],'Phi_P')
    subplot(3,2,2*j-1)
    % Reconstruct the fields from the columns
    V_X=reshape(Phi_P(1:n_s/2),[n_x,n_y]);
    V_Y=reshape(Phi_P(n_s/2+1:end),[n_x,n_y]);
    pcolor(Xg',Yg',sqrt(V_X'.^2+V_Y'.^2));
    shading interp
    %set(gca,'YDir','reverse');
    STR=streamslice(Xg',Yg',V_X',V_Y',5,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([-13 13])  
    xlim([4 66])
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

rmdir temp s