


%% Set Up and Create Data Matrix D

close all; clc; clear all
% 1 . Data Preparation
% If not done yet, unzip the folder provided with the TR_piv data.
disp('Unzipping Folder Data')
websave('CFD_Vortices.zip','https://osf.io/zgujk/download');
unzip('CFD_Vortices.zip','Data');
disp('Folder Ready')

% IMPORTANT: To keep the size of the GitHub folder within a reasonable
% limit, the dataset has been reduced. Therefore, you might not obtains
% exactly the same results as in the paper. If you want to have the
% complete set, contact mendez@vki.ac.be


% Observe that for demonstration purposes we use half of the data that was
% used in the article.
FOLDER='Data';
n_t=1030; % number of steps.
Fs=50; dt=1/Fs; % This data was sampled at 2kHz.
t=[0:1:n_t-1]*dt; % prepare the time axis

% Read only one snapshot to have more info.
file = [FOLDER,filesep,'Step',num2str(5,'%04.f'),'.dat'];
Data=importdata(file); % import data
DATA=Data.data; % Take only the numerical data
% This is an example of scalar data. Hence the number of spatial points is
n_s=size(DATA,1); 
% Reshape the scalar quantity
% In this dataset the mesh is not in each file. Hence we read from 1:
Omega=DATA(:,1); % Vorticity Field


%% 1. Reconstruct Mesh from file
% The mesh on the other hand is stored in a different file
file_M = [FOLDER,filesep,'XY_grid.dat'];
Data=importdata(file_M); % import data
XY=Data.data; % Take only the numerical data
X_S=XY(:,1);
Y_S=XY(:,2);

% Number of n_X/n_Y from forward differences
GRAD_X=diff(X_S); 
GRAD_Y=diff(Y_S);
% Depending on the reshaping performed, one of the two will start with
% non-zero gradient. The other will have zero gradient only on the change.
IND_X=find(GRAD_X~=0,1);
IND_Y=find(GRAD_Y~=0,1);
% Reshape from the column of the data file.
n_x=IND_X; n_y=(n_s/(n_x));
Xg=reshape(X_S,[n_x,n_y]); % Cartesian Mesh Grid, X Coordinates
Yg=reshape(Y_S,[n_x,n_y]); % Cartesian Mesh Grid, Y Coordinates


%% 2. Assembly Data Matrix D
D=zeros(n_s,n_t);
for k=1:1:n_t
    
    % Loop over the file name
    file = [FOLDER,filesep,'Step',num2str(k-1,'%04.f'),'.dat'];
    % Import data
    Data=importdata(file);
    DATA=Data.data;
    % Read columns
    Omega=DATA(:,1); % Vorticity Field
    D(:,k)=Omega; % Assmbly the columns of D
    disp(['Reading File ',num2str(k),' of ',...
    num2str(n_t)]); % Message to follow the process

end


% Save the Data file and all the info about spatial/time discretization
save('Data.mat','D','t','dt','n_t','Xg','Yg')
 

%% Visualize entire evolution (Optional)
% We create a gif to illustrate what data we are dealing with

filename='Exercise_3.gif';
HFIG=figure(11);
HFIG.Units='normalized';
HFIG.Position=[0.3 0.3 0.5 0.5];
NA=1000; % Number of steps included in the video

for k=1:2:NA
    
    disp(['Animating ',num2str(k),' of ',num2str(NA)])
    % Loop over the file name
    file = [FOLDER,filesep,'Step',num2str(k,'%04.f'),'.dat'];
    % Import data
    Data=importdata(file);
    DATA=Data.data;
    % Read column
    Omega=reshape(DATA(:,1),[n_x n_y]); % Vorticity Field


    pcolor(Xg,Yg,Omega);
    shading interp
    daspect([1 1 1])
    ylim([-20 20])  
    xlim([-20 20])
    caxis([-40 40])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[-]$','Interpreter','Latex','fontsize',18)
    ylabel('$y[-]$','Interpreter','Latex','fontsize',18)
    set(gcf,'color','w')
    title('$\omega[-]$','Interpreter','Latex','fontsize',24)


    frame = getframe(11);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

     if k == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
     else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime', 0.05);
     end

end

close all 
clc
 
 
 
