


%% Set Up and Create Data Matrix D

close all; clc; clear all
% 1 . Data Preparation
% If not done yet, unzip the folder provided with the TR_piv data.
disp('Unzipping Folder TR_PIV')
unzip('TR_PIV','TR_PIV');
disp('Folder Ready')



FOLDER='TR_PIV';
n_t=2000; % number of steps.
Fs=2000; dt=1/Fs; % This data was sampled at 2kHz.
t=[0:1:n_t-1]*dt; % prepare the time axis
% Read only one snapshot to have more info.
file = [FOLDER,filesep,'Res',num2str(5,'%05.f'),'.dat'];
Data=importdata(file); % import data
DATA=Data.data; % Take only the numerical data
% This is an example of vectorial data. Hence the number of spatial points
nxny=size(DATA,1); % is the to be doubled
% at the end we will have n_s=2 * n_x * n_y
n_s=2*nxny;

%% 1. Reconstruct Mesh from file
X_S=DATA(:,1);
Y_S=DATA(:,2);
% Number of n_X/n_Y from forward differences
GRAD_X=diff(X_S); 
GRAD_Y=diff(Y_S);
% Depending on the reshaping performed, one of the two will start with
% non-zero gradient. The other will have zero gradient only on the change.
IND_X=find(GRAD_X~=0,1);
IND_Y=find(GRAD_Y~=0,1);

% Reshaping the grid from the data
    n_x=IND_X;
    n_y=(nxny/(n_x));
    Xg=reshape(X_S,[n_x,n_y]);
    Yg=reshape(Y_S,[n_x,n_y]);


% Reshape u and v
% In this dataset the mesh is repeated in each file. Hence we read from 3:
V_X=DATA(:,3); % U component
V_Y=DATA(:,4); % V component

%% 2. Assembly Data Matrix D

D=zeros(n_s,n_t);

for k=1:1:n_t
    
    % Loop over the file name
    file = [FOLDER,filesep,'Res',num2str(k,'%05.f'),'.dat'];
    % Import data
    Data=importdata(file);
    DATA=Data.data;
    % Read columns
    V_X=DATA(:,3);
    V_Y=DATA(:,4);
    D(:,k)=[V_X;V_Y]; % Assmbly the columns of D
    disp(['Reading File ',num2str(k),' of ',...
    num2str(n_t)]); % Message to follow the process

end

% Save the Data file and all the info about spatial/time discretization
save('Data.mat','D','t','dt','n_t','Xg','Yg')
 
%% Visualize entire evolution (Optional)
% We create a gif to illustrate what data we are dealing with
filename='Exercise_4.gif';
HFIG=figure(11);
HFIG.Units='normalized';
HFIG.Position=[0.3 0.3 0.5 0.5];
NA=150; % Number of steps included in the video

for k=1:5:NA
    
    disp(['Animating ',num2str(k),' of ',num2str(NA)])
    % Loop over the file name
    file = [FOLDER,filesep,'Res',num2str(k,'%05.f'),'.dat'];
    % Import data
    Data=importdata(file);
    DATA=Data.data;
    % Read columns
    V_X=reshape(DATA(:,3),[n_x,n_y]);
    V_Y=reshape(DATA(:,4),[n_x,n_y]);
    pcolor(Xg,Yg,sqrt(V_X.^2+V_Y.^2));
    shading interp
    set(gca,'YDir','reverse');
    STR=streamslice(Xg,Yg,V_X,V_Y,12,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([10 30])  
    xlim([-2 35.5])
    caxis([0 7])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[mm]$','Interpreter','Latex','fontsize',18)
    ylabel('$y[mm]$','Interpreter','Latex','fontsize',18)
    set(gcf,'color','w')


    frame = getframe(11);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

    if k == 1
       imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
       imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime', 0.1);
    end

end

close all 
clc
 
 
 
