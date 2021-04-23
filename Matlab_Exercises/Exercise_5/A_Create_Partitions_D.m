


%% Set Up and Create Partitions of Data Matrix D

close all; clc; clear all
% 1 . Data Preparation
% If not done yet, unzip the folder provided with the TR_piv data.
disp('Unzipping Folder TR_PIV_Cylinder')
websave('Ex_5_TR_PIV_Cylinder.zip','https://osf.io/47ftd/download');
unzip('Ex_5_TR_PIV_Cylinder.zip','Data');
disp('Folder Ready')


FOLDER='Data';
n_t=13200; % number of steps.
Fs=3000; dt=1/Fs; % This data was sampled at 3kHz.
t=[0:1:n_t-1]*dt; % prepare the time axis
% Read only one mesh file
file = [FOLDER,filesep,'MESH.dat'];
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
IND_X=find(abs(GRAD_X)>1);
IND_Y=find(abs(GRAD_Y)>1);

% Reshaping the grid from the data
    n_x=IND_X(1);
    n_y=(nxny/(n_x));
    Xg=reshape(X_S,[n_x,n_y]);
    Yg=reshape(Y_S,[n_x,n_y]);

%% 2. Start reading the column partitions

% Number of partitions for the memory saving: you can change it.
% The values below are suggested.

    partitions = 4;    
    % partitions = 8;
    % partitions = 12;

    dim_col = floor(n_t/partitions);    % Compute the dimension of each column partitions

    
    blocco = 1;     % Initialize the count for the blocks to be saved
    
    % Compute the effective number of blocks
    if rem(n_t,partitions)>0
        tot_blocks_col = partitions + 1;
    else
        tot_blocks_col = partitions;
    end
    
    files = dir(fullfile(FOLDER,'*.dat'));    % Takes the name of the dat files
    files(1) = [];                            % Remove the file containing the mesh
    
    mkdir('temp');                            % Create temporary folder
    
    for i=1:dim_col:n_t

        % dim_col of the small matrices of D
        dim_col = floor(n_t/partitions);
        di = zeros(n_s,dim_col);
        
        % Fix the value of dim_col if it is bigger than the remaining
        % files to be read
        if (n_t-i)<=dim_col-1
            dim_col = n_t-i+1;
            di = zeros(n_s,dim_col);
        end
        
        %Loop to construct di
        for kkk=1:dim_col
                       
            A = importdata([FOLDER,filesep,files(i+kkk-1).name]);
            Data = A.data;
            
            V_X = Data(:,1);
            V_Y = Data(:,2);
            

            % Allocate partition
            di(:,kkk) = [reshape(V_X,numel(V_X),1);reshape(V_Y,numel(V_Y),1)];
            
            disp(['Reading File ',num2str(i+kkk-1),' of ',...
                num2str(n_t)]); % Message to follow the process
        end

        
        name = ['temp',filesep,'dc_',num2str(blocco),'.mat'];
        save(name,'di')
        clear di
        blocco = blocco + 1;

    end

    save('Data.mat','t','dt','n_t','n_s','n_x','n_y','Xg','Yg','partitions');
 
    
    
%% Visualize entire evolution (Optional)
% We create a gif to illustrate what data we are dealing with
filename='Exercise_5.gif';
HFIG=figure(11);
HFIG.Units='normalized';
HFIG.Position=[0.3 0.3 0.5 0.5];
NA=2000; % Number of steps included in the video

for k=1:100:NA
    
    disp(['Animating ',num2str(k),' of ',num2str(NA)])
    % Loop over the file name
    file = [FOLDER,filesep,'Res',num2str(k,'%05.f'),'.dat'];
    % Import data
    Data=importdata(file);
    DATA=Data.data;
    % Read columns
    V_X=reshape(DATA(:,1),[n_x,n_y]);
    V_Y=reshape(DATA(:,2),[n_x,n_y]);
    pcolor(Xg',Yg',sqrt(V_X'.^2+V_Y'.^2));
    shading interp
    %set(gca,'YDir','reverse');
    STR=streamslice(Xg',Yg',V_X',V_Y',12,'cubic');
    set(STR,'LineWidth',0.7,'Color','k');
    ylim([-13 13])  
    xlim([4 66])
    %caxis([0 7])
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
 
 
 
