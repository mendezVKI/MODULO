


%% Set Up and Create Data Matrix D

close all; clc; clear all
 %% 1 . Data Preparation
% This dataset is composed of three modes which we can construct
% separately. These modes are orthogonal in space and in time: that is,
% they are already POD modes.... will the POD recognize some POD modes when
% you provide some? The answer is no: if they have same (actually similar)
% energy, the POD will be confused!


L = 40; %Computational domain
m = 2^8; %Number of discretizations in x and y
n = m^2; %Total size of matrix
xyspan2 = linspace(-L/2,L/2,m+1);
xyspan = xyspan2(1:m);
[Xg,Yg] = meshgrid(xyspan,xyspan); % Grid for the test case
%% Build Source 1 
sigma_Y=5; sigma_X=5; % Spreading of the spatial Gaussian (they are all equal)
% Mesh info
dt=0.01; n_t=512; n_x=m; n_y=m; % Parameters of the Space/Time Mesh
% Prepare dataset 1
D_1=zeros(n_x*n_y,n_t); % Initialize the Data Matrix
t=(0:1:n_t-1)*dt; % Prepare the time sequence

% Parameters for the Gaussians
offset2=-10;
S1 = exp(-((Xg+offset2)/sigma_X).^2 - ((Yg-offset2).^2/sigma_Y.^2));
S1=S1/norm(S1(:)); % Normalize the spatial structure
S_1_t=sin(2*pi*15*t); % Define temporal structure
S_1_t=S_1_t.*exp(-(t-3).^20/2); % //
S_1_t=S_1_t/norm(S_1_t); % Normalize the temporal structure

% To visualize this spatial structure uncomment the following
% HFIG=figure(1);
% pcolor(Xg,Yg,S1)
% colorbar
% shading interp
% daspect([1 1 1])

close all
for i=1:1:n_t % Loop over time
    
    S_time= S_1_t(i);
    S_Space=S_time*S1; % Make the Spatial structure evolve
    % Assembly D1 Matrix
    D1(:,i)=reshape(S_Space,[numel(S_Space),1]); 
%    figure(1) % If you want to see this mode
%    pcolor(Xg,Yg,S_Space)
%    shading interp
%    caxis([-1,1])
%    daspect([1 1 1])
%    drawnow
    MEX=['Source 1, step ',num2str(i),' of ',num2str(n_t)];
    disp(MEX)
    
end

%% Build Source 2--------------------------------------------
% Repeat the same construction but with different time evolution and
% different spatial location.
offset2=10;
S2 = exp(-((Xg+offset2)/sigma_X).^2 - ((Yg-offset2).^2/sigma_Y.^2));
S2=S2/norm(S2(:));
S_2_t=1*sin(2*pi*0.1*t-pi/3);
S_2_t=S_2_t/norm(S_2_t); 

% To visualize this spatial structure uncomment the following
% HFIG=figure(1);
% pcolor(Xg,Yg,S2)
% colorbar
% shading interp
% daspect([1 1 1])
close all

for i=1:1:n_t % This is a repetition of the previous loop for Mode 2
    
    S_time= S_2_t(i);
    S_Space=S_time*S2;
    D2(:,i)=reshape(S_Space,[numel(S_Space),1]);
%    figure(1) % If you want to see this mode
%    pcolor(X,Y,S_Space)
%    shading interp
%    caxis([-1,1])
%    daspect([1 1 1])
%    drawnow
    MEX=['Source 2, step ',num2str(i),' of ',num2str(n_t)];
    disp(MEX)
   
end


%% Build Source 3--------------------------------------------
% Same as before, but different temporal structure an location
offset2=0;
S3 = exp(-((Xg+offset2)/sigma_X).^2 - ((Yg-offset2).^2/sigma_Y.^2));
S3=S3/norm(S3(:));
S_3_t=(t-mean(t)).^2.*sin(2*pi*7*t);
S_3_t=-S_3_t/norm(S_3_t); 


% To make it funnier, take orthogonalize the temporal basis also.
[Q R]=qr([S_1_t;S_2_t; S_3_t]');
% This is a small variant from the original paper, but is further stress
% the point: right now we have modes that are orthogonal in space
% and time. This means that they are already POD modes.

% If R is approximately equal to the identiy, the original basis was
% already orthogonal.


% Have a look at the results
% plot(Q(:,1))
% hold on
% plot(Q(:,2))
% hold on
% plot(Q(:,3))

% We for the temporal structures to be orthonormal
T1=Q(:,1);
T2=Q(:,2);
T3=Q(:,3);


close all
for i=1:1:n_t % Repeat the usual loop in time.
    
    S_time= S_3_t(i);
    S_Space=S_time*S3;
    % Assembly D3 Matrix
    D3(:,i)=reshape(S_Space,[numel(S_Space),1]);
%    figure(1) % If you want to see this mode
%    pcolor(Xg,Yg,S_Space)
%    shading interp
%    caxis([-0.026,0.026])
%    daspect([1 1 1])
%    colorbar
%    drawnow
    
    MEX=['Source 3, step ',num2str(i),' of ',num2str(n_t)];
    disp(MEX)
    
end

%% Print the introduced modes

% First 3 Modes Results POD
HFIG=figure(3);
HFIG.Units='normalized';
HFIG.Position=[0.1 0.1 0.65 0.75];
HFIG.Name='Introduced Modes';

for j=1:3
    
    subplot(3,2,2*j-1)
    % Reconstruct the Spatial structure
    pcolor(Xg',Yg',eval(['S',num2str(j)]));
    shading interp
    daspect([1 1 1])
    ylim([-19 19])  
    xlim([-19 19])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    xlabel('$x[-]$','Interpreter','Latex')
    ylabel('$y[-]$','Interpreter','Latex')
    title(['Introduced Spatial Structure ',num2str(j)],'Interpreter','Latex')
    set(gcf,'color','white')
    
end



for j=1:3
    
    subplot(3,2,2*j)
    plot(t,eval(['T',num2str(j)]),'linewidth',1.5)
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    xlim([0 5])
    % Label Information
    xlabel('$t[-]$','Interpreter','Latex')
    ylabel('$T Structure$','Interpreter','Latex')
    title(['Introduced T. Structure ',num2str(j)],'Interpreter','Latex')
    set(gcf,'color','white')
 
end

print(HFIG,'Introduced_Modes.png','-dpng')



%% Sum up the data

D=(D1+D2+D3);
MAX=max(D(:));
% Normalize to have reasonable numbers
D=1/MAX*D;



% Save the Data file and all the info about spatial/time discretization
save('Data.mat','D','t','dt','n_t','Xg','Yg')

%% Visualize entire evolution (Optional)
% We create a gif to illustrate what data we are dealing with
% return <--- uncomment this if you want to prevent the gif
filename='Exercise_2.gif';
HFIG=figure(11);
HFIG.Units='normalized';
HFIG.Position=[0.3 0.3 0.5 0.5];
NA=500; % Number of steps included in the video
for k=1:5:NA
    
    disp(['Animating ',num2str(k),' of ',num2str(NA)])

    pcolor(Xg',Yg',reshape(D(:,k),[n_y,n_x]));
    shading interp
    daspect([1 1 1])
    ylim([-19 19])  
    xlim([-19 19])
    caxis([-1 1])
    set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
    % Label Information
    set(gca,'XTick',[-15:5:15])
    set(gca,'YTick',[-15:5:15])
    xlabel('$x[-]$','Interpreter','Latex','fontsize',18)
    ylabel('$y[-]$','Interpreter','Latex','fontsize',18)
    set(gcf,'color','w')
    %title('$\omega[-]$','Interpreter','Latex','fontsize',24)

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
 
 
 
