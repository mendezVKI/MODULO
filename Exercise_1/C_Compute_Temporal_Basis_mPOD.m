
%% C. Compute Temporal Basis

% This script computes the temporal basis either via standard proper
% Orthogonal Decomposition or via Multiscale Proper Orthogonal
% Decomposition.

clear all
clc
close all

%% mPOD Decomposition
load('Data.mat') % Load the Data matrix D
load('Correlation_K.mat') % Load the correlation matrix K



%% Study the frequency content of K
% We plot || K_hat || to look for frequencies
% This could be done via Matrix Multiplication but we use fft2 for fast
% computation. If you want to see the algebraic form of the code,
% do not hesitate to contact me at menez@vki.ac.be

Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis
FIG=figure(11); 
K_HAT_ABS=abs(fftshift(fft2(K-mean(K(:)))));
imagesc(Freq,Freq,K_HAT_ABS/(numel(D))) % We normalize the result
caxis([0 1]) % From here  downward is just for plotting purposes
axis([-0.5 0.5 -0.5 0.5])
set(gca,'XTick',-0.5:0.25:0.5)
set(gca,'YTick',-0.5:0.25:0.5)
daspect([1 1 1]) 
     set(gca, ...
      'Fontname', 'Palatino Linotype', ...
      'Fontsize', 16, ...
       'Box'         , 'off'     , ...   
        'LineWidth'   , 1         )
    % Label Information
     xlabel('$\hat{f}[-]$','Interpreter','Latex','fontsize',18)
    ylabel('$\hat{f}[-]$','Interpreter','Latex','fontsize',18)
   set(gcf,'color','w')

drawnow
% This matrix shows that there are two dominant frequencies in the problem.
% We set a frequency splitting to divide these two portions of the
% spectra. (See Sec. )

% A good example is :
F_V=[0.1 0.35]; % This will generate three scales: H_A, H_H_1, H_H_2. See Sec. 3.2
Keep=[1 1]; %These are the band-pass you want to keep (Approximation is always kept).
% If Keep=[1 0], then we will remove the highest portion of the spectra
% (e.g. f>0.4). If Keep=[0,0] then only the Approximation is kept.
Nf=[500 500]; % This vector collects the length of the filter kernels.
% Observe that Nf could be set as it is usually done in Wavelet Theory.

% Compute the mPOD Temporal Basis
PSI_M = mPOD(K,dt,Nf,F_V,Keep,'symmetric');


save('Psis_mPOD.mat','PSI_M')
   
% To make a comparison later, we also compute the POD basis
% Temporal structures are eigenvectors of K
[PSI_P,Lambda_P]=svd(K,'econ');
% The POD has the unique feature of providing the amplitude of the modes
% with no need of projection. The amplitudes are:
Sigma_P=sqrt(Lambda_P); 

save('Psis_POD.mat','PSI_P','Sigma_P')


return

F_V




