
%% C. Compute Temporal Basis

% This script computes the temporal basis via standard Proper
% Orthogonal Decomposition and via Multiscale Proper Orthogonal
% Decomposition.

clear all
clc
close all

%% mPOD Decomposition
load('Data.mat') % Load the Data matrix D
load('Correlation_K.mat') % Load the correlation matrix K



%% Study the frequency content of K
% We plot || K_hat || to look for frequencies
% This could be done via Matrix Multiplication (as it is done in the paper)
% but we use fft2 for fast computation. If you want to see the algebraic form of the code,
% do not hesitate to contact me at mendez@vki.ac.be

Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis
FIG=figure(11); 
K_HAT_ABS=abs(fftshift(fft2(K-mean(K(:)))));
imagesc(Freq,Freq,K_HAT_ABS/(numel(D))) % We normalize the result
caxis([0 1]) % From here  downward is just for plotting purposes
axis([-0.5 0.5 -0.5 0.5]) % We set the axis for better visualization
set(gca,'XTick',-0.5:0.25:0.5)
set(gca,'YTick',-0.5:0.25:0.5)
daspect([1 1 1]) 
set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
% Label Information
xlabel('$\hat{f}[-]$','Interpreter','Latex','fontsize',18)
ylabel('$\hat{f}[-]$','Interpreter','Latex','fontsize',18)
set(gcf,'color','w')
drawnow
% This matrix shows that there are two dominant frequencies in the problem.
% We set a frequency splitting to divide these two portions of the
% spectra. (See Sec. 3.1-3.2)

% A good example is :
F_V=[0.1 0.25]; % This will generate three scales: H_A, H_H_1, H_H_2. See Sec. 3.2
Keep=[1 0]; %These are the band-pass you want to keep (Approximation is always kept).
% If Keep=[1 0], then we will remove the highest portion of the spectra (H_H_2)
% If Keep=[0 1], then we will remove the intermediate portion (H_H_1)
% The current version does not allow to remove the Approximation (the low).
% If you are willing to do that you could do it in two steps: 
% First you compute the approximation using Keep=[0,0]. Then you remove it
% from the original data.
Nf=[500 500]; % This vector collects the length of the filter kernels.
% Observe that Nf could be set as it is usually done in Wavelet Theory.
% For example, using eq. A.5.

% We can visualize where these are acting
F_Bank_r = F_V*2/Fs; %(Fs/2 mapped to 1)
M=length(F_Bank_r); % Number of scales
% Plot the transfer functions along the diagonal of K (similar to Fig 1)
HFIG=figure(11);
% Extract the diagonal of K_F
K_HAT_ABS=fliplr(abs(fftshift(fft2(K-mean(K(:))))));
% For Plotting purposes remove the 0 freqs.
ZERO_F=find(Freq==0); diag_K=abs(diag((K_HAT_ABS)));
diag_K(ZERO_F-1:ZERO_F+1)=0;
plot(Freq,diag_K./max(diag_K),'linewidth',1.2);
% Loop over the scales to show the transfer functions
for m=1:length(Keep)
    
    hold on
    % Generate the 1d filter for this 
    if m==1
       h_A=fir1(Nf(m),F_Bank_r(m),'low')'; % 1d Kernel for Low Pass
       h1d_H{m} = fir1(Nf(m),[F_Bank_r(m),F_Bank_r(m+1)],'bandpass')';
       plot(Freq,fftshift(abs(fft(h_A,n_t))),'linewidth',1.5)
       plot(Freq,fftshift(abs(fft(h1d_H{m},n_t))),'linewidth',1.5)
     elseif  m>1 && m<M 
       % This is the 1d Kernel for Band pass
       h1d_H{m} = fir1(Nf(m),[F_Bank_r(m),F_Bank_r(m+1)],'bandpass')';
       plot(Freq,fftshift(abs(fft(h1d_H{m},n_t))))
    else
       % This is the 1d Kernel for High Pass (last scale)
       h1d_H{m} = fir1(Nf(m),[F_Bank_r(m)],'high')';
       plot(Freq,fftshift(abs(fft(h1d_H{m},n_t))),'linewidth',1.5)
    end

end 

xlim([0 0.4])  % show only the positive part 
set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
% Label Information
xlabel('$f[-]$','Interpreter','Latex','fontsize',18)
ylabel('$Normalized Spectra$','Interpreter','Latex','fontsize',18)
set(gcf,'color','w')

print(HFIG,'Frequency_Splitting.png','-dpng')


% Compute the mPOD Temporal Basis
PSI_M = mPOD_FAST(K,dt,Nf,F_V,Keep,'symmetric');


save('Psis_mPOD.mat','PSI_M')
   
% To make a comparison later, we also compute the POD basis
% Temporal structures are eigenvectors of K
[PSI_P,Lambda_P]=svd(K,'econ');
% The POD has the unique feature of providing the amplitude of the modes
% with no need of projection. The amplitudes are:
Sigma_P=sqrt(Lambda_P); 

% Obs: svd and eig on a symmetric positive matrix are equivalent.
save('Psis_POD.mat','PSI_P','Sigma_P')


