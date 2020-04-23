
%% C. Compute Temporal Basis

% This script computes the temporal basis either via standard proper
% Orthogonal Decomposition and via Multiscale Proper Orthogonal
% Decomposition.

clear all
clc
close all

%% mPOD Decomposition
load('Data.mat') % Load the Data matrix D
load('Correlation_K.mat') % Load the correlation matrix K



%% Study the frequency content of K
%For the comments of this part look at the same file from exercise 1

Fs=1/dt; % Sampling frequency
Freq = [-n_t/2:1:n_t/2-1]*(Fs)*1/n_t; % Frequency Axis
FIG=figure(11); 
K_HAT_ABS=fliplr(abs(fftshift(fft2(K-mean(K(:))))));
imagesc(Freq,Freq,K_HAT_ABS/(n_s*n_t)) % We normalize the result
caxis([0 0.5]) % From here  downward is just for plotting purposes
axis([-600 600 -600 600])
set(gca,'XTick',-600:200:600)
set(gca,'YTick',-600:200:600)
daspect([1 1 1]) 
set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
% Label Information
xlabel('$f[Hz]$','Interpreter','Latex','fontsize',18)
ylabel('$f[Hz]$','Interpreter','Latex','fontsize',18)
set(gcf,'color','w')

drawnow


F_V=[10 350 420 500]; % This will generate four scales: H_A, H_H_1, H_H_2, H_H_3. See Sec. 3.2

Keep=[1 1 1 1]; %These are the band-pass you want to keep (Approximation is always kept).
Nf=[500 500 500 500]; % This vector collects the length of the filter kernels.

% We can visualize where these are acting
F_Bank_r = F_V*2/Fs; %(Fs/2 mapped to 1)
M=length(F_Bank_r);
HFIG=figure(11);
% Extract the diagonal of K_F
K_HAT_ABS=fliplr(abs(fftshift(fft2(K-mean(K(:))))));
% For Plotting purposes remove the 0 freqs.
ZERO_F=find(Freq==0); diag_K=abs(diag((K_HAT_ABS)));
diag_K(ZERO_F-1:ZERO_F+1)=0;
plot(Freq,diag_K./max(diag_K),'linewidth',1.2);

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

xlim([0 600])  
set(gca,'Fontname','Palatino Linotype','Fontsize',16,'Box','off','LineWidth',1)
% Label Information
xlabel('$f[Hz]$','Interpreter','Latex','fontsize',18)
ylabel('$Normalized Spectra$','Interpreter','Latex','fontsize',18)
set(gcf,'color','w')


print(HFIG,'Frequency_Splitting.png','-dpng')

% Compute the mPOD Temporal Basis
PSI_M = mPOD_FAST(K,dt,Nf,F_V,Keep,'symmetric');

% Check that this is equivalent to this
%PSI_M2 = mPOD(K,dt,Nf,F_V,Keep,'symmetric');

save('Psis_mPOD.mat','PSI_M')
   
% To make a comparison later, we also compute the POD basis
% Temporal structures are eigenvectors of K
[PSI_P,Lambda_P]=svd(K,'econ');
% The POD has the unique feature of providing the amplitude of the modes
% with no need of projection. The amplitudes are:
Sigma_P=sqrt(Lambda_P); 

save('Psis_POD.mat','PSI_P','Sigma_P')



