function PSI_M_Final = mPOD(K,dt,Nf,F_V,Keep,BC)
%mPOD is the function to compute the Multiscale Proper Orthogonal
%Decomposition
%% Input Parameters
% 
% * K- Temporal Correlation Matrix, D'*W*D )
% * dt - Is the time step between the columns of D.
% * Nf- Is Vector Containig the length of the filter kernels (e.g. eq. A 5)
% * FV - Frequency splitting Vector, see Sec. 3.2
% * Keep - Vector controlling which band pass filter scale will be removed.
% * BC - Boundary condition for the filter. From Matlab's imfilter documentation:
% 
%       'symmetric'  Input array values outside the bounds of the array
%                    are computed by mirror-reflecting the array across
%                    the array border.
%
%       'replicate'  Input array values outside the bounds of the array
%                    are assumed to equal the nearest array border
%                    value.
%
%       'circular'   Input array values outside the bounds of the array
%                    are computed by implicitly assuming the input array
%                    is periodic.

n_t=size(K,1); % length of the temporal vector
M=length(F_V); % This is the number of scales 
% First map the frequency vector onto [-1,1].
Fs=1/dt; % Sampling Frequency
F_Bank_r = F_V*2/Fs; %(Fs/2 mapped to 1)


%% 1. Prepare all the Kernels of the Filter Bank 
disp('Preparing Kernels...')

for m=1:1:M
    
    disp(['Preparing Kernel ',num2str(m),' of ',num2str(M)]);
    
    if (m==1 && M>1)
        h1d_L = fir1(Nf(m),F_Bank_r(m),'low')'; % 1d Kernel for Low Pass
        h_A = h1d_L*h1d_L'; % 2d Kernel for Low Pass (Approximation)
        h1d_H{m} = fir1(Nf(m),[F_Bank_r(m),F_Bank_r(m+1)],'bandpass')';
        h_D{m} = h1d_H{m}*h1d_H{m}'; % 2d Kernel for Low Pass (Diagonal Detail)
    elseif (m==1 && M==1)
        h1d_L = fir1(Nf(m),F_Bank_r(m),'low')'; % 1d Kernel for Low Pass
        h_A = h1d_L*h1d_L'; % 2d Kernel for Low Pass (Approximation)
        h1d_H{m} = fir1(Nf(m),[F_Bank_r(m)],'high')';
        h_D{m} = h1d_H{m}*h1d_H{m}'; % 2d Kernel for Low Pass (Diagonal Detail)
    elseif  m>1 && m<M
        % This is the 1d Kernel for Band pass
        h1d_H{m} = fir1(Nf(m),[F_Bank_r(m),F_Bank_r(m+1)],'bandpass')';
        h_D{m} = h1d_H{m}*h1d_H{m}'; % 2d Kernel for Low Pass (Diagonal Detail)
    else
        % This is the 1d Kernel for High Pass (last scale)
        h1d_H{m} = fir1(Nf(m),[F_Bank_r(m)],'high')';
        h_D{m} = h1d_H{m}*h1d_H{m}'; % 2d Kernel for High Pass (finest scale)
    end   
    
end


% Freq=[-n_t/2:1:n_t/2-1]*(Fs)*1/n_t;
% %%%Visualize some of them if you want (comment/uncomment)
% Hfig=figure(11)
% pcolor(Freq,Freq,abs(freqz2(h_D{1},[n_t n_t])));
% shading interp; daspect([1 1 1])
% xlim([-1 1])
% ylim([-1 1])



%% 2. Perform MRA: Compute Scales contributions
%Large Scale Portion
disp('Getting K_L')
K_L = imfilter(K,h_A,BC);
%Band-pass Portions (using Keep Vector)
Ind=find(Keep==1);
for ll=1:1:length(Ind)
    
    MEX = ['Getting K_H',num2str(Ind(ll))];
    disp(MEX)
    %Do not take the full spectral frame
    K_H{ll} = imfilter(K,h_D{Ind(ll)},BC); 
    
end

%% 3. Diagonalize all the Scales
disp('Eigenspace of K_L')
% Estimate the span of its eigenbasis from the frequency content
[h,~] = freqz(h1d_L,size(K_L,1));             
H=abs(h)/max(abs(h));
N_S=length(find(H>0.05));
[Psi_L Sigma_L ~] = svds(K_L,rank(K_L));
% Release some memory
clear K_L


for ll=1:1:length(Ind)
    
    MEX = ['Eigenspace of K_H',num2str(Ind(ll))];
    disp(MEX)
    % Repeat the previous estimation and diagonalization for all the scales     
    [h,w] = freqz(h1d_H{ll},n_t); 
    H=abs(h)/max(abs(h));
    N_S=length(find(H>0.1));
    [Psi_H{ll} Sigma_H{ll} ~] = svds(K_H{Ind(ll)},rank(K_H{Ind(ll)}));
    % Release some memory
    K_H{Ind(ll)}={};
    
end

%% 4. Mount the global mPOD basis
disp('Mounting the global basis...')
PSI_M = [Psi_L(:,1:end)];
Sigma_v_M = [diag(Sigma_L(1:1:end,1:1:end))];

for ll=2:1:length(Ind)+1
    
    PSI_M = [PSI_M, Psi_H{ll-1}(:,1:end)];
    Sigma_v_M = [Sigma_v_M;diag(Sigma_H{ll-1}(1:1:end,1:1:end))];
    
end

%Sort energies descending order
[Sort_SM,Perm] = sort(Sigma_v_M,'descend');
PSI_Mo = PSI_M(:,Perm);
%First reinforce orthogonality (Phase-correction)
[Q R]=qr(PSI_Mo,0); % The matrix R should be as close as possible to Identiy.
PSI_M_Final=Q;
disp('mPOD Basis ready!')



end

