

%% Compute Time Correlation Matrix K

% For a Cartesian Domain, we use no Weight Matrix 
 
load('Data.mat','D') % Load Matrix D
K=D'*D; % Compute correlations
save('Correlation_K.mat','K') % Save K






