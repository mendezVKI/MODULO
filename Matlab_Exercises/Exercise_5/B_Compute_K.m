

%% Compute Time Correlation Matrix K

% For a Cartesian Domain, we use no Weight Matrix. 
% Therefore the correlation is computed simply via matrix multiplication.
clc; close all;

disp('Computing Correlation Matrix by blocks...')

% As before, define the dimension of columns partitions
dim_col = floor(n_t/partitions);

for k=0:tot_blocks_col-1
    
    % Load the partition
    name = ['temp',filesep,'dc_',num2str(k+1),'.mat'];
    load(name,'di')
    
    % Set the indices of the matrix for the computation
    ind_start = k*dim_col+1;
    ind_end = ind_start+dim_col-1;
    % Fix the values of dim_col e ind_end in case the last partition is
    % smaller than the previous one
    if k==tot_blocks_col-1 && n_t-dim_col*partitions>0
        dim_col = n_t-dim_col*partitions;
        ind_end = ind_start+dim_col-1;
    end
    % Perform matrix multiplication for the elements along the diagonal
    K(ind_start:ind_end,ind_start:ind_end) = di'*di;
    
    % Compute how many blocks there are out of the diagonal
    %outblocks = tot_blocks_col-(k+1);
    % Define which block has to be loaded first
    block = (k+1)+1;
    % Copy the actual partition of D in another vector not to overwrite
    dj = di;
    
    % Load the other partitions for the matrix multilpication to compute
    % the elements of out the diagonal
    while block<=tot_blocks_col
        name = ['temp',filesep,'dc_',num2str(block),'.mat'];
        load(name,'di')
        ind_start_out = (block-1)*dim_col+1;
        ind_end_out = ind_start_out+dim_col-1;
        if block==tot_blocks_col && n_t-dim_col*partitions>0
            dim_col = n_t-dim_col*partitions;
            ind_end_out = ind_start_out+dim_col-1;
            di = di(:,1:dim_col);
        end
        K(ind_start:ind_end,ind_start_out:ind_end_out) = dj'*di;
        % Perform symmetry
        K(ind_start_out:ind_end_out,ind_start:ind_end) = K(ind_start:ind_end,ind_start_out:ind_end_out)';
        block = block + 1;
        
        % Reset the value of dim_col in case it has been recomputed
        dim_col = floor(n_t/partitions);
        
    end
    
end

save('Correlation_K.mat','K')




