""" 
Automated script to run all decompositions with a given version of modulo.
This saves the result in a folder named as the version, to be further 
compared with version upgrades.


"""

import argparse
import os
import sys
import numpy as np
from modulo_vki import ModuloVKI
from modulo_vki.utils.read_db import ReadData # to read the data
from modulo_vki.utils.others import plot_grid_cylinder_flow,Plot_Field_TEXT_Cylinder

def main():
    parser = argparse.ArgumentParser(
        description='Run kDMD and DMD decompositions for a specified Modulo version.'
    )
    parser.add_argument(
        '--version', '-v',
        required=True,
        help='Version of ModuloVKI to use (e.g., v1.2.3)'
    )
    parser.add_argument(
        '--dataset-dir', '-d',
        required=True,
        help='Path to the dataset directory containing snapshots'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Base output directory where results will be saved'
    )
    args = parser.parse_args()

    # Check dataset directory exists
    if not os.path.isdir(args.dataset_dir):
        parser.error(f"Dataset directory '{args.dataset_dir}' does not exist or is not a directory.")
        
    # Create versioned output folder
    versioned_output = os.path.join(args.output_dir, args.version)
    if os.path.exists(versioned_output):
        if not os.path.isdir(versioned_output):
            parser.error(f"Output path '{versioned_output}' exists and is not a directory.")
    else:
        os.makedirs(versioned_output, exist_ok=True)
    
    
    # print('\t--------- First round of tests: Cylinder dataset ------- \t')
    
    # out_folder = versioned_output + '/T1'
    # os.makedirs(out_folder, exist_ok=True)
     
    # FOLDER= args.dataset_dir + 'Tutorial_1_2D_Cylinder_CFD_POD_DMD'

    # # Read one snapshot and plot it 
    # U=np.loadtxt(FOLDER + os.sep+ 'U_Cyl.txt')                              # U component
    # V=np.loadtxt(FOLDER + os.sep+ 'V_Cyl.txt')                              # V component
    # X=np.loadtxt(FOLDER + os.sep+ 'X_Cyl.txt')                            # X coordinates
    # Y=np.loadtxt(FOLDER + os.sep+ 'Y_Cyl.txt')                            # Y coordinates

    # # We rebuild the mesh
    # Xg,Yg=np.meshgrid(X,Y)
    # n_x=len(Y); n_y=len(X); nxny=n_x*n_y
    # n_s=2*nxny; n_t=np.shape(U)[1]

    # t=np.arange(0,n_t)*1/1000

    # # Crete the snapshot Matrix:
    # D = np.zeros((n_s, n_t))

    # for k in range(0, n_t):    
    #     D[:int(n_s/2), k] = U[:, k]
    #     D[int(n_s/2):, k] = V[:, k]
        
    # m = ModuloVKI(data=np.nan_to_num(D))

    # print('POD (K) no memory saving ... ')
    
    # pod_folder = os.path.join(out_folder, 'POD_K')
    # os.makedirs(pod_folder, exist_ok=True)
    
    # try:
    #     Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()
    # except: 
    #     Phi_POD, Psi_POD, Sigma_POD = m.POD(mode='K')
    
    # np.savez(os.path.join(pod_folder, 'out.npz'), Phi=Phi_POD, Psi=Psi_POD, Sigma=Sigma_POD)
    
    # print('OK')
    
    # print('POD (svd) no memory saving (always) ... ')
    
    # pod_folder = os.path.join(out_folder, 'POD_svd')
    # os.makedirs(pod_folder, exist_ok=True)
    
    # try:
    #     Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_svd()
    # except: 
    #     Phi_POD, Psi_POD, Sigma_POD = m.POD(mode='svd')
    
    # np.savez(os.path.join(pod_folder, 'out.npz'), Phi=Phi_POD, Psi=Psi_POD, Sigma=Sigma_POD)
    
    # print('OK')
    
    # print('DMD ... ')
    # dmd_folder = os.path.join(out_folder, 'dmd')
    # os.makedirs(dmd_folder, exist_ok=True)
     
    # try:
    #     Phi_D, Lambda, freqs, a0s = m.compute_DMD_PIP(False, F_S=1000)
    # except: 
    #     Phi_D, Lambda, freqs, a0s = m.DMD(False, F_S=1000)
        
    # np.savez(os.path.join(dmd_folder, 'out.npz'), Phi=Phi_D, 
    #          Lambda=Lambda, freqs=freqs, a0s=a0s)
    
    # # ---- Kernel POD ---- #
    # kpod_folder = os.path.join(out_folder, 'kpod')
    # os.makedirs(kpod_folder, exist_ok=True)

    # M_DIST=[1,19]
    
    # try:
    #     Phi_kPOD_1, Psi_kPOD_1, Sigma_kPOD_1,K_zeta_1 = m.compute_kPOD(M_DIST=M_DIST,k_m=1e-3,
    #                                                                    cent=True, K_out=True)
    # except: 
    #     Phi_kPOD_1, Psi_kPOD_1, Sigma_kPOD_1,K_zeta_1 = m.kPOD(M_DIST=M_DIST,k_m=1e-3,
    #                                                                    cent=True, K_out=True)
            
    # np.savez(os.path.join(kpod_folder, 'out.npz'), Phi=Phi_kPOD_1, Psi=Psi_kPOD_1, 
    #          Sigma=Sigma_kPOD_1, K_zeta=K_zeta_1)
    
    # # ----- kDMD ----- # 
    # kdmd_folder = os.path.join(out_folder, 'kdmd')
    # os.makedirs(kdmd_folder, exist_ok=True)
    
    # try:
    #    Phi_kD, Lambda_k, freqs_k, a0s_k, _ = m.kDMD(F_S=1000) 
       
    #    np.savez(os.path.join(kdmd_folder, 'out.npz'), Phi=Phi_kD, 
    #             Lambda=Lambda_k, freqs=freqs_k, a0s=a0s_k)
    # except Exception as e:
    #     print(e)
    #     print('kDMD is not available in this version of MODULO. Install MODULO>=2.1.')
        
    
    
    # # ----------------------------------
    # out_folder = versioned_output + '/T3'
    # os.makedirs(out_folder, exist_ok=True)
     
    # FOLDER= args.dataset_dir + 'Tutorial_5_2D_Cylinder_Memory_Saving'
 
    
    # print('Using Memory Saving now...')
    # n_t=13200; Fs=3000; dt=1/Fs 
    # H = 1; F = 0; C=0
    # # --- Read one sample snapshot (to get N_S)
    # Name = FOLDER+os.sep+'data'+os.sep+"Res00001.dat"
    # Dat = np.genfromtxt(Name, skip_header=H, skip_footer=F)
    # # --- Component fields (N=2 for 2D velocity fields, N=1 for pressure fields)
    # N = Dat.shape[1]
    # # --- Number of mesh points and snapshots
    # nxny = Dat.shape[0]; N_T=n_t

    # Name_Mesh=FOLDER+os.sep+'data'+os.sep+'MESH.dat'
    # Name_FIG=FOLDER+os.sep+'Cylinder_Flow_snapshot_'+str(2)+'.png'

    # # This function reads all the info about the grid
    # n_s,Xg,Yg,Vxg,Vyg,X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh,Name_FIG) 
    # # number of points in x and y
    # n_x,n_y=np.shape(Xg)

    # # Prepare 10 partitions
    # D = ReadData._data_processing(D=None,FOLDER_IN=FOLDER+os.sep+'data', 
    #                             filename='Res%05d',
    #                             N=2, N_S=2*nxny,N_T=N_T,
    #                             h = H, f = F, c=C,
    #                             N_PARTITIONS=10, MR= False,
    #                             FOLDER_OUT=FOLDER+os.sep+'MODULO_tmp')

    # # ---- POD K w/ memory saving ---- #
    # pod_ms_folder = os.path.join(out_folder, 'pod_ms')
    # os.makedirs(pod_ms_folder, exist_ok=True)
    
    # m = ModuloVKI(data=None,N_T=N_T,
    #        FOLDER_OUT=FOLDER+os.sep+'MODULO_tmp',
    #        N_S=2*nxny,
    #        n_Modes=100,
    #        N_PARTITIONS=10,eig_solver='svd_sklearn_randomized')

    # try:
    #     Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()
    # except:
    #     Phi_POD, Psi_POD, Sigma_POD = m.POD(mode='K')
    
    # np.savez(os.path.join(pod_ms_folder, 'out.npz'), 
    #          Phi=Phi_POD, 
    #          Psi=Psi_POD, Sigma=Sigma_POD)
    
    # # ---- mPOD w/ memory saving ----- #
    # mpod_ms_folder = os.path.join(out_folder, 'mpod_ms')
    # os.makedirs(mpod_ms_folder, exist_ok=True)

    # F_V=np.array([10,290,320,430,470])
    # Nf=np.array([1201, 1201,801,801,801,801]); 
    # Keep=np.array([1,1,0,1,1,0])
    # Ex=1203
    # boundaries = 'nearest'
    # MODE = 'reduced'
    
    # m = ModuloVKI(None,
    #         N_T=13200,
    #         FOLDER_OUT=FOLDER+os.sep+'MODULO_tmp',
    #         N_S=2*nxny,
    #         N_PARTITIONS=10,
    #         n_Modes = 500,           
    #         eig_solver='svd_sklearn_randomized')
    
    # try:
    #     Phi_M,Psi_M,Sigmas_M = m.compute_mPOD(Nf=Nf,
    #                                         Ex=Ex,
    #                                         F_V=F_V,
    #                                         Keep=Keep,
    #                                         SAT=5,
    #                                         boundaries=boundaries,
    #                                         MODE=MODE,dt=1/Fs,SAVE=True)
    # except:
    #     Phi_M,Psi_M,Sigmas_M = m.mPOD(Nf=Nf,
    #                                         Ex=Ex,
    #                                         F_V=F_V,
    #                                         Keep=Keep,
    #                                         SAT=5,
    #                                         boundaries=boundaries,
    #                                         MODE=MODE,dt=1/Fs,SAVE=True)
         
    # np.savez(os.path.join(mpod_ms_folder, 'out.npz'), 
    #          Phi=Phi_M, 
    #          Psi=Psi_M, Sigma=Sigmas_M)
    
    # print('\t ------ Tests dataset 1 completed ----- \t')
    # del m 
    # del D 
    print('\t--------- Second round of tests: JET PIV dataset ------- \t')
    
    out_folder = versioned_output + '/T2'
    os.makedirs(out_folder, exist_ok=True)
     
    FOLDER= args.dataset_dir + 'Tutorial_2_JET_PIV/Ex_4_TR_PIV_Jet'
    n_t = 2000
    Fs = 2000
    dt = 1/Fs 
    t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

    # --- Component fields (N=2 for 2D velocity fields, N=1 for pressure fields)
    N = 2 
    # --- Number of mesh points
    N_S = 6840
    # --- Header (H) and footer (F) to be skipped during acquisition
    H = 1; F = 0
    # --- Read one sample snapshot (to get N_S)
    Name = FOLDER+"/Res00001.dat"
    Dat = np.genfromtxt(Name, skip_header=H, skip_footer=F)

    D = ReadData._data_processing(D=None,
                                FOLDER_OUT='./',
                                FOLDER_IN=FOLDER+'/', 
                                filename='Res%05d', 
                                h=H,f=F,c=2,
                                N=2, N_S=2*Dat.shape[0],N_T=n_t)
    
    # --- Remove the mean from this dataset (stationary flow )!
    D,D_MEAN=ReadData._data_processing(D,MR=True)
    # We create a matrix of mean flow:
    D_MEAN_mat=np.array([D_MEAN, ] * n_t).transpose()    
    
    # ---------- DFT ---------- # 
    dft_folder = os.path.join(out_folder, 'dft')
    os.makedirs(dft_folder, exist_ok=True) 
    
    m = ModuloVKI(data=D)
    
    try: 
        Sorted_Freqs, Phi_F, Sorted_Sigmas = m.compute_DFT(Fs)
    except: 
        Phi_F, Sorted_Freqs, Sorted_Sigmas = m.DFT(Fs) # change of order of outputs to align modules in MODULO
        
    np.savez(os.path.join(dft_folder, 'out.npz'), Phi=Phi_F, 
             Sigmas=Sorted_Sigmas, Psi_F=Sorted_Freqs)
    
    # -------- POD (SVD) ------- #
    pod_folder = os.path.join(out_folder, 'pod')
    os.makedirs(pod_folder, exist_ok=True) 
    
    try:
        Phi_P, Psi_P, Sigma_P = m.compute_POD_svd() # POD via svd
    except:
        Phi_P, Psi_P, Sigma_P = m.POD(mode='svd')
        
    np.savez(os.path.join(pod_folder, 'out.npz'), Phi=Phi_P,
             Psi_P=Psi_P, Sigma_P=Sigma_P)
    
    # ----- SPOD Towne (CSD Matrix) ----- # 
    
    spod_folder = os.path.join(out_folder, 'spod_t')
    os.makedirs(spod_folder, exist_ok=True) 
    
    try:
        Phi_SP, Sigma_SP, Freqs_Pos = m.SPOD(mode='towne', 
                                            F_S=2000, # sampling frequency
                                            L_B=200, # Length of the chunks for time average
                                            O_B=150, # Overlap between chunks
                                            n_Modes=3, SAVE_SPOD=False) # number of modes PER FREQUENCY
    except: 
        Phi_SP, Sigma_SP, Freqs_Pos = m.compute_SPOD_t(F_S=2000, # sampling frequency
                                                    L_B=200, # Length of the chunks for time average
                                                    O_B=150, # Overlap between chunks
                                                    n_Modes=3) # number of modes PER FREQUENCY
        
    np.savez(os.path.join(spod_folder, 'out.npz'), Phi=Phi_SP, Psi_P=Freqs_Pos, Sigma_P=Sigma_SP) 

    spod_folder = os.path.join(out_folder, 'spod_t_parallel')
    os.makedirs(spod_folder, exist_ok=True)
    
    try:
        Phi_SP, Sigma_SP, Freqs_Pos = m.SPOD(mode='towne', 
                                            F_S=2000, # sampling frequency
                                            L_B=200, # Length of the chunks for time average
                                            O_B=150, # Overlap between chunks
                                            n_Modes=3, 
                                            n_processes=3) # number of modes PER FREQUENCY 
        
        np.savez(os.path.join(spod_folder, 'out.npz'), Phi=Phi_SP, Psi_P=Freqs_Pos, Sigma_P=Sigma_SP)  
    except Exception as e:
        print(e)
        print('No parallel spod in this version of MODULO, please upgrade to MODULO >= 2.1.')
        pass 
    
    # ----- SPOD Sieber (Diagonal-filtered K) ------ # 
    spod_folder = os.path.join(out_folder, 'spod_s')
    os.makedirs(spod_folder, exist_ok=True) 
     
    try:
        Phi_S, Psi_S, Sigma_S = m.SPOD(mode='sieber', 
                                       F_S=Fs, 
                                       n_Modes=25,
                                       SAVE_SPOD=False,
                                       N_O=100, 
                                       f_c=0.01,)
    except:
        Phi_S, Psi_S, Sigma_S = m.compute_SPOD_s(Fs,N_O=100,
                                               f_c=0.01,
                                               n_Modes=25,
                                               SAVE_SPOD=True)
    
    np.savez(os.path.join(spod_folder, 'out.npz'), Phi=Phi_S, Psi_P=Psi_S, Sigma_P=Sigma_S)    
    
    # ----- mPOD ----- #
    Keep = np.array([1, 1, 1, 1, 1])
    Nf = np.array([201, 201, 201, 201, 201])
    # --- Test Case Data:
    # + Stand off distance nozzle to plate
    H = 4 / 1000  
    # + Mean velocity of the jet at the outlet
    U0 = 6.5  
    # + Input frequency splitting vector in dimensionless form (Strohual Number)
    ST_V = np.array([0.1, 0.2, 0.25, 0.4])  
    # + Frequency Splitting Vector in Hz
    F_V = ST_V * U0 / H
    # + Size of the extension for the BC (Check Docs)
    Ex = 203  # This must be at least as Nf.
    dt = 1/2000; 
    boundaries = 'reflective'
    MODE = 'reduced'
    #K = np.load("./MODULO_tmp/correlation_matrix/k_matrix.npz")['K']
    
    mpod_folder = os.path.join(out_folder, 'mpod')
    os.makedirs(mpod_folder, exist_ok=True) 
      
    try: 
        Phi_M, Psi_M, Sigmas_M = m.compute_mPOD(Nf, Ex, F_V, Keep, 20 ,boundaries, MODE, dt, False)
    except: 
        Phi_M, Psi_M, Sigmas_M = m.mPOD(Nf, Ex, F_V, Keep, 20 , boundaries, MODE, dt, False) 

    np.savez(os.path.join(mpod_folder, 'out.npz'), Phi=Phi_M, Psi=Psi_M, Sigmas=Sigmas_M)
    
    # ------------------ #
    print('--- \t Second round of tests completed. --- \t')
    pass 

if __name__ == '__main__':
    main()
    
    
    
 

    

        
    
        
    
    
        
    
        
    
