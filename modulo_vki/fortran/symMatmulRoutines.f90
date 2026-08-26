
module sym_routines
  use iso_c_binding
  implicit none
  integer, parameter :: dp = kind(1.0d0)

contains

  !!! REAL ROUTINES !!!
  subroutine compute_ata(A, m, n, K)
    ! Computes K = A^T * A using symmetry, returning full symmetric matrix.
    ! Can give up to 2x speedup compared to Numpy K = np.matmul(A.T,A)
    !
    ! Parameters
    ! ----------
    ! A : real(8)
    !   m by n input matrix
    !   
    ! m : integer > 0
    !   number of rows in A
    !
    ! n : integer > 0
    !   number of columns in A 
    !
    ! Returns
    ! -------
    ! K : real(8)
    !   n x n output matrix, K = A^T * A 

    integer, intent(in) :: m, n
    real(8), intent(in) :: A(m, n)
    real(8), intent(out) :: K(n, n)

    ! Compute upper triangle using BLAS dsyrk
    call dsyrk('U', 'T', n, m, 1.0_dp, A, m, 0.0_dp, K, n)

    ! Copy upper triangle to lower
    call symmetrize_upper_to_lower(K, n)
  end subroutine compute_ata


  subroutine symmetrize_upper_to_lower(K, n)
    ! Helper routine to fill lower triangle of real K based on upper triangle.
    !
    ! Parameters
    ! ----------
    ! K : real(8)
    !   Allocated matrix containing the upper triangle of 
    !   n x n matrix K = A^T * A 
    !
    ! n : integer > 0
    !   number of rows and columns in K 
    !
    ! Returns
    ! -------
    ! K : real(8)
    !   Filled symmetric n x n output matrix

    integer, intent(in) :: n
    real(8), intent(inout) :: K(n, n)
    integer :: i, j

    do j = 1, n-1
      do i = j+1, n
        K(i, j) = K(j, i)
      end do
    end do
  end subroutine symmetrize_upper_to_lower


  !!! COMPLEX ROUTINES !!!
  subroutine compute_aha(A, m, n, K)
    ! Computes K = A^H * A using Hermitian symmetry, returning full Hermitian symmetric matrix.
    ! Can give up to 2x speedup compared to Numpy K = np.matmul(A.conj().T,A)
    !
    ! Parameters
    ! ----------
    ! A : complex(8)
    !   m by n input matrix
    !   
    ! m : integer > 0
    !   number of rows in A
    !
    ! n : integer > 0
    !   number of columns in A 
    !
    ! Returns
    ! -------
    ! K : complex(8)
    !   n x n output matrix, K = A^H * A 

    integer, intent(in) :: m, n
    complex(8), intent(in)  :: A(m, n)
    complex(8), intent(out) :: K(n, n)
    ! compute upper triangle using zherk
    call zherk('U', 'C', n, m, 1.0_dp, A, m, 0.0_dp, K, n)
    ! Copy and conjugate upper triangle to lower
    call hermitize_upper_to_lower(K, n)
  end subroutine compute_aha


  subroutine compute_persym_aha_fullmat(A, m, n, K)
    ! Computes K = A^H * A using Hermitian symmetry and persymmetry based on symmetries
    ! in DFT of real signal, returning full Hermitian symmetric matrix of the full frequency 
    ! range of data. Can give up to 2.7x speedup compared to Numpy K = np.matmul(A.conj().T,A)
    !
    ! Parameters
    ! ----------
    ! A : complex(8)
    !   m by n input matrix
    !   
    ! m : integer > 0
    !   number of rows in A
    !
    ! n : integer > 0
    !   number of columns in A 
    !
    ! Returns
    ! -------
    ! K : complex(8)
    !   n x n output matrix, K = A^H * A 

    integer, intent(in) :: m, n
    complex(8), intent(in)  :: A(m, n)
    complex(8), intent(out) :: K(n, n)
    ! Local variables
    integer :: i, j
    integer :: nloc

    if (modulo(n,2) == 0) then   ! Includes Nyquist
      nloc = (n-2)/2
      ! zherk call to get the top triangle (including DC)
      call zherk('U','C', nloc+2, m, 1.0_dp, A(1,1), m, 0.0_dp, K(1,1), n)
      ! zgemm call to get the top square (including DC and Nyq)
      call zgemm('C','N', nloc+2, nloc, m, (1.0_dp,0.0_dp), A(1,1), m, A(1,nloc+3), m, (0.0_dp,0.0_dp), K(1,nloc+3), n)

      ! mirror top triangle across anti-diagonal (w/o first row -- DC)
      do j = nloc+3, n
        do i = nloc+3,j
          K(i,j) = K(n+2-j,n+2-i)
        end do 
      end do

    else  ! w/o Nyquist
      nloc = (n-1)/2
      ! zherk call to get the top triangle (including DC)
      call zherk('U','C', nloc+1, m, 1.0_dp, A(1,1), m, 0.0_dp, K(1,1), n)
      ! zgemm call to get the top square (including DC and Nyq)
      call zgemm('C','N', nloc+1, nloc, m, (1.0_dp,0.0_dp), A(1,1), m, A(1,nloc+2), m, (0.0_dp,0.0_dp), K(1,nloc+2), n)

      ! mirror top triangle across anti-diagonal (w/o first row -- DC)
      do j = nloc+2, n
        do i = nloc+2,j
          K(i,j) = K(n+2-j,n+2-i)
        end do 
      end do
    end if
    
    ! Hermitian symmetry
    call hermitize_upper_to_lower(K, n)

  end subroutine compute_persym_aha_fullmat


  subroutine compute_persym_aha_dc(A, m, n, K)
    ! Computes K = A^H * A using Hermitian symmetry and persymmetry based on symmetries
    ! in DFT of real signal, returning full Hermitian symmetric matrix of the frequency 
    ! band containing DC frequency. Can give up to 2.7x speedup compared to Numpy 
    ! K = np.matmul(A.conj().T,A)
    !
    ! Parameters
    ! ----------
    ! A : complex(8)
    !   m by n input matrix
    !   
    ! m : integer > 0
    !   number of rows in A
    !
    ! n : integer > 0
    !   number of columns in A 
    !
    ! Returns
    ! -------
    ! K : complex(8)
    !   n x n output matrix, K = A^H * A 
    
    integer, intent(in) :: m, n
    complex(8), intent(in)  :: A(m, n)
    complex(8), intent(out) :: K(n, n)
    ! Local variables
    integer :: i, j
    integer :: nloc

    nloc = (n-1)/2  ! DC included, no fNyq
    ! zherk call to get the top triangle (including DC)
    call zherk('U','C', nloc+1, m, 1.0_dp, A(1,1), m, 0.0_dp, K(1,1), n)
    ! zgemm call to get the top square (including DC)
    call zgemm('C','N', nloc+1, nloc, m, (1.0_dp,0.0_dp), A(1,1), m, A(1,nloc+2), m, (0.0_dp,0.0_dp), K(1,nloc+2), n)

    ! mirror top triangle across anti-diagonal (w/o first row -- DC)
    do j = nloc+2, n
      do i = nloc+2,j
        K(i,j) = K(n+2-j,n+2-i)
      end do 
    end do

    ! Hermitian symmetry
    call hermitize_upper_to_lower(K, n)

  end subroutine compute_persym_aha_dc

  
  subroutine compute_persym_aha_band(A, m, n, K)
    ! Computes K = A^H * A using Hermitian symmetry and persymmetry based on symmetries
    ! in DFT of real signal, returning full Hermitian symmetric matrix of the frequency 
    ! band without DC and Nyq frequency. Can give up to 2.7x speedup compared to Numpy
    ! K = np.matmul(A.conj().T,A)
    !
    ! Parameters
    ! ----------
    ! A : complex(8)
    !   m by n input matrix
    !   
    ! m : integer > 0
    !   number of rows in A
    !
    ! n : integer > 0
    !   number of columns in A 
    !
    ! Returns
    ! -------
    ! K : complex(8)
    !   n x n output matrix, K = A^H * A 
    
    integer, intent(in) :: m, n
    complex(8), intent(in)  :: A(m, n)
    complex(8), intent(out) :: K(n, n)
    ! Local variables
    integer :: i, j
    integer :: nloc

    nloc = n/2  ! no DC, no fNyq
    ! zherk call to get the top triangle
    call zherk('U','C', nloc, m, 1.0_dp, A(1,1), m, 0.0_dp, K(1,1), n)
    ! zgemm call to get the top square
    call zgemm('C','N', nloc, nloc, m, (1.0_dp,0.0_dp), A(1,1), m, A(1,nloc+1), m, (0.0_dp,0.0_dp), K(1,nloc+1), n)

    ! mirror top triangle across anti-diagonal
    do j = nloc+1, n
      do i = nloc+1,j
        K(i,j) = K(n+1-j,n+1-i)   ! no DC to exclude
      end do 
    end do

    ! Hermitian symmetry
    call hermitize_upper_to_lower(K, n)

  end subroutine compute_persym_aha_band


  subroutine compute_persym_aha_nyq(A, m, n, K)
    ! Computes K = A^H * A using Hermitian symmetry and persymmetry based on symmetries
    ! in DFT of real signal, returning full Hermitian symmetric matrix of the frequency 
    ! band including Nyq frequency. Can give up to 2.7x speedup compared to Numpy
    ! K = np.matmul(A.conj().T,A)
    !
    ! Parameters
    ! ----------
    ! A : complex(8)
    !   m by n input matrix
    !   
    ! m : integer > 0
    !   number of rows in A
    !
    ! n : integer > 0
    !   number of columns in A 
    !
    ! Returns
    ! -------
    ! K : complex(8)
    !   n x n output matrix, K = A^H * A 
    
    integer, intent(in) :: m, n
    complex(8), intent(in)  :: A(m, n)
    complex(8), intent(out) :: K(n, n)
    ! Local variables
    integer :: i, j
    integer :: nloc

    nloc = (n-1)/2  ! no DC, includes fNyq
    ! zherk call to get the top triangle (including Nyq)
    call zherk('U','C', nloc+1, m, 1.0_dp, A(1,1), m, 0.0_dp, K(1,1), n)
    ! zgemm call to get the top square (including Nyq)
    call zgemm('C','N', nloc+1, nloc, m, (1.0_dp,0.0_dp), A(1,1), m, A(1,nloc+2), m, (0.0_dp,0.0_dp), K(1,nloc+2), n)

    ! mirror top triangle across anti-diagonal
    do j = nloc+2, n
      do i = nloc+2,j
        K(i,j) = K(n+1-j,n+1-i) ! no DC to exclude
      end do 
    end do

    ! Hermitian symmetry
    call hermitize_upper_to_lower(K, n)

  end subroutine compute_persym_aha_nyq


  subroutine hermitize_upper_to_lower(K, n)
    ! Helper routine to fill lower triangle of complex K based on upper triangle.
    !
    ! Parameters
    ! ----------
    ! K : complex(8)
    !   Allocated matrix containing the upper triangle of 
    !   n x n matrix K = A^H * A 
    !
    ! n : integer > 0
    !   number of rows and columns in K 
    !
    ! Returns
    ! -------
    ! K : real(8)
    !   Filled Hermitian symmetric n x n output matrix

    integer, intent(in) :: n
    complex(8), intent(inout) :: K(n, n)
    integer :: i, j
    do j = 1, n-1
      do i = j+1, n
        K(i, j) = conjg(K(j, i))
      end do
    end do
  end subroutine hermitize_upper_to_lower

end module sym_routines

