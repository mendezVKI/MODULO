Theory
---------------------------------

The DFT is the only not strictly data-driven decomposition present in MODULO. In fact, the temporal basis :math:`\mathbf{\psi}_F` is the Fourier basis, computed *a priori*. 
This reads:

.. math:: 
    \bm{\psi}_F = \frac{1}{\sqrt{n_t}} \exp{(2 \pi f_r t) 

in which :math:`f_r=r\Delta f` is the frequency of the :math:`r`-th Fourier mode, :math:`\Delta f=f_s/n_t` is the frequency resolution, and :math:`n_t` is the number of time samples. 
The term :math:`\frac{1}{\sqrt{n_t}}` is a normalization factor, which ensures that the Fourier basis is orthonormal, e.g. :math:`\|\mathbf{\psi}_F\|^2=1`.

Considering the DFT of the signal at one specific location :math:`\mathbf{x}_i`, the Fourier coefficients are computed as:

.. math:: 
    \bm{C}_F(\bm{x}_i) = \frac{1}{\sqrt{n_t}} \sum_{t=0}^{n_t-1} \bm{u}(\bm{x}_i,t) \exp{(-2 \pi f_r t)}

that effectively projects the signal onto the Fourier basis. For each of these projections, one can compute the norm as:

.. math::
    \sigma_F = \|\bm{C}_F\| 

that leads to the normalized projected fields:

.. math:: 
    \bm{\Phi}_F = \frac{\bm{C}_F}{\sigma_F}\;,

that are the spatial structure of the DFT. This completes the DFT decomposition, presented here as a matrix factorization:

.. math:: 
    \mathbf{D} = \bm{\Phi}_F \bm{\Sigma}_F \bm{\Psi}_F^T\;,

where :math:`\mathbf{\Sigma}_F` is a diagonal matrix containing the modal amplitudes :math:`\sigma_F`.


