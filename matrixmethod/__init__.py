r"""Matrix method algorithm to calculate X-ray reflectivity and transmittivity and electric fields for a stack of
homogeneous layers.


The functions in this module are:

reflec_and_trans: Calculate overall reflection and transmission amplitudes.
reflec_and_trans_parallel: As reflec_and_trans, but parallelized.
fields: Calculate the transmission and reflection amplitudes for each layer in the stack.
fields_parallel: As fields, but parallelized.
fields_at_positions: Calculate the transmission and reflection amplitudes at specific positions in the stack.
fields_at_positions_parallel: As fields_at_positions, but parallelized.


Matrix method algorithm for x-ray reflectivity and transmittivity as described by A. Gibaud and G. Vignaud in
J. Daillant, A. Gibaud (Eds.), "X-ray and Neutron Reflectivity: Principles and Applications", Lect. Notes Phys. 770
(Springler, Berlin Heidelberg 2009), DOI 10.1007/978-3-540-88588-7, chapter 3.2.1 "The Matrix Method".

Conventions used:
I'm following the conventions of A. Gibaud and G. Vignaud cited above:
There is a stack of j=0..N media on a substrate S, with j=0 and S being infinite. The interface between j and j+1
is Z_{j+1}, so Z_1 is the interface between the topmost layer (i.e. usually air or vacuum) and the first sample layer.
Electromagnetic waves are represented by their electric field \vec{E}, which is divided in one part travelling
downwards, \vec{E}^- and one travelling upwards, \vec{E}^+.

\vec{E}^{-/+} = A^{-/+} \exp\( +i(\omega t - k_{\text{in}x,j} x - k_\{\text{in}z,j} z) \) \, \hat{e}_y

The magnitude of the electric fields (which is time-independent) is denoted by:
U(-/+ k_{\text{in}z,j}, z) = A^{-/+}_j \exp(-/+ ik_{\text{in}z,j} z)

using
p_{j, j+1} = \frac{k_{z,j} + k_{z,j+1}}{2k_{z,j}}
m_{j, j+1} = \frac{k_{z,j} - k_{z,j+1}}{2k_{z,j}}

the refraction matrix \RR_{j, j+1} is given by:
 \( \begin{pmatrix}
  U(k_{z,j}, Z_{j+1}) \\
  U(-k_{z,j}, Z_{j+1})
 \end{pmatrix} \)
 =
 \( \begin{pmatrix}  % this is \RR_{j, j+1}
  p_{j, j+1} & m_{j, j+1} \\
  m_{j, j+1} & p_{j, j+1}
 \end{pmatrix} \)
 \( \begin{pmatrix}
  U(k_{z,j+1}, Z_{j+1}) \\
  U(-k_{z,j+1}, Z_{j+1})
 \end{pmatrix} \)

while the translation matrix \TT is defined as
 \( \begin{pmatrix}
  U(k_{z,j}, z) \\
  U(-k_{z,j}, z)
 \end{pmatrix} \)
 =
 \( \begin{pmatrix}  % this is \TT_{j}
  exp(-ik_{z,j} h) & 0 \\
  0 & exp(ik_{z,j} h)
 \end{pmatrix} \)
 \( \begin{pmatrix}
  U(k_{z,j}, z+h) \\
  U(-k_{z,j}, z+h)
 \end{pmatrix} \)

such that the transfer matrix \MM is
 \MM = \prod_{j=0}^N \( \RR_{j,j+1} \TT_{j+1} \) \RR_{N,s}
 =
 \( \begin{pmatrix}
  M_{11} & M_{12} \\
  M_{21} & M_{22}
 \end{pmatrix} \)

with this, the reflection coefficient is:
 r = \frac{M_{12}}{M_{22}}
and the transmission coefficient is:
 t = \frac{1}{M_{22}}
"""

from .mm_numba import reflec_and_trans, reflec_and_trans_parallel, fields, fields_parallel, fields_at_positions, \
    fields_at_positions_parallel
