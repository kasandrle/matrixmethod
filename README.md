Matrix method algorithm to calculate X-ray reflectivity and transmittivity and electric fields for a stack of
homogeneous layers.


The functions in this module are:

* `reflec_and_trans`: Calculate overall reflection and transmission amplitudes.
* `reflec_and_trans_parallel`: As reflec_and_trans, but parallelized.
* `fields`: Calculate the transmission and reflection amplitudes for each layer in the stack.
* `fields_parallel`: As fields, but parallelized.
* `fields_at_positions`: Calculate the transmission and reflection amplitudes at specific positions in the stack.
* `fields_at_positions_parallel`: As fields_at_positions, but parallelized.
