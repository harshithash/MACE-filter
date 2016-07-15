# MACE-filter
This code is implemented in python. It creates a MACE filter using the training images . X matrix contains 2D-FFT coefficients
of training images. D is a square matrix that contains average power spectrum of elements . Finally H is matrix that contains 
the 2D-FFT coefficients of resulting filter.
H=(D^-1)X((X+ D^-1 X)^-1)d
