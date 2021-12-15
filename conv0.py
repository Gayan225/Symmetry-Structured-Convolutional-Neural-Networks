import numpy as np
np.random.seed(32189)
'''
Note: In this implementation, we initialize the upper triangular part of the matrix and form the full kernel
of size [number of rows, number of columns, number of input channels, number of output channels]
'''


class Conv_kenel_init1:
    # A Convolution layer using 3x3 filters.

    def __init__(self, kernel_size, num_channel, num_filters):
        self.kernel_size = kernel_size
        self.num_channel = num_channel
        self.num_filters = num_filters
        
        # filters is a 4d array with dimensions (kernel_size,(kernel_size+1)/2,num_channel,num_filters)
        # uses half of the glorot normal initialization
        self.bais = np.zeros(num_filters)
        self.filters_UHC = np.random.randn(kernel_size, kernel_size, int(
            num_channel/2), num_filters)*np.sqrt(0.5/(num_channel+num_filters))

    def form_full_kernel1(self):
        '''
        Generates full kernel from the upper half part of the kernel
        '''
        filters = np.concatenate((self.filters_UHC, self.filters_UHC), axis=-2)
        filters = np.reshape(
            filters, (self.kernel_size, self.kernel_size, self.num_channel, self.num_filters))

        return filters
    
    def form_bias(self):
    	return self.bais
