import numpy as np
np.random.seed(32189)
'''
Note: In this implementation, we initialize the upper triangular part of the matrix and form the full kernel
of size [number of rows, number of columns, number of input channels, number of output channels]
'''


class Conv_kenel_init2:
    # A Convolution layer using 3x3 filters.

    def __init__(self, kernel_size, num_channel, num_filters):
        self.kernel_size = kernel_size
        self.num_channel = num_channel
        self.num_filters = num_filters
        
        # filters is a 4d array with dimensions (kernel_size,(kernel_size+1)/2,num_channel,num_filters)
        # uses glorot uniform initialization
        self.bais = np.zeros(num_filters)
        self.filters_UH = np.random.randn(kernel_size, int(
            (kernel_size+1)/2), num_channel, num_filters)*np.sqrt(0.5/(num_channel+num_filters))

    def form_full_kernel2(self):
        '''
        Generates full kernel from the upper half part of the kernel
        '''
        output = np.zeros((self.kernel_size, self.kernel_size,
                           self.num_channel, self.num_filters))
        filter_UH = np.reshape(self.filters_UH, (self.kernel_size *
                                                 int((self.kernel_size+1)/2), self.num_channel, self.num_filters))
        counter = -1

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if (i <= j):
                    counter = counter+1
                    output[i, j, :, :] = filter_UH[counter, :, :]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if (i < j):
                    output[j, i, :, :] = output[i, j, :, :]

        return output

        def form_bias(self):
            return self.bais
