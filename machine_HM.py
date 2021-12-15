# -*- coding: utf-8 -*-
"""
The code was created on Sep 15 for submission.
A machine that uses symetric kernel CNN with hyperparameter match architecture.
"""
######
from __future__ import print_function

import argparse
import numpy as np
import time, sys, os, datetime
from sklearn.metrics import confusion_matrix
from matplotlib import colors
import matplotlib.pyplot as plt


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("tensorflow version :",tf.__version__)

from makebatchesT import *
from customT import *
from arch_HM import *
from datanames import *

np.random.seed(32189)
tf.set_random_seed(32189)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default = None, type=str, required = True)
parser.add_argument("--iterations", default = 2000, type = int)
parser.add_argument("--displaysteps", default = 1, type = int)
parser.add_argument("--batchsize", default = 1, type = int)
parser.add_argument("--maxlength", default = 500, type = int) # None = no max length
parser.add_argument('--noBN', dest='BN', default=True, action='store_false')
parser.add_argument('--LSTMlayers', default = 1, type = int)
parser.add_argument("--weight", default = 5, type = int)
parser.add_argument("--reg", default = 0.00001, type = float)
parser.add_argument("--regtype", default = 'l2', type = str)
parser.add_argument("--lr", default= 0.001, type=float)
parser.add_argument("--load", default=False, action='store_true')
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--lrdecay", default=False, action='store_true')
args = parser.parse_args()

dataset = args.dataset
iterations = args.iterations
lr = args.lr
reg = args.reg
loadmodel = args.load
maxlength = args.maxlength
batchsize = args.batchsize
weightint = args.weight
BN = args.BN
threshold = args.threshold
LSTMlayers = args.LSTMlayers
lrdecay = args.lrdecay


idstring = 'lr={:.0e}_reg={:.0e}_{:s}BN_LSTMlayers={:d}_weight={:d}_length={:d}'.format(lr,reg, 
                                                                    'no'*(not BN),
                                                                    LSTMlayers, 
                                                                    weightint,
                                                                    maxlength)
today = datetime.datetime.today()
outputtopdir = 'outputs_HM/{:s}_{:02d}_{:02d}'.format(dataset, today.month, today.day)
outputdir = outputtopdir+'/'+idstring+'/'
savename = 'saved_HM/'+dataset+'/'+idstring

pathdict = {'strand16s' : ('data/16s-finaltrain.txt', 'data/16s-finalvalid.txt')}
if dataset == 'strand16s':
	SPE = 55
	d = 20
	batchsize = 10
	testpath = 'data/testdata/testdata.txt'
iterations = SPE*500
trainpath, validpath = pathdict[dataset]
zspath = 'data/testdata/testset.txt'
writepath_train = outputdir+'trainlosses_'+idstring+'.txt'
writepath_valid = outputdir+'validlosses_'+idstring+'.txt'
writepath_test = outputdir+'testlosses_'+idstring+'.txt'

inputs = tf.keras.layers.Input(shape=(None, 5),name='input_feature')
y = tf.placeholder("float", [batchsize, None,None,1],name='lable')


trainsize = findsize(trainpath)
validsize = findsize(validpath)
testsize = findsize(testpath)
monitor_indices = np.random.choice(trainsize, trainsize//d, replace=False)

for path in ['outputs_HM', outputtopdir, outputdir, 'saved_HM', 'saved_HM/'+dataset]:
    if not os.path.exists(path):
        os.makedirs(path)

zsnames, zsmfe, testsets, testsetnames, mfeaccuracy = getdatanames(dataset)
print(idstring+'   ', testsets)


#if loadmodel:
#    loadname = 'saved_HM/strand16s/lr=1e-03_reg=1e-05_BN_LSTMlayers=1_weight=5_length=500_iter-20000.hdf5'
#    model = keras.models.load_model(loadname, custom_objects = {
#            'tf': tf,
#            'weighted_binary_cross_entropy' : weighted_binary_cross_entropy})
#    
#    # get training, validation accuracy
#    #trainmetrics = testonset(model, trainpath, writepath_train, monitor_indices, 'training set')
#    #validmetrics = testonset(model, validpath, writepath_valid, range(validsize), 'validation set')
#    
#    totalstep = 20000
#    
#    # test sets
#    #testfile = open(writepath_test, 'a+')
#    #testfile.write('\n-----\ntest losses, iter {0:d}\n\n'.format(totalstep))
#    #testfile.close()
#    
#    ## david set
#    #davidsetmetrics = []
#    #for k, (testset, testnames, mfeacc) in enumerate(zip(testsets, testsetnames, mfeaccuracy)):
#        #davidsetmetrics += testonset(model, testpath, writepath_test, range(k*5, (k+1)*5), testset, testnames, mfeaccs = mfeacc)
#    #writeavgmetrics(writepath_test, 'david 16s test total', davidsetmetrics)
#    
#    # zs set
#    for thr in [0.999]:
#        z = testonset(model, zspath, writepath_test, indices = [0], testsetname = 'sz', testnames = ['cuniculi_' + str(thr)], mfeaccs = [zsmfe[0]], threshold = thr)
#        print('{:.2f} {:.3f} {:.3f}'.format(thr, z[0][0], z[0][1]))
#    # write total test set metrics
#    #writeavgmetrics(writepath_test, 'total', davidsetmetrics + zsmetrics)

#    quit()


def modify_gradients1(grads):
	s = grads.get_shape().as_list()
	I = get_Identity_like(grads.shape)
	channelsizeh = int(s[2]/2)
	out1 = grads[:,:,0:channelsizeh,:]
	out2 = grads[:,:,channelsizeh:int(s[2]),:]
	out1 = out1 + out2
	out= tf.keras.backend.concatenate([out1, out1], axis=-2)
				
	return out

def modify_gradients2(grads):
	s = grads.get_shape().as_list()
	I = get_Identity_like(grads.shape)
	out = grads[:,:,:,:]
	outtranspose = tf.keras.backend.permute_dimensions(out,(1,0,2,3))
	outtransposediag = tf.multiply(I,outtranspose)
	mod_grad = out + outtranspose -outtransposediag
				
	return mod_grad

def weighted_binary_cross_entropy(labels, logits,weight):
    weight_f= tf.keras.backend.constant(weight, dtype=tf.float32)
    class_weights = labels*weight_f + (1 - labels)
    unweighted_losses = tf.keras.backend.binary_crossentropy(target=labels, output=logits)
    
    weighted_losses = unweighted_losses * class_weights
    
    loss = tf.keras.backend.mean(tf.matrix_band_part(tf.keras.backend.squeeze(weighted_losses, -1), 0, -1))
    return loss



model, out = makemodel(LSTMlayers, BN, weightint,reg, lr,inputs)
if lrdecay:
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, SPE*25, 0.96, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
else:
    opt = tf.train.AdamOptimizer(learning_rate=lr)

loss1 = weighted_binary_cross_entropy(y, out, weightint)


convkervar0 = [v for v in tf.trainable_variables() if 'conv2d_0/kernel:0' in v.name][0]
convkervar1 = [v for v in tf.trainable_variables() if 'conv2d_1/kernel:0' in v.name][0]
convkervar2 = [v for v in tf.trainable_variables() if 'conv2d_2/kernel:0' in v.name][0]
convkervar3 = [v for v in tf.trainable_variables() if 'conv2d_3/kernel:0' in v.name][0]
convkervar4 = [v for v in tf.trainable_variables() if 'conv2d_4/kernel:0' in v.name][0]
convkervar5 = [v for v in tf.trainable_variables() if 'conv2d_5/kernel:0' in v.name][0]
convkervar6 = [v for v in tf.trainable_variables() if 'conv2d_6/kernel:0' in v.name][0]
convkervar7 = [v for v in tf.trainable_variables() if 'conv2d_7/kernel:0' in v.name][0]
convkervar8 = [v for v in tf.trainable_variables() if 'conv2d_8/kernel:0' in v.name][0]

othervarlist = [v for v in tf.trainable_variables() if v not in [convkervar0, convkervar1, convkervar2, convkervar3,convkervar4, convkervar5, convkervar6, convkervar7, convkervar8]]

grad_Without_kernel = tf.gradients(loss1,othervarlist)

#CNN kernel variable list with kernel size 1
ker1_list = [convkervar0,convkervar7]
grad_ker_list1 = tf.gradients(loss1,ker1_list)
#CNN kernel variable with size > 1
ker2_list = [convkervar1, convkervar2,convkervar3, convkervar4,convkervar5, convkervar6,convkervar8]
grad_ker_list2 = tf.gradients(loss1,ker2_list)
#Modify the kernel varaibles
mod_grad1 =[modify_gradients1(gv) for gv in grad_ker_list1]
mod_grad2 =[modify_gradients2(gv) for gv in grad_ker_list2]

trainwithoutkernal = opt.apply_gradients(zip(grad_Without_kernel, othervarlist))

# Applying recurrent gradients
new_kernel_grad1 = [tf.placeholder(tf.float32, grad_ker_list1[index].shape) for index in range(len(ker1_list))]
new_kernel_grad2 = [tf.placeholder(tf.float32, grad_ker_list2[index].shape) for index in range(len(ker2_list))]

applygradkernel = opt.apply_gradients(zip(new_kernel_grad1, ker1_list)) 
applygradkernel2 = opt.apply_gradients(zip(new_kernel_grad2, ker2_list))
sample_x, sample_y = makebatch(trainpath, batchsize, maxlength)

# training loop
with tf.Session() as sess:
    
	# Initializing the variables
    t = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    for i in range(iterations//SPE):
    	print(i)
    	totalsize = findsize(trainpath) 
    	print(totalsize)
    	totalsize = (totalsize//batchsize)*batchsize
    	print(totalsize)
    	indexlist = np.arange(totalsize)
	
    	for j in range(0, totalsize, batchsize):
        	indices = indexlist[j:j+batchsize]
        	x1,l = makebatch(trainpath, batchsize, maxlength,indices,totalsize)
        	L,GWK,TWK,KL1,KL2,GKL1,MGKL1,GKL2,MGKL2,O= sess.run([loss1,grad_Without_kernel,trainwithoutkernal, ker1_list, ker2_list, grad_ker_list1, mod_grad1, grad_ker_list2, mod_grad2,out ], feed_dict = {inputs: x1, y : l})
        	print("Loss *************: ",L)
        	# Applying gradients
        	ker1_grads_dict = {new_kernel_grad1[index]: MGKL1[index] for index in range(len(MGKL1))} 
        	sess.run(applygradkernel, feed_dict = ker1_grads_dict)
        	ker2_grads_dict = {new_kernel_grad2[index]: MGKL2[index] for index in range(len(MGKL2))}
        	sess.run(applygradkernel2, feed_dict = ker2_grads_dict)
        	  
        	#print("Output final :",O," \n")
    	totalstep = (i+1)*SPE
    	if i % 25 == 24:
		    # save model
		    model.save(savename+'_iter-{:05d}'.format(totalstep)+'.hdf5')
		    save_path = saver.save(sess, savename)
		    
		    #decay lr
		    if lrdecay:
		        newlr = 0.5*tf.keras.backend.get_value(model.optimizer.lr)
		        tf.keras.backend.set_value(model.optimizer.lr, newlr)
		        print('new lr: {0:f}'.format(newlr))
		    
		    # test everything
		    if i > 75:
		        # get training, validation accuracy
		        print("print Training accuracies*****")
		        trainmetrics = testonset(model, trainpath, writepath_train, monitor_indices, 'training set')
		        validmetrics = testonset(model, validpath, writepath_valid, range(validsize), 'validation set')
		        testmetrics = testonset(model, testpath, writepath_test, range(testsize), 'test set')
		        
		        
		        # test sets
		        testfile = open(writepath_test, 'a+')
		        testfile.write('\n-----\ntest losses, iter {0:d}\n\n'.format(totalstep))
		        testfile.close()
		        
		        # david set
		        davidsetmetrics = []
		        for k, (testset, testnames, mfeacc) in enumerate(zip(testsets, testsetnames, mfeaccuracy)):
		            davidsetmetrics += testonset(model, testpath, writepath_test, range(k*5, (k+1)*5), testset, testnames, mfeaccs = mfeacc)
		        writeavgmetrics(writepath_test, 'david 16s test total', davidsetmetrics)
		        
		        # zs set
		        zsmetrics = testonset(model, zspath, writepath_test, range(16), 'zs', mfeaccs = zsmfe)
		        
		        # write total test set metrics
		        writeavgmetrics(writepath_test, 'total', davidsetmetrics + zsmetrics)
		        
