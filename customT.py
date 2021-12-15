# -*- coding: utf-8 -*-

import numpy as np
import time
from matplotlib import colors
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import confusion_matrix
from makebatchesT import *

# CUSTOM KERAS LAYERS
def get_Identity_like(shape):
	A = np.zeros(shape).astype('float32')
	for i in range(shape[0]):
		A[i,i,:,:]=1
	return A
	

def SelfCartesian(x):
    newshape = tf.stack([1, 1, tf.keras.backend.shape(x)[1], 1])
    
    x_expanded = tf.expand_dims(x, axis = -2)
    x_tiled = tf.tile(x_expanded, newshape)
    x_transposed = tf.keras.backend.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = tf.keras.backend.concatenate([x_tiled, x_transposed], axis=-1)
    return x_concat

def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]



# COMBINE ALL TEST STEPS

def testonset(model, testpath, writepath, indices = None, testsetname = '', testnames = None, threshold = 0.5, mfeaccs = None):
    
    if indices is None:
        indices = range(findsize(testpath))
    
    if testnames is None:
        testnames = [str(d) for d in range(1, len(indices)+1)]
    
    writefile = open(writepath, 'a+')
    writefile.write('\n{:15s} test set\n\n'.format(testsetname))
    if mfeaccs is None:
        mfeaccs = [None]*len(indices)
    
    metrics = []
    for testname, ind, mfeacc in zip(testnames, indices, mfeaccs):
        metrics.append(test_on_sequence(writefile, testpath, str(testname), ind, model, threshold, mfeacc))
    
    avgppv, avgsen, avgacc = np.mean(metrics, axis = 0)
    writefile.write('\n{:15s} avg    ppv:  {:.4f}     sen:  {:.4f}     acc:  {:.4f}\n\n'.format(testname, avgppv, avgsen, avgacc))
    
    writefile.close()
        
    return metrics


def test_on_sequence(writefile, testpath, testname, ind, model, threshold, mfeacc = None):
    test_x, test_y, test_yhat, test_pred = get_xy(testpath, ind, model)
    
    tn, fp, fn, tp = get_confusion(test_y, test_pred)
    ppv, sen, acc = get_metrics(test_y, test_yhat, threshold)
    
    writeoutputs(writefile, testname, tn, fp, fn, tp, ppv, sen, acc, mfeacc)
    
    return ppv, sen, acc


# TP/TN/FP/FN

def get_confusion(y, pred):
    # return tn, fp, fn, tp
    return confusion_matrix(y[np.triu_indices(y.shape[1])].flatten(),
                                       pred[np.triu_indices(pred.shape[1])].flatten(),
                                       labels=[0,1]).ravel()


# WRITING

def writeoutputs(writefile, testname, tn, fp, fn, tp, ppv, sen, acc, mfeacc = None):
    writefile.write('{:20s} '.format(testname))
    writefile.write('tn: {:7d}  fp: {:7d}  fn: {:3d}  tp: {:3d}'.format(tn, fp, fn, tp))
    writefile.write(' ppv: {:0.3f}  sen: {:0.3f} ||'.format(tp/(tp+fp), tp/(tp+fn)))
    writefile.write(' ppv: {:0.3f}  sen: {:0.3f}  acc: {:0.3f}'.format(ppv, sen, acc))
    
    if not mfeacc is None:
        writefile.write('  mfe acc: {:0.3f}{:s}'.format(mfeacc, '  ***'*(acc < mfeacc)))
    
    writefile.write('\n')
    return

def writeavgmetrics(writepath, setname, metricslist):
    writefile = open(writepath, 'a+')
    avgppv, avgsen, avgacc = np.mean(metricslist, axis = 0)
    writefile.write('\n{:15s} avg    ppv:  {:.4f}     sen:  {:.4f}     acc:  {:.4f}\n\n'.format(setname, avgppv, avgsen, avgacc))
    writefile.close()
    return


# PPV, SEN, ACC FUNCTIONS

def get_metrics(y, yhat, threshold = 0.5):
    truepairs = makepairs(np.triu(y), threshold)
    predpairs = makepairs(np.triu(yhat), threshold)
    metrics = getmetrics_frompairs(set(truepairs), set(predpairs))
    
    return metrics


def makepairs(originalstructure, threshold = 0.5, nested = False):
    structure = np.copy(originalstructure)
    pairs = []
    while np.any(structure > threshold):
        newpair = np.unravel_index(np.argmax(structure), structure.shape)
        #pairs.add(newpair)
        pairs.append(newpair)
        
        if nested:
            structure[:newpair[0],newpair[0]:newpair[1]+1] = 0
            structure[newpair[0]:newpair[1]+1,newpair[1]:] = 0
        else:
            structure[newpair[0]] = 0
            structure[:,newpair[1]] = 0
    
    #print(threshold, len(pairs))
    return pairs


def getmetrics_frompairs(native, predicted):
    
    if not len(predicted) or not len(native):
        return 0.0, 0.0, 0.0
    tp = native.intersection(predicted)
    fn = native.difference(predicted)
    fp = predicted.difference(native)
    
    PPV = len(tp)/float(len(predicted))
    sen = len(tp)/float(len(native))
    accuracy = 0.5*(PPV + sen)
    
    return PPV, sen, accuracy


def get_xy(filename, ind, model):
    x, y = makebatch(filename, 1, None, batchindices = [ind])
    y = np.squeeze(y)
    #y = np.triu(y)
    yhat = np.squeeze(model.predict_on_batch(x))
    pred = np.rint(yhat)
    
    return x, y, yhat, pred

