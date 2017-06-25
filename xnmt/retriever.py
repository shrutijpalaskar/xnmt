from __future__ import division, generators

import dynet as dy
import numpy as np
from batcher import *
from search_strategy import *
from vocab import Vocab
from encoder import *

DEBUG = True

def stack(src, nwin, skip=128):
    '''
        for each skip step, stack the +/-nwin/2 feature frames together to form a list of 1d features for the RNN encoder.
        '''
    assert src.shape[1] % skip == 0
    [nchannel, nframe] = src.shape
    src_st = []
    nstack = int(src.shape[1]/skip)
    
    for i in range(nstack):
        curstack = src[:, i*skip:(i*skip+nwin)]
        src_st.append(curstack)
    
    return src_st

class Retriever:
    '''
    A template class implementing an speech to image retrieval network that can retrieve an
    image given a speech description and vice versa
    
    '''

    '''
    Calculate the loss of the input and output
    '''

    def loss(self, x, y):
        raise NotImplementedError('loss mush be implemented for Translator subclasses')

    def batch_loss(self, xs, zs):
        return self.loss(x, y)

class DefaultRetriever(Retriever):
    '''def __init__(self, sp_encoder, sp_attender, im_encoder, im_attender):'''
    def __init__(self, src_encoder, trg_encoder, src_data, trg_data):
        self.src_encoder = src_encoder
        self.trg_encoder = trg_encoder
        self.src_data = src_data
        self.trg_data = trg_data
        '''#self.im_attender = im_attender'''
    
    def calc_loss(self, src, trg):
        ''' For single pair only now'''
        '''nrow_trg = self.trg_data[trg].shape[0]
        ncol_trg = self.trg_data[trg].shape[1]'''
        
        src_encodings = self.src_encoder.transduce(self.src_data[src])
        trg_encodings = self.trg_encoder.transduce(self.trg_data[trg])
        if DEBUG:
            print(src_encodings)
            print(trg_encodings)

        similarity = []
        for src_encoding in src_encodings:
            similarity_row = []
            for trg_encoding in trg_encodings:
                cur_s = dy.dot(src_encoding, trg_encoding)

    def retrieve(self, src):


#############
# Test Code #
#############
dy.renew_cg()
model = dy.Model()

sp_name = 'captions_40k.npz'
im_name = 'images_40k.npz'
sp_data = np.load(sp_name)
im_data = np.load(im_name)
sp = sp_data['arr_0']
im = im_data['arr_0']
nmf = sp[0].shape[0]
nframe = sp[0].shape[1]
nim = im[0].shape[0]
nembed = 1024

# Experimenting parameters...
nlayer_sp = 3
nlayer_im = 2 # Two layer to decrease length 4096 by a factor of 4 to embedding length
nwin = 128
nin = nmf*nwin

# Choose one pair for now
sp_stack = stack(sp[0], nwin)
sp_data_e = [dy.inputTensor(cur_stack.flatten('F'), (nmf*nwin)) for cur_stack in sp_stack]
print(im[0].shape)
im_data_e = dy.inputTensor(im[0], (im[0].shape,))

#im_stack = stack(im[0], nwin)
if DEBUG:
    print(len(sp_stack))
    print(sp_stack[0].shape)


# Test the encoders
sp_encoder = PyramidalLSTMEncoder(nlayer_sp, nin, nembed, model)
im_encoder = PyramidalLSTMEncoder(nlayer_im, nim, nembed, model)

# Test methods
retriever = DefaultRetriever(sp_encoder, im_encoder, sp_data_e, im_data_e)
retriever.calc_loss(0, 0)

