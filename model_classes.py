# unused file

from pymor.basic import *
import time
import numpy as np
# For ANN
import torch.optim as optim
from pymor.reductors.neural_network import NeuralNetworkReductor
# For ROM
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.greedy import rb_greedy

class ArtificialNeuralNetworkModel:

    def __init__(self, 
                 fom, 
                 training_set, 
                 validation_set, 
                 basis_size = None, 
                 l2_err = None,            #l2_err=1e-5 
                 ann_mse = None,           #ann_mse=1e-5
                 hidden_layers = None,
                 restarts = None,
                 optimizer = None,
                 epochs = None,
                 learning_rate = None):
        
        self.fom = fom
        self.training_set = training_set
        self.validation_set = validation_set
        self.basis_size = basis_size
        self.l2_err = l2_err
        self.ann_mse = ann_mse

        self.hidden_layers = hidden_layers
        self.restarts = restarts
        self.optimizer = optimizer
        self.epochs = epochs
        self.learning_rate = learning_rate

    def build(self):

        ann_reductor = NeuralNetworkReductor(
            self.fom, 
            self.training_set, 
            self.validation_set,
            self.basis_size, 
            #l2_err=1e-5, 
            self.ann_mse
        )

        ann_rom = ann_reductor.reduce(
            #hidden_layers = self.hidden_layers, 
            #restarts = self.restarts, 
            #optimizer = self.optimizer, 
            #epochs = self.epochs, 
            #learning_rate = self.learning_rate, 
            #log_loss_frequency = 100
        )
        
        return ann_rom, ann_reductor

class CoerciveRBModel:
    
    def __init__(self, 
                 fom,
                 product = None, #= fom.h1_0_semi_product
                 coercivity_estimator = None, #= ExpressionParameterFunctional('min(diffusion)', fom.parameters)
                 training_set = None, #1000
                 max_extensions = None
                 ):

        self.fom = fom
        self.product = product
        self.coercivity_estimator = coercivity_estimator
        self.training_set = training_set, #initially 1000
        self.max_extensions = max_extensions

    def build(self):

        rb_reductor = CoerciveRBReductor(
            self.fom,
            self.product,
            self.coercivity_estimator
        )
        
        greedy_data = rb_greedy(self.fom, 
                                rb_reductor, 
                                self.training_set,
                                self.max_extensions)

        rb_rom = greedy_data['rom']

        return rb_rom, rb_reductor
