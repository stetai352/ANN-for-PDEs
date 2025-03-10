print(f'Initializing libraries...')

from pymor.basic import *
from pymor.reductors.neural_network import NeuralNetworkReductor
import time
import numpy as np
import torch.optim as optim
# For ROM
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.greedy import rb_greedy
# For pretty output
from prettytable import PrettyTable
# For functionality
from problem_definition import defineSinusoidProblem
from model_classes import *

### define wave problem 5 #

print(f'Defining elliptic PDE probem...')
fom, data = defineSinusoidProblem()

### Set parameters #

parameter_space = fom.parameters.space((0.1, 1))
basis_size = 10

training_set = parameter_space.sample_uniformly(3) #initially 100
validation_set = parameter_space.sample_randomly(20)

### Build models #
"""
U_rb = CoerciveRBModel(
    fom = fom, 
    product = fom.h1_0_semi_product,
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters),
    training_set = training_set, #1000
    max_extensions = basis_size
)

rb_rom, rb_reductor = U_rb.build()
"""
"""U_ann = ArtificialNeuralNetworkModel(
    fom = fom , 
    training_set = training_set, 
    validation_set = validation_set, 
    basis_size = basis_size, 
    # l2_err,            #l2_err=1e-5 
    ann_mse = 1e-5,           #ann_mse=1e-5
    hidden_layers = [30, 30, 30],
    restarts = 10,
    optimizer = optim.Adam,
    epochs = 1e5,
    learning_rate = 1e-5
    )

ann_rom, ann_reductor =  U_ann.build()"""

rb_reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)',
                                                    fom.parameters)
)

#parameter_space = fom.parameters.space(0.1, 1)
tac = time.perf_counter()
greedy_data = rb_greedy(fom, rb_reductor, parameter_space.sample_randomly(81), #initially 1000
                        rtol=1e-2)
rb_rom = greedy_data['rom']
rb_training_time = time.perf_counter() - tac

test_set = parameter_space.sample_randomly(10)

def fom_test(fom, test_set = test_set):
                U_fom = fom.solution_space.empty(reserve=len(test_set))
                #slvtime_fom = []
                for mu in test_set:
                    tic = time.perf_counter()
                    U_fom.append(fom.solve(mu))
                    time_fom = time.perf_counter() - tic
                return U_fom, time_fom

U_fom, time_fom = fom_test(fom, test_set = test_set)

def model_test(model_rom, model_reductor, test_set = test_set):
                U_model = fom.solution_space.empty(reserve=len(test_set)) 
                model_speedups = []
                #slvtime_ann = []
                for i in range(len(test_set)):
                    mu = test_set[i]
                    tic = time.perf_counter()
                    U_model.append(model_reductor.reconstruct(model_rom.solve(mu)))
                    time_model = time.perf_counter() - tic
                    model_speedups.append(time_fom/time_model)
                absolute_model_errors = (U_fom - U_model).norm()
                relative_model_errors = (U_fom - U_model).norm() / U_fom.norm()
                return U_model, model_speedups, absolute_model_errors, relative_model_errors

U_rb,  rb_speedups,  absolute_rb_errors,  relative_rb_errors  =  model_test(rb_rom, rb_reductor)

log_file = "log250309.txt"

for iLayers in [[42, 42], [56, 56], [100, 100], [30, 30, 30], [100, 100, 100]]:
    
    f = open(log_file, "a")
    f.write(f'Hidden layers have structure {iLayers}\n')
    f.close()

    for iOptimizer in [optim.LBFGS, optim.Adam]:
        for iLearning_rate in [1e-5, 1e-3, 0.1, 1]:

            f = open(log_file, "a")
            f.write(f'ANN has {iLayers}, {iOptimizer}, {iLearning_rate}\n') #adjust layers
            t = PrettyTable(['', 'avg absolute error', 'var absolute error', 'max absolute error', 'avg relative error', 'var relative error', 'max relative error', 'avg speedup', 'var speedup', 'min speedup', 'training time'])
            t.align = 'l'
            t.add_row(['RB', f'{np.average(absolute_rb_errors)}', f'{np.var(absolute_rb_errors)}', f'{np.max(absolute_rb_errors)}', 
                    f'{np.average(relative_rb_errors)}', f'{np.var(relative_rb_errors)}', f'{np.max(relative_rb_errors)}', 
                    f'{np.average(rb_speedups)}', f'{np.var(rb_speedups)}', f'{np.min(rb_speedups)}',
                    f'{rb_training_time}'])

            for iBasis_size in [5, 10, 30, 100, 1e3]:

                try:
                    toc = time.perf_counter()

                    ann_reductor = NeuralNetworkReductor(
                        fom, training_set, validation_set, basis_size = iBasis_size, ann_mse = None #initially l2_err=1e-5, ann_mse=1e-5
                    )

                    ann_rom = ann_reductor.reduce(hidden_layers = iLayers, restarts=10, optimizer = iOptimizer, epochs = 15000, learning_rate = iLearning_rate, log_loss_frequency = 100)

                    ann_training_time = time.perf_counter() - toc

                    ### Speedup testing

                    U_ann, ann_speedups, absolute_ann_errors, relative_ann_errors =  model_test(ann_rom, ann_reductor)                

                    # output

                    t.add_row([f'ANN basis {iBasis_size}', f'{np.average(absolute_ann_errors)}', f'{np.var(absolute_ann_errors)}', f'{np.max(absolute_ann_errors)}', 
                            f'{np.average(relative_ann_errors)}', f'{np.var(relative_ann_errors)}', f'{np.max(relative_ann_errors)}', 
                            f'{np.average(ann_speedups)}', f'{np.var(ann_speedups)}', f'{np.min(ann_speedups)}',
                            f'{ann_training_time}'])
                    #t.add_row(['training time', f'{ann_train}', f'{rb_train}'])
                    print(t)

                    #print(f'ts_solve: {np.average(ts_solve)}\nts_recon: {np.average(ts_recon)}\nts_rb:    {np.average(ts_rb)}\nts_fom:   {np.average(ts_fom)}')
                
                except KeyboardInterrupt:
                    print("Process terminated.")
                    exit()
                except:
                    f.write(f'An error has occurred\n')
            
            f.write(str(t))
            f.write("\n")
            f.close()