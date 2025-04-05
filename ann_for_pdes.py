print(f'Initializing libraries...')

from pymor.basic import *
from pymor.reductors.neural_network import NeuralNetworkReductor
import time
import numpy as np
import torch
import torch.optim as optim
# For ROM
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.greedy import rb_greedy
# For pretty output
from prettytable import PrettyTable
# For functionality
from problem_definition import defineSinusoidProblem
# gpu access
#torch.set_default_device('rocm')

### define wave problem 5 #

print(f'Defining elliptic PDE probem...')
fom, data = defineSinusoidProblem()

### Set parameters #

parameter_space = fom.parameters.space((0.1, 1))
basis_size = 10

training_set = parameter_space.sample_uniformly(3) #initially 100
validation_set = parameter_space.sample_randomly(20)

### Build models #

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
                times_solve = []
                times_recon = []
                #slvtime_ann = []
                for i in range(len(test_set)):
                    mu = test_set[i]
                    tic = time.perf_counter()
                    model_solution = model_rom.solve(mu)
                    time_solve = time.perf_counter() - tic
                    U_model.append(model_reductor.reconstruct(model_solution))
                    time_model = time.perf_counter() - tic

                    time_recon = time_model - time_solve
                    model_speedups.append(time_fom/time_model)
                    times_solve.append(time_solve)
                    times_recon.append(time_recon)
                avg_time_solve = np.average(times_solve)
                avg_time_recon = np.average(times_recon)
                absolute_model_errors = (U_fom - U_model).norm()
                relative_model_errors = (U_fom - U_model).norm() / U_fom.norm()
                return U_model, model_speedups, avg_time_solve, avg_time_recon, absolute_model_errors, relative_model_errors

### Run Tests #

log_file = "log250405.txt"

iLayers_list = [[42, 42], [30, 30, 30]]
iBasis_size_list = [5, 10, 20, 30, 50] # values beyond 50 are too high to sensibly show any improvement because they are too close to test quantity 3^3 = 81
iActivation_function_list = [torch.relu, torch.sigmoid, torch.tanh]
#iOptimizer_list = [optim.LBFGS, optim.Adam]
#iLearning_rate_list = [1e-5, 1e-3, 0.1, 1]
iOptim_LR_list = [(optim.LBFGS, 1), (optim.Adam, 1e-5), (optim.Adam, 1e-3), (optim.Adam, 0.1), (optim.Adam, 1)]

for iLayers in iLayers_list:
    
    f = open(log_file, "a")
    f.write(f'Hidden layers have structure {iLayers}\n')
    f.write(f'Coercive RB\n')
    t = PrettyTable(['', 'AVG ABSOLUTE ERROR', 'var absolute error', 'max absolute error', 'AVG RELATIVE ERROR', 'var relative error', 'max relative error', 'AVG T_SOLVE', 'AVG T_RECON', 'AVG SPEEDUP', 'var speedup', 'min speedup', 'training time'])
    t.align = 'l'

    for iBasis_size in iBasis_size_list:

        try:
            # rb reductor
            tac = time.perf_counter()
            
            rb_reductor = CoerciveRBReductor(
                fom,
                product=fom.h1_0_semi_product,
                coercivity_estimator=ExpressionParameterFunctional('min(diffusion)',
                                                                fom.parameters))

            greedy_data = rb_greedy(fom, rb_reductor, parameter_space.sample_randomly(81), #initially 1000
                                    max_extensions = iBasis_size) #rtol=1e-2)
            
            rb_rom = greedy_data['rom']

            rb_training_time = time.perf_counter() - tac

            ### Speedup testing

            U_rb, rb_speedups, avg_time_rb_solve, avg_time_rb_recon, absolute_rb_errors, relative_rb_errors =  model_test(rb_rom, rb_reductor)   

            # output

            t.add_row([f'RB basis {iBasis_size}', f'{np.average(absolute_rb_errors)}', f'{np.var(absolute_rb_errors)}', f'{np.max(absolute_rb_errors)}', 
                f'{np.average(relative_rb_errors)}', f'{np.var(relative_rb_errors)}', f'{np.max(relative_rb_errors)}', 
                f'{avg_time_rb_solve}', f'{avg_time_rb_recon}',
                f'{np.average(rb_speedups)}', f'{np.var(rb_speedups)}', f'{np.min(rb_speedups)}',
                f'{rb_training_time}'])

            print(t)

        except KeyboardInterrupt:
            print("Process terminated.")
            exit()

        except:
            f.write(f'An error has occurred\n')

    f.write(str(t))
    f.write("\n")
    f.close()
    for iActivation_function in iActivation_function_list:
        for iOptimizer, iLearning_rate in iOptim_LR_list:

            f = open(log_file, "a")
            f.write(f'ANN has {iLayers}, {iActivation_function}, {iOptimizer}, {iLearning_rate}\n') #adjust layers
            print(f'ANN has {iLayers}, {iActivation_function}, {iOptimizer}, {iLearning_rate}\n')
            t = PrettyTable(['', 'AVG ABSOLUTE ERROR', 'var absolute error', 'max absolute error', 'AVG RELATIVE ERROR', 'var relative error', 'max relative error', 'AVG T_SOLVE', 'AVG T_RECON', 'AVG SPEEDUP', 'var speedup', 'min speedup', 'training time'])
            t.align = 'l'

            for iBasis_size in iBasis_size_list:

                try:
                    
                    # ann reductor
                    toc = time.perf_counter()

                    ann_reductor = NeuralNetworkReductor(
                        fom, training_set, validation_set, basis_size = iBasis_size, ann_mse = None #initially l2_err=1e-5, ann_mse=1e-5
                    )

                    ann_rom = ann_reductor.reduce(hidden_layers = iLayers, activation_function=iActivation_function, restarts=10, optimizer = iOptimizer, epochs = 10000, learning_rate = iLearning_rate, log_loss_frequency = 100)

                    ann_training_time = time.perf_counter() - toc

                    ### Speedup testing

                    U_ann, ann_speedups, avg_time_ann_solve, avg_time_ann_recon, absolute_ann_errors, relative_ann_errors =  model_test(ann_rom, ann_reductor)

                    # output

                    t.add_row([f'ANN basis {iBasis_size}', f'{np.average(absolute_ann_errors)}', f'{np.var(absolute_ann_errors)}', f'{np.max(absolute_ann_errors)}', 
                            f'{np.average(relative_ann_errors)}', f'{np.var(relative_ann_errors)}', f'{np.max(relative_ann_errors)}', 
                            f'{avg_time_ann_solve}', f'{avg_time_ann_recon}',
                            f'{np.average(ann_speedups)}', f'{np.var(ann_speedups)}', f'{np.min(ann_speedups)}',
                            f'{ann_training_time}'])                    

                    print(t)

                    losses = ann_reductor.losses
                    f.write(f'{str(losses)}\n')
                
                except KeyboardInterrupt:
                    print("Process terminated.")
                    exit()

                except:
                    f.write(f'An error has occurred\n')
            
            f.write(str(t))
            f.write("\n")
            f.close()