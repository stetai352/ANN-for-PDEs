# Bachelor's Thesis Outline

## Contents

## Introduction
- Motivation
    - Non-intrusive solver
    - Versatile adaptability
    - Scalability to high-dimensional systems

## Elliptic PDEs
- Definition
    - partial differential equations
    - Used to simulate behaviour of closed systems 
- TODO Examples

### Parameterized PDEs
- Motivation
    - Parameters enable the description of classes of systems depending on a set of variable parameters
- Functionality
    - Enables efficient exploration of parameter space
    - Allows for quick adaptation to new scenarios without solving from scratch

### Linear vs non-linear systems
- Definitions
    - Linear systems
        - typical behaviour
    - Non-linear systems
        - formal definition
        - ex. turbulence, material deformation
        - expensive to simulate

## Model Order Reduction
- Definition
    - computational technique for solving PDEs
    - simplify large-scale systems by projecting onto lower-dimensional space
- Motivation
    - project solution onto smaller subspace that captures the dominant influential parameter ranges
    - preserve essential dynamics
    - improve computability 
    - Used in fluid dynamics, structural mechanics, thermal simulations
- list well-known methods (expain their functionalities roughly?)
    - Galerkin projection -> TODO
    - Often used in conjuction with POD -> Segue

### POD
- Motivation
    - Identifies dominant modes in a dataset
    - Reduces complexity while retaining essential features
    - Common in fluid dynamics and control applications
    - Reduces the dimensionality of high-dimensional PDE solutions while retaining accuracy
    - Uses singular value decomposition (SVD) to extract dominant basis functions
    - Constructs a reduced-order model (ROM) using a few significant modes
- Functionality
    - Compute covariance matrix
    $$C = \frac{1}{N}U'^TU'$$
    - Singular Value Decomposition of U'. Columns of $\Phi$ are POD basis modes
    $$U' = \Phi\Sigma V^T$$
    - Select dominant modes $\Phi_r = [\phi_1, \phi_2, ..., \phi_r]$ such that 
    $$\Sigma_{i=1}^r \sigma_i^2 \approx 99\% \text{ of total variance}$$
    - Project full model onto the reduced basis:
      - Represent solution as linear combination of basis functions
      $$u\approx \Sigma_{i = 1}^r a_i(t)\phi_i$$ 
      - Solve for coefficients $a_i(t)$

### Offline-online-decomposition
- Motivation
    - precompute as much as possible in the offline phase. Computational effort here is practically irrelevant
    - actual final computation is then as fast as possible
    - enables real-time simulation
- Functionality
    - Offline phase: Compute reduced basis, store precomputed solutions
    - Online phase: Solve reduced problem efficiently using stored data

### Intrusive vs. non-intrusive methods
- Definitions
    - intrusive methods require modifications to the governing equations
    - non-intrusive methods can work with empiric data, thus more adaptable to complex systems
- list well-known methods
    - statistical techniques
    - Machine Learning -> Segue

## Artificial Neural Networks
- Motivation
    - Learn from training data
    - Prediction of behaviour without proper solution of equations
- Functionality
    - Interconnected nodes with adjustable weights take input (parameters) and give predicted output (solution)
    - Training procedurally updates adjustable weights to optimize for prediction accuracy
    - Introduce Adam, LBFGS
    - Adaptive learning rates
    - Hidden layers. Network depth and width impact approximation accuracy and computational cost
    - Activation function
    - high-dimensional input (simulation data) informs a low-dimensional representation, then efficiently computes approximation and reconstructs original data.

### Training a feedforward artificial neural network
- Training
    - realized using `pytorch` Python library
    - train on empiric data (here simulated by previously computed FOM data, established as accurate technique)
    - gather these from snapshots $(\mu, u)$
    - 
    - compare predictions with actual data, minimizing Loss Function using MSE (Mean Squared Error)
    $$\mathcal{L} = \frac{1}{N}\Sigma_{i=1}^N ||y_i - \hat{y}_i||^2$$
    - Update weights using gradient descent (with learning rate $\eta$)
    $$W^{(l)} = W^{(l)} - \eta \frac{\partial\mathcal{L}}{\partial W^{(l)}}$$
    - Iterate through multiple epochs until $\mathcal{L}$ stops decreasing (early stopping when it increases again)
- Validation
    - Confirm training's accuracy on test points different from training data
- Manually tune hyperparameters (one of the subjects of this work)
- Drawbacks and counter-measures
    - Overfitting -> early stopping when error on validation set increases 
    - Vanishing/exploding gradient problem -> use activation functions like ReLU, Tanh, Sigmoid
    - Computational expense -> Use of specialized hardware (GPU, TPU)

### A non-intrusive reduced basis method using artificial neural networks
- Description
    - Transfer FANN to solving PDEs
- Functionality
    - Input are sets of parameters
    - Output are approximations of solutions
    - Generate training data
    - Generate validation data
    - Train ANN using training data with a chosen optimizer

## Numerical Experiments

### Experiment 1. Parameterized Sinusoidal Domain
- Definition and Description
    - Expression definition
    - Poisson problem
    - Dirichlet boundary condition
- Execute with both ANN and RB method
- Try various combinations in hidden layers' layout

### Experiment 2
- Consider non-linear equations for comparison?
- Execute with both ANN and RB method

### Experiment 3
- Consider real-life experiment, context depends on the previous two experiments, which one shows best improvement
- Exxecute with both ANN and RB method

### Comparison
- Compare the benefits and drawbacks of ANN for solving PDEs using the three (or more) examples addressed in this paper.
- Sensitivity to hyperparameter tuning (lr, net layout)
- Also compare with RB method

## Conclusion (Expected, subject to change)
- ANN is expensive in offline phase, extremely efficient but moderately accurate in the online phase
- ANN is versatile due to non-intrusive property and scalability to high-dimensional systems
- Indispensable for very complex situations which cannot be approached using conservative methods
- Requires large amount of high-quality training data 
- Optimization using GPU access possible

## Future
- Further research in optimal network layout
- Explore hybrid models combining ANN and classical numerical solvers

## Appendix