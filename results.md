# Experimental results
## 250308
- LBFGS optimizer with learning rate 1 yields the best results for error with a similar magnitude to RB methods. The variance is similarly small too.
- A higher `base_size` tends to yield a slightly worse result for errors which is counter-intuitive
    - check code?
- An ANN of with `basis_size` 30 and LBFGS optimizer with learning rate 1e-3 seems to give marginally better results regarding speedup.
- The structure of the hidden layers seem not to be of big significance. Neither for error nor for speedup.
- An increase in the number of layers seems to have no effect on the speedup, but the number of nodes in each layer does.

## 250316
- speedup in range 12 - 15 without apparent trends across different settings.