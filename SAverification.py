import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from fishery import fish_game
import matplotlib.pyplot as plt

# Set up dictionary with system parameters
problem = {
  'num_vars': 6,
  'names': ['a', 'b', 'c','h',
            'K','m'],
  'bounds': [[ 0.002, 2],
             [0.005, 1],
             [0.2, 1],
             [0.001, 1],
             [100, 5000],
             [0.1, 1.5]]
}

defaultvalues = np.array([0.005, 0.5, 0.5, 0.1, 2000, 0.7])

# Generate samples
nsamples = np.arange(500, 5000, 50)
index_estimates = np.zeros([6,len(nsamples)])

for i in range(len(nsamples)):
    print('n= '+ str(nsamples[i]))
    sampleset = saltelli.sample(problem, nsamples[i],calc_second_order=False) # This is Set 1
    # Run model for all samples
    output = [fish_game(*sampleset[j,:]) for j in range(len(sampleset))]
    # Perform analysis
    results = sobol.analyze(problem, np.asarray(output), calc_second_order=False,print_to_console=False)
    index_estimates[:,i]=results['ST']


