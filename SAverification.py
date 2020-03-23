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
    
set = saltelli.sample(problem, nsamples[i]) # This is Set 1

# Run model for all samples
output = [fish_game(*X_Set1[j,:]) for j in range(nsamples[30])]
t = time.process_time()
output = [fish_game(*X_Set1[j,:]) for j in range(nsamples[30])]
elapsed_time = time.process_time() - t
# Perform analysis
results = delta.analyze(problem, X_Set1, np.asarray(output), print_to_console=True)

# Sort factors by importance
factors_sorted = np.argsort(results['delta'])[::-1]

# Set up DataFrame of default values to use for experiment
X_defaults = np.tile(defaultvalues,(nsamples, 1))

# Create initial Sets 2 and 3
X_Set2 = np.copy(X_defaults)
X_Set3 = np.copy(X_Set1)

