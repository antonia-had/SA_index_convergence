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

# Generate samples
nsamples = np.arange(50, 4050, 50)
S1_estimates = np.zeros([6,len(nsamples)])
ST_estimates = np.zeros([6,len(nsamples)])


for i in range(len(nsamples)):
    print('n= '+ str(nsamples[i]))
    sampleset = saltelli.sample(problem, nsamples[i],calc_second_order=False) # This is Set 1
    # Run model for all samples
    output = [fish_game(*sampleset[j,:]) for j in range(len(sampleset))]
    # Perform analysis
    results = sobol.analyze(problem, np.asarray(output), calc_second_order=False,print_to_console=False)
    ST_estimates[:,i]=results['ST']
    S1_estimates[:,i]=results['S1']

np.save('ST_estimates.npy', ST_estimates)
np.save('S1_estimates.npy', S1_estimates)


for j in range(6):
    plt.plot(nsamples, ST_estimates[j,:])
    
for j in range(6):
    plt.plot(nsamples, S1_estimates[j,:])