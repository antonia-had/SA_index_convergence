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

# Array with n's to use
nsamples = np.arange(50, 4050, 50)

# Arrays to store the index estimates
S1_estimates = np.zeros([6,len(nsamples)])
ST_estimates = np.zeros([6,len(nsamples)])

# Loop through all n values, create sample, evaluate model and estimate S1 & ST
for i in range(len(nsamples)):
    print('n= '+ str(nsamples[i]))
    # Generate samples
    sampleset = saltelli.sample(problem, nsamples[i],calc_second_order=False)
    # Run model for all samples
    output = [fish_game(*sampleset[j,:]) for j in range(len(sampleset))]
    # Perform analysis
    results = sobol.analyze(problem, np.asarray(output), calc_second_order=False,print_to_console=False)
    # Store estimates
    ST_estimates[:,i]=results['ST']
    S1_estimates[:,i]=results['S1']

np.save('ST_estimates.npy', ST_estimates)
np.save('S1_estimates.npy', S1_estimates)

S1_estimates = np.load('S1_estimates.npy')
ST_estimates = np.load('ST_estimates.npy')

# Generate figure showing evolution of indices
fig =  plt.figure(figsize=(18,9))
ax1 = fig.add_subplot(1,2,1)
handles = []
for j in range(problem['num_vars']):
    handles += ax1.plot(nsamples, S1_estimates[j,:], linewidth=5)
ax1.set_title('Evolution of S1 index estimates', fontsize=20)
ax1.set_ylabel('S1', fontsize=18)
ax1.set_xlabel('Number of samples (n)', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2 = fig.add_subplot(1,2,2)
for j in range(problem['num_vars']):
    ax2.plot(nsamples, ST_estimates[j,:], linewidth=5)
ax2.set_title('Evolution of ST index estimates', fontsize=20)
ax2.set_ylabel('ST', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xlabel('Number of samples (n)', fontsize=18)
fig.legend(handles, problem['names'], loc = 'right', fontsize=11)
plt.savefig('indexevolution.png')

# Calculate parameter rankings
S1_ranks = np.zeros_like(S1_estimates)
ST_ranks = np.zeros_like(ST_estimates)
for i in range(len(nsamples)):
    orderS1 = np.argsort(S1_estimates[:,i])
    orderST = np.argsort(ST_estimates[:,i])
    S1_ranks[:,i] = orderS1.argsort()
    ST_ranks[:,i] = orderST.argsort()
    
# Generate figure showing evolution of ranks
fig =  plt.figure(figsize=(18,9))
ax1 = fig.add_subplot(1,2,1)
handles = []
for j in range(problem['num_vars']):
    handles += ax1.plot(nsamples, S1_ranks[j,:], linewidth=3)
ax1.set_title('Parameter ranking based on S1', fontsize=20)
ax1.set_ylabel('S1', fontsize=18)
ax1.set_xlabel('Number of samples (n)', fontsize=18)
ax1.set_yticklabels(np.arange(problem['num_vars']+1, 0, -1))
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2 = fig.add_subplot(1,2,2)
for j in range(problem['num_vars']):
    ax2.plot(nsamples, ST_ranks[j,:], linewidth=3)
ax2.set_title('Parameter ranking based on ST', fontsize=20)
ax2.set_ylabel('ST', fontsize=18)
ax2.set_yticklabels(np.arange(problem['num_vars']+1, 0, -1))
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xlabel('Number of samples (n)', fontsize=18)
fig.legend(handles, problem['names'], loc = 'right', fontsize=14)
plt.savefig('rankingevolution.png')

