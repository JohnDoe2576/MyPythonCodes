import numpy as np
import matplotlib.pyplot as plt

def aprbs(**parms):
    # Generate an Amplitude modulated Pseudo-Random Binary Sequence (APRBS)
    #
    # The Pseudo-Random Binary Sequence (PRBS) is extensively used as an
    # excitation signal for System Identification of linear systems. It is 
    # characterized by randomly delayed shifts in amplitude between a 
    # user-defined minimum and maximum. These delayed shifts are usually 
    # range-bound, and are very helpful in capturing the system behaviour 
    # close to the operating frequency of the system.
    # 
    # A nonlinear system usually will have different behaviour at different 
    # amplitudes and cannot be predicted with the princlipe of superposition. 
    # Hence, the excitation signal also need to be modified to accomodate 
    # for capturing system behaviour at different amplitudes. The APRBS is 
    # an extension of PRBS by introducing randomly delayed shifts to random 
    # levels of range-bound amplitudes (rather than between a maximum and 
    # minimum).
    # 
    # Input parameters:
    #   n_samples: Number of required samples
    #   alpha: tuple of (min_amplitude, max_amplitude)
    #   tau: tuple of (min_delay, max_delay)

    # Extract signal parameters
    n_samples = parms['n_samples']  # Number of samples
    tau = parms['tau']              # Delay vector
    alpha = parms['alpha']          # Amplitude vector
    
    # Convert to usable parameters
    tau_min = tau[0]
    tau_max = tau[1]
    tau_range = tau_max - tau_min
    
    alpha_min = alpha[0]
    alpha_max = alpha[1]
    alpha_range = alpha_max - alpha_min
    
    # Initialize arrays
    tau_array = np.zeros((n_samples),dtype=int)
    alpha_array = np.zeros((n_samples))
    signal = np.zeros((n_samples))
    
    # Initialize counters
    sample_count = 0
    shift_count = 0
    
    while sample_count < n_samples:
        # Generate a random shift to perturb 'tau' and 'alpha'
        tau_shift = np.random.uniform(0.0, 1.0, 1)
        alpha_shift = np.random.uniform(0.0, 1.0, 1)

        # Introduce the random delay such that it range bound between 'tau_min' and 'tau_max'
        tau_array[shift_count] = np.fix(tau_min + (tau_shift * tau_range) ).astype(int)
        alpha_array[shift_count] = alpha_min + (alpha_shift * alpha_range)

        # Update counters
        sample_count += tau_array[shift_count]
        #print("Shift, Sample, tau, alpha: {}, {}, {}, {}".format(shift_count, sample_count, tau_array[shift_count], alpha_array[shift_count]))
        shift_count += 1

    tau_array[shift_count-1] -= (sample_count - n_samples)
    #print(tau_array[shift_count-1])

    idx = 0
    for i in range(0,shift_count):
        idx_tmp = idx + np.arange(0,tau_array[i],1,dtype=int)
        signal[idx_tmp] = alpha_array[i]
        idx = idx + tau_array[i]

    return signal

# Time parameters
t0 = 0.                         # Start time
dt = 0.01                       # Time step
t1 = 100.                       # End time

# Time vector
t = np.arange(t0, t1, dt)

# Signal parameters
n_samples = len(t)
alpha = (-2.5, 2.5)
tau = tuple(np.array([dt, 1.])/dt)

u = aprbs(n_samples=n_samples, alpha=alpha, tau=tau)

plt.plot(t,u)
plt.show()