import numpy as np

def euler_cromer(acc_fn, dims = 3, timesteps = 1000, dt = 0.001):
    r = np.zeros([timesteps, dims])
    v = np.zeros([timesteps, dims])
    t = np.zeros(timesteps)
    for i in range(timesteps):
        v[i+1] = v[i] + acc_fn(r[i], v[i])*dt
        r[i+1] = r[i] + v[i+1]*dt
        t[i+1] = t[i] + dt

    return r, v, t
