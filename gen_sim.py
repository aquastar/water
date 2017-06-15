import numpy as np

# 50,100,400,800,1000 days
# Faucet: 42.856
# Dis: 1.0784
# Toilet: 12.9203
# Shower: 2.3668
# Cloth: 2.1761
daily_times = 100
d_size = 1000 * daily_times
faucet = np.random.poisson(lam=42.856, size=d_size)
dish = np.random.poisson(lam=1.0784, size=d_size)
toilet = np.random.poisson(lam=12.9203, size=d_size)
shower = np.random.poisson(lam=2.3668, size=d_size)
cloth = np.random.poisson(lam=2.1761, size=d_size)

sum_vol = np.sum((faucet, dish, toilet, shower, cloth), axis=0)

all_data = np.stack((faucet, dish, toilet, shower, cloth, sum_vol), axis=0).transpose()

np.save('sim.npy', all_data)