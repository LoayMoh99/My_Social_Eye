import numpy as np


x = 1.0+1.0j
x = np.array([x])
print(np.arctan2(x.imag, np.sqrt(x.real)))