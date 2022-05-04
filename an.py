import numpy as np
print(np.random.rand(2,5))
x = np.where(np.random.rand(1, 10) < 0.5, 1, 0)#(1, 100)
print(x)