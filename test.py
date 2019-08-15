import numpy as np
a = np.ones([128])
b = a*2
c = b.reshape([-1,1])
print(c)