import numpy as np
aa = np.array(range(30)).reshape((2,3,5))
bb = np.array(range(30)).reshape((2,3,5))

print(aa)
print(bb)

# print(aa.argmax(0))
print(np.concatenate((aa,bb),axis=2))