import numpy as np
aa = np.array(range(30)).reshape((2,3,5))
bb = np.array(range(30)).reshape((2,3,5))
#[[[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]

# [[15 16 17 18 19]
#  [20 21 22 23 24]
#  [25 26 27 28 29]]]

# print(aa.argmax(0))
print(np.concatenate((aa,bb),axis=2))