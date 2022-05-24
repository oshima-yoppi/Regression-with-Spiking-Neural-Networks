import time
a = time.time()

import matplotlib.pyplot as plt
b = 1241231234
t = a-b
t = '{:.2f}'.format(t)
fi = plt.figure(f'{t}')
plt.show()