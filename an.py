import matplotlib.pyplot as plt
a = {}
a[-5] = 5
a[1] = 1
a[-12] = 12
print(a.items())
a = a.items()
a = sorted(a)
print(a)
x, y = zip(*a) 

plt.plot(x, y)
plt.show()
# a = sorted(a.keys())
# print(a)