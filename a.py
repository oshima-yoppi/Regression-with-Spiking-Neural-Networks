import matplotlib.pyplot as plt
a = [5,4,3,2,1]
b = []
for i in range(len(a)):
    b.append(i + 1)
fi = plt.figure()
plt.plot(b,a)

plt.show()