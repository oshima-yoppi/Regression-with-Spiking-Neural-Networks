from math import fabs
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
DF_PATH = 'loss2.csv'
df = pd.read_csv(DF_PATH)
print(df.head())
x = df['label_x']
y = df['label_y']
z = df['label_z']
w = df['label_w']
df['theta'] = np.degrees(np.arccos(z /np.sqrt(x**2 +y ** 2 + z** 2)))
df['phi'] = np.degrees(np.arctan2(y, x))
df['rate_w'] = np.abs(100*df['loss_w']/w)
print(df.head())
df_0 = df[df['label_w'] <= 5]
df_1 = df[(5 <= df['label_w'] ) & (df['label_w'] < 10)]
df_2 = df[(10 <= df['label_w'] ) & (df['label_w'] < 15)]
df_3 = df[(15 <= df['label_w'] ) & (df['label_w'] <= 20)]
print(df['rate_w'].describe())


lst_omega_rate = []
axis_oemga_rate = []

th = 5
for i in range(20 // th):
    axis_oemga_rate.append(f'[{th*i},{th*(i+1)})')
    lst_omega_rate.append(df[(th*i <= df['label_w'] ) & (df['label_w'] < th*(i+1))]['rate_w'].values)

df = df[df['label_w'] >= 10]
x = df['phi']
y = df['theta']
z = df['rate_w']



lst_theta = []
axis_label = []
th = 20
for i in range(180//th):
    axis_label.append(f'[{th*i},{th*(i+1)})')
    lst_theta.append(df[(th*i <= df['theta'] ) & (df['theta'] < th*(i+1))]['rate_w'].values)




fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x,y,z, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('phi')
ax.set_ylabel('theta')
ax.set_zlabel('rate_w')
plt.show()

# df_theta.plot.box()
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.boxplot([df for df in lst_theta], sym='', labels=axis_label, showmeans=False)
ax1.set_xlabel('theta[deg]')
ax1.set_ylabel('Error Rate[%]')

ax2.boxplot([d for d in lst_omega_rate], sym='', labels=axis_oemga_rate, showmeans=False)
ax2.set_xlabel('Angular Velocity[deg/s]')
ax2.set_ylabel('Error Rate[%]')
plt.tight_layout()
plt.show()