import numpy as np
import matplotlib.pyplot as plt

A_Cal, B_Cal, C_Cal = (20.34959396,2615.588964,-115.5216744)
A, B, C = (19.84644143,2370.741612,-127.05)
Temp = [250.0, 290.0, 320.0, 349.25, 378.5, 407.75, 437.0, 480.0, 500.0, 550.0]

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('2,4-Dimethyl-3-pentanol')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('lnVapor Pressure')
plt.legend()

#%%
A_Cal, B_Cal, C_Cal = (18.24623613, 2020.211899, -188.3467016)
A, B, C = (20.90917656, 3679.556307, -96.528)
Temp = [400.0, 420.0, 439.0, 451.5, 464.0, 476.5, 489.0, 500.0, 520.0]

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('1-Dodecyne')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('lnVapor Pressure')
plt.legend()

#%%
A_Cal, B_Cal, C_Cal = (18.42977708, 2018.52108, -189.9943014)
A, B, C = (25.14284766, 6022.181052, -28.25)
Temp = [371.0, 401.75, 432.5, 463.25, 494.0]

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('1,2-Ethanediol')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('lnVapor Pressure')
plt.legend()

#%%
A_Cal, B_Cal, C_Cal = (20.15067703, 2950.793647, -93.90061773)
A, B, C = (20.76791296, 3366.885975, -73.15)
Temp = [280.0, 305.0, 329.0, 363.25, 397.5, 431.75, 466.0, 500.0]

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('2,3-Dimethyloctane')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('lnVapor Pressure')
plt.legend()