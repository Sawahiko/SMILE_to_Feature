import numpy as np
import matplotlib.pyplot as plt

A_Cal, B_Cal, C_Cal = (20.34959396,2615.588964,-115.5216744)
A, B, C = (19.84644143,2370.741612,-127.05)
#Temp = range(195, 574, 15)
Temp = [320.0, 349.25, 378.5, 407.75, 437.0] #Trange(320,437)

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
#plt.plot([320,320], [-15.0,15], 'k--', label='Train Range')
#plt.plot([437,437], [-15.0,15], 'k--')
plt.scatter(Temp, lnPsat_Cal, marker='o', color = 'r', label='This Work')
plt.title('2,4-Dimethyl-3-pentanol')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('ln(Vapor Pressure)')
plt.legend()

#%%
A_Cal, B_Cal, C_Cal = (18.6700589, 2006.157383, -96.45823)
A, B, C = (21.23741006, 3083.562089, -52.861)
#Temp = range(195, 574, 15)
Temp = [283.0, 290.5, 298.0, 305.5, 313.0] #Trange(283,313)

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
#plt.plot([283,283], [-2.4,15], 'k--', label='Train Range')
#plt.plot([313,313], [-2.4,15], 'k--')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('2-Butanamine, N-ethyl-')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('ln(Vapor Pressure)')
plt.legend()

#%%
# Not acceptable chemical
A_Cal, B_Cal, C_Cal = (27.77937567, 10187.28727, 80.93778788)
A, B, C = (17.00661719,3873.153056,0)
#Temp = range(195, 574, 15)
Temp = [298.0, 324.5, 351.0, 377.5, 404.0] #Trange(298,404)

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
#plt.plot([320,320], [-15.0,15], 'k--', label='Train Range')
#plt.plot([437,437], [-15.0,15], 'k--')
plt.scatter(Temp, lnPsat_Cal, marker='o', color = 'r', label='This Work')
plt.title('N,N-Diethyl-m-toluamide')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('ln(Vapor Pressure)')
plt.legend()

#%%
A_Cal, B_Cal, C_Cal = (22.3388533, 5156.898929, -32.24239159)
A, B, C = (22.99808175, 5920.63705, -39.839)
#Temp = range(195, 574, 15)
Temp = [366.0, 413.25, 460.5, 507.75, 555.0] #Trange(366,555)

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
#plt.plot([283,283], [-2.4,15], 'k--', label='Train Range')
#plt.plot([313,313], [-2.4,15], 'k--')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('2-Methylglutaric anhydride')
#plt.title('CC(C)C(C(C)C)O') SMILES
plt.xlabel('Temperature (K)')
plt.ylabel('ln(Vapor Pressure)')
plt.legend()

#%%
'''
#%%
A_Cal, B_Cal, C_Cal = (18.46856313, 2031.371653, -262.3911931)
A, B, C = (16.76555955,1345.073503,-294.004)
Temp = [450.0, 480.0, 495.0, 502.0, 509.0, 516.0, 523.0, 538.0, 563.0] #Trange(495,523)

lnPsat_Cal = []
for i in range(len(Temp)):
    temp = A_Cal-(B_Cal/(Temp[i]+C_Cal))
    lnPsat_Cal.append(temp)

lnPsat = []
for i in range(len(Temp)):
    temp = A-(B/(Temp[i]+C))
    lnPsat.append(temp)

plt.plot(Temp, lnPsat, 'k-', label='Antoine')
plt.plot([495,495], [0,15], 'k--', label='Train Range')
plt.plot([523,523], [0,15], 'k--')
plt.scatter(Temp, lnPsat_Cal, marker='o',color = 'r', label='This Work')
plt.title('Dimethyl phthalate')
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
plt.ylabel('ln(Vapor Pressure)')
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
'''