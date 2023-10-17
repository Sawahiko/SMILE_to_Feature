import dwsim



# %%
import clr

clr_1 = clr

#from system.IO import Directory, Path, File
#from system import String, Double, Array, Reflection, Exception

dtlpath = "C:\\Users\\Kan\\AppData\\Local\\DWSIM"

clr.AddReference(dtlpath + "DWSIM.Thermodynamics.StandaloneLibrary.dll")

from DWSIM.Thermodynamics import Streams, PropertyPackages, CalculatorInterface

import CapeOpen

# %%
dtlc = CalculatorInterface.Calculator()

print(String.Format("DTL version: {0}", Reflection.Assembly.GetAssembly(dtlc.GetType()).GetName().Version))
print()

dtlc.Initialize()

nrtl = PropertyPackages.NRTLPropertyPackage(True)

dtlc.TransferCompounds(nrtl)

T = 355.0 #K
P = 101325.0 #Pa

compprops = dtlc.GetCompoundConstPropList()

print("Ethanol constant properties:\n")
for prop in compprops:
    pval = dtlc.GetCompoundConstProp("Ethanol", prop)
    print(prop + "\t" + pval)

print()