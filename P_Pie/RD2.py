#checking for validity
import pandas as pd
import numpy as np

def check(ar):
    if isinstance(ar,pd.DataFrame):
        ar=ar.to_numpy().T
        
    CRe=ar[0]
    DoubleCCRe=ar[1]
    TripleCC=ar[2]
    Bracket=ar[3]
    Benzene=ar[4]
    CycleRe=ar[5]
    SingleCO=ar[6]
    DoubleCO=ar[7]

    if CRe < 1:
        return False
    elif (2*DoubleCCRe)+(2*TripleCC)+(Bracket)+(6*Benzene)+(3*CycleRe)+(SingleCO)+(2*DoubleCO) > CRe:
        return False
    elif DoubleCCRe+TripleCC+Benzene+SingleCO+DoubleCO > 2*CRe:
        return False
    elif DoubleCCRe+TripleCC > CRe:
        return False
    elif Bracket+CycleRe > CRe:
        return False
    elif SingleCO+2*DoubleCO > CRe:
        return False
    elif Benzene+CycleRe > CRe:
        return False
    else :
       return True