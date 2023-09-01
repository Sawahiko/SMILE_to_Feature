import numpy as np
import pandas as pd
import pubchempy as pcp



#pcp.get_synonyms('Aspirin', 'smiles')
names = list()
names = ['Methane', 'Ethane', 'Propane', 'N-butane', 'N-pentane', 'N-hexane', 'N-heptane', 'N-octane', 'N-nonane', 'N-decane', 'N-undecane', 'N-dodecane', 'N-tridecane', 'N-tetradecane', 'N-pentadecane', 'N-hexadecane', 'N-heptadecane', 'N-octadecane', 'N-nonadecane', 'N-eicosane', 'N-heneicosane', 'N-docosane', 'N-tricosane', 'N-tetracosane', 'N-pentacosane', 'N-hexacosane', 'N-heptacosane', 'N-octacosane', 'N-nonacosane', 'N-triacontane', 'N-hentriacontane', 'N-dotriacontane', 'N-tritriacontane', 'N-pentatriacontane', 'N-tetracontane', 'N-pentacontane', 'N-hexacontane', 'N-heptacontane', 'N-octacontane', 'N-nonacontane', 'N-hectane', '2-methylpropane', '2-methylbutane', '2-methylpentane', '2-methylhexane', '2-methylheptane', '2-methyloctane', '2-methylnonane', '2-methyldecane', '2-methylundecane', '3-methylpentane', '3-methylhexane', '3-methylheptane', '3-methyloctane', '3-methylnonane', '3-methyldecane', '3-methylundecane', '2,2-dimethylpropane', '2,2-dimethylbutane', '2,2-dimethylpentane', '2,2-dimethylhexane', '2,2-dimethylheptane', '2,2-dimethyloctane', '2,2,4-Trimetylpentane', 'Ethene', 'Propene', 'But-1-ene', '1-pentene', '1-hexene', '1-heptene', '1-octene', '1-nonene', '1-decene', '1-undecene', '1-dodecene', '1-tridecene', '1-tetradecene', '1-pentadecene', '1-hexadecene', '1-heptadecene', '1-octadecene', '1-nonadecene', '1-docosene', '2-methyl-1-propene', '2-methyl-1-butene', '2-methyl-1-pentene', '2-methyl-1-hexene', '2-methyl-1-heptene', '2-methyl-1-octene', '2-methyl-1-nonene', '2-methyl-1-decene', 'Ethyne', 'Propyne', '1-butyne', '1-pentyne', '1-hexyne', '1-heptyne', '1-octyne', '1-nonyne', '1-decyne', '1-undecyne', '1-dodecyne', '1-tridecyne', '1-tetradecyne', '1-pentadecyne', '1-hexadecyne', '2-butyne', '2-pentyne', '2-hexyne', 'Cyclopropane', 'Cyclobutane', 'Cyclopentane', 'Cyclohexane', 'Cycloheptane', 'Cyclooctane', 'Cyclononane', 'Cyclodecane', 'Cyclopropene', 'Cyclobutene', 'Cyclopentene', 'Cyclohexene', 'Cycloheptene', 'Cyclooctene', 'Methylcyclopentane', 'Ethylcyclopentane', 'propylcyclopentane', 'butylcyclopentane', 'pentylcyclopentane', 'hexylcyclopentane', 'heptylcyclopentane', 'octylcyclopentane', 'nonylcyclopentane', 'decylcyclopentane', 'Methylcyclohexane', 'Ethylcyclohexane', 'Propylcyclohexane', 'butylcyclohexane', 'pentylcyclohexane', 'hexylcyclohexane', 'heptylcyclohexane', 'octylcyclohexane', 'nonylcyclohexane', 'decylcyclohexane', 'dodecylcyclohexane', 'tetradecylcyclohexane', 'pentadecylcyclohexane', 'octadecylcyclohexane', 'trans-Decahydronaphthalene', 'cis-Decahydronaphthalene', 'Benzene', 'Methylbenzene', 'Ethylbenzene', '1,2-Dimethylbenzene', '1,3-Dimethylbenzene', '1,4-Dimethylbenzene', 'Propylbenzene', 'Butylbenzene', 'Pentylbenzene', 'Hexylbenzene', 'Heptylbenzene', 'Octylbenzene', 'Nonylbenzene', 'Decylbenzene', 'Dodecylbenzene', 'Tetradecylbenzene', 'Pentadecylbenzene', 'Hexadecylbenzene', 'Heptadecylbenzene', 'Octadecylbenzene', 'Nonadecylbenzene', '1,2,3,4-tetrahydro-naphthalene', '1,2,3,4-tetrahydro-1-methyl-naphthalene', '1,2,3,4-tetrahydro-5-methyl-naphthalene', '1,2,3,4-tetrahydro-6-methyl-naphthalene', '1-Hexyl-1,2,3,4-tetrahydro-naphthalene', 'Naphthalene', '1-methylnaphthalene', '1-ethylnaphthalene', '1-propylnaphthalene', '1-butylnaphthalene', '1-pentylnaphthalene', '1-hexylnaphthalene', '1-nonylnaphthalene', '1-decylnaphthalene', '2-methylnaphthalene', '2-ethylnaphthalene', '1,2-dimethylnaphthalene', '1,3-dimethylnaphthalene', '1,4-dimethylnaphthalene', '1,5-dimethylnaphthalene', '1,6-dimethylnaphthalene', '1,7-dimethylnaphthalene', '1,8-dimethylnaphthalene', '2.3-dimethylnaphthalene', '2,6-dimethylnaphthalene', '2.7-dimethylnaphthalene', 'Phenanthrene', 'Anthracene', '1-methylanthracene', '2-methylanthracene', '9-methylanthracene', 'Benzophenanthrene', 'Chrysene', 'Triphenylene', 'Pyrene'
    ]
#Tb = 
names = ['Triisobutylmethylphosphonium tosylate']

names=names[:20]
List1=list()

for name1 in names :
    print(name1)
    results = pcp.get_compounds(name1, 'name')
    for compound in results:
        print(compound.cid,"",compound.isomeric_smiles)
        List1.append(compound.isomeric_smiles)
    
df =pd.DataFrame()
data = {
    "Name":names,
    "SMILES":List1
    }
df =pd.DataFrame(data)
a=results[0]
#a.
#pcp.get_properties('IsomericSMILES', 'CC', 'smiles')

#p = pcp.get_properties('IsomericSMILES', 'CC[N+](CC)=C(C)OCC', 'smiles')
