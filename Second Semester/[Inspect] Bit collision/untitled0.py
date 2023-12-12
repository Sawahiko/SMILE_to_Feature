import csv
from rdkit.Chem import Draw
import rdkit
from rdkit import Chem,DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd
from collections import defaultdict


# https://rdkit.blogspot.com/2014/02/colliding-bits.html
# https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-number-of-fp-bits-set.html
# https://github.com/greglandrum/rdkit_blog/blob/eae1643d2ad4788f9f1942ea681e9c001931e418/notebooks/Bits%20Set%20By%20Fingerprint%20Type.ipynb
# https://rdkit.blogspot.com/2016/02/morgan-fingerprint-bit-statistics.html
# http://rdkit.blogspot.com/2016/02/colliding-bits-iii.html

with open('chembl_5HT.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    i=0
    counts=defaultdict(list)
    for row in reader:
        i+=1
        #print(row['first_name'], row['last_name'])
        #print(row['canonical_smiles'])
        
        m = Chem.MolFromSmiles(row['canonical_smiles'])
        if not m: continue
        for v in 1,2,3:
            counts[(v,-1)].append(len(rdmd.GetMorganFingerprint(m,v).GetNonzeroElements()))
            for l in 1024,2048,4096,8192:
                counts[(v,l)].append(rdmd.GetMorganFingerprintAsBitVect(m,v,l).GetNumOnBits())
        if not (i+1)%5000:
            print("Done {0}".format(i+1))
# %%
import matplotlib as plt
import numpy as np

#fig1=plt.figure(figsize=(16,20))
fig1=plt.figure.Figure(figsize=(16,20))
pidx=0
#----------------------------
fig,ax=fig1.subplots(2,2)
for nbits in (8192,4096,2048,1024):
    
    v1=np.array(counts[1,-1])
    v2=np.array(counts[1,nbits])
    d1 = v1-v2
    d1p=np.array(d1).astype(float)
    d1p/=v1
    v1=np.array(counts[2,-1])
    v2=np.array(counts[2,nbits])
    d2 = v1-v2
    d2p=np.array(d2).astype(float)
    d2p/=v1
    v1=np.array(counts[3,-1])
    v2=np.array(counts[3,nbits])
    d3 = v1-v2
    d3p=np.array(d3).astype(float)
    d3p/=v1
    
    #subfig1=plt.figure.Figure(figsize=(16,20))
    ax[pidx].hist((d1,d2,d3),bins=20,log=True,label=("r=1","r=2","r=3"))
    #subfig1.title('%d bits'%nbits)
    #subfig1.legend()
    
    #fig1.subplots(4,2,pidx)
    pidx+=1
    #subfig2=plt.pyplot.hist((d1p,d2p,d3p),bins=20,log=True,label=("r=1","r=2","r=3"))
    #subfig2.hist((d1p,d2p,d3p),bins=20,histtype='step',cumulative=-1,normed=True, color=['b','g','r'])
    #subfig2.legend()