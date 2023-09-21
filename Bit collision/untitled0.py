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
counts=dict(list)
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
