import os
import numpy as np
root='d:/CSE/DeepFake/data/features'
shapes={}
for sub in ['real','fake']:
    p=os.path.join(root,sub)
    if not os.path.exists(p):
        continue
    for f in os.listdir(p):
        if not f.endswith('.npy'):
            continue
        a=np.load(os.path.join(p,f))
        shapes[a.shape]=shapes.get(a.shape,0)+1
print('Unique shapes and counts:')
for s,c in sorted(shapes.items(), key=lambda x:-x[1]):
    print(s,c)
