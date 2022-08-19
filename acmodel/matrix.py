# Matrix library for things like HMM in NumPy or PyTorch or similar


import numpy as np


import torch
#device = torch.device("cuda:0")
device = torch.device("cpu")
#dtype = torch.float
dtype = torch.double

#x = torch.rand(2, 3, device=device, dtype=dtype)
#print(x)

use_torch = True
use_sparse = True






class m:
    if use_sparse:
        def __init__(self, val=None):
            if val is not None:
                self.val = torch.tensor(val, device=device).float().to_sparse()
            else:
                self.val = None
    else:
        def __init__(self, val=None):
            if val is not None:
                self.val = torch.tensor(val, device=device).float()
            else:
                self.val = None

    #def __str__(self): # print() will use __repr__ as a backup
    def __repr__(self): # what Jupiter will show as cell output (and should be parsable)
        if self.val is None:
            return "m()"
        return "m(" + str(self.val) + ")"
                
    def __add__(self, val):
        result = m()
        if type(val)==float or type(val)==int:
            #result.val = torch.full(self.size(), val, device=device) + self.val
            result.val = (torch.full(self.size(), val, device=device) + self.val).to_sparse()
            return result

        if type(self.val)==list: # need to convert from rowlist first
            self.val = torch.tensor(self.val, device=device).float().to_sparse()
        if type(val.val)==list: # dtto
            val.val = torch.tensor(val.val, device=device).float().to_sparse()


        if type(val)==m:
            result.val = self.val+val.val
            return result


        raise Exception(f"Unsupported type {type(val)}")
        
        
    def to_dense(self):
        result = m()
        result.val = self.val.to_dense()
        return result

    def __mul__(self, val):

        result = m()

        if type(val)==float or type(val)==int:
            result.val = self.val * val
            return result


        if type(self.val)==list: # need to convert from rowlist first
            self.rowlist2sparse()
        if type(val.val)==list: # dtto
            val.rowlist2sparse()



        #result.val = self.val*(val.val.view(1,4)) # HACK!!!

        if type(val)==m:
            #result.val = self.val.mul(val.val.to_sparse())
            result.val = self.val.mul(val.val)
            return result

        raise Exception(f"Unsupported type {type(val)}")




    def __matmul__(self, val):
        result = m()
        #result.val = self.val@val.val
        result.val = torch.sparse.mm(self.val, val.val)
        return result
    
    
    

    def T(self):
        result = m()
        result.val = self.val.transpose(1,0)
        return result

    def size(self):
        return self.val.size()
    
    def max(self): # WARNING: ALWAYS COMPUTES A SCALAR, GLOBAL MAX
        return self.val.values().max()
    
    @staticmethod
    def zeros(siz):
        result = m()
        #result.val = torch.zeros(siz, layout=torch.sparse_coo, device=device)
        result.val = torch.zeros(siz, layout=torch.sparse_coo, device=device)
        return result
       
    @staticmethod
    def rowlist(rows_cols): # cols ignored for now
        rows, cols = rows_cols
        result = m()
        result.val = [None]*rows # rely on duck-typing
        return result
 
    
    def __setitem__(self, index, val):
        if type(val)==float or type(val)==int:
            self.val[index] = val
            return
        self.val[index] = val.val
        return
    
    def __getitem__(self, index):
        result = m()
        result.val = self.val[index][None] # return size (1, n)
        return result

    def rowlist2sparse(self):
        self.val = torch.cat(self.val)



print(f"VH Matrix Library, {use_torch=}, {use_sparse=}")


x = m([[1, 2, 3, 4], [5, 6, 0, 8]])
a = m([[3,0,4,0]])
b = m([[2,0,3,0]])

print(f"{a.to_dense()=}")
print(f"{b.to_dense()=}")
print(f"{x.to_dense()=}")

print(f"{(a*b).to_dense()=}")
print(f"{(a+b).to_dense()=}")

print(f"{x[0].to_dense()=}")

print(f"{(x[0]*b).to_dense()=}")

#print(f"{(x[0][None]*b).to_dense()=}")

#r = m.rowlist((3, 4))

#print(f"{r=}")



