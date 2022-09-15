# Matrix library for things like HMM in NumPy or PyTorch or similar


import numpy as np


import torch
#device = torch.device("cuda:0")
#device = torch.device("cpu")
#dtype = torch.float
#dtype = torch.double


torch.set_default_dtype(torch.float64)


#x = torch.rand(2, 3, device=device, dtype=dtype)
#print(x)

use_torch = True
use_sparse = False


def sparsify(t):
    return t.to_sparse()
    #return t.to_sparse_csr()
    #return t.to_sparse_bsr(32)


class m:
    if use_sparse:
        def __init__(self, val=None):
            if val is not None:
                self.val = sparsify(torch.tensor(val, device=device).float())
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
        #print("In add")

        result = m()
        if type(val)==float or type(val)==int:
            #result.val = torch.full(self.size(), val, device=device) + self.val
            if use_sparse:
                result.val = sparsify(torch.full(self.size(), val, device=device) + self.val)
            else:
                result.val = (torch.full(self.size(), val, device=device) + self.val)
            return result

        if type(self.val)==list: # need to convert from rowlist first
            if use_sparse:
                self.val = sparsify(torch.tensor(self.val, device=device).float())
            else:
                self.val = torch.tensor(self.val, device=device).float()
        if type(val.val)==list: # dtto
            if use_sparse:
                val.val = sparsify(torch.tensor(val.val, device=device).float())
            else:
                val.val = torch.tensor(val.val, device=device).float()


        if type(val)==m:
            result.val = self.val+val.val
            return result


        raise Exception(f"Unsupported type {type(val)}")
        
        
    def to_dense(self):
        #print("In to_dense")

        result = m()
        result.val = self.val.to_dense()
        return result

    def __mul__(self, val):
        #print("In mul")

        result = m()

        if type(val)==float or type(val)==int:
            result.val = self.val * val
            #result.val = (self.val * val).to_dense()
            return result


        if type(self.val)==list: # need to convert from rowlist first
            self.rowlist2sparse()
        if type(val.val)==list: # dtto
            val.rowlist2sparse()



        #result.val = self.val*(val.val.view(1,4)) # HACK!!!

        if type(val)==m:


            if not self.val.is_sparse and val.val.is_sparse:
                #print("mul dense * sparse")
                result.val = self.val.mul(val.val.to_dense())
                return result



            #result.val = self.val.mul(val.val.to_sparse())
            result.val = self.val.mul(val.val)
            return result

        raise Exception(f"Unsupported type {type(val)}")




    def __matmul__(self, val):
        #print("In matmul")

        result = m()
        #result.val = self.val@val.val
        if use_sparse:
            result.val = torch.sparse.mm(self.val, val.val)
        else:
            result.val = torch.mm(self.val, val.val)
        return result
   
    

    def T(self):
        result = m()
        result.val = self.val.transpose(1,0)
        return result

    def size(self):
        return self.val.size()
    
    def max(self): # WARNING: ALWAYS COMPUTES A SCALAR, GLOBAL MAX
        if not self.val.is_sparse:
            return self.val.max()
        return self.val._values().max()
    def min(self): # WARNING: ALWAYS COMPUTES A SCALAR, GLOBAL MIN
        if not self.val.is_sparse:
            return self.val.min()
        return self.val._values().min()
    
    @staticmethod
    def zeros(siz):
        result = m()
        #result.val = torch.zeros(siz, layout=torch.sparse_coo, device=device)
        if use_sparse:
            result.val = torch.zeros(siz, layout=torch.sparse_coo, device=device)
        else:
            result.val = torch.zeros(siz, device=device)
        return result
       
    @staticmethod
    def rowlist(rows_cols): # cols ignored for now
        rows, cols = rows_cols
        result = m()
        result.val = [None]*rows # rely on duck-typing
        return result
 
    
    def __setitem__(self, index, val):
        #print("In setitem")
        if type(val)==float or type(val)==int:
            self.val[index] = val
            return

        if type(self.val)==list:
            self.val[index] = val.val
            return

        if type(self.val)==torch.Tensor and self.val.is_sparse:
            #print("Converting to dense in setitem")
            self.val = self.val.to_dense()
            #print(f"{self.val=}")


        if val.val.is_sparse:
            self.val[index] = val.val[0].to_dense()
            return

        self.val[index] = val.val[0] # remove dimension to match, [] = [[]][0]
        return
    
    def __getitem__(self, index):
        #print("In getitem")
        result = m()
        result.val = self.val[index][None] # return size (1, n)
        #result.val = self.val[index] # return size (1, n)
        return result

    def rowlist2sparse(self):
        #print("In rowlist2sparse")
        self.val = torch.cat(self.val)





"""
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
"""


