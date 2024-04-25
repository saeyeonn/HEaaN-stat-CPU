import os
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions

from pathlib import Path
import numpy as np
import pandas as pd
import time
import heaan_sdk as heaan
import math

# set key_dir_path
key_dir_path = Path('./BinaryDT/keys')

# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
# if set generate_keys=True, then make key
# if set generate_keys=False, then not make key. just use existing key.
context = heaan.Context(
    params,
    key_dir_path=key_dir_path,
    load_keys="all",
    generate_keys=False,
)

num_slot = context.num_slots # num of slots
log_num_slot = context.log_slots 


# Encrypt + Decrypt

a = [1,2,3,4] # no need to fill 0 to make list length to num_slot
b = [0.1,0.2,0.3,0.4]

# if set encrypted = False, a will not be encrypted
# just encoding list to message Block
a = heaan.Block(context,encrypted = False, data = a)
b = heaan.Block(context,encrypted = False, data = b)

ctxt1 = a.encrypt()
ctxt2 = b.encrypt()

# save ciphertext to ctxt file format
ctxt1.save('/root/tutorial/python/x1.ctxt')
ctxt2.save('/root/tutorial/python/x2.ctxt')

# result : Ciphertext(log(num_slot): 15, device: CPU, level: 12)


# load ctxt file
# make emtpy ctxt
empty_msg= heaan.Block(context,encrypted = False) # default data == None
load_ctxt1 = empty_msg.encrypt()   
load_ctxt2 = empty_msg  # after call 'empty_msg.encrypt()', then 'empty_msg' is encrypted as ciphertext

load_ctxt1 = load_ctxt1.load('/root/tutorial/python/x1.ctxt')
load_ctxt2 = load_ctxt2.load('/root/tutorial/python/x2.ctxt')

# Ciphertext print ctxt1 
print_ctxt(load_ctxt1, 5)
print(' ')
print_ctxt(load_ctxt2, 5)
print(' ')

## Ciphertext print ctxt2 
x1 = load_ctxt1.decrypt() # then load_ctxt1 is decrypted as message
x2 = load_ctxt2.decrypt()

for i in range(5):
    print('x1: ',x1[i])
    print('x2: ',x2[i])
    
    
def check_boot(x):
    if x.level == 3: # what is level?
        x.bootstrap() # why?
    elif x.level < 3:
        exit(0)
    return x

def print_ctxt(c,size):
    m = c.decrypt(inplace=False)
    for i in range(size):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)