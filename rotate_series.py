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


# rotate
# don't need to rot_idx as block. it can be just int value

a = [0.1, 0.2, 0.3, 0.4]
a = heaan.Block(context, data = a)
a.encrypt()
print(a)

rot_idx = 1
res1 = a.__lshift__(rot_idx) # left rotate 'a' ciphertext 'rot_idx' slots 

res1.decrypt()
for i in range(5):
    print(res1[i])
# result : 
# Ciphertext(log(num_slot): 15, device: CPU, level: 12)
# (0.19999999320903158+3.2849852222637134e-09j)
# (0.3000000074784588-8.453653715750744e-09j)
# (0.40000004558468816-2.440390060491175e-09j)
# (1.836140808337516e-08-2.7250364062917074e-09j)
# (-7.84106134746879e-09+3.401902441924129e-08j)

print(' ')

res2 = a.__rshift__(rot_idx) # right rotate 'a' ciphertext 'rot_idx' slots 

res2.decrypt()
for i in range(5): 
    print(res2[i])
# result : 
# (3.250580490950235e-08-2.0899486241772277e-08j) 
# (0.10000000904632533-6.527107664591949e-10j)
# (0.19999999265960336+5.034915853588839e-09j)
# (0.3000000071572689-7.872942290715758e-09j)
# (0.4000000450399934-2.1893719406148973e-09j)



# rotate_reduce

def left_rotate_reduce(context, data, gs, interval):
    # data = Block
   
    # m0 = heaan.Message(logN-1, 0)
    m0 = heaan.Block(context, encrypted = False, data = [0] * context.num_slots)
    res = m0.encrypt()
    
    empty_msg= heaan.Block(context, encrypted = False)
    rot = empty_msg.encrypt(inplace = False) # what is inplace?
    
    binary_list = []
    while gs > 1: # gs : context(n) * data(d)
        if gs % 2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs // 2
        
    binary_list.append(gs)

    i = len(binary_list) - 1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data
            while ind < i:
                rot = tmp.__lshift__(s)
                tmp += rot
                s *= 2
                ind += 1
            if sdind > 0:
                tmp = tmp.__lshift__(sdind)
            res += tmp
            sdind += s
        i -= 1            

    del  rot, tmp
    
    return res

n = 2
d = 3
data = [1]*32768
msg = heaan.Block(context, data = data,encrypted=False)
tmp = msg.encrypt()

result = left_rotate_reduce(context,tmp,n * d,1)

result.decrypt()
for i in range(10):
    print(result[i])