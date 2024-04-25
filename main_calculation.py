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



# ciphertext + num

block = heaan.Block(context, encrypted = False, data = [0] * num_slot) # block은 ciphertext가 아니라 message
ctxt = block.encrypt()

ctxt = ctxt + 1

ctxt.decrypt()
for i in range(5):
    print(ctxt[i].real) # what is real?
    
    
    
# inverse

a= [0.1, 0.01, 0.001, 0.0001]

a_block = heaan.Block(context, data = a) 
ctxt = a_block.encrypt()

inverse = ctxt.inverse(greater_than_one = False)
inverse.decrypt()
for i in range(5):
    print(inverse[i].real)
    


# add / sub / mult    

a, b = [1, 2, 3, 4] 

a_block = heaan.Block(context, data = a) # block은 ciphertext가 아니라 message
ctxt1 = a_block.encrypt()
b_block = heaan.Block(context, data = b) 
ctxt2 = b_block.encrypt()

# add
add_ctxt = ctxt1 + ctxt2

# sub
sub_ctxt = ctxt1 - ctxt2

# mult
# excute mult operation, then cihertext level 1 down -> why?
# init level of ciphertext = 12
# print(ctxt1.level) ## 12
mult_ctxt = ctxt1 * ctxt2
# print(mult_ctxt.level) ## 11



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



# negate
# change the sign (ex. -2 -> +2)

li = [2, 0, -2] + [0] * (num_slot - 3)
li = heaan.Block(context, data = li)
li.encrypt()

res1 = 0 - li ## 0 - ciphertext is also make same result of negate method
res2 = li.__neg__()

res1.decrypt()
for i in range(5): 
    print(res1[i])

print(' ')

res2.decrypt()
for i in range(5): 
    print(res2[i])



# sign
# 양수면 1, 음수면 -1 

# log_range : Log of the input range 
## Integer <= 38 
## Defaults to 0 -> domain of approximation is [-1, 1]
li = [0.2, 0, -0.2]
li = heaan.Block(context, data = li)
res = li.encrypt()

res.sign(inplace = True, log_range = 0)

res.decrypt()
for i in range(5): 
    print(res[i])



# greater than zero
# 음수면 0, 양수면 1, 0이면 0.5

li = [-0.1, 0.2, 0.5]
li = heaan.Block(context, data = li)
li = li.encrypt()

res = li.greater_than_zero()

res.decrypt()
for i in range(5): 
    print(res[i])
# reulst : 
# (3.849505991748714e-10+6.773722927550588e-11j)
# (0.9999999992998507-1.1215461761812492e-10j)
# (1.000000002811444+8.352405424189487e-10j)
# (0.4948779082648287-2.026023887153482e-09j)
# (0.49723285326616995+1.5938540162428722e-08j)

context.min_level_for_bootstrap
# result : 3



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