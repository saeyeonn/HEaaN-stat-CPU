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