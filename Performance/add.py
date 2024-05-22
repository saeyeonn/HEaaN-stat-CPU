import os
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions

from pathlib import Path
import numpy as np
import pandas as pd
import time
import heaan_sdk as heaan
import math
import random

# set key_dir_path
key_dir_path = Path('./keys')

# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
# if set generate_keys=True, then make key
# if set generate_keys=False, then not make key. just use existing key.
context = heaan.Context(
    params,
    key_dir_path=key_dir_path,
    load_keys="all",
    generate_keys=True,
)

num_slot = context.num_slots
log_num_slot = context.log_slots


################################# 시작 #####################################

a = random.uniform(0, 1)
b = random.uniform(0, 1)


print("a : ", a, ", b : ", b)
print('plain text result')
plain_res = a + b
print(plain_res)

a = [a] * num_slot
a_ctxt = heaan.Block(context, encrypted = False, data = a)
a_ctxt.encrypt()

b = [b] * num_slot
b_ctxt = heaan.Block(context, encrypted = False, data = b)
b_ctxt.encrypt()


m_ctxt = heaan.Block(context, encrypted = False, data = [1] * num_slot)


print('first ciphertext : ',a_ctxt.level)
print('second ciphertext : ',b_ctxt.level)


print()
print('===========================================================')
print()

level_list = []
for i in range(12, 0, -1):
    a_ctxt = heaan.Block(context, encrypted = False, data = a)
    a_ctxt.encrypt()
    b_ctxt = heaan.Block(context, encrypted = False, data = b)
    b_ctxt.encrypt()

    print('***************** ciphertext a level : ', a_ctxt.level, 'ciphertext b level : ', b_ctxt.level, '***********************')
    time_list = []
    for j in range(33):
        res_ctxt = heaan.Block(context)
        start = time.time()
        res_ctxt = a_ctxt + b_ctxt
        end = time.time()
        add_time = end - start
        time_list.append(add_time)
        
        
        print('====== Noist check part =====')
        # print('result ciphertext : ', res_ctxt)
        res_ctxt.decrypt()
        print("plain : ", plain_res)
        print("res_ctxt : ", res_ctxt[0])
        diff = abs(round(a[0] + b[0] - res_ctxt[0].real, 12))
        print('test result', diff)
        
        
    print("time list : ", time_list[2:])
    avg_time = round(sum(time_list[2:]) / len(time_list[2:]) * 1000, 5)
    print("avg time : ", avg_time)
    level_list.append(avg_time) ## 단위 ms로 변경

    a_ctxt = a_ctxt * m_ctxt
    b_ctxt = b_ctxt * m_ctxt
    
    print('after ciphertext level',i)
    print('first ciphertext : ',a_ctxt.level)
    print('second ciphertext : ',b_ctxt.level)
    
    print()
    print(' ──────────────────────────────────────────────────────────')
    print()
    

print('레벨 단위 : ms')
for i in level_list:
    print(i) ## 단위 ms
    