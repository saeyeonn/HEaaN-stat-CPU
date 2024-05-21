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


print("a : ", a)
print('plain text result')
print(a)

a_list = [a] * num_slot
a_ctxt = heaan.Block(context, encrypted = False, data = a_list)
a_ctxt.encrypt()


m_block = heaan.Block(context, encrypted = False, data = [1] * num_slot)



print('first ciphertext : ',a_ctxt.level)

print()
print('===========================================================')
print()

level_list = []
for i in range(12, 0, -1):
    
    print('***************** ciphertext a level : ', a_ctxt.level, '***********************')
    time_list = []
    for j in range(33):
        res = [0] * num_slot
        res_ctxt = heaan.Block(context, encrypted= False, data = res)
        res_ctxt.encrypt()  
        for i in range(15):
            num = 2 ** i         
            start = time.time()
            res_ctxt = a_ctxt.__lshift__(num)
            end = time.time()
            rot_time = end - start
            time_list.append(rot_time)
            print("index", 2 ** i)        
            print('====== Noist check part =====')
            # print('result ciphertext : ', res_ctxt)
            res_ctxt = res_ctxt.decrypt()
            print("plain : ", a)
            print("res_ctxt : ", res_ctxt[0])
            diff = abs(round(a - res_ctxt[0].real, 12))
            print('test result', diff)
        
        
    print("time list : ", time_list[2:])
    avg_time = round(sum(time_list[2:]) / len(time_list[2:]) * 1000, 5)
    print("avg time : ", avg_time)
    level_list.append(avg_time) ## 단위 ms로 변경

    a_ctxt = a_ctxt * m_block

   
    print('after ciphertext level',i)
    print('first ciphertext : ',a_ctxt.level)

    
    print()
    print(' ──────────────────────────────────────────────────────────')
    print()

print('레벨 단위 : ms')
for i in level_list:
    print(i) ## 단위 ms
    