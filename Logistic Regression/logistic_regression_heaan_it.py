import os
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions

from pathlib import Path
import numpy as np
import pandas as pd
import time
import heaan_sdk as heaan
import math

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

####################################


def normalize_data(arr):
    S = 0
    for i in range(len(arr)):
        S += arr[i]
    return [arr[i] / S for i in range(len(arr))]


#####################################


def step(learning_rate, ctxt_X, ctxt_Y, ctxt_beta, n):

    # Step 1 
    ## beta 0
    index = 8 * n
    ctxt_beta0 = ctxt_beta.__lshift__(index)
    
    # compute ctxt_tmp = beta1 * x1 + beta2 * x2 + ... + beta8 * x8 + beta0
    ctxt_tmp = ctxt_beta * ctxt_X 
    for i in range(3):
        index = n * (2 ** (2 - i))
        ctxt_rot = ctxt_tmp.__lshift__(index)
        ctxt_tmp = ctxt_tmp + ctxt_rot
    ctxt_tmp = ctxt_tmp + ctxt_beta0
    
    # masking
    mask = heaan.Block(context, encrypted = False, data = [1 for __ in range(num_slot)])
    ctxt_tmp = ctxt_tmp * mask

    
    
    # Step 2
    ## compute sigmoid
    ctxt_tmp = ctxt_tmp.sigmoid(8.0)
    ctxt_tmp.bootstrap()
    
    ## if sigmoid(0) -> return 0.5
    mask2 = heaan.Block(context, encrypted = False, data = [2 for __ in range(num_slot)])
    mask2 = mask2.inverse()
    ctxt_tmp = ctxt_tmp - mask2
    
    
    
    # step 3 
    ## compute (learning_rate / n) * (y_(j) - p_(j))
    ctxt_d = ctxt_Y - ctxt_tmp
    ctxt_d = ctxt_d * (learning_rate / n)

    index = 8 * n
    ctxt_tmp = ctxt_d.__rshift__(index) # for beta0

    for i in range(3):
        index = n * (2 ** i)
        ctxt_rot = ctxt_d.__rshift__(index)
        ctxt_d = ctxt_d + ctxt_tmp
    ctxt_d = ctxt_d + ctxt_tmp
    
    
    
    # step 4 
    ## compute (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
    ctxt_d = ctxt_d * ctxt_X
    msg_X0 = [0] * num_slot
    for i in range(8 * n, 9 * n):
        msg_X0[i] = 1
    msg_X0 = heaan.Block(context, encrypted = False, data = msg_X0)
    msg_X0.encrypt()
    ctxt_X_j = ctxt_X + msg_X0
    ctxt_d = ctxt_d + ctxt_X_j

    
    
    # step 5
    ## Sum_(all j) (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
    for i in range(9):
        index = 2 ** (8 - i)
        ctxt_rot = ctxt_d.__lshift__(index)
        ctxt_d = ctxt_d + ctxt_rot
    ctxt_d = ctxt_d * mask
    
    for i in range(9):
        index = 2 ** i
        ctxt_rot = ctxt_d.__rshift__(index)
        ctxt_d = ctxt_d + ctxt_rot
        
        
        
    # step 6
    ## update beta    
    ctxt_d = ctxt_d + ctxt_beta
    return ctxt_d



def compute_sigmoid(ctxt_X, ctxt_beta, n):
    
    # beta0
    index = 8 * n
    ctxt_beta0 = ctxt_beta.__lshift__(index)
    ctxt_tmp = ctxt_X * ctxt_beta
    
    # compute x * beta + beta0
    for i in range(3):
        index = n * (2 ** (2 - i))
        ctxt_rot = ctxt_tmp.__lshift__(index)
        ctxt_tmp = ctxt_tmp + ctxt_rot
    ctxt_tmp = ctxt_tmp + ctxt_beta0
    
    # masking
    msg_mask1 = [1] * num_slot
    msg_mask1 = heaan.Block(context, encrypted = False, data = msg_mask1)
    msg_mask1.encrypt()
    ctxt_tmp = ctxt_tmp * msg_mask1
    
    # compute sigmoid    
    ctxt_tmp.sigmoid(8.0) 
    ctxt_tmp.bootstrap()
    msg_mask2 = [2] * num_slot   
    msg_mask2 = heaan.Block(context, encrypted = False, data = msg_mask2)
    msg_mask2.encrypt()
    msg_mask2 = msg_mask2.inverse()
    ctxt_tmp = ctxt_tmp - msg_mask2
    
    return ctxt_tmp
    
    



############################################



csv_train = pd.read_csv('train.csv')
df = pd.DataFrame(csv_train) 

train_n = df.shape[0]
X = [0] * 8
X[0] = normalize_data(df['LVR'].values) 
X[1] = list(df['REF'].values)
X[2] = list(df['INSUR'].values)
X[3] = normalize_data(df['RATE'].values)
X[4] = normalize_data(df['AMOUNT'].values)
X[5] = normalize_data(df['CREDIT'].values)
X[6] = normalize_data(df['TERM'].values)
X[7] = list(df['ARM'].values)
Y = list(df['DELINQUENT'].values)

msg_X = [0] * num_slot 
msg_Y = [0] * num_slot  

for i in range(8):
    for j in range(train_n):
        msg_X[train_n * i + j] = X[i][j]
     
ctxt_X = heaan.Block(context, encrypted = False, data = msg_X)
ctxt_X = ctxt_X.encrypt()

for i in range(train_n):
    msg_Y[i] = Y[i]

ctxt_Y = heaan.Block(context, encrypted = False, data = msg_Y)
ctxt_Y = ctxt_Y.encrypt()

beta = 2 * np.random.rand(9) - 1 
print("beta : ", beta)
print() 


msg_beta = [0] * num_slot

ctxt_next = heaan.Block(context, encrypted = False, data = msg_beta)
ctxt_next = ctxt_next.encrypt()


for i in range(8):
    for j in range(train_n):
        msg_beta[train_n * i + j] = beta[i + 1]

for j in range(train_n):
    msg_beta[8 * train_n + j] = beta[0]
    
print("msg_beta : ", msg_beta)

ctxt_beta = heaan.Block(context, encrypted = False, data = msg_beta)
ctxt_beta = ctxt_beta.encrypt()

learning_rate = 0.28
num_steps = 100

for i in range(num_steps):
    print("=== Step", i, "===") 
    ctxt_next = step(0.2, ctxt_X, ctxt_Y, ctxt_next, train_n)



######### Evaluation ##########



test_n = df.shape[0]

X_test = [0] * 8
X_test[0] = normalize_data(df['LVR'].values)
X_test[1] = list(df['REF'].values)
X_test[2] = list(df['INSUR'].values)
X_test[3] = normalize_data(df['RATE'].values)
X_test[4] = normalize_data(df['AMOUNT'].values)
X_test[5] = normalize_data(df['CREDIT'].values)
X_test[6] = normalize_data(df['TERM'].values)
X_test[7] = list(df['ARM'].values)
Y_test = df['DELINQUENT'].values

msg_X_test = [0] * num_slot


for i in range(8):
    for j in range(test_n):
        msg_X_test[test_n * i + j] = X_test[i][j]

ctxt_X_test = heaan.Block(context, encrypted = False, data = msg_X_test)
ctxt_X_test.encrypt()



######### Accuracy ##########


ctxt_infer = compute_sigmoid(ctxt_X_test, ctxt_next, test_n)

res = ctxt_infer.decrypt()
cnt = 0
for i in range(test_n): 
    if res[i].real >= 0.6:
        if Y_test[i] == 1:
            cnt += 1
    else:
        if Y_test[i] == 0:
            cnt += 1

print("Accuracy : ", cnt / test_n) 
