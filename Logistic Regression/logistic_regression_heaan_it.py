import os
import numpy as np
import pandas as pd
import time
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import heaan_sdk as heaan

# Set the number of CPU threads to use for parallel regions
os.environ["OMP_NUM_THREADS"] = "8"

# Set key_dir_path
key_dir_path = Path('./keys')

# Set parameter
params = heaan.HEParameter.from_preset("FGb")

# Initialize context and load all keys
context = heaan.Context(
    params,
    key_dir_path=key_dir_path,
    load_keys="all",
    generate_keys=True,
)

num_slot = context.num_slots
log_num_slot = context.log_slots

# Normalize the data
def normalize_data(arr):
    S = 0
    for i in range(len(arr)):
        S += arr[i]
    return [arr[i] / S for i in range(len(arr))]

# Step function for HE
def step(learning_rate, ctxt_X, ctxt_Y, ctxt_beta, n):
    # Step 1
    index = 8 * n
    ctxt_beta0 = ctxt_beta.__lshift__(index)
    
    # Compute ctxt_tmp = beta1 * x1 + beta2 * x2 + ... + beta8 * x8 + beta0
    ctxt_tmp = ctxt_beta * ctxt_X
    for i in range(3):
        index = n * (2 ** (2 - i))
        ctxt_rot = ctxt_tmp.__lshift__(index)
        ctxt_tmp = ctxt_tmp + ctxt_rot
    ctxt_tmp = ctxt_tmp + ctxt_beta0
    ctxt_tmp.bootstrap()
    
    mask = [1] * num_slot
    mask = heaan.Block(context, encrypted = False, data = mask)
    ctxt_tmp = ctxt_tmp * mask

    # Step 2
    # Compute sigmoid -> exp -> prob
    ctxt_tmp = ctxt_tmp.sigmoid(8.0)
    ctxt_tmp.bootstrap()
    
    # if sigmoid(0) -> return 0.5
    msg_mask2 = [2] * num_slot
    msg_mask2 = heaan.Block(context, encrypted = False, data = msg_mask2)
    msg_mask2 = msg_mask2.inverse()
    ctxt_tmp = ctxt_tmp - msg_mask2
    
    # Step 3
    # Compute (learning_rate / n) * (y_(j) - p_(j)) -> exp vs. real
    ctxt_d = ctxt_Y - ctxt_tmp
    ctxt_d = ctxt_d * (learning_rate / n)

    index = 8 * n
    ctxt_tmp = ctxt_d.__rshift__(index)  # for beta0
    ctxt_tmp.bootstrap()

    for i in range(3):
        index = n * (2 ** i)
        ctxt_rot = ctxt_d.__rshift__(index)
        ctxt_d = ctxt_d + ctxt_tmp
    ctxt_d = ctxt_d + ctxt_tmp
    
    # Step 4
    # Compute (learning_rate/n) * (y_(j) - p_(j)) * x_(j) to update beta
    ctxt_d = ctxt_d * ctxt_X
    msg_X0 = [0] * num_slot
    for i in range(8 * n, 9 * n):
        msg_X0[i] = 1
    msg_X0 = heaan.Block(context, encrypted = False, data = msg_X0)
    ctxt_X_j = ctxt_X + msg_X0
    ctxt_d = ctxt_d + ctxt_X_j
    ctxt_d.bootstrap()
    msg_X0.bootstrap()

    # Step 5
    # Sum_(all j) (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
    for i in range(9):
        index = 2 ** (8 - i)
        ctxt_rot = ctxt_d.__lshift__(index)
        ctxt_d = ctxt_d + ctxt_rot
    ctxt_d = ctxt_d * mask
    ctxt_d.bootstrap()
    mask.bootstrap()
    
    for i in range(9):
        index = 2 ** i
        ctxt_rot = ctxt_d.__rshift__(index)
        ctxt_d = ctxt_d + ctxt_rot
    ctxt_d.bootstrap()
        
    # Step 6
    # Update beta
    ctxt_d = ctxt_d + ctxt_beta
    return ctxt_d

# Compute sigmoid in HE
def compute_sigmoid(ctxt_X, ctxt_beta, n):
    # Beta0
    index = 8 * n
    ctxt_beta0 = ctxt_beta.__lshift__(index)
    ctxt_tmp = ctxt_X * ctxt_beta
    ctxt_tmp.bootstrap()
    ctxt_beta.bootstrap()
    ctxt_beta0.bootstrap()
    
    # Compute x * beta + beta0
    for i in range(3):
        index = n * (2 ** (2 - i))
        ctxt_rot = ctxt_tmp.__lshift__(index)
        ctxt_tmp = ctxt_tmp + ctxt_rot
    ctxt_tmp = ctxt_tmp + ctxt_beta0
    
    msg_mask1 = [1] * num_slot
    msg_mask1 = heaan.Block(context, encrypted = False, data = msg_mask1)
    ctxt_tmp = ctxt_tmp * msg_mask1
    ctxt_tmp.bootstrap()
    msg_mask1.bootstrap()
    
    # Compute sigmoid
    ctxt_tmp.sigmoid(domain_range = 8.0)
    ctxt_tmp.bootstrap()
    msg_mask2 = [2] * num_slot
    msg_mask2 = heaan.Block(context, encrypted = False, data = msg_mask2)
    msg_mask2 = msg_mask2.inverse()
    ctxt_tmp = ctxt_tmp - msg_mask2
    msg_mask2.bootstrap()
    
    return ctxt_tmp

# Load and preprocess data
csv_train = pd.read_csv('train.csv')
df = pd.DataFrame(csv_train)

# Timing the preprocessing
start_time_preprocess_he = time.time()

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

end_time_preprocess_he = time.time()

beta = 2 * np.random.rand(9) - 1
print("Initial beta:", beta)

msg_beta = [0] * num_slot

ctxt_next = heaan.Block(context, encrypted = False, data = msg_beta)
ctxt_next = ctxt_next.encrypt()

for i in range(8):
    for j in range(train_n):
        msg_beta[train_n * i + j] = beta[i + 1]

for j in range(train_n):
    msg_beta[8 * train_n + j] = beta[0]

print("Initial msg_beta:", msg_beta)

ctxt_beta = heaan.Block(context, encrypted = False, data = msg_beta)
ctxt_beta = ctxt_beta.encrypt()

learning_rate = 0.28
num_steps = 10

# Training the model in HE
start_time_train_he = time.time()
for i in range(num_steps):
    print("=== HE Step", i, "===")
    ctxt_next = step(learning_rate, ctxt_X, ctxt_Y, ctxt_next, train_n)
    beta_ctxt = ctxt_next
    beta_ctxt.decrypt()
    for j in range(10):
        print("ctxt beta : ", beta_ctxt[j])
end_time_train_he = time.time()

print("Homomorphic Encryption Training Time: ", end_time_train_he - start_time_train_he)

# Evaluation in HE
test_n = df.shape[0]

start_time_eval_he = time.time()

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

# Accuracy in HE
ctxt_infer = compute_sigmoid(ctxt_X_test, ctxt_next, test_n)
res = ctxt_infer.decrypt()
cnt = 0
for i in range(test_n):
    cnt += (1 if (res[i].real >= 0.5) == Y_test[i] else 0)
accuracy_he = cnt / test_n
end_time_eval_he = time.time()

print("Homomorphic Encryption Evaluation Time: ", end_time_eval_he - start_time_eval_he)
print("Homomorphic Encryption Accuracy: ", accuracy_he)

# Plaintext logistic regression for comparison
class PlaintextLogisticRegression:
    def __init__(self, learning_rate, num_steps):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.beta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, m = X.shape
        self.beta = np.random.randn(m)
        for step in range(self.num_steps):
            z = np.dot(X, self.beta)
            predictions = self.sigmoid(z)
            gradient = np.dot(X.T, (predictions - y)) / n
            self.beta -= self.learning_rate * gradient
            print(f"Step {step} - Beta: {self.beta}")

    def predict(self, X):
        z = np.dot(X, self.beta)
        return self.sigmoid(z) >= 0.5

# Preprocessing and training in plaintext
start_time_preprocess_plain = time.time()
X_plain = np.column_stack([normalize_data(df[col].values) if col in ['LVR', 'RATE', 'AMOUNT', 'CREDIT', 'TERM'] else df[col].values for col in ['LVR', 'REF', 'INSUR', 'RATE', 'AMOUNT', 'CREDIT', 'TERM', 'ARM']])
y_plain = df['DELINQUENT'].values
end_time_preprocess_plain = time.time()

start_time_train_plain = time.time()
model_plain = PlaintextLogisticRegression(learning_rate=0.28, num_steps=100)
model_plain.fit(X_plain, y_plain)
end_time_train_plain = time.time()

# Evaluation in plaintext
start_time_eval_plain = time.time()
predictions_plain = model_plain.predict(X_plain)
accuracy_plain = accuracy_score(y_plain, predictions_plain)
end_time_eval_plain = time.time()

print("Plaintext Preprocessing Time: ", end_time_preprocess_plain - start_time_preprocess_plain)
print("Plaintext Training Time: ", end_time_train_plain - start_time_train_plain)
print("Plaintext Evaluation Time: ", end_time_eval_plain - start_time_eval_plain)
print("Plaintext Accuracy: ", accuracy_plain)

# Comparing timings
print("Homomorphic Encryption Preprocessing Time: ", end_time_preprocess_he - start_time_preprocess_he)
print("Plaintext Preprocessing Time: ", end_time_preprocess_plain - start_time_preprocess_plain)
print("Homomorphic Encryption Training Time: ", end_time_train_he - start_time_train_he)
print("Plaintext Training Time: ", end_time_train_plain - start_time_train_plain)
print("Homomorphic Encryption Evaluation Time: ", end_time_eval_he - start_time_eval_he)
print("Plaintext Evaluation Time: ", end_time_eval_plain - start_time_eval_plain)

# Comparing accuracies
print("Homomorphic Encryption Accuracy: ", accuracy_he)
print("Plaintext Accuracy: ", accuracy_plain)
