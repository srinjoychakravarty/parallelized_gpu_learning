import numpy as np
import tensorflow
import datetime

def matrix_indices(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tensorflow.matmul(M, matrix_indices(M, n-1))


log_device_placement = True # Processing Units logs

n = 10  # Num of multiplications to perform

A = np.random.rand(10000, 10000).astype('float32')  # Create random large matrix
B = np.random.rand(10000, 10000).astype('float32')  # Create random large matrix

#a = np.zeros([10000, 10000], dtype = 'float16')
#A = tensorflow.convert_to_tensor(a)                         # Create large matrix of 0's
#b = np.zeros([10000, 10000], dtype = 'float16')
#B = tensorflow.convert_to_tensor(b)                         # Create large matrix of 0's

# Create 2 empty lists to store results
c1 = []
c2 = []

'''
Single GPU computing
'''
with tensorflow.device('/gpu:0'):   # The first GPU of our machine
    a = tensorflow.placeholder(tensorflow.float32, [10000, 10000])
    b = tensorflow.placeholder(tensorflow.float32, [10000, 10000])
    # Compute A^n and B^n and store results in c1
    c1.append(matrix_indices(a, n))
    c1.append(matrix_indices(b, n))

with tensorflow.device('/cpu:0'):   # "/cpu:0": The CPU of your machine.
  sum = tensorflow.add_n(c1) # Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()


'''
Multi GPU computing
'''
# GPU:0 computes A^n
with tensorflow.device('/gpu:0'):   # The first GPU of our machine
    # Compute A^n and store result in c2
    a = tensorflow.placeholder(tensorflow.float32, [10000, 10000])
    c2.append(matrix_indices(a, n))

# GPU:1 computes B^n
with tensorflow.device('/gpu:1'):   # The second GPU of our machine
    # Compute B^n and store result in c2
    b = tensorflow.placeholder(tensorflow.float32, [10000, 10000])
    c2.append(matrix_indices(b, n))

with tensorflow.device('/cpu:0'):
  sum = tensorflow.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))