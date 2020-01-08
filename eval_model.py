import sys
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
from utils import *
from layers import onn_layer
import matplotlib.pyplot as plt

N = 200
sess = tf.Session()
saver = tf.train.import_meta_graph('my_model_final.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

# eval data
#normalize all images to 1
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train[(y_train == 1) | (y_train == 3)]
x_test = x_test[(y_test==1) | (y_test==3)]
y_test = y_test[(y_test==1) | (y_test==3)]
y_train = y_train[(y_train == 1) | (y_train == 3)]

print('preprocessing')
init_op = tf.initialize_all_variables()
x_train = x_train/255.0
x_test = x_test/255.0
print(x_train.shape,'shape')
len_train = len(x_train)
len_test = len(x_test)
x_train = tf.reshape(x_train, [len(x_train),28,28,1])
x_test = tf.reshape(x_test, [len(x_test),28,28,1])
x_train = tf.image.resize_images(x_train,[N,N])
x_test = tf.image.resize_images(x_test,[N,N])
x_train = tf.reshape(x_train,[len_train,N,N])
x_test = tf.reshape(x_test,[len_test,N,N])

x_test = sess.run(x_test)
x_train = sess.run(x_train)


print(x_train.shape,'train shape')
# get graph
graph = tf.get_default_graph()

inputs = graph.get_tensor_by_name("inputs:0")
labels = graph.get_tensor_by_name("labels:0")

op = graph.get_tensor_by_name("MatMul:0")

batch_size = 30
n_batches = 333

index = np.arange(0,10)
acc = 0
total = 0
conf_mat = np.zeros((10,10))
for batch_index in range(n_batches):
    batch_acc = 0
    X_batch, y_batch = get_next_batch(batch_index, batch_size, x_test, y_test, N)
    feed_dict = {inputs: X_batch, labels: y_batch}
    a = sess.run(op,feed_dict)
    b = np.argmax(a, axis=1)
    print(b.shape)
    batch_sum = np.sum(b==y_batch)
    batch_acc = batch_sum/batch_size
    print('batch accuracy = ', batch_acc)
    total += batch_sum
    acc = total/((batch_index+1)*batch_size)
    print('total accuracy = ', acc)

    for i in range(0,30):
        true_value = y_batch[i]
        classified_value = b[i]
        conf_mat[true_value, classified_value] += 1


np.savetxt('confusion_matrix.csv',conf_mat)
