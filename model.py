import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from physics import *
import matplotlib.pyplot as plt

###################################
# ONN model defs ##################
###################################
N = 200 # neurons per 1 dimension (assume square)
c = 3e8 # speed of light
wavelength = 1.55 # microns
NEURON_SIZE = 1.2 # 3x3 scatterers/neuron, periodicity is 1.2
LAYER_SIZE =  NEURON_SIZE*N
LAYER_SPACING = 50*1.5 # 50 um substrate, 75 um total light pathlength
detector_width = 30*wavelength
DEBUG_MODE = 1


##################################
# Define coordiante space ########
##################################
xx1 = np.arange(-LAYER_SIZE/2,LAYER_SIZE/2,NEURON_SIZE)
yy1 = np.arange(-LAYER_SIZE/2,LAYER_SIZE/2,NEURON_SIZE)
print(xx1.shape)
XX1,YY1 = np.meshgrid(xx1,yy1)
XX1 = XX1.reshape(N**2)
YY1 = YY1.reshape(N**2)

########################
# define detector system
########################

pix_size = LAYER_SIZE/N
det_width_pix = int(detector_width/pix_size)
det_pos = {1:(-50,0), 
               2:(1000,1000), 
               3:(50,0),
               4:(1000,1000),
               5:(1000,1000),
               6:(1000,1000),
               7:(1000,1000),
               8:(1000,1000),
               9:(1000,1000),
               0:(1000,1000)
}

div_fact = 200/N
for key in det_pos:
    print(div_fact, det_pos[key])
    det_pos[key] = (int(det_pos[key][0]/div_fact),int(det_pos[key][1]/div_fact))
    print(det_pos[key][0],det_pos[key][1])

detector_matrix = np.zeros((10,N**2))
for i in range(0,10):
    pos_x = det_pos[i][0]
    pos_y = det_pos[i][1]
    detector_matrix[i][(XX1<=pos_x*pix_size+detector_width/2)
                   &(XX1>=pos_x*pix_size-detector_width/2)
                   &(YY1<=pos_y*pix_size+detector_width/2)
                   &(YY1>=pos_y*pix_size-detector_width/2)] = 1
    print(np.sum(detector_matrix[i]))

# free up memory
del XX1, YY1

all_detectors = np.zeros((N,N))
for j in range(0,10):
    all_detectors += detector_matrix[j].reshape((N,N))

if DEBUG_MODE:
    plt.figure()
    plt.imshow(all_detectors)
    plt.colorbar()
    plt.show()


#########################################
# preprocess all images #################
#########################################
mnist = tf.keras.datasets.mnist # load mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train[(y_train == 1) | (y_train == 3)]
x_test = x_test[(y_test==1) | (y_test==3)]
y_test = y_test[(y_test==1) | (y_test==3)]
y_train = y_train[(y_train == 1) | (y_train == 3)]

init_op = tf.initialize_all_variables() # define a tf op
x_train = x_train/255.0 # normalize all to 1
x_test = x_test/255.0 # ---//---

###########################################
# reshape all images to fit ONN size ######
###########################################
len_train = len(x_train)
len_test = len(x_test)
x_train = tf.reshape(x_train, [len(x_train),28,28,1])
x_test = tf.reshape(x_test, [len(x_test),28,28,1])
x_train = tf.image.resize_images(x_train,[N,N])
x_test = tf.image.resize_images(x_test,[N,N])
x_train = tf.reshape(x_train,[len_train,N,N])
x_test = tf.reshape(x_test,[len_test,N,N])
print(x_train,'train me fam')
with tf.Session() as sess:
    sess.run(init_op)
    x_test = sess.run(x_test)
    x_train = sess.run(x_train)

print(x_train.shape,'train shape')
print('processed')

################################
# training defs ################
################################
m = int(x_train.shape[0])
n_epochs = 8
batch_size = 30
n_batches = int(np.floor(m / batch_size))
print(n_batches)
learning_rate = 0.001

# construct model
print('constructing model')

################################
# input/label tensor alloc #####
################################
inputs = tf.placeholder(tf.complex128, shape=(batch_size,N**2),name='inputs')
labels = tf.placeholder(tf.int32, shape=(batch_size,),name='labels')

d = tf.constant(detector_matrix.reshape((10,N**2)),
               dtype = tf.float64,
               name = 'detectors')
all_d = tf.constant(all_detectors.reshape(1,N**2), dtype =tf.float64, name = 'all_detectors')

################################
# propagates light to the onn ##
# ##############################
#
# -> |                                 |               |
# -> | slide thickness - LAYER_SPACING | LAYER_SPACING | ONN BEGINS
# -> |                                 |               |
# -> |                                 |               |
# -> |                                 |               |
#    |       TOTAL SLIDE THICKNESS                     |                                      
slide_index = 1.5
to_nn_space = 700*slide_index-LAYER_SPACING
propagator_to_NN = tf_propagation_padded(xx1,yy1,wavelength,to_nn_space)
inputs1 = propagator_to_NN.propagate_tf_fft(inputs)

#####################
# ONN model begins ##
#####################
propagator_inside_NN = tf_propagation_padded(xx1,yy1,wavelength,LAYER_SPACING)
outputs1 = onn_layer_amp_and_phase(inputs1,N)
outputs1 = physical_nonlinearity(outputs1)

outputs1 = propagator_inside_NN.propagate_tf_fft(outputs1)
print('layer 1 propagated')

#measured_im = measurement_layer(outputsf, batch_size, N, 3)
measured_im = np.abs(outputs1)**2
normalized_im = normalize_output_2(measured_im,N)*10

#out = electronic_layer_N2x10_general(normalized_im)


#Cross Entropy
#print('computing loss')
out = tf.matmul(normalized_im,tf.transpose(d)) #only sum over detectors
print(out,'this is out')
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=out,name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
print(tf.trainable_variables)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer() 
saver = tf.train.Saver()

#Summary
#loss_summary = tf.summary.scalar('Loss', loss)
file_writer = tf.summary.FileWriter("tf_logs", tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        epoch_loss = 0 #reset loss
        for batch_index in range(n_batches):
            X_batch, y_batch = get_next_batch(batch_index, batch_size, x_train, y_train, N)
            
            #print(X_batch.shape)
            if batch_index % 100 == 0:
                #summary_str = loss_summary.eval(feed_dict={inputs: X_batch, labels: y_batch})
                step = epoch * n_batches + batch_index
                #file_writer.add_summary(summary_str, step)
            _, batch_loss = sess.run([training_op,loss], feed_dict={inputs: X_batch, labels: y_batch})
            epoch_loss += batch_loss
            #print(batch_loss)
        print("Epoch", epoch, "Loss =", epoch_loss/n_batches) #print out loss averaged over all batches              
        save_path = saver.save(sess, "tmp/my_model.ckpt")
    
    save_path = saver.save(sess, "tmp/my_model_final.ckpt")
    
    
     
file_writer.close()

# test session
'''
with tf.Session() as sess:
    sess.run(init)
    #X_batch = tf.reshape(x_train[0],[batch_size,N**2])
    X_batch = x_train[0:batch_size].reshape([batch_size,N**2])
    print(X_batch)
    y_batch = y_train[0:batch_size]
    print(y_batch)
    loss_out = sess.run([loss],feed_dict={inputs: X_batch, labels: y_batch})
    #outputsffs = sess.run(outputs1,feed_dict={inputs: X_batch, labels: y_batch})
    
    print(loss_out)
'''
