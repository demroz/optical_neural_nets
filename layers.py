import tensorflow as tf
import numpy as np
import pandas as pd

def onn_layer(X,N):
    '''
    X > input field

    N > number of neurons in one dimension
    
    '''
    theta = tf.Variable(tf.constant(np.pi/2,shape=[1,N**2],dtype=tf.float64),name='theta')
    phase_mod = tf.complex(tf.cos(theta),tf.sin(theta),name='phase_mod')
    X=X*phase_mod
    return X

def onn_layer_for_testing(X,N, phase):
    theta = phase
    #theta = tf.Variable(tf.constant(np.pi/2,shape=[1,N**2],dtype=tf.float64),name='theta')
    phase_mod = tf.complex(tf.cos(theta),tf.sin(theta),name='phase_mod')
    X=X*phase_mod
    return X

def onn_layer_noisy(X, N, noise_percentage):
    noise = tf.random.uniform(shape = [1, N**2],
                              minval = -noise_percentage*2*np.pi,
                              maxval = noise_percentage*2*np.pi,
                              dtype = tf.float64)
    theta = tf.Variable(tf.constant(np.pi/2,shape=[1,N**2],dtype=tf.float64),name='theta')+noise
    phase_mod = tf.complex(tf.cos(theta),tf.sin(theta),name='phase_mod')
    X=X*phase_mod
    return X

def onn_layer_amp_and_phase(X, N):
    alpha = tf.Variable(tf.constant(0,shape=[1,N**2],dtype=tf.float64),name='amp_train')
    temp_0 = tf.constant(0,shape=[1,N**2],dtype=tf.float64)
    transmission = tf.complex(tf.sigmoid(alpha,name='t_coeff'),temp_0)
    theta = tf.Variable(tf.constant(np.pi/2,shape=[1,N**2],dtype=tf.float64),name='theta')
    phase_mod = tf.complex(tf.cos(theta),tf.sin(theta),name='phase_mod')
    print(X,phase_mod,transmission)
    X=X*phase_mod
    X=X*transmission
    return X

def physical_nonlinearity(X):
    p1 = 0.594
    p2 = -1.868
    p3 = 2.321
    p4 = -1.478
    p5 = 0.5502
    p6 = 0.03134
    p7 = -0.0002703

    out = p1*X**6+p2*X**5+p3*X**4+p4*X**3+p5*X**2+p6*X+p7
    
    return out

def inverse_physical_nonlinearity(X):
    p1 = -0.594
    p2 = 1.868
    p3 = -2.321
    p4 = 1.478
    p5 = -0.5502
    p6 = -0.03134
    p7 = 0.0002703

    out = p1*X**6+p2*X**5+p3*X**4+p4*X**3+p5*X**2+p6*X+p7

    return out
def electronic_layer_N2(X,N):
    '''
    electronic layer for neural network

    X > intensity field. assumes it is normalized to 1
    
    N > number of neurons in one dimension
    '''
    
    # nonlinear activation function
    Y = (tf.math.tanh(10*(X-0.5))+1)/2

    e_layer_matrix = tf.Variable(tf.random.uniform(shape=[N**2,N**2],dtype=tf.float64),name='e_layer')
    out = tf.matmul(Y,e_layer_matrix)
    

    #sess = tf.Session()
    #print(sess.run(out))
    return out

def electronic_layer_N2x10(X,N):
    '''
    electronic layer for neural network

    X > intensity field. assumes it is normalized to 1
    
    N > number of neurons in one dimension
    '''
    
    # nonlinear activation function
    # Y = tf.math.sigmoid(X)
    # Y = (tf.math.tanh(10*(X-0.5))+1)/2
    Y = tf.nn.relu(X+0.3)
    e_layer_matrix = tf.Variable(tf.random.uniform(shape=[N**2,10],dtype=tf.float64),name='e_layer')
    out = tf.matmul(Y,e_layer_matrix)
    out  = tf.nn.relu(out)
    return out

def electronic_layer_N2x10_general(X):
    '''
    electronic layer for neural network

    X > intensity field. assumes it is normalized to 1
    
    N > number of neurons in one dimension
    '''
    N = int(X.shape[1])
    # nonlinear activation function
    # Y = tf.math.sigmoid(X)
    Y = (tf.math.tanh((X-5))+1.0)/2.0
    # Y = tf.nn.relu(X+0.3)
    e_layer_matrix = tf.Variable(tf.random.uniform(shape=[N,10],dtype=tf.float64),name='e_layer')
    out = tf.matmul(Y,e_layer_matrix)
    out  = tf.nn.relu(out)
    return out

def convolutional_layer(X, N, batch_size, layer_size):
    '''
    convolutional layer

    X  > input
    
    N > number of neurons in 1 dimension of optical layer

    layer_size > size of convolutional layer. since I'm using
                 N = 90, a reasonable size is probably 9x9, since
                 images are 1-D, layer_size**2 is taken to be the size
                 of the layer
    '''

    # nonlinearity
    X = (tf.math.tanh(10*(X-0.5))+1)/2
    X = tf.reshape(X, [batch_size, N, N, 1])
    kernel = tf.Variable(tf.random.uniform(shape = [layer_size,layer_size,1,1], dtype = tf.float64),
                         name = 'conv_layer')
    
    Y = tf.nn.conv2d(X, kernel, strides = [1,1,1,1], padding='SAME')
    Y = tf.reshape(Y, [batch_size, N**2])

    return Y

def convolutional_layer_2(X,N,batch_size, layer_size, num_outputs):
    '''
    convolutional layer

    X  > input
    
    N > number of neurons in 1 dimension of optical layer
    
    layer_size > size of convolutional layer. since I'm using
    N = 90, a reasonable size is probably 9x9, since
    images are 1-D, layer_size**2 is taken to be the size
    of the layer
    '''
    X = (tf.math.tanh(10*(X-0.5))+1)/2
    X = tf.reshape(X, [batch_size, N, N, 1])

    kernel = tf.Variable(tf.random.uniform(shape = [layer_size,layer_size,1,num_outputs], 
                                           dtype = tf.float64),name = 'conv_layer')
    print(kernel)
    Y = tf.nn.conv2d(X, kernel, strides = [1,1,1,1], padding='SAME')
    print('convolved',Y)
    Y = (tf.math.tanh(10*(Y-0.5))+1)/2
    print('nonlinear',Y)
    out = tf.nn.max_pool(Y,(1,layer_size, layer_size,1),strides = (1,layer_size, layer_size, 1),
                         padding = 'SAME')
    print(out)
    out_shape = int(out.shape[1]*out.shape[2]*out.shape[3])
    out = tf.reshape(out, (batch_size, out_shape))
    return out

def measurement_layer(X, batch_size, N_pix_model, pool_size):
    '''
    essentially average pooling layer

    X > input
    
    N_pix_model > number of neurons in 1 dim on optical layer

    pool_size > size of average pool
    
    batch_size > batch size
    '''
    X = tf.abs(X)**2
    X = tf.reshape(X, [batch_size,N_pix_model,N_pix_model,1])
    Y = tf.layers.average_pooling2d(X, [pool_size,pool_size], strides = [pool_size, pool_size])

    # flatten
    Y = tf.reshape(Y,[batch_size,int(N_pix_model/pool_size)**2])
    return Y
