import numpy as np
import tensorflow as tf

def get_next_batch(index, batch_size, x_train, y_train, N):
    # just gets batches of images to
    # train or validate the model
    #
    # @inputs
    #
    # index -> some integer from iteration loop
    # batch_size -> number of images to process simultaneously
    # x_train -> image set
    # y_train -> label set
    # N -> image size (assume square)
    print(index)
    print(x_train.shape)
    print(y_train.shape)
    
    X_batch = x_train[index*batch_size:(index+1)*batch_size].reshape(
        batch_size,N**2)
    y_batch = y_train[index*batch_size:(index+1)*batch_size]
    return X_batch, y_batch

def normalize_output(A):
    m0,n0 = A.shape
    print(m0,n0)
    norm = tf.tile(tf.reshape(tf.reduce_max(A,axis=1),[m0,1]),[1,n0])
    norm_A = tf.square(A/norm)
    return norm_A

def normalize_output_2(X,N):
    '''
    normalizes tensor X to have max value = 1
    '''
    max_tensor = tf.math.reduce_max(X,axis=1)
    max_tensor_list = []
    for i in range(0,N**2):
        max_tensor_list.append(max_tensor)

    max_tensor = tf.transpose(tf.stack(max_tensor_list))
    print(X, max_tensor)
    out = X/max_tensor
    return out

def get_next_batch_misaligned(index, batch_size, 
                              x_train, y_train,
                              N, pixel_misalignment):
    # just gets batches of images to
    # train or validate the model
    #
    # @inputs
    #
    # index -> some integer from iteration loop
    # batch_size -> number of images to process simultaneously
    # x_train -> image set
    # y_train -> label set
    # N -> image size (assume square)

    #X_batch = x_train[index*batch_size:(index+1)*batch_size].reshape(
    #    batch_size,N**2)

    X_batch = x_train[index*batch_size:(index+1)*batch_size]

    for i in range(0, batch_size):
        shift_x = np.random.randint(-pixel_misalignment,
                                    pixel_misalignment)
        shift_y = np.random.randint(-pixel_misalignment,
                                    pixel_misalignment)
        X_batch[i] = np.roll(X_batch[i], shift_x, axis=0)
        X_batch[i] = np.roll(X_batch[i], shift_y, axis=1)

    X_batch = X_batch.reshape(batch_size, N**2)
    y_batch = y_train[index*batch_size:(index+1)*batch_size]
    return X_batch, y_batch

def get_next_batch_and_inject_random_phase(index, batch_size,
                                           x_train, y_train,
                                           N):
    # just gets batches of images to
    # train or validate the model
    # and injects a random phase function
    # to model image as illuminated by
    # incoherent light source
    #
    # @inputs
    #
    # index -> some integer from iteration loop
    # batch_size -> number of images to process simultaneously
    # x_train -> image set
    # y_train -> label set
    # N -> image size (assume square)

    X_batch = x_train[index*batch_size:(index+1)*batch_size]
    x_batch_size = X_batch.shape
    X_batch = X_batch*np.exp(1j*np.random.uniform(0,2*np.pi,size=x_batch_size))
    
    X_batch = X_batch.reshape(batch_size, N**2)
    y_batch = y_train[index*batch_size:(index+1)*batch_size]
    return X_batch, y_batch
