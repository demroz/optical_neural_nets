import tensorflow as tf
import csv
import numpy as np

# constants
N = 90 # neurons per 1 dimension (assume square)
c = 3e8 # speed of light
wavelength = 1.55 # microns
NEURON_SIZE = 1.2 # 3x3 scatterers/neuron, periodicity is 1.2
LAYER_SIZE =  NEURON_SIZE*N
#detector_width = 10*wavelength
#LAYER_SPACING = 40*wavelength
LAYER_SPACING = 50*1.5 # 50 um substrate, 75 um total light pathlength

DEBUG_MODE = 1

# physical coordinates
xx1 = np.arange(-LAYER_SIZE/2,LAYER_SIZE/2,NEURON_SIZE)
yy1 = np.arange(-LAYER_SIZE/2,LAYER_SIZE/2,NEURON_SIZE)
print(xx1.shape)
XX1,YY1 = np.meshgrid(xx1,yy1)
XX1 = XX1.reshape(N**2)
YY1 = YY1.reshape(N**2)

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_model_final.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
 
 
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
 
graph = tf.get_default_graph()
graph = tf.get_default_graph()
c1 = graph.get_tensor_by_name("phase_mod:0")
#c2 = graph.get_tensor_by_name("phase_mod_1:0")
#c3 = graph.get_tensor_by_name("phase_mod_2:0")
#c4 = graph.get_tensor_by_name("phase_mod_3:0")
#c5 = graph.get_tensor_by_name("phase_mod_4:0")

theta1 = graph.get_tensor_by_name("theta:0")
#theta2 = graph.get_tensor_by_name("theta_1:0")
#theta3 = graph.get_tensor_by_name("theta_2:0")
#theta4 = graph.get_tensor_by_name("theta_3:0")
#theta5 = graph.get_tensor_by_name("theta_4:0")
print(theta1)
theta1 = theta1.eval(session=sess)
#theta2 = theta2.eval(session=sess)
#theta3 = theta3.eval(session=sess)
#theta4 = theta4.eval(session=sess)
#theta5 = theta5.eval(session=sess)
#theta1 = theta1.reshape(28,28)
import numpy as np
#phase_mask = c1.eval(session=sess)
#phase_mask = phase_mask.reshape(28,28)
#print(phase_mask.shape)
#print(phase_mask.dtype)
#import matplotlib.pyplot as plt
#plt.figure()
#plt.imshow(theta1)
#plt.colorbar()
#plt.show()

theta = np.asarray([theta1,theta2,theta3,theta4,theta5])
theta = theta%(2*np.pi)
print(theta.shape)
for i in range(0,len(theta)):
    with open('phase_mask_'+str(i)+'.csv','w') as writefile:
        writer = csv.writer(writefile)
        for j in range(0,len(XX1)):
            writer.writerow([XX1[j],YY1[j],theta[i,0,j]])
