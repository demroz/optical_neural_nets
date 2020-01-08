import tensorflow as tf
import numpy as np
class tf_propagation_focal_axis():
    '''
    tensorflow propagation along optical axis
    DEPRECIATED use tf_propagation_padded
    '''
    def __init__(self, xx, yy, lmb, distance):
        '''
        xx, yy -> location of input field x,y
        
        lmb -> wavelength

        distance -> propagation distance along z axis

        Nx, Ny -> number of field points in x/y direction
        '''
        WARNING = '''WARNING!!!! Its possible the boundary conditions of this
                   function make the field wrap around on itself. Use
                   tf_propagation_padded() to ensure accurate results'''
        print(WARNING)
        self.debug=0
        self.xx = xx
        self.Nx = len(self.xx)
        self.yy = yy
        self.Ny = len(self.yy)
        self.lmb = lmb
        self.distance = distance

    def _propagate_tf_fft(self, input):
        '''
        propagation for tensor input

        Ur code is poo
        '''
        input = tf.reshape(input,[self.Nx,self.Ny])

        Ax = self.xx[0]
        Ay = self.yy[0]

        # create recipocal grid space
        k_xlist_pos = 2*np.pi*np.linspace(-0.5*self.Nx/(2*Ax),0.5*self.Nx/(2*Ax),self.Nx)
        k_ylist_pos = 2*np.pi*np.linspace(-0.5*self.Ny/(2*Ay),0.5*self.Ny/(2*Ay),self.Ny)
        k_x, k_y = np.meshgrid(k_xlist_pos,k_ylist_pos)
        k = 2*np.pi/self.lmb

        k_z = np.sqrt(k**2-k_x**2-k_y**2)

        # propagator kernel
        H_freq = np.exp(1.0j*k_z*self.distance)

        # convolution
        #out = tf.ifft2d(np.fft.fftshift(H_freq)*tf.fft2d(input))
        out = tf.reshape(out,[self.Nx*self.Ny,])
        return out
    
    def propagate_tf_fft(self,input):
        # wrapper to accomodate batch training
        return tf.map_fn(self._propagate_tf_fft,input)


class tf_propagation_waveguide():
    '''
    generalized tensorflow propagation in a waveguide
    
    DEPRECIATED use tf_propagation_padded
    '''
    def __init__(self, xx1, yy1, xx2, yy2, lmb, z):
        '''
        @inputs
        xx1, yy1 -> x,y coordinates of known field at some point z0
        xx2, yy2 -> x,y coordinates of desired field at some point z

        coordinates are assumed to be increasing with distance, i.e.
        field propagates in positive x and y directions. So
        yy2 => yy1
        xx2 => xx1
        necessarily
        
        z - > propagation distance

        lmb -> wavelength
        '''
        WARNING = '''WARNING!!!! Its possible the boundary conditions of this
                   function make the field wrap around on itself. Use
                   tf_propagation_padded() to ensure accurate results'''
        print(WARNING)
        # turn on/off for debugging
        self.DEBUG_FLAG = 1
        
        self.xx1 = xx1
        self.yy1 = yy1
        self.xx2 = xx2
        self.yy2 = yy2
        self.lmb = lmb
        self.distance = z

        self._construct_propagation_grid()
    
    def random_defocus_on(self,minval,maxval):
        self.distance = self.distance+np.random.uniform(minval,maxval)
        print(self.distance)
    def _construct_propagation_grid(self):
        '''
        creates coordinate grid for angular
        propagation. It is assumed that field propagates
        along the z direction

        coordinates are assumed to be increasing with distance, i.e.
        field propagates in positive x and y directions
        '''
        self.xmax = np.max(self.xx2)
        self.xmin = np.min(self.xx1)
        self.ymax = np.max(self.yy2)
        self.ymin = np.min(self.yy1)

        dx = self.xx1[1]-self.xx1[0]
        dy = self.yy1[1]-self.yy1[0]
        
        self.xx = np.arange(self.xmin,self.xmax+dx/2,dx)
        self.yy = np.arange(self.ymin,self.ymax+dy/2,dy)
        self.Nx = len(self.xx)
        self.Ny = len(self.yy)

    def _pad_input(self, input):
        nx_og = len(self.xx1)
        ny_og = len(self.yy1)
        input = tf.reshape(input,[nx_og, ny_og])
        
        pad_x = len(self.xx)-len(self.xx1)
        pad_y = len(self.yy)-len(self.yy1)

        print(tf.pad(input, [[0,pad_y],[0,pad_x]]))
        return tf.pad(input, [[0,pad_y],[0,pad_x]])


        
    def _propagate_tf_fft(self, input):
        '''
        propagation for tensor input
        '''
        input = self._pad_input(input)

        Ax = self.xx[0]
        Ay = self.yy[0]

        # create recipocal grid space
        k_xlist_pos = 2*np.pi*np.linspace(-0.5*self.Nx/(2*Ax),0.5*self.Nx/(2*Ax),self.Nx)
        k_ylist_pos = 2*np.pi*np.linspace(-0.5*self.Ny/(2*Ay),0.5*self.Ny/(2*Ay),self.Ny)

        k_x, k_y = np.meshgrid(k_xlist_pos,k_ylist_pos)

        k = 2*np.pi/self.lmb

        k_z = np.sqrt(k**2-k_x**2-k_y**2+0j)

        # propagator kernel
        H_freq = np.exp(1.0j*k_z*self.distance)

        # convolution
        self.out = tf.ifft2d(tf.fft2d(input)*np.fft.ifftshift(H_freq))
        nx_og = len(self.xx)
        ny_og = len(self.yy)
        #self.out = self.out[self.Nx-nx_og-1:-1, self.Ny-ny_og-1:-1]
        print(self.Nx, nx_og, self.out)
        self.out = self.out[self.Nx-nx_og:self.Nx,self.Ny-ny_og:self.Ny]
        #out = tf.reshape(out,[self.Nx*self.Ny,])
        self.out = tf.reshape(self.out, [self.Nx*self.Ny])
        
        return self.out

    def propagate_tf_fft(self,input):
        # wrapper to accomodate batch training
        return tf.map_fn(self._propagate_tf_fft,input)

class tf_propagation_padded():
    ''' 
    angular spectrum propagation, along focal axis.
    Pads inputs to 2x the og size to avoid
    periodic boundary condition overlap
    '''
    def __init__(self, xx, yy, lmb, z):
        '''
        @inputs
        xx, yy -> x,y coordinates of known field at some point z0
        
        z - > propagation distance

        lmb -> wavelength
        '''        
        self.xx = xx
        self.yy = yy
        self.lmb = lmb
        self.distance = z

        self._construct_propagation_grid()

        self.debug = 0
    def _construct_propagation_grid(self):
        '''
        creates coordinate grid for angular
        spectruc propagation. Field propagates
        in the ''z'' direction
        '''

        xmax_og = np.max(self.xx)
        xmin_og = np.min(self.xx)
        ymax_og = np.max(self.yy)
        ymin_og = np.min(self.yy)

        len_x_og = xmax_og-xmin_og
        len_y_og = ymax_og-ymin_og

        xmax_new = xmax_og+len_x_og/2.
        xmin_new = xmin_og-len_x_og/2.
        ymax_new = ymax_og+len_y_og/2.
        ymin_new = ymin_og-len_y_og/2.        

        self.num_pix_x_og = len(self.xx)
        self.num_pix_y_og = len(self.yy)
        
        #self.num_pix_x_new = self.num_pix_x_og*2
        #self.num_pix_y_new = self.num_pix_y_og*2

        # new grid
        #self.xx_new = np.arange(xmin_new, xmax_new, self.num_pix_x_new)
        #self.yy_new = np.arange(ymin_new, ymax_new, self.num_pix_x_new)

        # pad size
        self.pad_x = np.int32(np.ceil(len(self.xx)/2.))
        self.pad_y = np.int32(np.ceil(len(self.yy)/2.))

        # specify lhs and rhs to append to old
        # array
        l_x = np.linspace(xmin_new,xmin_og,self.pad_x)
        r_x = np.linspace(xmax_og,xmax_new,self.pad_x)
        l_y = np.linspace(ymin_new,ymin_og,self.pad_y)
        r_y = np.linspace(ymax_og,ymax_new,self.pad_y)

        # construct new coordinate grid
        self.xx_new = np.insert(self.xx,0,l_x)
        self.xx_new = np.append(self.xx_new,r_x)
        self.yy_new = np.insert(self.yy,0,l_y)
        self.yy_new = np.append(self.yy_new,r_y)

        # new lengths
        self.num_pix_x_new = len(self.xx_new)
        self.num_pix_y_new = len(self.yy_new)
    def _pad_input(self, input):
        ''' 
        pads input to 2x the size. so 0.5x the grid
        on each side of the image
        '''
        input = tf.reshape(input, [self.num_pix_x_og, self.num_pix_y_og])

        ''' pad a single size of the image
        on each side'''

        out = tf.pad(input, [[self.pad_y,self.pad_y],
                             [self.pad_x,self.pad_x]])

        if self.debug:
            with tf.Session() as sess:
                out_disp = sess.run(out)
                print(out_disp)
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(np.abs(out_disp)**2)
                plt.show()
                
        return out

    def _propagate_tf_fft(self, input):
        '''
        propagation for tensor input
        '''
        input = self._pad_input(input)

        Ax = self.xx_new[0]
        Ay = self.yy_new[0]

        # create recipocal grid space
        k_xlist_pos = 2*np.pi*np.linspace(-0.5*self.num_pix_x_new/(2*Ax)
                                          ,0.5*self.num_pix_x_new/(2*Ax),self.num_pix_x_new)
        k_ylist_pos = 2*np.pi*np.linspace(-0.5*self.num_pix_y_new/(2*Ay),
                                          0.5*self.num_pix_y_new/(2*Ay),self.num_pix_y_new)

        k_x, k_y = np.meshgrid(k_xlist_pos,k_ylist_pos)

        k = 2*np.pi/self.lmb

        k_z = np.sqrt(k**2-k_x**2-k_y**2+0j)

        # propagator kernel
        H_freq = np.exp(1.0j*k_z*self.distance)

        # convolution
        self.out = tf.ifft2d(tf.fft2d(input)*np.fft.ifftshift(H_freq))
        
        self.out = self.out[self.pad_x:self.num_pix_x_og+self.pad_x,
                            self.pad_y:self.num_pix_y_og+self.pad_x]
        self.out = tf.reshape(self.out, [self.num_pix_x_og*self.num_pix_y_og])
        
        return self.out

    def propagate_tf_fft(self,input):
        # wrapper to accomodate batch training
        return tf.map_fn(self._propagate_tf_fft,input)
        
    
