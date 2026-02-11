import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, hermite
#import tensorflow_addons as tfa
from PhaseplateNetwork.utils.logic_functions import logical_and, SR_latch
from PhaseplateNetwork.utils.ImageUtils import get_gaussian_beam_image, get_minerva_image, get_impossible_cube_image, get_penrose_triangle_image
from PIL import Image, ImageDraw
import tensorflow_datasets as tfds
import h5py
import os

centers512 = np.array([[120, 90],
                       [270, 90],
                       [420, 90],
                       [120, 220],
                       [270, 220],
                       [420, 220],
                       [120, 320],
                       [279, 320],
                       [420, 320],
                       [220, 470]],
                       dtype='float')

def resize_complex_image(image, size):
    abs = tf.image.resize( tf.math.abs(image), size)
    angle = tf.image.resize( tf.math.angle(image), size)
    return tf.complex( abs,0.0) * tf.math.exp(1j * tf.complex(angle,0.0))


def get_scaled_hole_pos( side_length_mm, N = 30 ):
    centers = (centers512/512*N).astype(int)
    centers_mm = centers * side_length_mm / N
    return centers_mm


def create_5el_array_input(Nx=60, Lx=50, r0=7, x0=15):
    # Nx : grid size (# pixels)

    # Lx : grid size (physical length, mm)

    # r0 : transducer radius (mm)

    # x0 : transducer position (mm)

    # define input fields
    YY, XX = np.mgrid[-Lx / 2:Lx / 2:Nx * 1j, -Lx / 2:Lx / 2:Nx * 1j]
    input_fields = np.zeros((5, Nx, Nx), dtype=float)
    transducer_pos = [(0, 0), (x0, x0), (-x0, x0), (-x0, -x0), (x0, -x0)]
    for j, (xc, yc) in enumerate(transducer_pos):
        R = np.sqrt((XX - xc) ** 2 + (YY - yc) ** 2)
        input_fields[j, :, :] = R < r0
    return input_fields

def create_combinatory_array_input( Nx = 60, Lx = 50, r0 = 7, x0 = 15):
    inputs = create_5el_array_input(Nx, Lx, r0, x0)
    combinations = ( [0,1], [0,2], [0,3], [0,4], [1,2],[1,3],[1,4], [2,3],[2,4],[3,4])
    combinatory_inputs = []
    for comb in combinations:
        combinatory_inputs.append(inputs[comb[0]] + inputs[comb[1]])
    return np.array(combinatory_inputs)



def Radial_TEM_pl(rho, phi, l , p):
    I_0 = 1.0
    lag = genlaguerre(p,l)
    res = I_0 * rho**l*( lag(rho))**2 * np.cos(l*phi)**2 * np.exp(-rho)
    return res

def XYZ_TEM_mn( X,Y, w, m,n,  ):
    I_0 = 10.0
    w_0 = 1.0
    H_m = hermite(m)
    H_n = hermite(n)
    res = I_0 * ( w_0/w)**2 * (H_m(np.sqrt(2)*X/w)*np.exp((-X**2)/w**2))**2 * ( H_n(np.sqrt(2)*Y/w)*np.exp((-Y**2)/w**2))**2
    return res


def get_TEM_pl_modes( p,l, w, L, N_img):
    dx = L/N_img
    X,Y = np.mgrid[-L/2:L/2:dx,-L/2:L/2:dx]
    rho = 2*(X**2 + Y**2)**2/w**2 + 0.0000000000000001
    phi = np.arctan(Y/(X+0.00000000000001))
    img = Radial_TEM_pl(rho,phi, l,p)
    return img

def get_TEM_mn_modes(m, n, w, L, N_img):
    dx = L/N_img
    X,Y = np.mgrid[-L/2:L/2:dx, -L/2:L/2:dx]
    img = XYZ_TEM_mn(X,Y, w,m,n )
    return img





def create_input_image(NUM, X, Y, Lx, Nx, ri=2):

    # define input mask as a hole
    ci = get_scaled_hole_pos(Lx,Nx)[NUM]
    #print(ci)
    ci -= Lx/2
    #print(ci)
    input_mask = np.zeros_like(X)
    input_mask[np.sqrt((X-ci[0])**2 + (Y-ci[1])**2) <= ri] = 1

    return np.expand_dims(input_mask,axis = 2)


def create_input_image_v2(Num, X, Y, L, N, ri = None ):
    centers_mm = np.array([[12.5,10],
                           [25,10],
                           [37.5,10],
                           [12.5,20],
                           [25,20],
                           [37.5,20],
                           [12.5,30],
                           [25,30],
                           [37.5,30],
                           [25,40]])
    centers_mm = (centers_mm/50.0)*N
    ci = centers_mm[Num]
    input_mask = np.zeros_like(X)

    if ri == None:
        ri = L/10
    r_n = (ri/L) * N
    input_mask[np.sqrt((X-ci[0])**2 + (Y-ci[1])**2) <= r_n ] = 1
    return np.expand_dims(input_mask, axis = 2)


def create_input_image_v3(Num, X, Y,L):
    #L = 50
    #D0 = 12.7/L
    D0 = 0.254
    #D0 = 0.0127
    #D0 = 12.7/50.0
    dx0 = 0.32
    dx1 = dx0 / 2

    dy0 = 0.46
    dy1 = dy0 / 2

    y0 = 3 * dy1 / 2
    x0 = -dx0
    xy0 = np.array([[x0, y0],
                    [x0 + dx0, y0],
                    [x0 + 2 * dx0, y0],
                    [x0 + dx1, y0 - dy1],
                    [x0 + dx1 + dx0, y0 - dy1],
                    [x0, y0 - dy0],
                    [x0 + dx0, y0 - dy0],
                    [x0 + 2 * dx0, y0 - dy0],
                    [x0 + dx1, y0 - dy0 - dy1],
                    [x0 + dx1 + dx0, y0 - dy0 - dy1]])
    xy0_l = xy0*L
    radius = D0*L/2
    input_mask = np.zeros_like(X)
    ci = xy0_l[Num]
    input_mask[np.sqrt((X-ci[0])**2 + (Y-ci[1])**2) <= radius ] = 1
    return np.expand_dims(input_mask, axis = 2)









def create_circle_image_xy(x,y , L, N, r = 2):
    #x = np.array(x)
    #y = np.array(y)
    ci = np.array((x,y),dtype = 'float64')

    ci -= L/2
    dx = L/N
    Y, X = np.mgrid[0:L:dx, 0:L:dx] - L / 2


    if not (np.isscalar(x) or np.isscalar(y)):
        print(X.shape)
        print(x.shape)
        X = np.tile(X,(x.shape[0],1,1))
        Y = np.tile(Y,(y.shape[0],1,1))
        #print(np.sqrt((X - ci[0]) ** 2))
        input_mask = np.zeros_like(X,dtype = 'float32')
        for i in range(0, x.shape[0]):
            input_mask[i][np.sqrt((X[i] - ci[0,i]) ** 2 + (Y[i] - ci[1,i]) ** 2) <= r] = 1



    else:
        #print(ci.shape)
        #print(X.shape)
        input_mask = np.zeros_like(X,dtype = 'float32')
        #print(Y)
        input_mask[np.sqrt((X-ci[0])**2 + (Y-ci[1])**2) <= r] = 1
        input_mask = np.expand_dims(input_mask,axis = 0)

        #plt.imshow(input_mask)
        #plt.show()
    #print(input_mask.dtype)
    return np.expand_dims(input_mask, axis = 3)


def create_combinatorial_image_dataset(N, num_images = 6):
    def draw_ellipse(draw, middle,size, line_width):

        p1 = middle(-size/2)
        p2 = middle(+size/2)
        ellipse_pos = [p1[0], p0[1], p1[0],p1[1]]
        draw.ellipse(ellipse_pos, fill=(1), outline=(1))

        ellipse_2_pos = [p0[0] + line_width, p0[1] + line_width, p1[0] - line_width,
                         p1[1] - line_width]
        draw.ellipse(ellipse_2_pos, fill=(0), outline=(0))

    def draw_rectangle(draw, middle, size, rotation, line_width):

        topleft = np.array([(-size[0]/2), -size[1]/2])
        bottomleft = np.array([(size[0]/2), -size[1]/2])
        bottomright = np.array([(size[0]/2), size[1]/2])
        topright = np.array([(-size[0]/2), size[1]/2])

        rot_mat = np.array( [[ np.cos(rotation), -np.sin(rotation)],
                             [ np.sin(rotation), np.cos(rotation)]])
        topleft=  np.matmul(rot_mat, topleft)
        bottomleft = np.matmul(rot_mat, bottomleft)
        bottomright = np.matmul(rot_mat, bottomright)
        topright = np.matmul(rot_mat, topright)


        rect_pos = [int(0.2 * N), int(0.2 * N), int(0.8 * N), int(0.8 * N)]
        rect_pos_2 = [int(0.2 * N) + line_width, int(0.2 * N) + line_width, int(0.8 * N) - line_width,
                      int(0.8 * N) - line_width]
        draw.rectangle(rect_pos, fill=(1), outline=(1))
        draw.rectangle(rect_pos_2, fill=(0), outline=(0))

    def draw_cross(draw, middle, x_size, y_size, rotation, line_width):
        line_pos = [int(0.2 * N), int(0.2 * N), int(0.8 * N), int(0.8 * N)]
        draw.line(line_pos, fill=(1), width=line_width)
        line_pos2 = [int(0.2 * N), int(0.8 * N), int(0.8 * N), int(0.2 * N)]
        draw.line(line_pos2, fill=(1), width=line_width)

    def draw_triangle(draw, middle, x_size, y_size, rotation, line_width):
        polygon_pos = [int(0.5 * N), int(0.5 * N), N * 0.4]
        polygon_pos_2 = [int(0.5 * N), int(0.5 * N), N * 0.4 - line_width * 1.5]
        draw.regular_polygon(polygon_pos, 3, rotation=0, fill=(1), outline=(1))
        draw.regular_polygon(polygon_pos_2, 3, rotation=0, fill=(0), outline=(0))




    images = []
    for i in range(0,num_images):
        Img = Image.new('1',(N,N))
        draw = ImageDraw.Draw(Img)

def create_crystal_phase_retrieval_data(num = 50000, N = 128, M = 100):
    '''
    Creates images for the standard phase retrieval problem of crystal structure retrieval
    '''
    #images = np.zeros( (50000, N,N))
    images = []
    for i in range(0,num):
        image = np.zeros( (N*4, N*4))
        positions = []
        while len(positions) < M:
            pos = np.random.randint(0,N*4, 2)
            #print(positions)
            #print(len(positions))
            add = True
            if len(positions) > 1:
                dist = np.sqrt(np.sum((np.array(positions) - pos)**2,1))
                if np.min(dist) < 4.0:
                    add = False
            if add:
                positions.append(pos)
                X, Y = np.mgrid[0:128 * 4, 0:128 * 4]
                image[((X-pos[0]) ** 2 + (Y-pos[1]) ** 2) < 4 ** 2] = np.random.randint(1.0, 4.0)
                #image[(X ** 2 + Y ** 2) < 4 ** 2] = 100
                #print('{}, {}'.format(pos[0],pos[1]))
                #print(((X-pos[0]) ** 2 + (Y-pos[1]) ** 2) < 8 ** 2)
        #print(len(positions))
        #print(np.array(positions))

        images.append(image)
    images = np.array(images)
    fourier_transformed = np.fft.fft2( images)
    real = np.abs(fourier_transformed)
    return images, real




def num_to_image_measured_v3(num, scale = 1, file_path  = 'PhaseplateNetwork/utils/Datasets/input_fields_v3_normalized.npy'):
    '''
    returns the measured complex input images for 10 different digits
    '''
    measured_inputs = np.load(file_path)
    #measured_inputs = np.repeat(measured_inputs, 2, axis = 1)
    #measured_inputs = np.repeat(measured_inputs, 2, axis = 2)


    def zero(): return tf.constant(np.expand_dims(measured_inputs[0,:,:],axis = 2),dtype = tf.complex64)
    def one(): return tf.constant(np.expand_dims(measured_inputs[1,:,:],axis = 2),dtype = tf.complex64)
    def two(): return tf.constant(np.expand_dims(measured_inputs[2,:,:],axis = 2),dtype = tf.complex64)
    def three(): return tf.constant(np.expand_dims(measured_inputs[3,:,:],axis = 2),dtype = tf.complex64)
    def four(): return tf.constant(np.expand_dims(measured_inputs[4,:,:],axis = 2), dtype = tf.complex64)
    def five(): return tf.constant(np.expand_dims(measured_inputs[5,:,:],axis = 2), dtype = tf.complex64)
    def six(): return tf.constant(np.expand_dims(measured_inputs[6,:,:],axis = 2), dtype = tf.complex64)
    def seven(): return tf.constant(np.expand_dims(measured_inputs[7,:,:],axis = 2), dtype = tf.complex64)
    def eight(): return tf.constant(np.expand_dims(measured_inputs[8,:,:],axis = 2), dtype = tf.complex64)
    def nine(): return tf.constant(np.expand_dims(measured_inputs[9,:,:],axis = 2), dtype = tf.complex64)

    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])
    #res = tf.image.resize(res, (res.shape[0]*scale, res.shape[1]*scale), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #print(res.shape)
    return res

def num_to_image_measured_v2(num, scale = 1, file_path  = './measured_mnist_v2.npy'):
    '''
    returns the measured complex input images for 10 different digits
    '''
    measured_inputs = np.load(file_path)
    #measured_inputs = np.repeat(measured_inputs, 2, axis = 1)
    #measured_inputs = np.repeat(measured_inputs, 2, axis = 2)


    def zero(): return tf.constant(np.expand_dims(measured_inputs[0,:,:],axis = 2),dtype = tf.complex64)
    def one(): return tf.constant(np.expand_dims(measured_inputs[1,:,:],axis = 2),dtype = tf.complex64)
    def two(): return tf.constant(np.expand_dims(measured_inputs[2,:,:],axis = 2),dtype = tf.complex64)
    def three(): return tf.constant(np.expand_dims(measured_inputs[3,:,:],axis = 2),dtype = tf.complex64)
    def four(): return tf.constant(np.expand_dims(measured_inputs[4,:,:],axis = 2), dtype = tf.complex64)
    def five(): return tf.constant(np.expand_dims(measured_inputs[5,:,:],axis = 2), dtype = tf.complex64)
    def six(): return tf.constant(np.expand_dims(measured_inputs[6,:,:],axis = 2), dtype = tf.complex64)
    def seven(): return tf.constant(np.expand_dims(measured_inputs[7,:,:],axis = 2), dtype = tf.complex64)
    def eight(): return tf.constant(np.expand_dims(measured_inputs[8,:,:],axis = 2), dtype = tf.complex64)
    def nine(): return tf.constant(np.expand_dims(measured_inputs[9,:,:],axis = 2), dtype = tf.complex64)

    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])
    #res = tf.image.resize(res, (res.shape[0]*scale, res.shape[1]*scale), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #print(res.shape)
    return res

def num_to_image_measured(num, scale = 1, file_path  = './measured_mnist_30px.npy'):
    '''
    returns the measured complex input images for 10 different digits
    '''
    measured_inputs = np.load(file_path)
    measured_inputs = np.repeat(measured_inputs, 2, axis = 1)
    measured_inputs = np.repeat(measured_inputs, 2, axis = 2)


    def zero(): return tf.constant(np.expand_dims(measured_inputs[0,:,:],axis = 2),dtype = tf.complex64)
    def one(): return tf.constant(np.expand_dims(measured_inputs[1,:,:],axis = 2),dtype = tf.complex64)
    def two(): return tf.constant(np.expand_dims(measured_inputs[2,:,:],axis = 2),dtype = tf.complex64)
    def three(): return tf.constant(np.expand_dims(measured_inputs[3,:,:],axis = 2),dtype = tf.complex64)
    def four(): return tf.constant(np.expand_dims(measured_inputs[4,:,:],axis = 2), dtype = tf.complex64)
    def five(): return tf.constant(np.expand_dims(measured_inputs[5,:,:],axis = 2), dtype = tf.complex64)
    def six(): return tf.constant(np.expand_dims(measured_inputs[6,:,:],axis = 2), dtype = tf.complex64)
    def seven(): return tf.constant(np.expand_dims(measured_inputs[7,:,:],axis = 2), dtype = tf.complex64)
    def eight(): return tf.constant(np.expand_dims(measured_inputs[8,:,:],axis = 2), dtype = tf.complex64)
    def nine(): return tf.constant(np.expand_dims(measured_inputs[9,:,:],axis = 2), dtype = tf.complex64)

    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])
    #res = tf.image.resize(res, (res.shape[0]*scale, res.shape[1]*scale), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #print(res.shape)
    return res



def num_to_image_tensorflow( num, images):

    #def zero(): return tf.constant(result_0,dtype = tf.float32)
    #def one(): return tf.constant(result_1,dtype = tf.float32)
    #def two(): return tf.constant(result_2,dtype = tf.float32)
    #def three(): return tf.constant(result_3,dtype = tf.float32)
    #def four(): return tf.constant(result_4, dtype = tf.float32)
    #def five(): return tf.constant(result_5, dtype = tf.float32)
    #def six(): return tf.constant(result_6, dtype = tf.float32)
    #def seven(): return tf.constant(result_7, dtype = tf.float32)
    #def eight(): return tf.constant(result_8, dtype = tf.float32)
    #def nine(): return tf.constant(result_9, dtype = tf.float32)
    cases = []
    for i in range(0, images.shape[0]):
        cases.append( (tf.math.equal(num, i), lambda : images[i,:,:,:]) )
    #print(cases)
    res = tf.case(cases)
    #res = tf.case( [ ( tf.math.equal(num,0), zero),
    #                 ( tf.math.equal(num, 1), one),
    #                 ( tf.math.equal(num, 2), two),
    #                 ( tf.math.equal(num, 3), three),
    #                 ( tf.math.equal(num, 4), four),
    #                 ( tf.math.equal(num, 5), five),
    #                 ( tf.math.equal(num, 6), six),
    #                 ( tf.math.equal(num, 7), seven),
    #                 ( tf.math.equal(num, 8), eight),
    #                 ( tf.math.equal(num, 9), nine),
    #                   ])

    return res


def num_to_image_thanasi_v2( num, L, N, radius = 4):
    '''
    :param num: number from 0 to 9
    :param L: size of the propagation plate in mm
    :param N: number of pixels on the propagation plate (already upscaled)
    :param radius: radius of the holes in mm
    :return: an image of the input plate with an image at a certain position
    '''
    dx = L/N
    Y, X = np.mgrid[0:L:dx, 0:L:dx] - L / 2
    result_0 = create_input_image_v2(0,X,Y, L, N, radius)
    result_1 = create_input_image_v2(1,X,Y, L, N, radius)
    result_2 = create_input_image_v2(2,X,Y, L, N, radius)
    result_3 = create_input_image_v2(3,X,Y, L, N, radius)
    result_4 = create_input_image_v2(4,X,Y, L, N, radius)
    result_5 = create_input_image_v2(5,X,Y, L, N, radius)
    result_6 = create_input_image_v2(6,X,Y, L, N, radius)
    result_7 = create_input_image_v2(7,X,Y, L, N, radius)
    result_8 = create_input_image_v2(8,X,Y, L, N, radius)
    result_9 = create_input_image_v2(9,X,Y, L, N, radius)

    def zero(): return tf.constant(result_0,dtype = tf.float32)
    def one(): return tf.constant(result_1,dtype = tf.float32)
    def two(): return tf.constant(result_2,dtype = tf.float32)
    def three(): return tf.constant(result_3,dtype = tf.float32)
    def four(): return tf.constant(result_4, dtype = tf.float32)
    def five(): return tf.constant(result_5, dtype = tf.float32)
    def six(): return tf.constant(result_6, dtype = tf.float32)
    def seven(): return tf.constant(result_7, dtype = tf.float32)
    def eight(): return tf.constant(result_8, dtype = tf.float32)
    def nine(): return tf.constant(result_9, dtype = tf.float32)

    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])

    return res

def num_to_image_thanasi_v3( num, L, N,radius = 0.05*0.127):
    '''
    :param num: number from 0 to 9
    :param L: size of the propagation plate in mm
    :param N: number of pixels on the propagation plate (already upscaled)
    :param radius: radius of the holes in mm
    :return: an image of the input plate with an image at a certain position
    '''
    dx = L/N
    Y, X = np.mgrid[0:L:dx, 0:L:dx] - L / 2
    result_0 = create_input_image_v3(0,X,Y, L)
    result_1 = create_input_image_v3(1,X,Y, L)
    result_2 = create_input_image_v3(2,X,Y, L)
    result_3 = create_input_image_v3(3,X,Y, L)
    result_4 = create_input_image_v3(4,X,Y, L)
    result_5 = create_input_image_v3(5,X,Y, L)
    result_6 = create_input_image_v3(6,X,Y, L)
    result_7 = create_input_image_v3(7,X,Y, L)
    result_8 = create_input_image_v3(8,X,Y, L)
    result_9 = create_input_image_v3(9,X,Y, L)

    def zero(): return tf.constant(result_0,dtype = tf.float32)
    def one(): return tf.constant(result_1,dtype = tf.float32)
    def two(): return tf.constant(result_2,dtype = tf.float32)
    def three(): return tf.constant(result_3,dtype = tf.float32)
    def four(): return tf.constant(result_4, dtype = tf.float32)
    def five(): return tf.constant(result_5, dtype = tf.float32)
    def six(): return tf.constant(result_6, dtype = tf.float32)
    def seven(): return tf.constant(result_7, dtype = tf.float32)
    def eight(): return tf.constant(result_8, dtype = tf.float32)
    def nine(): return tf.constant(result_9, dtype = tf.float32)

    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])

    return res

def num_to_image_thanasi( num, L, N, radius = 4):
    '''
    :param num: number from 0 to 9
    :param L: size of the propagation plate in mm
    :param N: number of pixels on the propagation plate (already upscaled)
    :param radius: radius of the holes in mm
    :return: an image of the input plate with an image at a certain position
    '''
    dx = L/N
    Y, X = np.mgrid[0:L:dx, 0:L:dx] - L / 2
    result_0 = create_input_image(0,X,Y, L, N, radius)
    result_1 = create_input_image(1,X,Y, L, N, radius)
    result_2 = create_input_image(2,X,Y, L, N, radius)
    result_3 = create_input_image(3,X,Y, L, N, radius)
    result_4 = create_input_image(4,X,Y, L, N, radius)
    result_5 = create_input_image(5,X,Y, L, N, radius)
    result_6 = create_input_image(6,X,Y, L, N, radius)
    result_7 = create_input_image(7,X,Y, L, N, radius)
    result_8 = create_input_image(8,X,Y, L, N, radius)
    result_9 = create_input_image(9,X,Y, L, N, radius)

    def zero(): return tf.constant(result_0,dtype = tf.float32)
    def one(): return tf.constant(result_1,dtype = tf.float32)
    def two(): return tf.constant(result_2,dtype = tf.float32)
    def three(): return tf.constant(result_3,dtype = tf.float32)
    def four(): return tf.constant(result_4, dtype = tf.float32)
    def five(): return tf.constant(result_5, dtype = tf.float32)
    def six(): return tf.constant(result_6, dtype = tf.float32)
    def seven(): return tf.constant(result_7, dtype = tf.float32)
    def eight(): return tf.constant(result_8, dtype = tf.float32)
    def nine(): return tf.constant(result_9, dtype = tf.float32)

    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])

    return res


def num_to_image(num, N = 512, radius = 10):

    #print(tf.equal(num,5).numpy())
    #if num == 0:
    y,x = np.ogrid[-radius:radius,-radius:radius]
    ind = x**2 + y**2 < radius**2

    result_0 = np.zeros((N, N,1))
    cx, cy = int(120/512* N), int(90/512 * N)
    print(np.shape(result_0[cy-radius:cy+radius, cx-radius: cx + radius,0]))
    result_0[cy-radius:cy+radius, cx-radius: cx + radius,0][ind] = 1.0
    #result_0[100:140, 70:110,0] = np.ones((40, 40))

    result_1 = np.zeros((N, N,1))
    #result_1[250:290, 70:110,0] = np.ones((40, 40))
    cx, cy = int(270/512 * N), int(90/512 * N)
    result_1[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_2 = np.zeros((N, N,1))
    #result_2[400:440, 70:110,0] = np.ones((40, 40))
    cx, cy = int(420/512 * N), int(90/512 * N)
    result_2[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_3 = np.zeros((N, N,1))
    #result_3[100:140, 200:240,0] = np.ones((40, 40))
    cx, cy = int(120/512 * N), int(220/512 * N)
    result_3[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_4 = np.zeros((N, N,1))
    #result_4[250:290, 200:240,0] = np.ones((40, 40))
    cx, cy = int(270/512 * N), int(220/512 * N)
    result_4[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_5 = np.zeros((N, N,1))
    #result_5[400:440, 200:240,0] = np.ones((40, 40))
    cx, cy = int(420/512 *N), int(220/512 * N)
    result_5[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_6 = np.zeros((N, N,1))
    #result_6[100:140, 300:340,0] = np.ones((40, 40))
    cx, cy = int(120/512 * N), int(320/512 * N)
    result_6[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_7 = np.zeros((N, N,1))
    #result_7[250:290, 300:340,0] = np.ones((40, 40))
    cx, cy = int(279/512 * N), int(320/512 * N)
    result_7[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_8 = np.zeros((N, N,1))
    #result_8[400:440, 300:340,0] = np.ones((40, 40))
    cx, cy = int(420/512 * N), int(320/512 * N)
    result_8[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    result_9 = np.zeros((N, N,1))
    #result_9[200:240, 450:490,0] = np.ones((40, 40))
    cx, cy = int(220/512 * N), int(470/512 * N)
    result_9[cy - radius:cy + radius, cx - radius: cx + radius,0][ind] = 1.0

    def zero(): return tf.constant(result_0,dtype = tf.float32)
    def one(): return tf.constant(result_1,dtype = tf.float32)
    def two(): return tf.constant(result_2,dtype = tf.float32)
    def three(): return tf.constant(result_3,dtype = tf.float32)
    def four(): return tf.constant(result_4, dtype = tf.float32)
    def five(): return tf.constant(result_5, dtype = tf.float32)
    def six(): return tf.constant(result_6, dtype = tf.float32)
    def seven(): return tf.constant(result_7, dtype = tf.float32)
    def eight(): return tf.constant(result_8, dtype = tf.float32)
    def nine(): return tf.constant(result_9, dtype = tf.float32)

    #res = tf.case( { tf.math.equal(num, 0): zero,
    #                 tf.math.equal(num, 1): one,
    #                 tf.math.equal(num, 2): two,
    #                tf.math.equal(num, 3): three,
    #                 tf.math.equal(num, 4): four,
    #                 tf.math.equal(num, 5): five,
    #                 tf.math.equal(num, 6): six,
    #                 tf.math.equal(num, 7): seven,
    #                 tf.math.equal(num, 8): eight,
    #                 tf.math.equal(num, 9): nine,
    #                 })
    res = tf.case( [ ( tf.math.equal(num,0), zero),
                     ( tf.math.equal(num, 1), one),
                     ( tf.math.equal(num, 2), two),
                     ( tf.math.equal(num, 3), three),
                     ( tf.math.equal(num, 4), four),
                     ( tf.math.equal(num, 5), five),
                     ( tf.math.equal(num, 6), six),
                     ( tf.math.equal(num, 7), seven),
                     ( tf.math.equal(num, 8), eight),
                     ( tf.math.equal(num, 9), nine),
                       ])
    return res

def get_scaled_mnist_images(img, resolution = 512, scale = 5):
    mnist_size = 28*scale
    img = tf.reshape(img, (img.shape[0], img.shape[1], 1))
    img = tf.image.resize_with_crop_or_pad (tf.image.resize(img, (mnist_size,mnist_size)), resolution,resolution)/255.0
    return img

def get_scaled_mnist_images_3D(img, resolution = 512, scale =5):
    mnist_size = 28*scale
    #img = tf.reshape(img, ())
    img = tf.image.resize_with_crop_or_pad(tf.image.resize(img, (mnist_size, mnist_size)), resolution,
                                           resolution) / 255.0
    return img

def get_activation_images_TEM_mn(inp, image_size, L, radius):
    inp_num = inp.shape[1]
    per_row = np.ceil(np.sqrt(inp_num))
    dx = L/image_size
    Y,X = np.mgrid[0:L:dx, 0:L:dx]
    image = np.zeros_like(X)
    image = np.tile(image,(inp.shape[0],1,1))
    L_TEM = 15.0
    one = get_TEM_mn_modes(0,2, 0.2*L_TEM, L_TEM, L_TEM/dx )
    zero = get_TEM_mn_modes(2,0, 0.2*L_TEM, L_TEM, L_TEM/dx)
    for j in range(0, inp.shape[0]):
        for i in range(0,inp_num):
            cx, cy = ((i % per_row) * L * 0.9 / per_row + L / (per_row * 2)), (
                    (i // per_row) * L * 0.9 / per_row + L / (per_row * 2))
            #print(one.shape)
            print(one.shape)
            print(image[:,int(cx)-int(one.shape[0]/(2)):int(cx) + int(one.shape[0]/(2)), int(cy)-int(one.shape[1]/(2)):int(cy) + int(one.shape[1]/(2))].shape)
            print(cx,cy)
            print(i)
            cx = cx /dx
            cy = cy /dx
            image[j,int(cy)-int(one.shape[1]/(2)):int(cy) + int(one.shape[1]/(2)), int(cx)-int(one.shape[0]/(2)):int(cx) + int(one.shape[0]/(2))] = one* inp[j,i] + zero*(np.abs(1-inp[j,i]))
    return image

def get_square_activation_images_jones_prop(inp, image_size, L):
    inp_num = inp.shape[1]
    assert(inp_num%2 == 0)
    inp_num = inp_num//2
    img_x = get_square_activation_images(inp[:,0:inp_num],image_size,L)
    img_y = get_square_activation_images(inp[:,inp_num:],image_size,L)
    print(img_x.shape)
    print(img_y.shape)
    imgs = np.stack((img_x,img_y), axis = 3)
    return imgs

def get_square_activation_images(inp,image_size, L ):
    inp_num = inp.shape[1]
    per_row = np.ceil(np.sqrt(inp_num))
    print(per_row)
    square_size = np.floor(image_size//per_row).astype('int')
    print(square_size)

    dx = L/image_size
    Y,X = np.mgrid[0:L:dx, 0:L:dx]
    image = np.zeros_like(X)
    image = np.tile(image,(inp.shape[0],1,1))
    for j in range(0,inp.shape[0]):
        for i in range(0,inp_num):
            #cx, cy = ((i % per_row) * L * 0.9 / per_row + L / (per_row * 2)), (
            #    (i // per_row) * L * 0.9 / per_row + L / (per_row * 2))
            #print((i%per_row)*square_size)
            #print(((i+1)%per_row)*square_size)
            iy, ix= int((i%per_row)*square_size), int((i//per_row)*square_size)

            #ix2, iy2 = int( ((i+1)%per_row)*square_size), int(((i+1)//per_row)*square_size)
            #print(ix)
            #print(ix2)
            #print(iy)
            #print(iy2)
            image[j, ix: ix+square_size, iy: iy+square_size] = inp[j,i]

            #image[j,np.sqrt((X-cx)**2 + (Y - cy)**2) <= radius] =inp[j,i]

    return image

#def get_full_square_activation_images(inp, image_size, L, radius):
#    inp_num = inp.shape[1]
#    per_row = np.ceil(np.sqrt(inp_num))
#    square_size = np.floor()


def get_activation_images(inp, image_size,L, radius, type = 'rows'):
    #image = np.zeros((inp.shape[0], image_size,image_size))
    inp_num = inp.shape[1]
    per_row = np.ceil(np.sqrt(inp_num))

    dx = L/image_size
    #if len(np.arange(0,L,dx)) != image_size:
    Y, X = np.mgrid[0:L-(1e-15):dx, 0:L-(1e-15):dx]# - L / 2
    #print(X.shape)
    image = np.zeros_like(X)
    image = np.tile(image,(inp.shape[0],1,1))

    for j in range(0,inp.shape[0]):
        for i in range(0,inp_num):
            if type == 'rows':
                cx, cy = ((i % per_row) * L * 0.9 / per_row + L / (per_row * 2)), (
                    (i // per_row) * L * 0.9 / per_row + L / (per_row * 2))
            elif type == 'circle':
                cx = L / 2.0 + np.math.cos(2 * np.pi * float(i) / inp_num) * circ_rad
                cy = L / 2.0 - np.math.sin(2 * np.pi * float(i) / inp_num) * circ_rad


            image[j,np.sqrt((X-cx)**2 + (Y - cy)**2) <= radius] =inp[j,i]

    return image

def get_measured_activation_images(inp,  image_size, L):
    inp_num = inp.shape[1]
    if inp_num > 10:
        raise SystemExit('Can not create measured dataset with more than 10 inputs. Input nums: {}'.format(inp_num))
    dx = L/image_size

    #Y,X = np.mgrid[0:image_size, 0:image_size]
    image = np.zeros((image_size,image_size), 'complex64')
    image = np.tile(image,(inp.shape[0],1,1))

    image_dict = { 0: 3, 1:4, 2:6, 3:8}
    for j in range(0, inp.shape[0]):
        for i in range(0,inp_num):

            image[j,:] = image[j,:] + resize_complex_image(num_to_image_measured_v2(image_dict[i],1), (image_size,image_size))[:,:,0] * inp[j,i]


    return image

def get_measured_activation_images_v2(inp, image_size, L):
    inp_num = inp.shape[1]
    if inp_num > 10:
        raise SystemExit('Can not create measured dataset with more than 10 inputs. Input nums: {}'.format(inp_num))
    dx = L / image_size

    # Y,X = np.mgrid[0:image_size, 0:image_size]
    image = np.zeros((image_size, image_size), 'complex64')
    image = np.tile(image, (inp.shape[0], 1, 1))

    image_dict = {0: 0, 1: 2, 2: 6, 3: 8}
    for j in range(0, inp.shape[0]):
        for i in range(0, inp_num):
            image[j, :] = image[j, :] + resize_complex_image(num_to_image_measured_v3(image_dict[i], 1),
                                                             (image_size, image_size))[:, :, 0] * inp[j, i]

    return image




def get_circular_activation_images(inp, image_size, L, radius):
    inp_num = inp.shape[1]
    dx = L/image_size
    Y, X = np.mgrid[0:L-(1e-15):dx, 0:L-(1e-15):dx]# - L / 2
    #print(X.shape)
    image = np.zeros_like(X)
    image = np.tile(image,(inp.shape[0],1,1))

    circ_rad = L/3.0

    for j in range(0,inp.shape[0]):
        for i in range(0,inp_num):
            #cx, cy = ((i % per_row) * L * 0.9 / per_row + L / (per_row * 2)), (
            #    (i // per_row) * L * 0.9 / per_row + L / (per_row * 2))
            cx = L/2.0 + np.math.cos(2*np.pi * float(i)/inp_num)*circ_rad
            cy = L/2.0 - np.math.sin(2*np.pi * float(i)/inp_num)*circ_rad

            image[j,np.sqrt((X-cx)**2 + (Y - cy)**2) <= radius] =inp[j,i]

    return image

def get_activation_images_from_number(N, image_size,L, radius, type = 'line'):
    ar = np.zeros( (N,N), dtype = 'float32')
    for i in range(0,N):
        ar[i,i] = 1.0
    if type == 'circle':
        imgs = np.expand_dims(get_circular_activation_images(ar, image_size,L, radius), axis = 3).astype('float32')
    else:
        imgs = np.expand_dims(get_activation_images(ar, image_size, L, radius), axis=3).astype('float32')
    return imgs



def weave_arrays(a,b):
    c = np.empty((a.shape[0],a.shape[1] + b.shape[1]), dtype = a.dtype)
    c[:,0::2] = a
    c[:,1::2] = b
    return c

def get_logical_dataset(input_num, logical_function, double = False):
    inp = np.zeros((2**input_num,input_num))
    for i in range(0, 2**input_num):
        for j in range(0,input_num):
            inp[i,input_num - j -1 ] = 1 & i >> j
    out = logical_function(inp)
    inverse_inp = np.abs(inp -1)
    print(inverse_inp.shape, inp.shape)
    inverse_out = np.abs(out - 1)
    if double:
        inp = weave_arrays(inp,inverse_inp)
        out = weave_arrays(out,inverse_out)
        #inp = np.concatenate((inp, inverse_inp),1)
        #out = np.concatenate((out, inverse_out),1)
    return inp, out
    #input_images = get_activation_images(inp, 30,2)


def get_SR_latch_dataset(batch_num, image_size, L):
    input_logical, output_logical = get_logical_dataset(4, SR_latch, True)
    output_logical_inv = np.abs(output_logical-1)
    output_logical = np.concatenate((output_logical, output_logical_inv), axis = 1)
    print(input_logical)
    print(output_logical)
    input_images = get_square_activation_images_jones_prop(input_logical, image_size, L )
    output_images = get_square_activation_images_jones_prop(output_logical, image_size, L )
    input_images = np.reshape(input_images,
                              (input_images.shape[0], input_images.shape[1], input_images.shape[2], 2)).astype(
        'float32')
    output_images = np.reshape(output_images,
                               (output_images.shape[0], output_images.shape[1], output_images.shape[2], 2)).astype(
        'float32')
    train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images)
    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images)
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_output_dataset)).shuffle(16).batch(batch_num).repeat(30000)
    return train_dataset


def get_logic_dataset(batch_num, input_num, logical_function, image_size,L ,radius = 4, input_type = 'circles', double = False):
    input_logical, output_logical = get_logical_dataset(input_num, logical_function, double)
    if input_type == 'circles':
        input_images = get_activation_images(input_logical, image_size,L, radius)
        output_images = get_activation_images(output_logical, image_size,L, radius)
    elif input_type == 'TEM_mn':
        input_images = get_activation_images_TEM_mn(input_logical, image_size,L,radius)
        output_images = get_activation_images_TEM_mn(output_logical,image_size, L,  radius)
    elif input_type == 'squares':
        input_images = get_square_activation_images(input_logical,image_size,L)
        output_images = get_square_activation_images(output_logical, image_size,L)

    input_images = np.reshape(input_images,
                              (input_images.shape[0], input_images.shape[1], input_images.shape[2], 1)).astype(
        'float32')
    output_images = np.reshape(output_images,
                               (output_images.shape[0], output_images.shape[1], output_images.shape[2], 1)).astype(
        'float32')
    # diff = image_size*(1- L_plate/L)
    #diff = image_size - (trainable_pixel * scale)
    #if image_size > trainable_pixel:
    #    train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
    #        lambda x: tf.image.pad_to_bounding_box(x, int(diff / 2), int(diff / 2), image_size, image_size))
    #    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
    #        lambda x: tf.image.pad_to_bounding_box(x, int(diff / 2), int(diff / 2), image_size, image_size))
    #else:
    #    train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
    #        lambda x: tf.image.resize(x, (image_size, image_size)))
    #    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
    #        lambda x: tf.image.resize(x, (image_size, image_size)))
    train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images)
    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images)
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_output_dataset)).shuffle(2**input_num).batch(batch_num)
    return train_dataset

def get_spiral_dataset(batch_num, image_size = (28,28), L =0.00175, result_detector_radius = 0.000175 ):
    def get_scaled_dataset_from_categorical(data, num_cat, image_size, L , result_detector_radius ):
        ac_imgs = get_activation_images_from_number(num_cat, image_size[0], L, result_detector_radius)
        dataset = tf.data.Dataset.from_tensor_slices(data).map(
            lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs), image_size)
        )
        return dataset

    dir = os.path.dirname(os.path.abspath(__file__))
    spiral_data = h5py.File(dir+'/Datasets/spiraldata.h5','r')
    data = spiral_data['data']
    classes = spiral_data['classes']

    #data = data
    train_input = data[0:5000,:,:]/255.0
    test_input = data[5000:6000,:,:]/255.0

    train_output = classes[0,0:5000]
    test_output = classes[0,5000:6000]
    #tf.reshape(img, (img.shape[0], img.shape[1], 1))
    train_input_dataset = tf.data.Dataset.from_tensor_slices(train_input).map(
        lambda x: tf.image.resize(tf.expand_dims(x,axis = 2), image_size)
    )
    test_input_dataset = tf.data.Dataset.from_tensor_slices(test_input).map(
        lambda x: tf.image.resize(tf.expand_dims(x,axis = 2),image_size)
    )
    train_output_dataset = get_scaled_dataset_from_categorical(train_output, 4, image_size, L, result_detector_radius)
    test_output_dataset = get_scaled_dataset_from_categorical(test_output, 4, image_size, L, result_detector_radius)

    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_output_dataset)).shuffle(1000).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_output_dataset)).shuffle(1000).batch(batch_num)
    return train_dataset, test_dataset




def get_kMNIST_dataset(batch_num, image_size, L , Trainable_Pixel, scale = 1, mnist_scale = 1, result_detector_radius = 4, train_on_mse = True):
    data_train = tfds.as_numpy(tfds.load(
        'kmnist',
        split='train',
        batch_size=-1,
    ))
    data_test = tfds.as_numpy(tfds.load(
        'kmnist',
        split='test',
        batch_size=-1,
    ))
    x_train, y_train = data_train['image'], data_train['label']
    x_test, y_test = data_test['image'], data_test['label']

    #(x_train, y_train), (x_test,y_test) = mnist.load_data()

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        '''
        if image_size > Trainable_Pixel:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        '''
        ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)
        if image_size > Trainable_Pixel:
            #ac_imgs = get_activation_images_from_number(out.shape[1], N, L, 4)
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_tensorflow(x, ac_imgs),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size)))
    else:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_train))


    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_num_dataset)).shuffle(1000).batch(batch_num)
    #train_dataset_vec = tf.data.Dataset.zip(( train_img_dataset, train_num_vec_dataset)).

    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        '''
        if image_size > Trainable_Pixel:
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        '''
        ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)
        if image_size > Trainable_Pixel:
            #ac_imgs = get_activation_images_from_number(out.shape[1], N, L, 4)
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_tensorflow(x, ac_imgs),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size)))
    else:
        test_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_test))

    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_num_dataset)).shuffle(1000).batch(batch_num)

    return train_dataset, test_dataset



def get_MNIST_dataset(batch_num, image_size, L , Trainable_Pixel, scale = 1, mnist_scale = 1, result_detector_radius = 4, train_on_mse = True):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test,y_test) = mnist.load_data()

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        '''
        if image_size > Trainable_Pixel:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        '''
        ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)
        if image_size > Trainable_Pixel:
            #ac_imgs = get_activation_images_from_number(out.shape[1], N, L, 4)
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_tensorflow(x, ac_imgs),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size)))
    else:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_train))


    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_num_dataset)).shuffle(1000).batch(batch_num)
    #train_dataset_vec = tf.data.Dataset.zip(( train_img_dataset, train_num_vec_dataset)).

    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        '''
        if image_size > Trainable_Pixel:
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        '''
        ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)
        if image_size > Trainable_Pixel:
            #ac_imgs = get_activation_images_from_number(out.shape[1], N, L, 4)
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_tensorflow(x, ac_imgs),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size)))
    else:
        test_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_test))

    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_num_dataset)).shuffle(1000).batch(batch_num,drop_remainder= True)

    return train_dataset, test_dataset


def get_cifar10_dataset(batch_num, image_size, L, pic_scale = 1, result_detector_radius = 4, train_on_mse = True):
    (x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

    min_sum = np.min(np.sum(np.mean(x_train,axis = 3), axis = (1,2) ))
    #max_sum = np.max(np.sum(np.mean(x_train,axis = 3), axis = (1,2)))



    def get_scaled_images(img, resolution=512, scale=5):
        mnist_size = int(32 * scale)
        img = tf.reshape(img, (img.shape[0], img.shape[1], 1))
        img = tf.image.resize_with_crop_or_pad(tf.image.resize(img, (mnist_size, mnist_size)), resolution,
                                               resolution) / 255.0
        img = img * min_sum/tf.reduce_sum(img)
        return img

    def normalize_output(img):
        return img * min_sum/tf.reduce_sum(img)*0.1


    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_images(tf.math.reduce_mean(x,axis = 2), image_size, pic_scale ))

    #diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        ac_imgs = get_activation_images_from_number(10, image_size, L, result_detector_radius, type ='circle')
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: normalize_output(tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size))))
    else:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_train))


    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_num_dataset)).shuffle(1).batch(batch_num)
    #train_dataset_vec = tf.data.Dataset.zip(( train_img_dataset, train_num_vec_dataset)).

    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
        lambda x: get_scaled_images(tf.math.reduce_mean(x,axis = 2), image_size, pic_scale ))

    #diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        ac_imgs = get_activation_images_from_number(10, image_size, L, result_detector_radius,type ='circle')
        test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: normalize_output(tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size))))
    else:
        test_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_test))

    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_num_dataset)).shuffle(1).batch(batch_num,drop_remainder= True)

    return train_dataset, test_dataset


def get_fashion_MNIST_dataset(batch_num, image_size, L , Trainable_Pixel, scale = 1, mnist_scale = 1, result_detector_radius = 4, train_on_mse = True):
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test,y_test) = mnist.load_data()

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        '''
        if image_size > Trainable_Pixel:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        '''
        ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)
        if image_size > Trainable_Pixel:
            #ac_imgs = get_activation_images_from_number(out.shape[1], N, L, 4)
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_tensorflow(x, ac_imgs),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train).map(
                lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size)))
    else:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_train))


    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_num_dataset)).shuffle(1).batch(batch_num)
    #train_dataset_vec = tf.data.Dataset.zip(( train_img_dataset, train_num_vec_dataset)).

    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Trainable_Pixel * scale

    if train_on_mse == True:
        '''
        if image_size > Trainable_Pixel:
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        '''
        ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)
        if image_size > Trainable_Pixel:
            #ac_imgs = get_activation_images_from_number(out.shape[1], N, L, 4)
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_tensorflow(x, ac_imgs),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        else:
            # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
            test_num_dataset = tf.data.Dataset.from_tensor_slices(y_test).map(
                lambda x: tf.image.resize(num_to_image_tensorflow(x, ac_imgs),
                                          (image_size, image_size)))
    else:
        test_num_dataset = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(y_test))

    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_num_dataset)).shuffle(1).batch(batch_num)

    return train_dataset, test_dataset

##########################################
def create_denoising_diffusion_dataset(train_dataset, test_dataset= None, noisy_output = False, batch_num = 16, timesteps = 100, beta_start = 0.001, beta_end = 0.01):
    def get_noisy_img(image, alpha_bar, noise = None):
        if noise == None:
            noise = tf.random.normal(image.shape)
        img = image*tf.math.sqrt(alpha_bar) + noise*(1-alpha_bar)# + tf.math.sqrt(alpha_bar[t])*image
        return img

    def get_noisy_dataset_fn(inp,alphas_bar):
        noise = tf.random.normal(shape= inp.shape)
        t = tf.random.uniform(shape = [1], minval = 1, maxval = timesteps, dtype = tf.int32)
        alphas = tf.reshape(tf.gather(alphas_bar,t),(1,1,1))
        alphas_wanted = tf.reshape( tf.gather(alphas_bar, t-1), (1, 1,1))
        noisy_img = get_noisy_img(inp, alphas,noise)
        wanted_img = get_noisy_img(inp, alphas_wanted, noise)
        return ((noisy_img, t), (wanted_img))

    def get_noisy_dataset_fn_noisy_output(inp, alphas_bar):
        noise = tf.random.normal(shape= inp.shape)
        t = tf.random.uniform(shape = [1], minval = 1, maxval = timesteps, dtype = tf.int32)
        alphas = tf.reshape(tf.gather(alphas_bar,t),(1,1,1))
        alphas_wanted = tf.reshape( tf.gather(alphas_bar, t-1), (1, 1,1))
        noisy_img = get_noisy_img(inp, alphas,noise)
        wanted_img = get_noisy_img(inp, alphas_wanted, noise)
        return ((noisy_img, t), (noise))
    

    def linear_beta_schedule(timesteps, beta_start=0.001, beta_end=0.01):
        return tf.linspace(beta_start, beta_end, timesteps)
    

    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    alphas = 1- betas
    alphas_bar = tf.math.cumprod(alphas, axis = 0)

    if noisy_output:
        preprop_fn = lambda x: get_noisy_dataset_fn_noisy_output(x,alphas_bar)
    else:
        preprop_fn = lambda x: get_noisy_dataset_fn(x, alphas_bar)
            
    
    noisy_train_dataset = train_dataset.map(preprop_fn).shuffle(1).batch(batch_num)
    if test_dataset != None:

        noisy_test_dataset = test_dataset.map(preprop_fn).shuffle(1).batch(batch_num)
        return (noisy_train_dataset, noisy_test_dataset)
    return noisy_train_dataset



def get_mnist_images(image_size = 30, scale = 1, get_test_set = False):
    mnist = tf.keras.datasets.mnist 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
    lambda x: get_scaled_mnist_images(x, image_size, scale))

    if get_test_set:
        test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
             lambda x: get_scaled_mnist_images(x, image_size, scale))   
        return train_img_dataset, test_img_dataset

    return train_img_dataset

def get_kmnist_images(image_size = 30, scale = 1, get_test_set = False):
    #kmnist = tf.keras.datasets.kmnist
    (x_train, y_train), (x_test,y_test) = kmnist.load_data()
    data_train = tfds.as_numpy(tfds.load(
            'kmnist',
            split='train',
            batch_size=-1,
        ))
    data_test = tfds.as_numpy(tfds.load(
            'kmnist',
            split='test',
            batch_size=-1,
        ))
    x_train, y_train = data_train['image'][:,:,:,0], data_train['label']
    x_test, y_test = data_test['image'][:,:,:,0], data_test['label']

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_mnist_images(x, image_size, scale))

    if get_test_set:
        test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
             lambda x: get_scaled_mnist_images(x, image_size, scale))   
        return train_img_dataset, test_img_dataset

    return train_img_dataset

def get_fashion_mnist_images(image_size = 30, scale = 1, get_test_set = False):
    f_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test,y_test) = f_mnist.load_data()

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_mnist_images(x, image_size, scale))
    
    if get_test_set:
        test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
             lambda x: get_scaled_mnist_images(x, image_size, scale))   
        return train_img_dataset, test_img_dataset

    return train_img_dataset

def get_cifar10_images(image_size = 30, scale = 1, get_test_set = False):
    (x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

    min_sum = np.min(np.sum(np.mean(x_train,axis = 3), axis = (1,2) ))

    def get_scaled_images(img, resolution=512, scale=5):
        mnist_size = 32 * scale
        img = tf.reshape(img, (img.shape[0], img.shape[1], 1))
        img = tf.image.resize_with_crop_or_pad(tf.image.resize(img, (mnist_size, mnist_size)), resolution,
                                               resolution) / 255.0
        img = img * min_sum/tf.reduce_sum(img)
        return img

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train).map(
        lambda x: get_scaled_images(tf.math.reduce_mean(x,axis = 2), image_size, scale ))

    if get_test_set:
        test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test).map(
            lambda x: get_scaled_images(tf.math.reduce_mean(x,axis = 2), image_size, scale ))
        return train_img_dataset, test_img_dataset

    return train_img_dataset





def get_inverse_MNIST_dataset(images, image_size, L, Trainable_Pixel, scale=1, mnist_scale=1, result_detector_radius = 4):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_images = []
    y_train_categories = []
    i = 0
    assert(images <= 10)
    while len(x_train_images) < images:

        if not (y_train[i] in y_train_categories):
            x_train_images.append(x_train[i])
            y_train_categories.append(y_train[i])
        i = i + 1


    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_images).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))
    diff = image_size - Trainable_Pixel * scale

    ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)


    if image_size > Trainable_Pixel:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: tf.image.pad_to_bounding_box(num_to_image_tensorflow(x, ac_imgs),
                                                   int(diff / 2), int(diff / 2), image_size, image_size))
    else:
        #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image_tensorflow(x,ac_imgs), (image_size, image_size) ))


    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).repeat(10000).shuffle(1).batch(images)
    test_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).shuffle(1).batch(images)
    return train_dataset, test_dataset

def get_inverse_MNIST_thanasi_dataset(images, image_size, L, Trainable_Pixel, scale=1, mnist_scale=1, result_detector_radius = 4, version = 0):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_images = []
    y_train_categories = []
    i = 0
    while len(x_train_images) < images:

        if not (y_train[i] in y_train_categories):
            x_train_images.append(x_train[i])
            y_train_categories.append(y_train[i])
        i = i + 1

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_images).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))
    diff = image_size - Trainable_Pixel * scale
    #ac_imgs = get_activation_images_from_number(10, Trainable_Pixel * scale, L, result_detector_radius)

    if image_size > Trainable_Pixel:
        if version == 0:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
                lambda x: tf.image.pad_to_bounding_box( num_to_image_thanasi(x, L, Trainable_Pixel * scale,result_detector_radius),
                                                       int(diff / 2), int(diff / 2), image_size, image_size))
        elif version == 1:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi_v2(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
        elif version == 2:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
                lambda x: tf.image.pad_to_bounding_box(
                    num_to_image_thanasi_v3(x, L, Trainable_Pixel * scale, result_detector_radius),
                    int(diff / 2), int(diff / 2), image_size, image_size))
    else:
        #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
        if version == 0 :
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image_thanasi(x, L, Trainable_Pixel * scale,result_detector_radius), (image_size, image_size) ))
        elif version == 1:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
                lambda x: tf.image.resize(num_to_image_thanasi_v2(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))
        elif version ==2:
            train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
                lambda x: tf.image.resize(num_to_image_thanasi_v3(x, L, Trainable_Pixel * scale, result_detector_radius),
                                          (image_size, image_size)))

    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).repeat(10000).shuffle(1).batch(images)
    test_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).shuffle(1).batch(images)

    return train_dataset, test_dataset

def get_thanasi_measured_dataset_focus(images, image_size, L=1.0,radius=0.1,  version =2):

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    def get_middle_focus_image(N, L, radius):
        img = np.zeros(shape = (N,N), dtype = 'float32')
        #X,Y = np.meshgrid([])
        dx = L/N
        Y, X = np.mgrid[0:L:dx, 0:L:dx] - L / 2
        img[ X**2 + Y**2 < radius**2] =1.0
        return img

    image = get_middle_focus_image(image_size,L, radius)
    image = np.reshape(image, [1, image_size, image_size,1])
    output_images = np.tile(image,[images, 1,1,1])
    train_out_dataset = tf.data.Dataset.from_tensor_slices(output_images)

    #print(x_train.shape)
    #print(output_images.shape)

    y_train_categories = np.array([0,1,2,3,4,5,6,7,8,9])


        #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
    if version == 1:
        train_inp_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: resize_complex_image( num_to_image_measured(x, 1), ( image_size, image_size)))
    elif version == 2:
        train_inp_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured_v2(x, 1), (image_size, image_size)))
    elif version == 3:
        train_inp_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured_v3(x, 1), (image_size, image_size)))
    else:
        raise ValueError('version must be in [1,2,3], was {}'.format(version))

    train_dataset = tf.data.Dataset.zip((train_inp_dataset, train_out_dataset)).repeat(10000).shuffle(1).batch(images)
    test_dataset = tf.data.Dataset.zip((train_inp_dataset, train_out_dataset)).shuffle(1).batch(images)
    return train_dataset, test_dataset


def get_inverse_measured_MNIST_dataset(images, image_size, L, Trainable_Pixel, mnist_scale=1, result_detector_radius = 4,version = 1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_images = []
    y_train_categories = []
    i = 0
    while len(x_train_images) < images:

        if not (y_train[i] in y_train_categories):
            x_train_images.append(x_train[i])
            y_train_categories.append(y_train[i])
        i = i + 1
    print(mnist_scale)
    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_images).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    #diff = image_size - Trainable_Pixel * scale
    if version == 1:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: resize_complex_image( num_to_image_measured(x, 1), ( image_size, image_size)))
    elif version == 2:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured_v2(x, 1), (image_size, image_size)))
    elif version == 3:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured_v3(x, 1), (image_size, image_size)))
    else:
        raise ValueError('version must be in [1,2,3], was {}'.format(version))

    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).repeat(10000).shuffle(1).batch(images)
    test_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).shuffle(1).batch(images)
    return train_dataset,test_dataset

def get_inverse_measured_MNIST_dataset_single( image_size, mnist_scale=1, version = 1, mnist_num = 0):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_images = []
    y_train_categories = []
    i = 0
    while len(x_train_images) == 0:
        #print(y_train[i])
        #print(y_train[i], mnist_num)
        #print(y_train[i] == mnist_num)
        if (y_train[i] == mnist_num):
            x_train_images.append(x_train[i])
            y_train_categories.append(y_train[i])

            #break
        i = i + 1

    #print(mnist_scale)
    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_images).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    # diff = image_size - Trainable_Pixel * scale
    if version == 1:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured(x, 1), (image_size, image_size)))
    elif version == 2:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured_v2(x, 1), (image_size, image_size)))
    elif version == 3:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: resize_complex_image(num_to_image_measured_v3(x, 1), (image_size, image_size)))
    else:
        raise ValueError('version must be in [1,2,3], was {}'.format(version))

    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).repeat(10000).shuffle(1).batch(1)
    test_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).repeat(10).shuffle(1).batch(10)
    return train_dataset, test_dataset

def get_inverse_measured_MNIST_v2_dataset(images, image_size, L, Propagation_Pixel, mnist_scale=1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_images = []
    y_train_categories = []
    i = 0
    while len(x_train_images) < images:

        if not (y_train[i] in y_train_categories):
            x_train_images.append(x_train[i])
            y_train_categories.append(y_train[i])
        i = i + 1

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_images).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))

    diff = image_size - Propagation_Pixel

    train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: resize_complex_image( num_to_image_measured_v2(x, 1), ( image_size, image_size)))
    #if image_size > Propagation_Pixel:
    #    train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
    #        lambda x: tf.image.pad_to_bounding_box(num_to_image_measured_v2(x,1),
    #                                               int(diff / 2), int(diff / 2), image_size, image_size))
    #elif image_size < Propagation_Pixel:
    #    #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
    #    train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image_measured_v2(x,1), (image_size, image_size) ))
    #else:
    #    train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: num_to_image_measured_v2(x, 1))
    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_img_dataset)).shuffle(1).batch(images)
    return train_dataset


def get_and_anim_dataset(batch_num, image_size,L, trainable_pixel , scale =1 , radius = 0.4, input_type = 'circles', double = True):
#(batch_num, input_num, logical_function, image_size,L , trainable_pixel, scale=1,radius = 4, input_type = 'circles', double = False)
    input_logical, output_logical = get_logical_dataset(2, logical_and, double)
    print(input_logical)
    print(output_logical)
    if input_type == 'circles':
        input_images = get_activation_images(input_logical, int(trainable_pixel * scale), L, radius)
        output_images = get_activation_images(output_logical, int(trainable_pixel * scale), L, radius)
    elif input_type == 'TEM_mn':
        input_images = get_activation_images_TEM_mn(input_logical, int(trainable_pixel * scale), L, radius)
        output_images = get_activation_images_TEM_mn(output_logical, int(trainable_pixel * scale), L, radius)
    elif input_type == 'squares':
        input_images = get_square_activation_images(input_logical, int(trainable_pixel * scale), L, radius)
        output_images = get_square_activation_images(output_logical, int(trainable_pixel * scale), L, radius)

    input_images = np.reshape(input_images,
                              (input_images.shape[0], input_images.shape[1], input_images.shape[2], 1)).astype(
        'float32')
    output_images = np.reshape(output_images,
                               (output_images.shape[0], output_images.shape[1], output_images.shape[2], 1)).astype(
        'float32')
    N = int(trainable_pixel * scale)
    image_arr = np.zeros( (4,N,N,1), 'float32')
    line_width = 3
    for i in range(0,4):
        Img = Image.new('1',(N,N))
        draw = ImageDraw.Draw(Img)

        if i == 0 :

            ellipse_pos = [ int( 0.2*N), int(0.2*N), int(0.8*N), int(0.8*N)]
            draw.ellipse(ellipse_pos, fill = (1), outline= (1))

            ellipse_2_pos = [ int( 0.2*N)+line_width, int(0.2*N)+line_width, int(0.8*N)-line_width, int(0.8*N)-line_width]
            draw.ellipse(ellipse_2_pos, fill = (0), outline= (0))

        if i == 1:
            rect_pos = [  int( 0.2*N), int(0.2*N), int(0.8*N), int(0.8*N)]
            rect_pos_2 = [int(0.2 * N)+line_width, int(0.2 * N)+line_width, int(0.8 * N)-line_width, int(0.8 * N)-line_width]
            draw.rectangle(rect_pos, fill = (1), outline=  (1))
            draw.rectangle(rect_pos_2, fill = (0), outline = (0))
        if i ==2:
            line_pos =  [int( 0.2*N), int(0.2*N), int(0.8*N), int(0.8*N)]
            draw.line(line_pos, fill = (1), width = line_width)
            line_pos2 = [int(0.2*N), int(0.8*N), int(0.8*N), int(0.2*N)]
            draw.line(line_pos2, fill = (1), width = line_width)
        if i ==3:
            polygon_pos = [int(0.5*N),int(0.2*N), int(0.2*N), int(0.5*N), int(0.5*N), int(0.8*N), int(0.8*N), int(0.5*N) ]
            polygon_pos_2 =  [int(0.5*N),int(0.2*N) + line_width, int(0.2*N) + line_width, int(0.5*N), int(0.5*N), int(0.8*N)- line_width, int(0.8*N)-line_width, int(0.5*N) ]
            draw.polygon(polygon_pos, fill = (1), outline = (1))
            draw.polygon(polygon_pos_2, fill = (0), outline = (0))

        #plt.imshow(Img)
        #plt.show()
        image_arr[i,:,:,0] = np.array(Img)
    #print(input_images.shape)

    output_images = image_arr.astype('float32')


    #plt.imshow(output_images[0, :, :, 0])
    #plt.show()
    diff = image_size - (trainable_pixel * scale)
    if image_size > trainable_pixel:
        train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
            lambda x: tf.image.pad_to_bounding_box(x, int(diff / 2), int(diff / 2), image_size, image_size))
        train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
            lambda x: tf.image.pad_to_bounding_box(x, int(diff / 2), int(diff / 2), image_size, image_size))
    else:
        train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
            lambda x: tf.image.resize(x, (image_size, image_size)))
        train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
            lambda x: tf.image.resize(x, (image_size, image_size)))
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_output_dataset)).batch(batch_num)

    return train_dataset







def get_and_anim_6inp_dataset(batch_num, image_size,L , radius = 0.4, input_type = 'circles'):
#(batch_num, input_num, logical_function, image_size,L , trainable_pixel, scale=1,radius = 4, input_type = 'circles', double = False)
    #input_logical, output_logical = get_logical_dataset(4, logical_and, False)
    #print(input_logical)
    #print(output_logical)
    #input()
    input_logical = np.array( [[0,0,1,1],[0,1,0,1],[1,0,0,1],[0,1,1,0],[1,0,1,0],[1,1,0,0]])
    if radius == None:
        radius = L/6.0

    if input_type == 'circles':
        input_images = get_activation_images(input_logical, image_size, L, radius)
        #output_images = get_activation_images(output_logical, trainable_pixel, L, radius)
    elif input_type == 'TEM_mn':
        input_images = get_activation_images_TEM_mn(input_logical, image_size, L, radius)
        #output_images = get_activation_images_TEM_mn(output_logical, trainable_pixel, L, radius)
    elif input_type == 'squares':
        input_images = get_square_activation_images(input_logical, image_size , L, radius)
        #output_images = get_square_activation_images(output_logical, trainable_pixel, L, radius)
    elif input_type == 'measured':
        input_images = get_measured_activation_images(input_logical, image_size,L)
    elif input_type == 'measured_v2':
        input_images = get_measured_activation_images_v2(input_logical, image_size,L)

    input_images = np.reshape(input_images,
                              (input_images.shape[0], input_images.shape[1], input_images.shape[2], 1))
    #output_images = np.reshape(output_images,
    #                           (output_images.shape[0], output_images.shape[1], output_images.shape[2], 1)).astype(
    #    'float32')
    N = image_size
    image_arr = np.zeros( (6,N,N,1), 'float32')
    line_width = int(N/10)

    for i in range(0,6):
        Img = Image.new('1',(N,N))
        draw = ImageDraw.Draw(Img)

        if i == 0 :

            ellipse_pos = [ int( 0.2*N), int(0.2*N), int(0.8*N), int(0.8*N)]
            draw.ellipse(ellipse_pos, fill = (1), outline= (1))

            ellipse_2_pos = [ int( 0.2*N)+line_width, int(0.2*N)+line_width, int(0.8*N)-line_width, int(0.8*N)-line_width]
            draw.ellipse(ellipse_2_pos, fill = (0), outline= (0))

        if i == 1:
            rect_pos = [  int( 0.2*N), int(0.2*N), int(0.8*N), int(0.8*N)]
            rect_pos_2 = [int(0.2 * N)+line_width, int(0.2 * N)+line_width, int(0.8 * N)-line_width, int(0.8 * N)-line_width]
            draw.rectangle(rect_pos, fill = (1), outline=  (1))
            draw.rectangle(rect_pos_2, fill = (0), outline = (0))
        if i ==2:
            line_pos =  [int( 0.2*N), int(0.2*N), int(0.8*N), int(0.8*N)]
            draw.line(line_pos, fill = (1), width = line_width)
            line_pos2 = [int(0.2*N), int(0.8*N), int(0.8*N), int(0.2*N)]
            draw.line(line_pos2, fill = (1), width = line_width)
        if i ==3:
            polygon_pos = [int(0.5*N),int(0.2*N), int(0.2*N), int(0.5*N), int(0.5*N), int(0.8*N), int(0.8*N), int(0.5*N) ]
            polygon_pos_2 =  [int(0.5*N),int(0.2*N) + line_width, int(0.2*N) + line_width, int(0.5*N), int(0.5*N), int(0.8*N)- line_width, int(0.8*N)-line_width, int(0.5*N) ]
            draw.polygon(polygon_pos, fill = (1), outline = (1))
            draw.polygon(polygon_pos_2, fill = (0), outline = (0))
        if i == 4:
            polygon_pos = [int(0.5*N), int(0.5*N), N*0.4]
            polygon_pos_2 = [int(0.5*N), int(0.5*N), N*0.4-line_width*1.5]
            draw.regular_polygon(polygon_pos, 3, rotation = 0, fill = (1), outline = (1))
            draw.regular_polygon(polygon_pos_2, 3, rotation = 0, fill = (0), outline = (0))
        if i == 5:
            polygon_pos = [int(0.5*N), int(0.5*N), N*0.4]
            polygon_pos_2 = [int(0.5*N), int(0.5*N), N*0.4-line_width*1.5]
            draw.regular_polygon(polygon_pos, 3, rotation = 180, fill = (1), outline = (1))
            draw.regular_polygon(polygon_pos_2, 3, rotation = 180, fill = (0), outline = (0))
        #if i == 6:
        #    polygon_pos = [int(0.5*N), int(0.5*N), N*0.3]
        #    polygon_pos_2 = [int(0.5*N), int(0.5*N), N*0.3-line_width]
        #    draw.regular_polygon(polygon_pos, 6, rotation = 0, fill = (1), outline = (1))
        #    draw.regular_polygon(polygon_pos_2, 6, rotation = 0, fill = (0), outline = (0))
        #if i == 7:
        #    polygon_pos = [int(0.5*N), int(0.5*N), N*0.3]
        #    polygon_pos_2 = [int(0.5*N), int(0.5*N), N*0.3-line_width]
        #    draw.regular_polygon(polygon_pos, 6, rotation = 90, fill = (1), outline = (1))
        #    draw.regular_polygon(polygon_pos_2, 6, rotation = 90, fill = (0), outline = (0))


        #plt.imshow(Img)
        #plt.show()
        image_arr[i,:,:,0] = np.array(Img)
    #print(input_images.shape)

    output_images = image_arr.astype('float32')


    #diff = image_size - trainable_pixel
    #if image_size > trainable_pixel:
    #    train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
    #        lambda x: tf.image.pad_to_bounding_box(x, int(diff / 2), int(diff / 2), image_size, image_size))
    #    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
    #        lambda x: tf.image.pad_to_bounding_box(x, int(diff / 2), int(diff / 2), image_size, image_size))
    #else:
    #    train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
    #        lambda x: tf.image.resize(x, (image_size, image_size)))
    #    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
    #        lambda x: tf.image.resize(x, (image_size, image_size)))

    if input_images.dtype == 'complex64':
        train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
            lambda x: resize_complex_image(x, (image_size, image_size)))
    elif input_images.dtype == 'float32':
        train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
            lambda x: tf.image.resize(x, (image_size, image_size)))
    elif input_images.dtype == 'float64':
        train_img_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(
            lambda x: tf.image.resize(tf.cast(x,tf.float32), (image_size, image_size)))


    train_output_dataset = tf.data.Dataset.from_tensor_slices(output_images).map(
        lambda x: tf.image.resize(x, ( image_size, image_size)))

    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_output_dataset)).repeat(10000).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((train_img_dataset, train_output_dataset)).batch(batch_num)
    return train_dataset, test_dataset







def get_seperate_input_anim_dataset(batch_num, image_size, L, scale = 1, anim_path = './snake_anim_30px.npy'):
    snake_images = np.reshape(np.load(anim_path), (batch_num,image_size, image_size,1))
    diff = image_size - Trainable_Pixel * scale
    train_snake_dataset = tf.data.Dataset.from_tensor_slices(snake_images).map(
        lambda x: tf.image.pad_to_bounding_box(tf.image.resize(x,(Trainable_Pixel*scale, Trainable_Pixel*scale)), int(diff/2), int(diff/2), image_size, image_size)
    )

    y_train_categories = [0,1,2,3,4,5,6,7,8,9]
    if image_size > Trainable_Pixel:
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
            lambda x: tf.image.pad_to_bounding_box(num_to_image_measured(x,scale),
                                                   int(diff / 2), int(diff / 2), image_size, image_size))
    else:
        #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
        train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image_thanasi(x,L,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
    #print(snake_images.shape)
    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_snake_dataset)).batch(batch_num)
    return train_dataset


def get_numpy_dataset(batch_num, image_size, L, Trainable_Pixel, scale = 1, input_path = './vortex_beam_30px_input.npy', output_path = './vortex_beam_30px_output.npy'):
    input = np.load(input_path)
    input = tf.cast(np.reshape( input, (input.shape[0], input.shape[1], input.shape[2], 1)), dtype =tf.dtypes.complex64)
    output = np.load(output_path)
    output = tf.cast(np.reshape( output, ( output.shape[0], output.shape[1], output.shape[2], 1)),dtype = tf.dtypes.complex64)
    diff = image_size - Trainable_Pixel*scale
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input).map(
        lambda x: tf.image.pad_to_bounding_box( tf.repeat(tf.repeat(x, scale,0), scale, 1), int(diff/2), int(diff/2), image_size, image_size)
    )

    train_output_dataset = tf.data.Dataset.from_tensor_slices(output).map(
        lambda x:  tf.repeat(tf.repeat(x, scale,0), scale, 1)
    )



    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_output_dataset)).batch(batch_num)
    return train_dataset


def get_anim_dataset(batch_num,  image_size, L, Trainable_Pixel,scale = 1, anim_path = './snake_anim_30px.npy'):
    snake_images =np.reshape(np.load(anim_path),(batch_num,Trainable_Pixel,Trainable_Pixel,1))
    diff = image_size - Trainable_Pixel * scale


    train_snake_dataset = tf.data.Dataset.from_tensor_slices(snake_images).map(
        lambda x: tf.image.pad_to_bounding_box(tf.image.resize(x,(Trainable_Pixel*scale, Trainable_Pixel*scale)), int(diff/2), int(diff/2), image_size, image_size)
    )

    y_train = np.array([0,1,2,3,4,5,6,7,8,9])

    input_array = tf.expand_dims(create_combinatory_array_input(Trainable_Pixel*scale, L),3)
    input_array = input_array[0:batch_num, :,:,:]
    #print(input_array)
    #print(input_array.shape)
    #print(snake_images.shape)
    #plt.imshow(input_array[0,:,:,0])
    #plt.show()
    if image_size > Trainable_Pixel:
        train_input_dataset = tf.data.Dataset.from_tensor_slices(input_array).map(
            lambda x: tf.image.pad_to_bounding_box(x,
                                               int(diff / 2), int(diff / 2), image_size, image_size)
        )
    else:
        train_input_dataset = tf.data.Dataset.from_tensor_slices(input_array).map(
            lambda x: tf.image.resize(x,
                                      (image_size, image_size))
        )
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_snake_dataset)).shuffle(1).batch(batch_num)
    return train_dataset




def get_nonlinear_MNIST_dataset(image_size,L, Trainable_Pixel, scale= 1, mnist_scale = 1, result_detector_radius = 4):
    mnist=  tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_images = []
    y_train_categories = []

    i = 0
    category = 0
    while len(x_train_images) < 3:
        if category == y_train[i]:
            x_train_images.append(x_train[i])
            y_train_categories.append(y_train[i])
            category = category +1
        i = i +1

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_images).map(
        lambda x: get_scaled_mnist_images(x, image_size, mnist_scale))
    output_img_data = get_activation_images(np.array([[1,0],[0,1],[1,1]]), int(Trainable_Pixel*scale),L,  result_detector_radius)
    output_img_data = np.reshape(output_img_data,
                               (output_img_data.shape[0], output_img_data.shape[1], output_img_data.shape[2], 1)).astype(
        'float32')
    diff = image_size - Trainable_Pixel * scale
    if image_size > Trainable_Pixel:
        #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
         #   lambda x: tf.image.pad_to_bounding_box(
         #       num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
         #       int(diff / 2), int(diff / 2), image_size, image_size))
        train_out_dataset = tf.data.Dataset.from_tensor_slices(output_img_data).map(
            lambda x: tf.image.pad_to_bounding_box(
                x, int(diff/2), int(diff/2), image_size, image_size
            )
        )

    else:
        # train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(lambda x: tf.image.resize( num_to_image(x,Trainable_Pixel*scale,result_detector_radius), (image_size, image_size) ))
        #train_num_dataset = tf.data.Dataset.from_tensor_slices(y_train_categories).map(
         #   lambda x: tf.image.resize(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
        #                              (image_size, image_size)))
        train_out_dataset = tf.data.Dataset.from_tensor_slices(output_img_data).map(
            lambda x: tf.image.resize(x, (image_size,image_size))
        )


    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_out_dataset)).shuffle(1).batch(3)
    return train_dataset


def get_crystal_phase_retrieval_data( batch_num, N = 128, M = 100, path = 'data/crystal_phase_retrieval.npy'):
    create_new_data = False
    try:
        data = np.load(path)
    except:
        print('Data not found')
        create_new_data = True

    if data.shape[1] != N or data.shape[2] != N:
        create_new_data = True

    if create_new_data:
        print('Creating new data. This could take a while')

        images, res = create_crystal_phase_retrieval_data(40000, N, M)
        data = np.concatenate(([images],[res]), axis = 0)
        np.save(path, data)

    images, fourier_images = (data[0,:,:], data[0,:,:])
    train_images = tf.data.Dataset.from_tensor_slices(images[0:30000,:,:])
    train_measurements = tf.data.Dataset.from_tensor_slices(fourier_images[0:30000,:,:,])
    train_dataset = tf.data.Dataset.zip( (train_images, train_measurements)).shuffle(1).batch(batch_num)


    test_images = tf.data.Dataset.from_tensor_slices(images[30000:40000,:,:])
    test_measurements = tf.data.Dataset.from_tensor_slices(fourier_images[30000:40000,:,:])
    test_dataset = tf.data.Dataset.zip( (test_images, test_measurements)).shuffle(1).batch(batch_num)

    return train_dataset, test_dataset





def get_MNIST_dataset_3D(batch_num, N = 50, num_3D = 7, dataset = 'mnist', scale = 1, transducer_radius = 25):
    size = 28
    if dataset == 'mnist':
        set = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = set.load_data()
        x_train = x_train
        x_test = x_test
        size = 28
    if dataset == 'kmnist':
        data_train = tfds.as_numpy(tfds.load(
            'kmnist',
            split='train',
            batch_size=-1,
        ))
        data_test = tfds.as_numpy(tfds.load(
            'kmnist',
            split='test',
            batch_size=-1,
        ))
        x_train, y_train = data_train['image'][:,:,:,0], data_train['label']
        x_test, y_test = data_test['image'][:,:,:,0], data_test['label']
        size = 28
    if dataset == 'fashion':
        set = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = set.load_data()
        size = 28
    if dataset == 'cifar10':
        set = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = set.load_data()
        x_train = np.average(x_train, axis = 3)
        x_test = np.average(x_test, axis = 3)
        size = 32
    #print(x_train.shape)
    x_train_3D = np.zeros((x_train.shape[0]//num_3D, size,size,num_3D))
    x_test_3D = np.zeros((x_test.shape[0]//num_3D, size,size, num_3D))
    for i in range(0, x_train.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_train_3D[i,:,:,j] = x_train[i*num_3D + j,:,:]
    for i in range(0, x_test.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_test_3D[i,:,:,j] = x_test[i*num_3D + j,:,:]
    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, scale))
    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, scale)
    )
    #X, Y = np.mgrid[-N/2:N / 2:dx, -N / 2:N / 2:dx]
    #amplitude = np.zeros_like(X)
    #amplitude[X ** 2 + Y ** 2 < transducer_radius ** 2] = 1.0
    #amplitude = np.reshape(amplitude, (1,amplitude.shape[0], amplitude.shape[1], 1))

    #train_transducer_dataset = tf.data.Dataset.from_tensor_slices(amplitude).repeat(x_train_3D.shape[0])
    #test_transducer_dataset = tf.data.Dataset.from_tensor_slices(amplitude).repeat(x_test_3D.shape[0])
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_img_dataset)).shuffle(1000).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_img_dataset)).shuffle(1000).batch(batch_num)
    return train_dataset, test_dataset

def get_double_MNIST_dataset_3D( batch_num, N, num_3D = 7, mnist_scale = 1):
    mnist=  tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_3D = np.zeros((x_train.shape[0]//num_3D, 28,28,num_3D))
    x_test_3D = np.zeros((x_test.shape[0]//num_3D, 28,28, num_3D))
    for i in range(0, x_train.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_train_3D[i,:,:,j] = x_train[i*num_3D + j,:,:]
    for i in range(0, x_test.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_test_3D[i,:,:,j] = x_test[i*num_3D + j,:,:]

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale))
    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale)
    )
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_img_dataset)).shuffle(1).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_img_dataset)).shuffle(1).batch(batch_num)

    return train_dataset, test_dataset

def get_double_MNIST_3D_dataset_with_distances( batch_num, N, num_3D = 7, mnist_scale = 1):
    mnist=  tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_3D = np.zeros((x_train.shape[0]//num_3D, 28,28,num_3D))
    x_test_3D = np.zeros((x_test.shape[0]//num_3D, 28,28, num_3D))
    for i in range(0, x_train.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_train_3D[i,:,:,j] = x_train[i*num_3D + j,:,:]
    for i in range(0, x_test.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_test_3D[i,:,:,j] = x_test[i*num_3D + j,:,:]

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale))
    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale)
    )
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_img_dataset)).shuffle(1).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_img_dataset)).shuffle(1).batch(batch_num)

    return train_dataset, test_dataset

def get_double_fashion_MNIST_dataset_3D( batch_num, N, num_3D = 7, mnist_scale = 1):
    mnist=  tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_3D = np.zeros((x_train.shape[0]//num_3D, 28,28,num_3D))
    x_test_3D = np.zeros((x_test.shape[0]//num_3D, 28,28, num_3D))
    for i in range(0, x_train.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_train_3D[i,:,:,j] = x_train[i*num_3D + j,:,:]
    for i in range(0, x_test.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_test_3D[i,:,:,j] = x_test[i*num_3D + j,:,:]

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale))
    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale)
    )
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_img_dataset)).shuffle(1).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_img_dataset)).shuffle(1).batch(batch_num)

    return train_dataset, test_dataset

def get_double_MNIST_dataset_3D_augmented( batch_num, N, num_3D = 7, mnist_scale = 1):
    mnist=  tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_3D = np.zeros((x_train.shape[0]//num_3D, 28,28,num_3D))
    x_test_3D = np.zeros((x_test.shape[0]//num_3D, 28,28, num_3D))
    for i in range(0, x_train.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_train_3D[i,:,:,j] = x_train[i*num_3D + j,:,:]
    for i in range(0, x_test.shape[0]//num_3D):
        for j in range(0,num_3D):
            x_test_3D[i,:,:,j] = x_test[i*num_3D + j,:,:]

    train_img_dataset = tf.data.Dataset.from_tensor_slices(x_train_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale))
    test_img_dataset = tf.data.Dataset.from_tensor_slices(x_test_3D).map(
        lambda x: get_scaled_mnist_images_3D(x, N, mnist_scale)
    )

    train_img_dataset = train_img_dataset.map( lambda x: tfa.image.rotate( x, np.random.rand()*2*np.pi))
    test_img_dataset = test_img_dataset.map( lambda x : tfa.image.rotate(x, np.random.rand()*2*np.pi))

    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_img_dataset)).shuffle(1).batch(batch_num)
    test_dataset = tf.data.Dataset.zip((test_img_dataset, test_img_dataset)).shuffle(1).batch(batch_num)

    return train_dataset, test_dataset


def get_two_input_xy_set(num,batch_size,image_size,L, Trainable_pixel,scale):
    one = np.random.rand((num)).astype('float32')
    two = np.random.rand((num)).astype('float32')
    output_position_x = one* L
    output_position_y = two* L
    one = np.reshape(one, (num,1,1,1))
    two = np.reshape(two,(num,1,1,1))

    input_image_1 = one* create_circle_image_xy(int(1/3*L), int(1/2*L), L, int(Trainable_pixel*scale), r=3)
    input_image_2 = two* create_circle_image_xy(int(2/3*L), int(1/2*L), L, int(Trainable_pixel*scale), r=3)
    input_image = input_image_1 + input_image_2
    #print(input_image_1.dtype)
    #print(input_image_1.shape)
    #print(input_image_2.shape)
    #print(output_position_x.shape)
    #print(one[0])
    #print(two[0])
    #plt.imshow(input_image_2[:,:,0])
    #plt.figure()

    output_image = create_circle_image_xy(output_position_x, output_position_y, L, int(Trainable_pixel*scale), r = 3)

    diff = image_size - Trainable_pixel * scale
    input_image = tf.image.pad_to_bounding_box(input_image, int(diff/2), int(diff/2), image_size, image_size)
    output_image = tf.image.pad_to_bounding_box(output_image, int(diff/2), int(diff/2), image_size, image_size)
    #tf.image.pad_to_bounding_box(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
    #                            int(diff / 2), int(diff / 2), image_size, image_size))


    train_dataset = tf.data.Dataset.from_tensor_slices((input_image,output_image)).shuffle(1).batch(batch_size)
    return train_dataset

def get_two_input_interpolate_set(num,batch_size,image_size, L, Trainable_pixel, scale, radius = 4, radius_out = 1):
    one = np.random.rand((num)).astype('float32')
    #two = np.random.rand((num)).astype('float32')
    output_position_x = L/5 + one*(L*3/5)
    output_position_y = np.repeat(L/2, num)
    one = np.reshape(one, (num,1,1,1))
    #two = np.reshape(two,(num,1,1,1))

    input_image_1 = (1-one)* create_circle_image_xy(int(1/5*L), int(1/2*L), L, int(Trainable_pixel*scale), r=radius)
    input_image_2 = one* create_circle_image_xy(int(4/5*L), int(1/2*L), L, int(Trainable_pixel*scale), r=radius)
    input_image = input_image_1 + input_image_2
    #print(input_image_1.dtype)
    #print(input_image_1.shape)
    #print(input_image_2.shape)
    #print(output_position_x.shape)
    #print(one[0])
    #print(two[0])
    #plt.imshow(input_image_2[:,:,0])
    #plt.figure()

    output_image = create_circle_image_xy(output_position_x, output_position_y, L, int(Trainable_pixel*scale), r = radius_out)

    diff = image_size - Trainable_pixel * scale
    input_image = tf.image.pad_to_bounding_box(input_image, int(diff/2), int(diff/2), image_size, image_size)
    output_image = tf.image.pad_to_bounding_box(output_image, int(diff/2), int(diff/2), image_size, image_size)
    #tf.image.pad_to_bounding_box(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
    #                            int(diff / 2), int(diff / 2), image_size, image_size))


    train_dataset = tf.data.Dataset.from_tensor_slices((input_image,output_image)).shuffle(1).batch(batch_size)
    return train_dataset

def get_four_input_interpolate_set(num,batch_size,image_size, L, Trainable_pixel, scale, radius = 4, radius_out = 1):
    one = np.random.rand((num)).astype('float32')
    two = np.random.rand((num)).astype('float32')
    output_position_x = L/5 + one*(L*4/5)
    #output_position_y = np.repeat(L/2, num)
    output_position_y = L/5 + two*(L*4/5)
    one = np.reshape(one, (num,1,1,1))
    two = np.reshape(two,(num,1,1,1))

    input_image_1 = (1-one)* create_circle_image_xy(int(1/5*L), int(1/5*L), L, int(Trainable_pixel*scale), r=radius)
    input_image_2 = one* create_circle_image_xy(int(4/5*L), int(1/5*L), L, int(Trainable_pixel*scale), r=radius)
    input_image_3 = (1-two) * create_circle_image_xy(int(1/5*L), int(4/5*L),L, int(Trainable_pixel*scale), r = radius)
    input_image_4 = two * create_circle_image_xy(int(4/5*L), int(4/5*L),L, int(Trainable_pixel*scale), r = radius)
    input_image = input_image_1 + input_image_2 + input_image_3 + input_image_4
    input_image = input_image * 2*np.pi
    #print(input_image_1.dtype)
    #print(input_image_1.shape)
    #print(input_image_2.shape)
    #print(output_position_x.shape)
    #print(one[0])
    #print(two[0])
    #plt.imshow(input_image_2[:,:,0])
    #plt.figure()

    output_image = create_circle_image_xy(output_position_x, output_position_y, L, int(Trainable_pixel*scale), r = radius_out)

    diff = image_size - Trainable_pixel * scale
    input_image = tf.image.pad_to_bounding_box(input_image, int(diff/2), int(diff/2), image_size, image_size)
    output_image = tf.image.pad_to_bounding_box(output_image, int(diff/2), int(diff/2), image_size, image_size)
    #tf.image.pad_to_bounding_box(num_to_image_thanasi(x, L, Trainable_Pixel * scale, result_detector_radius),
    #                            int(diff / 2), int(diff / 2), image_size, image_size))


    train_dataset = tf.data.Dataset.from_tensor_slices((input_image,output_image)).shuffle(1).batch(batch_size)
    return train_dataset


def get_minerva_optimization_dataset(batch_num, image_size,L , radius = 0.4):
    gaussian_image = np.reshape(get_gaussian_beam_image((L,L), radius, image_pixels = (image_size,image_size)),(1,image_size, image_size, 1)).astype('float32')
    minerva = (np.reshape(get_minerva_image(image_size = (image_size,image_size)), (1,image_size,image_size,1))).astype('float32')
    tf.data.Dataset.from_tensor_slices((gaussian_image,minerva))
    dataset = tf.data.Dataset.from_tensor_slices((gaussian_image, minerva)).repeat(10000).batch(batch_num)
    test_dataset = tf.data.Dataset.from_tensor_slices((gaussian_image, minerva)).batch(1)
    return dataset, test_dataset


def get_minerva_real_input_dataset(batch_num, image_size, L, radius = 0.4):
    #gaussian_image = np.reshape(get_gaussian_beam_image((L,L), 0.05, image_pixels = (image_size,image_size)),(1,image_size, image_size, 1)).astype('float32')
    x = tf.constant([0.5], shape = (1,1))
    minerva = (np.reshape(get_minerva_image(image_size = (image_size,image_size)), (1,image_size,image_size,1))).astype('float32')
    #print(gaussian_image.shape)
    #print(minerva.shape)
    tf.data.Dataset.from_tensor_slices((x,minerva))
    dataset = tf.data.Dataset.from_tensor_slices((x, minerva)).repeat(10000).batch(batch_num)
    test_dataset = tf.data.Dataset.from_tensor_slices((x, minerva)).batch(1)
    return dataset, test_dataset


def get_impossible_cube_optimization_dataset(batch_num, image_size, L, radius = 0.4):
    gaussian_image = np.reshape(get_gaussian_beam_image((L,L), radius, image_pixels = (image_size,image_size)),(1,image_size, image_size, 1)).astype('float32')
    print(image_size)
    cube = (np.reshape(get_impossible_cube_image(image_size = (image_size,image_size)), (1,image_size,image_size,1))).astype('float32')
    tf.data.Dataset.from_tensor_slices((gaussian_image,cube))
    dataset = tf.data.Dataset.from_tensor_slices((gaussian_image, cube)).repeat(10000).batch(batch_num)
    test_dataset = tf.data.Dataset.from_tensor_slices((gaussian_image, cube)).batch(1)
    return dataset, test_dataset   

def get_impossible_triangle_optimization_dataset(batch_num, image_size, L, radius = 0.4):
    gaussian_image = np.reshape(get_gaussian_beam_image((L,L), radius, image_pixels = (image_size,image_size)),(1,image_size, image_size, 1)).astype('float32')
    print(image_size)
    img = (np.reshape(get_penrose_triangle_image(image_size = (image_size,image_size)), (1,image_size,image_size,1))).astype('float32')
    tf.data.Dataset.from_tensor_slices((gaussian_image, img))
    dataset = tf.data.Dataset.from_tensor_slices((gaussian_image, img)).repeat(10000).batch(batch_num)
    test_dataset = tf.data.Dataset.from_tensor_slices((gaussian_image, img)).batch(1)
    return dataset, test_dataset   

def get_regression_function( dataset_str):
    if dataset_str == 'regression_sin':
        f = lambda x: 0.1*np.sin(x*np.pi*2)+0.2
    elif dataset_str == 'regression_x2':
        f = lambda x: 0.1*x**2 + 0.2
    elif dataset_str == 'regression_steps':
        f = lambda x: 0.2*((x>0.2)*0.2 + (x>0.4)*0.2 + (x>0.6)*0.2 + (x>0.8)*0.2)+ 0.2
    elif dataset_str == 'regression_sin2x':
        f = lambda x: 0.1*np.sin(x*np.pi*4)*0.1+0.2
    elif dataset_str == 'regression_sin3x':
        f = lambda x: 0.1*np.sin(x*np.pi*6)*0.1+0.2
    elif dataset_str == 'regression_sin4x':
        f = lambda x: 0.1*np.sin(x*np.pi*8)*0.1+0.2
    elif dataset_str == 'regression_x2exp':
        f = lambda x: 3*x**2*np.exp(-4*x) + 0.4
    elif dataset_str == 'regression_linear':
        f = lambda x : x
    elif dataset_str == 'regression_neg_linear':
        f = lambda x : 1-x
    elif dataset_str == 'regression_const':
        f = lambda x : 0.5*np.ones_like(x)
    elif dataset_str == 'regression_sin_big':
        f = lambda x : 0.5*np.sin(x*np.pi*2) + 1.0
    elif dataset_str == 'regression_big_sin2x':
        f = lambda x: 0.1*np.sin(x*np.pi*4)+0.2
    elif dataset_str == 'regression_big_sin3x':
        f = lambda x: 0.1*np.sin(x*np.pi*6)+0.2
    elif dataset_str == 'regression_big_sin4x':
        f = lambda x: 0.1*np.sin(x*np.pi*8)+0.2
    elif dataset_str == 'regression_neg_sin':
        f = lambda x: 0.1*np.sin(x*np.pi*2)
    elif dataset_str == 'regression_gaussian':
        f = lambda x: 1/((0.6)*np.sqrt(2*np.pi)) * np.exp(-0.5*(x - 0.5)**2/((0.05)**2))
    elif dataset_str == 'regression_big_gaussian':
        f = lambda x: 1/((2)*np.sqrt(2*np.pi)) * np.exp(-0.5*(x - 0.5)**2/((0.2)**2))
                        
    elif dataset_str == 'regression_step':
        f = lambda x: 0.2 * ((x > 0.35) * (x < 0.65))
    elif dataset_str == 'regression_small_step':
        f = lambda x: 0.05 * ((x > 0.35) * (x < 0.65))
    elif dataset_str == 'regression_single_step':
        f = lambda x: 0.05 * (x > 0.35)
    return f

def multidimensional_sin(x, output_dim = 2, frequency = [1.0]):
    out = np.zeros((x.shape[0], output_dim))
    input_dim = x.shape[1]
    
    #print(x.shape)
    if isinstance(frequency, (float, np.floating)):
        frequency = np.full(output_dim, frequency)
        #frequeny = np.tile(frequency, (output_dim))
        
        
    assert(len(frequency) == output_dim)
    for i in range(0,output_dim):
        
        for j in range(0,input_dim):
            sin = np.sin(x[:,j] * frequency[i])
            print(sin.shape)
            out[:,i] += np.sin(x[:,j]*frequency[i])
    return out

def get_regression_dataset( batch_num, n = 10000, dataset_str = 'regression_sin'):
    f = get_regression_function(dataset_str)

    x_train = np.random.uniform(0,1, size = (n, 1))
    y_train = f(x_train)

    x_test = np.random.uniform(0,1, size = (int(n*0.1),1))
    y_test = f(x_test)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(n).batch(batch_num)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(n).batch(batch_num)

    return train_dataset, test_dataset





