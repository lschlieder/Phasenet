import tensorflow as tf
import numpy as np
from numpy.fft import fft2,ifft2,fftshift,ifftshift,fftfreq
import PIL.Image as Image
import os
from PhaseplateNetwork.utils.conversion_utils import cube_to_2d, create_field 
from skimage.draw import polygon

def divide(x,div = 10):
    return x/div

def unpad2D(x,n):
    if n==0:
        return x
    else:
        return x[...,n:-n,n:-n]

def filter_waveprop_tf(inp, z, N, L, f0=2.00e6, cM=1484, padding = None):
    dp = L / (N - 1)
    if padding == None:
        padding = L / 2

    wavelength = cM / f0
    df_p = 1 / (L + padding)
    padding_pixel = int(padding / dp) + 1
    actual_N = (N + padding_pixel)

    [fx, fy] = np.meshgrid((np.arange(-actual_N / 2, actual_N / 2, 1)) *df_p,
                                     (np.arange(-actual_N / 2, actual_N / 2, 1)) * df_p)
    kx = (2 * np.pi * fx).astype('float32')
    ky = (2 * np.pi * fy).astype('float32')

    ka = np.float32(2 * np.pi / wavelength)

    K_threshold = ka * tf.sqrt((L ** 2 / 2) / ((L ** 2 / 2) + z ** 2))
    is_smaller_threshold = tf.cast((kx ** 2 + ky ** 2) < K_threshold ** 2, dtype=tf.complex64)
    mask = tf.reshape(is_smaller_threshold, (is_smaller_threshold.shape[0], is_smaller_threshold.shape[1]))

    inp = tf.cast(inp, dtype = tf.complex64)
    img = tf.image.pad_to_bounding_box(inp, padding_pixel // 2, padding_pixel // 2, actual_N,
                                     actual_N)
    img = tf.transpose(img, (0, 3, 1, 2))

    fourier_pressure = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(img)))
    #print('mask')
    #print(mask.shape)
    #print('f pressure')
    #print(fourier_pressure.shape)
    prop_kspace = tf.multiply(fourier_pressure, mask)
    propagated_image = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(prop_kspace)))
    propagated_image = tf.transpose(propagated_image, (0, 2, 3, 1))

    propagated_image = tf.image.crop_to_bounding_box(propagated_image, padding_pixel // 2, padding_pixel // 2,
                                                     N,N)
    #print('prop img')
    #print(propagated_image.shape)

    return tf.math.abs(propagated_image)


def filter_waveprop(p0, dz, dx, f, c, D=None, Npad=0):
    """ apply a frequency-domain filter to a 2D field
    equivalent to the spectral filtering that takes place after propagating a distance dz
    this function is meant to help identify what realistic fields can look like after propagation

    Parameters
    ----------
    p0 : input pressure field (complex array)
    dz : propagation distance (mm)
    dx : pixel size (mm)
    f : acoustic frequency (MHz)
    c : medium sound speed (mm/us)
    D : window size to use for cutoff freq. defaults to window size.
    Npad : number of pixels to pad pressure field with for calculation, to remove wrap-around errors
    """
    print(dz)
    print(dx)
    print(f)
    print(p0.shape)
    p0 = p0[0,:,:,0]
    omega = 2 * np.pi * f
    k = omega / c

    N = p0.shape[0]

    if D is None:
        D = N * dx

    # cutoff spatial freq for window of size D
    # as used in Zheng & McGough (2008)
    kc = k * np.sqrt(D ** 2 / 2 / (D ** 2 / 2 + dz ** 2))

    spat_freq = fftfreq(N + 2 * Npad, dx)
    kx, ky = np.meshgrid(2 * np.pi * spat_freq, 2 * np.pi * spat_freq)
    kz2 = kx ** 2 + ky ** 2

    # mask of where kz is real
    #     mkz = kz2 < k**2
    # kc <= k so just use kc to define the mask
    mkz = kz2 < kc ** 2

    tf.image.pad_to_bounding_box()

    return unpad2D(ifft2(fft2(np.pad(p0, Npad)) * mkz), Npad)


def get_gaussian_beam_image(image_size, sigma, amp = 1.0, image_pixels = (100,100)):
    def gaus_beam(x, sigma, amp):
        #return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5* ((x- mu)/(sigma))**2)
        return amp *np.exp(- ((x)/(sigma))**2)

    xv, yv = np.meshgrid(np.linspace(-image_size[0]/2, image_size[0]/2, image_pixels[0]), np.linspace(-image_size[1]/2, image_size[1]/2, image_pixels[1]))

    g = lambda r: gaus_beam(r,sigma, amp)
    gaussian_beam_image = np.array( list(map(g, np.sqrt(xv**2 + yv**2))))
    return gaussian_beam_image

def get_minerva_image(image_size = (581,572), mode = 'bw-inv'):
    dirname = os.path.dirname(__file__)
    imgpath = os.path.join(dirname, './Datasets/Images/Minerva.jpg')
    img = Image.open(imgpath).resize(image_size)
    if mode == 'bw-inv':
        img = img.convert('L')
        img = np.abs((np.array(img)/255.0)-1.0)
    elif mode == 'bw':
        img = img.convert('L')
    else:
        raise Warning("Mode not known. Using native image mode")
    return np.array(img)


def get_impossible_cube_image(image_size = (1000,1000), mode = 'bw-inv'):
    dirname = os.path.dirname(__file__)
    imgpath = os.path.join(dirname, './Datasets/Images/impossible_cube2.png')
    img = Image.open(imgpath).resize(image_size)
    if mode == 'bw-inv':
        img = img.convert('L')
        img = np.abs((np.array(img)/255.0)-1.0)
    elif mode == 'bw':
        img = img.convert('L')    
    else:
        raise Warning("Mode not known. Using native image mode")
    return np.array(img)  

def get_penrose_triangle_image(image_size = (500,500), mode = 'bw-inv'):
    dirname = os.path.dirname(__file__)
    imgpath = os.path.join(dirname, './Datasets/Images/Penrose-dreieck.png')
    img = Image.open(imgpath).resize(image_size)
    if mode == 'bw-inv':
        img = img.convert('L')
        img = np.abs((np.array(img)/255.0)-1.0)
    elif mode == 'bw':
        img = img.convert('L')    
    else:
        raise Warning("Mode not known. Using native image mode")
    return np.array(img)  


def set_pixels_in_circle(image, x, y, r):
    """
    Set the pixels to 1 in a circular area with radius r around the position (x, y) in the image.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        x (int): X-coordinate of the center of the circle.
        y (int): Y-coordinate of the center of the circle.
        r (tuple): Radius (x,y) of the circle.

    Returns:
        numpy.ndarray: Updated image with pixels set to 1 in the circular area.
    """
    if np.any(r <= 0):
        raise ValueError("The radius (r) must be a positive value.")
    
    # Ensure the circle is entirely within the image bounds
    if x - r[1] < 0 or x + r[1] >= image.shape[1] or y - r[0] < 0 or y + r[0] >= image.shape[0]:
        raise ValueError("The circle with the given radius and position extends beyond the image bounds.")

    # Create a copy of the image to modify
    updated_image = np.copy(image)

    # Calculate the indices within the circle using meshgrid
    y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
    circle_mask = ((x_coords - x)/(r[1]))** 2 + ((y_coords - y)/(r[0]))** 2 <= 1

    # Set the pixels inside the circle to 1
    updated_image[circle_mask] = 1.0

    return updated_image

def set_pixels_in_rectangle(image, x, y, s):
    """
    Set the pixels to 1 in a rectangle with sidelength s around the position (x, y) in the image.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        x (int): X-coordinate of the center of the circle.
        y (int): Y-coordinate of the center of the circle.
        s (tuple): sidelength (x,y) of the circle.

    Returns:
        numpy.ndarray: Updated image with pixels set to 1 in the circular area.
    """
    if x - s[1]/2 < 0 or x + s[1]/2 >= image.shape[1] or y - s[0]/2 < 0 or y + s[0]/2 >= image.shape[0]:
        raise ValueError("The circle with the given sidelength and position extends beyond the image bounds.")

    # Create a copy of the image to modify
    updated_image = np.copy(image)

    # Calculate the indices within the circle using meshgrid
    y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
    rectangle_mask = (np.abs(x_coords - x)<= s[0]/2) * (np.abs(y_coords - y)<= s[1]/2)# + ((y_coords - y)/(r[0]))** 2 <= 1
    # Set the pixels inside the circle to 1
    updated_image[rectangle_mask] = 1.0

    return updated_image

def set_pixels_in_hex(image, x, y, s, orientation = 'pointy'):
    """
    Set the pixels to 1 in a hexagon with size s around the position (x, y) in the image.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        x (int): X-coordinate of the center of the circle.
        y (int): Y-coordinate of the center of the circle.
        s (int): size of hex.
    Returns:
        numpy.ndarray: Updated image with pixels set to 1 in the circular area.
    """
    if x - s < 0 or x + s >= image.shape[1] or y - s < 0 or y + s >= image.shape[0]:
        raise ValueError("The hex with the given sidelength and position extends beyond the image bounds.")

    # Create a copy of the image to modify
    updated_image = np.copy(image)

    # Calculate the indices within the circle using meshgrid
    #y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]


    hex_mask = np.zeros_like(image, dtype=bool)
    if orientation == 'pointy':
        angle = np.deg2rad(60)
        angles = np.linspace(0, 2 * np.pi, 7)
    elif orientation == 'flat':
        angle = np.deg2rad(30)
        angles = np.linspace(0, 2 * np.pi, 7) - angle

    x_coords = x + s * np.cos(angles)
    y_coords = y + s * np.sin(angles)

    rr, cc = polygon(y_coords,x_coords)

    hex_mask[rr, cc] = 1.0

    # Set the pixels inside the hex to 1
    updated_image[hex_mask] = 1.0

    return updated_image



def create_hexagonal_image(n, img_size, output_size, radius, distance= None, orientation = 'pointy'):
    """
    Create an image with img_size that contains n circular output regions positioned around the middle point on a hexagonal grid

    Parameters:
        n (int): The q coordinate in cube coordinates.
        img_size (tuple:int): the image size (x,y) in pixels
        output_size (tuple:float): the output region size (x,y) in [m]
        radius (float): radius of points in [m]
        distance (float): distance of the points in [m]
        orientation (str, optional): Orientation of the hexagons, either 'pointy' or 'flat'. Default is 'pointy'.
    Returns:
        image (np-tensor) of size img_size with circles in hexagonal grid
    """
    field = create_field(n)
    if distance == None:
        distance = 2*radius
        
    field_2d = np.array([cube_to_2d(s,distance/np.sqrt(3), orientation) for s in field])
    
    field_2d = field_2d - np.mean(field_2d, axis = 0)
    img = np.zeros(img_size)
    
    c_factor = np.array(img_size)/np.array(output_size) # conversion factor to go from size in m to pixels
    
    radius_pixels = c_factor * radius
    for p in field_2d:

        position_pixel = p* c_factor + np.array(img_size)/2
        img = set_pixels_in_circle(img, position_pixel[1], position_pixel[0], radius_pixels )
        
    return img


def create_hexagonal_image_stack(n, img_size, output_size, radius, distance= None, orientation = 'pointy', shape = 'rect'):
    """
    Create an n images with (img_size_y, img_size_x, n) that contains 1 circular output regions positioned around the middle point on a hexagonal grid

    Parameters:
        n (int): The q coordinate in cube coordinates.
        img_size (tuple:int): the image size (x,y) in pixels
        output_size (tuple:float): the output region size (x,y) in [m]
        radius (float): radius of points in [m]
        distance (float): distance of the points in [m]
        orientation (str, optional): Orientation of the hexagons, either 'pointy' or 'flat'. Default is 'pointy'.
        shape (str, optional): shape of the object to be put onto the hex grid, defaults to 'circle'. Either 'circle', 'hex', 'rect'.
    Returns:
        image (np-tensor) of size img_size with circles in hexagonal grid
    """    
    assert( shape in ['circle', 'hex', 'rect'])

    field = create_field(n)
    if distance == None:
        distance = 2*radius
        
    field_2d = np.array([cube_to_2d(s,distance/np.sqrt(3), orientation) for s in field])
    field_2d = field_2d - np.mean(field_2d, axis = 0)
    
    imgs = np.zeros((img_size[0],img_size[1], n))
    
    c_factor = np.array(img_size)/np.array(output_size) # conversion factor to go from size in m to pixels
    
    radius_pixels = c_factor * radius

    for i,p in enumerate(field_2d):
        img = np.zeros(img_size)
        position_pixel = p* c_factor + np.array(img_size)/2
        if shape =='circle':
            imgs[:,:,i] = set_pixels_in_circle(img, position_pixel[1], position_pixel[0], radius_pixels )
        elif shape == 'rect':
            imgs[:,:,i] = set_pixels_in_rectangle(img, position_pixel[1], position_pixel[0], radius_pixels)
        elif shape == 'hex':
            imgs[:,:,i] = set_pixels_in_hex(img, position_pixel[1], position_pixel[0], radius_pixels[0], orientation = orientation)



        
    return imgs


def get_gaussian_beam_image_2(pixel_number = [100,100], size = 0.001972, waist =0.0009, offset = [0,0], **kwargs):
        def return_array(param, dtype = float):
            if isinstance(param, dtype):
                ret = [param, param]
                return ret
            elif isinstance(param, list) or isinstance(param, tuple):
                assert (all(isinstance(item, dtype) for item in param))
                ret = np.array(param)
                return ret
            raise Exception(f"Could not convert parameters {param} with data type {dtype}!")
    
        
        pixel_number = return_array(pixel_number, int)
        size  =return_array(size, float)
        offset = return_array(offset, float)
        waist = waist

        X,Y = np.meshgrid(np.linspace(-size[0]/2, size[0]/2, pixel_number[0]),
                          np.linspace(-size[1]/2, size[1]/2, pixel_number[1]))
        r = np.sqrt(X**2 + Y**2)
        

        w0 = waist

        field = np.exp(- r**2/w0**2)
        return field



        
    




