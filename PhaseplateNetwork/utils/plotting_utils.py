########## Plotting Utils
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
from PhaseplateNetwork.utils.data_utils import get_regression_function
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib.patches as patches

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  figure.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  #plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image



def get_polarization_plot(jones_image, num_positions = (6,6), scale = 5.0 , width = 1.0, plot_circle = True, plot_intensity = False, fig = None, ax = None):
    if plot_intensity:
        if fig == None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax_anim = ax[0]
        else:
            assert(len(ax) == 2)
            ax_anim = ax
    else:
        if fig == None:
            fig, ax = plt.subplots(1,1, figsize= (5,5))
            ax_anim = ax
        else:
            assert(ax != None )
            ax_anim = ax

    
    xshape = jones_image.shape[1]
    yshape = jones_image.shape[0]

    grid_num_x = num_positions[0]
    grid_num_y = num_positions[1]

    factor_arr_x = (np.linspace(0, xshape - 0.00000001, grid_num_x)).astype('int32')
    factor_arr_y = (-np.linspace(0, yshape - 0.0000001, grid_num_y)).astype('int32')
    # print(factor_arr_x)

    xgrid, ygrid = np.meshgrid(np.linspace(0, xshape, grid_num_x), np.linspace(0, yshape, grid_num_y))
    if plot_circle:
        segs = np.zeros((grid_num_y * grid_num_x, 100, 2))
        line = LineCollection(segs, linestyle='solid', colors='black')

    Q = ax_anim.quiver(xgrid, ygrid, np.zeros_like(xgrid), np.zeros_like(ygrid), color=['b'], angles='xy',
                     scale=1 / scale, scale_units='xy', width = width)
    # Q2 = ax[1].quiver(xgrid, ygrid,np.zeros_like(xgrid), np.zeros_like(ygrid), color = ['b'], angles = 'xy', scale = 1/scale, scale_units = 'xy')
    # line = LineCollection(segs, linestyle='solid', colors = 'black')

    xabs = np.abs(jones_image[:, :, 0])
    yabs = np.abs(jones_image[:, :, 1])

    if plot_intensity:
        im = ax[1].imshow(np.sqrt(xabs ** 2 + yabs ** 2))
    ax_anim.add_collection(line)
    xlines = []
    ylines = []
    xline = []
    yline = []

    jones_image = np.take(jones_image, factor_arr_y, axis=0)
    jones_image = np.take(jones_image, factor_arr_x, axis=1)

    xabs = np.abs(jones_image[:, :, 0])
    xangle = np.angle(jones_image[:, :, 0])
    yabs = np.abs(jones_image[:, :, 1])
    yangle = np.angle(jones_image[:, :, 1])

    ax_anim.set_xlim(-1, xshape)
    ax_anim.set_ylim(-1, yshape)

    line.set_segments(segs)
    frame = 0
    x = np.cos(xangle - frame) * xabs
    y = np.cos(yangle - frame) * yabs



    frame_100 = np.tile(np.expand_dims(np.linspace(frame - np.pi, frame, 100), axis=1),
                        (1, grid_num_x * grid_num_y))
    # frame_100_y = np.tile(np.expand_dims(np.linspace(frame-np.pi, frame, 100),axis = 1),(1,grid_num_y*grid_num_y))

    #print(frame_100.shape)
    #print(xangle.shape)
    x_100 = np.cos(xangle.flatten() - frame_100) * xabs.flatten()

    y_100 = np.cos(yangle.flatten() - frame_100) * yabs.flatten()

    xlines = xgrid.flatten() + x_100 * scale
    ylines = ygrid.flatten() + y_100 * scale

    xlines_np = np.transpose(np.array(xlines), (1, 0))
    ylines_np = np.transpose(np.array(ylines), (1, 0))

    segs = np.stack((xlines_np, ylines_np), axis=2)

    alphas = np.ones_like(segs) * 0.1
    Q.set_UVC(x, y)
    if plot_circle:
        line.set_segments(segs)
        line.set_alpha(0.1)

    return fig, ax



def get_field_animation(fields, fps = 100, plot_angle = False ):
    def get_max_img_size(fields):
        all_max = 0
        for field in fields:
            max_size = np.max(field.shape[1], field.shape[2])
            if max_size > all_max:
                all_max = max_size
        return all_max

    def get_max_size(fields, dim = 0):
        all_max = 0
        for field in fields:
            if field.shape[dim] > all_max:
                all_max = field.shape[dim]
        return all_max


    def get_resized_image(img, size):
        img_sliced = img[0,:,:,0]
        diff_x = size - img_sliced.shape[0]
        diff_y = size - img_sliced.shape[1]
        return np.pad(img_sliced, [[diff_x//2 + diff_x % 2, diff_y//2 + diff_y %2], [ diff_x//2, diff_y//2]])

    def unset_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    #(len(fields))
    max_channel = get_max_size(fields, 3)
    max_size = max(get_max_size(fields,1),get_max_size(fields,2))

    start_img = np.zeros(  (max_size, max_size), dtype = 'complex64')
    #print(max_channel)
    if max_channel == 1:
        if plot_angle == False:
            fig, ax = plt.subplots(1,1, figsize= (5,5))
            im = ax.imshow(np.abs(start_img), interpolation='none', aspect='auto', vmin=0, vmax=1)
            unset_ticks(ax)
        else:
            fig, ax = plt.subplots(2,1, figsize= (5,10))
            im = ax[0].imshow(np.abs(start_img), interpolation='none', aspect='auto', vmin=0, vmax=1)
            im_angle = ax[1].imshow(np.angle(start_img), interpolation='none', aspect='auto', vmin=0, vmax=np.pi*2)
            unset_ticks(ax[0])
            unset_ticks(ax[1])
            #im_angle = ax.imshow(np.angle(start_img), interpolation='none', aspect='auto', vmin=0, vmax=1)
    elif max_channel == 2:
        if plot_angle == False:
            fig, [ax, ax2] = plt.subplots(1,2, figsize = (10,5))
            im = ax.imshow(np.abs(start_img),interpolation='none', aspect='auto', vmin=0, vmax=1)
            im2 = ax2.imshow(np.abs(start_img), interpolation='none', aspect='auto', vmin=0, vmax=1)
            unset_ticks(ax)
            unset_ticks(ax2)
        else:
            fig, ax = plt.subplots(2,2, figsize = (10,10))
            im = ax[0,0].imshow(np.abs(start_img),interpolation='none', aspect='auto', vmin=0, vmax=1)
            im2 = ax[0,1].imshow(np.abs(start_img), interpolation='none', aspect='auto', vmin=0, vmax=1)
            
            im_angle = ax[1,0].imshow(np.angle(start_img),interpolation='none', aspect='auto', vmin=0, vmax=np.pi*2)
            im2_angle = ax[1,1].imshow(np.angle(start_img),interpolation='none', aspect='auto', vmin=0, vmax=np.pi*2)
            unset_ticks(ax[0,0])
            unset_ticks(ax[0,1])
            unset_ticks(ax[1,0])
            unset_ticks(ax[1,1])

    def animate(frame):
        res = []
        if max_channel == 1:
            if not plot_angle:
                im.set_array(np.abs(get_resized_image(fields[frame][:,:,:,0:1], max_size)))
                res = [im]
            else:
                im.set_array(np.abs(get_resized_image(fields[frame][:,:,:,0:1], max_size)))
                im_angle.set_array(np.angle(get_resized_image(fields[frame][:,:,:,0:1], max_size)))
                res = [im, im_angle]            
        elif max_channel == 2:
            if not plot_angle:
                im.set_array(np.abs(get_resized_image(fields[frame][:,:,:,0:1], max_size)))
                if fields[frame].shape[3] == 2:
                    im2.set_array(np.abs(get_resized_image(fields[frame][:,:,:,1:2], max_size)))
                else:
                    im2.set_array(np.abs(start_img))
                res = [im, im2]
            else:
                im.set_array(np.abs(get_resized_image(fields[frame][:,:,:,0:1], max_size)))
                im_angle.set_array(np.angle(get_resized_image(fields[frame][:,:,:,0:1], max_size)))
                if fields[frame].shape[3] == 2:
                    im2.set_array(np.abs(get_resized_image(fields[frame][:,:,:,1:2], max_size)))
                    im2_angle.set_array(np.angle(get_resized_image(fields[frame][:,:,:,1:2], max_size)))
                else:
                    im2.set_array(np.abs(start_img))
                    im2_angle.set_array(np.angle(start_img))

   
                res = [im, im2, im_angle, im2_angle]     
        return res


    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(fields),
        #init_func = init,
        interval=1000/fps,  # in ms
        blit = True
    )
    return anim

def get_polarization_animation(jones_image, num_positions=(6, 6), scale=5.0, plot_circle=False, plot_intensity = False):
    if plot_intensity:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax_anim = ax[0]
    else:
        fig, ax = plt.subplots(1,1, figsize= (5,5))
        ax_anim = ax
    xshape = jones_image.shape[1]
    yshape = jones_image.shape[0]

    grid_num_x = num_positions[0]
    grid_num_y = num_positions[1]

    factor_arr_x = (np.linspace(0, xshape - 0.00000001, grid_num_x)).astype('int32')
    factor_arr_y = (-np.linspace(0, yshape - 0.0000001, grid_num_y)).astype('int32')
    # print(factor_arr_x)

    xgrid, ygrid = np.meshgrid(np.linspace(0, xshape, grid_num_x), np.linspace(0, yshape, grid_num_y))
    if plot_circle:
        segs = np.zeros((grid_num_y * grid_num_x, 100, 2))
        line = LineCollection(segs, linestyle='solid', colors='black')

    Q = ax_anim.quiver(xgrid, ygrid, np.zeros_like(xgrid), np.zeros_like(ygrid), color=['b'], angles='xy',
                     scale=1 / scale, scale_units='xy')
    # Q2 = ax[1].quiver(xgrid, ygrid,np.zeros_like(xgrid), np.zeros_like(ygrid), color = ['b'], angles = 'xy', scale = 1/scale, scale_units = 'xy')
    #line = LineCollection(segs, linestyle='solid', colors = 'black')

    xabs = np.abs(jones_image[:, :, 0])
    yabs = np.abs(jones_image[:, :, 1])
    # im = ax[2].imshow(np.sqrt(xabs**2 + yabs**2))
    if plot_intensity:
        im = ax[1].imshow(np.sqrt(xabs ** 2 + yabs ** 2))
    if plot_circle:
        ax_anim.add_collection(line)
    xlines = []
    ylines = []
    xline = []
    yline = []

    jones_image = np.take(jones_image, factor_arr_y, axis=0)
    jones_image = np.take(jones_image, factor_arr_x, axis=1)

    xabs = np.abs(jones_image[:, :, 0])
    xangle = np.angle(jones_image[:, :, 0])
    yabs = np.abs(jones_image[:, :, 1])
    yangle = np.angle(jones_image[:, :, 1])

    # print(jones_image.shape)

    def init():
        ax_anim.set_xlim(-1, xshape)
        ax_anim.set_ylim(-1, yshape)
        if plot_intensity:
            ax[1].set_xticks([])
            ax[1].set_yticks([])
        # ax[1].set_xlim(-1,xshape)
        # ax[1].set_ylim(-1,yshape)
        line.set_segments(segs)
        if plot_circle:
            ret = [line, Q]
        else:
            ret = [Q]
        return ret

    def update(frame):

        x = np.cos(xangle - frame) * xabs
        y = np.cos(yangle - frame) * yabs

        frame_100 = np.tile(np.expand_dims(np.linspace(frame - np.pi, frame, 100), axis=1),
                            (1, grid_num_x * grid_num_y))
        # frame_100_y = np.tile(np.expand_dims(np.linspace(frame-np.pi, frame, 100),axis = 1),(1,grid_num_y*grid_num_y))

        # print(xangle.shape)
        # print(frame_100.shape)
        x_100 = np.cos(xangle.flatten() - frame_100) * xabs.flatten()

        y_100 = np.cos(yangle.flatten() - frame_100) * yabs.flatten()

        # print(xgrid.shape)
        # print(y_100.shape)

        xlines = xgrid.flatten() + x_100 * scale
        ylines = ygrid.flatten() + y_100 * scale

        xlines_np = np.transpose(np.array(xlines), (1, 0))
        ylines_np = np.transpose(np.array(ylines), (1, 0))

        segs = np.stack((xlines_np, ylines_np), axis=2)

        alphas = np.ones_like(segs) * 0.1
        Q.set_UVC(x, y)
        if plot_circle:
            line.set_segments(segs)
            line.set_alpha(0.1)
            ret = [line, Q]

        # Q.set_UVC(x,y)
        else:
            ret = [Q]

        return ret

    anim = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 100), init_func=init, blit=True,
                                   interval=40)

    return anim

def save_gif_animation(anim, path):
    writergif = animation.PillowWriter(fps=30)
    writergif.setup(anim, path, dpi=200)
    anim.save(path, writer=writergif, dpi=200)
    return



def get_regression_plot_function(function_str, fig = None, ax = None,num = 20, **kwargs):
    fn = get_regression_function(function_str)
    x = np.reshape(np.linspace(0,1,num), (num,1))
    y = fn(x)
    if fig == None:
        fig, ax = plt.subplots(1,1, figsize= (5,5))
        
    ax.plot(x,y, **kwargs)
    return fig, ax 

def get_regression_plot_model(model, fig = None, ax  = None,num = 20, **kwargs):
    x = np.reshape(np.linspace(0,1,num), (num,1))
    y = model(x)
    if fig == None:
        fig, ax = plt.subplots(1,1, figsize = (5,5))
    ax.plot(x,y, **kwargs)
    return fig, ax 


def get_output_fields(model):
    #fig,ax = plt.subplots( 1,1, figsize = (6,6))
    
    inputs = np.linspace(0,1,100)
    output_fields = []
    for i,x in tqdm(enumerate(inputs)):
        fields = model.get_propagation_fields(np.reshape(x,(1,1)))
        output_fields.append(fields[-2])
    return output_fields


def get_regression_animation(model, function_str):
    fig, ax = plt.subplots(1,1,figsize = (6,6))
    inax = inset_axes(ax,
                    width="30%",
                    height="30%")
    ax.axis('off')
    
    fields = get_output_fields(model)
    fig, inax = get_regression_plot_model(model, fig = fig, ax = inax)
    fig, inax = get_regression_plot_function(function_str, fig = fig, ax = inax)


    #inax.legend(bbox_to_anchor = ax.bbox)
    fig.legend()
    line = inax.axvline(0, ls = '-', color ='r', zorder = 10)
    def update(frame):
        
        ax.imshow(np.abs(fields[frame][0,:,:,0]))
                    # Plot Red Squares               
        size = model.mean_size  # The side length of the square
        # Calculate the coordinates of the square's corners
        x = fields[frame].shape[2] // 2 - size // 2
        y = fields[frame].shape[1] // 2 - size // 2
        # Create a rectangle patch with red color
        rect = patches.Rectangle((x, y), size, size, linewidth=1, edgecolor='red', facecolor='none')
        rect2 = patches.Rectangle((x, y), size, size, linewidth=1, edgecolor='red', facecolor='none')


        line.set_xdata([frame/100,frame/100])
        return line,
        
                                                           
                                                           
    
    ani = FuncAnimation(fig, update, frames = 100, interval = 1)
    return ani


