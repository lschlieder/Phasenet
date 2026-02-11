import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from PhaseplateNetwork.Losses.Losses import *
from PIL import Image
from tqdm import tqdm
from tensorflow.python.keras.engine import data_adapter


def flatten(tensor):
    return tf.reshape(tensor, shape = (tf.size(tensor)))


def to_vector(list):
    new_list = []
    for element in list:
        new_list.append(tf.reshape(element, shape=(tf.size(element))))
    return tf.concat(new_list, axis=0)


def to_list(vector, shapes):
    offset = tf.constant(0)
    new_list = []
    #i = tf.constant(0)


    for i in range(0,len(shapes)):
        slice = tf.slice(vector, [offset], [shapes[i].num_elements()])
        new_list.append(tf.reshape(slice, shape=shapes[i]))
        offset = offset + shapes[i].num_elements()

    return new_list

def repeat_image_tensor(input, rep = 2):
    '''
    Repeats a Tensor of shape [batch, n, m , channels] 'repeat' times in the image dimensions
    :param input: image tensor
    :param rep: number of times to scale the image by repeating
    :return: Scaled image tensor
    '''
    temp = repeat(input, rep, 1)
    res = repeat(temp, rep, 2)
    return res

def repeat_2d_tensor(input,rep = 2):
    '''
    Repeats a Tensor of shape [n,m] 'repeat' times in each dimension just like np.repeat would.
    inp: [[1,2],[3,4]] -> out [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
    :param input: Tensor of shape [n,m]
    :param repeat: Number of times each element should be repeated
    :return: Tensorf of shape [repeat*n, repeat,m]
    '''
    temp = repeat(input,rep, 0)
    res = repeat(temp,rep,1)
    return res


def repeat(input, repeat = 2, dimension = 0):
    '''
    Repeats a Tensor in the dimension specified in dimension
    :param input: Tensor of arbitrary shape
    :param repeat: number of times the dimension should be repeated
    :param dimension: a dimension in the range of [0,len(input.shape)] (no -1)
    :return:
    '''
    assert dimension < len(input.shape) and dimension >= 0, 'dimension out of range for input tensor'
    temp = tf.expand_dims(input, dimension+1)
    tile = []
    for i in range(0,len(temp.shape)):
        if i == dimension+1:
            tile.append(repeat)
        else:
            tile.append(1)
    temp = tf.tile(temp, tile)
    shape = list(input.shape)
    shape[dimension] = shape[dimension] * repeat
    res = tf.reshape(temp, shape)
    return res

def create_output_directories(PATH):
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'
    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(CheckpointsPath)
    except FileExistsError:
        print('Directory {} already exists'.format(CheckpointsPath))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    try:
        os.mkdir(ImagesPath)
    except FileExistsError:
        print('Directory {} already exists'.format(ImagesPath))

    return



@tf.function
def compute_gradients_and_perform_update_step_new(model, optimizer,inp, wanted_output, loss_fn):
    with tf.GradientTape() as g:
        result = model(inp)
        loss = loss_fn(result, wanted_output)
        mse_loss = loss

        for mloss in model.losses:
            loss = loss + mloss

    #print(model.weights)
    gradients = g.gradient(loss, model.weights)
    #print(gradients)

    optimizer.apply_gradients(zip(gradients, model.weights))

    return [mse_loss] + model.losses, result


@tf.function
def compute_gradients_and_perform_update_step(model, optimizer,inp, wanted_output, loss_fn):
    with tf.GradientTape() as g:
        result, u_array = model(inp)

        loss = loss_fn(result, wanted_output)

        mse_loss = loss

        for mloss in model.losses:
            loss = loss + mloss

    gradients = g.gradient(loss, model.weights)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, model.weights))

    return [mse_loss] + model.losses, result, u_array


@tf.function
def train_step_gan( inp, generator, discriminator, gen_optimizer = tf.keras.optimizers.Adam(0.01), disc_optimizer = tf.keras.optimizers.Adam(0.01), gen_loss = generator_loss, disc_loss = discriminator_loss, alpha = 1.0):
    with tf.GradientTape() as g_disc, tf.GradientTape() as g_gen:
        generated_images, u_array = generator(inp)
        generated_images = tf.math.abs(generated_images)
        real_output = discriminator(inp)
        fake_output = discriminator(generated_images)
        #print(input)
        #print(generated_images)
        #print(real_output)
        #print(fake_output)
        generator_loss = gen_loss(fake_output, generated_images, inp, alpha = alpha)
        discriminator_loss  = disc_loss(real_output, fake_output)
        image_mse_loss = mse(generated_images, inp)
        #print(discriminator_loss)
    gradients_of_generator = g_gen.gradient(generator_loss, generator.trainable_variables)
    gradients_of_discriminator = g_disc.gradient(discriminator_loss, discriminator.trainable_variables)
    #print(gradients_of_discriminator[2])
    #input()
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #print(gradients_of_discriminator[0])
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return generator_loss, discriminator_loss, generated_images, u_array, image_mse_loss

def train_gan_model( generator, discriminator, train_dataset, gen_optimizer= tf.keras.optimizers.Adam(0.001), disc_optimizer = tf.keras.optimizers.Adam(0.001),
                    loss_fn_generator = generator_loss, alpha = 1.0, loss_fn_discriminator = discriminator_loss, epochs = 10, test_dataset = None, training = True, output_images = 3,
                    images_output_steps = 100, loss_output_steps = 100, checkpoints_output_steps = 100, PATH = '.', print_propagation_images = True):
    warnings.simplefilter('error', UserWarning)
    ImagesPath = PATH + '/InverseImageNet_Images'
    CheckpointsPath = PATH + '/InverseImageNet_chkpts'
    LossPath = PATH + '/InverseImageNet_Losses'

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), gen_optimizer=gen_optimizer, net_gen=generator,
                                                    disc_optimizer = disc_optimizer, net_disc = discriminator)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(CheckpointsPath)
    except FileExistsError:
        print('Directory {} already exists'.format(CheckpointsPath))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    try:
        os.mkdir(ImagesPath)
    except FileExistsError:
        print('Directory {} already exists'.format(ImagesPath))
    if training:
        i = 0
        i_val = 0
        generator_loss = []
        validation_gen_loss = []
        discriminator_loss = []
        validation_disc_loss = []
        generator_mse = []
        validation_gen_mse = []
        for epoch in range(0, epochs):
            ##################################
            ###Train discriminator
            ##################################
            for data in train_dataset:
                input, output = data
                #generator_batch, u_array = generator(input)

            ##################################
            ###Train generator
            ##################################
            for data in train_dataset:

                input, output = data
                if i_val % 100 != 0:
                    #loss, result, u_array = compute_gradients_and_perform_update_step(
                    #    model, optimizer, input, output, loss_fn)
                    #input, generator, discriminator, gen_optimizer = tf.keras.optimizers.Adam(0.01), disc_optimizer = tf.keras.optimizers.Adam(0.01), gen_loss = generator_loss, disc_loss = discriminator_loss):
                    gen_loss, disc_loss , result, u_array, gen_mse = train_step_gan(input, generator, discriminator, gen_optimizer, disc_optimizer, loss_fn_generator, loss_fn_discriminator, alpha = alpha)
                    wanted_output = output
                    #aggregated_loss.append((loss.numpy(), int(ckpt.step)))
                    generator_loss.append((gen_loss.numpy(), int(ckpt.step)))
                    discriminator_loss.append((disc_loss.numpy(), int(ckpt.step)))
                    generator_mse.append((gen_mse.numpy(), int(ckpt.step)))

                    if int(ckpt.step) % checkpoints_output_steps == 0:
                        save_path = manager.save()
                    if i % loss_output_steps == 0:
                        print('generator loss: {}, generator mse: {}, discriminator loss: {}'.format(gen_loss.numpy(),gen_mse.numpy(), disc_loss.numpy()))
                        np.save(LossPath + '/generator_loss', generator_loss)
                        np.save(LossPath + '/discriminator_loss', discriminator_loss)
                        np.save(LossPath + '/validation_gen_loss', validation_gen_loss)
                        np.save(LossPath + '/validation_disc_loss', validation_disc_loss)
                        np.save(LossPath + '/generator_mse', generator_mse)
                        np.save(LossPath + '/validation_gen_mse', validation_gen_mse)
                    if i % images_output_steps == 0:
                        for k in range(0, wanted_output.shape[3]):
                            plt.figure()
                            plt.imshow(wanted_output.numpy()[0, :, :, k])
                            plt.colorbar()
                            plt.savefig(ImagesPath + '/wanted_output_{}_{}'.format(int(ckpt.step),k))

                            plt.close()

                            plt.figure()
                            plt.imshow(result.numpy()[0, :, :, k])
                            plt.colorbar()
                            plt.savefig(ImagesPath + '/output_{}_{}'.format(int(ckpt.step),k))

                            plt.close()

                        if print_propagation_images:
                            #u_array = generator.get_propagation()
                            for img in range(0, len(u_array)):
                                #print(u_array[img])
                                #print(u_array[img].numpy())
                                plt.figure()
                                plt.imshow(np.angle(u_array[img].numpy()[0, :, :, 0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath + '/propagation_{}_{}'.format(int(ckpt.step), img))
                                plt.close()
                                plt.figure()
                                plt.imshow(np.abs(u_array[img].numpy()[0,:,:,0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath+ '/result_abs_{}_{}'.format(int(ckpt.step),img))
                                plt.close()

                    ckpt.step.assign_add(1)
                    i = i + 1
                else:
                    val_generated_images, u_array = generator(input)
                    val_generated_images = tf.math.abs(val_generated_images)
                    val_disc_real_out = discriminator(input)
                    val_disc_fake_out = discriminator(val_generated_images)

                    val_gen_loss = loss_fn_generator(val_disc_fake_out, val_generated_images, input, alpha = alpha)
                    val_disc_loss = loss_fn_discriminator(val_disc_real_out, val_disc_fake_out)
                    val_gen_mse = mse(val_generated_images, input)
                    #print(val_gen_loss)
                    #print(val_disc_loss)
                    validation_gen_loss.append((float(val_gen_loss), int(ckpt.step)))
                    validation_disc_loss.append((float(val_disc_loss), int(ckpt.step)))
                    validation_gen_mse.append((float(val_gen_mse), int(ckpt.step)))

                i_val = i_val + 1



        ##########SAVE LOSSES
        np.save(LossPath + '/aggregated_loss', aggregated_loss)
        np.save(LossPath + '/validation_loss', validation_loss)
    else:
        #################################################### Testing
        aggregated_loss = []
        validation_loss = []
        correct_count = 0
        all_count = 0
        for data in test_dataset:
            image, number = data
            res, u_array = model(image)
            #accuracy, count = PhaseplateNet.get_accuracy(res, number, 2)
            correct_count += count
            all_count += batch_size
        print('Test dataset had an accuracy of {}'.format(correct_count / all_count))
        np.save(LossPath + '/accuracy_test', (correct_count / all_count))

    return aggregated_loss, validation_loss



def load_model(model, CheckpointsPath='.'):
    print('Loading model from {}'.format(CheckpointsPath))

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    return model

def save_model(model,dataset, PATH = '.'):
    print('Loading Network Parameters')
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'

    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))

    try:
        f = open(PATH + '/network_config.txt', 'w+')
        f.write(model.to_json(indent=4))
        f.close()
    except:
        print('Could not save network hyperparamters')

    print('Save Model Parameters')
    model.save_phase_plates_to_numpy(PATH+'/phase_plates.npy')



    print('Save Model examples Input/Outputs')
    try:
        os.mkdir(PATH+'/ExampleImages')
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    iterator = iter(dataset)
    inp,out = next(iterator)
    model_output = model(inp)


    print('Save propagation fields')
    model.save_propagation_fields(inp,PATH+'/ExampleImages')
    np.save(PATH+'/example_data_input.npy',np.array(tf.concat([tf.cast(inp,tf.complex64),tf.cast(out,tf.complex64),tf.cast(model_output, tf.complex64)],axis = 3)))

    print('Save Phase plates to images')
    model.save_phase_plates_to_images(PATH+'/ExampleImages')

    print('Save Input Output')
    fig,axs = plt.subplots(3,inp.shape[0],figsize= (5*inp.shape[0],15))
    print(inp.shape)
    for i in range(0,inp.shape[0]):

        ax0 = axs[0,i] if inp.shape[0] > 1 else axs[0]
        ax1 = axs[1,i] if inp.shape[0] > 1 else axs[1]
        ax2 = axs[2,i] if inp.shape[0] > 1 else axs[2]

        im = ax0.imshow(np.abs(inp[i,:,:,0]))
        ax0.axis('off')
        ax0.set_title('Input {}'.format(i))
        #fig.colorbar(im,axs[0,i])

        #plt.imshow(inp[i,:,:,0])
        #plt.show()

        im = ax1.imshow(out[i,:,:,0])
        ax1.axis('off')
        ax1.set_title('Wanted output {}'.format(i))
        #fig.colorbar(im,axs[1,i])

        im = ax2.imshow(np.abs(model_output[i,:,:,0]))
        ax2.axis('off')
        ax2.set_title('Actual output {}'.format(i))
        #fig.colorbar(im,axs[2,i])
    fig.savefig(PATH+'/ExampleImages/Joint_Output.png')


    for i in range(0, inp.shape[0]):
        #model_output_image = np.zeros( (model_output.shape[1],model_output.shape[2]))
        #Image.fromarray(np.abs(model_output[i,:,:,:].numpy()))

        #plt.figure()
        img_dims = model_output.shape[3]
        fig, axs = plt.subplots(2*img_dims,2, figsize = (10*img_dims,10))
            #ax1.imshow(np.abs(model_output[i,:,:,0]))
        for j in range(0, model_output.shape[3]):
            im1 = axs[j*2,0].imshow(np.abs(model_output[i,:,:,j]))
            axs[j*2,0].axis('off')
            axs[j*2,0].set_title('Output (Abs) {}'.format(j))
            fig.colorbar(im1,ax = axs[j*2,0])

            im2 = axs[j*2+1,0].imshow(np.angle(model_output[i,:,:,j]))
            axs[j*2+1,0].axis('off')
            axs[j*2+1,0].set_title('Output (Angle) {}'.format(j))
            fig.colorbar(im2, ax = axs[j*2+1,0])

            im3 = axs[j*2,1].imshow(np.abs(out[i,:,:,j]))
            axs[j*2,1].axis('off')
            axs[j*2,1].set_title('Ideal Output (Abs) {}'.format(j))
            fig.colorbar(im3,ax = axs[j*2,1])

            im4 = axs[j*2+1,1].imshow(np.angle(out[i,:,:,j]))
            axs[j*2+1,1].axis('off')
            axs[j*2+1,1].set_title('Ideal Output (Angle) {}'.format(j))
            fig.colorbar(im3, ax = axs[j*2+1,1])
        #ax2 = plt.subplot(122)
        #ax2.imshow(np.angle(model_output[i,:,:,0]))
        #plt.colorbar()
        #ax2.axes('off')
        fig.savefig(PATH+ '/ExampleImages/Actual_output_{}.png'.format(i))



        #plt.imshow(model_output[i,:,:,:])
        #plt.axes('off')

        #plt.colorbar()


    return








def train_model( model, optimizer, train_dataset, loss_fn = mse, epochs = 10, test_dataset = None, training = True, output_images = 3,
                 images_output_steps = 100, loss_output_steps = 100, checkpoints_output_steps = 100, PATH = '.', print_propagation_images = True):
    warnings.simplefilter('error', UserWarning)
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer_model=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)


    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(CheckpointsPath)
    except FileExistsError:
        print('Directory {} already exists'.format(CheckpointsPath))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    try:
        os.mkdir(ImagesPath)
    except FileExistsError:
        print('Directory {} already exists'.format(ImagesPath))

    try:
        f = open(PATH + '/network_parameters.txt', 'w+')
        f.write(model.get_network_hyperparameters())
        f.close()
    except:
        print('Could not save network hyperparamters')

    if training:
        i = 0
        i_val = 0
        aggregated_loss = []
        validation_loss = []
        regularization_loss = []
        for epoch in range(0, epochs):
            #train_dataset = train_dataset.shuffle(1)
            for data in train_dataset:
                # image, amplitude_inp = data
                input, output = data
                #print(epoch)
                # plt.imshow(image[0,:,:,0])
                # plt.figure()
                # plt.imshow(amplitude_inp[0,:,:,0])
                # plt.show()
                # print(image.shape)
                # print(amplitude_inp.shape)
                # print(image.shape)
                # print(amplitude_inp.shape)
                if i_val % 100 != 0:
                    #if train_on_mse:
                    loss, result, u_array = compute_gradients_and_perform_update_step(
                        model, optimizer, input, output, loss_fn)

                    wanted_output = output
                        # model.perform_single_wavefront_matching_step( amplitude_inp, image)
                        # model.perform_projection_steps(amplitude_inp,image)
                        # print(image.shape)
                        # loss, result, wanted_output, u_array  = PhaseplateNet.compute_gradients_and_perform_update_step_PONCS(model, optimizer, amplitude_inp, image)
                        # result, u_array = model(amplitude_inp)
                        # result = tf.math.abs(result)
                        # loss = tf.reduce_mean((result - np.max(result) * image) ** 2)
                        # wanted_output = image
                    aggregated_loss.append((loss[0].numpy(), int(ckpt.step)))
                    if len(loss) >= 2:
                        regularization_loss.append((loss[1].numpy(), int(ckpt.step)))

                    # print(wanted_output.shape)
                    # print(result.shape)
                    # reshaped_wanted_output = np.reshape(wanted_output, (batch_size,image_size,image_size))
                    # reshaped_result = np.reshape(result,(batch_size,image_size,image_size))
                    # np.save(PATH+'/wanted_output_numpy.npy',reshaped_wanted_output)
                    # np.save(PATH+'/actual_result.npy',reshaped_result)
                    # input()
                    if int(ckpt.step) % checkpoints_output_steps == 0:
                        save_path = manager.save()
                    if i % loss_output_steps == 0:
                        print('loss: {}'.format(loss[0].numpy()))
                        np.save(LossPath + '/aggregated_loss', aggregated_loss)
                        np.save(LossPath + '/validation_loss', validation_loss)
                        if len(loss) >= 2:
                            np.save(LossPath + '/regularization_loss', regularization_loss)

                    if i % images_output_steps == 0:
                        for batch in range(0, 3):
                            for k in range(0, wanted_output.shape[3]):
                                fig, (ax1, ax2) = plt.subplots(1,2)
                                im1 = ax1.imshow(np.abs(wanted_output.numpy()[batch, :, :, k]))
                                fig.colorbar(im1,ax = ax1)

                                im2 = ax2.imshow(np.angle(wanted_output.numpy()[batch,: ,: ,k]))
                                fig.colorbar(im2,ax = ax2)
                                fig.savefig(ImagesPath + '/wanted_output_batch_abs:{}_step:{}_channel{}'.format(batch,int(ckpt.step),k))
                                plt.close()


                                #print(result.numpy())
                                #print(np.sum(np.isnan(result.numpy())))
                                fig, (ax1, ax2) = plt.subplots(1,2)

                                im1 = ax1.imshow(np.abs(result.numpy()[batch,:,:,k]))
                                fig.colorbar(im1,ax = ax1)
                                im2 = ax2.imshow(np.angle(result.numpy()[batch,:,:,k]))
                                fig.colorbar(im2,ax = ax2)

                                fig.savefig(ImagesPath + '/output_batch:{}_step:{}_channel{}'.format(batch,int(ckpt.step),k))
                                plt.close()

                                #plt.close()
                        #print(model.weights)
                        '''
                        plt.imshow(wanted_output.numpy()[1, :, :, 0])
                        plt.savefig(ImagesPath + '/wanted_output_1_{}'.format(int(ckpt.step)), vmin=0.0, vvmax=1.0)
                        plt.close()
                        plt.imshow(wanted_output.numpy()[2, :, :, 0])
                        plt.savefig(ImagesPath + '/wanted_output_2_{}'.format(int(ckpt.step)))
                        plt.close()
                        plt.figure()
                        plt.imshow(wanted_output.numpy()[3, :, :, 0])
                        plt.savefig(ImagesPath + '/wanted_output_3_{}'.format(int(ckpt.step)))
                        plt.close()

                        plt.figure()
                        plt.imshow(result.numpy()[0, :, :, 0])
                        plt.savefig(ImagesPath + '/output_0_{}'.format(int(ckpt.step)), vmin=0.0, vmax=1.0)
                        plt.close()
                        plt.figure()
                        plt.imshow(result.numpy()[1, :, :, 0])
                        plt.savefig(ImagesPath + '/output_1_{}'.format(int(ckpt.step)), vmin=0.0, vmax=1.0)
                        plt.close()
                        plt.figure()
                        plt.imshow(result.numpy()[2, :, :, 0])
                        plt.savefig(ImagesPath + '/output_2_{}'.format(int(ckpt.step)))
                        plt.close()
                        plt.figure()
                        plt.imshow(result.numpy()[3, :, :, 0])
                        plt.savefig(ImagesPath + '/output_3_{}'.format(int(ckpt.step)))
                        plt.close()
                        '''

                        if print_propagation_images:
                            #u_array = model.get_propagation()
                            #print(u_array.shape)
                            for img in range(0, len(u_array)):
                                # print(u_array[img].numpy())
                                plt.figure()
                                plt.imshow(np.angle(u_array[img].numpy()[0, :, :, 0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath + '/propagation_{}_{}'.format(int(ckpt.step), img))
                                plt.close()
                                plt.figure()
                                plt.imshow(np.abs(u_array[img].numpy()[0,:,:,0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath+ '/propagation_abs_{}_{}'.format(int(ckpt.step),img))
                                plt.close()


                        # print(len(model.weights))
                        for plate in range(0, len(model.weights) - 1):
                            plt.figure()
                            if model.use_tanh_activation:
                                plt.imshow(np.tanh(model.weights[plate].numpy()) * np.pi)
                            else:
                                plt.imshow(model.weights[plate].numpy())

                            plt.savefig(ImagesPath + '/phase_plates_{}_{}'.format(int(ckpt.step), plate))
                            plt.close()


                    ckpt.step.assign_add(1)
                    i = i + 1
                else:
                    val_res, u_array = model(input)
                    #print(val_res.shape)
                    #if train_on_mse:
                    #print(output.shape)
                    #val_loss = tf.reduce_mean((np.abs(val_res) - np.max(np.abs(val_res)) * output) ** 2)
                    val_loss = loss_fn(tf.abs(val_res), output)
                    validation_loss.append((float(val_loss), int(ckpt.step)))

                i_val = i_val + 1

        ##########SAVE LOSSES
        np.save(LossPath + '/aggregated_loss', aggregated_loss)
        np.save(LossPath + '/validation_loss', validation_loss)
    else:
        #################################################### Testing
        aggregated_loss = []
        validation_loss = []
        correct_count = 0
        all_count = 0
        for data in test_dataset:
            image, number = data
            res, u_array = model(image)
            accuracy, count = PhaseplateNet.get_accuracy(res, number, 2)
            correct_count += count
            all_count += batch_size
        print('Test dataset had an accuracy of {}'.format(correct_count / all_count))
        np.save(LossPath + '/accuracy_test', (correct_count / all_count))

    return aggregated_loss, validation_loss


def output_images( result, wanted_output,input, model, path,i ):
    for batch in range(0, min(3, result.shape[0])):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        im1 = ax1.imshow(np.abs(wanted_output[batch, :, :, 0]))
        fig.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(np.angle(wanted_output[batch, :, :, 0]))
        fig.colorbar(im2, ax=ax2)
        fig.savefig(path + '/wanted_output_batch_abs:{}_step:{}'.format(batch, i))
        plt.close()

        # print(result.numpy())
        # print(np.sum(np.isnan(result.numpy())))
        fig, (ax1, ax2) = plt.subplots(1, 2)

        im1 = ax1.imshow(np.abs(result[batch, :, :, 0]))
        fig.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(np.angle(result[batch, :, :, 0]))
        fig.colorbar(im2, ax=ax2)

        fig.savefig(path + '/output_batch:{}_step:{}'.format(batch, i))
        plt.close()
    return

def output_vector(result, wanted_output, input, model,path, class_images, i ):
    index = np.where(wanted_output == 1)
    for batch in range(0, min(3, result.shape[0])):

        fig, (ax1, ax2) = plt.subplots(1, 2)

        im1 = ax1.imshow(np.abs(class_images[index[1][batch]][:,:,0]))
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(np.angle(class_images[index[1][batch]][:,:,0]))
        fig.colorbar(im2, ax=ax2)
        fig.savefig(path + '/wanted_output_batch_abs:{}_step:{}'.format(batch, i ))
        plt.close()

        # print(result.numpy())
        # print(np.sum(np.isnan(result.numpy())))
        fig, (ax1, ax2) = plt.subplots(1, 2)

        im1 = ax1.imshow(np.abs(result.numpy()[batch, :, :, 0]))
        fig.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(np.angle(result.numpy()[batch, :, :, 0]))
        fig.colorbar(im2, ax=ax2)
        fig.savefig(path + '/output_batch:{}_step:{}'.format(batch, i))
        plt.close()
    return

def train_new_model( model, optimizer, train_dataset, loss_fn = mse, metrics = [], epochs = 10, test_dataset = None, training = True, output_function = output_images,
                 images_output_steps = 100, loss_output_steps = 100, checkpoints_output_steps = 100, PATH = '.', print_propagation_images = True):
    warnings.simplefilter('error', UserWarning)
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer_model=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)


    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(CheckpointsPath)
    except FileExistsError:
        print('Directory {} already exists'.format(CheckpointsPath))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    try:
        os.mkdir(ImagesPath)
    except FileExistsError:
        print('Directory {} already exists'.format(ImagesPath))

    try:
        f = open(PATH + '/network_parameters.txt', 'w+')
        f.write(model.get_network_hyperparameters())
        f.close()
    except:
        print('Could not save network hyperparamters')

    if training:
        i = 0
        i_val = 0
        aggregated_loss = []
        validation_loss = []
        regularization_loss = []

        metrics_array = []
        for metric in metrics:
            name, m = metric
            m.reset_states()
            metrics_array.append([])

        for epoch in tqdm(range(0, epochs)):
            #train_dataset = train_dataset.shuffle(1)
            for i_m in range(0, len(metrics)):
                name, m = metrics[i_m]
                m.reset_states()

            for data in train_dataset:
                # image, amplitude_inp = data
                input, output = data

                if i_val % 100 != 0:
                    loss, result = compute_gradients_and_perform_update_step_new(
                        model, optimizer, input, output, loss_fn)

                    wanted_output = output


                    for metric in metrics:
                        name, m = metric
                        m.update_state(result, wanted_output)



                    aggregated_loss.append((loss[0].numpy(), int(ckpt.step)))
                    if len(loss) >= 2:
                        regularization_loss.append((loss[1].numpy(), int(ckpt.step)))

                    if int(ckpt.step) % checkpoints_output_steps == 0:
                        save_path = manager.save()
                    if i % loss_output_steps == 0:
                        if len(loss) > 1:
                            o_str = 'loss: {}, model loss: {}'.format(loss[0].numpy(), loss[1].numpy())
                        else:
                            o_str = 'loss: {}'.format(loss[0].numpy())
                        for i_m in range(0,len(metrics)):

                            name, m  = metrics[i_m]
                            o_str = o_str + ' ' + name+': {}'.format(m.result())
                            metrics_array[i_m].append((m.result().numpy(), int(ckpt.step)))
                            m.reset_states()
                        print(o_str)

                        np.save(LossPath + '/metrics', np.array(metrics_array))
                        np.save(LossPath + '/aggregated_loss', aggregated_loss)
                        np.save(LossPath + '/validation_loss', validation_loss)
                        if len(loss) >= 2:
                            np.save(LossPath + '/regularization_loss', regularization_loss)


                    if i % images_output_steps == 0:
                        output_function(result,wanted_output,input,model, ImagesPath, int(ckpt.step))
                        model_images = model.get_image_variables()
                        #print(len(model_images))
                        #print(model_images)
                        for plate in range(0, len(model_images)):
                            plt.figure()
                            plt.imshow(model_images[plate].numpy())

                            plt.savefig(ImagesPath + '/phase_plates_{}_{}'.format(int(ckpt.step), plate))
                            plt.close()


                    ckpt.step.assign_add(1)
                    i = i + 1
                else:
                    val_res = model(input)
                    #print(val_res.shape)
                    #if train_on_mse:
                    #print(output.shape)
                    #val_loss = tf.reduce_mean((np.abs(val_res) - np.max(np.abs(val_res)) * output) ** 2)
                    val_loss = loss_fn(tf.abs(val_res), output)
                    validation_loss.append((float(val_loss), int(ckpt.step)))

                i_val = i_val + 1

        ##########SAVE LOSSES
        np.save(LossPath + '/aggregated_loss', aggregated_loss)
        np.save(LossPath + '/validation_loss', validation_loss)
    else:
        #################################################### Testing
        aggregated_loss = []
        validation_loss = []
        correct_count = 0
        all_count = 0
        for data in test_dataset:
            image, number = data
            res, u_array = model(image)
            accuracy, count = PhaseplateNet.get_accuracy(res, number, 2)
            correct_count += count
            all_count += batch_size
        print('Test dataset had an accuracy of {}'.format(correct_count / all_count))
        np.save(LossPath + '/accuracy_test', (correct_count / all_count))

    return aggregated_loss, validation_loss

def test_model( model, test_dataset, loss_fn = mse, metrics = [], PATH = '.'):
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'

    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    print('evaluationg model')
    '''
    data_handler = data_adapter.get_data_handler(test_dataset, model = model)
    print(data_handler.enumerate_epochs())
    for _, iterator in data_handler.enumerate_epochs():
        with data_handler.catch_stop_iteration():
            for step in data_handler.steps():
                print(step)
                print(next(iterator)[0].shape)
                data = data_adapter.expand_1d(iterator)
                x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
                print(x)
                print(y)
                input()
    '''

    #metrics = model.evaluate( test_dataset, steps = 1)
    #x,y = next(iter(test_dataset))
    #metrics = model.test_on_batch(x,y)
    metrics = model.evaluate(test_dataset)

    o_str = ''
    i = 0
    header = ''

    if isinstance(metrics, float):
        metrics = [metrics]
    for val in metrics:
        o_str = o_str+ ' {}:{} '.format(model.metrics_names[i],val)
        header = header + ' '+model.metrics_names[i]
        i = i+1

    print(o_str)
    #print(np.array(metrics).shape)
    np.savetxt(LossPath + '/test_metrics.txt',np.expand_dims(np.array(metrics),0), header = header)


    np.save(LossPath + '/test_metrics', np.array(metrics))

    TestImagesOutput = PATH+'/TestDatasetOutput'
    try:
        os.mkdir(TestImagesOutput)
    except FileExistsError:
        print('Directory {} already exists'.format(TestImagesOutput))
    save_model(model, test_dataset,TestImagesOutput)

    return metrics









def train_hologram_model( model, optimizer, train_dataset, loss_fn = mse, epochs = 10, test_dataset = None, training = True, output_images = 3,
                 images_output_steps = 100, loss_output_steps = 100, checkpoints_output_steps = 100, PATH = '.', print_propagation_images = True):
    warnings.simplefilter('error', UserWarning)
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer_model=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)


    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(CheckpointsPath)
    except FileExistsError:
        print('Directory {} already exists'.format(CheckpointsPath))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    try:
        os.mkdir(ImagesPath)
    except FileExistsError:
        print('Directory {} already exists'.format(ImagesPath))

    try:
        f = open(PATH + '/network_parameters.txt', 'w+')

        f.write(model.get_network_hyperparameters())
        f.close()
    except:
        print('Could not save network hyperparamters')


    if training:
        i = 0
        i_val = 0
        aggregated_loss = []
        validation_loss = []
        regularization_loss = []
        try:
            aggregated_loss = np.load(LossPath + '/aggregated_loss.npy').tolist()

            validation_loss = np.load(LossPath + '/validation_loss.npy').tolist()

        except:
            print('Could not load old Losses. Maybe there aren\'t any yet')

        for epoch in range(0, epochs):
            #train_dataset = train_dataset.shuffle(1)
            for data in train_dataset:
                # image, amplitude_inp = data
                input, output = data
                #print(epoch)
                # plt.imshow(image[0,:,:,0])
                # plt.figure()
                # plt.imshow(amplitude_inp[0,:,:,0])
                # plt.show()
                # print(image.shape)
                # print(amplitude_inp.shape)
                # print(image.shape)
                # print(amplitude_inp.shape)
                if i_val % 100 != 0:
                    #if train_on_mse:

                    loss, result, u_array = compute_gradients_and_perform_update_step(
                        model, optimizer, input, output, loss_fn)

                    #print(result)
                    #print(loss[0].numpy())

                    wanted_output = output
                        # model.perform_single_wavefront_matching_step( amplitude_inp, image)
                        # model.perform_projection_steps(amplitude_inp,image)
                        # print(image.shape)
                        # loss, result, wanted_output, u_array  = PhaseplateNet.compute_gradients_and_perform_update_step_PONCS(model, optimizer, amplitude_inp, image)
                        # result, u_array = model(amplitude_inp)
                        # result = tf.math.abs(result)
                        # loss = tf.reduce_mean((result - np.max(result) * image) ** 2)
                        # wanted_output = image
                    aggregated_loss.append((loss[0].numpy(), int(ckpt.step)))
                    if len(loss) >= 2:
                        regularization_loss.append((loss[1].numpy(), int(ckpt.step)))

                    # print(wanted_output.shape)
                    # print(result.shape)
                    # reshaped_wanted_output = np.reshape(wanted_output, (batch_size,image_size,image_size))
                    # reshaped_result = np.reshape(result,(batch_size,image_size,image_size))
                    # np.save(PATH+'/wanted_output_numpy.npy',reshaped_wanted_output)
                    # np.save(PATH+'/actual_result.npy',reshaped_result)
                    # input()
                    if int(ckpt.step) % checkpoints_output_steps == 0:
                        save_path = manager.save()
                    if i % loss_output_steps == 0:
                        print('loss: {}'.format(loss[0].numpy()))
                        np.save(LossPath + '/aggregated_loss', aggregated_loss)
                        np.save(LossPath + '/validation_loss', validation_loss)
                        if len(loss) >= 2:
                            np.save(LossPath + '/regularization_loss', regularization_loss)

                    if i % images_output_steps == 0:
                        for batch in range(0, 3):
                            for k in range(0, wanted_output.shape[3]):
                                plt.figure()
                                plt.imshow(wanted_output.numpy()[batch, :, :, k])
                                plt.colorbar()
                                plt.savefig(ImagesPath + '/wanted_output_batch:{}_step:{}_channel{}'.format(batch,int(ckpt.step),k))

                                plt.close()

                                plt.figure()
                                #print(result.numpy())
                                #print(np.sum(np.isnan(result.numpy())))
                                plt.imshow(np.abs(result.numpy()[batch, :, :, k]))
                                plt.colorbar()
                                plt.savefig(ImagesPath + '/output_batch:{}_step:{}_channel{}'.format(batch,int(ckpt.step),k))

                                plt.close()
                        #print(model.weights)
                        '''
                        plt.imshow(wanted_output.numpy()[1, :, :, 0])
                        plt.savefig(ImagesPath + '/wanted_output_1_{}'.format(int(ckpt.step)), vmin=0.0, vvmax=1.0)
                        plt.close()
                        plt.imshow(wanted_output.numpy()[2, :, :, 0])
                        plt.savefig(ImagesPath + '/wanted_output_2_{}'.format(int(ckpt.step)))
                        plt.close()
                        plt.figure()
                        plt.imshow(wanted_output.numpy()[3, :, :, 0])
                        plt.savefig(ImagesPath + '/wanted_output_3_{}'.format(int(ckpt.step)))
                        plt.close()

                        plt.figure()
                        plt.imshow(result.numpy()[0, :, :, 0])
                        plt.savefig(ImagesPath + '/output_0_{}'.format(int(ckpt.step)), vmin=0.0, vmax=1.0)
                        plt.close()
                        plt.figure()
                        plt.imshow(result.numpy()[1, :, :, 0])
                        plt.savefig(ImagesPath + '/output_1_{}'.format(int(ckpt.step)), vmin=0.0, vmax=1.0)
                        plt.close()
                        plt.figure()
                        plt.imshow(result.numpy()[2, :, :, 0])
                        plt.savefig(ImagesPath + '/output_2_{}'.format(int(ckpt.step)))
                        plt.close()
                        plt.figure()
                        plt.imshow(result.numpy()[3, :, :, 0])
                        plt.savefig(ImagesPath + '/output_3_{}'.format(int(ckpt.step)))
                        plt.close()
                        '''

                        if print_propagation_images:
                            #u_array = model.get_propagation()
                            #print(u_array.shape)
                            for img in range(0, len(u_array)):
                                # print(u_array[img].numpy())
                                plt.figure()
                                plt.imshow(np.angle(u_array[img].numpy()[0, :, :, 0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath + '/propagation_{}_{}'.format(int(ckpt.step), img))
                                plt.close()
                                plt.figure()
                                plt.imshow(np.abs(u_array[img].numpy()[0,:,:,0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath+ '/propagation_abs_{}_{}'.format(int(ckpt.step),img))
                                plt.close()


                        # print(len(model.weights))
                        #for plate in range(0, len(model.weights) - 1):
                        #    plt.figure()
                            #if model.use_tanh_activation:
                            #    plt.imshow(np.tanh(model.weights[plate].numpy()) * np.pi)
                            #else:
                        #    plt.imshow(model.weights[plate].numpy())

                        #    plt.savefig(ImagesPath + '/phase_plates_{}_{}'.format(int(ckpt.step), plate))
                        #    plt.close()


                    ckpt.step.assign_add(1)
                    i = i + 1
                else:
                    val_res, u_array = model(input)
                    #print(val_res.shape)
                    #if train_on_mse:
                    #print(output.shape)
                    #val_loss = tf.reduce_mean((np.abs(val_res) - np.max(np.abs(val_res)) * output) ** 2)
                    val_loss = loss_fn(tf.abs(val_res), output)
                    validation_loss.append((float(val_loss), int(ckpt.step)))

                i_val = i_val + 1

        ##########SAVE LOSSES
        np.save(LossPath + '/aggregated_loss', aggregated_loss)
        np.save(LossPath + '/validation_loss', validation_loss)
    else:
        #################################################### Testing
        aggregated_loss = []
        validation_loss = []
        correct_count = 0
        all_count = 0
        for data in test_dataset:
            image, number = data
            res, u_array = model(image)
            accuracy, count = PhaseplateNet.get_accuracy(res, number, 2)
            correct_count += count
            all_count += batch_size
        print('Test dataset had an accuracy of {}'.format(correct_count / all_count))
        np.save(LossPath + '/accuracy_test', (correct_count / all_count))

    return aggregated_loss, validation_loss


@tf.function
def compute_gradients_and_perform_update_step_gradient_descent(
                        generator, regularizer, optimizer, inp, output, loss_fn= discriminator_loss, reg_optimizer = tf.keras.optimizers.Adam(0.0001)):
    generated_images, u_array, generator_loss = generator.call(inp, regularizer)
    with tf.gradient_tape() as g:
        reg_out_fake = regularizer(generated_images)
        reg_out_real = regularizer(inp)
        reg_loss = loss_fn(reg_out_real, reg_out_fake)
        image_mse_loss = mse(generated_images, inp)

    gradients_of_regularizer = g_disc.gradient(reg_loss, regularizer.trainable_variables)

    #gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    reg_optimizer.apply_gradients(zip(gradients_of_regularizer, regularizer.trainable_variables))

    return generator_loss, discriminator_loss, generated_images, u_array, image_mse_loss




def train_gradient_descent_discriminator(generator, regularizer, optimizer, train_dataset, loss_fn_reg = discriminator_loss, epochs = 10, test_dataset = None, training = True, output_images = 3,
                 images_output_steps = 100, loss_output_steps = 100, checkpoints_output_steps = 100, PATH = '.', print_propagation_images = True):

    warnings.simplefilter('error', UserWarning)
    create_output_directories(PATH)
    CheckpointsPath = PATH + '/InverseImageNet_chkpts'
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), gen_optimizer=gen_optimizer, net_gen=generator,
                                                    disc_optimizer = disc_optimizer, net_disc = discriminator)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if training:
        i = 0
        i_val = 0
        generator_loss = []
        validation_gen_loss = []
        discriminator_loss = []
        validation_disc_loss = []
        generator_mse = []
        validation_gen_mse = []
        for epoch in range(0, epochs):
            ##################################
            ###Train discriminator
            ##################################
            for data in train_dataset:
                input, output = data
                #generator_batch, u_array = generator(input)

            ##################################
            ###Train generator
            ##################################
            for data in train_dataset:

                input, output = data
                if i_val % 100 != 0:
                    #loss, result, u_array = compute_gradients_and_perform_update_step(
                    #    model, optimizer, input, output, loss_fn)
                    #input, generator, discriminator, gen_optimizer = tf.keras.optimizers.Adam(0.01), disc_optimizer = tf.keras.optimizers.Adam(0.01), gen_loss = generator_loss, disc_loss = discriminator_loss):
                    gen_loss, disc_loss, result, u_array, gen_mse = compute_gradients_and_perform_update_step_gradient_descent(generator, regularizer, input, output, loss_fn_reg, optimizer)
                                                                        #generator, regularizer, optimizer, inp, output, loss_fn = discriminator_loss):
                    wanted_output = output
                    #aggregated_loss.append((loss.numpy(), int(ckpt.step)))
                    generator_loss.append((gen_loss.numpy(), int(ckpt.step)))
                    discriminator_loss.append((disc_loss.numpy(), int(ckpt.step)))
                    generator_mse.append((gen_mse.numpy(), int(ckpt.step)))

                    if int(ckpt.step) % checkpoints_output_steps == 0:
                        save_path = manager.save()
                    if i % loss_output_steps == 0:
                        print('generator loss: {}, generator mse: {}, discriminator loss: {}'.format(gen_loss.numpy(),gen_mse.numpy(), disc_loss.numpy()))
                        np.save(LossPath + '/generator_loss', generator_loss)
                        np.save(LossPath + '/discriminator_loss', discriminator_loss)
                        np.save(LossPath + '/validation_gen_loss', validation_gen_loss)
                        np.save(LossPath + '/validation_disc_loss', validation_disc_loss)
                        np.save(LossPath + '/generator_mse', generator_mse)
                        np.save(LossPath + '/validation_gen_mse', validation_gen_mse)
                    if i % images_output_steps == 0:
                        for k in range(0, wanted_output.shape[3]):
                            plt.figure()
                            plt.imshow(wanted_output.numpy()[0, :, :, k])
                            plt.colorbar()
                            plt.savefig(ImagesPath + '/wanted_output_{}_{}'.format(int(ckpt.step),k))

                            plt.close()

                            plt.figure()
                            plt.imshow(result.numpy()[0, :, :, k])
                            plt.colorbar()
                            plt.savefig(ImagesPath + '/output_{}_{}'.format(int(ckpt.step),k))

                            plt.close()

                        if print_propagation_images:
                            #u_array = generator.get_propagation()
                            for img in range(0, len(u_array)):
                                #print(u_array[img])
                                #print(u_array[img].numpy())
                                plt.figure()
                                plt.imshow(np.angle(u_array[img].numpy()[0, :, :, 0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath + '/propagation_{}_{}'.format(int(ckpt.step), img))
                                plt.close()
                                plt.figure()
                                plt.imshow(np.abs(u_array[img].numpy()[0,:,:,0]))
                                plt.colorbar()
                                plt.savefig(ImagesPath+ '/result_abs_{}_{}'.format(int(ckpt.step),img))
                                plt.close()

                    ckpt.step.assign_add(1)
                    i = i + 1
                else:
                    val_generated_images, u_array = generator(input)
                    val_generated_images = tf.math.abs(val_generated_images)
                    val_disc_real_out = discriminator(input)
                    val_disc_fake_out = discriminator(val_generated_images)

                    val_gen_loss = loss_fn_generator(val_disc_fake_out, val_generated_images, input, alpha = alpha)
                    val_disc_loss = loss_fn_discriminator(val_disc_real_out, val_disc_fake_out)
                    val_gen_mse = mse(val_generated_images, input)
                    #print(val_gen_loss)
                    #print(val_disc_loss)
                    validation_gen_loss.append((float(val_gen_loss), int(ckpt.step)))
                    validation_disc_loss.append((float(val_disc_loss), int(ckpt.step)))
                    validation_gen_mse.append((float(val_gen_mse), int(ckpt.step)))

                i_val = i_val + 1



        ##########SAVE LOSSES
        np.save(LossPath + '/aggregated_loss', aggregated_loss)
        np.save(LossPath + '/validation_loss', validation_loss)
    else:
        #################################################### Testing
        aggregated_loss = []
        validation_loss = []
        correct_count = 0
        all_count = 0
        for data in test_dataset:
            image, number = data
            res, u_array = model(image)
            #accuracy, count = PhaseplateNet.get_accuracy(res, number, 2)
            correct_count += count
            all_count += batch_size
        print('Test dataset had an accuracy of {}'.format(correct_count / all_count))
        np.save(LossPath + '/accuracy_test', (correct_count / all_count))

    return aggregated_loss, validation_loss



def train_projector_lbfgs(model, train_dataset, loss_fn = mse, epochs = 10, PATH = '.', print_propagation_images = True ):


    warnings.simplefilter('error', UserWarning)
    ImagesPath = PATH + '/Images'
    CheckpointsPath = PATH + '/chkpts'
    LossPath = PATH + '/Losses'

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, CheckpointsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)


    try:
        os.mkdir(PATH)
    except FileExistsError:
        print('Directory {} already exists'.format(PATH))
    try:
        os.mkdir(CheckpointsPath)
    except FileExistsError:
        print('Directory {} already exists'.format(CheckpointsPath))
    try:
        os.mkdir(LossPath)
    except FileExistsError:
        print('Directory {} already exists'.format(LossPath))
    try:
        os.mkdir(ImagesPath)
    except FileExistsError:
        print('Directory {} already exists'.format(ImagesPath))

    try:
        f = open(PATH + '/network_parameters.txt', 'w+')
        f.write(model.get_network_hyperparameters())
        f.close()
    except:
        print('Could not save network hyperparamters')

    it = iter(train_dataset)
    inp, wanted_out = next(it)


    #shapes = [x.shape for x in model.trainable_variables]
    shapes =model.get_phaseshift_shape()
    with tf.GradientTape() as g:
        vars = tf.zeros(shapes)
        g.watch(vars)
        test_output = model(tf.cast(inp,dtype = tf.complex64),vars)
        loss = loss_fn( test_output, wanted_out)
    grad = g.gradient(loss, vars)

    #ii = 0

    def model_call( model, loss_fn, input, wanted_output,shapes, x):
        #shapes = [x.shape for x in model.trainable_variables]
        #shapes = tf.map_fn(fn = lambda x: x.shape, elems = tf.constant(model.trainable_variables))
        #print(shapes)
        '''
        list = to_list(x, shapes)

        #print(list)
        assert len(list) == len(model.trainable_variables)
        #print(x)
        #assert x.shape[2] == len(model.trainable_variables)
        #print(x.shape)
        for i in range(0, len( model.trainable_variables)):
            #model.trainable_variables[i].assign(tf.reshape(tf.slice(x, (0,0,i), (x.shape[0], x.shape[1],1)),(x.shape[0], x.shape[1])) )
            model.trainable_variables[i].assign( list[i])

        output,u_array = model(input)
        loss = loss_fn(output, wanted_output)
        '''
        #x = tf.stack(to_list(x,shapes))
        x = tf.reshape(x, shapes)
        output = model.call(tf.cast(input, dtype = tf.complex64), x)
        loss = loss_fn(output,wanted_output)

        print('{}, {}'.format(ckpt.step.numpy(), loss.numpy()))
        ckpt.step.assign_add(1)
        #TODO: put sensible condition here
        if ckpt.step.numpy() % 100 == 0:

            for batch in range(0, 4):
                for k in range(0, wanted_output.shape[3]):
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    im1 = ax1.imshow(np.abs(wanted_output.numpy()[batch, :, :, k]))
                    fig.colorbar(im1, ax=ax1)

                    im2 = ax2.imshow(np.angle(wanted_output.numpy()[batch, :, :, k]))
                    fig.colorbar(im2, ax=ax2)
                    fig.savefig(
                        ImagesPath + '/wanted_output_batch_abs:{}_step:{}_channel{}'.format(batch, int(ckpt.step), k))
                    plt.close()

                    # print(result.numpy())
                    # print(np.sum(np.isnan(result.numpy())))
                    fig, (ax1, ax2) = plt.subplots(1, 2)

                    im1 = ax1.imshow(np.abs(output.numpy()[batch, :, :, k]))
                    fig.colorbar(im1, ax=ax1)
                    im2 = ax2.imshow(np.angle(output.numpy()[batch, :, :, k]))
                    fig.colorbar(im2, ax=ax2)

                    fig.savefig(ImagesPath + '/output_batch:{}_step:{}_channel{}'.format(batch, int(ckpt.step), k))
                    plt.close()


        return loss

    def loss_and_gradient(x):

        ##def l(x):
         #   return model_call( model,loss_fn, inp,wanted_out,x)

        #loss, grad = tfp.math.value_and_gradient( l, x)
        # return loss, grad
        '''
        with tf.GradientTape() as g:
            g.watch(x)
            loss = model_call(model, loss_fn, inp,wanted_out, shapes, x)
        gradient = g.gradient(loss,model.trainable_variables)
        #gradient = tf.transpose( tf.stack(gradient), (1,2,0))
        gradient = to_vector(gradient)
        #print(tf.reduce_sum(gradient))

        return loss, gradient
        '''

        return tfp.math.value_and_gradient(
            lambda y: model_call( model, loss_fn, inp, wanted_out,shapes,y),
            x
        )


    #start = to_vector(model.trainable_variables)
    print(shapes)
    #start = tf.reshape(tf.zeros(shape=  shapes), shape = (shapes))
    start = tf.zeros(shape = shapes)
    start = tf.reshape( start, shape= (tf.size(start)))



    #start = model.trainable_variables
    #print(start)
    #start = tf.stack(start)
    #start = tf.transpose(start, (1,2,0))
    #print(start)
    #print(epochs)
    #print('model_call')
    #print(model_call(model, loss_fn, inp, wanted_out,start))
    #print('loss')
    #print(loss_and_gradient(start))
    #print('grad')
    #print(quadratic_loss_and_gradient(start))
    print(epochs)
    optim_results = tfp.optimizer.lbfgs_minimize(
        loss_and_gradient, initial_position = start,tolerance = 0,x_tolerance= 0, f_relative_tolerance=0, max_iterations = epochs, stopping_condition= lambda x,y: x, max_line_search_iterations= 1000000
    )
    print(optim_results.position)
    print(optim_results.objective_value)
    model_call(model, loss_fn, inp, wanted_out, shapes, optim_results.position)
    print(optim_results.converged)
    print(optim_results.failed)

    #if print_propagation_images:
    save_path = manager.save()

    return optim_results



