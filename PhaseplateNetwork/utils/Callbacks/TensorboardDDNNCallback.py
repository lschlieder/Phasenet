import tensorflow as tf
from PhaseplateNetwork.utils.plotting_utils import plot_to_image
import matplotlib.pyplot as plt
import numpy as np



class TensorboardDDNNCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, PATH= '.', images_output_steps = 100, loss_output_steps = 100, print_propagation_images = True):
        super(TensorboardDDNNCallback, self).__init__()        
        self.PATH = PATH
        self.images_output_steps = images_output_steps
        self.loss_output_steps = loss_output_steps
        #self.checkpoints_output_steps = checkpoints_output_steps
        #print(test_data)
        self.input, self.wanted_output = test_data

    def on_train_begin(self, logs=None):
        self.file_writer_images = tf.summary.create_file_writer(self.PATH + '/images')
        self.file_writer_metrics = tf.summary.create_file_writer(self.PATH + '/metrics')



    def on_train_batch_end(self, batch, logs=None):
        if batch % self.images_output_steps == 0:
            #result = self.model.predict(self.input)
            result = self.model.call(self.input)
            output_image = self.model.get_output_image(self.input)
            #self.output_function(result, self.wanted_output, self.input, self.model, self.ImagesPath, int(self.ckpt.step))
            output_comparison_images = []
            for i in range(0,result.shape[0]):
                
                fig, ax = plt.subplots(1,3, figsize = (15,5))
                img = ax[0].imshow(self.input[i,:,:,0])
                ax[0].set_title('Input')
                fig.colorbar(img, ax = ax[0])
                img = ax[1].imshow(output_image[i,:,:,0])
                #ax[1].text(-5,-5, '{}/{}'.format(self.input[1][i,0], self.model.timesteps))
                ax[1].set_title('Output')
                fig.colorbar(img, ax= ax[1])


                wanted_img = ax[2].imshow(self.wanted_output[i,:,:,0])
                ax[2].set_title('Wanted Output')
                fig.colorbar(wanted_img, ax = ax[2])
                output_comparison_images.append(plot_to_image(fig))
                plt.close(fig)
            

            
            model_images = self.model.get_image_variables()
            num_plates = len(model_images)
            if num_plates > 0:

                fig, ax = plt.subplots(1,num_plates, figsize = (5*num_plates, 5))
                if num_plates == 1:
                    ax = [ax]
                for i,a in enumerate(ax):
                    img = a.imshow(model_images[i])
                    fig.colorbar(img, ax = a)

                pplate_image = plot_to_image(fig)
                plt.close(fig)

            def trim_input( input, n=1):
                return input[0:n, :, :, :]
            input_fields = trim_input(self.input)
            save_fields = self.model.get_propagation_fields(input_fields)

            def get_prop_images(fields):
                ret_imgs = []
                for i in range(0, len(save_fields)):
                    fig, axs = plt.subplots(2, fields[i].shape[3], figsize=(10, 10))
                    if fields[i].shape[3] > 1:
                        for j in range(0, fields[i].shape[3]):
                        # ax1.imshow(np.abs(model_output[i,:,:,0]))
                            im1 = axs[0,j].imshow(np.abs(fields[i][0, :, :, j]))
                            axs[0,j].axis('off')
                            axs[0,j].set_title('Output (Abs)')
                            fig.colorbar(im1, ax=axs[0,j])

                            im2 = axs[1,j].imshow(np.angle(fields[i][0, :, :, j]))
                            axs[1,j].axis('off')
                            axs[1,j].set_title('Output (phase)')
                            fig.colorbar(im2, ax=axs[1,j])
                            #ret_imgs.append(plot_to_image(fig))
                            #fig.savefig(PATH + '/Propagation_field_{}.png'.format(i))

                    else:
                        im1 = axs[0].imshow(np.abs(fields[i][0, :, :, 0]))
                        axs[0].axis('off')
                        axs[0].set_title('Output (Abs)')
                        fig.colorbar(im1, ax=axs[0])

                        im2 = axs[1].imshow(np.angle(fields[i][0, :, :, 0]))
                        axs[1].axis('off')
                        axs[1].set_title('Output (phase)')
                        fig.colorbar(im2, ax=axs[1])

                    ret_imgs.append(plot_to_image(fig))
                    plt.close(fig)
                return ret_imgs

            prop_images = get_prop_images(save_fields)


            with self.file_writer_images.as_default():
                if num_plates > 0:
                    tf.summary.image('Phase Plates', pplate_image, step = batch)

                for i in range(0,len(output_comparison_images)):
                    tf.summary.image('Output Comparison, batch: {}'.format(i),output_comparison_images[i], step = batch)

                for i in range(0, len(prop_images)):
                    tf.summary.image(f'Propagation Image step {i}', prop_images[i], step = batch)

        if batch % self.loss_output_steps == 0 :
            with self.file_writer_metrics.as_default():
                for name, item in logs.items():
                    tf.summary.scalar(name, item, step = batch)

            self.model.reset_metrics()

        return



        #self.ckpt.step.assign_add(1)
