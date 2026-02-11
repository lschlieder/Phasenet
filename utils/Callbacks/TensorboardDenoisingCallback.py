import tensorflow as tf
from PhaseplateNetwork.utils.plotting_utils import plot_to_image
import matplotlib.pyplot as plt



class TensorboardDenoisingCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, PATH= '.', images_output_steps = 100, loss_output_steps = 100, print_propagation_images = True):
        super(TensorboardDenoisingCallback, self).__init__()        
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
            #self.output_function(result, self.wanted_output, self.input, self.model, self.ImagesPath, int(self.ckpt.step))
            output_comparison_images = []
            for i in range(0,result.shape[0]):
                fig, ax = plt.subplots(1,3, figsize = (15,5))

                img = ax[0].imshow(self.input[0][i,:,:,0])
                ax[0].set_title('Input')
                fig.colorbar(img, ax = ax[0])
                img = ax[1].imshow(result[i,:,:,0])
                ax[1].text(-5,-5, '{}/{}'.format(self.input[1][i,0], self.model.timesteps))
                ax[1].set_title('Output')
                fig.colorbar(img, ax= ax[1])

                wanted_img = ax[2].imshow(self.wanted_output[i,:,:,0])
                ax[2].set_title('Wanted Output')
                fig.colorbar(wanted_img, ax = ax[2])
                output_comparison_images.append(plot_to_image(fig))
                plt.close(fig)
            

            
            model_images = self.model.get_image_variables()
            num_plates = len(model_images)
            fig, ax = plt.subplots(1,num_plates, figsize = (5*num_plates, 5))
            if num_plates == 1:
                ax = [ax]
            for i,a in enumerate(ax):
                img = a.imshow(model_images[i])
                fig.colorbar(img, ax = a)

            pplate_image = plot_to_image(fig)
            plt.close(fig)

            with self.file_writer_images.as_default():
                tf.summary.image('Phase Plates', pplate_image, step = batch)
                for i in range(0,len(output_comparison_images)):
                    tf.summary.image('Output Comparison, batch: {}'.format(i),output_comparison_images[i], step = batch)

        if batch % self.loss_output_steps == 0 :
            with self.file_writer_metrics.as_default():
                for name, item in logs.items():
                    tf.summary.scalar(name, item, step = batch)

            self.model.reset_metrics()

        return



        #self.ckpt.step.assign_add(1)

