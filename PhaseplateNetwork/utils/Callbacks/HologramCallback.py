import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

class HologramTrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, PATH='.', images_output_steps=100, loss_output_steps=100,
                 checkpoints_output_steps=100,**kwargs):
        super(HologramTrainCallback,self).__init__(**kwargs)
        #self.output_function = output_function
        self.PATH = PATH
        self.images_output_steps = images_output_steps
        self.loss_output_steps = loss_output_steps
        self.checkpoints_output_steps = checkpoints_output_steps
        self.input, self.wanted_output = test_data

    def on_train_begin(self, logs=None):
        self.ImagesPath = self.PATH + '/Images'
        self.CheckpointsPath = self.PATH + '/chkpts'
        self.LossPath = self.PATH + '/Losses'

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer_model=self.model.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.CheckpointsPath, max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        try:
            os.mkdir(self.PATH)
        except FileExistsError:
            print('Directory {} already exists'.format(self.PATH))
        try:
            os.mkdir(self.CheckpointsPath)
        except FileExistsError:
            print('Directory {} already exists'.format(self.CheckpointsPath))
        try:
            os.mkdir(self.LossPath)
        except FileExistsError:
            print('Directory {} already exists'.format(self.LossPath))
        try:
            os.mkdir(self.ImagesPath)
        except FileExistsError:
            print('Directory {} already exists'.format(self.ImagesPath))

        try:
            with open(self.PATH + '/network_configuration.txt', 'w') as outfile:
                outfile.write(self.model.to_json(indent=4))
        except:
            print('Could not save network configuration')

        try:
            self.aggregated_loss = np.load(self.LossPath + '/aggregated_loss.npy')
        except:
            print('No old loss files found. Starting a new one!')
            self.aggregated_loss = np.zeros((0,2))

        try:
            self.metrics = np.load(self.LossPath + '/metrics.npy')
        except:
            print('No old metrics file found. Starting a new one!')
            #print(self.model.metrics_names)
            self.metrics = np.zeros((0))
            #print(self.metrics.shape)
            #input()

        try:
            self.validation_metrics = np.load(self.LossPath+'/validation_metrics.npy')
        except:
            print('No old validation metrics file found. Starting a new one!')
            self.validation_metrics = np.zeros((0))






    def on_train_batch_end(self, batch, logs=None):
        #print(logs[0])
        if int(self.ckpt.step) % self.checkpoints_output_steps == 0:
            save_path = self.manager.save()

        if int(self.ckpt.step) % self.images_output_steps == 0:
            result = self.model.predict(self.input)
            for i in range(0, result.shape[3]):
                plt.figure()
                plt.imshow(tf.math.abs(result[0, :, :, i]))
                plt.colorbar()
                plt.savefig(self.ImagesPath + '/output_images_{}_{}'.format(int(self.ckpt.step), i))
                plt.close()

                plt.figure()
                plt.imshow(self.wanted_output[0,:,:,i])
                plt.colorbar()
                plt.savefig(self.ImagesPath+'/wanted_output_images_{}_{}'.format(int(self.ckpt.step),i))
                plt.close()

            model_images = self.model.get_image_variables(self.input)
            for plate in range(0, len(model_images)):

                plt.figure()
                fig, ax = plt.subplots(1,2, figsize = (10,5))
                im1 = ax[0].imshow(np.abs(model_images[plate].numpy()))
                fig.colorbar(im1, ax = ax[0])
                im2 = ax[1].imshow(np.angle(model_images[plate].numpy()))
                fig.colorbar(im2, ax = ax[1])
                plt.savefig(self.ImagesPath + '/hologram_{}_{}'.format(int(self.ckpt.step), plate))
                plt.close()


        self.aggregated_loss = np.append(self.aggregated_loss, np.array([[logs['loss'], int(self.ckpt.step)]]), axis = 0)

        if int(self.ckpt.step) % self.loss_output_steps == 0:
            np.save(self.LossPath + '/aggregated_loss', self.aggregated_loss)
            temp_m = []
            for name, val in logs.items():
                temp_m.append(val)

            temp_m.append(int(self.ckpt.step))

            if self.metrics.shape[0] == 0:
                self.metrics = np.zeros((0,np.array(temp_m).shape[0]))
            self.metrics = np.concatenate((self.metrics, np.expand_dims(np.array(temp_m),0)), axis = 0)
            np.save(self.LossPath + '/metrics.npy',self.metrics)
            self.model.reset_metrics()

        self.ckpt.step.assign_add(1)