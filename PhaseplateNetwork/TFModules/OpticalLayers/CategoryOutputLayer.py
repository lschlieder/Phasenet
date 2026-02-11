import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.ImageUtils import create_hexagonal_image_stack


class CategoryOutputLayer(OpticalLayer):
    def __init__(self, categories, image_size, output_size, radius, distance = None, output_function = 'sum', orientation = 'pointy', output_shape = 'circle' ,negative_output = False, **kwargs):
        """
        Creates a new categoryOutputLayer
        This layer takes an image of size (batch_num, image_size[1], image_size[2], 1) and returns a vector of size (batch_num, categories),
        that either sum up or take the mean from output regions in a hexagonal pattern

        categories (int): number of output categories
        image_size ([x,y]): input image size
        output_size ([x,y]): size of the output region in [m]
        radius (float or [rx,ry]): radius of the output regions
        distance (float): distance of the output regions in the hex-grid
        negative_output (bool): Weather to include negative weights in the output region (doubles the output categories)
        output_function: \'sum\' or \'mean\'. determines the reduction function
        """
        super(CategoryOutputLayer, self).__init__(**kwargs)
        self.categories = categories
        if distance ==None:
            distance = radius * 2.5
        if output_function not in ['sum', 'mean']:
            raise ValueError("Invalid output function. use \"sum\" or \"mean\" ")
        self.output_fn = output_function

        self.negative_output = negative_output

        if negative_output == False:
            self.masks = create_hexagonal_image_stack( categories, image_size, output_size, radius,distance, orientation, output_shape)
        elif negative_output == True:
            self.masks = create_hexagonal_image_stack( categories*2, image_size, output_size, radius, distance, orientation, output_shape)
            self.positive_masks = self.masks[:,:,::2]
            self.negative_masks = self.masks[:,:,1::2]
        
    def call(self, input, **kwargs):
        assert(input.shape[1] == self.masks.shape[0])
        assert(input.shape[2] == self.masks.shape[1])
        assert(input.shape[3] == 1)
        
        if self.negative_output == False:

            masked_output = input*self.masks
        elif self.negative_output == True:
            masked_output = input * self.positive_masks -  input * self.negative_masks

        if self.output_fn == 'sum':
            out = tf.reduce_sum(masked_output, axis = [1,2])
        elif self.output_fn == 'mean':
            out = tf.reduce_mean(masked_output, axis = [1,2])

        return out
    
    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.categories]
    
    def get_joint_masks(self):
        return tf.reduce_sum(self.masks, axis = 2)
    
    def get_wanted_output_image(self, input_category):
        assert(input_category < self.categories)
        assert(input_category >= 0 )

        if self.negative_output:
            return self.positive_masks[:,:,input_category] + self.negative_masks[:,:,input_category]
        else:
            return self.masks[:,:,input_category]
        
    
