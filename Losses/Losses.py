import tensorflow as tf
from PhaseplateNetwork.utils.data_utils import get_scaled_hole_pos, get_activation_images_from_number, get_activation_images

@tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.backend.binary_crossentropy(tf.ones_like(real_output), real_output, False)
    fake_loss = tf.keras.backend.binary_crossentropy(tf.zeros_like(fake_output), fake_output, False)
    total_loss = real_loss + fake_loss
    return tf.math.reduce_mean(total_loss)

#@tf.function
def generator_loss(fake_output, generated_images, wanted_images,alpha= 1.0):
    #print(fake_output.shape)
    #print(wanted_images.shape)
    mse = mse_loss(generated_images, wanted_images)
    loss_from_disc = tf.math.reduce_mean(tf.keras.backend.binary_crossentropy(tf.ones_like(fake_output), fake_output, True))
    return mse + alpha * loss_from_disc



@tf.function
def complex_mse(result, wanted_output):
    abs_result = tf.math.abs(result)
    phase_result = tf.math.angle(result)
    abs_output = tf.math.abs(wanted_output)
    phase_output = tf.math.angle(wanted_output)
    return tf.reduce_mean( ((abs_result - abs_output)**2 + (phase_result - phase_output)**2)/2)

@tf.function
def normalized_complex_mse(result,wanted_output):
    abs_result = tf.math.abs(result)
    phase_result = tf.math.angle(result)
    abs_output = tf.math.abs(wanted_output)
    phase_output = tf.math.abs(wanted_output)
    abs_result_normalized = abs_result/tf.reshape(tf.math.reduce_max(abs_result, (1,2)), ( abs_result.shape[0], 1,1, abs_result.shape[3])) * 1.0
    abs_output_normalized = abs_output/tf.reshape(tf.math.reduce_max(abs_output, (1,2)), (abs_output.shape[0], 1, 1, abs_output.shape[3])) * 1.0

    return tf.reduce_mean( ((abs_result_normalized - abs_output_normalized)**2 + (phase_result - phase_output)**2)/2)


@tf.function
def crop(result,wanted_output):
    if result.shape[1] != wanted_output.shape[1] or result.shape[2] != wanted_output.shape[2]:
        difference_x = result.shape[1] - wanted_output.shape[1]
        difference_y = result.shape[2] - wanted_output.shape[2]
        result = tf.image.crop_to_bounding_box(result, int(difference_x/2), int(difference_y/2), wanted_output.shape[1], wanted_output.shape[2])
    return result, wanted_output

@tf.function
def cropped_complex_mse(result,wanted_output):
    result,wanted_output = crop(result,wanted_output)

    return complex_mse(result,wanted_output)

@tf.function
def cropped_normalized_complex_mse(result,wanted_output):
    result,wanted_output = crop(result,wanted_output)
    return normalized_complex_mse(result,wanted_output)

@tf.function
def complex_mse_normalized(result, wanted_output):
    abs_result = tf.image.per_image_standardization(tf.abs(result))
    phase_result = tf.math.angle(result)
    abs_output = tf.image.per_image_standardization(tf.abs(wanted_output))
    phase_output = tf.math.angle(wanted_output)

    return tf.reduce_mean( ((abs_result - abs_output)**2 + (phase_result - phase_output)**2)/2.0)


@tf.function
def mse_loss_per_image_standardized(result, wanted_output):
    result = tf.math.abs(result)
    wanted_output = tf.abs(wanted_output)
    return tf.reduce_mean((tf.image.per_image_standardization(result) - tf.image.per_image_standardization(wanted_output))**(2))

@tf.function
def normalized_mse(result,wanted_output):
    result = tf.math.abs(result)
    wanted_output = tf.math.abs(wanted_output)
    result = result/tf.reshape(tf.math.reduce_max(result, (1,2)), ( -1, 1,1, result.shape[3])) * 1.0
    wanted_output = wanted_output/tf.reshape(tf.math.reduce_max(wanted_output, (1,2)), (-1, 1, 1, wanted_output.shape[3])) * 1.0
    return tf.reduce_mean((result - wanted_output)**2)

@tf.function
def acc(inp1, inp2):
    inp1 = tf.math.abs(inp1)
    inp2 = tf.math.abs(inp2)
    mul = tf.reduce_sum(inp1*inp2, (1,2,3))
    res = mul / tf.sqrt(tf.reduce_sum(inp1**2, (1,2,3)) * tf.reduce_sum(inp2**2,(1,2,3)))
    return tf.reduce_mean(1.0-res)

@tf.function
def dssim(inp1, inp2):
    inp1 = tf.math.abs(inp1)
    inp2 = tf.math.abs(inp2)
    inp1 = inp1/tf.reshape(tf.math.reduce_max(inp1, (1,2)), ( inp1.shape[0], 1,1, inp1.shape[3])) * 255.0
    inp2 = inp2/tf.reshape(tf.math.reduce_max(inp2, (1,2)), (inp2.shape[0], 1, 1, inp2.shape[3])) * 255.0
    return tf.reduce_mean(1-(tf.image.ssim(inp1,inp2,255.0))/2)

@tf.function
def ssim(inp1, inp2):
    inp1 = tf.math.abs(inp1)
    inp2 = tf.math.abs(inp2)
    inp1 = inp1/tf.reshape(tf.math.reduce_max(inp1, (1,2)), ( inp1.shape[0], 1,1, inp1.shape[3])) * 255.0
    inp2 = inp2/tf.reshape(tf.math.reduce_max(inp2, (1,2)), (inp2.shape[0], 1, 1, inp2.shape[3])) * 255.0
    return tf.reduce_mean(tf.image.ssim(inp1,inp2,255.0))

@tf.function
def mse(result, wanted_output):
    if result.dtype == tf.complex64:
        result = tf.math.abs(result)
    if wanted_output.dtype == tf.complex64:
        wanted_output = tf.abs(wanted_output)
    return tf.reduce_mean((result - wanted_output)**2)






@tf.function
def mse_cropped(result,wanted_output):
    result,wanted_output = crop(result,wanted_output)
    return mse(result,wanted_output)



@tf.function
def mse_only_standardized(result,wanted_output):
    result = tf.math.abs(result)
    wanted_output = tf.abs(wanted_output)
    res_N = tf.cast(result.shape[1]*result.shape[2], tf.float32)
    wanted_N = tf.cast(wanted_output.shape[1]*wanted_output.shape[2], tf.float32)
    result = result/ tf.math.maximum( tf.math.reduce_std( result, axis = ( 1,2), keepdims = True), 1/tf.math.sqrt(res_N))
    wanted_output = wanted_output/ tf.math.maximum( tf.math.reduce_std( wanted_output, axis = ( 1,2), keepdims = True), 1/tf.math.sqrt(wanted_N))
    return tf.reduce_mean((result - wanted_output)**2)

@tf.function
def TV_loss(result,wanted_output):
    return tf.reduce_mean(tf.image.total_variation(result-wanted_output))

@tf.function
def softmax_crossent(result,wanted_output):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(wanted_output, result)

@tf.function
def weight_ortogonalization(weights):
    '''
    loss function for orthonormal weight regularization after https://arxiv.org/abs/1901.11352
    Deep learning for Inverse Problems: Bounds and Regularizers, J. Amjad, Z.Lyu, M. Rodriges.
    weights: List of network weights to be regularized over (can only handle dimensions 2 and 4, for dense and convolutional layers)
    '''
    loss = tf.constant(0.0)

    for weight in weights:
        if len(weight.get_shape()) == 2:
            loss = loss + tf.reduce_sum(tf.abs(tf.linalg.matmul(weight,weight, transpose_a = True) - tf.eye(weight.shape[1], weight.shape[1])))

        elif len(weight.get_shape()) == 4: #handle convolutions
            # weights should come in as w x h x inp x out and need to be out x inp*w*h according to
            # https://arxiv.org/pdf/1705.10941.pdf Y. Yoshida and T. Miyato, Spectral Norm Regulariztion for improving the generalizability of deep learning
            # and
            # Deep learning for Inverse Problems: Bounds and Regularizers, J. Amjad, Z.Lyu, M. Rodriges.
            weight = tf.reshape(tf.transpose(weight, (3,0,1,2)), (weight.shape[3], weight.shape[0]*weight.shape[1]*weight.shape[2]))
            loss = loss + tf.reduce_sum(
                tf.abs(tf.linalg.matmul(weight, weight, transpose_a=True) - 0.1*tf.eye(weight.shape[1], weight.shape[1])))
    return loss

@tf.function
def get_classification_vector(image,class_images):
    '''
    Get the classification vector of an image. The classification positions are given by p_arr of shape [num, 2]
    :param image: classification image [batch, N,M, c]
    :param class_images: classification images of shape [num, N,M, c] (should be binary, 1 or 0 on positions where the position should be counted
    '''
    res = []
    #print(image.shape)
    #print(class_images.shape)
    assert (image.shape[1] == class_images.shape[1])
    assert (image.shape[2] == class_images.shape[2])
    assert (image.shape[3] == class_images.shape[3])
    for i in range(0,class_images.shape[0]):
        res.append([ tf.reduce_sum( image * class_images[i,:,:,:], axis = (1,2,3))])
    #print(res)
    res = tf.concat(res, axis = 0)
    res = tf.transpose(res, (1,0))
    #print(res.shape)
    return res

import matplotlib.pyplot as plt
def image_softmax_loss( image, wanted_image, class_images):
    if image.dtype == tf.complex64:
        image = tf.math.abs(image)
    if wanted_image.dtype == tf.complex64:
        wanted_image = tf.math.abs(wanted_image)
    class_vec = get_classification_vector(image,class_images)
    wanted_vec = get_classification_vector(wanted_image, class_images)
    '''
    print(class_vec[0])
    print(wanted_vec[0])
    plt.imshow(image[0,:,:,0])
    plt.figure()
    plt.imshow(wanted_image[0,:,:,0])
    plt.show()
    '''
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(wanted_vec, class_vec))# + 1/tf.math.reduce_sum(class_vec)
    loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy( tf.math.softmax(wanted_vec), tf.math.softmax( class_vec), from_logits = True))

    #print(loss)
    return loss

def image_detector_mse(image, wanted_image, output_area):
    if image.dtype == tf.complex64:
        image = tf.math.abs(image)
    if wanted_image.dtype == tf.complex64:
        wanted_image = wanted_image
    proj_img = image * output_area
    return mse(proj_img, wanted_image)

def image_detector_normalized_mse(image, wanted_image, output_area):
    if image.dtype == tf.complex64:
        image = tf.math.abs(image)
    if wanted_image.dtype == tf.complex64:
        wanted_image =  tf.math.abs(wanted_image)
    proj_img = image * output_area
    return normalized_mse(proj_img, wanted_image)


def image_mse_loss(image, wanted_vector,L, radius):
    if image.dtype == tf.complex64:
        image = tf.math.abs(image)
    assert image.shape[1] == image.shape[2]
    print(L)
    print(image.shape[1])
    act_imgs = tf.expand_dims(tf.cast(get_activation_images(wanted_vector, image.shape[1],L, radius), tf.float32), axis = 3)
    return mse(image, act_imgs)

def mnist_softmax_accuracy_metric(output, wanted_output, class_images):
    if output.dtype == tf.complex64:
        output = tf.math.abs(output)
    class_vec = get_classification_vector(output, class_images)
    wanted_output_vec = get_classification_vector(wanted_output, class_images)

    #tf.keras.metrics.Accuracy(class_vec, wanted_output_ve

