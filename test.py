import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from inpaint_model import InpaintCAModel

from label_detection import get_text_mask

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    # ng.get_gpus(1)
    args = parser.parse_args()

    model = InpaintCAModel()
    image = cv2.imread(args.image)
    image = cv2.resize(image, (256, 256))
    if (args.mask != ""):
        mask = cv2.imread(args.mask)
        assert image.shape == mask.shape
    
    h, w, _ = image.shape    
    
    mask = get_text_mask(args.image, h, w)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(args.output, result[0][:, :, ::-1])

    # plot original images, mask, image with with removed region, predicted image
    
    image = cv2.cvtColor(np.squeeze(image, axis=0), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(np.squeeze(mask, axis=0), cv2.COLOR_BGR2RGB)
    masked_image = cv2.subtract(image, mask)
    output = cv2.cvtColor(result[0][:, :, ::-1], cv2.COLOR_BGR2RGB)
    _, axarr = plt.subplots(1,4)
    axarr[0].set_title("Original Image")
    axarr[0].imshow(image)
    axarr[1].set_title("Mask for the text detected")
    axarr[1].imshow(mask)
    axarr[2].set_title("Image with detected text removed")
    axarr[2].imshow(masked_image)
    axarr[3].set_title("Image with removed regions filled")
    axarr[3].imshow(output)
    plt.show()
