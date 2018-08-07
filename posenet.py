import json
import time
import struct
import tensorflow as tf
import cv2
import numpy as np
import os
import yaml

# GLOBALS
MANIFEST_FILENAME = "manifest.json"
CONFIG_PATH = "converter/config.yaml"
WEIGHTS_PATH = "converter/waits"
INPUT_IMAGE_PATH = "converter/images/tennis_in_crowd.jpg"
OUTPUT_IMAGE_PATH = "output.jpg"


def toOutputStridedLayers(convolutionDefinition, outputStride):
    currentStride = 1
    rate = 1
    blockId = 0
    buff = []
    for _a in convolutionDefinition:
        convType = _a[0]
        stride = _a[1]

        if (currentStride == outputStride):
            layerStride = 1
            layerRate = rate
            rate *= stride
        else:
            layerStride = stride
            layerRate = 1
            currentStride *= stride

        buff.append({
            'blockId': blockId,
            'convType': convType,
            'stride': layerStride,
            'rate': layerRate,
            'outputStride': currentStride
        })
        blockId += 1

    return buff


def read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width,height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    return img


def convToOutput(mobileNetOutput, outputLayerName):
    w = tf.nn.conv2d(mobileNetOutput, weights(outputLayerName),[1,1,1,1],padding='SAME')
    w = tf.nn.bias_add(w,biases(outputLayerName), name=outputLayerName)
    return w

def conv(inputs, stride, blockId):
    return tf.nn.relu6(
        tf.nn.conv2d(inputs, weights("Conv2d_" + str(blockId)), stride, padding='SAME') 
        + biases("Conv2d_" + str(blockId)))


def weights(layerName):
    return variables["MobilenetV1/" + layerName + "/weights"]['x']


def biases(layerName):
    return variables["MobilenetV1/" + layerName + "/biases"]['x']


def depthwiseWeights(layerName):
    return variables["MobilenetV1/" + layerName + "/depthwise_weights"]['x']


def separableConv(inputs, stride, blockID, dilations):
    if (dilations == None):
        dilations = [1,1]
    
    dwLayer = "Conv2d_" + str(blockID) + "_depthwise"
    pwLayer = "Conv2d_" + str(blockID) + "_pointwise"
    
    w = tf.nn.depthwise_conv2d(inputs, depthwiseWeights(dwLayer), stride, 'SAME',rate=dilations, data_format='NHWC')
    w = tf.nn.bias_add(w, biases(dwLayer))
    w = tf.nn.relu6(w)

    w = tf.nn.conv2d(w, weights(pwLayer), [1,1,1,1], padding='SAME')
    w = tf.nn.bias_add(w, biases(pwLayer))
    w = tf.nn.relu6(w)

    return w


# Set up network configuration
with open(CONFIG_PATH, "r+") as f:
    cfg = yaml.load(f)
checkpoints = cfg['checkpoints']
imageSize = cfg['imageSize']
chk = cfg['chk']
outputStride = cfg['outputStride']
chkpoint = checkpoints[chk]

if chkpoint == 'mobilenet_v1_050':
    mobileNetArchitectures = cfg['mobileNet50Architecture']
elif chkpoint == 'mobilenet_v1_075':
    mobileNetArchitectures = cfg['mobileNet75Architecture']
else:
    mobileNetArchitectures = cfg['mobileNet100Architecture']

width = imageSize
height = imageSize


# Load weights into layers
with open(os.path.join(WEIGHTS_PATH, chkpoint, MANIFEST_FILENAME)) as f:
    variables = json.load(f)

for x in variables:
    filename = variables[x]["filename"]
    byte = open( os.path.join(WEIGHTS_PATH, chkpoint, filename),'rb').read()
    fmt = str (int (len(byte) / struct.calcsize('f'))) + 'f'
    d = struct.unpack(fmt, byte) 
    # d = np.array(d,dtype=np.float32)
    d = tf.cast(d, tf.float32)
    d = tf.reshape(d,variables[x]["shape"])
    variables[x]["x"] = tf.Variable(d,name=x)

image = tf.placeholder(tf.float32, shape=[1, imageSize, imageSize, 3],name='image')
x = image

# Define base network and load its weights
rate = [1,1]
layers = toOutputStridedLayers(mobileNetArchitectures, outputStride)
with tf.variable_scope(None, 'MobilenetV1'):
    for m in layers:
        stride = [1, m['stride'], m['stride'], 1]
        rate = [m['rate'], m['rate']]
        if (m['convType'] == "conv2d"):
            x = conv(x, stride, m['blockId'])
        elif (m['convType'] == "separableConv"):
            x = separableConv(x, stride, m['blockId'], rate)

# Define personlab layers and load their weights
heatmaps = convToOutput(x, 'heatmap_2')
offsets = convToOutput(x, 'offset_2')
displacementFwd = convToOutput(x, 'displacement_fwd_2')
displacementBwd = convToOutput(x, 'displacement_bwd_2')
heatmaps = tf.sigmoid(heatmaps,'heatmap')

# Define init operations
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialize network
    sess.run(init)

    # Process input
    input_image = read_imgfile(INPUT_IMAGE_PATH, width, height)
    input_image = np.array(input_image, dtype=np.float32)
    input_image = input_image.reshape(1, width, height, 3)

    # Run input through model
    heatmaps_result, offsets_result, displacementFwd_result, displacementBwd_result = sess.run(
        [heatmaps, offsets, displacementFwd, displacementBwd], feed_dict={image: input_image})
    end = time.time()
    
    # DRAW MASK
    heatmaps_img = (heatmaps_result * 255).astype(np.uint8)[:, :, :, 1][0]
    res = cv2.resize(heatmaps_img, tuple(image.shape[1:3].as_list()), interpolation = cv2.INTER_CUBIC)
    draw_image = cv2.imread(INPUT_IMAGE_PATH)
    cv2.imshow('image', (0.7 * np.expand_dims(res, axis=2) + 0.3 * draw_image).astype(np.uint8))
    cv2.waitKey(0)
