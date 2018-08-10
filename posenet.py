import json
import time
import struct
import tensorflow as tf
import cv2
import numpy as np
import os
import heapq
import yaml

from utils import get_image_coords, get_offset_point, within_radius_of_corresponding_point
from decode_pose import decode_pose


# GLOBALS
MANIFEST_FILENAME = "manifest.json"
CONFIG_PATH = "converter/config.yaml"
WEIGHTS_PATH = "converter/waits"
INPUT_IMAGE_PATH = "converter/images/tennis_in_crowd.jpg"
OUTPUT_IMAGE_PATH = "output.jpg"

K_LOCAL_MAXIMUM_RADIUS = 1  # TODO: WHAT? They use 1 in implementation but not in paper
SCORE_THRESHOLD = 0.5
NMS_RADIUS = 20  # TODO: Not used?? the same as K_LOCAL_MAXIMUM_RADIUS?
OUTPUT_STRIDE = 16
MAX_POSE_DETECTIONS = 5  # TODO: Increase!


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


def decode_multiple_poses(heatmap_scores, offsets, displacements_fwd, displacements_bwd,
                          output_stride, max_pose_detections, score_threshold=0.5, nms_radius=20):
    poses = []
    queue = build_part_with_score_queue(score_threshold, K_LOCAL_MAXIMUM_RADIUS, heatmap_scores)
    squared_nms_radius = nms_radius ** 2

    while len(poses) < max_pose_detections and len(queue) != 0:
        root = heapq.heappop(queue)
        root_image_coords = get_image_coords(root[1], output_stride, offsets)

        if within_radius_of_corresponding_point(poses, squared_nms_radius,
                                                root_image_coords, root[1]['keypoint_id']):
            continue

        keypoints = decode_pose(root, heatmap_scores, offsets, output_stride,
                                displacements_fwd, displacements_bwd)


def build_part_with_score_queue(score_threshold, local_max_radius, heatmap_scores):
    height, width, num_keypoints = heatmap_scores.shape
    queue = []  # We'll use a reversed heapq to implement a max heap
    for heatmap_y in range(height):
        for heatmap_x in range(width):
            for keypoint_id in range(num_keypoints):
                score = heatmap_scores[heatmap_y, heatmap_x, keypoint_id]  # TODO check index order

                # Only consider parts with score greater or equal to threshold as root candidates.
                if score < score_threshold:
                    continue

                # Only consider keypoints whose score is maximum in a local window.
                if score_is_maximum_in_local_window(keypoint_id, score, heatmap_y, heatmap_x,
                                                    local_max_radius, heatmap_scores):
                    # For some reason python only allows min heaps (not max heaps)
                    # so I negate the score 🤮
                    heapq.heappush(
                        queue,
                        (-score, {'heatmap_y': heatmap_y, 'heatmap_x': heatmap_x, 'keypoint_id': keypoint_id})
                    )
    return queue


def score_is_maximum_in_local_window(keypoint_id, score, heatmap_y, heatmap_x,
                                     local_max_radius, heatmap_scores):
    # TODO: I could easily vectorize this whole function.
    #       Don't know if it will be faster, due to break saving me iterations though.
    height, width, _ = heatmap_scores.shape  # We recieve a single layer

    local_maximum = True
    y_start = max(heatmap_y - local_max_radius, 0)
    y_end = min(heatmap_y + local_max_radius + 1, height)
    for y_current in range(y_start, y_end):
        # TODO: these x_start, x_end, definitions should be defined outside of this loop right?
        x_start = max(heatmap_x - local_max_radius, 0)
        x_end = min(heatmap_x + local_max_radius + 1, width)
        for x_current in range(x_start, x_end):
            if heatmap_scores[y_current, x_current, keypoint_id] > score:
                local_maximum = False
                break
        if not local_maximum:
            break
    return local_maximum


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


# Load weights from harddrive into `variables` list
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
displacements_fwd = convToOutput(x, 'displacement_fwd_2')
displacements_bwd = convToOutput(x, 'displacement_bwd_2')
heatmaps = tf.sigmoid(heatmaps,'heatmap')

# Define init operations
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialize network
    sess.run(init)

    # Format input
    input_image = read_imgfile(INPUT_IMAGE_PATH, width, height)
    input_image = np.array(input_image, dtype=np.float32)
    input_image = input_image.reshape(1, width, height, 3)

    # Run input through model
    heatmaps_result, offsets_result, displacements_fwd_result, displacements_bwd_result = sess.run(
        [heatmaps, offsets, displacements_fwd, displacements_bwd], feed_dict={image: input_image})

    # Generate poses from model's output
    poses = decode_multiple_poses(
        heatmaps_result[0], offsets_result[0], displacements_fwd_result[0],
        displacements_bwd_result[0], OUTPUT_STRIDE, MAX_POSE_DETECTIONS, SCORE_THRESHOLD, NMS_RADIUS
    )

    # Draw mask
    heatmaps_img = (heatmaps_result * 255).astype(np.uint8)[:, :, :, 1][0]
    res = cv2.resize(heatmaps_img, tuple(image.shape[1:3].as_list()), interpolation = cv2.INTER_CUBIC)
    draw_image = cv2.imread(INPUT_IMAGE_PATH)
    cv2.imshow('image', (0.7 * np.expand_dims(res, axis=2) + 0.3 * draw_image).astype(np.uint8))
    cv2.waitKey(0)
