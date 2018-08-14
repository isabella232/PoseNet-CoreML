from keypoints import NUM_KEYPOINTS


def get_image_coords(part, outputStride, offsets):
    offset_point = get_offset_point(part['heatmap_y'], part['heatmap_x'], part['keypoint_id'], offsets)
    return {
        'x': part['heatmap_x'] * outputStride + offset_point['x'],
        'y': part['heatmap_y'] * outputStride + offset_point['y']
    }


def get_offset_point(y, x, keypoint, offsets):
    return {'y': offsets[y, x, keypoint], 'x': offsets[y, x, keypoint + NUM_KEYPOINTS]}


def within_radius_of_corresponding_point(poses, radius, keypoint_image_coords, keypoint_id):
    # TODO: possible bugs, took some liberties here
    return any([
        squared_distance(
            keypoint_image_coords['y'], keypoint_image_coords['x'],
            corr_point['keypoints'][keypoint_id]['position']['y'], corr_point['keypoints'][keypoint_id]['position']['x']
        ) <= radius
        for corr_point in poses
    ])


def clamp(a, min, max):
    if a < min:
        return min
    if a > max:
        return max;
    return a


def add_vectors(a, b):
  return {'x': a['x'] + b['x'], 'y': a['y'] + b['y']}


def squared_distance(y1, x1, y2, x2):
  dy = y2 - y1
  dx = x2 - x1
  return dy * dy + dx * dx


def get_valid_resolution(image_scale_factor, input_dimension, output_stride):
    # TODO even_resolution is missnamed right?
    even_resolution = input_dimension * image_scale_factor - 1
    return even_resolution - (even_resolution % output_stride) + 1


def scale_poses(poses, scale_y, scale_x):
    if scale_x == 1 and scale_y == 1:
        return poses
    return [scale_pose(pose, scale_x, scale_y) for pose in poses]


def scale_pose(pose, scale_x, scale_y):
    return {
        'score': pose['score'],
        'keypoints': [{
            'score': kp['score'],
            'part': kp['part'],
            'position': {'x': kp['position']['x'] * scale_x, 'y': kp['position']['y'] * scale_y}
        } for kp in pose['keypoints']]
    }


def convert_to_cv2_point(position):
    return tuple([int(round(position['x'])), int(round(position['y']))])
