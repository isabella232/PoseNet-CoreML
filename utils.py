from keypoints import NUM_KEYPOINTS


def get_image_coords(part, outputStride, offsets):
    y, x = get_offset_point(part['heatmap_y'], part['heatmap_x'], part['keypoint_id'], offsets)
    return {
        'x': part['heatmap_x'] * outputStride + x,
        'y': part['heatmap_y'] * outputStride + y
    }


def get_offset_point(y, x, keypoint, offsets):
    return offsets[y, x, keypoint], offsets[y, x, keypoint + NUM_KEYPOINTS]


def within_radius_of_corresponding_point(poses, radius, keypoint_image_coords, keypoint_id):
    # TODO: possible bugs, took some liberties here
    return any([
        squared_distance(y, x, corr_point['position']['y'], corr_point['position']['x']) <= radius
        for corr_point in poses
    ])

def clamp(a, min, max):
    if a < min:
        return min
    if a > max:
        return max;
    return a
