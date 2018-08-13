from keypoints import part_names, pose_chain, part_name_to_id_map
from utils import (get_image_coords, clamp, add_vectors, get_offset_point,
                   within_radius_of_corresponding_point)


parent_children_id_tuples = [
    (part_name_to_id_map[parent_joint_name], part_name_to_id_map[child_joint_name])
    for parent_joint_name, child_joint_name in pose_chain
]
parent_to_child_edges = tuple([t[1] for t in parent_children_id_tuples])
child_to_parent_edges = tuple([t[0] for t in parent_children_id_tuples])


def decode_pose(root, heatmap_scores, offsets, output_stride, displacements_fwd, displacements_bwd):
    num_parts = heatmap_scores.shape[2]
    num_edges = len(parent_to_child_edges)
    # TODO: check if this is buggy
    instance_keypoints = [None] * num_parts

    # Start the new detection instance at the position of root.
    root_score, root_part = -root[0], root[1]  # The `-` is due to python not having a max heap 🤮
    root_point = get_image_coords(root_part, output_stride, offsets)
    instance_keypoints[root_part['keypoint_id']] = {
        'score': root_score, 'part': part_names[root_part['keypoint_id']], 'position': root_point
    }

    # Decode the part positions upwards in the tree, following the backward displacements.
    # TODO: This is absolutely disgusting code, please rewrite.
    for edge in reversed(range(num_edges)):
        # TODO: have some doubts bout this code
        source_keypoint_id = parent_to_child_edges[edge]
        target_keypoint_id = child_to_parent_edges[edge]
        if instance_keypoints[source_keypoint_id] and not instance_keypoints[target_keypoint_id]:
            instance_keypoints[target_keypoint_id] = traverse_to_target_keypoint(
                edge, instance_keypoints[source_keypoint_id], target_keypoint_id, heatmap_scores,
                offsets, output_stride, displacements_bwd
            )

    for edge in range(num_edges):
        source_keypoint_id = child_to_parent_edges[edge]
        target_keypoint_id = parent_to_child_edges[edge]
        if instance_keypoints[source_keypoint_id] and not instance_keypoints[target_keypoint_id]:
            instance_keypoints[target_keypoint_id] = traverse_to_target_keypoint(
                edge, instance_keypoints[source_keypoint_id], target_keypoint_id, heatmap_scores,
                offsets, output_stride, displacements_fwd
            )

    return instance_keypoints


def traverse_to_target_keypoint(edge_id, source_keypoint, target_keypoint_id, heatmap_scores,
                                offsets, output_stride, displacements):
        
    height, width, _ = heatmap_scores.shape

    # Nearest neighbor interpolation for the source->target displacements.
    source_keypoint_indices = get_strided_index_near_point(
        source_keypoint['position'], output_stride, height, width
    )

    displacement = get_displacement(edge_id, source_keypoint_indices, displacements)

    displaced_point = add_vectors(source_keypoint['position'], displacement)

    displaced_point_indices = get_strided_index_near_point(displaced_point, output_stride,
                                                           height, width)

    offset_point = get_offset_point(displaced_point_indices['y'], displaced_point_indices['x'],
                                    target_keypoint_id, offsets)

    score = heatmap_scores[displaced_point_indices['y'],
                           displaced_point_indices['x'],
                           target_keypoint_id]

    target_keypoint = add_vectors(
        {'x': displaced_point_indices['x'] * output_stride,
         'y': displaced_point_indices['y'] * output_stride},
        offset_point  # TODO: I refactored it a bit here, check in case something fails
    )

    return {'position': target_keypoint, 'part': part_names[target_keypoint_id], 'score': score}


def get_strided_index_near_point(point, output_stride, height, width):
    # TODO: Isn't this clamp unnecesary?
    return {
        'y': int(clamp(round(point['y'] / output_stride), 0, height - 1)),
        'x': int(clamp(round(point['x'] / output_stride), 0, width - 1))
    }


def get_displacement(edge_id, point, displacements):
    num_edges = int(displacements.shape[2] / 2)  # TODO: convert to int?
    return {
        'y': displacements[point['y'], point['x'], edge_id],
        'x': displacements[point['y'], point['x'], num_edges + edge_id]
    }


def get_instance_score(existing_poses, squared_nms_radius, instance_keypoints):
    # TODO is this generated score used at all? (Maybe I added some bugs here).
    not_overlapped_keypoint_scores = 0.0
    for keypoint_id, keypoint in enumerate(instance_keypoints):
        if not within_radius_of_corresponding_point(existing_poses, squared_nms_radius,
                                                    keypoint['position'], keypoint_id):
            not_overlapped_keypoint_scores += keypoint['score']

    return not_overlapped_keypoint_scores / len(instance_keypoints)
