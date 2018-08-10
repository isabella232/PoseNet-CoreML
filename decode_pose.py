from keypoints import part_names, pose_chain, part_name_to_id_map
from utils import get_image_coords, clamp


parent_children_id_tuples = [
    (part_name_to_id_map[parent_joint_name], part_name_to_id_map[child_joint_name])
    for parent_joint_name, child_joint_name in pose_chain
]
parent_to_child_edges = tuple([t[1] for t in parent_children_id_tuples])
child_to_parent_edges = tuple([t[0] for t in parent_children_id_tuples])



def decode_pose(root, heatmap_scores, offsets, output_stride, displacements_fwd, displacements_bwd):
    num_parts = heatmap_scores.shape[2]
    num_edges = len(parent_to_child_edges)
    instance_keypoints = []

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
            instanceKeypoints[target_keypoint_id] = traverse_to_target_keypoint(
                edge, instance_keypoints[source_keypoint_id], target_keypoint_id, scores, offsets, output_stride, displacements_bwd
            )


def traverse_to_target_keypoint(edge_id, source_keypoint, target_keypoint_id, scores_buffer,
                                offsets, output_stride, displacements):
        
    height, width = scores.shape

    # Nearest neighbor interpolation for the source->target displacements.
    source_keypoint_indices = get_strided_index_near_point(
        source_keypoint['position'], output_stride, height, width
    )

    displacement = get_displacement(edge_id, sourceKeypoint_indices, displacements);

    const displacedPoint = addVectors(sourceKeypoint.position, displacement);

    const displacedPointIndices =
        getStridedIndexNearPoint(displacedPoint, outputStride, height, width);

    const offsetPoint = getOffsetPoint(
        displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
        offsets);

    const score = scoresBuffer.get(
        displacedPointIndices.y, displacedPointIndices.x, targetKeypointId);

    const targetKeypoint = addVectors(
        {
          x: displacedPointIndices.x * outputStride,
          y: displacedPointIndices.y * outputStride
        },
        {x: offsetPoint.x, y: offsetPoint.y});

    return {position: targetKeypoint, part: partNames[targetKeypointId], score};
}  


def get_strided_index_near_point(point, output_stride, height, width):
    return {
        'y': clamp(round(point['y'] / output_stride), 0, height - 1),
        'x': clamp(round(point['x'] / output_stride), 0, width - 1)
    }
