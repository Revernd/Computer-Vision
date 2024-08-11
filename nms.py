import torch
from IOU import intersect_over_union

def non_max_suppression(bboxes, 
                        iou_threshold, 
                        prob_threshold,
                        format='corners'):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key= lambda x : x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop()

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersect_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                format = format
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms