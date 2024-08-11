import torch

def intersect_over_union(box_preds, box_labels, format='midpoint'):
    
    if format == 'corners':
        box1_x1 = box_preds[..., 0:1]
        box1_y1 = box_preds[..., 1:2]
        box1_x2 = box_preds[..., 2:3]
        box1_y2 = box_preds[..., 3:4] 

        box2_x1 = box_labels[..., 0:1]
        box2_y1 = box_labels[..., 1:2]
        box2_x2 = box_labels[..., 2:3]
        box2_y2 = box_labels[..., 3:4] 

    if format == 'midpoint':
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3]/2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4]/2 
        box1_x2 = box_preds[..., 0:1] + box_preds[..., 2:3]/2
        box1_y2 = box_preds[..., 1:2] + box_preds[..., 3:4]/2

        box2_x1 = box_labels[..., 0:1] - box_labels[..., 2:3]/2
        box2_y1 = box_labels[..., 1:2] - box_labels[..., 3:4]/2 
        box2_x2 = box_labels[..., 0:1] + box_labels[..., 2:3]/2
        box2_y2 = box_labels[..., 1:2] + box_labels[..., 3:4]/2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    union = box1_area + box2_area - intersection

    return intersection/(union + 1e-6)