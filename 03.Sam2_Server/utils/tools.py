import numpy as np
import cv2 
import torch

# 마스크로부터 가장 큰 바운딩 박스 좌표 계산
def get_largest_bbox_from_masks(out_obj_ids, out_mask_logits, all_mask):
    
    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).byte()
        all_mask = torch.bitwise_or(all_mask, out_mask.squeeze(0))

    combined_mask = (all_mask > 0).byte().cpu().numpy().astype(np.uint8)
    coords = np.argwhere(combined_mask)
    if coords.size == 0:
        return None, None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    current_bbox = [x_min, y_min, x_max, y_max]
    
    all_mask = all_mask.cpu().numpy() * 255
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
    
    return current_bbox, all_mask