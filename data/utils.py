from torch.nn.utils.rnn import pad_sequence
import torch

def pad_bounding_boxes(bboxes, padding_value=0.0):
    
    """
    Pad a list of bounding boxes to the same length.
    
    Args:
        bboxes (list of Tensors): List of bounding boxes, each of shape (?, 4).
        padding_value (float): Value to use for padding.
        
    Returns:
        Tensor: Padded bounding boxes of shape (Bsz, max_length, 4).
        Tensor: Mask indicating valid bounding boxes.

    Objective : to remove nested form bounding boxes.
    """
    # Convert list of Tensors to a list of padded Tensors
    padded_bboxes = pad_sequence(bboxes, batch_first=True, padding_value=padding_value)
    
    # Create a mask to indicate valid bounding boxes
    mask = torch.zeros_like(padded_bboxes[..., 0], dtype=torch.bool)
    for i, bbox in enumerate(bboxes):
        mask[i, :bbox.size(0)] = 1
    
    return padded_bboxes, mask