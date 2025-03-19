from visual_grounding_model import VisualGroundingModel
from PIL import Image
import torch
if __name__ == "__main__":
    """
    Inference Scenario
    """
    model = VisualGroundingModel()
    model.load_state_dict(torch.load('path/to/model.pth'))
    model.eval()
    # context examples
    context_examples = [
        {
            'image': Image.open('path/to/image1.jpg'),
            'text': 'text1',
            'bboxes': [bbox1, bbox2, ...]
        }
    ]

    image = Image.open('path/to/image.jpg')
    text_prompt = 'text2'
    bboxes = [bbox3, bbox4, ...]

    # inference
    with torch.no_grad():
        predicted_bboxes, generated_texts = model.predict_with_context(image, text_prompt, context_examples, bboxes)

    # print results
    print(predicted_bboxes)
    print(generated_texts)

    # visualize results    
