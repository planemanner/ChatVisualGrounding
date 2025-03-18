# Visual Grounding with Vision-Language Model

This implementation provides a vision-language model for visual grounding tasks, where the model predicts bounding box coordinates for objects described in natural language.
# Note
- This repository is not fully implemented and is not ready to use.
  - 아직 완전히 구현 되지 않았습니다 !
- Also, this README page also is not providing proper information to use.
## Features

- Uses a generative approach with special tokens for bounding box representation
- Supports both training and inference modes
- Includes visualization utilities
- Built on top of popular transformer models

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from visual_grounding_model import VisualGroundingModel
from PIL import Image

# Initialize the model
model = VisualGroundingModel()

# Make a prediction
image_path = "path/to/your/image.jpg"
text_prompt = "Find the red car in the image"
predicted_bbox = model.predict(image_path, text_prompt)
```

### Training Mode

```python
# Prepare your data
images = [image1, image2, ...]  # List of PIL Images
text_prompts = ["Find the red car", "Locate the blue chair", ...]
target_bboxes = [[x1, y1, x2, y2], ...]  # List of bounding box coordinates

# Forward pass for training
loss = model(images, text_prompts, target_bboxes)
```

### Visualization

```python
from example import visualize_grounding

# Visualize the results
visualize_grounding(image_path, text_prompt, predicted_bbox)
```

## Model Architecture

The model uses:
- A vision encoder (CLIP) for processing images
- A language model (GPT-2) for text generation
- Special tokens `<box>` and `</box>` to represent bounding box coordinates
- The bounding box coordinates are represented as text in the format: `<box>x1,y1,x2,y2</box>`

## Notes

- The bounding box coordinates should be normalized between 0 and 1
- The model expects images to be in RGB format
- Text prompts should be clear and specific for better results 

# To do
- 데이터셋, 로더 파트 구현 필요
  - Conversation form ?
