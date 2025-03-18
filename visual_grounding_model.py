import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
from torch.nn import functional as F

class BBoxEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, bbox):
        # bbox shape: [batch_size, num_boxes, 4] or [num_boxes, 4] or [4]
        if len(bbox.shape) == 1:
            bbox = bbox.unsqueeze(0)
        if len(bbox.shape) == 2:
            bbox = bbox.unsqueeze(0)
        return self.encoder(bbox)

class BBoxDecoder(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        
    def forward(self, features):
        # features shape: [batch_size, num_boxes, hidden_size] or [num_boxes, hidden_size] or [hidden_size]
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
        batch_size, num_boxes, _ = features.shape
        features = features.reshape(-1, features.size(-1))
        decoded = self.decoder(features)
        return decoded.reshape(batch_size, num_boxes, 4)

class VisualGroundingModel(nn.Module):
    def __init__(self, model_name="gpt2", max_length=512, hidden_size=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Add special tokens for bounding box and image
        special_tokens = {
            "additional_special_tokens": ["<box>", "</box>", "<image>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Get special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.box_token_id = self.tokenizer.convert_tokens_to_ids("<box>")
        
        # Initialize bbox encoder and decoder
        self.bbox_encoder = BBoxEncoder(hidden_size)
        self.bbox_decoder = BBoxDecoder(hidden_size)
        
        # Projection layers
        self.image_projection = nn.Linear(self.processor.config.hidden_size, self.model.config.hidden_size)
        
        self.max_length = max_length
        self.hidden_size = hidden_size
    
    def validate_box_tokens(self, text, num_boxes):
        """Validate that the number of box tokens matches the number of bounding boxes"""
        box_count = text.count("<box>")
        if box_count != num_boxes:
            raise ValueError(
                f"Mismatch between number of <box> tokens ({box_count}) "
                f"and number of bounding boxes ({num_boxes}) in text: {text}"
            )
    
    def encode_bbox(self, bboxes):
        """Convert bounding box coordinates to feature representation"""
        bbox_features = self.bbox_encoder(bboxes)
        return bbox_features
    
    def decode_bbox(self, features):
        """Convert feature representation back to bounding box coordinates"""
        return self.bbox_decoder(features)
    
    def process_image(self, image):
        """Process a single image and return its features"""
        if isinstance(image, str):
            image = Image.open(image)
        
        image_inputs = self.processor(images=[image], return_tensors="pt")
        image_features = self.processor.get_image_features(**image_inputs)
        image_features = self.image_projection(image_features)
        return image_features
    
    def process_context_examples(self, examples):
        """
        Process a list of context examples for in-context learning.
        
        Args:
            examples: List of dictionaries, each containing:
                    {'image': PIL Image or path, 'text': text prompt, 'bboxes': [box1, box2, ...]}
        
        Returns:
            Tuple of (context_text, context_image_features, context_bboxes)
        """
        context_text = []
        context_image_features = []
        context_bboxes = []
        
        for example in examples:
            # Process image
            if 'image' in example:
                img_feature = self.process_image(example['image'])
                context_image_features.append(img_feature)
            
            # Add text prompt
            if 'text' in example:
                context_text.append(example['text'])
            
            # Add bounding boxes
            if 'bboxes' in example:
                context_bboxes.extend(example['bboxes'])
        
        # Combine all text prompts with appropriate separators
        combined_text = " ".join(context_text) if context_text else ""
        
        return combined_text, context_image_features, context_bboxes
    
    def prepare_in_context_inputs(self, image, text_prompt, bboxes=None, context_examples=None):
        """
        Prepare inputs for in-context learning.
        
        Args:
            image: Current query image
            text_prompt: Current query text
            bboxes: Bounding boxes for current query
            context_examples: List of context examples
            
        Returns:
            Dictionary with prepared inputs
        """
        # Process current query image
        query_image_features = self.process_image(image)
        
        # Process context examples if provided
        context_text = ""
        context_image_features = []
        context_bboxes = []
        
        if context_examples:
            context_text, context_image_features, context_bboxes = self.process_context_examples(context_examples)
        
        # Combine context with current query
        combined_text = context_text + " " + text_prompt if context_text else text_prompt
        combined_bboxes = context_bboxes + (bboxes if bboxes is not None else [])
        
        # Create a list of all image features (context + query)
        all_image_features = context_image_features + [query_image_features]
        
        return {
            'combined_text': combined_text,
            'combined_bboxes': combined_bboxes,
            'all_image_features': all_image_features
        }
    
    def embed_inputs_with_special_tokens(self, text, input_ids, attention_mask, image_features, bboxes=None):
        """
        Embed inputs and replace special tokens with corresponding features.
        
        Args:
            text: Text input
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            image_features: List of image features
            bboxes: List of bounding boxes
            
        Returns:
            input_embeddings: Embeddings with special tokens replaced
        """
        # Get word embeddings
        word_embeddings = self.model.get_input_embeddings()
        input_embeddings = word_embeddings(input_ids)
        
        # Replace <image> tokens with image features
        image_token_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
        
        # For each <image> token, use the corresponding image feature
        for idx, (batch_idx, pos_idx) in enumerate(zip(image_token_positions[0], image_token_positions[1])):
            if idx < len(image_features):
                input_embeddings[batch_idx, pos_idx] = image_features[idx][0]  # [0] to get the first item from batch dim
        
        # Replace <box> tokens with bbox features if available
        if bboxes is not None and len(bboxes) > 0:
            box_token_positions = (input_ids == self.box_token_id).nonzero(as_tuple=True)
            
            # Encode bounding boxes
            bbox_tensor = torch.tensor(bboxes)
            bbox_features = self.encode_bbox(bbox_tensor)
            projected_bbox_features = self.bbox_projection(bbox_features)
            
            # Replace each <box> token with its corresponding bbox feature
            for idx, (batch_idx, pos_idx) in enumerate(zip(box_token_positions[0], box_token_positions[1])):
                if idx < len(bboxes):
                    input_embeddings[batch_idx, pos_idx] = projected_bbox_features[idx]
        
        return input_embeddings
    
    def in_context_learning(self, image, text_prompt, context_examples=None, bboxes=None):
        """
        Perform in-context learning with provided examples.
        
        Args:
            image: Current query image (PIL Image or path)
            text_prompt: Current query text
            context_examples: List of context examples
            bboxes: Bounding boxes for current query
            
        Returns:
            Tuple of (predicted_bboxes, generated_text)
        """
        # Prepare inputs with context
        inputs = self.prepare_in_context_inputs(
            image=image,
            text_prompt=text_prompt,
            bboxes=bboxes,
            context_examples=context_examples
        )
        
        # Tokenize combined text
        tokenized = self.tokenizer(
            inputs['combined_text'],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Embed inputs and replace special tokens
        input_embeddings = self.embed_inputs_with_special_tokens(
            text=inputs['combined_text'],
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            image_features=inputs['all_image_features'],
            bboxes=inputs['combined_bboxes']
        )
        
        # Generate output
        outputs = self.model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=tokenized['attention_mask'],
            max_length=self.max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        # Process output
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Extract bbox features and decode them
        bbox_features = self.model.get_bbox_features(outputs)
        predicted_bboxes = self.decode_bbox(bbox_features)
        
        return predicted_bboxes[0], generated_texts[0]
    
    def forward(self, images, texts, bboxes=None, labels=None, bboxes_mask=None):
        """
        Args:
            images: List of images
            text_prompts: List of pre-formatted text prompts with <box> and <image> tokens
            bboxes: List of bounding boxes corresponding to <box> tokens in text_prompts
            bounding box 는 nested form 이 아닌 채로 들어오며, 유효한 bounding box 의 위치를 나타내는 bboxes_mask 가 존재함.
        """
        # Process images
        image_inputs = self.processor(images=images, return_tensors="pt")
        image_features = self.processor.get_image_features(**image_inputs)
        image_features = self.image_projection(image_features)
        
        batch_size = len(texts)
        
        # Validate inputs
        if bboxes is not None:
            for i in range(batch_size):
                self.validate_box_tokens(texts[i], len(bboxes[i]))
                
            # Encode bounding boxes
            bbox_features = self.encode_bbox(torch.tensor(bboxes))
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create input embeddings
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        word_embeddings = self.model.get_input_embeddings()
        input_embeddings = word_embeddings(input_ids)
        
        # Replace <image> tokens with image features
        image_token_positions = (input_ids == self.image_token_id)
        input_embeddings[image_token_positions] = image_features[image_token_positions]
        
        # Replace <box> tokens with bbox features
        if bboxes is not None:
            box_token_positions = (input_ids == self.box_token_id)
            input_embeddings[box_token_positions] = bbox_features[bboxes_mask]
        
        if self.training:
            # Training mode
            outputs = self.model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                labels=input_ids  # Use input_ids as labels for causal LM
            )
            
            if bboxes is not None:
                # Add reconstruction loss for bbox features
                # reconstructed_bboxes = self.decode_bbox(bbox_features)
                # generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=False)
                # Next-Chat 은 예측된 token 을 기반으로하지 않고 label 을 기반으로 강제하여 box 위치를 할당함
                box_positions = labels.eq(self.box_token_id)
                box_hidden_states = outputs.hidden_states[box_positions, ...] # -1, hidden_size
                predicted_bboxes = self.bbox_decoder(box_hidden_states)
                box_loss = self.box_loss(predicted_bboxes, bboxes=bboxes[bboxes_mask])

                return outputs.loss + box_loss

            return outputs.loss
            
        else:
            # Inference mode
            outputs = self.model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            # 여기에서 나오는 결과들에는 bbox 정보가 들어있음
            # Extra 로 처리해야하는 부분은 bbox 정보를 추출하는 것.
            box_positions = outputs == self.box_token_id
            box_hidden_states = outputs.hidden_states[box_positions, ...] # -1, hidden_size
            predicted_bboxes = self.bbox_decoder(box_hidden_states)
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            return predicted_bboxes, generated_texts

    def box_loss(self, predicted_bboxes, bboxes):
        loss_bbox = F.l1_loss(predicted_bboxes, bboxes)
        giou_loss = self.giou_loss(pred_boxes=predicted_bboxes, target_boxes=bboxes)
        return loss_bbox + 0.1 * giou_loss

    def predict(self, image, text_prompt, bboxes=None):
        """Convenience method for single prediction"""
        if isinstance(image, str):
            image = Image.open(image)
            
        predicted_bboxes, generated_text = self.forward(
            [image], 
            [text_prompt],
            [bboxes] if bboxes is not None else None
        )
        return predicted_bboxes[0], generated_text[0] 

    def predict_with_context(self, image, text_prompt, context_examples, bboxes=None):
        """
        Convenience method for prediction with in-context learning.
        
        Args:
            image: Query image
            text_prompt: Query text
            context_examples: List of context examples for in-context learning
            bboxes: Bounding boxes for the current query (optional)
            
        Returns:
            Tuple of (predicted_bboxes, generated_text)
        """
        return self.in_context_learning(
            image=image,
            text_prompt=text_prompt,
            context_examples=context_examples,
            bboxes=bboxes
        )

    def giou_loss(self, pred_boxes, target_boxes):
        """
        Calculate the Generalized Intersection over Union (GIoU) loss between two sets of boxes.
        
        Args:
            pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4).
            target_boxes (Tensor): Ground truth bounding boxes of shape (N, 4).
            
        Returns:
            Tensor: GIoU loss.
        """
        # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)  # Add epsilon to avoid division by zero

        # Calculate enclosing box
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        # Calculate GIoU
        giou = iou - (enc_area - union_area) / (enc_area + 1e-6)

        # GIoU loss
        giou_loss = 1 - giou

        return giou_loss.mean()