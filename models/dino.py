from transformers import AutoImageProcessor, Dinov2Model
import torch

class Dino:
    def __init__(self, model_name="dinov2-base", device=torch.device("cuda")):
        self.device = device
        self.model = Dinov2Model.from_pretrained(f"facebook/{model_name}").to(device)
        self.processor = AutoImageProcessor.from_pretrained(f"facebook/{model_name}")

    def encode(self, inputs):
        outputs = self.model(**inputs.to(self.device))
        image_embeds = outputs.pooler_output
        return image_embeds
