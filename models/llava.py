from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy

class Llava:

    def __init__(self, device):
        self.device = device
        self.layer = 22
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                                   torch_dtype=torch.float16,
                                                                   device_map=self.device)
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.base_CLIPEncoderLayerPostMlpResidual = copy.deepcopy(
            self.model.vision_tower.vision_model.encoder.layers[self.layer]
        )

    def prompt(self, text, image, max_tokens=5):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=[prompt],
                                padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        output = [x.split('ASSISTANT: ')[-1] for x in output]
        return output

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self.model.vision_tower.vision_model.encoder.layers[self.layer] = CLIPEncoderLayerPostMlpResidual(
            self.base_CLIPEncoderLayerPostMlpResidual,
            modified_sae,
        )


class SAEWrapper(nn.Module):

    def __init__(self, sae, neurons_to_fix, pre_zero):
        super().__init__()
        self.sae = sae
        self.neurons_to_fix = neurons_to_fix
        self.pre_zero = pre_zero

    def encode(self, x):
        x = self.sae.encode(x)
        if self.pre_zero:
            x = torch.zeros_like(x)
        for neuron_id, value in self.neurons_to_fix.items():
            x[:, :, neuron_id] = value
        return x

    def decode(self, x):
        x = self.sae.decode(x)
        x = x.to(dtype=torch.float16)
        return x




class CLIPEncoderLayerPostMlpResidual(nn.Module):

    def __init__(self, base, sae):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2
        self.sae = sae

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        encoded_hidden_states = self.sae.encode(hidden_states)
        decoded_hidden_states = self.sae.decode(encoded_hidden_states)
        hidden_states = decoded_hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs