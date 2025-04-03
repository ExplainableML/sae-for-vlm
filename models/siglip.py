from transformers import AutoProcessor, SiglipVisionModel
import torch
import torch.nn as nn
from typing import Optional, Tuple


# siglip-so400m-patch14-384
class Siglip:
    def __init__(self, model_name, device):
        self.device = device
        self.model = SiglipVisionModel.from_pretrained(f"google/{model_name}").to(device)
        self.processor = AutoProcessor.from_pretrained(f"google/{model_name}")
        self.register = {}
        self.attach_methods = {
            'post_mlp_residual': self._attach_post_mlp_residual,
            'post_projection': self._attach_post_projection,
        }
        self.sae = None
        self.layer = None

    def encode(self, inputs):
        for hook in self.register.keys():
            self.register[hook] = []
        outputs = self.model(**inputs.to(self.device))
        pooled_output = outputs.pooler_output

        if self.sae is not None:
            pooled_output = self.sae.encode(pooled_output)
            self.register[f'post_projection_{self.layer}'].append(pooled_output.detach().cpu())
            pooled_output = self.sae.decode(pooled_output)
        elif self.layer is not None:
            self.register[f'post_projection_{self.layer}'].append(pooled_output.detach().cpu())

        return pooled_output

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point in self.attach_methods:
            self.attach_methods[attachment_point](layer, sae)
            self.register[f'{attachment_point}_{layer}'] = []
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.vision_model.encoder.layers[layer] = SiglipEncoderLayerPostMlpResidual(
            self.model.vision_model.encoder.layers[layer],
            sae,
            layer,
            self.register,
        )

    def _attach_post_projection(self, layer, sae):
        self.sae = sae
        self.layer = layer


class SiglipEncoderLayerPostMlpResidual(nn.Module):
    def __init__(self, base, sae, layer, register):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2

        self.sae = sae
        self.layer = layer
        self.register = register

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.sae is not None:
            hidden_states = self.sae.encode(hidden_states)
            self.register[f'post_mlp_residual_{self.layer}'].append(hidden_states.detach().cpu())
            hidden_states = self.sae.decode(hidden_states)
        else:
            self.register[f'post_mlp_residual_{self.layer}'].append(hidden_states.detach().cpu())

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs