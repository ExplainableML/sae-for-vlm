from transformers import AutoProcessor, CLIPVisionModelWithProjection
import torch
import torch.nn as nn
from typing import Optional, Tuple

class Clip:
    def __init__(self, model_name, device):
        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained(f"openai/{model_name}").to(device)
        self.processor = AutoProcessor.from_pretrained(f"openai/{model_name}")
        self.register = {}
        self.attach_methods = {
            'in_mlp': self._attach_in_mlp,
            'post_mlp': self._attach_post_mlp,
            'post_mlp_residual': self._attach_post_mlp_residual,
            'post_projection': self._attach_post_projection,
        }

    def encode(self, inputs):
        for hook in self.register.keys():
            self.register[hook] = []
        outputs = self.model(**inputs.to(self.device))
        # pooled_output = outputs.pooler_output
        # return pooled_output
        image_embeds = outputs.image_embeds
        return image_embeds

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point in self.attach_methods:
            self.attach_methods[attachment_point](layer, sae)
            self.register[f'{attachment_point}_{layer}'] = []
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")

    def _attach_in_mlp(self, layer, sae):
        raise NotImplementedError

    def _attach_post_mlp(self, layer, sae):
        raise NotImplementedError

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.vision_model.encoder.layers[layer] = CLIPEncoderLayerPostMlpResidual(
            self.model.vision_model.encoder.layers[layer],
            sae,
            layer,
            self.register,
        )

    def _attach_post_projection(self, layer, sae):
        self.model.visual_projection = CLIPProjectionLayer(
            self.model.visual_projection,
            sae,
            layer,
            self.register,
        )

class CLIPProjectionLayer(nn.Module):
    def __init__(self, projector, sae, layer, register):
        super().__init__()
        self.projector = projector
        self.sae = sae
        self.layer = layer
        self.register = register

    def forward(self, inputs):
        outputs = self.projector(inputs)
        if self.sae is not None:
            outputs = self.sae.encode(outputs)
            self.register[f'post_projection_{self.layer}'].append(outputs.detach().cpu())
            outputs = self.sae.decode(outputs)
        else:
            self.register[f'post_projection_{self.layer}'].append(outputs.detach().cpu())
        return outputs

class CLIPEncoderLayerPostMlpResidual(nn.Module):
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
