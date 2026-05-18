from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .model import ModelConfig, Transformer
except ImportError:
    from model import ModelConfig, Transformer


def _past_key_values_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except TypeError:
            return int(past_key_values.get_seq_length(0))
        except Exception:
            pass
    try:
        if len(past_key_values) == 0:
            return 0
    except TypeError:
        return 0
    first_layer = past_key_values[0] if len(past_key_values) else None
    if first_layer is None:
        return 0
    try:
        return int(first_layer[0].size(2))
    except Exception:
        return 0


class LightLMConfig(PretrainedConfig):
    model_type = "lightlm"

    def __init__(
        self,
        vocab_size=49152,
        num_dims=512,
        num_heads=16,
        num_kv_heads=4,
        num_layers=32,
        ffn_hidden_dims=2048,
        rmsnorm_eps=1e-6,
        rope_theta=100000.0,
        context_len=2048,
        use_cache=False,
        use_flash=True,
        use_moe=False,
        moe_num_experts=2,
        moe_active_experts=2,
        moe_eps=1e-6,
        moe_aux_loss_coef=0.01,
        moe_shared_experts=1,
        use_lossfreebalance=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.ffn_hidden_dims = ffn_hidden_dims
        self.rmsnorm_eps = rmsnorm_eps
        self.rope_theta = rope_theta
        self.context_len = context_len
        self.use_cache = use_cache
        self.use_flash = use_flash
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_active_experts = moe_active_experts
        self.moe_eps = moe_eps
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.moe_shared_experts = moe_shared_experts
        self.use_lossfreebalance = use_lossfreebalance

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            num_dims=self.num_dims,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            num_layers=self.num_layers,
            ffn_hidden_dims=self.ffn_hidden_dims,
            rmsnorm_eps=self.rmsnorm_eps,
            rope_theta=self.rope_theta,
            context_len=self.context_len,
            use_cache=self.use_cache,
            use_flash=self.use_flash,
            use_moe=self.use_moe,
            moe_num_experts=self.moe_num_experts,
            moe_active_experts=self.moe_active_experts,
            moe_eps=self.moe_eps,
            moe_aux_loss_coef=self.moe_aux_loss_coef,
            moe_shared_experts=self.moe_shared_experts,
            use_lossfreebalance=self.use_lossfreebalance,
        )


class LightLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LightLMConfig
    base_model_prefix = "lightlm"
    supports_gradient_checkpointing = False
    _tied_weights_keys = ["lightlm.tokens_embedding.weight", "lightlm.ll_head.weight"]

    def __init__(self, config: LightLMConfig):
        super().__init__(config)
        self.lightlm = Transformer(config.to_model_config())
        self.all_tied_weights_keys = {key: key for key in self._tied_weights_keys}
        self.tie_weights()
        self.generation_config.use_cache = True

    def tie_weights(self, *args, **kwargs):
        del args, kwargs
        self.lightlm.ll_head.weight = self.lightlm.tokens_embedding.weight
        self.all_tied_weights_keys = {key: key for key in self._tied_weights_keys}

    def get_input_embeddings(self):
        return self.lightlm.tokens_embedding

    def set_input_embeddings(self, value):
        self.lightlm.tokens_embedding = value
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lightlm.ll_head

    def set_output_embeddings(self, new_embeddings):
        self.lightlm.ll_head = new_embeddings
        self.lightlm.tokens_embedding.weight = self.lightlm.ll_head.weight
        self.all_tied_weights_keys = {key: key for key in self._tied_weights_keys}

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        past_length = _past_key_values_length(past_key_values)
        if past_length > 0:
            input_ids = input_ids[:, past_length:]
            if input_ids.size(1) == 0:
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Optional[int] = None,
        num_logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        if logits_to_keep is None:
            logits_to_keep = num_logits_to_keep
        del attention_mask, kwargs
        if use_cache is None:
            use_cache = bool(self.config.use_cache)
        if labels is not None:
            use_cache = False

        if _past_key_values_length(past_key_values) == 0:
            past_key_values = None

        outputs = self.lightlm(
            input_ids,
            targets=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        )
        if use_cache:
            logits, _, _, present_key_values = outputs
        else:
            logits, _, _ = outputs
            present_key_values = None

        loss = None
        if labels is not None:
            if logits.size(1) != labels.size(1):
                labels = labels[:, -logits.size(1) :]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=present_key_values)
