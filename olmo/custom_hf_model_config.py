from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig

from olmo.config import (
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
    MoERouterType,
    MoEType,
)


class CustomOlmo2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Olmo2Model`]. It is used to instantiate an OLMo2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/Olmo2-7B-1124-hf](https://huggingface.co/allenai/Olmo2-7B-1124-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the Olmo2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Olmo2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50279):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.

    ```python
    >>> from transformers import Olmo2Model, Olmo2Config

    >>> # Initializing a Olmo2 7B style configuration
    >>> configuration = Olmo2Config()

    >>> # Initializing a model from the Olmo2 7B style configuration
    >>> model = Olmo2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "customolmo2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        attention_layer_norm=False,
        layer_norm_scale=False,
        attention_center=False,
        center_method="attn",
        norm_after=False,
        # MoE specific parameters
        moe_type="moe",
        num_experts=32,
        num_experts_per_tok=1,
        router_type="linear",
        moe_hidden_size=None,
        jitter_eps=None,
        normalize_expert_weights=None,
        uniform_expert_assignment=False,
        bias_gamma=None,
        gating_function="softmax",
        lb_loss_weight=None,
        z_loss_weight=None,
        scale_loss_by_num_layers=True,
        capacity_factor=1.2,
        skip_layer_ids=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_layer_norm = attention_layer_norm
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_scale = layer_norm_scale
        self.attention_center = attention_center
        self.center_method = center_method
        self.norm_after = norm_after

        # MoE specific parameters
        self.moe_type = moe_type
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_type = router_type
        self.moe_hidden_size = (
            moe_hidden_size if moe_hidden_size is not None else intermediate_size
        )
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment
        self.bias_gamma = bias_gamma
        self.gating_function = gating_function
        self.lb_loss_weight = lb_loss_weight
        self.z_loss_weight = z_loss_weight
        self.scale_loss_by_num_layers = scale_loss_by_num_layers
        self.capacity_factor = capacity_factor
        self.skip_layer_ids = skip_layer_ids
        if isinstance(skip_layer_ids, list):
            self.skip_layer_ids = [int(id) for id in skip_layer_ids]
        elif isinstance(skip_layer_ids, int):
            self.skip_layer_ids = [skip_layer_ids]
        elif isinstance(skip_layer_ids, str):
            self.skip_layer_ids = [
                int(id) for id in skip_layer_ids.replace("|", ",").split(",")
            ]

    def __setattr__(self, key, value):
        if key == "skip_layer_ids":
            if isinstance(value, list):
                value = [int(id) for id in value]
            elif isinstance(value, int):
                value = [value]
            elif isinstance(value, str):
                value = eval(value.replace("|", ","))
        return super().__setattr__(key, value)

    def get_moe_config(self) -> MoEConfig:
        """Convert HuggingFace config to OLMo MoEConfig."""
        return MoEConfig(
            hidden_size=self.moe_hidden_size,
            router_type=MoERouterType(self.router_type),
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            jitter_eps=self.jitter_eps,
            normalize_expert_weights=self.normalize_expert_weights,
            uniform_expert_assignment=self.uniform_expert_assignment,
            bias_gamma=self.bias_gamma,
            gating_function=MoERouterGatingFunction(self.gating_function),
            lb_loss_weight=self.lb_loss_weight,
            lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
            z_loss_weight=self.z_loss_weight,
            scale_loss_by_num_layers=self.scale_loss_by_num_layers,
            capacity_factor=self.capacity_factor,
            moe_type=MoEType(self.moe_type),
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                f"`rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )


class CustomOlmo2MoEConfig(CustomOlmo2Config):
    """
    Configuration class for Custom OLMo2 MoE models.
    """

    model_type = "customolmo2moe"


__all__ = ["CustomOlmo2Config", "CustomOlmo2MoEConfig"]
