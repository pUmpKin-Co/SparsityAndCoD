import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from olmo_core.distributed.checkpoint import unshard_checkpoint
from tokenizers import Tokenizer
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

from olmo.checkpoint import build_sharded_checkpointer
from olmo.config import TrainConfig
from olmo.custom_hf_model import CustomOlmo2MoEForCausalLM
from olmo.custom_hf_model_config import CustomOlmo2Config, CustomOlmo2MoEConfig

"""
Sample usage:

For single checkpoint (model.pt):
```
python scripts/convert_tools/convert_olmo_moe_hf.py \
    --input_dir /path/to/downloaded/olmo/moe/weights --output_dir /output/path
```

For distributed checkpoints (.distcp files):
```
python scripts/convert_tools/convert_olmo_moe_hf.py \
    --input_dir /path/to/distributed/checkpoint --output_dir /output/path
```

Force unsharding even if model.pt exists:
```
python scripts/convert_tools/convert_olmo_moe_hf.py \
    --input_dir /path/to/checkpoint --output_dir /output/path --force_unshard
```

Thereafter, models can be loaded via:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
The script automatically detects and unshards distributed checkpoints before conversion.
"""


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * (
        (int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of
    )


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def is_moe_model(olmo_config):
    """Check if the model is a MoE model."""
    return (
        olmo_config.get("block_type") == "moe"
        or olmo_config.get("moe_config") is not None
        or any("ff." in key for key in olmo_config.keys() if isinstance(key, str))
    )


def is_distributed_checkpoint(input_base_path):
    """Check if the checkpoint is a distributed checkpoint with .distcp files."""
    checkpoint_path = Path(input_base_path)

    # Check for model.pt first (single checkpoint)
    if (checkpoint_path / "model.pt").exists():
        return False

    # Check for distributed checkpoint files
    model_dir = checkpoint_path
    if model_dir.exists():
        distcp_files = list(model_dir.glob("*.distcp"))
        return len(distcp_files) > 0

    return False


def unshard_distributed_checkpoint(input_base_path, output_path=None):
    """Unshard a distributed checkpoint to create a single model.pt file."""
    print(f"Detected distributed checkpoint at {input_base_path}")
    print("Unsharding checkpoint...")

    # Load training config
    config_path = Path(input_base_path) / "config.yaml"
    train_config = TrainConfig.load(config_path)

    # Build checkpointer
    # checkpointer = build_sharded_checkpointer(train_config)

    # Unshard the checkpoint
    # model_state, _, _ = checkpointer.unshard_checkpoint(
    # load_path=input_base_path, load_optimizer_state=False, load_trainer_state=False
    # )

    # Save as model.pt
    # model_pt_path = Path(input_base_path) / "model.pt"
    # torch.save(model_state, model_pt_path)

    output_path = unshard_checkpoint(
        input_base_path,
        str(output_path) if output_path else str(Path(input_base_path) / "unsharded"),
        optim=False,
        save_overwrite=True,
    )
    if isinstance(output_path, tuple):
        output_path = output_path[0]
    print(f"Saved unsharded model to {output_path}")

    return output_path


def write_moe_model(
    model_path,
    input_base_path,
    tokenizer_path=None,
    safe_serialization=True,
    fix_eos_token_id=True,
    norm_after=False,
    layer_norm_scale=False,
    attention_center=False,
    center_method="attn",
    attention_layer_norm=False,
    force_unshard=False,
):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.yaml"
    olmo_config = yaml.safe_load(config_path.read_text())["model"]

    n_layers = olmo_config["n_layers"]
    n_heads = olmo_config["n_heads"]
    dim = olmo_config["d_model"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
    )
    max_position_embeddings = olmo_config["max_sequence_length"]

    vocab_size = olmo_config.get("embedding_size", olmo_config["vocab_size"])

    if olmo_config.get("n_kv_heads", None) is not None:
        num_key_value_heads = olmo_config["n_kv_heads"]  # for GQA / MQA
    elif olmo_config["multi_query_attention"]:  # compatibility with other checkpoints
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    # Extract MoE configuration
    moe_config = olmo_config.get("moe_config", {})
    if not moe_config:
        # Try to infer MoE config from block_type
        if olmo_config.get("block_type") == "moe":
            moe_config = {
                "num_experts": 8,  # Default values, should be in config
                "top_k": 2,
                "hidden_size": olmo_config.get(
                    "mlp_hidden_size", olmo_config.get("mlp_ratio", 4) * dim
                ),
                "moe_type": "moe",
            }

    print(f"MoE configuration: {moe_config}")
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    # Check if this is a distributed checkpoint and unshard if necessary
    if is_distributed_checkpoint(input_base_path) or force_unshard:
        output_path = unshard_distributed_checkpoint(input_base_path, model_path)
        loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    else:
        output_path = input_base_path
        loaded = torch.load(
            os.path.join(output_path, "model.pt"), map_location="cpu", weights_only=True
        )

    # Debug: Print available keys to understand the structure
    print("Available keys in checkpoint:")
    ff_keys = []
    for key in sorted(loaded.keys()):
        if "ff." in key or "moe" in key.lower():
            ff_keys.append(key)
            print(
                f"  {key}: {loaded[key].shape if hasattr(loaded[key], 'shape') else type(loaded[key])}"
            )

    if not ff_keys:
        print("  No MoE-related keys found!")
        print("  All available keys:")
        for i, key in enumerate(sorted(loaded.keys())):
            if i < 20:  # Show first 20 keys
                print(
                    f"    {key}: {loaded[key].shape if hasattr(loaded[key], 'shape') else type(loaded[key])}"
                )
            elif i == 20:
                print(f"    ... and {len(loaded.keys()) - 20} more keys")
                break
    print()

    param_count = 0
    index_dict = {"weight_map": {}}

    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"

        # Attention weights (same as dense model)
        fused_dims = [
            dim,
            dims_per_head * num_key_value_heads,
            dims_per_head * num_key_value_heads,
        ]
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
        )

        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                f"transformer.blocks.{layer_i}.attn_out.weight"
            ],
        }

        # Layer norms - MoE model uses input_layernorm and post_attention_layernorm
        state_dict[f"model.layers.{layer_i}.input_layernorm.weight"] = loaded[
            f"transformer.blocks.{layer_i}.attn_norm.weight"
        ]
        state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = loaded[
            f"transformer.blocks.{layer_i}.ff_norm.weight"
        ]

        if attention_layer_norm:
            state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"] = loaded[
                f"transformer.blocks.{layer_i}.q_norm.weight"
            ]
            state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"] = loaded[
                f"transformer.blocks.{layer_i}.k_norm.weight"
            ]

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        # MoE weights
        router_key = f"transformer.blocks.{layer_i}.ff.router.weight"
        if router_key in loaded:
            print(f"Found MoE router for layer {layer_i}")
            # Router weights
            state_dict[f"model.layers.{layer_i}.moe.router.weight"] = loaded[router_key]

            # Expert weights - these are stored as flattened tensors
            num_experts = moe_config.get("num_experts", 8)
            hidden_size = moe_config.get("hidden_size", dim * 4)
            print(f"MoE config: num_experts={num_experts}, hidden_size={hidden_size}")

            # Check for different possible expert weight key patterns
            w1_key = f"transformer.blocks.{layer_i}.ff.experts.mlp.w1"
            w2_key = f"transformer.blocks.{layer_i}.ff.experts.mlp.w2"
            w3_key = f"transformer.blocks.{layer_i}.ff.experts.mlp.w3"

            print(f"Looking for expert keys: {w1_key}, {w2_key}, {w3_key}")

            if w1_key in loaded and w2_key in loaded and w3_key in loaded:
                print(f"Found MoE expert weights for layer {layer_i}")
                w1_weight = loaded[f"transformer.blocks.{layer_i}.ff.experts.mlp.w1"]
                w2_weight = loaded[f"transformer.blocks.{layer_i}.ff.experts.mlp.w2"]
                w3_weight = loaded[f"transformer.blocks.{layer_i}.ff.experts.mlp.w3"]

                # Reshape from (num_experts * d_model, hidden_size) to individual expert weights
                w1_key = f"model.layers.{layer_i}.moe.experts.mlp.w1"
                w2_key = f"model.layers.{layer_i}.moe.experts.mlp.w2"
                w3_key = f"model.layers.{layer_i}.moe.experts.mlp.w3"

                state_dict[w1_key] = w1_weight
                state_dict[w2_key] = w2_weight
                state_dict[w3_key] = w3_weight

                if layer_i == 0:
                    print(f"Created keys: {w1_key}, {w2_key}, {w3_key}")

                # for expert_i in range(num_experts):
                #     start_idx = expert_i * dim
                #     end_idx = (expert_i + 1) * dim

                #     w1_key = (
                #         f"model.layers.{layer_i}.moe.experts.{expert_i}.mlp.w1.weight"
                #     )
                #     w2_key = (
                #         f"model.layers.{layer_i}.moe.experts.{expert_i}.mlp.w2.weight"
                #     )
                #     w3_key = (
                #         f"model.layers.{layer_i}.moe.experts.{expert_i}.mlp.w3.weight"
                #     )

                #     state_dict[w1_key] = w1_weight[start_idx:end_idx]
                #     state_dict[w2_key] = w2_weight[
                #         expert_i * hidden_size : (expert_i + 1) * hidden_size
                #     ]
                #     state_dict[w3_key] = w3_weight[start_idx:end_idx]

                #     if (
                #         layer_i == 0 and expert_i == 0
                #     ):  # Print for first expert of first layer
                #         print(f"Created keys: {w1_key}, {w2_key}, {w3_key}")

            # Handle bias if present
            if f"transformer.blocks.{layer_i}.ff.experts.mlp.w1.bias" in loaded:
                w1_bias = loaded[f"transformer.blocks.{layer_i}.ff.experts.mlp.w1.bias"]
                w2_bias = loaded[f"transformer.blocks.{layer_i}.ff.experts.mlp.w2.bias"]
                w3_bias = loaded[f"transformer.blocks.{layer_i}.ff.experts.mlp.w3.bias"]

                for expert_i in range(num_experts):
                    start_idx = expert_i * dim
                    end_idx = (expert_i + 1) * dim

                    state_dict[
                        f"model.layers.{layer_i}.moe.experts.{expert_i}.mlp.w1.bias"
                    ] = w1_bias[start_idx:end_idx]
                    state_dict[
                        f"model.layers.{layer_i}.moe.experts.{expert_i}.mlp.w2.bias"
                    ] = w2_bias[expert_i * hidden_size : (expert_i + 1) * hidden_size]
                    state_dict[
                        f"model.layers.{layer_i}.moe.experts.{expert_i}.mlp.w3.bias"
                    ] = w3_bias[start_idx:end_idx]
            else:
                print(f"WARNING: MoE expert weights not found for layer {layer_i}")
                print(f"Available keys containing 'ff' for layer {layer_i}:")
                for key in sorted(loaded.keys()):
                    if f"transformer.blocks.{layer_i}.ff" in key:
                        print(
                            f"  {key}: {loaded[key].shape if hasattr(loaded[key], 'shape') else type(loaded[key])}"
                        )
        else:
            print(f"WARNING: No MoE router found for layer {layer_i}")
            print(f"Looking for key: {router_key}")
            print(f"Available keys containing 'ff' for layer {layer_i}:")
            for key in sorted(loaded.keys()):
                if f"transformer.blocks.{layer_i}.ff" in key:
                    print(
                        f"  {key}: {loaded[key].shape if hasattr(loaded[key], 'shape') else type(loaded[key])}"
                    )

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
            if "moe.experts" in k and layer_i == 0:  # Print MoE keys for first layer
                print(f"  Saving to {filename}: {k}")
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"

    # Embedding and final layer weights
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "model.norm.weight": loaded["transformer.ln_f.weight"],
        "lm_head.weight": (
            loaded["transformer.ff_out.weight"]
            if "transformer.ff_out.weight" in loaded
            else loaded["transformer.wte.weight"]
        ),
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    if olmo_config.get("mlp_hidden_size", None) is not None:
        intermediate_size = olmo_config["mlp_hidden_size"] // 2
    else:
        intermediate_size = (dim * olmo_config["mlp_ratio"]) // 2

    if fix_eos_token_id and olmo_config["eos_token_id"] == 0:
        # Fixing a bug in OLMo where eos token id was incorrectly set
        print("Changing eos_token_id from 0 to 50279.")
        olmo_config["eos_token_id"] = 50279

    # Create HuggingFace config with MoE parameters
    config = CustomOlmo2MoEConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=olmo_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=olmo_config["eos_token_id"],
        tie_word_embeddings=olmo_config["weight_tying"],
        rope_theta=base,
        clip_qkv=olmo_config.get("clip_qkv"),
        norm_after=norm_after,
        layer_norm_scale=layer_norm_scale,
        attention_center=attention_center,
        center_method=center_method,
        attention_layer_norm=attention_layer_norm,
        # MoE specific parameters
        moe_type=moe_config.get("moe_type", "moe"),
        num_experts=moe_config.get("num_experts", 8),
        num_experts_per_tok=moe_config.get("top_k", 2),
        router_type="linear",
        moe_hidden_size=moe_config.get("hidden_size", intermediate_size),
        capacity_factor=moe_config.get("capacity_factor", 1.2),
        lb_loss_weight=moe_config.get("lb_loss_weight"),
        z_loss_weight=moe_config.get("z_loss_weight"),
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    _write_tokenizer(model_path, config, input_base_path, tokenizer_path)

    print("Loading the checkpoint in a OLMo MoE model.")
    model = CustomOlmo2MoEForCausalLM.from_pretrained(
        tmp_model_path, dtype=torch.bfloat16
    )
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


def _write_tokenizer(
    output_path: Path,
    config,
    checkpoint_dir: str,
    input_tokenizer_path: Path | None,
) -> None:
    print(f"Saving a {GPTNeoXTokenizerFast.__name__} to {output_path}.")

    if input_tokenizer_path is not None:
        base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))
    else:
        config_path = Path(checkpoint_dir) / "config.yaml"
        tokenizer_config = yaml.safe_load(config_path.read_text())["tokenizer"]

        # Initialize tokenizer and validate vocab size.
        if Path(tokenizer_config["identifier"]).is_file():
            base_tokenizer = Tokenizer.from_file(tokenizer_config["identifier"])
        else:
            base_tokenizer = Tokenizer.from_pretrained(tokenizer_config["identifier"])

    eos_token_id = (
        config.eos_token_id
        if config.eos_token_id is not None
        else base_tokenizer.get_vocab_size() - 1
    )
    pad_token_id = (
        config.pad_token_id if config.pad_token_id is not None else eos_token_id
    )

    tokenizer = GPTNeoXTokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
        unk_token="<unk>",
        bos_token="<|begin_of_text|>",
    )

    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo MoE weights, which contains config.yaml and either model.pt or distributed checkpoint files (.distcp).",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        default=None,
        help="Location of OLMo tokenizer json file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--no_fix_eos_token_id",
        action="store_false",
        dest="fix_eos_token_id",
        help="If set, does not change eos token id from 0 to 50279 if it is 0. Changing 0 to 50279 is a bug fix, so use this option with care.",
    )
    parser.add_argument(
        "--safe_serialization",
        type=bool,
        help="Whether or not to save using `safetensors`.",
    )
    parser.add_argument(
        "--norm_after",
        action="store_true",
        help="Whether to apply layer norm after the attention and feedforward layers.",
    )
    parser.add_argument(
        "--layer_norm_scale",
        action="store_true",
        help="Whether to scale the layer norm by the layer index.",
    )
    parser.add_argument(
        "--attention_center",
        action="store_true",
        help="Whether to center the attention output.",
    )
    parser.add_argument(
        "--center_method",
        choices=["attn", "value"],
        help="The method to use to center the attention output.",
        default="attn",
    )
    parser.add_argument(
        "--attention_layer_norm",
        action="store_true",
        help="Whether to apply layer norm to the attention output.",
    )
    parser.add_argument(
        "--force_unshard",
        action="store_true",
        help="Force unsharding of distributed checkpoints even if model.pt already exists.",
    )
    # Different OLMo versions used different default values for max_position_embeddings, hence the need to be able to specify which version is being used.
    args = parser.parse_args()
    write_moe_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        tokenizer_path=args.tokenizer_json_path,
        fix_eos_token_id=args.fix_eos_token_id,
        norm_after=args.norm_after,
        layer_norm_scale=args.layer_norm_scale,
        attention_center=args.attention_center,
        center_method=args.center_method,
        attention_layer_norm=args.attention_layer_norm,
        force_unshard=args.force_unshard,
    )


if __name__ == "__main__":
    main()
