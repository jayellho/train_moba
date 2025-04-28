
import torch
from transformers.models.gpt2 import modeling_gpt2
from moba.wrapper   import moba_layer, fa_to_hf
from moba.config    import MoBAConfig
from moba.moba_efficient import moba_attn_varlen

def patch_model_with_moba(model, block_size: int = 512, top_k: int = 3):
    """
    1) Registers a new 'moba' backend in GPT-2's ALL_ATTENTION_FUNCTIONS.
    2) Switches each GPT2Attention to use that 'moba' backend.
    """
    cfg = MoBAConfig(moba_chunk_size=block_size, moba_topk=top_k)

    def moba_attention_forward(
        module,
        query,               # [batch, heads, seq, head_dim]
        key,                 # [batch, kv_heads, seq, head_dim]
        value,               # [batch, kv_heads, seq, head_dim]
        attention_mask=None,
        head_mask=None,
        *args, **kwargs,
    ):
        batch, heads, seq, head_dim = query.shape

        # Call MoBA wrapper; returns packed output [batch*seq, heads, head_dim]
        packed_out, _ = moba_layer(
            moba_impl=moba_attn_varlen,
            moba_config=cfg,
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            *args, **kwargs,
        )
        # Unpack to HF shape [batch, seq, heads, head_dim]
        out_hf = fa_to_hf(packed_out, batch)
        # Permute to [batch, heads, seq, head_dim]
        attn_output = out_hf.permute(0, 2, 1, 3)

        # Return exactly what HF's attention_interface expects:
        # (attn_output, attn_weights), here we have no weights
        return attn_output, None

    # 1) Register backend
    modeling_gpt2.ALL_ATTENTION_FUNCTIONS["moba"] = moba_attention_forward

    # 2) Set each attention layer to use 'moba'
    for module in model.modules():
        if isinstance(module, modeling_gpt2.GPT2Attention):
            module.config._attn_implementation = "moba"

    return model
