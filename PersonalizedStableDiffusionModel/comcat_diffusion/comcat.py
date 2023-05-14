from typing import List
import torch

import torch.nn as nn

from comcat_diffusion.comcat_attention import CrossAttentionSVD, CLIPAttentionSVD


def inject_trainable_comcat(
        model: nn.Module,
        rank=2,
        target_replace_module="CrossAttention",
):
    if target_replace_module not in ['CLIPAttention', 'CrossAttention']:
        print('Unsupported Attention type:', target_replace_module)
        exit(1)

    require_grad_params = []
    names = []
    count = 0
    if target_replace_module == "CrossAttention":
        blocks = model.down_blocks[:3] + model.up_blocks[1:4] + [model.mid_block]
        for block in blocks:
            for attn in block.attentions:
                for transformer_block in attn.transformer_blocks:
                    count += 2
                    transformer_block.attn1 = CrossAttentionSVD(transformer_block.attn1, rank)
                    transformer_block.attn2 = CrossAttentionSVD(transformer_block.attn2, rank)
                    # for param in transformer_block.attn1.parameters():
                    #     require_grad_params.append(param)
                    # for param in transformer_block.attn2.parameters():
                    #     require_grad_params.append(param)
                    for a in [transformer_block.attn1, transformer_block.attn2]:
                        for params in [a.to_q.parameters(), a.to_k.parameters(), a.to_v.parameters(),
                                       a.to_out.parameters()]:
                            for param in params:
                                require_grad_params.append(param)

    elif target_replace_module == "CLIPAttention":
        for layer in model.text_model.encoder.layers:
            count += 1
            layer.self_attn = CLIPAttentionSVD(layer.self_attn, rank)
            a = layer.self_attn
            for params in [a.q_proj.parameters(), a.k_proj.parameters(), a.v_proj.parameters(),
                           a.out_proj.parameters()]:
                for param in params:
                    require_grad_params.append(param)
                require_grad_params.append(a.qk_bias)
            # for param in layer.self_attn.parameters():
            #     require_grad_params.append(param)
    print('target_replace_module:', target_replace_module, count)
    return require_grad_params, names


def save_comcat_weight(
        params, path="./comcat_weight.pt"
):
    torch.save(params, path)


def tune_comcat_scale(model_type, alpha: float = 1.0):
    if model_type == 'unet':
        CrossAttentionSVD.alpha = alpha
    elif model_type == 'text_encoder':
        CLIPAttentionSVD.alpha = alpha
    else:
        print('Only support unet and text_encoder!')


def patch_from_comcat_weight(
        model, params, target_replace_module="CrossAttention", rank=2
):
    if target_replace_module == "CrossAttention":
        blocks = model.down_blocks[:3] + model.up_blocks[1:4] + [model.mid_block]
        for block in blocks:
            for attn in block.attentions:
                for transformer_block in attn.transformer_blocks:
                    transformer_block.attn1 = CrossAttentionSVD(transformer_block.attn1, rank)
                    transformer_block.attn2 = CrossAttentionSVD(transformer_block.attn2, rank)
                    for a in [transformer_block.attn1, transformer_block.attn2]:
                        a.to_q.weight.data = params.pop(0)
                        a.to_k.weight.data = params.pop(0)
                        a.to_v.weight.data = params.pop(0)
                        a.to_out.weight.data = params.pop(0)
                        # a.to_out.bias.data = params.pop(0)
    elif target_replace_module == "CLIPAttention":
        for layer in model.text_model.encoder.layers:
            layer.self_attn = CLIPAttentionSVD(layer.self_attn, rank)
            a = layer.self_attn
            a.q_proj.weight.data = params.pop(0)
            a.k_proj.weight.data = params.pop(0)
            a.v_proj.weight.data = params.pop(0)
            a.out_proj.weight.data = params.pop(0)
            a.out_proj.bias.data = params.pop(0)
            a.qk_bias.data = params.pop(0)
    if len(params) > 0:
        print('Load Error!!')
        exit(1)


def load_learned_embed_in_clip(
        learned_embeds_path, text_encoder, tokenizer, token=None, idempotent=False
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    if num_added_tokens == 0 and idempotent:
        return token

    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token
