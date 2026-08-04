# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://github.com/baudm/parseq/blob/main/strhub/models/parseq/system.py
# reference: https://arxiv.org/abs/2207.06966

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import math
from itertools import permutations
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        if attn_mask.dtype == torch.bool or attn_mask.dtype == torch.int:
            attn_mask = (attn_mask.to(dtype) - 1.0) * 1e9
        else:
            attn_mask = attn_mask.to(dtype)
    return attn_mask


class MultiheadAttention(nn.Module):
    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=True,
        bias=True,
    ):
        super().__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            f"but received {embed_dim}"
        )
        assert num_heads > 0, (
            "Expected num_heads to be greater than 0, "
            f"but received {num_heads}"
        )

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.kdim, embed_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.vdim, embed_dim, bias=bias
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias
        )

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = torch.reshape(q, [q.shape[0], -1, self.num_heads, self.head_dim])
        q = q.permute(0, 2, 1, 3)

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = torch.reshape(k, [k.shape[0], -1, self.num_heads, self.head_dim])
        k = k.permute(0, 2, 1, 3)
        v = torch.reshape(v, [v.shape[0], -1, self.num_heads, self.head_dim])
        v = v.permute(0, 2, 1, 3)
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        if type == MultiheadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            fill_shape = (torch.shape(key)[0].item(), self.num_heads, 0, self.head_dim)
            k = torch.full(fill_shape, fill_value=0, dtype=key.dtype)
            v = torch.full(fill_shape, fill_value=0, dtype=key.dtype)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        product = torch.matmul(
            q * (self.head_dim ** -0.5), k.transpose(-1, -2)
        )
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product, dim=-1)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
            )

        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, [out.shape[0], -1, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


def create_combined_mask(tgt_mask, tgt_key_padding_mask):
    # Create a boolean mask from tgt_mask
    tgt_mask_bool = (tgt_mask != float("-inf")).unsqueeze(0).unsqueeze(1)

    # Create a boolean mask from tgt_key_padding_mask
    key_padding_mask_bool = (~tgt_key_padding_mask).unsqueeze(1).unsqueeze(2)

    # Combine the masks
    combined_mask = tgt_mask_bool & key_padding_mask_bool

    return combined_mask


class DecoderLayer(torch.nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(
        #     d_model, nhead, dropout=dropout, batch_first=True
        # )  # paddle.nn.MultiHeadAttention默认为batch_first模式
        # self.cross_attn = nn.MultiheadAttention(
        #     d_model, nhead, dropout=dropout, batch_first=True
        # )

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout
        )  # paddle.nn.MultiHeadAttention默认为batch_first模式
        self.cross_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.linear1 = nn.Linear(
            in_features=d_model, out_features=dim_feedforward
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(
            in_features=dim_feedforward, out_features=d_model
        )
        self.norm1 = nn.LayerNorm(
            normalized_shape=d_model, eps=layer_norm_eps
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=d_model, eps=layer_norm_eps
        )
        self.norm_q = nn.LayerNorm(
            normalized_shape=d_model, eps=layer_norm_eps
        )
        self.norm_c = nn.LayerNorm(
            normalized_shape=d_model, eps=layer_norm_eps
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        if activation == "gelu":
            self.activation = nn.GELU()

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.gelu
        super().__setstate__(state)

    def forward_stream(
        self, tgt, tgt_norm, tgt_kv, memory, tgt_mask, tgt_key_padding_mask
    ):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        if tgt_key_padding_mask is not None:
            tgt_mask1 = create_combined_mask(tgt_mask, tgt_key_padding_mask)
            tgt2, sa_weights = self.self_attn(
                tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask1
            )
        else:
            tgt2, sa_weights = self.self_attn(
                tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask
            )

        tgt = tgt + self.dropout1(tgt2)
        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm2(tgt))))
        )
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask=None,
        content_mask=None,
        content_key_padding_mask=None,
        update_content=True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query,
            query_norm,
            content_norm,
            memory,
            query_mask,
            content_key_padding_mask,
        )[0]
        if update_content:
            content = self.forward_stream(
                content,
                content_norm,
                content_norm,
                memory,
                content_mask,
                content_key_padding_mask,
            )[0]
        return query, content


def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Decoder(torch.nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[torch.Tensor] = None,
        content_mask: Optional[torch.Tensor] = None,
        content_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query,
                content,
                memory,
                query_mask,
                content_mask,
                content_key_padding_mask,
                update_content=not last,
            )
        query = self.norm(query)
        return query


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=charset_size, embedding_dim=embed_dim
        )
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens.type(torch.int64))


class ParseQHead(nn.Module):
    def __init__(
        self,
        out_channels,
        max_text_length,
        embed_dim,
        dec_num_heads,
        dec_mlp_ratio,
        dec_depth,
        perm_num,
        perm_forward,
        perm_mirrored,
        decode_ar,
        refine_iters,
        dropout,
        **kwargs,
    ):
        super().__init__()

        self.bos_id = out_channels - 2
        self.eos_id = 0
        self.pad_id = out_channels - 1

        self.max_label_length = max_text_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        decoder_layer = DecoderLayer(
            embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout
        )
        self.decoder = Decoder(
            decoder_layer,
            num_layers=dec_depth,
            norm=torch.nn.LayerNorm(normalized_shape=embed_dim),
        )
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.head = nn.Linear(
            in_features=embed_dim, out_features=out_channels - 2
        )
        self.text_embed = TokenEmbedding(out_channels, embed_dim)
        self.pos_queries = nn.Parameter(
            torch.Tensor(1, max_text_length + 1, embed_dim),
        )
        self.pos_queries.stop_gradient = not True
        self.dropout = torch.nn.Dropout(p=dropout)
        self._device = next(self.parameters(recurse=False)).device
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, torch.nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.GroupNorm)
        ):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def no_weight_decay(self):
        param_names = {"text_embed.embedding.weight", "pos_queries"}
        enc_param_names = {("encoder." + n) for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img):
        return self.encoder(img)

    def decode(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_padding_mask=None,
        tgt_query=None,
        tgt_query_mask=None,
    ):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        if L != 1:
            tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
            tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        else:
            tgt_emb = self.dropout(null_ctx)
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(
            tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask
        )

    def forward_test(self, memory, max_length=None):
        testing = max_length is None
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = memory.shape[0]
        num_steps = max_length + 1

        pos_queries = self.pos_queries[:, :num_steps].expand([bs, -1, -1])
        tgt_mask = query_mask = torch.triu(
            torch.full((num_steps, num_steps), fill_value=float("-inf"), device=self._device),
            diagonal=1,
        )
        # tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), fill_value=self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(torch.as_tensor(num_steps)):
                j = i + 1
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                    #     break
            logits = torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full((bs, 1), fill_value=self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)
        if self.refine_iters:
            # temp = torch.triu(
            #     torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), diagonal=2
            # )
            # posi = np.where(temp.cpu().numpy() == True)
            # query_mask[posi] = 0

            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), fill_value=self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # tgt_padding_mask = (tgt_in == self.eos_id).type(dtype="int32")
                # tgt_padding_mask = tgt_padding_mask.cpu()
                # tgt_padding_mask = tgt_padding_mask.cumsum(dim=-1) > 0
                tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(-1) > 0
                # tgt_padding_mask = (
                #     # tgt_padding_mask.cuda().astype(dtype="float32") == 1.0
                #     tgt_padding_mask.type(dtype="float32") == 1.0
                # )
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    tgt_query=pos_queries,
                    tgt_query_mask=query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        # transfer to probility
        logits = F.softmax(logits, dim=-1)

        final_output = {"predict": logits}

        return final_output

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
        This works because the same attention mask can be used for the shorter sequences
        because of the padding mask.
        """
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(end=3, device=self._device).unsqueeze(dim=0)
        perms = [torch.arange(end=max_num_chars, device=self._device)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(
                    len(perm_pool), size=num_gen_perms - len(perms), replace=False
                )
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [
                    torch.randperm(max_num_chars, device=self._device)
                    for _ in range(num_gen_perms - len(perms))
                ]
            )
            perms = torch.stack(perms)
        if self.perm_mirrored:
            comp = perms.flip(-1)
            # x = torch.stack(tensors=[perms, comp])
            # perm_2 = list(range(x.ndim))
            # perm_2[0] = 1
            # perm_2[1] = 0
            # perms = x.transpose(perm_2[0], perm_2[1]).reshape((-1, max_num_chars))
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(end=max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            if len(masked_keys) == 0:
                break
            mask[query_idx, masked_keys] = float("-inf")
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float("-inf")
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def forward_train(self, memory, tgt):
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        logits_list = []
        final_out = {}
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(
                tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask
            )
            logits = self.head(out)
            if i == 0:
                final_out["predict"] = logits
            logits = logits.flatten(end_dim=1)
            logits_list.append(logits)

        final_out["logits_list"] = logits_list
        final_out["pad_id"] = self.pad_id
        final_out["eos_id"] = self.eos_id

        return final_out

    def forward(self, feat, targets=None):
        # feat : B, N, C
        # targets : labels, labels_len

        if self.training:
            label = targets[0]  # label
            label_len = targets[1]
            max_step = torch.max(label_len).cpu().numpy()[0] + 2
            crop_label = label[:, :max_step]
            final_out = self.forward_train(feat, crop_label)
        else:
            final_out = self.forward_test(feat)

        return final_out
