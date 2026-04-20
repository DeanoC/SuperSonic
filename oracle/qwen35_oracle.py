#!/usr/bin/env python3

import argparse
import json
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--device", default="cpu", help="device to run on (cpu, cuda:0, etc.)")
    parser.add_argument(
        "--trace-linear-layer",
        type=int,
        help="Optional Qwen linear-attention layer index to dump prefill internals for.",
    )
    parser.add_argument(
        "--trace-full-layer",
        type=int,
        help="Optional Qwen full-attention layer index to dump prefill internals for.",
    )
    parser.add_argument(
        "--trace-mlp-layer",
        type=int,
        help="Optional Qwen MLP layer index to dump prefill internals for.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_ids = [int(part) for part in args.prompt_ids.split(",") if part]
    if not prompt_ids:
        raise SystemExit("prompt ids must not be empty")

    target_device = args.device
    load_started = time.perf_counter()
    load_kwargs: dict[str, Any] = dict(
        torch_dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16,
        trust_remote_code=True,
    )
    if target_device != "cpu":
        load_kwargs["device_map"] = target_device
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    load_elapsed_ms = (time.perf_counter() - load_started) * 1000.0

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
    prefill_started = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    prefill_elapsed_ms = (time.perf_counter() - prefill_started) * 1000.0

    embedding_output = None
    first_layer_output = None
    first_layer_input_layernorm_output = None
    decode_first_layer_input_layernorm_output = None
    first_layer_linear_qkv_output = None
    decode_first_layer_linear_qkv_output = None
    first_layer_linear_z_output = None
    decode_first_layer_linear_z_output = None
    first_layer_linear_b_output = None
    decode_first_layer_linear_b_output = None
    first_layer_linear_a_output = None
    decode_first_layer_linear_a_output = None
    first_layer_linear_conv_weight = None
    first_layer_linear_pre_conv_value_focus_head_output = None
    first_layer_linear_post_conv_output = None
    first_layer_linear_direct_conv_output = None
    first_layer_linear_prepared_query_output = None
    decode_first_layer_linear_prepared_query_output = None
    first_layer_linear_prepared_key_output = None
    decode_first_layer_linear_prepared_key_output = None
    first_layer_linear_prepared_value_output = None
    decode_first_layer_linear_prepared_value_output = None
    first_layer_linear_prepared_beta_output = None
    decode_first_layer_linear_prepared_beta_output = None
    first_layer_linear_prepared_g_output = None
    decode_first_layer_linear_prepared_g_output = None
    first_layer_linear_direct_recurrent_output = None
    decode_first_layer_linear_direct_recurrent_output = None
    decode_first_layer_conv_state_before = None
    decode_first_layer_recurrent_state_before = None
    first_layer_linear_focus_kv_mem_output = None
    first_layer_linear_focus_delta_output = None
    first_layer_linear_focus_state_output = None
    first_layer_linear_focus_output = None
    first_layer_linear_focus_kv_mem_steps = None
    first_layer_linear_focus_delta_steps = None
    first_layer_linear_focus_state_steps = None
    first_layer_linear_focus_output_steps = None
    first_layer_linear_prepared_value_focus_head_output = None
    first_layer_linear_pre_norm_output = None
    decode_first_layer_linear_pre_norm_output = None
    first_layer_linear_pre_norm_mean_square = None
    first_layer_linear_pre_norm_rsqrt = None
    first_layer_linear_pre_norm_focus_head_output = None
    first_layer_linear_norm_gate_input = None
    first_layer_linear_norm_weight = None
    first_layer_linear_norm_weighted_hidden = None
    first_layer_linear_norm_silu_gate = None
    first_layer_linear_norm_output = None
    decode_first_layer_linear_norm_output = None
    first_layer_token_mixer_output = None
    decode_first_layer_token_mixer_output = None
    first_layer_post_attention_layernorm_output = None
    decode_first_layer_post_attention_layernorm_output = None
    first_layer_mlp_output = None
    decode_first_layer_mlp_output = None
    decode_first_layer_output = None
    decode_final_hidden_output = None
    decode_final_norm_input = None
    decode_final_norm_mean_square = None
    decode_final_norm_rsqrt = None
    decode_final_norm_weighted_hidden = None
    decode_final_norm_output = None
    decoder_layer_outputs = []
    decode_decoder_layer_outputs = []
    layer3_input_layernorm_output = None
    decode_layer3_input_layernorm_output = None
    layer3_input_layernorm_input = None
    decode_layer3_input_layernorm_input = None
    layer3_input_layernorm_mean_square = None
    decode_layer3_input_layernorm_mean_square = None
    layer3_input_layernorm_rsqrt = None
    decode_layer3_input_layernorm_rsqrt = None
    layer3_input_layernorm_weight = None
    layer3_input_layernorm_weighted_hidden = None
    decode_layer3_input_layernorm_weighted_hidden = None
    layer3_q_and_gate_output = None
    layer3_k_proj_output = None
    layer3_v_proj_output = None
    layer3_prepared_query_output = None
    layer3_gate_output = None
    layer3_prepared_key_output = None
    layer3_prepared_value_output = None
    layer3_attention_output = None
    layer3_token_mixer_output = None
    decode_layer3_token_mixer_output = None
    layer3_post_attention_layernorm_output = None
    decode_layer3_post_attention_layernorm_output = None
    layer3_mlp_output = None
    decode_layer3_mlp_output = None
    layer3_output = None
    decode_layer3_output = None
    decode_layer23_input_layernorm_output = None
    decode_layer23_input_layernorm_input = None
    decode_layer23_input_layernorm_mean_square = None
    decode_layer23_input_layernorm_rsqrt = None
    decode_layer23_input_layernorm_weighted_hidden = None
    decode_layer23_token_mixer_output = None
    decode_layer23_post_attention_layernorm_output = None
    decode_layer23_mlp_gate_proj_output = None
    decode_layer23_mlp_up_proj_output = None
    decode_layer23_mlp_activated_hidden = None
    decode_layer23_mlp_down_proj_output = None
    decode_layer23_mlp_output = None
    decode_layer23_output = None
    layer4_input_layernorm_output = None
    layer4_input_layernorm_input = None
    layer4_input_layernorm_mean_square = None
    layer4_input_layernorm_rsqrt = None
    layer4_input_layernorm_weight = None
    layer4_input_layernorm_weighted_hidden = None
    layer4_token_mixer_output = None
    layer4_post_attention_layernorm_output = None
    layer4_mlp_output = None
    layer4_output = None
    trace_linear_layer_idx = args.trace_linear_layer
    trace_full_layer_idx = args.trace_full_layer
    trace_mlp_layer_idx = args.trace_mlp_layer
    trace_linear_qkv_output = None
    trace_linear_z_output = None
    trace_linear_input_layernorm_output = None
    trace_linear_post_conv_output = None
    trace_linear_prepared_query_output = None
    trace_linear_prepared_key_output = None
    trace_linear_prepared_value_output = None
    trace_linear_prepared_beta_output = None
    trace_linear_prepared_g_output = None
    trace_linear_direct_recurrent_output = None
    trace_linear_norm_output = None
    trace_linear_token_mixer_output = None
    trace_full_q_and_gate_output = None
    trace_full_gate_output = None
    trace_full_k_proj_output = None
    trace_full_v_proj_output = None
    trace_full_prepared_query_output = None
    trace_full_prepared_key_output = None
    trace_full_prepared_value_output = None
    trace_full_attention_output = None
    trace_mlp_post_attention_layernorm_output = None
    trace_mlp_gate_proj_output = None
    trace_mlp_up_proj_output = None
    trace_mlp_activated_hidden = None
    trace_mlp_down_proj_output = None

    # Dictionary-based capture for middle decode layers (15-18)
    mid_layer_captures: dict[str, Any] = {}

    def embed_hook(_module, _inputs, output):
        nonlocal embedding_output
        if capture_phase != "prefill":
            return
        embedding_output = output.detach().to(dtype=torch.float32).cpu()

    def layer_hook(_module, _inputs, output):
        nonlocal first_layer_output
        nonlocal decode_first_layer_output
        if capture_phase == "decode":
            layer_output = output[0] if isinstance(output, tuple) else output
            decode_first_layer_output = layer_output.detach().to(dtype=torch.float32).cpu()
            return
        if capture_phase != "prefill":
            return
        layer_output = output[0] if isinstance(output, tuple) else output
        first_layer_output = layer_output.detach().to(dtype=torch.float32).cpu()

    capture_phase = "prefill"

    def make_decoder_layer_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            layer_output = output[0] if isinstance(output, tuple) else output
            captured = layer_output.detach().to(dtype=torch.float32).cpu()
            if capture_phase == "prefill":
                decoder_layer_outputs[layer_idx] = captured
            elif capture_phase == "decode":
                decode_decoder_layer_outputs[layer_idx] = captured

        return hook

    def capture_tensor(output):
        tensor = output[0] if isinstance(output, tuple) else output
        return tensor.detach().to(dtype=torch.float32).cpu()

    def focus_token_head(tensor: torch.Tensor, token_axis: int = 1, head_axis: int = 2) -> torch.Tensor:
        token_idx = min(2, tensor.shape[token_axis] - 1)
        head_idx = min(6, tensor.shape[head_axis] - 1)
        return tensor.select(token_axis, token_idx).select(head_axis - 1, head_idx).cpu()

    def compute_linear_prefill_trace(
        layer_idx: int,
        z_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        a_tensor: torch.Tensor,
        conv_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        linear_attn = model.model.layers[layer_idx].linear_attn
        value_dim = z_tensor.shape[-1]
        key_dim = (conv_output.shape[-1] - value_dim) // 2
        num_k_heads = int(linear_attn.num_k_heads)
        num_v_heads = int(linear_attn.num_v_heads)
        head_k_dim = int(linear_attn.head_k_dim)
        head_v_dim = int(linear_attn.head_v_dim)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        query = conv_output[..., :key_dim].reshape(
            batch_size, seq_len, num_k_heads, head_k_dim
        )
        key = conv_output[..., key_dim : key_dim * 2].reshape(
            batch_size, seq_len, num_k_heads, head_k_dim
        )
        value = conv_output[..., key_dim * 2 : key_dim * 2 + value_dim].reshape(
            batch_size, seq_len, num_v_heads, head_v_dim
        )
        query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)

        head_repeat = num_v_heads // num_k_heads
        if head_repeat > 1:
            query = query.repeat_interleave(head_repeat, dim=2)
            key = key.repeat_interleave(head_repeat, dim=2)

        beta = torch.sigmoid(b_tensor).to(dtype=torch.float32)
        dt_bias = linear_attn.dt_bias.detach().to(dtype=torch.float32).reshape(1, 1, num_v_heads)
        a_log_exp = linear_attn.A_log.detach().to(dtype=torch.float32).exp().reshape(1, 1, num_v_heads)
        g = -torch.log1p(torch.exp(a_tensor.to(dtype=torch.float32) + dt_bias)) * a_log_exp

        q = query.transpose(1, 2).contiguous() * (1.0 / (head_k_dim ** 0.5))
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        beta_t = beta.transpose(1, 2).contiguous()
        g_t = g.transpose(1, 2).contiguous()
        state = torch.zeros(
            (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=torch.float32
        )
        outputs = []
        for step in range(seq_len):
            q_step = q[:, :, step, :]
            k_step = k[:, :, step, :]
            v_step = v[:, :, step, :]
            beta_step = beta_t[:, :, step].unsqueeze(-1)
            g_step = g_t[:, :, step].exp().unsqueeze(-1).unsqueeze(-1)
            state = state * g_step
            kv_mem = (state * k_step.unsqueeze(-1)).sum(dim=2)
            delta = (v_step - kv_mem) * beta_step
            state = state + k_step.unsqueeze(-1) * delta.unsqueeze(2)
            out_step = (state * q_step.unsqueeze(-1)).sum(dim=2)
            outputs.append(out_step.unsqueeze(2))

        direct_recurrent = (
            torch.cat(outputs, dim=2)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, -1)
            .cpu()
        )
        return {
            "query": query.cpu(),
            "key": key.cpu(),
            "value": value.cpu(),
            "beta": beta.cpu(),
            "g": g.cpu(),
            "direct_recurrent": direct_recurrent,
        }

    def reconstruct_decode_linear_outputs():
        nonlocal decode_first_layer_linear_prepared_query_output
        nonlocal decode_first_layer_linear_prepared_key_output
        nonlocal decode_first_layer_linear_prepared_value_output
        nonlocal decode_first_layer_linear_prepared_beta_output
        nonlocal decode_first_layer_linear_prepared_g_output
        nonlocal decode_first_layer_linear_direct_recurrent_output
        if decode_first_layer_linear_prepared_query_output is not None:
            return
        if decode_first_layer_conv_state_before is None:
            raise RuntimeError("decode linear_attn pre-hook must capture conv state before decode reconstruction")
        if decode_first_layer_recurrent_state_before is None:
            raise RuntimeError("decode linear_attn pre-hook must capture recurrent state before decode reconstruction")
        if decode_first_layer_linear_qkv_output is None:
            raise RuntimeError("decode linear_qkv hook must run before decode reconstruction")
        if decode_first_layer_linear_z_output is None:
            raise RuntimeError("decode linear_z hook must run before decode reconstruction")
        if decode_first_layer_linear_b_output is None:
            raise RuntimeError("decode linear_b hook must run before decode reconstruction")
        if decode_first_layer_linear_a_output is None:
            raise RuntimeError("decode linear_a hook must run before decode reconstruction")

        batch_size = decode_first_layer_linear_qkv_output.shape[0]
        seq_len_local = decode_first_layer_linear_qkv_output.shape[1]
        linear_attn = model.model.layers[0].linear_attn
        qkv_t = decode_first_layer_linear_qkv_output.transpose(1, 2).contiguous()
        conv_weight = linear_attn.conv1d.weight.detach().squeeze(1).to(dtype=torch.float32)
        conv_bias = (
            linear_attn.conv1d.bias.detach().to(dtype=torch.float32)
            if linear_attn.conv1d.bias is not None
            else None
        )
        hidden_states_new = torch.cat(
            [decode_first_layer_conv_state_before, qkv_t.to(dtype=torch.float32)], dim=-1
        )
        direct_conv = F.conv1d(
            hidden_states_new,
            conv_weight.unsqueeze(1),
            conv_bias,
            padding=0,
            groups=hidden_states_new.shape[1],
        )
        direct_conv = F.silu(direct_conv[:, :, -seq_len_local:]).transpose(1, 2).contiguous()
        value_dim = decode_first_layer_linear_z_output.shape[-1]
        key_dim = (direct_conv.shape[-1] - value_dim) // 2
        num_k_heads = int(linear_attn.num_k_heads)
        num_v_heads = int(linear_attn.num_v_heads)
        head_k_dim = int(linear_attn.head_k_dim)
        head_v_dim = int(linear_attn.head_v_dim)
        query = direct_conv[..., :key_dim].reshape(
            batch_size, seq_len_local, num_k_heads, head_k_dim
        )
        key = direct_conv[..., key_dim : key_dim * 2].reshape(
            batch_size, seq_len_local, num_k_heads, head_k_dim
        )
        value = direct_conv[..., key_dim * 2 : key_dim * 2 + value_dim].reshape(
            batch_size, seq_len_local, num_v_heads, head_v_dim
        )
        query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)
        head_repeat = num_v_heads // num_k_heads
        if head_repeat > 1:
            query = query.repeat_interleave(head_repeat, dim=2)
            key = key.repeat_interleave(head_repeat, dim=2)
        decode_first_layer_linear_prepared_query_output = query.cpu()
        decode_first_layer_linear_prepared_key_output = key.cpu()
        decode_first_layer_linear_prepared_value_output = value.cpu()
        beta = torch.sigmoid(decode_first_layer_linear_b_output).to(dtype=torch.float32)
        decode_first_layer_linear_prepared_beta_output = beta.cpu()
        dt_bias = linear_attn.dt_bias.detach().to(dtype=torch.float32).reshape(1, 1, num_v_heads)
        a_log_exp = linear_attn.A_log.detach().to(dtype=torch.float32).exp().reshape(1, 1, num_v_heads)
        a_input = decode_first_layer_linear_a_output.to(dtype=torch.float32)
        g = -torch.log1p(torch.exp(a_input + dt_bias)) * a_log_exp
        decode_first_layer_linear_prepared_g_output = g.cpu()
        q = query.transpose(1, 2).contiguous() * (1.0 / (head_k_dim ** 0.5))
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        beta_t = beta.transpose(1, 2).contiguous()
        g_t = g.transpose(1, 2).contiguous()
        state = decode_first_layer_recurrent_state_before.to(dtype=torch.float32).clone()
        outputs = []
        for step in range(seq_len_local):
            q_step = q[:, :, step, :]
            k_step = k[:, :, step, :]
            v_step = v[:, :, step, :]
            beta_step = beta_t[:, :, step].unsqueeze(-1)
            g_step = g_t[:, :, step].exp().unsqueeze(-1).unsqueeze(-1)
            state = state * g_step
            kv_mem = (state * k_step.unsqueeze(-1)).sum(dim=2)
            delta = (v_step - kv_mem) * beta_step
            state = state + k_step.unsqueeze(-1) * delta.unsqueeze(2)
            out_step = (state * q_step.unsqueeze(-1)).sum(dim=2)
            outputs.append(out_step.unsqueeze(2))
        decode_first_layer_linear_direct_recurrent_output = (
            torch.cat(outputs, dim=2)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len_local, -1)
            .cpu()
        )

    def input_layernorm_hook(_module, _inputs, output):
        nonlocal first_layer_input_layernorm_output
        nonlocal decode_first_layer_input_layernorm_output
        if capture_phase == "decode":
            decode_first_layer_input_layernorm_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_input_layernorm_output = capture_tensor(output)

    def token_mixer_hook(_module, _inputs, output):
        nonlocal first_layer_token_mixer_output
        nonlocal decode_first_layer_token_mixer_output
        if capture_phase == "decode":
            decode_first_layer_token_mixer_output = capture_tensor(output)
            reconstruct_decode_linear_outputs()
            return
        if capture_phase != "prefill":
            return
        first_layer_token_mixer_output = capture_tensor(output)

    def linear_qkv_hook(_module, _inputs, output):
        nonlocal first_layer_linear_qkv_output
        nonlocal first_layer_linear_pre_conv_value_focus_head_output
        nonlocal decode_first_layer_linear_qkv_output
        if capture_phase == "decode":
            decode_first_layer_linear_qkv_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_linear_qkv_output = capture_tensor(output)
        qkv = first_layer_linear_qkv_output
        value_dim = first_layer_linear_z_output.shape[-1] if first_layer_linear_z_output is not None else 2048
        key_dim = (qkv.shape[-1] - value_dim) // 2
        num_v_heads = 16
        head_v_dim = value_dim // num_v_heads
        value = qkv[..., key_dim * 2 : key_dim * 2 + value_dim].reshape(
            input_ids.shape[0], input_ids.shape[1], num_v_heads, head_v_dim
        )
        first_layer_linear_pre_conv_value_focus_head_output = focus_token_head(value)

    def linear_z_hook(_module, _inputs, output):
        nonlocal first_layer_linear_z_output
        nonlocal decode_first_layer_linear_z_output
        if capture_phase == "decode":
            decode_first_layer_linear_z_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_linear_z_output = capture_tensor(output)

    def linear_b_hook(_module, _inputs, output):
        nonlocal first_layer_linear_b_output
        nonlocal decode_first_layer_linear_b_output
        if capture_phase == "decode":
            decode_first_layer_linear_b_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_linear_b_output = capture_tensor(output)

    def linear_a_hook(_module, _inputs, output):
        nonlocal first_layer_linear_a_output
        nonlocal decode_first_layer_linear_a_output
        if capture_phase == "decode":
            decode_first_layer_linear_a_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_linear_a_output = capture_tensor(output)

    def linear_conv_hook(_module, _inputs, output):
        nonlocal first_layer_linear_post_conv_output
        nonlocal first_layer_linear_direct_conv_output
        nonlocal first_layer_linear_prepared_query_output
        nonlocal first_layer_linear_prepared_key_output
        nonlocal first_layer_linear_prepared_value_output
        nonlocal first_layer_linear_prepared_beta_output
        nonlocal first_layer_linear_prepared_g_output
        nonlocal first_layer_linear_direct_recurrent_output
        nonlocal first_layer_linear_focus_kv_mem_output
        nonlocal first_layer_linear_focus_delta_output
        nonlocal first_layer_linear_focus_state_output
        nonlocal first_layer_linear_focus_output
        nonlocal first_layer_linear_focus_kv_mem_steps
        nonlocal first_layer_linear_focus_delta_steps
        nonlocal first_layer_linear_focus_state_steps
        nonlocal first_layer_linear_focus_output_steps
        nonlocal first_layer_linear_prepared_value_focus_head_output
        nonlocal first_layer_linear_conv_weight
        nonlocal decode_first_layer_linear_prepared_query_output
        nonlocal decode_first_layer_linear_prepared_key_output
        nonlocal decode_first_layer_linear_prepared_value_output
        nonlocal decode_first_layer_linear_prepared_beta_output
        nonlocal decode_first_layer_linear_prepared_g_output
        nonlocal decode_first_layer_linear_direct_recurrent_output
        if capture_phase != "prefill":
            if capture_phase != "decode":
                return
            return
        tensor = capture_tensor(output)
        first_layer_linear_conv_weight = _module.weight.detach().squeeze(1).to(dtype=torch.float32).cpu()
        seq_len = input_ids.shape[1]
        post_conv = tensor.transpose(1, 2)[:, -seq_len:, :].contiguous()
        first_layer_linear_post_conv_output = post_conv
        if first_layer_linear_qkv_output is None:
            raise RuntimeError("linear_qkv hook must run before linear_conv hook")
        qkv = first_layer_linear_qkv_output.transpose(1, 2).contiguous()
        direct_conv = F.conv1d(
            qkv,
            _module.weight.detach().to(dtype=torch.float32),
            bias=_module.bias.detach().to(dtype=torch.float32) if _module.bias is not None else None,
            stride=_module.stride[0],
            padding=_module.padding[0],
            dilation=_module.dilation[0],
            groups=_module.groups,
        )
        direct_conv = F.silu(direct_conv[:, :, :seq_len]).transpose(1, 2).contiguous().cpu()
        first_layer_linear_direct_conv_output = direct_conv
        if first_layer_linear_z_output is None:
            raise RuntimeError("linear_z hook must run before linear_conv hook")
        value_dim = first_layer_linear_z_output.shape[-1]
        key_dim = (direct_conv.shape[-1] - value_dim) // 2
        linear_attn = model.model.layers[0].linear_attn
        num_k_heads = int(linear_attn.num_k_heads)
        num_v_heads = int(linear_attn.num_v_heads)
        head_k_dim = int(linear_attn.head_k_dim)
        head_v_dim = int(linear_attn.head_v_dim)
        query = direct_conv[..., :key_dim].reshape(
            input_ids.shape[0], input_ids.shape[1], num_k_heads, head_k_dim
        )
        key = direct_conv[..., key_dim : key_dim * 2].reshape(
            input_ids.shape[0], input_ids.shape[1], num_k_heads, head_k_dim
        )
        value = direct_conv[..., key_dim * 2 : key_dim * 2 + value_dim].reshape(
            input_ids.shape[0], input_ids.shape[1], num_v_heads, head_v_dim
        )
        query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)
        head_repeat = num_v_heads // num_k_heads
        if head_repeat > 1:
            query = query.repeat_interleave(head_repeat, dim=2)
            key = key.repeat_interleave(head_repeat, dim=2)
        first_layer_linear_prepared_query_output = query.cpu()
        first_layer_linear_prepared_key_output = key.cpu()
        first_layer_linear_prepared_value_output = value.cpu()
        beta = torch.sigmoid(
            first_layer_linear_b_output
        ).to(dtype=torch.float32)
        first_layer_linear_prepared_beta_output = beta.cpu()
        dt_bias = linear_attn.dt_bias.detach().to(dtype=torch.float32).reshape(1, 1, num_v_heads)
        a_log_exp = linear_attn.A_log.detach().to(dtype=torch.float32).exp().reshape(
            1, 1, num_v_heads
        )
        a_input = first_layer_linear_a_output.to(dtype=torch.float32)
        g = (
            -torch.log1p(torch.exp(a_input + dt_bias)) * a_log_exp
        )
        first_layer_linear_prepared_g_output = g.cpu()
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        beta_t = beta.transpose(1, 2).contiguous()
        g_t = g.transpose(1, 2).contiguous()
        q = q * (1.0 / (head_k_dim ** 0.5))
        state = torch.zeros(
            (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=torch.float32
        )
        outputs = []
        focus_kv_mem_steps = []
        focus_delta_steps = []
        focus_state_steps = []
        focus_output_steps = []
        focus_step = min(2, seq_len - 1)
        focus_head = min(6, num_v_heads - 1)
        for step in range(seq_len):
            q_step = q[:, :, step, :]
            k_step = k[:, :, step, :]
            v_step = v[:, :, step, :]
            beta_step = beta_t[:, :, step].unsqueeze(-1)
            g_step = g_t[:, :, step].exp().unsqueeze(-1).unsqueeze(-1)
            state = state * g_step
            kv_mem = (state * k_step.unsqueeze(-1)).sum(dim=2)
            delta = (v_step - kv_mem) * beta_step
            state = state + k_step.unsqueeze(-1) * delta.unsqueeze(2)
            out_step = (state * q_step.unsqueeze(-1)).sum(dim=2)
            focus_kv_mem_steps.append(kv_mem[0, focus_head].cpu())
            focus_delta_steps.append(delta[0, focus_head].cpu())
            focus_state_steps.append(state[0, focus_head].cpu())
            focus_output_steps.append(out_step[0, focus_head].cpu())
            if step == focus_step:
                first_layer_linear_focus_kv_mem_output = kv_mem[0, focus_head].cpu()
                first_layer_linear_focus_delta_output = delta[0, focus_head].cpu()
                first_layer_linear_focus_state_output = state[0, focus_head].cpu()
                first_layer_linear_focus_output = out_step[0, focus_head].cpu()
            outputs.append(out_step.unsqueeze(2))
        first_layer_linear_focus_kv_mem_steps = focus_kv_mem_steps
        first_layer_linear_focus_delta_steps = focus_delta_steps
        first_layer_linear_focus_state_steps = focus_state_steps
        first_layer_linear_focus_output_steps = focus_output_steps
        first_layer_linear_direct_recurrent_output = (
            torch.cat(outputs, dim=2).transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1).cpu()
        )
        first_layer_linear_prepared_value_focus_head_output = focus_token_head(value)

    def linear_norm_hook(_module, _inputs, output):
        nonlocal first_layer_linear_norm_output
        nonlocal decode_first_layer_linear_norm_output
        if capture_phase == "decode":
            tensor = capture_tensor(output)
            decode_first_layer_linear_norm_output = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
            return
        if capture_phase != "prefill":
            return
        tensor = capture_tensor(output)
        first_layer_linear_norm_output = tensor.reshape(input_ids.shape[0], input_ids.shape[1], -1)

    def linear_norm_pre_hook(_module, inputs):
        nonlocal first_layer_linear_pre_norm_output
        nonlocal decode_first_layer_linear_pre_norm_output
        if capture_phase == "decode":
            tensor = capture_tensor(inputs[0])
            decode_first_layer_linear_pre_norm_output = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
            return
        if capture_phase != "prefill":
            return
        tensor = capture_tensor(inputs[0])
        hidden = tensor.reshape(input_ids.shape[0], input_ids.shape[1], -1)
        first_layer_linear_pre_norm_output = hidden
        head_dim = _module.weight.shape[0]
        num_heads = hidden.shape[-1] // head_dim
        hidden_heads = hidden.reshape(input_ids.shape[0], input_ids.shape[1], num_heads, head_dim)
        nonlocal first_layer_linear_pre_norm_mean_square
        nonlocal first_layer_linear_pre_norm_rsqrt
        nonlocal first_layer_linear_pre_norm_focus_head_output
        mean_square = hidden_heads.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + _module.variance_epsilon)
        first_layer_linear_pre_norm_mean_square = mean_square.squeeze(-1).cpu()
        first_layer_linear_pre_norm_rsqrt = rsqrt.squeeze(-1).cpu()
        first_layer_linear_pre_norm_focus_head_output = focus_token_head(hidden_heads)
        nonlocal first_layer_linear_norm_gate_input
        gate_tensor = capture_tensor(inputs[1])
        first_layer_linear_norm_gate_input = gate_tensor.reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        nonlocal first_layer_linear_norm_weight
        first_layer_linear_norm_weight = _module.weight.detach().to(dtype=torch.float32).cpu()
        hidden = inputs[0].detach().to(dtype=torch.float32)
        gate = inputs[1].detach().to(dtype=torch.float32)
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        weighted_hidden = hidden * torch.rsqrt(variance + _module.variance_epsilon)
        weighted_hidden = weighted_hidden * _module.weight.detach().to(dtype=torch.float32)
        silu_gate = F.silu(gate)
        nonlocal first_layer_linear_norm_weighted_hidden
        nonlocal first_layer_linear_norm_silu_gate
        first_layer_linear_norm_weighted_hidden = weighted_hidden.reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        ).cpu()
        first_layer_linear_norm_silu_gate = silu_gate.reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        ).cpu()

    def make_trace_linear_hooks(layer_idx: int):
        layer = model.model.layers[layer_idx]
        linear_attn = layer.linear_attn

        def input_layernorm_hook(_module, _inputs, output):
            nonlocal trace_linear_input_layernorm_output
            if capture_phase != "prefill":
                return
            trace_linear_input_layernorm_output = capture_tensor(output)

        def qkv_hook(_module, _inputs, output):
            nonlocal trace_linear_qkv_output
            if capture_phase != "prefill":
                return
            trace_linear_qkv_output = capture_tensor(output)

        def z_hook(_module, _inputs, output):
            nonlocal trace_linear_z_output
            if capture_phase != "prefill":
                return
            trace_linear_z_output = capture_tensor(output)

        def b_hook(_module, _inputs, output):
            nonlocal trace_linear_prepared_beta_output
            if capture_phase != "prefill":
                return
            trace_linear_prepared_beta_output = capture_tensor(output)

        def a_hook(_module, _inputs, output):
            nonlocal trace_linear_prepared_g_output
            if capture_phase != "prefill":
                return
            trace_linear_prepared_g_output = capture_tensor(output)

        def conv_hook(_module, _inputs, output):
            nonlocal trace_linear_post_conv_output
            nonlocal trace_linear_prepared_query_output
            nonlocal trace_linear_prepared_key_output
            nonlocal trace_linear_prepared_value_output
            nonlocal trace_linear_prepared_beta_output
            nonlocal trace_linear_prepared_g_output
            nonlocal trace_linear_direct_recurrent_output
            if capture_phase != "prefill":
                return
            if (
                trace_linear_z_output is None
                or trace_linear_prepared_beta_output is None
                or trace_linear_prepared_g_output is None
            ):
                raise RuntimeError("trace linear projections must run before trace linear conv hook")
            tensor = capture_tensor(output)
            seq_len = input_ids.shape[1]
            post_conv = F.silu(tensor[:, :, :seq_len]).transpose(1, 2).contiguous()
            trace_linear_post_conv_output = post_conv.cpu()
            trace = compute_linear_prefill_trace(
                layer_idx,
                trace_linear_z_output,
                trace_linear_prepared_beta_output,
                trace_linear_prepared_g_output,
                post_conv,
            )
            trace_linear_prepared_query_output = trace["query"]
            trace_linear_prepared_key_output = trace["key"]
            trace_linear_prepared_value_output = trace["value"]
            trace_linear_prepared_beta_output = trace["beta"]
            trace_linear_prepared_g_output = trace["g"]
            trace_linear_direct_recurrent_output = trace["direct_recurrent"]

        def norm_hook(_module, _inputs, output):
            nonlocal trace_linear_norm_output
            if capture_phase != "prefill":
                return
            tensor = capture_tensor(output)
            trace_linear_norm_output = tensor.reshape(input_ids.shape[0], input_ids.shape[1], -1)

        def token_mixer_hook(_module, _inputs, output):
            nonlocal trace_linear_token_mixer_output
            if capture_phase != "prefill":
                return
            trace_linear_token_mixer_output = capture_tensor(output)

        return [
            layer.input_layernorm.register_forward_hook(input_layernorm_hook),
            linear_attn.in_proj_qkv.register_forward_hook(qkv_hook),
            linear_attn.in_proj_z.register_forward_hook(z_hook),
            linear_attn.in_proj_b.register_forward_hook(b_hook),
            linear_attn.in_proj_a.register_forward_hook(a_hook),
            linear_attn.conv1d.register_forward_hook(conv_hook),
            linear_attn.norm.register_forward_hook(norm_hook),
            linear_attn.register_forward_hook(token_mixer_hook),
        ]

    def make_trace_full_hooks(layer_idx: int):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        def q_proj_hook(_module, _inputs, output):
            nonlocal trace_full_q_and_gate_output
            nonlocal trace_full_gate_output
            if capture_phase != "prefill":
                return
            tensor = capture_tensor(output)
            trace_full_q_and_gate_output = tensor
            head_dim = int(attn.head_dim)
            num_heads = tensor.shape[-1] // (head_dim * 2)
            q_and_gate = tensor.reshape(
                input_ids.shape[0], input_ids.shape[1], num_heads, head_dim * 2
            )
            trace_full_gate_output = q_and_gate[..., head_dim:].reshape(
                input_ids.shape[0], input_ids.shape[1], num_heads * head_dim
            ).cpu()

        def k_proj_hook(_module, _inputs, output):
            nonlocal trace_full_k_proj_output
            if capture_phase != "prefill":
                return
            trace_full_k_proj_output = capture_tensor(output)

        def v_proj_hook(_module, _inputs, output):
            nonlocal trace_full_v_proj_output
            nonlocal trace_full_prepared_value_output
            if capture_phase != "prefill":
                return
            tensor = capture_tensor(output)
            trace_full_v_proj_output = tensor
            head_dim = int(attn.head_dim)
            num_kv_heads = tensor.shape[-1] // head_dim
            trace_full_prepared_value_output = tensor.reshape(
                input_ids.shape[0], input_ids.shape[1], num_kv_heads, head_dim
            ).transpose(1, 2).contiguous().cpu()

        def o_proj_pre_hook(_module, inputs):
            nonlocal trace_full_attention_output
            if capture_phase != "prefill":
                return
            trace_full_attention_output = capture_tensor(inputs[0])

        return [
            attn.q_proj.register_forward_hook(q_proj_hook),
            attn.k_proj.register_forward_hook(k_proj_hook),
            attn.v_proj.register_forward_hook(v_proj_hook),
            attn.o_proj.register_forward_pre_hook(o_proj_pre_hook),
        ]

    def make_trace_mlp_hooks(layer_idx: int):
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp

        def post_attention_layernorm_hook(_module, _inputs, output):
            nonlocal trace_mlp_post_attention_layernorm_output
            if capture_phase != "prefill":
                return
            trace_mlp_post_attention_layernorm_output = capture_tensor(output)

        def gate_proj_hook(_module, _inputs, output):
            nonlocal trace_mlp_gate_proj_output
            if capture_phase != "prefill":
                return
            trace_mlp_gate_proj_output = capture_tensor(output)

        def up_proj_hook(_module, _inputs, output):
            nonlocal trace_mlp_up_proj_output
            if capture_phase != "prefill":
                return
            trace_mlp_up_proj_output = capture_tensor(output)

        def down_proj_pre_hook(_module, inputs):
            nonlocal trace_mlp_activated_hidden
            if capture_phase != "prefill":
                return
            trace_mlp_activated_hidden = capture_tensor(inputs[0])

        def down_proj_hook(_module, _inputs, output):
            nonlocal trace_mlp_down_proj_output
            if capture_phase != "prefill":
                return
            trace_mlp_down_proj_output = capture_tensor(output)

        return [
            layer.post_attention_layernorm.register_forward_hook(post_attention_layernorm_hook),
            mlp.gate_proj.register_forward_hook(gate_proj_hook),
            mlp.up_proj.register_forward_hook(up_proj_hook),
            mlp.down_proj.register_forward_pre_hook(down_proj_pre_hook),
            mlp.down_proj.register_forward_hook(down_proj_hook),
        ]

    def post_attention_layernorm_hook(_module, _inputs, output):
        nonlocal first_layer_post_attention_layernorm_output
        nonlocal decode_first_layer_post_attention_layernorm_output
        if capture_phase == "decode":
            decode_first_layer_post_attention_layernorm_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_post_attention_layernorm_output = capture_tensor(output)

    def mlp_hook(_module, _inputs, output):
        nonlocal first_layer_mlp_output
        nonlocal decode_first_layer_mlp_output
        if capture_phase == "decode":
            decode_first_layer_mlp_output = capture_tensor(output)
            return
        if capture_phase != "prefill":
            return
        first_layer_mlp_output = capture_tensor(output)

    def final_norm_hook(_module, _inputs, output):
        nonlocal decode_final_norm_output
        if capture_phase == "decode":
            decode_final_norm_output = capture_tensor(output)

    def final_norm_pre_hook(_module, inputs):
        nonlocal decode_final_norm_input
        nonlocal decode_final_norm_mean_square
        nonlocal decode_final_norm_rsqrt
        nonlocal decode_final_norm_weighted_hidden
        if capture_phase != "decode":
            return
        eps = getattr(_module, "variance_epsilon", getattr(_module, "eps"))
        hidden = inputs[0].detach().to(dtype=torch.float32).cpu()
        hidden_fp32 = inputs[0].detach().to(dtype=torch.float32)
        mean_square = hidden_fp32.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + eps)
        weight = (1.0 + _module.weight.detach().to(dtype=torch.float32)).view(1, 1, -1)
        weighted_hidden = (hidden_fp32 * rsqrt) * weight
        decode_final_norm_input = hidden
        decode_final_norm_mean_square = mean_square.cpu()
        decode_final_norm_rsqrt = rsqrt.cpu()
        decode_final_norm_weighted_hidden = weighted_hidden.cpu()

    def layer3_input_layernorm_hook(_module, _inputs, output):
        nonlocal layer3_input_layernorm_output
        nonlocal decode_layer3_input_layernorm_output
        if capture_phase == "decode":
            decode_layer3_input_layernorm_output = capture_tensor(output)
            return
        if capture_phase == "prefill":
            layer3_input_layernorm_output = capture_tensor(output)

    def layer3_input_layernorm_pre_hook(_module, inputs):
        nonlocal layer3_input_layernorm_input
        nonlocal decode_layer3_input_layernorm_input
        nonlocal layer3_input_layernorm_mean_square
        nonlocal decode_layer3_input_layernorm_mean_square
        nonlocal layer3_input_layernorm_rsqrt
        nonlocal decode_layer3_input_layernorm_rsqrt
        nonlocal layer3_input_layernorm_weight
        nonlocal layer3_input_layernorm_weighted_hidden
        nonlocal decode_layer3_input_layernorm_weighted_hidden
        eps = getattr(_module, "variance_epsilon", getattr(_module, "eps"))
        hidden = capture_tensor(inputs[0])
        hidden_f32 = inputs[0].detach().to(dtype=torch.float32)
        mean_square = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + eps)
        weighted_hidden = hidden_f32 * rsqrt
        weighted_hidden = weighted_hidden * (
            _module.weight.detach().to(dtype=torch.float32) + 1.0
        )
        if capture_phase == "decode":
            decode_layer3_input_layernorm_input = hidden
            decode_layer3_input_layernorm_mean_square = mean_square.cpu()
            decode_layer3_input_layernorm_rsqrt = rsqrt.cpu()
            decode_layer3_input_layernorm_weighted_hidden = weighted_hidden.cpu()
            return
        if capture_phase == "prefill":
            layer3_input_layernorm_input = hidden
            layer3_input_layernorm_mean_square = mean_square.cpu()
            layer3_input_layernorm_rsqrt = rsqrt.cpu()
            layer3_input_layernorm_weight = _module.weight.detach().to(dtype=torch.float32).cpu()
            layer3_input_layernorm_weighted_hidden = weighted_hidden.cpu()

    def layer3_token_mixer_hook(_module, _inputs, output):
        nonlocal layer3_token_mixer_output
        nonlocal decode_layer3_token_mixer_output
        if capture_phase == "decode":
            decode_layer3_token_mixer_output = capture_tensor(output)
            return
        if capture_phase == "prefill":
            layer3_token_mixer_output = capture_tensor(output)

    def layer3_q_proj_hook(_module, _inputs, output):
        nonlocal layer3_q_and_gate_output
        nonlocal layer3_gate_output
        if capture_phase != "prefill":
            return
        tensor = capture_tensor(output)
        layer3_q_and_gate_output = tensor
        layer3_attn = model.model.layers[3].self_attn
        head_dim = int(layer3_attn.head_dim)
        num_heads = tensor.shape[-1] // (head_dim * 2)
        q_and_gate = tensor.reshape(input_ids.shape[0], input_ids.shape[1], num_heads, head_dim * 2)
        layer3_gate_output = q_and_gate[..., head_dim:].reshape(
            input_ids.shape[0], input_ids.shape[1], num_heads * head_dim
        ).cpu()

    def layer3_k_proj_hook(_module, _inputs, output):
        nonlocal layer3_k_proj_output
        if capture_phase != "prefill":
            return
        layer3_k_proj_output = capture_tensor(output)

    def layer3_v_proj_hook(_module, _inputs, output):
        nonlocal layer3_v_proj_output
        nonlocal layer3_prepared_value_output
        if capture_phase != "prefill":
            return
        tensor = capture_tensor(output)
        layer3_v_proj_output = tensor
        layer3_attn = model.model.layers[3].self_attn
        head_dim = int(layer3_attn.head_dim)
        num_kv_heads = tensor.shape[-1] // head_dim
        layer3_prepared_value_output = tensor.reshape(
            input_ids.shape[0], input_ids.shape[1], num_kv_heads, head_dim
        ).transpose(1, 2).contiguous().cpu()

    def layer3_o_proj_pre_hook(_module, inputs):
        nonlocal layer3_attention_output
        if capture_phase != "prefill":
            return
        layer3_attention_output = capture_tensor(inputs[0])

    def layer3_post_attention_layernorm_hook(_module, _inputs, output):
        nonlocal layer3_post_attention_layernorm_output
        nonlocal decode_layer3_post_attention_layernorm_output
        if capture_phase == "decode":
            decode_layer3_post_attention_layernorm_output = capture_tensor(output)
            return
        if capture_phase == "prefill":
            layer3_post_attention_layernorm_output = capture_tensor(output)

    def layer3_mlp_hook(_module, _inputs, output):
        nonlocal layer3_mlp_output
        nonlocal decode_layer3_mlp_output
        if capture_phase == "decode":
            decode_layer3_mlp_output = capture_tensor(output)
            return
        if capture_phase == "prefill":
            layer3_mlp_output = capture_tensor(output)

    def layer3_hook(_module, _inputs, output):
        nonlocal layer3_output
        nonlocal decode_layer3_output
        tensor = output[0] if isinstance(output, tuple) else output
        if capture_phase == "decode":
            decode_layer3_output = tensor.detach().to(dtype=torch.float32).cpu()
            return
        if capture_phase == "prefill":
            layer3_output = tensor.detach().to(dtype=torch.float32).cpu()

    def layer4_input_layernorm_hook(_module, _inputs, output):
        nonlocal layer4_input_layernorm_output
        if capture_phase != "prefill":
            return
        layer4_input_layernorm_output = capture_tensor(output)

    def layer4_input_layernorm_pre_hook(_module, inputs):
        nonlocal layer4_input_layernorm_input
        nonlocal layer4_input_layernorm_mean_square
        nonlocal layer4_input_layernorm_rsqrt
        nonlocal layer4_input_layernorm_weight
        nonlocal layer4_input_layernorm_weighted_hidden
        if capture_phase != "prefill":
            return
        eps = getattr(_module, "variance_epsilon", getattr(_module, "eps"))
        hidden = capture_tensor(inputs[0])
        layer4_input_layernorm_input = hidden
        hidden_f32 = inputs[0].detach().to(dtype=torch.float32)
        mean_square = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + eps)
        weighted_hidden = hidden_f32 * rsqrt
        weighted_hidden = weighted_hidden * (
            _module.weight.detach().to(dtype=torch.float32) + 1.0
        )
        layer4_input_layernorm_mean_square = mean_square.cpu()
        layer4_input_layernorm_rsqrt = rsqrt.cpu()
        layer4_input_layernorm_weight = _module.weight.detach().to(dtype=torch.float32).cpu()
        layer4_input_layernorm_weighted_hidden = weighted_hidden.cpu()

    def layer4_token_mixer_hook(_module, _inputs, output):
        nonlocal layer4_token_mixer_output
        if capture_phase != "prefill":
            return
        layer4_token_mixer_output = capture_tensor(output)

    def layer4_post_attention_layernorm_hook(_module, _inputs, output):
        nonlocal layer4_post_attention_layernorm_output
        if capture_phase != "prefill":
            return
        layer4_post_attention_layernorm_output = capture_tensor(output)

    def layer4_mlp_hook(_module, _inputs, output):
        nonlocal layer4_mlp_output
        if capture_phase != "prefill":
            return
        layer4_mlp_output = capture_tensor(output)

    def layer4_hook(_module, _inputs, output):
        nonlocal layer4_output
        if capture_phase != "prefill":
            return
        tensor = output[0] if isinstance(output, tuple) else output
        layer4_output = tensor.detach().to(dtype=torch.float32).cpu()

    def layer23_input_layernorm_hook(_module, _inputs, output):
        nonlocal decode_layer23_input_layernorm_output
        if capture_phase == "decode":
            decode_layer23_input_layernorm_output = capture_tensor(output)

    def layer23_input_layernorm_pre_hook(_module, inputs):
        nonlocal decode_layer23_input_layernorm_input
        nonlocal decode_layer23_input_layernorm_mean_square
        nonlocal decode_layer23_input_layernorm_rsqrt
        nonlocal decode_layer23_input_layernorm_weighted_hidden
        if capture_phase != "decode":
            return
        eps = getattr(_module, "variance_epsilon", getattr(_module, "eps"))
        hidden = capture_tensor(inputs[0])
        hidden_f32 = inputs[0].detach().to(dtype=torch.float32)
        mean_square = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + eps)
        weight = (1.0 + _module.weight.detach().to(dtype=torch.float32)).view(1, 1, -1)
        weighted_hidden = (hidden_f32 * rsqrt) * weight
        decode_layer23_input_layernorm_input = hidden
        decode_layer23_input_layernorm_mean_square = mean_square.cpu()
        decode_layer23_input_layernorm_rsqrt = rsqrt.cpu()
        decode_layer23_input_layernorm_weighted_hidden = weighted_hidden.cpu()

    def layer23_token_mixer_hook(_module, _inputs, output):
        nonlocal decode_layer23_token_mixer_output
        if capture_phase == "decode":
            decode_layer23_token_mixer_output = capture_tensor(output)

    def layer23_post_attention_layernorm_hook(_module, _inputs, output):
        nonlocal decode_layer23_post_attention_layernorm_output
        if capture_phase == "decode":
            decode_layer23_post_attention_layernorm_output = capture_tensor(output)

    def layer23_mlp_gate_proj_hook(_module, _inputs, output):
        nonlocal decode_layer23_mlp_gate_proj_output
        if capture_phase == "decode":
            decode_layer23_mlp_gate_proj_output = capture_tensor(output)

    def layer23_mlp_up_proj_hook(_module, _inputs, output):
        nonlocal decode_layer23_mlp_up_proj_output
        if capture_phase == "decode":
            decode_layer23_mlp_up_proj_output = capture_tensor(output)

    def layer23_mlp_down_proj_pre_hook(_module, inputs):
        nonlocal decode_layer23_mlp_activated_hidden
        if capture_phase == "decode":
            decode_layer23_mlp_activated_hidden = capture_tensor(inputs[0])

    def layer23_mlp_down_proj_hook(_module, _inputs, output):
        nonlocal decode_layer23_mlp_down_proj_output
        if capture_phase == "decode":
            decode_layer23_mlp_down_proj_output = capture_tensor(output)

    def layer23_mlp_hook(_module, _inputs, output):
        nonlocal decode_layer23_mlp_output
        if capture_phase == "decode":
            decode_layer23_mlp_output = capture_tensor(output)

    def layer23_hook(_module, _inputs, output):
        nonlocal decode_layer23_output
        if capture_phase == "decode":
            tensor = output[0] if isinstance(output, tuple) else output
            decode_layer23_output = tensor.detach().to(dtype=torch.float32).cpu()

    def make_mid_layer_decode_hooks(layer_idx):
        """Create decode-only coarse hooks for a given layer, storing into mid_layer_captures."""
        layer = model.model.layers[layer_idx]
        prefix = f"decode_layer{layer_idx}"
        handles = []

        def input_layernorm_pre_hook(_module, inputs):
            if capture_phase != "decode":
                return
            eps = getattr(_module, "variance_epsilon", getattr(_module, "eps"))
            hidden = capture_tensor(inputs[0])
            hidden_f32 = inputs[0].detach().to(dtype=torch.float32)
            mean_square = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
            rsqrt_val = torch.rsqrt(mean_square + eps)
            weight = (1.0 + _module.weight.detach().to(dtype=torch.float32)).view(1, 1, -1)
            weighted_hidden = (hidden_f32 * rsqrt_val) * weight
            mid_layer_captures[f"{prefix}_input_layernorm_input"] = hidden
            mid_layer_captures[f"{prefix}_input_layernorm_mean_square"] = mean_square.cpu()
            mid_layer_captures[f"{prefix}_input_layernorm_rsqrt"] = rsqrt_val.cpu()
            mid_layer_captures[f"{prefix}_input_layernorm_weighted_hidden"] = weighted_hidden.cpu()

        def input_layernorm_hook(_module, _inputs, output):
            if capture_phase != "decode":
                return
            mid_layer_captures[f"{prefix}_input_layernorm_output"] = capture_tensor(output)

        def token_mixer_hook(_module, _inputs, output):
            if capture_phase != "decode":
                return
            mid_layer_captures[f"{prefix}_token_mixer_output"] = capture_tensor(output)

        def post_attention_layernorm_hook(_module, _inputs, output):
            if capture_phase != "decode":
                return
            mid_layer_captures[f"{prefix}_post_attention_layernorm_output"] = capture_tensor(output)

        def mlp_hook(_module, _inputs, output):
            if capture_phase != "decode":
                return
            mid_layer_captures[f"{prefix}_mlp_output"] = capture_tensor(output)

        def layer_hook(_module, _inputs, output):
            if capture_phase != "decode":
                return
            tensor = output[0] if isinstance(output, tuple) else output
            mid_layer_captures[f"{prefix}_output"] = tensor.detach().to(dtype=torch.float32).cpu()

        handles.append(layer.input_layernorm.register_forward_pre_hook(input_layernorm_pre_hook))
        handles.append(layer.input_layernorm.register_forward_hook(input_layernorm_hook))
        token_mixer_module = (
            layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        )
        handles.append(token_mixer_module.register_forward_hook(token_mixer_hook))
        handles.append(layer.post_attention_layernorm.register_forward_hook(post_attention_layernorm_hook))
        handles.append(layer.mlp.register_forward_hook(mlp_hook))
        handles.append(layer.register_forward_hook(layer_hook))
        return handles

    mid_layer_ids = [15, 16, 17, 18]
    mid_layer_handles: list[Any] = []
    for _mid_lid in mid_layer_ids:
        mid_layer_handles.extend(make_mid_layer_decode_hooks(_mid_lid))

    decoder_layer_outputs = [None] * len(model.model.layers)
    decode_decoder_layer_outputs = [None] * len(model.model.layers)

    embed_handle = model.model.embed_tokens.register_forward_hook(embed_hook)
    layer_handle = model.model.layers[0].register_forward_hook(layer_hook)
    decoder_layer_handles = [
        layer.register_forward_hook(make_decoder_layer_hook(layer_idx))
        for layer_idx, layer in enumerate(model.model.layers)
    ]
    input_layernorm_handle = model.model.layers[0].input_layernorm.register_forward_hook(
        input_layernorm_hook
    )
    token_mixer_handle = model.model.layers[0].linear_attn.register_forward_hook(
        token_mixer_hook
    )
    linear_qkv_handle = model.model.layers[0].linear_attn.in_proj_qkv.register_forward_hook(
        linear_qkv_hook
    )
    linear_z_handle = model.model.layers[0].linear_attn.in_proj_z.register_forward_hook(
        linear_z_hook
    )
    linear_b_handle = model.model.layers[0].linear_attn.in_proj_b.register_forward_hook(
        linear_b_hook
    )
    linear_a_handle = model.model.layers[0].linear_attn.in_proj_a.register_forward_hook(
        linear_a_hook
    )
    linear_conv_handle = model.model.layers[0].linear_attn.conv1d.register_forward_hook(
        linear_conv_hook
    )
    linear_norm_handle = model.model.layers[0].linear_attn.norm.register_forward_hook(
        linear_norm_hook
    )
    linear_norm_pre_handle = model.model.layers[0].linear_attn.norm.register_forward_pre_hook(
        linear_norm_pre_hook
    )
    post_attention_layernorm_handle = (
        model.model.layers[0]
        .post_attention_layernorm.register_forward_hook(post_attention_layernorm_hook)
    )
    mlp_handle = model.model.layers[0].mlp.register_forward_hook(mlp_hook)
    final_norm_pre_handle = model.model.norm.register_forward_pre_hook(final_norm_pre_hook)
    final_norm_handle = model.model.norm.register_forward_hook(final_norm_hook)
    layer3_input_layernorm_handle = (
        model.model.layers[3]
        .input_layernorm.register_forward_hook(layer3_input_layernorm_hook)
    )
    layer3_input_layernorm_pre_handle = (
        model.model.layers[3]
        .input_layernorm.register_forward_pre_hook(layer3_input_layernorm_pre_hook)
    )
    layer3_token_mixer_handle = model.model.layers[3].self_attn.register_forward_hook(
        layer3_token_mixer_hook
    )
    layer3_q_proj_handle = model.model.layers[3].self_attn.q_proj.register_forward_hook(
        layer3_q_proj_hook
    )
    layer3_k_proj_handle = model.model.layers[3].self_attn.k_proj.register_forward_hook(
        layer3_k_proj_hook
    )
    layer3_v_proj_handle = model.model.layers[3].self_attn.v_proj.register_forward_hook(
        layer3_v_proj_hook
    )
    layer3_o_proj_pre_handle = model.model.layers[3].self_attn.o_proj.register_forward_pre_hook(
        layer3_o_proj_pre_hook
    )
    layer3_post_attention_layernorm_handle = (
        model.model.layers[3]
        .post_attention_layernorm.register_forward_hook(layer3_post_attention_layernorm_hook)
    )
    layer3_mlp_handle = model.model.layers[3].mlp.register_forward_hook(layer3_mlp_hook)
    layer3_handle = model.model.layers[3].register_forward_hook(layer3_hook)
    layer4_input_layernorm_handle = (
        model.model.layers[4]
        .input_layernorm.register_forward_hook(layer4_input_layernorm_hook)
    )
    layer4_input_layernorm_pre_handle = (
        model.model.layers[4]
        .input_layernorm.register_forward_pre_hook(layer4_input_layernorm_pre_hook)
    )
    layer4_token_mixer_handle = model.model.layers[4].linear_attn.register_forward_hook(
        layer4_token_mixer_hook
    )
    layer4_post_attention_layernorm_handle = (
        model.model.layers[4]
        .post_attention_layernorm.register_forward_hook(layer4_post_attention_layernorm_hook)
    )
    layer4_mlp_handle = model.model.layers[4].mlp.register_forward_hook(layer4_mlp_hook)
    layer4_handle = model.model.layers[4].register_forward_hook(layer4_hook)
    layer23_input_layernorm_handle = (
        model.model.layers[23]
        .input_layernorm.register_forward_hook(layer23_input_layernorm_hook)
    )
    layer23_input_layernorm_pre_handle = (
        model.model.layers[23]
        .input_layernorm.register_forward_pre_hook(layer23_input_layernorm_pre_hook)
    )
    layer23_token_mixer_module = (
        model.model.layers[23].linear_attn
        if hasattr(model.model.layers[23], "linear_attn")
        else model.model.layers[23].self_attn
    )
    layer23_token_mixer_handle = layer23_token_mixer_module.register_forward_hook(
        layer23_token_mixer_hook
    )
    layer23_post_attention_layernorm_handle = (
        model.model.layers[23]
        .post_attention_layernorm.register_forward_hook(layer23_post_attention_layernorm_hook)
    )
    layer23_mlp_gate_proj_handle = model.model.layers[23].mlp.gate_proj.register_forward_hook(
        layer23_mlp_gate_proj_hook
    )
    layer23_mlp_up_proj_handle = model.model.layers[23].mlp.up_proj.register_forward_hook(
        layer23_mlp_up_proj_hook
    )
    layer23_mlp_down_proj_pre_handle = (
        model.model.layers[23]
        .mlp.down_proj.register_forward_pre_hook(layer23_mlp_down_proj_pre_hook)
    )
    layer23_mlp_down_proj_handle = model.model.layers[23].mlp.down_proj.register_forward_hook(
        layer23_mlp_down_proj_hook
    )
    layer23_mlp_handle = model.model.layers[23].mlp.register_forward_hook(layer23_mlp_hook)
    layer23_handle = model.model.layers[23].register_forward_hook(layer23_hook)
    trace_linear_handles: list[Any] = []
    if trace_linear_layer_idx is not None:
        if not hasattr(model.model.layers[trace_linear_layer_idx], "linear_attn"):
            raise RuntimeError(
                f"trace-linear-layer={trace_linear_layer_idx} is not a linear-attention layer"
            )
        trace_linear_handles = make_trace_linear_hooks(trace_linear_layer_idx)
    trace_full_handles: list[Any] = []
    if trace_full_layer_idx is not None:
        if not hasattr(model.model.layers[trace_full_layer_idx], "self_attn"):
            raise RuntimeError(
                f"trace-full-layer={trace_full_layer_idx} is not a full-attention layer"
            )
        trace_full_handles = make_trace_full_hooks(trace_full_layer_idx)
    trace_mlp_handles: list[Any] = []
    if trace_mlp_layer_idx is not None:
        if trace_mlp_layer_idx < 0 or trace_mlp_layer_idx >= len(model.model.layers):
            raise RuntimeError(
                f"trace-mlp-layer={trace_mlp_layer_idx} is out of range for {len(model.model.layers)} layers"
            )
        trace_mlp_handles = make_trace_mlp_hooks(trace_mlp_layer_idx)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
            prefill_last_token_logits = (
                outputs.logits[0, -1, :].to(dtype=torch.float32).cpu().tolist()
            )
            past_key_values = outputs.past_key_values
            next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            if args.max_new_tokens > 0:
                capture_phase = "decode"
                decode_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=target_device)
                decode_first_layer_conv_state_before = (
                    past_key_values.layers[0].conv_states.detach().to(dtype=torch.float32).clone()
                )
                decode_first_layer_recurrent_state_before = (
                    past_key_values.layers[0].recurrent_states.detach().to(dtype=torch.float32).clone()
                )
                decode_outputs = model(
                    input_ids=decode_input_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                first_decode_step_logits = (
                    decode_outputs.logits[0, -1, :].to(dtype=torch.float32).cpu().tolist()
                )
                decode_final_hidden_output = (
                    decode_outputs.hidden_states[-1]
                    .detach()
                    .to(dtype=torch.float32)
                    .cpu()
                )
            else:
                first_decode_step_logits = None
    finally:
        embed_handle.remove()
        layer_handle.remove()
        input_layernorm_handle.remove()
        token_mixer_handle.remove()
        linear_qkv_handle.remove()
        linear_z_handle.remove()
        linear_b_handle.remove()
        linear_a_handle.remove()
        linear_conv_handle.remove()
        linear_norm_pre_handle.remove()
        linear_norm_handle.remove()
        post_attention_layernorm_handle.remove()
        mlp_handle.remove()
        final_norm_pre_handle.remove()
        final_norm_handle.remove()
        layer3_input_layernorm_pre_handle.remove()
        layer3_input_layernorm_handle.remove()
        layer3_token_mixer_handle.remove()
        layer3_q_proj_handle.remove()
        layer3_k_proj_handle.remove()
        layer3_v_proj_handle.remove()
        layer3_o_proj_pre_handle.remove()
        layer3_post_attention_layernorm_handle.remove()
        layer3_mlp_handle.remove()
        layer3_handle.remove()
        layer4_input_layernorm_handle.remove()
        layer4_input_layernorm_pre_handle.remove()
        layer4_token_mixer_handle.remove()
        layer4_post_attention_layernorm_handle.remove()
        layer4_mlp_handle.remove()
        layer4_handle.remove()
        layer23_input_layernorm_pre_handle.remove()
        layer23_input_layernorm_handle.remove()
        layer23_token_mixer_handle.remove()
        layer23_post_attention_layernorm_handle.remove()
        layer23_mlp_gate_proj_handle.remove()
        layer23_mlp_up_proj_handle.remove()
        layer23_mlp_down_proj_pre_handle.remove()
        layer23_mlp_down_proj_handle.remove()
        layer23_mlp_handle.remove()
        layer23_handle.remove()
        for handle in trace_linear_handles:
            handle.remove()
        for handle in trace_full_handles:
            handle.remove()
        for handle in trace_mlp_handles:
            handle.remove()
        for handle in mid_layer_handles:
            handle.remove()
        for handle in decoder_layer_handles:
            handle.remove()

    if trace_full_q_and_gate_output is not None and trace_full_k_proj_output is not None:
        trace_full_attn = model.model.layers[trace_full_layer_idx].self_attn
        head_dim = int(trace_full_attn.head_dim)
        num_heads = trace_full_q_and_gate_output.shape[-1] // (head_dim * 2)
        num_kv_heads = trace_full_k_proj_output.shape[-1] // head_dim
        q = trace_full_q_and_gate_output.reshape(
            input_ids.shape[0], input_ids.shape[1], num_heads, head_dim * 2
        )[..., :head_dim]
        k = trace_full_k_proj_output.reshape(
            input_ids.shape[0], input_ids.shape[1], num_kv_heads, head_dim
        )
        q_weight = trace_full_attn.q_norm.weight.detach().to(dtype=torch.float32)
        k_weight = trace_full_attn.k_norm.weight.detach().to(dtype=torch.float32)
        q_eps = getattr(trace_full_attn.q_norm, "variance_epsilon", getattr(trace_full_attn.q_norm, "eps"))
        k_eps = getattr(trace_full_attn.k_norm, "variance_epsilon", getattr(trace_full_attn.k_norm, "eps"))
        q_ms = q.pow(2).mean(dim=-1, keepdim=True)
        k_ms = k.pow(2).mean(dim=-1, keepdim=True)
        q_normed = q * torch.rsqrt(q_ms + q_eps)
        k_normed = k * torch.rsqrt(k_ms + k_eps)
        trace_full_prepared_query_output = (
            q_normed * (q_weight + 1.0)
        ).transpose(1, 2).contiguous().cpu()
        trace_full_prepared_key_output = (
            k_normed * (k_weight + 1.0)
        ).transpose(1, 2).contiguous().cpu()

    if layer3_q_and_gate_output is not None and layer3_k_proj_output is not None:
        layer3_attn = model.model.layers[3].self_attn
        head_dim = int(layer3_attn.head_dim)
        num_heads = layer3_q_and_gate_output.shape[-1] // (head_dim * 2)
        num_kv_heads = layer3_k_proj_output.shape[-1] // head_dim
        q = layer3_q_and_gate_output.reshape(
            input_ids.shape[0], input_ids.shape[1], num_heads, head_dim * 2
        )[..., :head_dim]
        k = layer3_k_proj_output.reshape(
            input_ids.shape[0], input_ids.shape[1], num_kv_heads, head_dim
        )
        q_weight = layer3_attn.q_norm.weight.detach().to(dtype=torch.float32)
        k_weight = layer3_attn.k_norm.weight.detach().to(dtype=torch.float32)
        q_eps = getattr(layer3_attn.q_norm, "variance_epsilon", getattr(layer3_attn.q_norm, "eps"))
        k_eps = getattr(layer3_attn.k_norm, "variance_epsilon", getattr(layer3_attn.k_norm, "eps"))
        q_ms = q.pow(2).mean(dim=-1, keepdim=True)
        k_ms = k.pow(2).mean(dim=-1, keepdim=True)
        q_normed = q * torch.rsqrt(q_ms + q_eps)
        k_normed = k * torch.rsqrt(k_ms + k_eps)
        layer3_prepared_query_output = (
            q_normed * (q_weight + 1.0)
        ).transpose(1, 2).contiguous().cpu()
        layer3_prepared_key_output = (
            k_normed * (k_weight + 1.0)
        ).transpose(1, 2).contiguous().cpu()

    missing = []
    required_scalars = {
        "embedding_output": embedding_output,
        "first_layer_output": first_layer_output,
        "first_layer_input_layernorm_output": first_layer_input_layernorm_output,
        "first_layer_linear_qkv_output": first_layer_linear_qkv_output,
        "first_layer_linear_z_output": first_layer_linear_z_output,
        "first_layer_linear_b_output": first_layer_linear_b_output,
        "first_layer_linear_a_output": first_layer_linear_a_output,
        "first_layer_linear_conv_weight": first_layer_linear_conv_weight,
        "first_layer_linear_pre_conv_value_focus_head_output": first_layer_linear_pre_conv_value_focus_head_output,
        "first_layer_linear_post_conv_output": first_layer_linear_post_conv_output,
        "first_layer_linear_direct_conv_output": first_layer_linear_direct_conv_output,
        "first_layer_linear_prepared_query_output": first_layer_linear_prepared_query_output,
        "first_layer_linear_prepared_key_output": first_layer_linear_prepared_key_output,
        "first_layer_linear_prepared_value_output": first_layer_linear_prepared_value_output,
        "first_layer_linear_prepared_beta_output": first_layer_linear_prepared_beta_output,
        "first_layer_linear_prepared_g_output": first_layer_linear_prepared_g_output,
        "first_layer_linear_direct_recurrent_output": first_layer_linear_direct_recurrent_output,
        "first_layer_linear_focus_kv_mem_output": first_layer_linear_focus_kv_mem_output,
        "first_layer_linear_focus_delta_output": first_layer_linear_focus_delta_output,
        "first_layer_linear_focus_state_output": first_layer_linear_focus_state_output,
        "first_layer_linear_focus_output": first_layer_linear_focus_output,
        "first_layer_linear_focus_kv_mem_steps": first_layer_linear_focus_kv_mem_steps,
        "first_layer_linear_focus_delta_steps": first_layer_linear_focus_delta_steps,
        "first_layer_linear_focus_state_steps": first_layer_linear_focus_state_steps,
        "first_layer_linear_focus_output_steps": first_layer_linear_focus_output_steps,
        "first_layer_linear_prepared_value_focus_head_output": first_layer_linear_prepared_value_focus_head_output,
        "first_layer_linear_pre_norm_output": first_layer_linear_pre_norm_output,
        "first_layer_linear_pre_norm_mean_square": first_layer_linear_pre_norm_mean_square,
        "first_layer_linear_pre_norm_rsqrt": first_layer_linear_pre_norm_rsqrt,
        "first_layer_linear_pre_norm_focus_head_output": first_layer_linear_pre_norm_focus_head_output,
        "first_layer_linear_norm_gate_input": first_layer_linear_norm_gate_input,
        "first_layer_linear_norm_weight": first_layer_linear_norm_weight,
        "first_layer_linear_norm_weighted_hidden": first_layer_linear_norm_weighted_hidden,
        "first_layer_linear_norm_silu_gate": first_layer_linear_norm_silu_gate,
        "first_layer_linear_norm_output": first_layer_linear_norm_output,
        "first_layer_token_mixer_output": first_layer_token_mixer_output,
        "first_layer_post_attention_layernorm_output": first_layer_post_attention_layernorm_output,
        "first_layer_mlp_output": first_layer_mlp_output,
        "layer3_input_layernorm_output": layer3_input_layernorm_output,
        "layer3_input_layernorm_input": layer3_input_layernorm_input,
        "layer3_input_layernorm_mean_square": layer3_input_layernorm_mean_square,
        "layer3_input_layernorm_rsqrt": layer3_input_layernorm_rsqrt,
        "layer3_input_layernorm_weight": layer3_input_layernorm_weight,
        "layer3_input_layernorm_weighted_hidden": layer3_input_layernorm_weighted_hidden,
        "layer3_q_and_gate_output": layer3_q_and_gate_output,
        "layer3_k_proj_output": layer3_k_proj_output,
        "layer3_v_proj_output": layer3_v_proj_output,
        "layer3_prepared_query_output": layer3_prepared_query_output,
        "layer3_gate_output": layer3_gate_output,
        "layer3_prepared_key_output": layer3_prepared_key_output,
        "layer3_prepared_value_output": layer3_prepared_value_output,
        "layer3_attention_output": layer3_attention_output,
        "layer3_token_mixer_output": layer3_token_mixer_output,
        "layer3_post_attention_layernorm_output": layer3_post_attention_layernorm_output,
        "layer3_mlp_output": layer3_mlp_output,
        "layer3_output": layer3_output,
        "layer4_input_layernorm_output": layer4_input_layernorm_output,
        "layer4_input_layernorm_input": layer4_input_layernorm_input,
        "layer4_input_layernorm_mean_square": layer4_input_layernorm_mean_square,
        "layer4_input_layernorm_rsqrt": layer4_input_layernorm_rsqrt,
        "layer4_input_layernorm_weight": layer4_input_layernorm_weight,
        "layer4_input_layernorm_weighted_hidden": layer4_input_layernorm_weighted_hidden,
        "layer4_token_mixer_output": layer4_token_mixer_output,
        "layer4_post_attention_layernorm_output": layer4_post_attention_layernorm_output,
        "layer4_mlp_output": layer4_mlp_output,
        "layer4_output": layer4_output,
        "decode_layer23_input_layernorm_output": decode_layer23_input_layernorm_output,
        "decode_layer23_input_layernorm_input": decode_layer23_input_layernorm_input,
        "decode_layer23_input_layernorm_mean_square": decode_layer23_input_layernorm_mean_square,
        "decode_layer23_input_layernorm_rsqrt": decode_layer23_input_layernorm_rsqrt,
        "decode_layer23_input_layernorm_weighted_hidden": decode_layer23_input_layernorm_weighted_hidden,
        "decode_layer23_token_mixer_output": decode_layer23_token_mixer_output,
        "decode_layer23_post_attention_layernorm_output": decode_layer23_post_attention_layernorm_output,
        "decode_layer23_mlp_gate_proj_output": decode_layer23_mlp_gate_proj_output,
        "decode_layer23_mlp_up_proj_output": decode_layer23_mlp_up_proj_output,
        "decode_layer23_mlp_activated_hidden": decode_layer23_mlp_activated_hidden,
        "decode_layer23_mlp_down_proj_output": decode_layer23_mlp_down_proj_output,
        "decode_layer23_mlp_output": decode_layer23_mlp_output,
        "decode_layer23_output": decode_layer23_output,
    }
    missing.extend(name for name, value in required_scalars.items() if value is None)
    if any(layer_output is None for layer_output in decoder_layer_outputs):
        missing.append("decoder_layer_outputs")
    if args.max_new_tokens > 0:
        required_decode = {
            "decode_decoder_layer_outputs": None if any(layer_output is None for layer_output in decode_decoder_layer_outputs) else True,
            "decode_first_layer_input_layernorm_output": decode_first_layer_input_layernorm_output,
            "decode_first_layer_linear_qkv_output": decode_first_layer_linear_qkv_output,
            "decode_first_layer_linear_z_output": decode_first_layer_linear_z_output,
            "decode_first_layer_linear_b_output": decode_first_layer_linear_b_output,
            "decode_first_layer_linear_a_output": decode_first_layer_linear_a_output,
            "decode_first_layer_linear_prepared_query_output": decode_first_layer_linear_prepared_query_output,
            "decode_first_layer_linear_prepared_key_output": decode_first_layer_linear_prepared_key_output,
            "decode_first_layer_linear_prepared_value_output": decode_first_layer_linear_prepared_value_output,
            "decode_first_layer_linear_prepared_beta_output": decode_first_layer_linear_prepared_beta_output,
            "decode_first_layer_linear_prepared_g_output": decode_first_layer_linear_prepared_g_output,
            "decode_first_layer_linear_direct_recurrent_output": decode_first_layer_linear_direct_recurrent_output,
            "decode_first_layer_linear_pre_norm_output": decode_first_layer_linear_pre_norm_output,
            "decode_first_layer_linear_norm_output": decode_first_layer_linear_norm_output,
            "decode_first_layer_token_mixer_output": decode_first_layer_token_mixer_output,
            "decode_first_layer_post_attention_layernorm_output": decode_first_layer_post_attention_layernorm_output,
            "decode_first_layer_mlp_output": decode_first_layer_mlp_output,
            "decode_first_layer_output": decode_first_layer_output,
            "decode_final_hidden_output": decode_final_hidden_output,
            "decode_layer3_input_layernorm_output": decode_layer3_input_layernorm_output,
            "decode_layer3_input_layernorm_input": decode_layer3_input_layernorm_input,
            "decode_layer3_input_layernorm_mean_square": decode_layer3_input_layernorm_mean_square,
            "decode_layer3_input_layernorm_rsqrt": decode_layer3_input_layernorm_rsqrt,
            "decode_layer3_input_layernorm_weighted_hidden": decode_layer3_input_layernorm_weighted_hidden,
            "decode_layer3_token_mixer_output": decode_layer3_token_mixer_output,
            "decode_layer3_post_attention_layernorm_output": decode_layer3_post_attention_layernorm_output,
            "decode_layer3_mlp_output": decode_layer3_mlp_output,
            "decode_layer3_output": decode_layer3_output,
            "decode_layer23_input_layernorm_output": decode_layer23_input_layernorm_output,
            "decode_layer23_input_layernorm_input": decode_layer23_input_layernorm_input,
            "decode_layer23_input_layernorm_mean_square": decode_layer23_input_layernorm_mean_square,
            "decode_layer23_input_layernorm_rsqrt": decode_layer23_input_layernorm_rsqrt,
            "decode_layer23_input_layernorm_weighted_hidden": decode_layer23_input_layernorm_weighted_hidden,
            "decode_layer23_token_mixer_output": decode_layer23_token_mixer_output,
            "decode_layer23_post_attention_layernorm_output": decode_layer23_post_attention_layernorm_output,
            "decode_layer23_mlp_gate_proj_output": decode_layer23_mlp_gate_proj_output,
            "decode_layer23_mlp_up_proj_output": decode_layer23_mlp_up_proj_output,
            "decode_layer23_mlp_activated_hidden": decode_layer23_mlp_activated_hidden,
            "decode_layer23_mlp_down_proj_output": decode_layer23_mlp_down_proj_output,
            "decode_layer23_mlp_output": decode_layer23_mlp_output,
            "decode_layer23_output": decode_layer23_output,
        }
        mid_layer_suffixes = [
            "input_layernorm_input", "input_layernorm_output",
            "input_layernorm_mean_square", "input_layernorm_rsqrt",
            "input_layernorm_weighted_hidden", "token_mixer_output",
            "post_attention_layernorm_output", "mlp_output", "output",
        ]
        for lid in mid_layer_ids:
            for suffix in mid_layer_suffixes:
                key = f"decode_layer{lid}_{suffix}"
                required_decode[key] = mid_layer_captures.get(key)
        missing.extend(name for name, value in required_decode.items() if value is None)
    if missing:
        raise RuntimeError(
            "failed to capture staged first-layer outputs from PyTorch model; missing="
            + ",".join(missing)
        )

    decode_started = time.perf_counter()
    decode_logits: list[list[float]] = []
    generated_token_ids: list[int] = []
    past_key_values = outputs.past_key_values
    next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
    for _ in range(args.max_new_tokens):
        generated_token_ids.append(next_token)
        decode_input_ids = torch.tensor([[next_token]], dtype=torch.long)
        with torch.no_grad():
            outputs = model(
                input_ids=decode_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        step_logits = outputs.logits[0, -1, :].to(dtype=torch.float32).cpu().tolist()
        decode_logits.append(step_logits)
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        past_key_values = outputs.past_key_values
    decode_elapsed_ms = (time.perf_counter() - decode_started) * 1000.0

    payload = {
        "load_ms": load_elapsed_ms,
        "prefill_ms": prefill_elapsed_ms,
        "decode_ms": decode_elapsed_ms,
        "embedding_output": embedding_output.tolist(),
        "first_layer_output": first_layer_output.tolist(),
        "first_layer_input_layernorm_output": first_layer_input_layernorm_output.tolist(),
        "first_layer_linear_qkv_output": first_layer_linear_qkv_output.tolist(),
        "first_layer_linear_pre_conv_value_focus_head_output": first_layer_linear_pre_conv_value_focus_head_output.tolist(),
        "first_layer_linear_z_output": first_layer_linear_z_output.tolist(),
        "first_layer_linear_b_output": first_layer_linear_b_output.tolist(),
        "first_layer_linear_a_output": first_layer_linear_a_output.tolist(),
        "first_layer_linear_conv_weight": first_layer_linear_conv_weight.tolist(),
        "first_layer_linear_post_conv_output": first_layer_linear_post_conv_output.tolist(),
        "first_layer_linear_direct_conv_output": first_layer_linear_direct_conv_output.tolist(),
        "first_layer_linear_prepared_query_output": first_layer_linear_prepared_query_output.tolist(),
        "first_layer_linear_prepared_key_output": first_layer_linear_prepared_key_output.tolist(),
        "first_layer_linear_prepared_value_output": first_layer_linear_prepared_value_output.tolist(),
        "first_layer_linear_prepared_beta_output": first_layer_linear_prepared_beta_output.tolist(),
        "first_layer_linear_prepared_g_output": first_layer_linear_prepared_g_output.tolist(),
        "first_layer_linear_direct_recurrent_output": first_layer_linear_direct_recurrent_output.tolist(),
        "first_layer_linear_focus_kv_mem_output": first_layer_linear_focus_kv_mem_output.tolist(),
        "first_layer_linear_focus_delta_output": first_layer_linear_focus_delta_output.tolist(),
        "first_layer_linear_focus_state_output": first_layer_linear_focus_state_output.tolist(),
        "first_layer_linear_focus_output": first_layer_linear_focus_output.tolist(),
        "first_layer_linear_focus_kv_mem_steps": [tensor.tolist() for tensor in first_layer_linear_focus_kv_mem_steps],
        "first_layer_linear_focus_delta_steps": [tensor.tolist() for tensor in first_layer_linear_focus_delta_steps],
        "first_layer_linear_focus_state_steps": [tensor.tolist() for tensor in first_layer_linear_focus_state_steps],
        "first_layer_linear_focus_output_steps": [tensor.tolist() for tensor in first_layer_linear_focus_output_steps],
        "first_layer_linear_prepared_value_focus_head_output": first_layer_linear_prepared_value_focus_head_output.tolist(),
        "first_layer_linear_pre_norm_output": first_layer_linear_pre_norm_output.tolist(),
        "first_layer_linear_pre_norm_mean_square": first_layer_linear_pre_norm_mean_square.tolist(),
        "first_layer_linear_pre_norm_rsqrt": first_layer_linear_pre_norm_rsqrt.tolist(),
        "first_layer_linear_pre_norm_focus_head_output": first_layer_linear_pre_norm_focus_head_output.tolist(),
        "first_layer_linear_norm_gate_input": first_layer_linear_norm_gate_input.tolist(),
        "first_layer_linear_norm_weight": first_layer_linear_norm_weight.tolist(),
        "first_layer_linear_norm_weighted_hidden": first_layer_linear_norm_weighted_hidden.tolist(),
        "first_layer_linear_norm_silu_gate": first_layer_linear_norm_silu_gate.tolist(),
        "first_layer_linear_norm_output": first_layer_linear_norm_output.tolist(),
        "first_layer_token_mixer_output": first_layer_token_mixer_output.tolist(),
        "first_layer_post_attention_layernorm_output": first_layer_post_attention_layernorm_output.tolist(),
        "first_layer_mlp_output": first_layer_mlp_output.tolist(),
        "layer3_input_layernorm_output": layer3_input_layernorm_output.tolist(),
        "layer3_input_layernorm_input": layer3_input_layernorm_input.tolist(),
        "layer3_input_layernorm_mean_square": layer3_input_layernorm_mean_square.tolist(),
        "layer3_input_layernorm_rsqrt": layer3_input_layernorm_rsqrt.tolist(),
        "layer3_input_layernorm_weight": layer3_input_layernorm_weight.tolist(),
        "layer3_input_layernorm_weighted_hidden": layer3_input_layernorm_weighted_hidden.tolist(),
        "layer3_q_and_gate_output": layer3_q_and_gate_output.tolist(),
        "layer3_k_proj_output": layer3_k_proj_output.tolist(),
        "layer3_v_proj_output": layer3_v_proj_output.tolist(),
        "layer3_prepared_query_output": layer3_prepared_query_output.tolist(),
        "layer3_gate_output": layer3_gate_output.tolist(),
        "layer3_prepared_key_output": layer3_prepared_key_output.tolist(),
        "layer3_prepared_value_output": layer3_prepared_value_output.tolist(),
        "layer3_attention_output": layer3_attention_output.tolist(),
        "layer3_token_mixer_output": layer3_token_mixer_output.tolist(),
        "layer3_post_attention_layernorm_output": layer3_post_attention_layernorm_output.tolist(),
        "layer3_mlp_output": layer3_mlp_output.tolist(),
        "layer3_output": layer3_output.tolist(),
        "layer4_input_layernorm_output": layer4_input_layernorm_output.tolist(),
        "layer4_input_layernorm_input": layer4_input_layernorm_input.tolist(),
        "layer4_input_layernorm_mean_square": layer4_input_layernorm_mean_square.tolist(),
        "layer4_input_layernorm_rsqrt": layer4_input_layernorm_rsqrt.tolist(),
        "layer4_input_layernorm_weight": layer4_input_layernorm_weight.tolist(),
        "layer4_input_layernorm_weighted_hidden": layer4_input_layernorm_weighted_hidden.tolist(),
        "layer4_token_mixer_output": layer4_token_mixer_output.tolist(),
        "layer4_post_attention_layernorm_output": layer4_post_attention_layernorm_output.tolist(),
        "layer4_mlp_output": layer4_mlp_output.tolist(),
        "layer4_output": layer4_output.tolist(),
        "decoder_layer_outputs": [layer_output.tolist() for layer_output in decoder_layer_outputs],
        "decode_decoder_layer_outputs": [
            layer_output.tolist() for layer_output in decode_decoder_layer_outputs
        ] if args.max_new_tokens > 0 else [],
        "decode_first_layer_input_layernorm_output": decode_first_layer_input_layernorm_output.tolist() if decode_first_layer_input_layernorm_output is not None else None,
        "decode_first_layer_linear_qkv_output": decode_first_layer_linear_qkv_output.tolist() if decode_first_layer_linear_qkv_output is not None else None,
        "decode_first_layer_linear_z_output": decode_first_layer_linear_z_output.tolist() if decode_first_layer_linear_z_output is not None else None,
        "decode_first_layer_linear_b_output": decode_first_layer_linear_b_output.tolist() if decode_first_layer_linear_b_output is not None else None,
        "decode_first_layer_linear_a_output": decode_first_layer_linear_a_output.tolist() if decode_first_layer_linear_a_output is not None else None,
        "decode_first_layer_linear_prepared_query_output": decode_first_layer_linear_prepared_query_output.tolist() if decode_first_layer_linear_prepared_query_output is not None else None,
        "decode_first_layer_linear_prepared_key_output": decode_first_layer_linear_prepared_key_output.tolist() if decode_first_layer_linear_prepared_key_output is not None else None,
        "decode_first_layer_linear_prepared_value_output": decode_first_layer_linear_prepared_value_output.tolist() if decode_first_layer_linear_prepared_value_output is not None else None,
        "decode_first_layer_linear_prepared_beta_output": decode_first_layer_linear_prepared_beta_output.tolist() if decode_first_layer_linear_prepared_beta_output is not None else None,
        "decode_first_layer_linear_prepared_g_output": decode_first_layer_linear_prepared_g_output.tolist() if decode_first_layer_linear_prepared_g_output is not None else None,
        "decode_first_layer_linear_direct_recurrent_output": decode_first_layer_linear_direct_recurrent_output.tolist() if decode_first_layer_linear_direct_recurrent_output is not None else None,
        "decode_first_layer_conv_state_before": decode_first_layer_conv_state_before.tolist() if decode_first_layer_conv_state_before is not None else None,
        "decode_first_layer_recurrent_state_before": decode_first_layer_recurrent_state_before.tolist() if decode_first_layer_recurrent_state_before is not None else None,
        "decode_first_layer_linear_pre_norm_output": decode_first_layer_linear_pre_norm_output.tolist() if decode_first_layer_linear_pre_norm_output is not None else None,
        "decode_first_layer_linear_norm_output": decode_first_layer_linear_norm_output.tolist() if decode_first_layer_linear_norm_output is not None else None,
        "decode_first_layer_token_mixer_output": decode_first_layer_token_mixer_output.tolist() if decode_first_layer_token_mixer_output is not None else None,
        "decode_first_layer_post_attention_layernorm_output": decode_first_layer_post_attention_layernorm_output.tolist() if decode_first_layer_post_attention_layernorm_output is not None else None,
        "decode_first_layer_mlp_output": decode_first_layer_mlp_output.tolist() if decode_first_layer_mlp_output is not None else None,
        "decode_first_layer_output": decode_first_layer_output.tolist() if decode_first_layer_output is not None else None,
        "decode_final_hidden_output": decode_final_hidden_output.tolist() if decode_final_hidden_output is not None else None,
        "decode_final_norm_input": decode_final_norm_input.tolist() if decode_final_norm_input is not None else None,
        "decode_final_norm_mean_square": decode_final_norm_mean_square.tolist() if decode_final_norm_mean_square is not None else None,
        "decode_final_norm_rsqrt": decode_final_norm_rsqrt.tolist() if decode_final_norm_rsqrt is not None else None,
        "decode_final_norm_weighted_hidden": decode_final_norm_weighted_hidden.tolist() if decode_final_norm_weighted_hidden is not None else None,
        "decode_final_norm_output": decode_final_norm_output.tolist() if decode_final_norm_output is not None else None,
        "decode_layer3_input_layernorm_output": decode_layer3_input_layernorm_output.tolist() if decode_layer3_input_layernorm_output is not None else None,
        "decode_layer3_input_layernorm_input": decode_layer3_input_layernorm_input.tolist() if decode_layer3_input_layernorm_input is not None else None,
        "decode_layer3_input_layernorm_mean_square": decode_layer3_input_layernorm_mean_square.tolist() if decode_layer3_input_layernorm_mean_square is not None else None,
        "decode_layer3_input_layernorm_rsqrt": decode_layer3_input_layernorm_rsqrt.tolist() if decode_layer3_input_layernorm_rsqrt is not None else None,
        "decode_layer3_input_layernorm_weighted_hidden": decode_layer3_input_layernorm_weighted_hidden.tolist() if decode_layer3_input_layernorm_weighted_hidden is not None else None,
        "decode_layer3_token_mixer_output": decode_layer3_token_mixer_output.tolist() if decode_layer3_token_mixer_output is not None else None,
        "decode_layer3_post_attention_layernorm_output": decode_layer3_post_attention_layernorm_output.tolist() if decode_layer3_post_attention_layernorm_output is not None else None,
        "decode_layer3_mlp_output": decode_layer3_mlp_output.tolist() if decode_layer3_mlp_output is not None else None,
        "decode_layer3_output": decode_layer3_output.tolist() if decode_layer3_output is not None else None,
        "decode_layer23_input_layernorm_output": decode_layer23_input_layernorm_output.tolist() if decode_layer23_input_layernorm_output is not None else None,
        "decode_layer23_input_layernorm_input": decode_layer23_input_layernorm_input.tolist() if decode_layer23_input_layernorm_input is not None else None,
        "decode_layer23_input_layernorm_mean_square": decode_layer23_input_layernorm_mean_square.tolist() if decode_layer23_input_layernorm_mean_square is not None else None,
        "decode_layer23_input_layernorm_rsqrt": decode_layer23_input_layernorm_rsqrt.tolist() if decode_layer23_input_layernorm_rsqrt is not None else None,
        "decode_layer23_input_layernorm_weighted_hidden": decode_layer23_input_layernorm_weighted_hidden.tolist() if decode_layer23_input_layernorm_weighted_hidden is not None else None,
        "decode_layer23_token_mixer_output": decode_layer23_token_mixer_output.tolist() if decode_layer23_token_mixer_output is not None else None,
        "decode_layer23_post_attention_layernorm_output": decode_layer23_post_attention_layernorm_output.tolist() if decode_layer23_post_attention_layernorm_output is not None else None,
        "decode_layer23_mlp_gate_proj_output": decode_layer23_mlp_gate_proj_output.tolist() if decode_layer23_mlp_gate_proj_output is not None else None,
        "decode_layer23_mlp_up_proj_output": decode_layer23_mlp_up_proj_output.tolist() if decode_layer23_mlp_up_proj_output is not None else None,
        "decode_layer23_mlp_activated_hidden": decode_layer23_mlp_activated_hidden.tolist() if decode_layer23_mlp_activated_hidden is not None else None,
        "decode_layer23_mlp_down_proj_output": decode_layer23_mlp_down_proj_output.tolist() if decode_layer23_mlp_down_proj_output is not None else None,
        "decode_layer23_mlp_output": decode_layer23_mlp_output.tolist() if decode_layer23_mlp_output is not None else None,
        "decode_layer23_output": decode_layer23_output.tolist() if decode_layer23_output is not None else None,
        "trace_linear_layer": trace_linear_layer_idx,
        "trace_full_layer": trace_full_layer_idx,
        "trace_mlp_layer": trace_mlp_layer_idx,
        "trace_linear_input_layernorm_output": trace_linear_input_layernorm_output.tolist() if trace_linear_input_layernorm_output is not None else None,
        "trace_linear_qkv_output": trace_linear_qkv_output.tolist() if trace_linear_qkv_output is not None else None,
        "trace_linear_z_output": trace_linear_z_output.tolist() if trace_linear_z_output is not None else None,
        "trace_linear_post_conv_output": trace_linear_post_conv_output.tolist() if trace_linear_post_conv_output is not None else None,
        "trace_linear_prepared_query_output": trace_linear_prepared_query_output.tolist() if trace_linear_prepared_query_output is not None else None,
        "trace_linear_prepared_key_output": trace_linear_prepared_key_output.tolist() if trace_linear_prepared_key_output is not None else None,
        "trace_linear_prepared_value_output": trace_linear_prepared_value_output.tolist() if trace_linear_prepared_value_output is not None else None,
        "trace_linear_prepared_beta_output": trace_linear_prepared_beta_output.tolist() if trace_linear_prepared_beta_output is not None else None,
        "trace_linear_prepared_g_output": trace_linear_prepared_g_output.tolist() if trace_linear_prepared_g_output is not None else None,
        "trace_linear_direct_recurrent_output": trace_linear_direct_recurrent_output.tolist() if trace_linear_direct_recurrent_output is not None else None,
        "trace_linear_norm_output": trace_linear_norm_output.tolist() if trace_linear_norm_output is not None else None,
        "trace_linear_token_mixer_output": trace_linear_token_mixer_output.tolist() if trace_linear_token_mixer_output is not None else None,
        "trace_full_q_and_gate_output": trace_full_q_and_gate_output.tolist() if trace_full_q_and_gate_output is not None else None,
        "trace_full_gate_output": trace_full_gate_output.tolist() if trace_full_gate_output is not None else None,
        "trace_full_k_proj_output": trace_full_k_proj_output.tolist() if trace_full_k_proj_output is not None else None,
        "trace_full_v_proj_output": trace_full_v_proj_output.tolist() if trace_full_v_proj_output is not None else None,
        "trace_full_prepared_query_output": trace_full_prepared_query_output.tolist() if trace_full_prepared_query_output is not None else None,
        "trace_full_prepared_key_output": trace_full_prepared_key_output.tolist() if trace_full_prepared_key_output is not None else None,
        "trace_full_prepared_value_output": trace_full_prepared_value_output.tolist() if trace_full_prepared_value_output is not None else None,
        "trace_full_attention_output": trace_full_attention_output.tolist() if trace_full_attention_output is not None else None,
        "trace_mlp_post_attention_layernorm_output": trace_mlp_post_attention_layernorm_output.tolist() if trace_mlp_post_attention_layernorm_output is not None else None,
        "trace_mlp_gate_proj_output": trace_mlp_gate_proj_output.tolist() if trace_mlp_gate_proj_output is not None else None,
        "trace_mlp_up_proj_output": trace_mlp_up_proj_output.tolist() if trace_mlp_up_proj_output is not None else None,
        "trace_mlp_activated_hidden": trace_mlp_activated_hidden.tolist() if trace_mlp_activated_hidden is not None else None,
        "trace_mlp_down_proj_output": trace_mlp_down_proj_output.tolist() if trace_mlp_down_proj_output is not None else None,
    }
    for lid in mid_layer_ids:
        for suffix in [
            "input_layernorm_input", "input_layernorm_output",
            "input_layernorm_mean_square", "input_layernorm_rsqrt",
            "input_layernorm_weighted_hidden", "token_mixer_output",
            "post_attention_layernorm_output", "mlp_output", "output",
        ]:
            key = f"decode_layer{lid}_{suffix}"
            val = mid_layer_captures.get(key)
            payload[key] = val.tolist() if val is not None else None
    payload.update({
        "prefill_last_token_logits": prefill_last_token_logits,
        "first_decode_step_last_token_logits": first_decode_step_logits,
        "decode_last_token_logits": decode_logits,
        "generated_token_ids": generated_token_ids,
    })
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
