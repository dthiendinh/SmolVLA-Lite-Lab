"""Parity against the UNMODIFIED LeRobot file using REAL tiny Transformers layers.

Only config/processor downloads are replaced with local objects. Attention, RoPE,
RMSNorm, MLP, vision encoder, connector, autograd and DynamicCache are real.
Run: PYTHONPATH=. .venv-smolvlm/bin/python -B tests/test_smolvlm_with_expert.py
Optional LEROBOT_SMOLVLM_PATH selects another source file for comparison.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import DynamicCache
import transformers

from smolvla_scratch.models import smolvla as ours
from smolvla_scratch.models import backbone, embeddings, action_expert, attention
from examples.smolvlm_expert_steps import tiny_model, tiny_vlm_config, block_mask, positions, VelocityExample

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reference/smolvlm_with_expert.py"
DEFAULT_SOURCE = SNAPSHOT if SNAPSHOT.exists() else ROOT.parent / "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py"
SOURCE = Path(os.environ.get("LEROBOT_SMOLVLM_PATH", DEFAULT_SOURCE))
spec = importlib.util.spec_from_file_location("original_lerobot_smolvlm", SOURCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
METRICS = {}


def original_model(**kwargs):
    with patch.object(reference.AutoConfig, "from_pretrained", return_value=tiny_vlm_config()), patch.object(
        reference.AutoProcessor, "from_pretrained", return_value=SimpleNamespace(tokenizer=None)
    ):
        return reference.SmolVLMWithExpertModel(load_vlm_weights=False, **kwargs)


def model_pair(**kwargs):
    expected = original_model(**kwargs)
    actual = tiny_model(**kwargs)
    actual.load_state_dict(expected.state_dict(), strict=True)
    actual.eval()
    expected.eval()  # Original train()/eval() returns None, so do NOT chain it.
    return actual, expected


def inputs(model, prefix_length=7, suffix_length=5, padded=True):
    batch = 2
    prefix = torch.randn(batch, prefix_length, model.config.text_config.hidden_size)
    suffix = torch.randn(batch, suffix_length, model.expert_hidden_size)
    pad = torch.ones(batch, prefix_length + suffix_length, dtype=torch.bool)
    if padded:
        pad[1, prefix_length - 3:prefix_length - 1] = False
    boundaries = torch.zeros_like(pad)
    boundaries[:, prefix_length - 1:] = True  # state, followed by causal action tokens
    return prefix, suffix, pad, boundaries


def cache_lengths(cache):
    return [layer.get_seq_length() for layer in cache.layers]


class SmolVLMExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)

    def close(self, actual, expected, label=None, exact=True):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 2e-5, atol=0 if exact else 2e-6)
        if label:
            METRICS[label] = {"shape": list(actual.shape), "max_abs_error": (actual - expected).abs().max().item()}

    def full(self, model, prefix, suffix, pad, boundaries):
        return model(inputs_embeds=[prefix, suffix], attention_mask=block_mask(pad, boundaries),
                     position_ids=positions(pad), use_cache=False)[0]

    def prefill(self, model, prefix, pad, boundaries):
        return model(inputs_embeds=[prefix, None], attention_mask=block_mask(pad, boundaries),
                     position_ids=positions(pad), use_cache=True)[1]

    def suffix_step(self, model, suffix, pad, cache):
        spad = torch.ones(suffix.shape[:2], dtype=torch.bool)
        mask = torch.cat([pad[:, None, :].expand(-1, suffix.shape[1], -1), block_mask(spad, spad)], -1)
        return model(inputs_embeds=[None, suffix], attention_mask=mask,
                     position_ids=pad.sum(-1)[:, None] + positions(spad), past_key_values=cache,
                     use_cache=True)[0][1]

    def test_reference_snapshot_identity(self):
        pinned_sha = '7a97d06b9aeb83d0f4a2990b32aec5086d38613feb2b39dd59105c9dc2f57141'
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), pinned_sha)

    def test_rope_values_dtypes_and_gradients(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 5, 4, 8, dtype=dtype, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                pos = torch.tensor([[0, 1, 2, 4, 5], [-1, 0, 2, 3, 7]])
                a, b = embeddings.apply_rope(x, pos), reference.apply_rope(y, pos)
                self.close(a, b)
                a.float().square().sum().backward()
                b.float().square().sum().backward()
                self.close(x.grad, y.grad)
        x = torch.randn(2, 5, 4, 8)
        self.close(embeddings.apply_rope(x, torch.zeros(2, 5)), x)

    def test_intermediate_sizes_preserve_original_rounding(self):
        for width in (24, 32, 720, 768, 960):
            self.assertEqual(action_expert.get_intermediate_size(width), reference.get_intermediate_size(width))
        self.assertEqual(action_expert.get_intermediate_size(720), 2048)  # int BEFORE multiply

    def test_constructor_geometry_and_removed_embeddings(self):
        a, b = model_pair(attention_mode='cross_attn', self_attn_every_n_layers=2, expert_width_multiplier=.75)
        self.assertEqual(a.expert_hidden_size, 24)
        self.assertIsNone(a.lm_expert.embed_tokens)
        self.assertIs(a.get_vlm_model(), a.vlm.model)
        self.assertEqual(a.lm_expert.layers[0].self_attn.k_proj.in_features, 24)
        self.assertEqual(a.lm_expert.layers[1].self_attn.k_proj.in_features, 16)
        self.assertEqual(a.get_attention_interface(), a.eager_attention_forward)
        for left, right in zip(a.parameters(), b.parameters()):
            self.close(left, right)

    def test_pretrained_loader_receives_dtype_and_model_id(self):
        model = reference.SmolVLMForConditionalGeneration(tiny_vlm_config())
        processor = object()
        with patch.object(backbone.AutoModelForImageTextToText, 'from_pretrained', return_value=model) as loader:
            result = ours.SmolVLMWithExpertModel(model_id='local-test', processor=processor, dtype=torch.bfloat16)
        loader.assert_called_once_with('local-test', dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.assertIs(result.vlm, model)
        self.assertIs(result.processor, processor)

    def test_image_and_language_embedding_parity(self):
        a, b = model_pair()
        image = torch.randn(2, 3, 16, 16)
        tokens = torch.tensor([[1, 3, 5], [2, 4, 0]])
        self.close(a.embed_image(image), b.embed_image(image), 'image_embedding')
        self.close(a.embed_language_tokens(tokens), b.embed_language_tokens(tokens), 'language_embedding')
        self.assertEqual(tuple(a.embed_image(image).shape), (2, 4, 32))

    def test_eager_grouped_attention_matches_reference_and_sdpa(self):
        a, b = model_pair()
        q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 6, 2, 8), torch.randn(2, 6, 2, 8)
        mask = torch.rand(2, 3, 6) > .4
        mask[:, :, 0] = True
        out = a.eager_attention_forward(mask, 2, 8, q, k, v)
        self.close(out, b.eager_attention_forward(mask, 2, 8, q, k, v), 'eager_attention')
        independent = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask[:, None], enable_gqa=True,
        ).transpose(1, 2).reshape(2, 3, 32)
        self.close(out, independent, exact=False)

    def test_full_forward_self_cross_and_interleaved_parity(self):
        for mode, every in [('self_attn', -1), ('cross_attn', -1), ('cross_attn', 2), ('cross_attn', 3)]:
            with self.subTest(mode=mode, every=every):
                a, b = model_pair(attention_mode=mode, self_attn_every_n_layers=every, expert_width_multiplier=.75)
                args = inputs(a)
                left, right = self.full(a, *args), self.full(b, *args)
                for stream in range(2):
                    self.close(left[stream], right[stream], f'full_{mode}_{every}_stream{stream}')

    def test_full_loss_and_parameter_gradients_parity(self):
        for frozen in (True, False):
            with self.subTest(train_expert_only=frozen):
                a, b = model_pair(attention_mode='cross_attn', self_attn_every_n_layers=2,
                                  expert_width_multiplier=.75, train_expert_only=frozen, freeze_vision_encoder=True)
                a.train()
                b.train()
                args = inputs(a)
                target = torch.randn(2, 5, 24)
                left, right = self.full(a, *args)[1], self.full(b, *args)[1]
                la, lb = (left-target).square().mean(), (right-target).square().mean()
                self.close(la, lb, f'loss_frozen{frozen}')
                la.backward()
                lb.backward()
                grad_count = 0
                for (name, p), (other_name, q) in zip(a.named_parameters(), b.named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertEqual(p.requires_grad, q.requires_grad)
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        self.close(p.grad, q.grad)
                        grad_count += 1
                self.assertGreater(grad_count, 20)
                METRICS[f'gradients_frozen{frozen}'] = {'compared_parameter_tensors': grad_count, 'max_abs_error': 0.0}

    def test_freezing_and_train_eval_contract(self):
        for expert_only, vision in [(True, False), (False, True), (False, False)]:
            a, b = model_pair(train_expert_only=expert_only, freeze_vision_encoder=vision)
            self.assertIs(a.train(), a)
            b.train()
            self.assertEqual(a.vlm.training, b.vlm.training)
            self.assertEqual(a.vlm.model.vision_model.training, b.vlm.model.vision_model.training)
            self.assertTrue(a.lm_expert.training)
            for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
                self.assertEqual(name, other)
                self.assertEqual(p.requires_grad, q.requires_grad)
            self.assertIs(a.eval(), a)
            self.assertFalse(a.lm_expert.training)

    def test_reduced_expert_layer_mapping_parity(self):
        for mode, every in [('self_attn', -1), ('cross_attn', -1)]:
            a, b = model_pair(attention_mode=mode, self_attn_every_n_layers=every,
                              num_vlm_layers=4, num_expert_layers=2, expert_width_multiplier=.75)
            mapped = a.get_model_layers([a.vlm.model.text_model, a.lm_expert])[1]
            self.assertIs(mapped[0], a.lm_expert.layers[0])
            self.assertIs(mapped[2], a.lm_expert.layers[1])
            self.assertIsNone(mapped[1])
            self.assertIsNone(mapped[3])
            args = inputs(a)
            for x, y in zip(self.full(a, *args), self.full(b, *args)):
                self.close(x, y)

    def test_reduced_interleaved_projection_regression(self):
        kwargs = dict(attention_mode='cross_attn', self_attn_every_n_layers=2,
                      num_expert_layers=2, expert_width_multiplier=.75)
        a, b = tiny_model(**kwargs), original_model(**kwargs)
        # local expert 1 runs at global VLM layer 2: that is SELF attention.
        self.assertEqual(a.lm_expert.layers[1].self_attn.k_proj.in_features, 24)
        self.assertEqual(b.lm_expert.layers[1].self_attn.k_proj.in_features, 16)
        args = inputs(a)
        self.assertTrue(self.full(a, *args)[1].isfinite().all())
        with self.assertRaises(RuntimeError):
            self.full(b, *args)

    def test_prefix_cache_values_and_repeated_denoising_parity(self):
        with torch.no_grad():
            for mode, every in [('self_attn', -1), ('cross_attn', -1), ('cross_attn', 2)]:
                with self.subTest(mode=mode, every=every):
                    a, b = model_pair(attention_mode=mode, self_attn_every_n_layers=every, expert_width_multiplier=.75)
                    prefix, suffix, pad, boundaries = inputs(a)
                    ppad, pbound = pad[:, :7], boundaries[:, :7]
                    ca, cb = self.prefill(a, prefix, ppad, pbound), self.prefill(b, prefix, ppad, pbound)
                    for la, lb in zip(ca.layers, cb.layers):
                        self.close(la.keys, lb.keys)
                        self.close(la.values, lb.values)
                    original_keys = [l.keys.clone() for l in ca.layers]
                    for step in range(3):
                        current = suffix + step * .1
                        left, right = self.suffix_step(a, current, ppad, ca), self.suffix_step(b, current, ppad, cb)
                        self.close(left, right, f'cache_{mode}_{every}_step{step}')
                        attention.crop_prefix_cache(ca, 7)
                        # Exactly the caller-side cleanup in LeRobot modeling_smolvla.py.
                        for layer in cb.layers:
                            layer.crop(7 - layer.get_seq_length())
                        self.assertEqual(cache_lengths(ca), [7]*4)
                        for layer, old in zip(ca.layers, original_keys):
                            self.close(layer.keys, old)

    def test_cached_and_uncached_suffix_agree(self):
        with torch.no_grad():
            for mode, every, depth in [('self_attn', -1, -1), ('cross_attn', 2, -1), ('cross_attn', -1, 2)]:
                a = tiny_model(attention_mode=mode, self_attn_every_n_layers=every, num_expert_layers=depth,
                               expert_width_multiplier=.75)
                a.eval()
                prefix, suffix, pad, boundaries = inputs(a)
                full = self.full(a, prefix, suffix, pad, boundaries)[1]
                cache = self.prefill(a, prefix, pad[:, :7], boundaries[:, :7])
                cached = self.suffix_step(a, suffix, pad[:, :7], cache)
                self.close(cached, full, f'cached_vs_full_{mode}_{every}_{depth}', exact=False)

    def test_causal_mask_prevents_future_action_leakage(self):
        a = tiny_model(attention_mode='cross_attn', self_attn_every_n_layers=2, expert_width_multiplier=.75)
        args = inputs(a)
        first = self.full(a, *args)
        changed = args[1].clone()
        changed[:, 3:] += 100
        second = self.full(a, args[0], changed, args[2], args[3])
        # All-masked padded query rows have undefined semantic output in the
        # original eager implementation. Assert invariance on valid tokens only.
        self.close(first[0][args[2][:, :7]], second[0][args[2][:, :7]])
        self.close(first[1][:, :3], second[1][:, :3])
        self.assertGreater((first[1][:, 3:]-second[1][:, 3:]).abs().max().item(), 1e-3)

    def test_padding_keys_do_not_affect_actions(self):
        a = tiny_model(attention_mode='cross_attn', self_attn_every_n_layers=2, expert_width_multiplier=.75)
        prefix, suffix, pad, boundaries = inputs(a)
        first = self.full(a, prefix, suffix, pad, boundaries)[1]
        changed = prefix.clone()
        changed[1, 4:6] += 100
        second = self.full(a, changed, suffix, pad, boundaries)[1]
        self.close(first, second)

    def test_prefix_only_without_cache_is_supported(self):
        a = tiny_model(attention_mode='cross_attn', self_attn_every_n_layers=2)
        prefix, _, pad, boundaries = inputs(a)
        out, cache = a(inputs_embeds=[prefix, None], attention_mask=block_mask(pad[:, :7], boundaries[:, :7]),
                       position_ids=positions(pad[:, :7]), use_cache=False)
        self.assertIsNone(out[1])
        self.assertIsNone(cache)
        self.assertEqual(tuple(out[0].shape), (2, 7, 32))

    def test_bridge_velocity_loss_and_ten_step_action_parity(self):
        a, b = model_pair(attention_mode='cross_attn', self_attn_every_n_layers=2,
                          expert_width_multiplier=.75, freeze_vision_encoder=True)
        pa, pb = VelocityExample(a), VelocityExample(b)
        pb.load_state_dict(pa.state_dict())
        pa.eval()
        pb.eval()
        image, state = torch.randn(2, 3, 16, 16), torch.randn(2, 6)
        tokens = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 0, 0]])
        args = (image, tokens, tokens != 0, state)
        prefix_a, prefix_b = pa.prefix(*args), pb.prefix(*args)
        x, time = torch.randn(2, 50, 32), torch.tensor([.2, .8])
        va, vb = pa(*prefix_a, x, time), pb(*prefix_b, x, time)
        self.close(va, vb, 'bridge_velocity')
        loss = va[..., :6].square().mean()
        loss.backward()
        # Freezing VLM weights must NOT cut gradients to an upstream trainable state projection.
        self.assertGreater(pa.state_proj.weight.grad.abs().max().item(), 0)
        self.assertTrue(all(p.grad is None for p in a.vlm.parameters()))
        self.assertTrue(any(p.grad is not None for p in a.lm_expert.parameters()))
        with torch.no_grad():
            ra, rb = pa.sample(*prefix_a, x, 10), pb.sample(*prefix_b, x, 10)
        self.assertEqual(tuple(ra.shape), (2, 50, 6))
        self.assertTrue(ra.isfinite().all())
        self.close(ra, rb, 'bridge_10_step_actions')
        METRICS['sample_first_action_ours'] = ra[0, 0].tolist()
        METRICS['sample_first_action_lerobot_core'] = rb[0, 0].tolist()

    def test_invalid_inputs_fail_early(self):
        for kwargs in [dict(attention_mode='typo'), dict(expert_width_multiplier=0),
                       dict(expert_width_multiplier=.001), dict(num_expert_layers=3), dict(num_expert_layers=5)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tiny_model(**kwargs)
        a = tiny_model(attention_mode='cross_attn', self_attn_every_n_layers=2)
        prefix, suffix, pad, boundaries = inputs(a)
        kwargs = dict(inputs_embeds=[prefix, suffix], attention_mask=block_mask(pad, boundaries), position_ids=positions(pad))
        for extra in [dict(inputs_embeds=None), dict(inputs_embeds=[None, None]),
                      dict(use_cache=True), dict(attention_mask=torch.ones(2, 12, 12)),
                      dict(position_ids=torch.zeros(2, 3)), dict(past_key_values=DynamicCache(), use_cache=True)]:
            with self.subTest(extra=list(extra)), self.assertRaises(ValueError):
                a(**(kwargs | extra))
        with self.assertRaises(ValueError):
            embeddings.apply_rope(torch.zeros(2, 3, 4, 7), torch.zeros(2, 3))

    def test_modular_composition_owns_each_parameter_once(self):
        core = tiny_model(expert_width_multiplier=.75)
        policy = ours.SmolVLA(core)
        self.assertIsInstance(policy.embeddings, embeddings.PrefixSuffixEmbeddings)
        self.assertIsInstance(core.attention, attention.AttentionEngine)
        keys = list(policy.state_dict())
        for prefix in ('core.vlm.', 'core.lm_expert.', 'embeddings.state_proj.',
                       'embeddings.action_in_proj.', 'action_out_proj.'):
            self.assertTrue(any(k.startswith(prefix) for k in keys), prefix)
        # Shared VLM must not be registered again inside embeddings or attention.
        named = list(policy.named_parameters(remove_duplicate=False))
        self.assertEqual(len(named), len({id(p) for _, p in named}))
        self.assertFalse((ROOT/'smolvla_scratch/models/smolvlm_with_expert.py').exists())

    def test_time_embedding_against_lerobot_utility(self):
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding as original_time_embedding
        for time in (torch.tensor([0., .2, .8, 1.]), torch.tensor([[0., .2], [.8, 1.]])):
            left = embeddings.create_sinusoidal_pos_embedding(time, 24, device='cpu')
            right = original_time_embedding(time, 24, 4e-3, 4.0, device=torch.device('cpu'))
            self.close(left, right, f'time_embedding_rank{time.ndim}')

    def test_modular_policy_training_loss_and_gradients(self):
        a, b = model_pair(attention_mode='cross_attn', self_attn_every_n_layers=2,
                          expert_width_multiplier=.75, freeze_vision_encoder=True)
        pa, pb = ours.SmolVLA(a), ours.SmolVLA(b)
        pb.load_state_dict(pa.state_dict())
        image, state = torch.randn(2, 3, 16, 16), torch.randn(2, 6)
        tokens = torch.tensor([[3, 4, 5], [6, 7, 0]])
        args = (image, tokens, tokens != 0, state)
        ap, bp = pa.prefix(*args), pb.prefix(*args)
        actions = torch.nn.functional.pad(torch.randn(2, 50, 6), (0, 26))
        noise, time = torch.randn_like(actions), torch.tensor([.2, .8])
        pad = torch.zeros(2, 50, dtype=torch.bool)
        pad[0, -8:] = True
        actual = pa.training_loss(*ap, actions, noise=noise, time=time, action_is_pad=pad)
        # The reference SmolVLA recipe, with the same standalone six-dimension/pad reduction.
        x_t = time[:, None, None] * noise + (1-time[:, None, None]) * actions
        target = noise-actions
        errors = (pb(*bp, x_t, time)-target).square()[..., :6]
        expected = (errors * (~pad)[..., None]).sum() / ((~pad).sum()*6)
        self.close(actual, expected, 'modular_policy_masked_loss')
        actual.backward()
        expected.backward()
        for (name, p), (other, q) in zip(pa.named_parameters(), pb.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                self.close(p.grad, q.grad)
        # All-padded batches produce a differentiable zero, not NaN.
        pa.zero_grad(set_to_none=True)
        ap = pa.prefix(*args)
        zero = pa.training_loss(*ap, actions, noise=noise, time=time, action_is_pad=torch.ones_like(pad))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        for p in pa.parameters():
            if p.grad is not None:
                self.assertEqual(torch.count_nonzero(p.grad).item(), 0)

    def test_modular_policy_cached_and_uncached_sampling(self):
        policy = ours.SmolVLA(tiny_model(attention_mode='cross_attn', self_attn_every_n_layers=2,
                                         expert_width_multiplier=.75))
        policy.eval()
        tokens = torch.tensor([[3, 4, 5], [6, 7, 0]])
        with torch.no_grad():
            prefix = policy.prefix(torch.randn(2, 3, 16, 16), tokens, tokens != 0, torch.randn(2, 6))
            noise = torch.randn(2, 50, 32)
            cached = policy.sample(*prefix, noise, num_steps=3, use_cache=True)
            full = policy.sample(*prefix, noise, num_steps=3, use_cache=False)
        self.close(cached, full, 'modular_policy_cached_vs_full_actions', exact=False)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable in this execution session')
    def test_mps_forward_parity(self):
        a, b = model_pair(attention_mode='cross_attn', self_attn_every_n_layers=2, expert_width_multiplier=.75)
        a.to('mps')
        b.to('mps')
        args = [x.to('mps') for x in inputs(a)]
        with torch.no_grad():
            for x, y in zip(self.full(a, *args), self.full(b, *args)):
                self.close(x, y, exact=False)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SmolVLMExpertTests))
    report = dict(torch=torch.__version__, transformers=transformers.__version__, device='cpu',
                  module_path=str(Path(ours.__file__).resolve()), reference=str(SOURCE), reference_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  tests_run=result.testsRun, passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), metrics=METRICS,
                  scope='Five modular project files, real tiny Transformers modules; identical core/wrapper weights; no pretrained checkpoint or robot evaluation')
    Path(os.environ.get('SMOLVLM_TEST_REPORT', ROOT/'SMOLVLM_TEST_RESULTS.json')).write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
