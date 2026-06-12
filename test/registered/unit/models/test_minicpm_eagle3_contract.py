from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import types
import unittest

import torch
from torch import nn

from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3, LlamaModel
from sglang.srt.models.minicpm import MiniCPMModel, MiniCPMSALAForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AddLayer(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, positions, hidden_states, forward_batch, residual):
        return hidden_states + self.delta, None


class _IdentityNorm(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _FakeEmbedding(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, input_ids):
        return torch.full(
            (input_ids.numel(), self.value.shape[-1]),
            self.value.item(),
            dtype=torch.float32,
        )


class _RecordEmbedsLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_embeds = None

    def forward(self, positions, embeds, hidden_states, forward_batch, residual):
        self.seen_embeds = embeds
        return hidden_states, residual


class _LlamaNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, hidden_states


class _ForwardMode:
    def is_extend(self):
        return False


class _FixedOutputModel:
    def __init__(self, output):
        self.embed_tokens = object()
        self._output = output

    def __call__(self, input_ids, positions, forward_batch, input_embeds):
        return self._output


class _RecordingLogitsProcessor:
    def __init__(self):
        self.hidden_states = None
        self.aux_hidden_states = None

    def __call__(self, input_ids, hidden_states, lm_head, forward_batch, aux):
        self.hidden_states = hidden_states
        self.aux_hidden_states = aux
        return hidden_states


class TestMiniCPMEagle3TargetContract(CustomTestCase):
    def test_set_eagle3_layers_to_capture_uses_next_layer_ids(self):
        model = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
        model.config = types.SimpleNamespace(num_hidden_layers=32)
        model.model = types.SimpleNamespace(layers_to_capture=[])
        model.capture_aux_hidden_states = False

        model.set_eagle3_layers_to_capture([1, 10, 22])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, [2, 11, 23])

    def test_set_eagle3_layers_to_capture_default_layers(self):
        model = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
        model.config = types.SimpleNamespace(num_hidden_layers=32)
        model.model = types.SimpleNamespace(layers_to_capture=[])
        model.capture_aux_hidden_states = False

        model.set_eagle3_layers_to_capture()

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, [2, 16, 29])

    def test_set_eagle3_layers_to_capture_rejects_empty_and_oob_layers(self):
        model = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
        model.config = types.SimpleNamespace(num_hidden_layers=4)
        model.model = types.SimpleNamespace(layers_to_capture=[])
        model.capture_aux_hidden_states = False

        with self.assertRaisesRegex(ValueError, "at least one"):
            model.set_eagle3_layers_to_capture([])
        with self.assertRaisesRegex(ValueError, "0..3"):
            model.set_eagle3_layers_to_capture([4])
        with self.assertRaisesRegex(ValueError, "0..3"):
            model.set_eagle3_layers_to_capture([-1])

    def test_get_embed_and_head_handles_tied_embeddings(self):
        weight = torch.empty(4, 8)
        model = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
        model.config = types.SimpleNamespace(tie_word_embeddings=True)
        model.model = types.SimpleNamespace(
            embed_tokens=types.SimpleNamespace(weight=weight)
        )

        embed, head = model.get_embed_and_head()

        self.assertIs(embed, weight)
        self.assertIs(head, weight)

    def test_get_embed_and_head_handles_untied_lm_head(self):
        embed_weight = torch.empty(4, 8)
        head_weight = torch.empty(4, 8)
        model = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
        model.config = types.SimpleNamespace(tie_word_embeddings=False)
        model.model = types.SimpleNamespace(
            embed_tokens=types.SimpleNamespace(weight=embed_weight)
        )
        model.lm_head = types.SimpleNamespace(weight=head_weight)

        embed, head = model.get_embed_and_head()

        self.assertIs(embed, embed_weight)
        self.assertIs(head, head_weight)

    def test_minicpm_model_captures_previous_layer_output_without_residual(self):
        model = MiniCPMModel.__new__(MiniCPMModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([_AddLayer(1.0), _AddLayer(10.0)])
        model.norm = _IdentityNorm()
        model.layers_to_capture = [1]
        model.embed_tokens = None
        model.config = types.SimpleNamespace(scale_emb=1.0)

        hidden_states, aux_hidden_states = model.forward(
            input_ids=None,
            positions=None,
            forward_batch=None,
            input_embeds=torch.tensor([[3.0]]),
        )

        self.assertTrue(torch.equal(hidden_states, torch.tensor([[14.0]])))
        self.assertEqual(len(aux_hidden_states), 1)
        self.assertTrue(torch.equal(aux_hidden_states[0], torch.tensor([[4.0]])))

    def test_minicpm_model_can_capture_final_layer_output(self):
        model = MiniCPMModel.__new__(MiniCPMModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([_AddLayer(1.0), _AddLayer(10.0)])
        model.norm = _IdentityNorm()
        model.layers_to_capture = [2]
        model.embed_tokens = None
        model.config = types.SimpleNamespace(scale_emb=1.0)

        hidden_states, aux_hidden_states = model.forward(
            input_ids=None,
            positions=None,
            forward_batch=None,
            input_embeds=torch.tensor([[3.0]]),
        )

        self.assertTrue(torch.equal(hidden_states, torch.tensor([[14.0]])))
        self.assertEqual(len(aux_hidden_states), 1)
        self.assertTrue(torch.equal(aux_hidden_states[0], torch.tensor([[14.0]])))

    def test_forward_divides_main_hidden_by_scale_width_but_not_aux(self):
        aux_hidden = torch.tensor([[3.0]])
        model = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
        model.config = types.SimpleNamespace(tie_word_embeddings=True)
        model.model = _FixedOutputModel((torch.tensor([[8.0]]), [aux_hidden]))
        model.capture_aux_hidden_states = True
        model.scale_width = 4.0
        model.logits_processor = _RecordingLogitsProcessor()

        model.forward(input_ids=torch.tensor([0]), positions=None, forward_batch=None)

        self.assertTrue(
            torch.equal(model.logits_processor.hidden_states, torch.tensor([[2.0]]))
        )
        self.assertIs(model.logits_processor.aux_hidden_states[0], aux_hidden)


class TestLlamaEagle3DraftContract(CustomTestCase):
    def test_draft_embedding_applies_scale_emb(self):
        model = LlamaModel.__new__(LlamaModel)
        nn.Module.__init__(model)
        model.embed_tokens = _FakeEmbedding(torch.tensor([2.0]))
        model.scale_emb = 12.0
        model.is_mrope_enabled = False
        model.fc_norm = None
        recorder = _RecordEmbedsLayer()
        model.layers = nn.ModuleList([recorder])
        model.norm = _LlamaNorm()
        model.norm_output = False
        forward_batch = types.SimpleNamespace(
            mm_input_embeds=None,
            forward_mode=_ForwardMode(),
            spec_info=types.SimpleNamespace(hidden_states=torch.ones(2, 1)),
        )

        model.forward(torch.tensor([1, 2]), None, forward_batch)

        self.assertTrue(torch.equal(recorder.seen_embeds, torch.full((2, 1), 24.0)))

    def test_draft_input_embeds_also_apply_scale_emb(self):
        model = LlamaModel.__new__(LlamaModel)
        nn.Module.__init__(model)
        model.scale_emb = 12.0
        model.is_mrope_enabled = False
        model.fc_norm = None
        recorder = _RecordEmbedsLayer()
        model.layers = nn.ModuleList([recorder])
        model.norm = _LlamaNorm()
        model.norm_output = False
        forward_batch = types.SimpleNamespace(
            mm_input_embeds=None,
            forward_mode=_ForwardMode(),
            spec_info=types.SimpleNamespace(hidden_states=torch.ones(2, 1)),
        )

        model.forward(
            torch.tensor([1, 2]),
            None,
            forward_batch,
            input_embeds=torch.full((2, 1), 2.0),
        )

        self.assertTrue(torch.equal(recorder.seen_embeds, torch.full((2, 1), 24.0)))

    def test_load_weights_loads_split_nvfp4_scales_through_stacked_mapping(self):
        model = LlamaForCausalLMEagle3.__new__(LlamaForCausalLMEagle3)
        qkv_input_scale = nn.Parameter(torch.zeros(3))
        gate_up_weight_scale_2 = nn.Parameter(torch.zeros(2))

        def qkv_scale_loader(param, loaded_weight, shard_id):
            param.data[{"q": 0, "k": 1, "v": 2}[shard_id]] = loaded_weight

        def gate_up_scale_loader(param, loaded_weight, shard_id):
            param.data[shard_id] = loaded_weight

        qkv_input_scale.weight_loader = qkv_scale_loader
        gate_up_weight_scale_2.weight_loader = gate_up_scale_loader
        model.hot_token_id = None
        model.named_parameters = lambda: iter(
            [
                (
                    "model.layers.0.self_attn.qkv_proj.input_scale",
                    qkv_input_scale,
                ),
                (
                    "model.layers.0.mlp.gate_up_proj.weight_scale_2",
                    gate_up_weight_scale_2,
                ),
            ]
        )

        model.load_weights(
            [
                (
                    "model.midlayer.self_attn.q_proj.input_scale",
                    torch.tensor(1.0),
                ),
                (
                    "model.midlayer.self_attn.k_proj.input_scale",
                    torch.tensor(2.0),
                ),
                (
                    "model.midlayer.self_attn.v_proj.input_scale",
                    torch.tensor(3.0),
                ),
                (
                    "model.midlayer.mlp.gate_proj.weight_scale_2",
                    torch.tensor(4.0),
                ),
                (
                    "model.midlayer.mlp.up_proj.weight_scale_2",
                    torch.tensor(5.0),
                ),
            ]
        )

        self.assertTrue(
            torch.equal(qkv_input_scale.data, torch.tensor([1.0, 2.0, 3.0]))
        )
        self.assertTrue(
            torch.equal(gate_up_weight_scale_2.data, torch.tensor([4.0, 5.0]))
        )


if __name__ == "__main__":
    unittest.main()
