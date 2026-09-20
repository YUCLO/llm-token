from types import SimpleNamespace

import pytest
import torch

from experiments.common_context_nll import score_utterance


class _IdentityBase(torch.nn.Module):
    def forward(self, input_ids, **_kwargs):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _IdentityBase()
        self.lm_head = torch.nn.Identity()
        self.config = SimpleNamespace(bos_token_id=3)
        # Mirror the released tokenizer/training path: no tokenizer BOS.
        self._qbc_tokenizer_bos_id = None


def test_common_prefix_uses_previous_clean_token_without_bos():
    model = _FakeCausalLM()
    audio_ids = torch.tensor([0, 1, 2])
    clean_codes = torch.tensor([0, 1, 2, 0])
    rows = [
        {
            "full_frame": 2,
            "full_token": 2,
            "slice_token": 1,
            "flip": 1,
        }
    ]
    scored = score_utterance(model, audio_ids, clean_codes, rows, torch.device("cpu"))

    # Target frame 2 is predicted at input position 1, whose clean-prefix token
    # is code 1.  Identity logits therefore prefer slice code 1 over clean code 2.
    assert scored[0]["clean_logit"] == pytest.approx(0.0)
    assert scored[0]["slice_logit"] == pytest.approx(1.0)
    assert scored[0]["delta_nll_common_context"] == pytest.approx(-1.0)
    assert scored[0]["prefix_mode"] == "clean_prefix_without_bos"


def test_no_flip_has_exact_zero_substitution_penalty():
    model = _FakeCausalLM()
    audio_ids = torch.tensor([0, 1, 2])
    clean_codes = torch.tensor([0, 1, 2, 0])
    rows = [
        {
            "full_frame": 2,
            "full_token": 2,
            "slice_token": 2,
            "flip": 0,
        }
    ]
    scored = score_utterance(model, audio_ids, clean_codes, rows, torch.device("cpu"))
    assert scored[0]["delta_nll_common_context"] == 0.0
    assert scored[0]["clean_nll_full_vocab"] == scored[0]["slice_nll_full_vocab"]
