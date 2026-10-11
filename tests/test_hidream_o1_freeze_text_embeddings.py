import pytest
import torch

from musubi_tuner.hidream_o1_train import HiDreamO1Trainer


class _TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])


class _TinyHiDream(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = _TinyLanguageModel()


def test_freeze_text_embeddings_only_freezes_the_token_table():
    transformer = _TinyHiDream()
    assert HiDreamO1Trainer._freeze_text_embeddings(transformer) == 32 * 8
    assert not transformer.model.language_model.embed_tokens.weight.requires_grad
    assert all(parameter.requires_grad for parameter in transformer.model.language_model.layers.parameters())


def test_freeze_text_embeddings_requires_the_table():
    with pytest.raises(ValueError, match="embed_tokens"):
        HiDreamO1Trainer._freeze_text_embeddings(torch.nn.Linear(4, 4))
