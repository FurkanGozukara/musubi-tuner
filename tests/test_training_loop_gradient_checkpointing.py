from musubi_tuner.training.training_loop import _enable_transformer_gradient_checkpointing


class _QwenStyleTransformer:
    def __init__(self):
        self.activation_cpu_offloading = None

    def enable_gradient_checkpointing(self, activation_cpu_offloading=False):
        self.activation_cpu_offloading = activation_cpu_offloading


class _LtxStyleTransformer:
    def __init__(self):
        self.options = None

    def enable_gradient_checkpointing(
        self, activation_cpu_offloading=False, weight_cpu_offloading=False, blocks_to_checkpoint=None
    ):
        self.options = activation_cpu_offloading, weight_cpu_offloading, blocks_to_checkpoint


def test_gradient_checkpointing_ignores_options_unsupported_by_qwen():
    transformer = _QwenStyleTransformer()

    _enable_transformer_gradient_checkpointing(
        transformer,
        True,
        weight_cpu_offloading=True,
        blocks_to_checkpoint=12,
    )

    assert transformer.activation_cpu_offloading is True


def test_gradient_checkpointing_preserves_ltx_options():
    transformer = _LtxStyleTransformer()

    _enable_transformer_gradient_checkpointing(
        transformer,
        True,
        weight_cpu_offloading=True,
        blocks_to_checkpoint=12,
    )

    assert transformer.options == (True, True, 12)
