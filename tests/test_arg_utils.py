"""--optimizer_args / --lr_scheduler_args / --network_args items as every trainer reads them (utils/arg_utils.py)."""

import pytest

from musubi_tuner.utils.arg_utils import literal_or_text, parse_literal_args, parse_text_args, split_key_value


def test_literal_args_strip_spaces_and_read_python_values():
    kwargs = parse_literal_args([" weight_decay = 0.01 ", "betas=(0.9, 0.999)", "relative_step=false", "mode=triangular2", "eps=None", ""])
    assert kwargs == {"weight_decay": 0.01, "betas": (0.9, 0.999), "relative_step": False, "mode": "triangular2", "eps": None}


def test_text_args_keep_values_with_equals():
    kwargs = parse_text_args([r"network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3", " conv_dim = 4", "verbose=True"])
    assert kwargs == {"network_reg_lrs": r"single_blocks_(\d|10)_=1e-3,double_blocks=2e-3", "conv_dim": "4", "verbose": "True"}


@pytest.mark.parametrize("item", ["no_equals", "=3", "  =x"])
def test_items_without_a_key_fail_clearly(item):
    with pytest.raises(ValueError, match="expected key=value"):
        split_key_value(item, "--network_args")


def test_literal_or_text():
    assert literal_or_text("1e-4") == 1e-4 and literal_or_text("'x'") == "x" and literal_or_text("TRUE") is True
    assert literal_or_text("cosine") == "cosine"
