"""FLUX.2 block swap: blocks_to_swap -> (double, single) blocks.

The proportional split stops early on FLUX Klein 9B (16 = 6 double + 18 single blocks, although the
offloaders can swap 6 + 22), so larger values extend it; values it already supported map unchanged.
"""

import pytest

from musubi_tuner.flux_2.flux2_models import Flux2

MODELS = {"flux2_dev": (8, 48), "klein_9b": (8, 24), "klein_4b": (5, 20)}


def _model(num_double: int, num_single: int) -> Flux2:
    model = object.__new__(Flux2)  # only the block counts are needed
    model.num_double_blocks = num_double
    model.num_single_blocks = num_single
    return model


def _original_split(num_double: int, num_single: int, num_blocks: int):
    """The split before the extension; None where it failed its assertion."""
    ratio = num_single / num_double
    double = int(round(num_blocks / (1.0 + ratio / 2.0)))
    single = int(round(double * ratio))
    if num_double * 2 < num_single:
        while double >= 1 and double > num_double - 2:
            double -= 1
            single += 2
    else:
        while single >= 2 and single > num_single - 2:
            single -= 2
            double += 1
    if double == 0 and single == 0:
        if num_single >= num_double:
            single = 1
        else:
            double = 1
    if double > num_double - 2 or single > num_single - 2:
        return None
    return double, single


@pytest.mark.parametrize("name", sorted(MODELS))
def test_values_the_proportional_split_supported_map_unchanged(name):
    num_double, num_single = MODELS[name]
    model = _model(num_double, num_single)
    supported = 0
    for num_blocks in range(1, num_double + num_single):
        expected = _original_split(num_double, num_single, num_blocks)
        if expected is not None:
            supported += 1
            assert model.split_blocks_to_swap(num_blocks) == expected, num_blocks
    assert supported > 0
    assert model.split_blocks_to_swap(0) == (0, 0)


@pytest.mark.parametrize(
    ("name", "old_max", "extended"),
    [
        ("klein_9b", 16, {17: (6, 20), 18: (6, 22)}),
        ("flux2_dev", 29, {30: (6, 46)}),
        ("klein_4b", 13, {}),
    ],
)
def test_larger_values_swap_the_remaining_single_blocks(name, old_max, extended):
    num_double, num_single = MODELS[name]
    model = _model(num_double, num_single)
    assert _original_split(num_double, num_single, old_max) is not None
    assert _original_split(num_double, num_single, old_max + 1) is None
    for num_blocks, split in extended.items():
        assert model.split_blocks_to_swap(num_blocks) == split
    new_max = old_max + len(extended)
    assert model.split_blocks_to_swap(new_max) == (num_double - 2, num_single - 2)
    with pytest.raises(ValueError, match=f"at most {new_max}"):
        model.split_blocks_to_swap(new_max + 1)
