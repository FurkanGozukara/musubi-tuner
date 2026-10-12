"""A blank negative prompt written by the GUI ("--n " at the end of a line) survives load_prompts' line stripping."""

from musubi_tuner.training.sampling_prompts import line_to_prompt_dict, load_prompts


def test_trailing_blank_negative_keeps_cfg_on(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("a photo of a cat --w 512 --l 4.0 --n  \nno negative here --w 512\n", encoding="utf-8")
    first, second = load_prompts(str(path))
    assert first["negative_prompt"] == " " and first["cfg_scale"] == 4.0
    assert "negative_prompt" not in second


def test_explicit_negative_prompts_are_unchanged():
    assert line_to_prompt_dict("a cat --n blurry, low quality --w 512")["negative_prompt"] == "blurry, low quality"
    assert line_to_prompt_dict("a cat --n   --w 512")["negative_prompt"] == " "
