"""Training-time samples show the trained weights: the LoRA network is in eval mode while sampling (no network /
rank / module dropout), the trainers' sampling hooks run around it (Krea 2 Turbo sampling, Ideogram 4 unconditional
DiT), and a step is sampled at most once (the step trigger can fire on the last step of an epoch)."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

from musubi_tuner.training.sampling_prompts import should_sample_images

TRAINING = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner" / "training"


def _nested_function(path: Path, outer: str, inner: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    outer_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == outer)
    inner_node = next(node for node in ast.walk(outer_node) if isinstance(node, ast.FunctionDef) and node.name == inner)
    return textwrap.dedent(ast.get_source_segment(source, inner_node, padded=True))


class Recorder:
    def __init__(self):
        self.calls = []
        self.training = True

    # the LoRA network
    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class Trainer:
    def __init__(self, network):
        self.network = network
        self.calls = []

    def on_before_sample_images(self, *args):
        self.calls.append(("before", self.network.training))

    def sample_images(self, accelerator, args, epoch, steps, *rest):
        self.calls.append(("sample", self.network.training, epoch, steps))

    def on_after_sample_images(self, *args):
        self.calls.append(("after", self.network.training))


def _fork_wrapper(args, network, trainer):
    body = textwrap.indent(_nested_function(TRAINING / "training_loop.py", "train", "sample_images_with_optional_ema"), "    ")
    factory = (
        "def factory(self, args, accelerator, vae, transformer, network, sample_parameters, dit_dtype,\n"
        "            apply_ema_weights_for_eval, apply_self_flow_weights_for_eval,\n"
        "            restore_self_flow_weights_for_eval, restore_ema_weights_for_eval):\n"
        "    last_sampled_step = None\n"
        f"{body}\n"
        "    return sample_images_with_optional_ema\n"
    )
    namespace = {"should_sample_images": should_sample_images}
    exec(factory, namespace)
    return namespace["factory"](
        trainer, args, None, None, None, network, [], None, lambda: None, lambda: None, lambda _: None, lambda _: None
    )


def test_fork_loop_samples_in_eval_mode_with_the_trainer_hooks_once_per_step():
    network = Recorder()
    trainer = Trainer(network)
    args = SimpleNamespace(sample_at_first=False, sample_every_n_steps=10, sample_every_n_epochs=1)
    sample = _fork_wrapper(args, network, trainer)

    sample(None, 5)  # not due: no hooks, no mode switch
    assert trainer.calls == []
    sample(None, 10)  # the step trigger on the last step of epoch 1 ...
    sample(1, 10)  # ... then the epoch-end call for the same step
    sample(2, 15)  # an epoch end that only the epoch trigger wants

    assert trainer.calls == [
        ("before", False),
        ("sample", False, None, 10),
        ("after", False),
        ("before", False),
        ("sample", False, 2, 15),
        ("after", False),
    ]
    assert network.training  # back in training mode


def test_fork_loop_restores_training_mode_when_sampling_fails():
    network = Recorder()

    class Failing(Trainer):
        def sample_images(self, *args):
            raise RuntimeError("sampling failed")

    trainer = Failing(network)
    sample = _fork_wrapper(SimpleNamespace(sample_at_first=True, sample_every_n_steps=None, sample_every_n_epochs=None), network, trainer)
    try:
        sample(0, 0)
    except RuntimeError:
        pass
    assert network.training and trainer.calls == [("before", False), ("after", False)]


def test_upstream_loop_samples_in_eval_mode_around_the_hooks():
    wrapper = _nested_function(TRAINING / "trainer_base.py", "_run_training_loop", "_do_sample")
    tokens = ("network.eval()", "self.on_before_sample_images(", "self.sample_images(", "self.on_after_sample_images(", "network.train()")
    positions = [wrapper.index(token) for token in tokens]
    assert positions == sorted(positions)
