from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from accelerate import Accelerator

import musubi_tuner.training.full_finetune as full_finetune
from test_full_finetune_runtime import TinyDataset, TinyFullTrainer, install_runtime_fakes, make_args


class ResumeDataset(TinyDataset):
    num_train_items = 5

    def __len__(self):
        return 5


class ResumeTrainer(TinyFullTrainer):
    interrupt_after_state_save = False

    def _build_dataset(self, _args):
        dataset = ResumeDataset()
        return dataset, lambda items: items[0], SimpleNamespace(value=0)

    def get_lr_scheduler(self, args, optimizer, num_processes):
        scheduler = super().get_lr_scheduler(args, optimizer, num_processes)
        self.scheduler = scheduler
        return scheduler

    def _save_step_state(self, accelerator, args, global_step):
        super()._save_step_state(accelerator, args, global_step)
        if self.interrupt_after_state_save:
            raise TrainingInterrupted


class TrainingInterrupted(Exception):
    pass


def _run_training(
    monkeypatch,
    output_dir,
    *,
    max_train_steps,
    resume=None,
    save_at_step=None,
    interrupt_after_state_save=False,
):
    trainer = ResumeTrainer()
    trainer.interrupt_after_state_save = interrupt_after_state_save
    trainer.skipped_batches = []
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1, mixed_precision="no")
    skip_first_batches = accelerator.skip_first_batches

    def record_skip(dataloader, num_batches):
        trainer.skipped_batches.append(num_batches)
        return skip_first_batches(dataloader, num_batches)

    accelerator.skip_first_batches = record_skip
    monkeypatch.setattr(full_finetune, "prepare_accelerator", lambda _args: accelerator)
    monkeypatch.setattr(full_finetune, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(
        full_finetune.sai_model_spec,
        "build_metadata",
        lambda *args, **kwargs: {"modelspec.architecture": "tiny", "is_lora": str(kwargs["is_lora"])},
    )

    trainer.train(
        make_args(
            output_dir,
            blocks_to_swap=0,
            compile=False,
            sample_prompts=None,
            max_train_steps=max_train_steps,
            save_every_n_steps=save_at_step,
            save_state=save_at_step is not None,
            resume=None if resume is None else str(resume),
        )
    )
    return trainer


def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    uninterrupted = _run_training(monkeypatch, tmp_path / "uninterrupted", max_train_steps=4)

    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(TrainingInterrupted):
        _run_training(
            monkeypatch,
            interrupted_dir,
            max_train_steps=4,
            save_at_step=2,
            interrupt_after_state_save=True,
        )
    state_dir = interrupted_dir / "tiny-step00000002-state"
    resumed = _run_training(monkeypatch, tmp_path / "resumed", max_train_steps=4, resume=state_dir)

    uninterrupted_state = uninterrupted.raw_model.state_dict()
    resumed_state = resumed.raw_model.state_dict()
    assert uninterrupted_state.keys() == resumed_state.keys()
    assert all(torch.equal(uninterrupted_state[key], resumed_state[key]) for key in uninterrupted_state)
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert resumed.training_progress == uninterrupted.training_progress
    assert resumed.skipped_batches == [2]
    assert resumed.training_progress.global_step == 4
    assert resumed.training_progress.epoch == 1
    assert resumed.training_progress.next_batch == 0


def test_resume_requires_registered_training_progress_file(tmp_path, monkeypatch):
    state_dir = tmp_path / "legacy-state"
    state_dir.mkdir()
    trainer = TinyFullTrainer()
    install_runtime_fakes(monkeypatch, trainer)

    with pytest.raises(RuntimeError, match="TrainingProgress"):
        trainer.train(make_args(tmp_path, resume=str(state_dir)))
