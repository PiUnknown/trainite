from unittest import mock

import pytest
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint

from trainite.shared.utils import (
    _ActiveTimeCheckpoint,
    setup_best_model_checkpoint,
    setup_training_checkpointing,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecordingSaver:
    def __init__(self, clock, save_duration=0.0, fail=False):
        self.clock = clock
        self.save_duration = save_duration
        self.fail = fail
        self.engine = None
        self.iterations = []
        self.checkpoints = []
        self.priorities = []

    def __call__(self, checkpoint, filename, metadata=None):
        if self.fail:
            raise OSError("disk full")
        self.iterations.append(self.engine.state.iteration)
        self.checkpoints.append((filename, checkpoint))
        self.priorities.append(metadata.get("priority") if metadata else None)
        self.clock.advance(self.save_duration)

    def remove(self, filename):
        pass


def make_engine(clock):
    def train_step(engine, duration):
        clock.advance(duration)
        return duration

    return Engine(train_step)


def make_saver(clock, engine, save_duration=0.0):
    saver = RecordingSaver(clock, save_duration=save_duration)
    saver.engine = engine
    return saver


def setup(engine, saver, clock, interval_minutes=1.0):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return setup_training_checkpointing(
        engine,
        {"model": model, "optimizer": optimizer},
        saver,  # type: ignore[arg-type]
        interval_minutes=interval_minutes,
        monotonic_clock=clock,
    )


def test_timed_checkpoint_waits_for_threshold_and_deduplicates_epoch_save():
    clock = FakeClock()
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def train_step(engine, duration):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        clock.advance(duration)

    engine = Engine(train_step)
    saver = make_saver(clock, engine)
    checkpoint = setup_training_checkpointing(
        engine,
        {"model": model, "optimizer": optimizer},
        saver,  # type: ignore[arg-type]
        interval_minutes=1.0,
        monotonic_clock=clock,
    )
    saves_after_step = []
    engine.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda _: saves_after_step.append(len(saver.checkpoints)),
    )

    engine.run([30.0, 29.0, 1.0], max_epochs=1)

    assert saves_after_step == [0, 0, 1]
    assert saver.iterations == [3]
    filename, payload = saver.checkpoints[0]
    assert filename == "last.pt"
    assert isinstance(checkpoint, Checkpoint)
    assert set(payload) == {"model", "optimizer"}
    assert payload["model"]["weight"].item() == pytest.approx(model.weight.item())


def test_variable_steps_overrun_without_catch_up_burst():
    clock = FakeClock()
    engine = make_engine(clock)
    saver = make_saver(clock, engine)
    setup(engine, saver, clock)
    saves_after_step = []
    saved_at_run_start = [0]
    engine.add_event_handler(
        Events.STARTED,
        lambda _: saved_at_run_start.__setitem__(0, len(saver.checkpoints)),
    )
    engine.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda _: saves_after_step.append(len(saver.checkpoints) - saved_at_run_start[0]),
    )

    engine.run([20.0, 55.0, 100.0, 1.0], max_epochs=1)

    # First save is after 75 active seconds; the 100-second overrun causes one
    # save, and the remaining 1-second step does not trigger a catch-up save.
    # The final iteration then receives its separate epoch save.
    assert saves_after_step == [0, 1, 2, 2]
    assert saver.iterations == [2, 3, 4]


def test_evaluation_and_epoch_save_duration_are_excluded():
    clock = FakeClock()
    engine = make_engine(clock)
    # Trainer registers its evaluation callback before checkpoint setup.
    engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: clock.advance(600.0))
    saver = make_saver(clock, engine, save_duration=900.0)
    setup(engine, saver, clock, interval_minutes=0.75)

    engine.run([20.0], max_epochs=2)

    # Forty active seconds total is below the 45-second interval, even though
    # evaluation and epoch-saving time make wall time much longer.
    assert saver.iterations == [1, 2]


def test_timer_resets_between_runs():
    clock = FakeClock()
    engine = make_engine(clock)
    saver = make_saver(clock, engine)
    setup(engine, saver, clock)
    saves_after_step = []
    saved_at_run_start = [0]
    engine.add_event_handler(
        Events.STARTED,
        lambda _: saved_at_run_start.__setitem__(0, len(saver.checkpoints)),
    )
    engine.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda _: saves_after_step.append(len(saver.checkpoints) - saved_at_run_start[0]),
    )

    engine.run([30.0], max_epochs=1)
    engine.run([30.0], max_epochs=1)

    # A fresh 30-second run must not combine with the previous run to trigger a
    # timed save; each run has only its existing epoch save.
    assert saver.iterations == [1, 1]
    assert saves_after_step == [0, 0]


def test_failed_timed_save_propagates_and_remains_due():
    clock = FakeClock()
    engine = Engine(lambda _engine, _batch: None)
    failing_checkpoint = mock.Mock(side_effect=OSError("disk full"))
    timer = _ActiveTimeCheckpoint(engine, failing_checkpoint, 60.0, clock)
    timer.start_step(engine)
    clock.advance(60.0)

    with pytest.raises(OSError, match="disk full"):
        timer.finish_step(engine)

    assert timer.elapsed == 60.0
    assert timer.last_saved_iteration is None
    failing_checkpoint.assert_called_once_with(engine)


def test_failed_saver_exception_propagates_through_engine():
    clock = FakeClock()
    engine = make_engine(clock)
    saver = make_saver(clock, engine)
    saver.fail = True
    checkpoint = setup(engine, saver, clock)

    with pytest.raises(OSError, match="disk full"):
        engine.run([60.0], max_epochs=1)

    assert isinstance(checkpoint, Checkpoint)
    assert checkpoint.last_checkpoint is None
    assert saver.checkpoints == []


def test_disabled_mode_keeps_epoch_only_checkpointing():
    clock = FakeClock()
    engine = make_engine(clock)
    saver = make_saver(clock, engine)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = setup_training_checkpointing(
        engine,
        {"model": model, "optimizer": optimizer},
        saver,  # type: ignore[arg-type]
    )

    engine.run([120.0], max_epochs=2)

    assert isinstance(checkpoint, Checkpoint)
    assert saver.iterations == [1, 2]


def test_best_checkpoint_selection_is_independent_of_timed_latest_saves():
    clock = FakeClock()
    trainer = Engine(lambda _engine, _batch: None)
    evaluator = Engine(lambda _engine, _batch: None)
    latest_saver = make_saver(clock, trainer)
    setup(trainer, latest_saver, clock)
    best_saver = make_saver(clock, trainer)
    model = torch.nn.Linear(2, 2)
    best_checkpoint = setup_best_model_checkpoint(
        trainer,
        evaluator,
        {"model": model},
        best_saver,  # type: ignore[arg-type]
        score_function=lambda engine: -engine.state.metrics["loss"],
        score_name="val_loss",
    )

    for loss in (1.0, 0.5, 0.75):
        evaluator.state.metrics = {"loss": loss}
        evaluator.fire_event(Events.COMPLETED)

    assert isinstance(best_checkpoint, Checkpoint)
    assert [name for name, _ in best_saver.checkpoints] == ["best.pt", "best.pt"]
    assert best_saver.priorities == [-1.0, -0.5]
    assert latest_saver.checkpoints == []
