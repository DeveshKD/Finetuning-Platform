"""Job lifecycle state machine.

Linear on purpose, with CANCELLED reachable from every non-terminal,
non-persisting state — a stop request can land at any point while a job is
actually running, but PERSISTING (writing final artifacts) is short and
deliberately not interruptible.
"""

from enum import Enum


class JobState(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"        # model load + quantization + PEFT wrap
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}

VALID_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {JobState.PREPARING, JobState.CANCELLED},
    JobState.PREPARING: {JobState.TRAINING, JobState.FAILED, JobState.CANCELLED},
    JobState.TRAINING: {JobState.CHECKPOINTING, JobState.EVALUATING, JobState.FAILED, JobState.CANCELLED},
    JobState.CHECKPOINTING: {JobState.TRAINING, JobState.EVALUATING, JobState.FAILED, JobState.CANCELLED},
    JobState.EVALUATING: {JobState.PERSISTING, JobState.FAILED, JobState.CANCELLED},
    JobState.PERSISTING: {JobState.COMPLETED, JobState.FAILED},
}


def assert_valid_transition(current: JobState, target: JobState) -> None:
    if target not in VALID_TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal job transition: {current} -> {target}")
