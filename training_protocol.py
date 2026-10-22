"""Small, testable guards for PI-EDG training-stage transitions."""


def stage_is_enabled(
    *,
    current_epoch: int,
    global_step: int,
    start_epoch: int,
    start_step: int,
) -> bool:
    """Return whether an optional stage has started.

    A nonnegative step takes precedence over the epoch schedule. If both
    schedules are negative, the stage is disabled.
    """

    if start_step >= 0:
        return global_step >= start_step
    if start_epoch >= 0:
        return current_epoch >= start_epoch
    return False
