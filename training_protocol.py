"""Training-stage switches."""


def stage_is_enabled(
    *,
    current_epoch: int,
    global_step: int,
    start_epoch: int,
    start_step: int,
) -> bool:
    """Use the step threshold when set; otherwise use the epoch threshold."""

    if start_step >= 0:
        return global_step >= start_step
    if start_epoch >= 0:
        return current_epoch >= start_epoch
    return False
