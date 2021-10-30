# Reference : https://huggingface.co/transformers/_modules/transformers/optimization.html#get_constant_schedule_with_warmup

import math
from enum import Enum
from typing import Optional

from numpy import e
from pydantic import BaseModel, validator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR


class Scheduler(Enum):
    cosine_warmup = 'cosine_warmup'
    constant_warmup = 'constant_warmup'
    step_warmup = 'step_warmup'


class ScheduleValidator(BaseModel):
    scheduler: Scheduler
    num_warmup_steps: Optional[int] = None
    num_training_steps: Optional[int] = None
    num_cycles: Optional[float] = None
    step_size: Optional[int] = None
    step_factor: Optional[float] = None

    @validator("num_warmup_steps", always=True)
    def validate_num_warmup_steps(cls, value, values):

        if values['scheduler'] in [
            Scheduler.cosine_warmup.value,
            Scheduler.constant_warmup.value,
            Scheduler.step_warmup.value
        ]:
            assert value is not None,\
                '`num_warmup_steps` cannot be None '\
                f'for scheduler {values["scheduler"]}.'
        return value

    @validator("num_training_steps", always=True)
    def validate_num_training_steps(cls, value, values):

        if values['scheduler'] in [
            Scheduler.cosine_warmup.value,
        ]:
            assert value is not None,\
                '`num_training_steps` cannot be None '\
                f'for scheduler {values["scheduler"]}.'
            return value

        else:
            return None

    @validator("num_cycles", always=True)
    def validate_num_cycles(cls, value, values):

        if values['scheduler'] in [
            Scheduler.cosine_warmup.value,
        ]:
            assert value is not None,\
                '`num_cycles` cannot be None '\
                f'for scheduler {values["scheduler"]}.'

            return value

        else:
            return None

    @validator("step_size", always=True)
    def validate_step_size(cls, value, values):

        if values['scheduler'] in [
            Scheduler.step_warmup.value,
        ]:
            assert value is not None,\
                '`step_size` cannot be None '\
                f'for scheduler {values["scheduler"]}.'

            return value

        else:
            return None

    @validator("step_factor", always=True)
    def validate_step_factor(cls, value, values):

        if values['scheduler'] in [
            Scheduler.step_warmup.value,
        ]:
            assert value is not None,\
                '`step_factor` cannot be None '\
                f'for scheduler {values["scheduler"]}.'

            return value

        else:
            return None

    class Config:
        use_enum_values = True


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_step_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        step_size: int,
        step_factor: float,
        last_epoch: int = -1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        else:
            step_pow = (current_step - num_warmup_steps) // step_size
            step_mult = step_factor ** step_pow
            return step_mult

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
