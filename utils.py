from abc import abstractmethod
import numpy
import operator
from subprocess import call
from shutil import copyfile
import logging
import os
import sys


GO_ID = 1
EOS_ID = 2
UNK_ID = 0
NOTAPPLICABLE_ID = 3
NEG_INF = float("-inf")
INF = float("inf")
EPS_P = 0.00001


def argmax_n(arr, n):
    if instance(arr, dict):
        return sorted(arr, key=arr.get, reverse=True)
    elif len(arr) <= n:
        return range(n)
    else:
        return numpy.argpartition(arr, -n)[-n:]


def common_get(obj, key, default):
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default


def log_sum_log_semiring(vals):
    """
    Uses the ``logsumexp`` function in scipy to calculate the log of
    the sum of a set of log values.
    """
    return logsumexp(numpy.asarray([val for val in vals]))

log_sum = log_sum_log_semiring


MESSAGE_TYPE_DEFAULT = 1
MESSAGE_TYPE_POSTERIOR = 2
MESSAGE_TYPE_FULL_HYPO = 3


class Observer(object):

    @abstractmethod
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        raise NotImplementedError


class Observable(object):

    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        for observer in self.observers:
            observer.notify(message, message_type)
