import os
import time
from typing import Callable

import torch


def _test_memory_once(func: Callable) -> float:

    torch.cuda.memory._record_memory_history()
    torch.cuda.memory.reset_peak_memory_stats()

    func()

    mem = torch.cuda.max_memory_allocated()

    # uncomment to save the visual memory snapshot
    # torch.cuda.memory._dump_snapshot(f"{func.__name__}.pickle")

    torch.cuda.memory._record_memory_history(enabled=None)
    return mem


def _test_memory(func: Callable, _iter: int = 10) -> float:
    total_mem = []

    for _ in range(_iter):
        mem = _test_memory_once(func)
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)


def get_current_file_directory() -> str:
    """
    Returns the directory path of the current Python file.
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    return os.path.dirname(current_file_path)


def sleep(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            time.sleep(seconds)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def _print_memory_banner():
    print("**************************************")
    print("*     BENCHMARKING GPU MEMORY        *")
    print("**************************************")


def _print_speed_banner():
    print("**************************************")
    print("*        BENCHMARKING SPEED          *")
    print("**************************************")