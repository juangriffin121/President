import random

import numpy as np
import pytest

from president.player import set_sleep_enabled
from president.ui import writes


@pytest.fixture(autouse=True)
def deterministic_and_quiet() -> None:
    random.seed(12345)
    np.random.seed(12345)
    set_sleep_enabled(False)
    writes.set_silent(True)

