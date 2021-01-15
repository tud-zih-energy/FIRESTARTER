#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This requires python>=3.7

This toy metric is an example for the --metric-from-stdin parameter from FIRESTARTER.
Execute with: ./test_metric.py | FIRESTARTER --metric-from-stdin=test-metric --measurement -t 10
"""

import time

try:
    while True:
        print(f"test-metric {time.time_ns()} {1.0}", flush=True)
        time.sleep(0.1)
except BrokenPipeError:
    pass
