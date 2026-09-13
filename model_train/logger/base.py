#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class MetricSink:
    def log(self, step: int, metrics: dict):
        """Internal helper."""
        raise NotImplementedError

    def close(self):
        """Internal helper."""
        pass
