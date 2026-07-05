from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


class Metrics:

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def record(self, name: str, value: float) -> None:
        self._timings[name].append(value)

    def count(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def get_gauge(self, name: str) -> Optional[float]:
        return self._gauges.get(name)

    def start_timer(self, name: str) -> None:
        self._start_times[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        if name not in self._start_times:
            return 0.0
        elapsed = (time.perf_counter() - self._start_times.pop(name)) * 1000
        self._timings[name].append(elapsed)
        return elapsed

    def time_function(self, name: str):
        class TimerContext:
            def __init__(self, metrics: Metrics, timer_name: str):
                self._metrics = metrics
                self._name = timer_name
                self._elapsed = 0.0

            def __enter__(self):
                self._start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._elapsed = (time.perf_counter() - self._start) * 1000
                self._metrics._timings[self._name].append(self._elapsed)
                return False

            @property
            def elapsed(self) -> float:
                return self._elapsed

        return TimerContext(self, name)

    def get_avg(self, name: str) -> float:
        values = self._timings.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_median(self, name: str) -> float:
        values = self._timings.get(name, [])
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    def get_min(self, name: str) -> float:
        values = self._timings.get(name, [])
        return min(values) if values else 0.0

    def get_max(self, name: str) -> float:
        values = self._timings.get(name, [])
        return max(values) if values else 0.0

    def get_total(self, name: str) -> float:
        values = self._timings.get(name, [])
        return sum(values)

    def get_count(self, name: str) -> int:
        return self._counters.get(name, 0)

    def get_timing_count(self, name: str) -> int:
        return len(self._timings.get(name, []))

    def get_percentile(self, name: str, percentile: float) -> float:
        values = self._timings.get(name, [])
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * (percentile / 100.0))
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]

    def get_all_timing_names(self) -> List[str]:
        return sorted(self._timings.keys())

    def get_all_counter_names(self) -> List[str]:
        return sorted(self._counters.keys())

    def report(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "timings": {},
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }

        for name in sorted(self._timings.keys()):
            values = self._timings[name]
            if values:
                result["timings"][name] = {
                    "count": len(values),
                    "avg_ms": round(self.get_avg(name), 3),
                    "median_ms": round(self.get_median(name), 3),
                    "min_ms": round(self.get_min(name), 3),
                    "max_ms": round(self.get_max(name), 3),
                    "total_ms": round(self.get_total(name), 3),
                    "p95_ms": round(self.get_percentile(name, 95), 3),
                    "p99_ms": round(self.get_percentile(name, 99), 3),
                }

        return result

    def summary(self) -> str:
        lines: List[str] = []

        if self._timings:
            lines.append("Timings:")
            for name in sorted(self._timings.keys()):
                values = self._timings[name]
                if values:
                    lines.append(
                        f"  {name}: avg={self.get_avg(name):.2f}ms, "
                        f"count={len(values)}, "
                        f"total={self.get_total(name):.2f}ms"
                    )

        if self._counters:
            lines.append("Counters:")
            for name in sorted(self._counters.keys()):
                lines.append(f"  {name}: {self._counters[name]}")

        if self._gauges:
            lines.append("Gauges:")
            for name in sorted(self._gauges.keys()):
                lines.append(f"  {name}: {self._gauges[name]}")

        return "\n".join(lines) if lines else "No metrics recorded."

    def reset(self) -> None:
        self._timings.clear()
        self._counters.clear()
        self._gauges.clear()
        self._start_times.clear()

    def merge(self, other: Metrics) -> None:
        for name, values in other._timings.items():
            self._timings[name].extend(values)
        for name, count in other._counters.items():
            self._counters[name] += count
        for name, value in other._gauges.items():
            self._gauges[name] = value