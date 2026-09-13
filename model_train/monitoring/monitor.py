#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

_MIB = 1024 * 1024
_GIB = 1024.0



_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASE = _REPO_ROOT / "logs" / "monitoring_log"



try:
    import pynvml as _nvml
    _nvml.nvmlInit()
    _N_GPU = _nvml.nvmlDeviceGetCount()
    _HAS_GPU = _N_GPU > 0
except Exception:
    _HAS_GPU = False
    _N_GPU = 0


def _probe(fn) -> bool:
    if not _HAS_GPU:
        return False
    try:
        fn(_nvml.nvmlDeviceGetHandleByIndex(0))
        return True
    except Exception:
        return False


if _HAS_GPU:
    _POWER_OK = _probe(lambda h: _nvml.nvmlDeviceGetPowerUsage(h))
    _CLOCK_OK = _probe(lambda h: _nvml.nvmlDeviceGetClockInfo(h, _nvml.NVML_CLOCK_SM))
    _PCIE_OK = _probe(lambda h: _nvml.nvmlDeviceGetPcieThroughput(
        h, _nvml.NVML_PCIE_UTIL_RX_BYTES))
    _TEMP_OK = _probe(lambda h: _nvml.nvmlDeviceGetTemperature(
        h, _nvml.NVML_TEMPERATURE_GPU))
else:
    _POWER_OK = _CLOCK_OK = _PCIE_OK = _TEMP_OK = False


def _sample_gpu() -> list:
    out = []
    for i in range(_N_GPU):
        h = _nvml.nvmlDeviceGetHandleByIndex(i)
        util = _nvml.nvmlDeviceGetUtilizationRates(h)
        mem = _nvml.nvmlDeviceGetMemoryInfo(h)
        rec = {
            "id": i,
            "util": util.gpu,
            "mem_used": mem.used >> 20,
            "mem_total": mem.total >> 20,
        }
        if _TEMP_OK:
            try:
                rec["temp"] = _nvml.nvmlDeviceGetTemperature(h, _nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
        if _POWER_OK:
            try:
                rec["power_w"] = round(_nvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
        if _CLOCK_OK:
            try:
                rec["sm_mhz"] = _nvml.nvmlDeviceGetClockInfo(h, _nvml.NVML_CLOCK_SM)
            except Exception:
                pass
        if _PCIE_OK:
            try:
                rx = _nvml.nvmlDeviceGetPcieThroughput(h, _nvml.NVML_PCIE_UTIL_RX_BYTES)
                tx = _nvml.nvmlDeviceGetPcieThroughput(h, _nvml.NVML_PCIE_UTIL_TX_BYTES)
                rec["pcie_rx_mib_s"] = round(rx / 1024.0, 1)   # KB/s -> MiB/s
                rec["pcie_tx_mib_s"] = round(tx / 1024.0, 1)
            except Exception:
                pass
        out.append(rec)
    return out


def _sample_mem() -> dict:
    """Internal helper."""
    try:
        vm = psutil.virtual_memory()
        return {"used_mib": vm.used >> 20, "total_mib": vm.total >> 20,
                "percent": round(vm.percent, 1)}
    except Exception:
        return {"used_mib": 0, "total_mib": 0, "percent": 0.0}


def _disk_io_counters():
    try:
        return psutil.disk_io_counters(nowrap=True)
    except TypeError:
        return psutil.disk_io_counters()
    except Exception:
        return None


def _net_io_counters():
    try:
        return psutil.net_io_counters(nowrap=True)
    except TypeError:
        return psutil.net_io_counters()
    except Exception:
        return None


def _counter_delta(now, prev, name: str) -> int:
    return max(0, int(getattr(now, name, 0) or 0) - int(getattr(prev, name, 0) or 0))


def _sample_disk_io(now, prev, dt: float) -> dict:
    if now is None or prev is None or dt <= 0:
        return {"read_mib_s": 0.0, "write_mib_s": 0.0,
                "read_iops": 0.0, "write_iops": 0.0, "busy_pct": 0.0}
    return {
        "read_mib_s": round(_counter_delta(now, prev, "read_bytes") / dt / _MIB, 2),
        "write_mib_s": round(_counter_delta(now, prev, "write_bytes") / dt / _MIB, 2),
        "read_iops": round(_counter_delta(now, prev, "read_count") / dt, 1),
        "write_iops": round(_counter_delta(now, prev, "write_count") / dt, 1),

        "busy_pct": round(_counter_delta(now, prev, "busy_time") / (dt * 10.0), 1),
    }


def _sample_net_io(now, prev, dt: float) -> dict:
    if now is None or prev is None or dt <= 0:
        return {"recv_mib_s": 0.0, "sent_mib_s": 0.0}
    return {
        "recv_mib_s": round(_counter_delta(now, prev, "bytes_recv") / dt / _MIB, 2),
        "sent_mib_s": round(_counter_delta(now, prev, "bytes_sent") / dt / _MIB, 2),
    }


def _read_jsonl(jsonl_path: Path) -> list:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render(rows: list, out_path: Path, quiet: bool = False) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        if not quiet:
            print("[train]")
        return
    if not rows:
        return

    ts = [r["t"] for r in rows]
    n_gpu = len(rows[0].get("gpus", []))

    def fmt_t(t):
        return f"{t/60:.1f}m" if t > 120 else f"{t:.0f}s"

    x_labels = [fmt_t(t) for t in ts]
    step = max(1, len(ts) // 10)
    xtick_pos = list(range(0, len(ts), step))

    GPU_COLORS = ["#4C8EFF", "#FF6B6B", "#51CF66", "#FFD43B",
                  "#F783AC", "#74C0FC", "#63E6BE", "#FFA94D"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if n_gpu > 0:

        ax = axes[0]
        for i in range(n_gpu):
            c = GPU_COLORS[i % len(GPU_COLORS)]
            ax.plot(ts, [r["gpus"][i]["util"] for r in rows], color=c,
                    linewidth=1.2, label=f"GPU{i}")
        ax.set_ylabel("Util (%)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, (n_gpu + 3) // 4))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("GPU Utilization & Memory", fontsize=11, fontweight="bold")


        ax = axes[1]
        gt = rows[0]["gpus"][0].get("mem_total", 0)
        for i in range(n_gpu):
            c = GPU_COLORS[i % len(GPU_COLORS)]
            ax.plot(ts, [r["gpus"][i]["mem_used"] / _GIB for r in rows], color=c,
                    linewidth=1.2, label=f"GPU{i}")
        if gt > 0:
            ax.axhline(gt / _GIB, color="#868E96", linewidth=0.7, linestyle=":",
                       label=f"Total {gt / _GIB:.0f} GiB")
        ax.set_ylabel("Mem (GiB)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, (n_gpu + 3) // 4))
        ax.grid(True, linestyle="--", alpha=0.4)
    else:

        ax = axes[0]
        cpu = [r.get("cpu", 0.0) for r in rows]
        ax.plot(ts, cpu, color="#339AF0", linewidth=1.2, label="CPU avg")
        ax.fill_between(ts, cpu, alpha=0.15, color="#339AF0")
        ax.set_ylabel("CPU (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("CPU & RAM (no GPU)", fontsize=11, fontweight="bold")

        ax = axes[1]
        mem_used = [r.get("mem", {}).get("used_mib", 0) / _GIB for r in rows]
        ax.plot(ts, mem_used, color="#7950F2", linewidth=1.2, label="RAM Used GiB")
        ax.fill_between(ts, mem_used, alpha=0.15, color="#7950F2")
        ax.set_ylabel("RAM (GiB)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    xtick_vals = [ts[i] for i in xtick_pos]
    xtick_labels = [x_labels[i] for i in xtick_pos]
    for ax in axes:
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels, fontsize=8, rotation=30, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Time")

    fig.tight_layout()

    tmp = out_path.with_suffix(".png.tmp")
    fig.savefig(tmp, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    tmp.replace(out_path)
    if not quiet:
        print(f"[train]  {out_path}.")


class ResourceMonitor:
    """Internal helper."""

    PLOT_INTERVAL = 30.0
    MAX_PLOT_POINTS = 6000

    def __init__(self, save_dir, interval: float = 1.0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval = float(interval)
        self._path = self.save_dir / "stats.jsonl"
        self._png = self.save_dir / "usage.png"
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="res-monitor")
        self._plot_th = threading.Thread(target=self._plot_loop, daemon=True,
                                         name="res-monitor-plot")
        self._t0 = time.time()
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._last_t = None
        self._last_disk = None
        self._last_net = None

        self._series = []
        self._series_lock = threading.Lock()
        self._sample_n = 0
        self._keep_every = 1
        psutil.cpu_percent(percpu=True)


    @classmethod
    def launch(cls, base_dir=None, interval: float = 1.0) -> "ResourceMonitor":
        """Internal helper."""
        base = Path(base_dir) if base_dir is not None else _DEFAULT_BASE
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        m = cls(base / ts, interval=interval)
        m.start()
        return m

    def __enter__(self):
        if not self._thread.is_alive():
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


    def start(self) -> None:
        self._thread.start()
        self._plot_th.start()
        print(f"[train]  {self.save_dir}."
              f"[train]  {self.interval:g}; {self.PLOT_INTERVAL:.0f}.")

    def stop(self, *, final_plot: bool = True) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop_evt.set()
        self._thread.join(timeout=5)
        self._plot_th.join(timeout=5)
        if not final_plot:
            print("[train]")
            return
        print("[train]")
        try:
            _render(_read_jsonl(self._path), self._png)
        except Exception as e:
            print(f"[train]  {e}.")


    def _append_plot_series(self, record: dict) -> None:
        self._sample_n += 1
        if self._sample_n % self._keep_every != 0:
            return
        with self._series_lock:
            self._series.append(record)
            if len(self._series) > self.MAX_PLOT_POINTS:
                self._series = self._series[::2]
                self._keep_every *= 2

    def _loop(self) -> None:
        with self._path.open("w") as f:
            while not self._stop_evt.is_set():
                now = time.time()
                disk_now = _disk_io_counters()
                net_now = _net_io_counters()
                dt = (now - self._last_t) if self._last_t is not None else 0.0
                per_cpu = psutil.cpu_percent(percpu=True)
                record = {
                    "t": round(now - self._t0, 1),
                    "cpu": round(sum(per_cpu) / len(per_cpu), 1) if per_cpu else 0.0,
                    "mem": _sample_mem(),
                    "disk_io": _sample_disk_io(disk_now, self._last_disk, dt),
                    "net_io": _sample_net_io(net_now, self._last_net, dt),
                    "gpus": _sample_gpu() if _HAS_GPU else [],
                }
                self._last_t = now
                self._last_disk = disk_now
                self._last_net = net_now
                f.write(json.dumps(record) + "\n")
                f.flush()
                self._append_plot_series(record)
                self._stop_evt.wait(self.interval)

    def _plot_loop(self) -> None:
        while not self._stop_evt.wait(self.PLOT_INTERVAL):
            try:
                with self._series_lock:
                    rows = list(self._series)
                if rows:
                    _render(rows, self._png, quiet=True)
            except Exception as e:
                print(f"[train]  {e}.")


def start(base_dir=None, interval: float = 1.0) -> ResourceMonitor:
    """Internal helper."""
    return ResourceMonitor.launch(base_dir=base_dir, interval=interval)


def stop(monitor: ResourceMonitor, *, final_plot: bool = True) -> None:
    monitor.stop(final_plot=final_plot)


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="[train]")

    ap.add_argument("--input", nargs="?", default=None,
                    help="[train]")
    ap.add_argument("--output", default=None,
                    help="[train]")
    ap.add_argument("--interval", type=float, default=1.0, help="[train]")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="[train]")
    args = ap.parse_args()


    if args.input:
        target = Path(args.input)
        if target.is_dir():
            target = target / "stats.jsonl"
        if not target.exists():
            print(f"[train]  {target}.")
            raise SystemExit(1)
        _render(_read_jsonl(target), target.parent / "usage.png")
        return


    m = ResourceMonitor.launch(base_dir=args.output, interval=args.interval)
    import signal

    def _sig(signum, frame):
        m.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        if args.duration > 0:
            time.sleep(args.duration)
            m.stop()
        else:
            while True:
                time.sleep(3600)
    except SystemExit:
        pass


if __name__ == "__main__":
    _cli()
