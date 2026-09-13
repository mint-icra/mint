#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端口自愈（杀本脚本旧实例 / 改用空闲端口）+ 启动后自动打开浏览器。纯标准库，无项目内依赖。"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path


def port_free(port: int) -> bool:
    """能否在 0.0.0.0:port 上 bind（判断端口是否空闲）。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def listen_inodes(port: int) -> set[str]:
    """从 /proc/net/tcp{,6} 找到在 port 上 LISTEN 的 socket inode 集合（纯读文件，无需外部命令）。"""
    inodes: set[str] = set()
    for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(fn).read_text().splitlines()[1:]
        except Exception:  # noqa: BLE001
            continue
        for ln in lines:
            f = ln.split()
            if len(f) < 10 or f[3] != "0A":      # f[3]==0A 即 TCP_LISTEN
                continue
            try:
                if int(f[1].rsplit(":", 1)[1], 16) == port:
                    inodes.add(f[9])             # f[9] = socket inode
            except Exception:  # noqa: BLE001
                continue
    return inodes


def pids_on_port(port: int) -> list[int]:
    """占用 port 的 LISTEN 进程 pid 列表：优先解析 /proc（Linux 原生），回退 lsof/ss。"""
    inodes = listen_inodes(port)
    if inodes:
        pids: list[int] = []
        for pd in Path("/proc").glob("[0-9]*"):
            try:
                for fd in (pd / "fd").iterdir():
                    try:
                        tgt = os.readlink(fd)
                    except OSError:
                        continue
                    if tgt.startswith("socket:[") and tgt[8:-1] in inodes:
                        pids.append(int(pd.name))
                        break
            except OSError:                      # 进程已退出/无权限
                continue
        if pids:
            return sorted(set(pids))
    # 回退：lsof / ss（补全 PATH，非 Linux 或 /proc 不可用时）
    import subprocess
    env = dict(os.environ, PATH=os.environ.get("PATH", "") + ":/usr/sbin:/sbin:/usr/bin:/bin")
    for cmd in (["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                ["ss", "-ltnpH", f"sport = :{port}"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env=env).stdout
        except Exception:  # noqa: BLE001
            continue
        got = sorted({int(p) for p in (re.findall(r"pid=(\d+)", out)
                                       or re.findall(r"^\s*(\d+)\s*$", out, re.M))})
        if got:
            return got
    return []


def cmd_of(pid: int) -> str:
    """尽量取进程命令行（Linux /proc 优先，回退 ps）。"""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
    except Exception:  # noqa: BLE001
        import subprocess
        try:
            return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                  capture_output=True, text=True, timeout=5).stdout
        except Exception:  # noqa: BLE001
            return ""


def ensure_port(port: int) -> int:
    """启动前自愈端口：固定端口、每次强制腾出。空闲直接用；被占用则不管是不是本脚本，
    一律 SIGTERM→（等不掉再）SIGKILL 把占该端口的 LISTEN 进程清掉，始终复用请求端口；
    只有实在杀不掉（如权限不足/杀不动的系统进程）才退回自动改用下一个空闲端口。"""
    import signal
    import time as _t
    if port_free(port):
        return port
    victims = [pid for pid in pids_on_port(port) if pid != os.getpid()]
    for pid in victims:
        own = "viewer_web.py" in cmd_of(pid)
        print(f"[port] {port} 被{'本脚本旧实例' if own else '其他进程'} PID {pid} 占用，SIGTERM 杀掉…",
              flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
    if victims:
        for _ in range(25):                 # 最多等 5s 让其退出
            _t.sleep(0.2)
            if port_free(port):
                print(f"[port] {port} 已释放。", flush=True)
                return port
        for pid in victims:                 # 还赖着就强杀
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        _t.sleep(0.5)
        if port_free(port):
            print(f"[port] {port} 已释放（SIGKILL）。", flush=True)
            return port
    for p in range(port + 1, port + 50):    # 实在腾不掉（权限/杀不动）才退回换端口
        if port_free(p):
            print(f"[port] {port} 无法腾出（杀不掉占用进程），退回改用空闲端口 {p}。", flush=True)
            return p
    return port


def _browser_candidates() -> list[str]:
    """GUI/remote browser openers only; skip terminal browsers that dump HTML."""
    candidates: list[str] = []
    browser = os.environ.get("BROWSER", "")
    if browser:
        candidates.extend(x for x in browser.split(os.pathsep) if x.strip())
    if os.name == "posix":
        candidates.extend(["xdg-open", "gio open", "sensible-browser", "open"])
    return candidates


def _spawn_browser(cmd: str, url: str) -> bool:
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return False
    if not argv:
        return False
    exe = argv[0]
    if Path(exe).name in {"w3m", "lynx", "links", "elinks", "www-browser"}:
        return False
    if os.path.isabs(exe):
        if not Path(exe).exists():
            return False
    elif shutil.which(exe) is None:
        return False

    if any("%s" in part for part in argv):
        argv = [part.replace("%s", url) for part in argv]
    else:
        argv.append(url)

    try:
        with open(os.devnull, "rb") as null_in, open(os.devnull, "wb") as null_out:
            subprocess.Popen(argv, stdin=null_in, stdout=null_out, stderr=null_out,
                             close_fds=True, start_new_session=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def open_browser(port: int) -> None:
    """后台线程里等 Flask 起来再尝试自动打开浏览器，并吞掉 opener 自身输出。"""
    import time as _t
    _t.sleep(1.2)
    url = f"http://localhost:{port}/"
    for cmd in _browser_candidates():
        if _spawn_browser(cmd, url):
            print(f"[open] 已尝试自动打开浏览器：{url}", flush=True)
            return
    print(f"[open] 未找到可用图形/远端浏览器命令，请手动访问：{url}", flush=True)
