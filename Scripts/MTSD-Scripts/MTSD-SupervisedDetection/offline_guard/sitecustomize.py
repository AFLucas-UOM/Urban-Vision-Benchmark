"""Process-wide network guard for reproducible offline MTSD experiments."""
from __future__ import annotations

import ipaddress
import os
import socket


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True
    host = str(address[0]).strip("[]").casefold()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


if os.getenv("UV_OFFLINE_GUARD") == "1":
    _original_connect = socket.socket.connect
    _original_connect_ex = socket.socket.connect_ex

    def _guarded_connect(self, address):
        if self.family in {socket.AF_INET, socket.AF_INET6} and not _is_loopback(address):
            raise OSError(f"Offline experiment blocked network connection to {address!r}")
        return _original_connect(self, address)

    def _guarded_connect_ex(self, address):
        if self.family in {socket.AF_INET, socket.AF_INET6} and not _is_loopback(address):
            return 10013
        return _original_connect_ex(self, address)

    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    os.environ["UV_OFFLINE_GUARD_ACTIVE"] = "1"
