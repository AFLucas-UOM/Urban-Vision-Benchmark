"""Tests for the UVB launcher's process lifecycle and command construction.

All subprocess spawning, port probing, browser opening, sleeping and process
termination are injected fakes: no real process is started, no real port is
opened, and no real browser tab appears anywhere in this suite.
"""

import json
import subprocess
from pathlib import Path

import pytest

import launch_uvb as lv


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class DummyProcess:
    """Stand-in for subprocess.Popen with controllable liveness."""

    _next_pid = 4000

    def __init__(self):
        DummyProcess._next_pid += 1
        self.pid = DummyProcess._next_pid
        self.exit_code = None
        self.wait_raises = False

    def poll(self):
        return self.exit_code

    def wait(self, timeout=None):
        if self.wait_raises and self.exit_code is None:
            raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout)
        return self.exit_code if self.exit_code is not None else 0


class Harness:
    """Bundles the injected fakes and the AppManager under test."""

    def __init__(self, tmp_path: Path):
        self.spawned = []          # (cmd, cwd) tuples
        self.terminated = []       # pids handed to terminate
        self.browser_urls = []     # urls opened
        self.browser_result = True
        self.port_open = {}        # port -> bool
        self.spawn_binds = True    # spawned apps bind their port immediately
        self.time = 0.0
        self.processes = []

        def spawn(cmd, cwd, log_handle):
            proc = DummyProcess()
            self.processes.append(proc)
            self.spawned.append((cmd, cwd))
            if self.spawn_binds and "--port" in cmd:
                # Simulate the server binding its port right after start.
                self.port_open[int(cmd[cmd.index("--port") + 1])] = True
            return proc

        def terminate(pid):
            self.terminated.append(pid)
            for proc in self.processes:
                if proc.pid == pid and proc.exit_code is None:
                    proc.exit_code = -9

        def port_check(port):
            return self.port_open.get(port, False)

        def browser(url):
            self.browser_urls.append(url)
            return self.browser_result

        def sleep(seconds):
            self.time += seconds

        self.manager = lv.AppManager(
            spawn=spawn, terminate=terminate, port_check=port_check,
            browser_open=browser, sleep=sleep, clock=lambda: self.time,
            state_file=tmp_path / "state.json", log_dir=tmp_path / "logs")


def make_tool(**overrides):
    defaults = dict(
        id="test-app", title="Test app", category="Local Web Applications",
        description="test", script="launch_uvb.py", cwd=".", env=None,
        port=7999, port_arg="--port", no_browser_arg="--no-browser",
        url="http://127.0.0.1:{port}", browser=True, confirm=False,
        ready_timeout=10.0)
    defaults.update(overrides)
    return lv.Tool(**defaults)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    # Module-level port probe used by resolve_port: nothing is listening.
    monkeypatch.setattr(lv, "port_is_listening", lambda port, **kw: h.port_open.get(port, False))
    monkeypatch.setattr(lv, "describe_port_owner", lambda port: "PID 1234 (other.exe)")
    return h


# ---------------------------------------------------------------------------
# Command construction and conda environment selection
# ---------------------------------------------------------------------------

def test_command_uses_current_python_without_conda(monkeypatch):
    monkeypatch.setattr(lv, "find_conda", lambda: None)
    tool = make_tool(env="mtsd-base")
    cmd = lv.command_for(tool, 7999)
    assert Path(cmd[0]).name.startswith("python")
    assert "--port" in cmd and "7999" in cmd


def test_command_selects_conda_env(monkeypatch):
    monkeypatch.setattr(lv, "find_conda", lambda: Path("C:/fake/conda.exe"))
    tool = make_tool(env="mtsd-attrcls")
    cmd = lv.command_for(tool, 8001)
    assert cmd[:4] == [str(Path("C:/fake/conda.exe")), "run", "-n", "mtsd-attrcls"]
    assert cmd[4] == "python"
    assert cmd[-2:] == ["--port", "8001"]


def test_managed_command_appends_no_browser_flag(monkeypatch):
    monkeypatch.setattr(lv, "find_conda", lambda: None)
    tool = make_tool()
    assert "--no-browser" in lv.command_for(tool, 7999, managed=True)
    assert "--no-browser" not in lv.command_for(tool, 7999, managed=False)


def test_command_script_relative_to_cwd(monkeypatch):
    monkeypatch.setattr(lv, "find_conda", lambda: None)
    tool = make_tool(script="Scripts/Automation/verify_repository_health.py",
                     cwd="Scripts/Automation", port=None, port_arg=None, url=None)
    cmd = lv.command_for(tool)
    assert "verify_repository_health.py" in cmd
    assert not any("Scripts" in part for part in cmd[1:2])


# ---------------------------------------------------------------------------
# Readiness polling
# ---------------------------------------------------------------------------

def test_wait_until_ready_success(harness):
    tool = make_tool()
    harness.spawn_binds = False  # the server takes a while to come up
    app = harness.manager.start(tool, 7999)
    ticks = []

    def tick(remaining):
        ticks.append(remaining)
        if harness.time >= 2.0:
            harness.port_open[7999] = True

    assert harness.manager.wait_until_ready(app, 10.0, on_tick=tick) == "ready"
    assert ticks  # polled at least once before readiness


def test_wait_until_ready_process_exit(harness):
    tool = make_tool()
    app = harness.manager.start(tool, 7999)
    app.process.exit_code = 3
    assert harness.manager.wait_until_ready(app, 10.0) == "exited"


def test_wait_until_ready_timeout(harness):
    tool = make_tool()
    harness.spawn_binds = False  # the server never comes up
    app = harness.manager.start(tool, 7999)
    assert harness.manager.wait_until_ready(app, 5.0) == "timeout"
    assert harness.time >= 5.0


# ---------------------------------------------------------------------------
# Web-app launch outcomes
# ---------------------------------------------------------------------------

def test_successful_startup_opens_browser_once(harness):
    tool = make_tool()
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert app is not None and app.running
    assert harness.browser_urls == ["http://127.0.0.1:7999"]
    assert len(harness.spawned) == 1


def test_startup_failure_opens_no_browser(harness):
    tool = make_tool()
    harness.spawn_binds = False
    # Process dies immediately, port never opens.
    original_spawn = harness.manager._spawn

    def dying_spawn(cmd, cwd, log_handle):
        proc = original_spawn(cmd, cwd, log_handle)
        proc.exit_code = 1
        return proc

    harness.manager._spawn = dying_spawn
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert app is None
    assert harness.browser_urls == []
    assert harness.manager.running_apps() == []


def test_startup_timeout_stops_process_and_opens_no_browser(harness):
    tool = make_tool(ready_timeout=3.0)
    harness.spawn_binds = False  # the server never comes up
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert app is None
    assert harness.browser_urls == []
    assert harness.terminated  # the unready process tree was cleaned up
    assert harness.manager.running_apps() == []


def test_browser_open_failure_keeps_app_running(harness):
    tool = make_tool()
    harness.browser_result = False
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert app is not None and app.running  # failure to open a tab never kills the app
    assert harness.manager.running_app(tool.id) is app


def test_missing_script_refuses_launch(harness):
    tool = make_tool(script="does/not/exist.py")
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert app is None
    assert harness.spawned == []


# ---------------------------------------------------------------------------
# Occupied ports
# ---------------------------------------------------------------------------

def test_occupied_port_with_port_arg_uses_verified_free_port(harness, monkeypatch):
    tool = make_tool()
    harness.port_open[7999] = True  # unrelated process owns the preferred port
    monkeypatch.setattr(lv, "find_free_port", lambda preferred, attempts=25: 8042)
    port, message = lv.resolve_port(tool, harness.manager)
    assert port == 8042
    assert "unrelated process" in message and "8042" in message


def test_occupied_port_without_port_arg_refuses(harness):
    tool = make_tool(port_arg=None)
    harness.port_open[7999] = True
    port, message = lv.resolve_port(tool, harness.manager)
    assert port is None
    assert "refused" in message


def test_occupied_port_names_launcher_owned_app(harness, monkeypatch):
    other = make_tool(id="other-app")
    app = harness.manager.start(other, 7999)
    harness.port_open[7999] = True
    monkeypatch.setattr(lv, "find_free_port", lambda preferred, attempts=25: 8042)
    tool = make_tool(id="second-app")
    port, message = lv.resolve_port(tool, harness.manager)
    assert port == 8042
    assert "other-app" in message
    assert app.running  # the other app was never touched


# ---------------------------------------------------------------------------
# Repeat launch, stop, restart
# ---------------------------------------------------------------------------

def test_repeat_launch_does_not_spawn_duplicate(harness):
    tool = make_tool()
    first = lv.launch_web_app(tool, harness.manager, interactive=False)
    again = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert again is first
    assert len(harness.spawned) == 1


def test_managed_stop_terminates_tree_and_forgets_app(harness):
    tool = make_tool()
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert harness.manager.stop(tool.id) is True
    assert harness.terminated == [app.process.pid]
    assert harness.manager.running_apps() == []
    assert not harness.manager._state_file.exists()


def test_restart_spawns_fresh_process(harness):
    tool = make_tool()
    first = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert harness.manager.stop(tool.id)
    harness.port_open[7999] = False  # the stopped server released its port
    second = lv.launch_web_app(tool, harness.manager, interactive=False)
    assert second is not None and second.process.pid != first.process.pid
    assert len(harness.spawned) == 2


def test_stop_reports_failure_when_tree_survives(harness):
    tool = make_tool()
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    app.process.wait_raises = True
    harness.manager._terminate = lambda pid: None  # termination has no effect
    assert harness.manager.stop(tool.id) is False


# ---------------------------------------------------------------------------
# Cleanup paths
# ---------------------------------------------------------------------------

def test_stop_all_terminates_every_managed_app(harness):
    a = harness.manager.start(make_tool(id="app-a"), 7999)
    b = harness.manager.start(make_tool(id="app-b", port=8000), 8000)
    harness.manager.stop_all()
    assert sorted(harness.terminated) == sorted([a.process.pid, b.process.pid])
    assert harness.manager.running_apps() == []


def test_ctrl_c_during_cli_tool_kills_process_tree(monkeypatch):
    tool = make_tool(id="cli", port=None, port_arg=None, url=None, browser=False,
                     script="launch_uvb.py")
    killed = []
    wait_calls = []

    class InterruptingProcess(DummyProcess):
        def wait(self, timeout=None):
            wait_calls.append(timeout)
            if len(wait_calls) == 1:
                raise KeyboardInterrupt  # user pressed Ctrl+C mid-run
            return 0

    interrupting = InterruptingProcess()
    monkeypatch.setattr(lv, "_default_spawn_foreground", lambda cmd, cwd: interrupting)
    monkeypatch.setattr(lv, "terminate_process_tree", lambda pid: killed.append(pid))
    lv.run_cli_tool(tool)
    assert killed == [interrupting.pid]
    assert len(wait_calls) == 2  # interrupted wait + post-termination wait


def test_state_file_written_and_cleared(harness):
    tool = make_tool()
    app = lv.launch_web_app(tool, harness.manager, interactive=False)
    state = json.loads(harness.manager._state_file.read_text(encoding="utf-8"))
    assert state["apps"][tool.id]["pid"] == app.process.pid
    assert state["apps"][tool.id]["port"] == 7999
    harness.manager.stop_all()
    assert not harness.manager._state_file.exists()


# ---------------------------------------------------------------------------
# Stale state: never terminate a PID that is no longer ours
# ---------------------------------------------------------------------------

def _write_stale_state(manager, pid, command="conda run -n x python app.py --port 7999"):
    manager._state_file.write_text(json.dumps({
        "apps": {"test-app": {"pid": pid, "port": 7999, "command": command,
                              "started_at": "2026-07-16T10:00:00"}}}), encoding="utf-8")


def test_stale_pid_reused_by_unrelated_process_is_never_killed(harness, monkeypatch):
    _write_stale_state(harness.manager, 5555)
    killed = []
    monkeypatch.setattr(lv, "pid_command_line", lambda pid: "C:/Windows/explorer.exe")
    monkeypatch.setattr(lv, "terminate_process_tree", lambda pid: killed.append(pid))
    monkeypatch.setattr(lv, "confirm", lambda prompt: True)
    lv.report_stale_state(harness.manager, interactive=True)
    assert killed == []
    assert not harness.manager._state_file.exists()  # stale entry dropped


def test_stale_pid_gone_is_dropped_silently(harness, monkeypatch):
    _write_stale_state(harness.manager, 5556)
    killed = []
    monkeypatch.setattr(lv, "pid_command_line", lambda pid: None)
    monkeypatch.setattr(lv, "terminate_process_tree", lambda pid: killed.append(pid))
    lv.report_stale_state(harness.manager, interactive=True)
    assert killed == []


def test_stale_pid_with_matching_command_offered_for_stop(harness, monkeypatch):
    command = "conda run -n x python app.py --port 7999"
    _write_stale_state(harness.manager, 5557, command)
    killed = []
    monkeypatch.setattr(lv, "pid_command_line", lambda pid: command)
    monkeypatch.setattr(lv, "terminate_process_tree", lambda pid: killed.append(pid))
    monkeypatch.setattr(lv, "confirm", lambda prompt: True)
    lv.report_stale_state(harness.manager, interactive=True)
    assert killed == [5557]


def test_stale_pid_not_killed_without_interactive_confirmation(harness, monkeypatch):
    command = "conda run -n x python app.py --port 7999"
    _write_stale_state(harness.manager, 5558, command)
    killed = []
    monkeypatch.setattr(lv, "pid_command_line", lambda pid: command)
    monkeypatch.setattr(lv, "terminate_process_tree", lambda pid: killed.append(pid))
    lv.report_stale_state(harness.manager, interactive=False)
    assert killed == []


# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------

def test_find_free_port_skips_bound_port():
    import socket

    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        taken = blocker.getsockname()[1]
        found = lv.find_free_port(taken, attempts=5)
        assert found is not None and found != taken


def test_port_is_listening_detects_live_listener():
    import socket

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert lv.port_is_listening(port) is True
    assert lv.port_is_listening(port) is False


# ---------------------------------------------------------------------------
# Non-Rich fallback and registry sanity
# ---------------------------------------------------------------------------

def test_non_rich_fallback_output(monkeypatch, capsys):
    monkeypatch.setattr(lv, "RICH", False)
    lv.say("hello", "bold cyan")
    lv.list_tools()
    out = capsys.readouterr().out
    assert "hello" in out
    assert "promptdetect" in out


def test_tool_registry_ids_unique_and_scripts_exist():
    ids = [t.id for t in lv.TOOLS]
    assert len(ids) == len(set(ids))
    for tool in lv.TOOLS:
        if tool.script:
            assert (lv.ROOT / tool.script).exists(), f"missing script for {tool.id}: {tool.script}"


def test_every_web_app_has_managed_launch_fields():
    for tool in lv.TOOLS:
        if tool.is_web_app:
            assert tool.port_arg, f"{tool.id} must accept an explicit port"
            assert tool.no_browser_arg, f"{tool.id} must accept a no-browser flag"
            assert "{port}" in tool.url


def test_web_app_wording_avoids_website_language():
    for tool in lv.TOOLS:
        text = (tool.title + " " + tool.description + " " + tool.notes).lower()
        assert "website" not in text
