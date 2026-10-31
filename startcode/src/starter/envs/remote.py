"""TCP environment used by the online fine-tunes.

The Isaac process is the server. The learner, running in the ``dp`` or ``act``
conda environment, is the client. Frames are newline-delimited JSON. Images
travel as JPEG so a 480p frame stays small.
"""

from __future__ import annotations

import base64
import json
import socket
import time
from typing import Any


class _Buffer:
    def __init__(self) -> None:
        self._buffer = b""


def serve(env, host: str, port: int, once: bool = False) -> None:
    """Serve batched ``reset`` / ``step`` until the client closes.

    One message steps every active scene. ``once`` exits after the first wave
    of ``num_envs`` episodes, so Diffusion Policy can start a fresh Isaac
    process instead of resetting this one a second time.
    """

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(1)
    width = int(getattr(env, "num_envs", 1))
    print(f"[env] listening on {host}:{port} num_envs {width}", flush=True)
    connection, _address = listener.accept()
    print("[env] learner connected", flush=True)
    incoming = _Buffer()
    finished = 0
    try:
        while True:
            message = _read(connection, incoming)
            if message is None:
                break
            op = message.get("op")
            if op == "ping":
                _write(connection, {"ok": True, "num_envs": width})
            elif op == "reset":
                seeds = message.get("seeds")
                if seeds is None:
                    seeds = [message.get("seed")]
                observations, infos = env.reset_batch(seeds)
                privileged = env.privileged_batch(list(range(width)))
                _write(
                    connection,
                    {
                        "observations": [_pack_observation(item) for item in observations],
                        "privileged": [_pack(item) for item in privileged],
                        "infos": [_pack(item) for item in infos],
                        "num_envs": width,
                    },
                )
            elif op == "step":
                active = [int(index) for index in message["active"]]
                results = env.step_batch(message["actions"], active)
                privileged = env.privileged_batch(active)
                terminated = [bool(item[2]) for item in results]
                truncated = [bool(item[3]) for item in results]
                _write(
                    connection,
                    {
                        "observations": [_pack_observation(item[0]) for item in results],
                        "privileged": [_pack(item) for item in privileged],
                        "scores": [float(item[1]) for item in results],
                        "terminated": terminated,
                        "truncated": truncated,
                        "infos": [_pack(item[4]) for item in results],
                    },
                )
                finished += sum(1 for done_flag, cut in zip(terminated, truncated) if done_flag or cut)
                if once and finished >= width:
                    break
            elif op == "close":
                _write(connection, {"ok": True})
                break
            else:
                _write(connection, {"error": f"unknown op {op}"})
    finally:
        connection.close()
        listener.close()
        print(f"[env] closed after {finished} episode(s)", flush=True)


class RemoteEnv:
    """Client side of :func:`serve`. ``reset`` waits until Isaac is listening."""

    def __init__(
        self,
        host: str,
        port: int,
        restart_each_episode: bool = False,
        timeout: float = 2400.0,
        reset_timeout: float = 10800.0,
    ):
        self.host = host
        self.port = int(port)
        self.restart_each_episode = restart_each_episode
        self.timeout = float(timeout)
        # The first reset builds every cluttered scene. That exceeds the
        # per-step socket timeout when the policy and Isaac share one GPU.
        self.reset_timeout = float(reset_timeout)
        self._socket: socket.socket | None = None
        self._buffer = b""
        self._privileged: list[dict] = []

    def reset_batch(self, seeds):
        # The simulator process is restarted until it stays up. A reset that
        # dies with the process should wait for the next one. A live reset
        # keeps the socket open for reset_timeout; closing it early kills
        # the scene build.
        deadline = time.monotonic() + self.reset_timeout
        last_error: Exception | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConnectionError(
                    f"Isaac env server did not complete reset within {self.reset_timeout:.0f}s"
                ) from last_error
            self._ensure(timeout=remaining)
            try:
                reply = self._request(
                    {"op": "reset", "seeds": [None if seed is None else int(seed) for seed in seeds]},
                    timeout=remaining,
                )
            except (TimeoutError, socket.timeout) as exc:
                self.close()
                raise ConnectionError(
                    f"Isaac env server did not complete reset within {self.reset_timeout:.0f}s"
                ) from exc
            except ConnectionError as exc:
                last_error = exc
                self.close()
                time.sleep(min(2.0, max(deadline - time.monotonic(), 0.0)))
                continue
            observations = [_unpack_observation(item) for item in reply["observations"]]
            return observations, reply.get("infos") or [{} for _ in observations]

    def step_batch(self, actions, active):
        reply = self._request(
            {
                "op": "step",
                "actions": [[float(item) for item in action] for action in actions],
                "active": [int(index) for index in active],
            }
        )
        results = []
        infos = reply.get("infos") or [{} for _ in reply["scores"]]
        for index, observation in enumerate(reply["observations"]):
            results.append(
                (
                    _unpack_observation(observation),
                    float(reply["scores"][index]),
                    bool(reply["terminated"][index]),
                    bool(reply["truncated"][index]),
                    infos[index],
                )
            )
        return results

    def privileged(self, index: int) -> dict:
        """Privileged state aligned with the last reset or step reply."""

        return dict(self._privileged[index])

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            _write(self._socket, {"op": "close"})
        except OSError:
            pass
        self._socket.close()
        self._socket = None
        self._buffer = b""

    def _ensure(self, timeout: float | None = None) -> None:
        if self._socket is not None:
            return
        limit = self.timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            connect_timeout = min(5.0, max(deadline - time.monotonic(), 0.1))
            try:
                sock = socket.create_connection((self.host, self.port), timeout=connect_timeout)
            except OSError as exc:
                last_error = exc
                time.sleep(min(2.0, max(deadline - time.monotonic(), 0.0)))
                continue
            sock.settimeout(self.timeout)
            self._socket = sock
            self._buffer = b""
            self._privileged = []
            return
        raise TimeoutError(
            f"Isaac env server did not accept {self.host}:{self.port} within {limit:.0f}s"
        ) from last_error

    def _request(self, message: dict, timeout: float | None = None) -> dict:
        if self._socket is None:
            raise RuntimeError("Remote env is not connected.")
        self._socket.settimeout(self.timeout if timeout is None else float(timeout))
        _write(self._socket, message)
        reply = _read(self._socket, self)
        if reply is None:
            self.close()
            raise ConnectionError("Isaac env server closed the connection.")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        if "privileged" in reply:
            self._privileged = list(reply["privileged"])
        return reply

def _read(connection: socket.socket, holder: Any) -> dict | None:
    buffer = b"" if holder is None else holder._buffer
    while b"\n" not in buffer:
        chunk = connection.recv(1024 * 1024)
        if not chunk:
            if holder is not None:
                holder._buffer = buffer
            return None
        buffer += chunk
    line, _, rest = buffer.partition(b"\n")
    if holder is not None:
        holder._buffer = rest
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


def _write(connection: socket.socket, message: dict) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
    connection.sendall(payload)


def _pack_observation(observation: dict) -> dict:
    packed = {}
    for key, value in observation.items():
        if key == "image":
            packed[key] = _pack_image(value)
        else:
            packed[key] = _pack(value)
    return packed


def _unpack_observation(payload: dict) -> dict:
    observation = {}
    for key, value in payload.items():
        if key == "image":
            observation[key] = _unpack_image(value)
        else:
            observation[key] = value
    return observation


def _pack_image(value) -> dict:
    import cv2
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode the camera frame.")
    return {
        "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "shape": list(array.shape),
    }


def _unpack_image(payload: dict):
    import cv2
    import numpy as np

    raw = base64.b64decode(payload["jpeg"])
    array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise RuntimeError("Failed to decode the camera frame.")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def _pack(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)
