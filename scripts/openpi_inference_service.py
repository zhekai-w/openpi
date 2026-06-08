"""
OpenPI Inference Service

Drop-in ZMQ server that serves an OpenPI UR5 policy behind the same wire
protocol as `pkgs/lerobot/src/lerobot/scripts/server/smolvla_inference_service.py`,
so the existing `pkgs/Isaac-GR00T/scripts/ur5_gr00t_simple_client.py` works
without any changes.

The client sends:
    video.azure_kinect                 : (1, H, W, 3) uint8
    video.wfov                         : (1, H, W, 3) uint8
    state.ur5_arm                      : (1, 6) float64
    state.gripper                      : (1, 1) float64
    annotation.human.task_description  : list[str]

The server returns:
    action.ur5_arm : (action_horizon, 6) float64
    action.gripper : (action_horizon,)   float64

Server usage:
    python pkgs/openpi/scripts/openpi_inference_service.py \
        --server --config pi0_ur5_lora \
        --checkpoint-dir /path/to/checkpoint --port 5555

Smoke-test client (no robot needed):
    python pkgs/openpi/scripts/openpi_inference_service.py --client --port 5555
"""

import io
import time
from dataclasses import dataclass

import msgpack
import numpy as np
import tyro
import zmq

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# ---------------------------------------------------------------------------
# Wire codec — msgpack + numpy, bit-compatible with gr00t/eval/service.py
# ---------------------------------------------------------------------------


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def pack(data: dict) -> bytes:
    return msgpack.packb(data, default=_encode)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=_decode)


# ---------------------------------------------------------------------------
# OpenPI ZMQ server
# ---------------------------------------------------------------------------


class OpenPIZMQServer:
    def __init__(
        self,
        policy,
        *,
        host: str,
        port: int,
        api_token: str | None,
    ):
        self.policy = policy
        self.api_token = api_token

        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"OpenPI server listening on {addr}")
        while self.running:
            try:
                request = unpack(self.socket.recv())
                if self.api_token is not None and request.get("api_token") != self.api_token:
                    self.socket.send(pack({"error": "Unauthorized: Invalid API token"}))
                    continue

                endpoint = request.get("endpoint", "get_action")
                if endpoint == "ping":
                    self.socket.send(pack({"status": "ok", "message": "OpenPI server running"}))
                elif endpoint == "kill":
                    self.running = False
                    self.socket.send(pack({"status": "ok"}))
                elif endpoint == "get_modality_config":
                    self.socket.send(pack(self._modality_config()))
                elif endpoint == "get_action":
                    action = self._get_action(request.get("data", {}))
                    self.socket.send(pack(action))
                else:
                    self.socket.send(pack({"error": f"Unknown endpoint: {endpoint}"}))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.socket.send(pack({"error": str(e)}))

    def _modality_config(self) -> dict:
        return {
            "video": {"modality_keys": ["video.azure_kinect", "video.wfov"]},
            "state": {"modality_keys": ["state.ur5_arm", "state.gripper"]},
            "action": {"modality_keys": ["action.ur5_arm", "action.gripper"]},
        }

    def _get_action(self, obs: dict) -> dict:
        # Translate GR00T observation format -> OpenPI UR5 format
        base_rgb = np.asarray(obs["video.azure_kinect"])[0]   # (H, W, 3) uint8
        wrist_rgb = np.asarray(obs["video.wfov"])[0]          # (H, W, 3) uint8
        state = np.concatenate(
            [np.asarray(obs["state.ur5_arm"][0], dtype=np.float32),
             np.asarray(obs["state.gripper"][0], dtype=np.float32)],
            axis=-1,
        )  # (7,) float32
        task_list = obs["annotation.human.task_description"]
        prompt = task_list[0] if isinstance(task_list, (list, tuple)) else task_list
        print(f"Task: {prompt}")

        t0 = time.time()
        output = self.policy.infer({
            "base_rgb": base_rgb,
            "wrist_rgb": wrist_rgb,
            "state": state,
            "prompt": prompt,
        })
        print(f"Inference took {time.time() - t0:.3f}s")

        # output["actions"]: (action_horizon, 7)  — UR5Outputs already slices [:, :7]
        actions = np.asarray(output["actions"])
        return {
            "action.ur5_arm": np.ascontiguousarray(actions[:, :6], dtype=np.float64),
            "action.gripper": np.ascontiguousarray(actions[:, 6], dtype=np.float64),
        }


# ---------------------------------------------------------------------------
# Smoke-test client
# ---------------------------------------------------------------------------


def _smoke_test_client(host: str, port: int, api_token: str | None):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{host}:{port}")

    def call(endpoint, data=None):
        req: dict = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        if api_token:
            req["api_token"] = api_token
        sock.send(pack(req))
        return unpack(sock.recv())

    print("ping:", call("ping"))
    print("modality_config:", call("get_modality_config"))

    obs = {
        "video.azure_kinect": np.random.randint(0, 256, (1, 360, 640, 3), dtype=np.uint8),
        "video.wfov": np.random.randint(0, 256, (1, 360, 640, 3), dtype=np.uint8),
        "state.ur5_arm": np.random.rand(1, 6).astype(np.float64),
        "state.gripper": np.random.rand(1, 1).astype(np.float64),
        "annotation.human.task_description": ["place the small cube on the red box."],
    }
    t0 = time.time()
    action = call("get_action", obs)
    print(f"get_action took {time.time() - t0:.3f}s")
    if "error" in action:
        print("ERROR:", action["error"])
        return
    for k, v in action.items():
        arr = np.asarray(v)
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class ArgsConfig:
    """OpenPI inference service (GR00T-protocol-compatible ZMQ server)."""

    config: str | None = None
    """Training config name, e.g. 'pi0_ur5_lora'. Required for --server."""

    checkpoint_dir: str | None = None
    """Path to checkpoint directory (local or GCS). Required for --server."""

    default_prompt: str | None = None
    """Fallback language prompt when client doesn't send one."""

    port: int = 5555
    """TCP port to bind (server) / connect (client)."""

    host: str = "0.0.0.0"
    """Host to bind (server) or connect (client)."""

    api_token: str | None = None
    """Optional API token for authentication."""

    server: bool = False
    """Run as ZMQ server."""

    client: bool = False
    """Run synthetic smoke-test client (no robot needed)."""


def main(args: ArgsConfig):
    if args.server:
        assert args.config is not None, "Need --config for server mode"
        assert args.checkpoint_dir is not None, "Need --checkpoint-dir for server mode"
        print(f"Loading OpenPI policy: config={args.config}, checkpoint={args.checkpoint_dir}")
        train_config = _config.get_config(args.config)
        policy = _policy_config.create_trained_policy(
            train_config,
            args.checkpoint_dir,
            default_prompt=args.default_prompt,
        )
        print("Policy loaded.")

        server = OpenPIZMQServer(
            policy,
            host=args.host,
            port=args.port,
            api_token=args.api_token,
        )
        server.run()
    elif args.client:
        host = "localhost" if args.host == "0.0.0.0" else args.host
        _smoke_test_client(host, args.port, args.api_token)
    else:
        raise ValueError("Pass --server or --client")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
