import json
import os
import subprocess
import time
from typing import List

import requests

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1337
DEFAULT_BIN = "target/release/fourier"
DEFAULT_SETUP_PATH = "setup"
DEFAULT_PRECOMPUTE_PATH = "precompute"
DEFAULT_SKIP_PRECOMPUTE = False


class RPCRequest:
    def __init__(self, method="ping", id=0, params=None):
        self.id = id
        self.method = method
        self.params = params
        self.jsonrpc = "2.0"

    def json(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def ping():
        return RPCRequest()

    @staticmethod
    def commit(poly):
        params = {"poly": poly}
        return RPCRequest(method="commit", params=params)

    @staticmethod
    def open(poly, x):
        params = {"poly": poly, "x": x}
        return RPCRequest(method="open", params=params)

    @staticmethod
    def verify(proof, x, y, commitment):
        params = {"proof": proof, "x": x, "y": y, "commitment": commitment}
        return RPCRequest(method="verify", params=params)

    @staticmethod
    def random_poly(degree):
        params = {"degree": degree}
        return RPCRequest(method="randomPoly", params=params)

    @staticmethod
    def random_point():
        return RPCRequest(method="randomPoint")

    @staticmethod
    def evaluate(poly, x):
        params = {"poly": poly, "x": x}
        return RPCRequest(method="evaluate", params=params)

    @staticmethod
    def prove(poly):
        params = {"poly": poly}
        return RPCRequest(method="prove", params=params)


class CLI:
    def __init__(self, bin=DEFAULT_BIN):
        if not os.path.exists(bin):
            print(f"Binary does not exist: {bin}")
            raise FileNotFoundError
        self.bin = bin
        self.process = None

    def cmd(self, args: List[str]):
        return [
            self.bin,
            *args,
        ]

    def wait_until_running(self) -> bool:
        time.sleep(1)
        total_sleep = 0
        while not self.is_running():
            total_sleep += 1
            time.sleep(1)
            if total_sleep > 10:
                print("Failed to start process.")
                return False
        return True

    def run(
        self,
        host=None,
        port=None,
        scale=None,
        setup_path=None,
        precompute_path=None,
        uncompressed=False,
    ) -> bool:
        HOST_LONG = "--host"
        PORT_LONG = "--port"
        SCALE_LONG = "--scale"
        SETUP_PATH_LONG = "--setup-path"
        PRECOMPUTE_PATH_LONG = "--precompute-path"
        UNCOMPRESSED_LONG = "--uncompressed"
        args = ["run"]
        if host:
            args.extend([HOST_LONG, host])
        if port:
            args.extend([PORT_LONG, str(port)])
        if scale:
            args.extend([SCALE_LONG, str(scale)])
        if setup_path:
            args.extend([SETUP_PATH_LONG, setup_path])
        if precompute_path:
            args.extend([PRECOMPUTE_PATH_LONG, precompute_path])
        if uncompressed:
            args.extend([UNCOMPRESSED_LONG])
        print(f"Running: {self.cmd(args)}")
        self.process = subprocess.Popen(args=self.cmd(args))
        return self.wait_until_running()

    def setup(
        self,
        setup_path=None,
        overwrite=False,
        scale=None,
        precompute_path=None,
        generate_setup=False,
        generate_precompute=False,
        uncompressed=False,
        compress_existing=False,
        decompress_existing=False,
    ):
        SETUP_PATH_LONG = "--setup-path"
        PRECOMPUTE_PATH_LONG = "--precompute-path"
        SCALE_LONG = "--scale"
        OVERWRITE_LONG = "--overwrite"
        GENERATE_SETUP_LONG = "--generate-setup"
        GENERATE_PRECOMPUTE_LONG = "--generate-precompute"
        UNCOMPRESSED_LONG = "--uncompressed"
        COMPRESS_EXISTING_LONG = "--compress-existing"
        DECOMPRESS_EXISTING_LONG = "--decompress-existing"
        args = ["setup"]
        if setup_path:
            args.extend([SETUP_PATH_LONG, setup_path])
        if precompute_path:
            args.extend([PRECOMPUTE_PATH_LONG, precompute_path])
        if overwrite:
            args.extend([OVERWRITE_LONG])
        if scale:
            args.extend([SCALE_LONG, str(scale)])
        if generate_setup:
            args.extend([GENERATE_SETUP_LONG])
        if generate_precompute:
            args.extend([GENERATE_PRECOMPUTE_LONG])
        if uncompressed:
            args.extend([UNCOMPRESSED_LONG])
        if compress_existing:
            args.extend([COMPRESS_EXISTING_LONG])
        if decompress_existing:
            args.extend([DECOMPRESS_EXISTING_LONG])
        self.process = subprocess.Popen(args=self.cmd(args))
        return self.wait_until_running()

    def stop(self) -> bool:
        if self.is_running():
            self.process.terminate()
        return self.is_running()

    def is_running(self):
        return self.process.poll() is None


class Client:
    def __init__(
        self,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        setup_path=DEFAULT_SETUP_PATH,
        precompute_path=DEFAULT_PRECOMPUTE_PATH,
        bin=DEFAULT_BIN,
        uncompressed=False,
    ):
        self.host = host
        self.port = port
        self.cli = CLI(bin=bin)
        self.setup_path = setup_path if os.path.exists(setup_path) else None
        self.precompute_path = (
            precompute_path if os.path.exists(precompute_path) else None
        )
        self.uncompressed = uncompressed

    def endpoint(self):
        return f"http://{self.host}:{self.port}"

    def start_rust(
        self,
        scale=None,
    ) -> bool:
        self.cli.run(
            host=self.host,
            port=self.port,
            setup_path=self.setup_path,
            precompute_path=self.precompute_path,
            uncompressed=self.uncompressed,
            scale=scale,
        )
        return self.cli.is_running()

    def stop_rust(self) -> bool:
        return self.cli.stop()

    def start(
        self,
        scale=None,
    ) -> bool:
        if not self.start_rust(scale=scale):
            return False
        if not self.ping().ok:
            print("Failed to ping Rust server.")
            return False
        print("Rust server is running.")

    def stop(self) -> bool:
        if not self.stop_rust():
            return False
        print("Rust server stopped.")

    # Ping the Rust server
    def ping(self) -> requests.Response:
        req = RPCRequest.ping()
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Commit to a polynomial
    def commit(self, poly: str) -> requests.Response:
        req = RPCRequest.commit(poly)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Open a commitment
    def open(self, poly: str, x: str) -> requests.Response:
        req = RPCRequest.open(poly, x)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Verify a proof
    def verify(self, proof: str, x: str, y: str, commitment: str) -> requests.Response:
        req = RPCRequest.verify(proof, x, y, commitment)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Generate a random polynomial
    def random_poly(self, degree: int) -> requests.Response:
        req = RPCRequest.random_poly(degree)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Generate a random point
    def random_point(self) -> requests.Response:
        req = RPCRequest.random_point()
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Evaluate a polynomial
    def eval(self, poly: str, x: str) -> requests.Response:
        req = RPCRequest.evaluate(poly, x)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Prove a polynomial
    # This is a combinatory method, and performs a commitment and an
    # opening on a randomly generated point.
    # This method provides convenience for miners and simplifies the code.
    def prove(self, poly: str) -> requests.Response:
        req = RPCRequest.prove(poly)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp


def commit(rpc, poly):
    with rpc.commit(poly) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("commitment")
    return None


def _open(rpc, poly, x):
    with rpc.open(poly, x) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("proof")
    return None


def verify(rpc, proof, x, y, commitment):
    with rpc.verify(proof, x, y, commitment) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("valid")
    return None


def random_poly(rpc, degree):
    with rpc.random_poly(degree) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("poly")
    return None


def random_point(rpc):
    with rpc.random_point() as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("point")
    return None


def eval_poly(rpc, poly, x):
    with rpc.eval(poly, x) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("y")
    return None


def prove(rpc, poly):
    with rpc.prove(poly) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return (
            data.get("result", {}).get("commitment"),
            data.get("result", {}).get("y"),
            data.get("result", {}).get("x"),
            data.get("result", {}).get("proof"),
        )
    return None


if __name__ == "__main__":
    os.environ["RUST_LOG"] = "debug"
    HOST = "localhost"
    PORT = 1337
    SETUP_PATH = "setup.decompressed"
    PRECOMPUTE_PATH = "precompute.decompressed"
    BIN = "target/release/fourier"
    setup_path = SETUP_PATH if os.path.exists(SETUP_PATH) else None

    rpc = Client(
        host=HOST,
        port=PORT,
        setup_path=SETUP_PATH,
        precompute_path=PRECOMPUTE_PATH,
        bin=BIN,
        uncompressed=True,
    )
    precompute_path = PRECOMPUTE_PATH if os.path.exists(PRECOMPUTE_PATH) else None
    rpc.start(scale=20)

    # Generate initial params
    f = random_poly(rpc, 5)
    x = random_point(rpc)
    y = eval_poly(rpc, f, x)
    print(f"Generated f: {f}")
    print(f"Generated x: {x}")
    print(f"Generated y: {y}")

    # Commit, open and verify
    commitment = commit(rpc, f)
    proof = _open(rpc, f, x)
    valid = verify(rpc, proof, x, y, commitment)
    assert valid
    print(f"Proof is valid: {valid}")

    rpc.stop()
