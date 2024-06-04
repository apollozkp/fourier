import json
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
DEFAULT_UNCOMPRESSED = False


class RPCRequest:
    def __init__(self, method="ping", id=0, params=None):
        self.id = id
        self.method = method
        self.params = params
        self.jsonrpc = "2.0"

    def json(self):
        return json.dumps(self.__dict__)

    # UTILS
    @staticmethod
    def ping():
        return RPCRequest()

    @staticmethod
    def random_poly():
        return RPCRequest(method="randomPoly")

    @staticmethod
    def random_point():
        return RPCRequest(method="randomPoint")

    @staticmethod
    def evaluate(poly: List[str], x: str):
        params = {"poly": poly, "x": x}
        return RPCRequest(method="evaluate", params=params)

    @staticmethod
    def fft(poly: List[str], left: bool, inverse: bool):
        params = {"poly": poly, "left": left, "inverse": inverse}
        return RPCRequest(method="fft", params=params)

    # WORKER METHODS
    @staticmethod
    def worker_commit(i: int, poly: List[str]):
        params = {"i": i, "poly": poly}
        return RPCRequest(method="workerCommit", params=params)

    @staticmethod
    def worker_open(i: int, poly: List[str], x: str):
        params = {"i": i, "poly": poly, "x": x}
        return RPCRequest(method="workerOpen", params=params)

    @staticmethod
    def worker_verify(i: int, alpha: str, proof: str, eval: str, commitment: str):
        params = {
            "i": i,
            "alpha": alpha,
            "proof": proof,
            "eval": eval,
            "commitment": commitment,
        }
        return RPCRequest(method="workerVerify", params=params)

    # MASTER METHODS
    @staticmethod
    def master_commit(commitments: List[str]):
        params = {"commitments": commitments}
        return RPCRequest(method="masterCommit", params=params)

    @staticmethod
    def master_open(evals: List[str], proofs: List[str], beta: str):
        params = {"evals": evals, "proofs": proofs, "beta": beta}
        return RPCRequest(method="masterOpen", params=params)

    @staticmethod
    def master_verify(
        commitment: str, beta: str, alpha: str, z: str, pi_0: str, pi_1: str
    ):
        params = {
            "commitment": commitment,
            "beta": beta,
            "alpha": alpha,
            "z": z,
            "pi_0": pi_0,
            "pi_1": pi_1,
        }
        return RPCRequest(method="masterVerify", params=params)


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
        machines_scale=None,
        setup_path=None,
        precompute_path=None,
        uncompressed=None,
    ) -> bool:
        HOST_LONG = "--host"
        PORT_LONG = "--port"
        SCALE_LONG = "--scale"
        SETUP_PATH_LONG = "--setup-path"
        PRECOMPUTE_PATH_LONG = "--precompute-path"
        UNCOMPRESSED_LONG = "--uncompressed"
        MACHINES_SCALE_LONG = "--machines-scale"
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
        if machines_scale:
            args.extend([MACHINES_SCALE_LONG, str(machines_scale)])
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
        machines_scale=None,
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
        MACHINES_SCALE_LONG = "--machines-scale"
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
        if machines_scale:
            args.extend([MACHINES_SCALE_LONG, str(machines_scale)])
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
        uncompressed=DEFAULT_UNCOMPRESSED,
        bin=DEFAULT_BIN,
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
        machines_scale=None,
    ) -> bool:
        self.cli.run(
            host=self.host,
            port=self.port,
            setup_path=self.setup_path,
            precompute_path=self.precompute_path,
            scale=scale,
            machines_scale=machines_scale,
            uncompressed=self.uncompressed,
        )
        return self.cli.is_running()

    def stop_rust(self) -> bool:
        return self.cli.stop()

    def start(
        self,
        scale=None,
        machines_scale=None,
    ) -> bool:
        if not self.start_rust(scale=scale, machines_scale=machines_scale):
            return False
        if not self.ping().ok:
            print("Failed to ping Rust server.")
            return False
        print("Rust server is running.")

    def stop(self) -> bool:
        if not self.stop_rust():
            return False
        print("Rust server stopped.")

    def post(self, req: RPCRequest) -> requests.Response:
        return requests.post(self.endpoint(), data=req.json())

    # UTILS
    # Ping the Rust server
    def ping(self) -> requests.Response:
        req = RPCRequest.ping()
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Generate a random polynomial
    def random_poly(self) -> requests.Response:
        return self.post(RPCRequest.random_poly())

    # Generate a random point
    def random_point(self) -> requests.Response:
        return self.post(RPCRequest.random_point())

    # Evaluate a polynomial
    def eval(self, poly: str, x: str) -> requests.Response:
        return self.post(RPCRequest.evaluate(poly, x))

    # FFT a polynomial
    def fft(self, poly: str, left: bool, inverse: bool) -> requests.Response:
        return self.post(RPCRequest.fft(poly, left, inverse))

    # WORKER METHODS
    # Commit a polynomial as worker i
    def worker_commit(self, i: int, poly: str) -> requests.Response:
        return self.post(RPCRequest.worker_commit(i, poly))

    # Open a polynomial as worker i
    def worker_open(self, i: int, poly: str, x: str) -> requests.Response:
        return self.post(RPCRequest.worker_open(i, poly, x))

    # Verify a proof for worker i
    def worker_verify(
        self, i: int, proof: str, alpha: str, eval: str, commitment: str
    ) -> requests.Response:
        return self.post(RPCRequest.worker_verify(i, alpha, proof, eval, commitment))

    # MASTER METHODS
    # Make master commitment from a list of worker commitments
    def master_commit(self, commitments: List[str]) -> requests.Response:
        return self.post(RPCRequest.master_commit(commitments))

    # Make master opening from a list of worker openings
    def master_open(
        self, evals: List[str], proofs: List[str], beta: str
    ) -> requests.Response:
        return self.post(RPCRequest.master_open(evals, proofs, beta))

    # Verify a proof for the master
    def master_verify(
        self,
        commitment: str,
        beta: str,
        alpha: str,
        z: str,
        pi_0: str,
        pi_1: str,
    ) -> requests.Response:
        return self.post(
            RPCRequest.master_verify(commitment, beta, alpha, z, pi_0, pi_1)
        )

    # Prove a polynomial
    # This is a combinatory method, and performs a commitment and an
    # opening on a randomly generated point.
    # This method provides convenience for miners and simplifies the code.
    def prove(self, poly: str) -> requests.Response:
        req = RPCRequest.prove(poly)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp


# UTILS
def random_poly(rpc: Client):
    with rpc.random_poly() as resp:
        data = resp.json()
        print(f"Data: {data}")
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        poly = data.get("poly")
        print(f"Poly: {poly}")
        return poly

    return None


def random_point(rpc: Client):
    with rpc.random_point() as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        return data.get("point")

    return None


def eval_poly(rpc: Client, poly: str, x: str):
    with rpc.eval(poly, x) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        return data.get("eval")

    return None


def fft(rpc: Client, poly: str, left: bool, inverse: bool):
    with rpc.fft(poly, left, inverse) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        return data.get("poly")

    return None


# WORKER METHODS
def worker_commit(rpc: Client, i: int, poly: str):
    with rpc.worker_commit(i, poly) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        return data.get("commitment")
    return None


def worker_open(rpc: Client, i: int, poly: str, x: str):
    with rpc.worker_open(i, poly, x) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("eval"), data.get("proof")
    return None


def worker_verify(
    rpc: Client, i: int, proof: str, alpha: str, eval: str, commitment: str
):
    with rpc.worker_verify(i, proof, alpha, eval, commitment) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        return data.get("valid")

    return None


# Convenience method for miners
def worker_commit_and_open(rpc: Client, i: int, poly: str, alpha: str):
    commitment = worker_commit(rpc, i, poly)
    eval, proof = worker_open(rpc, i, poly, alpha)
    return commitment, eval, proof


# MASTER METHODS
def master_commit(rpc: Client, commitments: List[str]):
    with rpc.master_commit(commitments) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")

        return data.get("commitment")

    return None


def master_open(rpc: Client, evals: List[str], proofs: List[str], beta: str):
    with rpc.master_open(evals, proofs, beta) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("z"), data.get("pi_0"), data.get("pi_1")

    return None


def master_verify(
    rpc: Client, commitment: str, beta: str, alpha: str, z: str, pi_0: str, pi_1: str
):
    with rpc.master_verify(commitment, beta, alpha, z, pi_0, pi_1) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("valid")
    return None


def test_routine(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    setup_path: str = "test_setup",
    precompute_path: str = "test_precompute",
    uncompressed: bool = True,
    bin: str = "target/release/fourier",
    scale: int = 6,
    machines_scale: int = 2,
):
    rpc = Client(
        host=host,
        port=port,
        bin=bin,
        setup_path=setup_path,
        precompute_path=precompute_path,
        uncompressed=uncompressed,
    )
    setup_path = setup_path if os.path.exists(setup_path) else None
    precompute_path = precompute_path if os.path.exists(precompute_path) else None

    n_workers = 2**machines_scale
    try:
        rpc.start(scale=scale, machines_scale=machines_scale)

        # Generate initial params
        f, alpha, beta = random_poly(rpc), random_point(rpc), random_point(rpc)
        worker_polys = [
            fft(rpc, f[i], left=True, inverse=True) for i in range(n_workers)
        ]

        # Commit and open for all workers
        commitments = []
        evals = []
        proofs = []
        for i in range(n_workers):
            print(f"Querying worker {i}")
            commitment, eval, proof = worker_commit_and_open(
                rpc, i, worker_polys[i], alpha
            )
            valid = worker_verify(rpc, i, proof, alpha, eval, commitment)
            assert valid
            print(f"Worker {i} submitted valid proof.")

            commitments.append(commitment)
            evals.append(eval)
            proofs.append(proof)

        # Commit and open for the master
        print("Assembling master proof.")
        master_commitment = master_commit(rpc, commitments)
        z, pi_0, pi_1 = master_open(rpc, evals, proofs, beta)
        valid = master_verify(rpc, master_commitment, beta, alpha, z, pi_0, pi_1)
        assert valid
        print("Proof is valid.")
    except Exception:
        rpc.stop()
        raise


if __name__ == "__main__":
    import os

    os.environ["RUST_LOG"] = "debug"
    HOST = "localhost"
    PORT = 1337
    SETUP_PATH = "test_setup"
    PRECOMPUTE_PATH = "test_precompute"
    UNCOMPRESSED = True
    BIN = "target/release/fourier"

    test_routine(
        host=HOST,
        port=PORT,
        setup_path=SETUP_PATH,
        precompute_path=PRECOMPUTE_PATH,
        uncompressed=UNCOMPRESSED,
        bin=BIN,
    )
