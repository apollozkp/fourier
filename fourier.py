import json
import os
import subprocess
import time

import requests

ADDRESS = "127.0.0.1"
RECV_PORT = 1337

RUST_BIN = "target/debug/fourier"


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
        params = {"p": poly}
        return RPCRequest(method="commit", params=params)

    @staticmethod
    def proof(poly, x, y):
        params = {"p": poly, "x": x, "y": y}
        return RPCRequest(method="prove", params=params)

    @staticmethod
    def verify(poly, x, y, proof):
        params = {"p": poly, "x": x, "y": y, "proof": proof}
        return RPCRequest(method="verify", params=params)


class Client:
    def __init__(self, address=ADDRESS, port=RECV_PORT):
        self.address = address
        self.port = port
        self.rust_rpc = None

    def endpoint(self):
        return f"http://{self.address}:{self.port}"

    def rpc_cmd(self):
        return f"./{RUST_BIN} {self.address} {self.port}"

    def start_rust(self) -> bool:
        if self.rust_rpc is not None:
            print("Rust server is already running.")
            return False

        cmd = f"./{RUST_BIN}"
        self.rust_rpc = subprocess.Popen(cmd)

        if self.rust_rpc is None:
            print("Failed to start Rust server.")
            return False
        print("waiting for Rust server to start...")
        time.sleep(3)
        return True

    def stop_rust(self) -> bool:
        if self.rust_rpc is None:
            print("No Rust server to stop.")
            return False
        self.rust_rpc.terminate()
        self.rust_rpc = None
        return True

    def start(self):
        if not self.start_rust():
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

    # Prove a commitment
    def prove(self, poly: str, x: str, y: str) -> requests.Response:
        req = RPCRequest.proof(poly, x, y)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp

    # Verify a proof
    def verify(self, poly: str, x: str, y: str, proof: str) -> requests.Response:
        req = RPCRequest.verify(poly, x, y, proof)
        resp = requests.post(self.endpoint(), data=req.json())
        return resp


if __name__ == "__main__":
    rpc = Client()
    rpc.start()
    with rpc.commit("1 2 3 4") as resp:
        print(resp.text)
    with rpc.prove("1 2 3 4", "1", "10") as resp:
        print(resp.text)
    with rpc.verify("1 2 3 4", "1", "10", "1 2 3") as resp:
        print(resp.text)
    rpc.stop()
