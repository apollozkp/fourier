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
        params = {"poly": poly}
        return RPCRequest(method="commit", params=params)

    @staticmethod
    def proof(poly, x, y):
        params = {"poly": poly, "x": x, "y": y}
        return RPCRequest(method="prove", params=params)

    @staticmethod
    def verify(poly, x, y, proof):
        params = {"poly": poly, "x": x, "y": y, "proof": proof}
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
    poly = [
        "6945DC5C4FF4DAC8A7278C9B8F0D4613320CF87FF947F21AC9BF42327EC19448",
        "68E40C088D827BCCE02CEF34BDC8C12BB025FBEA047BC6C00C0C8C5C925B7FAF",
        "67281FAC164E9348B80693BA30D5D4E311DE5878EB3D20E34A58507B484B243C",
        "5F7C377DAE6B9D9ABAD75DC15E4FFF9FE7520D1F85224C95F485F44978154C5A",
        "2D85C376A440B6E25C3F7C11559B6A27684023F36C3D7A0ACD7E7D019DE399C7",
        "4A6FB95F0241B3583771E799120C87AAE3C843ECDB50A38254A92E198968922F",
        "1005079F96EC412A719FE2E9FA67D421D98FB4DEC4181459E59430F5D502BD2A",
        "64960B8692062DCB01C0FFBAC569478A89AD880ED3C9DF710BED5CE75F484693",
        "03C2882155A447642BD21FB1CF2553F80955713F09BBBBD9724E2CBFD8B19D41",
        "0AB07FECB59EE3435F6129FCD602CB519E56D7B426941633E37A3B676A24830F",
        "12FA5861459EFFBAE654827D98BFDFEA5545DDF8BB9628579463DA21F17462B5",
        "6A6296A0376D807530DB09DC8BB069FFDEC3D7541497B82C722A199D6B7C5B06",
        "153D2C81B54D7E1C3E83EA61C7F66FD88155F1713EE581E2BE8438CA9FEE1A02",
        "216BCCC4AE97FE3E1D4B21C375C46140FA153E7868201A43480889047ACD0C2D",
        "381BD4FE924EB10E08F2A227D3DB2083AA0E5A1F661CD3C702C4B8A9385E7839",
        "723A7640FD7E65473131563AB5514916AC861C2695CE6513E5061E597E5E1A81",
    ]
    with rpc.commit(poly) as resp:
        print(resp.text)
    with rpc.prove("1 2 3 4", "1", "10") as resp:
        print(resp.text)
    with rpc.verify("1 2 3 4", "1", "10", "1 2 3") as resp:
        print(resp.text)
    rpc.stop()
