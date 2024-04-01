import json
import subprocess
import time

import requests

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1337
DEFAULT_BIN = "target/debug/fourier"


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


class Client:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.rust_rpc = None

    def endpoint(self):
        return f"http://{self.host}:{self.port}"

    def rust_cmd(self):
        return [f"./{DEFAULT_BIN}", "--host", self.host, "--port", str(self.port)]

    def start_rust(self) -> bool:
        if self.rust_rpc is not None:
            print("Rust server is already running.")
            return False

        self.rust_rpc = subprocess.Popen(args=self.rust_cmd())

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


def commit(rpc, poly):
    with rpc.commit(poly) as resp:
        data = resp.json()
        if data.get("error"):
            print(f"Error: {data.get('error')}")
        return data.get("result", {}).get("commitment")
    return None


def open(rpc, poly, x):
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


def test_pipeline(rpc, poly, x, y, expected_commitment=None, expected_proof=None):
    commitment = commit(rpc, poly)
    if not commitment:
        print("Failed to commit to polynomial.")
        return
    if expected_commitment and commitment != expected_commitment:
        print(
            f"Commitment mismatch. Expected: {expected_commitment}, Got: {commitment}"
        )
    else:
        print(f"Commitment: {commitment}")

    proof = open(rpc, poly, x)
    if not proof:
        print("Failed to open commitment.")
        return
    if expected_proof and proof != expected_proof:
        print(f"Proof mismatch. Expected: {expected_proof}, Got: {proof}")
    else:
        print(f"Proof: {proof}")

    valid = verify(rpc, proof, x, y, commitment)
    if not valid:
        print("Failed to verify proof.")
        return
    print(f"Verification: {valid}")


if __name__ == "__main__":
    HOST = "localhost"
    PORT = 1338
    rpc = Client(host=HOST, port=PORT)
    rpc.start()
    TEST_POLY = [
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

    TEST_POINT = "456006fff56412d329d527901d02877a581a89cfa677ca963eb9d680766234cc"
    TEST_EVAL = "29732a1e0e074ab05ee6a9e57794c5ad1965b98b6c8c6ecde96ac776ea06ff5b"

    # compressed hex
    TEST_COMMITMENT = "8424fb9dc224ab79efccf6710edea3b936d03bbd323f052bb9c4b2efe9f98239e7c3e48148f243065cee910054a10e71"
    TEST_PROOF = "895cdfe1bf26bbf10bdc0d90178ec89635269cca7c9b39836a76e91689ad3fa4d1772f8d60cdd86cd4bfd1dedbdec81d"

    test_pipeline(
        rpc,
        TEST_POLY,
        TEST_POINT,
        TEST_EVAL,
        expected_commitment=TEST_COMMITMENT,
        expected_proof=TEST_PROOF,
    )

    rpc.stop()
