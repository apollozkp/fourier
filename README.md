<div align="center">

# **Apollo ZK library**
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1222672871092912262)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![example workflow](https://github.com/apollozkp/fourier/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/apollozkp/fourier/badge.svg?branch=main)](https://coveralls.io/github/apollozkp/fourier?branch=main)

[Website](https://apollozkp.com)

</div>

# To use

1. Install from git
```bash
pip install git+http://github.com/apollozkp/fourier.git
```
2. Import client
```python
from fourier import Client
```
3. Start client
```python
client = Client(port=1337)
```

## API

### Start
Start the Rust RPC server and ping it to check if it is running.
```python
client.start()
```

### Stop
Stop the Rust RPC server.
```python
client.stop()
```

### Ping
Ping the Rust RPC server.
```python
client.ping()
# {'jsonrpc': '2.0', 'result': 'pong', 'id': 1}
```

### Commit
Compute a commitment to a polynomial.
```python
# base64 encoded coefficients of polynomial
poly = ["AQB...", ..., "AQB..."]
client.commit(poly=poly)
# {'jsonrpc': '2.0', 'result': {'commitment': '123...efg'}, 'id': 1}
```

### Open
Open a polynomial at a point.
```python
# base64 encoded coefficients of polynomial
poly = ["AQB...", ..., "AQB..."]
# base64 encoded point
x = "AQB..."
client.open(poly=poly, x=x)
# {'jsonrpc': '2.0', 'result': {'proof': '123...efg'}, 'id': 1}
```

### Verify
Verify a proof of polynomial at a point.
```python
# base64 encoded proof
proof = "AQB..."
# base64 encoded point
x = "AQB..."
# base64 encoded value of polynomial at x
y = "AQB..."
# base64 encoded commitment
commitment = "AQB..."
client.verify(proof=proof, x=x, y=y, commitment=commitment)
# {'jsonrpc': '2.0', {'valid': True}, 'id': 1}
```

### RandomPoly
Generate a random polynomial.
```python
# degree of polynomial
degree = 10
client.random_poly(degree=degree)
# {'jsonrpc': '2.0', 'result': {'poly': ['123...efg', ..., '123...efg']}, 'id': 1}
```

