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
# hex encoded coefficients of polynomial
poly = ["123...efg", ..., "123...efg"]
client.commit(poly=poly)
# {'jsonrpc': '2.0', 'result': {'commitment': '123...efg'}, 'id': 1}
```

### Open
Open a polynomial at a point.
```python
# hex encoded coefficients of polynomial
poly = ["123...efg", ..., "123...efg"]
# hex encoded point
x = "123...efg"
client.open(poly=poly, x=x)
# {'jsonrpc': '2.0', 'result': {'proof': '123...efg'}, 'id': 1}
```

### RandomPoly
Generate a random polynomial.
```python
# degree of polynomial
degree = 10
client.random_poly(degree=degree)
# {'jsonrpc': '2.0', 'result': {'poly': ['123...efg', ..., '123...efg']}, 'id': 1}
```

### Verify
Verify a proof of polynomial at a point.
```python
# hex encoded proof
proof = "123...efg"
# hex encoded point
x = "123...efg"
# hex encoded value of polynomial at x
y = "123...efg"
# hex encoded commitment
commitment = "123...efg"
client.verify(proof=proof, x=x, y=y, commitment=commitment)
# {'jsonrpc': '2.0', {'valid': True}, 'id': 1}
```

