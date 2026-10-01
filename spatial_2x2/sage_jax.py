"""CPU re-implementation of scripts/06_train_gnn.py GraphSAGE (PyG SAGEConv: mean aggregation over
neighbours incl. self-loop + root weight; 2 layers, hidden 64, ReLU, dropout 0.3, Adam 1e-3,
200 full-batch epochs; features and target standardised on training nodes only)."""
import numpy as np, jax, jax.numpy as jnp, optax
from sklearn.preprocessing import StandardScaler

def _lin(key, fin, fout, bias=True):
    k1, k2 = jax.random.split(key); b = 1.0 / np.sqrt(fin)
    W = jax.random.uniform(k1, (fin, fout), minval=-b, maxval=b)
    return W, (jax.random.uniform(k2, (fout,), minval=-b, maxval=b) if bias else None)

def _init(key, fin, hid):
    ks = jax.random.split(key, 4)
    Wl1, bl1 = _lin(ks[0], fin, hid); Wr1, _ = _lin(ks[1], fin, hid, False)
    Wl2, bl2 = _lin(ks[2], hid, 1);   Wr2, _ = _lin(ks[3], hid, 1, False)
    return dict(Wl1=Wl1, bl1=bl1, Wr1=Wr1, Wl2=Wl2, bl2=bl2, Wr2=Wr2)

def _fwd(p, x, src, dst, deg, key, train, dropout):
    n = x.shape[0]
    agg = lambda h: jax.ops.segment_sum(h[src], dst, n) / deg
    h = jax.nn.relu(agg(x) @ p["Wl1"] + p["bl1"] + x @ p["Wr1"])
    if train:
        keep = jax.random.bernoulli(key, 1 - dropout, h.shape)
        h = jnp.where(keep, h / (1 - dropout), 0.0)
    return (agg(h) @ p["Wl2"] + p["bl2"] + h @ p["Wr2"]).ravel()

@jax.jit
def _step(params, st, x, y, trmask, src, dst, deg, key):
    def loss_fn(p):
        out = _fwd(p, x, src, dst, deg, key, True, 0.3)
        return jnp.sum(jnp.where(trmask, (out - y) ** 2, 0.0)) / jnp.sum(trmask)
    l, gr = jax.value_and_grad(loss_fn)(params)
    upd, st = _OPT.update(gr, st, params)
    return optax.apply_updates(params, upd), st, l

_OPT = optax.adam(1e-3)
_pred = jax.jit(lambda p, x, src, dst, deg: _fwd(p, x, src, dst, deg, None, False, 0.3))

def run_sage(X, y, edge_index, train_idx, test_idx, seed=42, hid=64, epochs=200):
    n = X.shape[0]; sl = np.arange(n)
    src = jnp.array(np.concatenate([edge_index[0], sl])); dst = jnp.array(np.concatenate([edge_index[1], sl]))
    deg = jnp.maximum(jax.ops.segment_sum(jnp.ones(src.shape[0]), dst, n), 1.0)[:, None]
    xs = StandardScaler().fit(X[train_idx]); ys = StandardScaler().fit(y[train_idx, None])
    x = jnp.array(xs.transform(X), dtype=jnp.float32)
    yt = jnp.array(ys.transform(y[:, None]).ravel(), dtype=jnp.float32)
    trmask = jnp.zeros(n, bool).at[train_idx].set(True)
    key = jax.random.PRNGKey(seed); key, k0 = jax.random.split(key)
    params = _init(k0, X.shape[1], hid); st = _OPT.init(params)
    for _ in range(epochs):
        key, k = jax.random.split(key)
        params, st, l = _step(params, st, x, yt, trmask, src, dst, deg, k)
    pred = np.asarray(_pred(params, x, src, dst, deg))
    return ys.inverse_transform(pred[:, None]).ravel()[test_idx], float(l)
