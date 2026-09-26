from ipaddress import _RawNetworkPart

import numpy as np

def one_hot(i, n):
    v = np.zeros((n,1))
    v[i,0] = 1.0
    return v

def softmax(o):
    e = np.exp(o - np.max(o))
    return e / np.sum(e)

class RNN:
    PARAM_NAMES = ("Wxh", "Whh", "why", "bh", "by")

    def __init__(self, n_in, n_hidden, n_out, seed=0, scale=0.1):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.Wxh = rng.normal(0.0, scale, (n_hidden, n_in))
        self.Whh = rng.normal(0.0, scale, (n_hidden, n_hidden))
        self.Why = rng.normal(0.0, scale, (n_out, n_hidden))
        self.bh = np.zeros((n_hidden, 1))
        self.by = np.zeros((n_out, 1))

        self._cache = {n: np.zeros_like(getattr(self, n)) for n in self.PARAM_NAMES}

    def get(self, name):
        return getattr(self, name)

    def forward(self, xs, h_init=None):
        T = len(xs)
        hs = {-1: np.zeros(self.n_hidden, 1) if h_init is None else h_init.copy()}
        a_s,os_, ps = {},{},{}

        for t in range(T):
            a_s = self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh
            hs = np.tanh(a_s[t])
            os_[t] = self.Why @ hs[t] + self.by
            ps[t] = softmax(os_[t])

        return {"xs": xs, "hs": hs, "a": a_s, "o": os_, "ps": ps, "T": T}

    @staticmethod
    def loss(cache, targets):
        L= 0.0
        for t,y in enumerate(targets):
            if y is None:
                continue
            L -= np.log(cache["ps"][t][y,0])
        return L

    def backward(self, cache, targets):
        xs,hs,ps,T = cache["xs"], cache["hs"], cache["ps"], cache["T"]
        grads = {n: np.zeros_like(getattr(self,n)) for n in self.PARAM_NAMES}
        dh_next = np.zeros((self.n_hidden, 1))

        for t in reversed(range(T)):
            y = targets[t]
            if y is None:
                do = np.zeros((self.n_out,1))
            else:
                do = ps[t].copy()
                do[y,0] -= 1.0

            grads["Why"] += do @ hs[t].T
            grads["by"] += do
            dh = self.Why.T @ do + dh_next
            da = (1.0 - hs[t] **2)*dh
            grads["Wxh"] += da @ xs[t].T
            grads["Whh"] += da @ hs[t-1].T
            grads["bh"] += da

            dh_next = self.Whh.T @ da
        return grads

    def loss_and_grads(self, xs, targets, h_init=None):
        cache = self.forward(xs, h_init)
        return self.loss(cache, targets), self.backward(cache, targets), cache

    @staticmethod
    def clip(grads, limit=5.0):
        for g in grads.values():
            np.clip(g, -limit, limit, out=g)
        return grads

    def step(self, grads, lr=0.1, eps=1e-8):
        for n in self.PARAM_NAMES:
            g = grads[n]
            self._cache[n] += g * g
            getattr(self, n)[...] -= lr * g / (np.sqrt(self._cache[n]) + eps)
