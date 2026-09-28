import numpy as np
from rnn import RNN, one_hot

def numerical_grad(net, name, xs, targets, eps=1e-5):
    W = net.get(name)
    num = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = W[idx]
        W[idx] = original + eps
        L_plus = net.loss(net.forward(xs), targets)
        W[idx] = original - eps
        L_minus = net.loss(net.forward(xs), targets)
        W[idx] = original
        num[idx] = (L_plus - L_minus) / (2 * eps)
        it.iternext()
    return num

def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12))

def run_check(n_in=4, n_hidden=5, n_out=3, T=7, seed=1, sparse_targets=False):
    rng = np.random.default_rng(seed)
    net = RNN(n_in, n_hidden, n_out, seed=seed, scale=0.5)
    xs = [one_hot(int(rng.integers(n_in)), n_in) for _ in range(T)]
    if sparse_targets:
        targets = [None] * (T - 1) + [int(rng.integers(n_out))]
    else:
        targets = [int(rng.integers(n_out)) for _ in range(T)]
    _, analytic, _ = net.loss_and_grads(xs, targets)
    mode = "ответ только в конце" if sparse_targets else "ответ на каждом шаге"
    print(f"\n=== Проверка градиента: {mode}, T={T} ===")
    worst = 0.0
    for name in RNN.PARAM_NAMES:
        num = numerical_grad(net, name, xs, targets)
        err = relative_error(num, analytic[name])
        worst = max(worst, err)
        verdict = "OK " if err < 1e-7 else "ПЛОХО"
        print(f"  {name:4s}  отн. ошибка = {err:.3e}   {verdict}")
    return worst
  
if __name__ == "__main__":
    worst = 0.0
    worst = max(worst, run_check(sparse_targets=False))
    worst = max(worst, run_check(sparse_targets=True))
    worst = max(worst, run_check(T=25, sparse_targets=True, seed=7))
    print(f"\nХудшая относительная ошибка по всем тестам: {worst:.3e}")
    if worst < 1e-7:
        print("ВЫВОД BPTT ПОДТВЕРЖДЁН ЧИСЛЕННО.")
    else:
        print("Где-то ошибка в backward(). Ищи.")
