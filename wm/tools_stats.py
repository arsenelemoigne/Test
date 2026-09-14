"""La loi de Student, sans scipy. Extraite de tools/valeur.py pour etre importable
depuis le paquet : l'approximation normale est fausse aux petits n, et dans le
mauvais sens - a rho = +0,81 sur n = 10 elle rend p < 0,001 la ou Student rend
p = 0,005."""
import math


def _betacf(a, b, x, iters=200):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-12:
            break
    return h


def p_student(t: float, df: int) -> float:
    """p bilaterale d'un t de Student a df degres de liberte."""
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lb) * _betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lb) * \
            _betacf(b, a, 1 - x) / b
    return min(1.0, max(0.0, ib))
