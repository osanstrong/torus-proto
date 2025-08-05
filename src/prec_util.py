from mpmath import mp
from mpmath.ctx_mp_python import _mpf
from mpmath.libmp.libmpf import round_fast, normalize

# =-------=
# Constants
# =-------=

SINGLE_PREC = 23
DOUBLE_PREC = 53
QUAD_PREC = 113
OCT_PREC = 237
CONST_PREC = OCT_PREC

type MpfAble = str|float|int|_mpf

# =-------=
# Functions
# =-------=


def mp_const(val: MpfAble, reduced_prec: int = None):
    '''
    Returns a custom mpmath constant (in the mp context) of the given mpf-convertable value, up to octuple precision. (E.g. float, string like "1/3").
    Evaluates the value at octuple precision, and returns an mpmath constant function, which returns a conversion of this value to the current working precision.

    Parameters
    ----------
    val : MpfAble
        A value, such as an mpf instance, a float, or a mathematically interpretable string ("0.1", "1/3"), to be tracked with up to 237 bits of precision.
    reduced_prec : int, default None
        If specified, evaluates
    '''
    if (not reduced_prec is None) and (not reduced_prec < CONST_PREC): 
        raise ValueError(f"Reduced precision of {reduced_prec} is higher than stored precision of {CONST_PREC}")
    prev = mp.prec
    if not reduced_prec is None:
        eval_prec = reduced_prec
    else:
        eval_prec = CONST_PREC
    mp.prec = eval_prec
    hpval = mp.mpf(val)
    hpv_comps = hpval._mpf_
    mp.prec = prev
    func = lambda prec, rnd=round_fast: normalize(hpv_comps[0], hpv_comps[1], hpv_comps[2], hpv_comps[3], prec, rnd)
    return mp.constant(func, "custom_const")


CONST_TYPE = type(mp_const(1))
