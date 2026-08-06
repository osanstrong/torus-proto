import src.quartics.alg1010 as tt
import src.quartics.ferrari as fr
import src.quartics.numpyqs as nq


def print_case_compare(case:dict):
    coeffs = case['polynomial']
    fer_res = fr.FerrariSolver(coeffs)()
    ten_res = tt.Alg1010Solver(coeffs)()
    npq_res = nq.NumpySolver(coeffs)()
    print(f"numpy : {npq_res}")
    print(f"Ferrari:")
    print(f" - pyt: {fer_res}")
    print(f" - c++: {case['ferrari']}")
    print(f"Alg1010:")
    print(f" - pyt: {ten_res}")
    print(f" - c++: {case["alg1010"]}")
