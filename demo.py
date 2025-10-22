import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt 

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.Models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.SimulationSampler import Simulator
    from stochprocsim.Models.TransitionModel import QuantumTransitionModel, ExactTransitionModel, TransitionModel
    from stochprocsim.CausalModels import Causal_Models

    return (
        Causal_Models,
        QuantumTransitionModel,
        RenewalProcess,
        Simulator,
        eval_diverge,
        get_uniform_renewal,
        np,
        plt,
        sp,
    )


@app.cell
def _(Causal_Models, sp):
    def is_unitary(U, tol=1e-7):
        U = sp.Matrix(U)
        prod = U * U.H
        # Check each element numerically
        for i in range(prod.shape[0]):
            for j in range(prod.shape[1]):
                val = complex(prod[i, j])
                if i == j:
                    if abs(val - 1) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
        return True

    for idx, model in Causal_Models.items():
        model.set_U_mat()
        print(f"Model {idx} unitary: {is_unitary(model.U)}")
    return


@app.cell
def _(Causal_Models, sp):
    # Reorder my unitaries to ximing style then sample kl's and check is same 
    from stochprocsim.CausalModels import reorder_matrix, reorder_states, CausalModel
    N = 3
    m = Causal_Models[N]
    m.set_U_mat()
    U_theo = sp.Matrix(m.U)

    m.set_U_optics()
    U_mathematica = sp.Matrix(m.U)

    tol = 1e-8
    diff = U_theo - U_mathematica
    diff.applyfunc(lambda x: 0 if abs(complex(x)) < tol else sp.N(x))
    return


@app.cell
def _(
    Causal_Models,
    QuantumTransitionModel,
    RenewalProcess,
    Simulator,
    eval_diverge,
    get_uniform_renewal,
    np,
):
    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        # Reset probabilities (emitting 1) after k steps starting from state 0 are:
        q_emit = exp_data ## EXPERIEMTNATL DATA GO HERE

        # probability of emitting 0 at each causal state:
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

        return RenewalProcess([1-q for q in q_survive_st[:-1]]) # the last state always emit 1

    def simulate_quantum_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states

        CS = Causal_Models[N]
        CS.set_U_mat()
        q_sim = Simulator(QuantumTransitionModel(CS))
        q_model = generate_quantum_model(np.array(q_sim.get_quantities()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        kl_div = eval_diverge(p_dist, q_dist, his_steps=N-1)

        return kl_div

    def simulate_quantum_optics_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        CS = Causal_Models[N]
        CS.set_U_optics()
        q_sim = Simulator(QuantumTransitionModel(CS))
        q_model = generate_quantum_model(np.array(q_sim.get_quantities()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        kl_div = eval_diverge(p_dist, q_dist, his_steps=N-1)

        return kl_div

    def simulate_classical_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        classical_bd = exact_model.classical_bd(4,target_dim=2)
        return classical_bd
    return (
        simulate_classical_proc,
        simulate_quantum_optics_proc,
        simulate_quantum_proc,
    )


@app.cell
def _(
    plt,
    simulate_classical_proc,
    simulate_quantum_optics_proc,
    simulate_quantum_proc,
):
    NMAX = 6
    NMIN = 3
    C_BUFF = 3
    kl_div = []

    x_quant = list(range(NMIN, NMAX + 1))
    x_class = list(range(NMIN, NMAX + C_BUFF))

    y_quantum   = [simulate_quantum_proc(n) for n in x_quant]
    y_qopt      = [simulate_quantum_optics_proc(n) for n in x_quant]
    y_classical = [simulate_classical_proc(n) for n in x_class]

    plt.plot(x_class, y_classical, '-s', label='Classical')  
    plt.plot(x_quant, y_quantum, '-o', label='Quantum (Theo)') 
    plt.plot(x_quant, y_qopt, '-p', label='Quantum (Exp)') 

    plt.xticks(x_class)
    plt.xlabel('N')
    plt.ylabel('KL Divergence')
    # plt.ylim(0,0.1)
    plt.title('Quantum vs Classical KL Divergence')
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
