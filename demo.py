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
    from stochprocsim.Models.TransitionModel import QuantumTransitionModel, ExactTransitionModel, print_model_probabilities
    from stochprocsim.CausalModels import Causal_Models
    return (
        Causal_Models,
        ExactTransitionModel,
        QuantumTransitionModel,
        RenewalProcess,
        Simulator,
        eval_diverge,
        get_uniform_renewal,
        np,
        plt,
        sp,
    )


@app.cell(hide_code=True)
def _(Causal_Models, sp):
    N = 3
    CS = Causal_Models[N]
    # S0 
    print(CS[1])

    # Calculate s1 
    s1 = CS.U@CS[0]
    # s1.simplify()
    # print(s1)
    v13 = sp.Matrix([[s1[0]],[s1[2]]])
    v24 = sp.Matrix([[s1[1]],[s1[3]]])

    α, β = v13.norm(), v24.norm()
    u13, u24 = v13/α, v24/β
    α_val, β_val = α**2, β**2
    ip1 = sp.Abs((u13.conjugate().T @ sp.Matrix([[CS[0+1][0]],[CS[0+1][2]]]))[0])
    ip2 = sp.Abs((u24.conjugate().T @ sp.Matrix([[CS.s0[0]],[CS.s0[2]]]))[0]) 
    print(f'ip1: {round(float(sp.N(ip1)),2)}')
    print(f'ip2: {round(float(sp.N(ip2)),2)}')

    # print_model_probabilities()
    return


@app.cell(hide_code=True)
def _(Causal_Models, ExactTransitionModel, QuantumTransitionModel, Simulator):
    # Now lets compare probabilities between Pure Quantum and Optics

    mod2 = Causal_Models[3]

    sim2 = Simulator(transition_model=QuantumTransitionModel(mod2))

    # We are sampling both unitaries the SAME way
    # print(f'Pure Quantum:')
    mod2.set_U_mat()
    n = sim2.get_quantities()
    print(n)
    sim2.get_probabilities()

    print('\n')
    e = Simulator(transition_model=ExactTransitionModel(mod2))
    e.get_probabilities()
    return


@app.cell(hide_code=True)
def _(RenewalProcess, np):
    def get_output_string_quantitites(model, n):
        probs = []
        for i in range(n):
            str = ''
            for j in range(i):
                str += '0'
            str += '1'
            probs.append(model.prob_seq(str, past='1'))
        return probs

    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        # Reset probabilities (emitting 1) after k steps starting from state 0 are:
        q_emit = exp_data ## EXPERIEMTNATL DATA GO HERE

        # probability of emitting 0 at each causal state:
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

        return RenewalProcess([1-q for q in q_survive_st[:-1]]) # the last state always emit 1
    return (generate_quantum_model,)


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    eval_diverge,
    generate_quantum_model,
    get_uniform_renewal,
    np,
):
    def simulate_quantum_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        m = 100
        n = 10000
        CS = Causal_Models[N]

        CS.set_U_mat()
        # print(f'N{N} U: {CS.U}')
        q_sim = Simulator(QuantumTransitionModel(CS), m, n)
        q_model = generate_quantum_model(np.array(q_sim.get_quantities()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        kl_div = eval_diverge(p_dist, q_dist, his_steps=N-1)

        return kl_div

    def simulate_classical_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        classical_bd = exact_model.classical_bd(4,target_dim=2)
        return classical_bd

    def simulate_quantum_optics_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        m = 100
        n = 10000
        CS = Causal_Models[N]

        CS.set_U_optics()
        # print(f'N{N} U OPT: {CS.U}')
        q_sim_opt = Simulator(QuantumTransitionModel(CS), m, n)
        q_model_opt = generate_quantum_model(np.array(q_sim_opt.get_quantities()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist_opt = q_model_opt.gen_dists(N)[0]
        kl_div_opt = eval_diverge(p_dist, q_dist_opt, his_steps=N-1)

        return kl_div_opt
    return simulate_classical_proc, simulate_quantum_proc


@app.cell(hide_code=True)
def _(plt, simulate_classical_proc, simulate_quantum_proc):
    NMAX = 6
    NMIN = 3
    C_BUFF = 5
    kl_div = []

    x_quant = list(range(NMIN, NMAX + 1))
    x_class = list(range(NMIN, NMAX + C_BUFF))

    y_quantum   = [simulate_quantum_proc(n) for n in x_quant]
    # y_qopt      = [simulate_quantum_optics_proc(n) for n in x_quant]
    y_classical = [simulate_classical_proc(n) for n in x_class]

    plt.plot(x_class, y_classical, '-s', label='Classical')  
    plt.plot(x_quant, y_quantum, '-o', label='Quantum') 
    # plt.plot(x_quant, y_qopt, '-p', label='Quantum (Optics)') 

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
