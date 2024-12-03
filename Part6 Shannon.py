import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

EPS = 10 ** -8  # Numerical threshold
dispersal = 10 ** -6  # Species dispersal rate


def run(S=20, cp=100., days=15, pH0=7, kdeath=-0.1, kgrowth=10.,
        acid_lover_index=0, acid_lover_initial_amount=0.1, acid_lover_kgrowth=15.0,
        alpha=1000, beta=0, movement_factor=0.4, **kwargs):
    ppref = kwargs.get('ppref', np.random.normal(pH0, 1, S))
    ppref[acid_lover_index] = 3.5

    cs = np.random.uniform(-1, 1, S) * cp  # Ensuring all cs are negative
    cs[acid_lover_index] = -np.abs(np.random.uniform(2, 4) * cp)

    initial_populations = np.random.random(S) * 0.05
    initial_populations[acid_lover_index] = acid_lover_initial_amount
    x0 = kwargs.get('x0', np.concatenate([[pH0], np.log(initial_populations)]))

    def fun(p):
        ingate = np.zeros(S)
        for i in range(S):
            if i != acid_lover_index:
                low_bound = np.random.normal(3.65, 0.5)
                high_bound = low_bound + 1.5
                ingate[i] = float(low_bound <= p <= high_bound)
            else:
                if i == acid_lover_index:
                    ingate[acid_lover_index] = float(3.5 <= p <= 4.5)
        return ingate

    def eqs(t, x):
        p = x[0]
        x = np.exp(np.clip(x[1:], np.log(EPS), 0))
        dx = np.zeros_like(x)

        for i in range(S):
            growth_rate = kgrowth
            if i == acid_lover_index:
                growth_rate = acid_lover_kgrowth

            competition_term = 0
            for j in range(S):
                if i != j:
                    competition_term += (alpha if j == acid_lover_index else beta) * x[j]

            dx[i] = x[i] * (growth_rate * fun(p)[i] * (1 - x[i] - competition_term))

        for i in range(S):
            if i != acid_lover_index:
                movement_change = 1 + movement_factor * (np.random.choice([-1, 1]))
            if i == acid_lover_index:
                movement_change = 1
            x[i] *= movement_change

        if kdeath < 0:
            dx += np.log(-kdeath) * x
        dx += dispersal
        dp = np.sum(cs * x)
        if kdeath < 0:
            dp += np.log(-kdeath) * (p - pH0)

            pKa = 3.0
            buffer_concentration = 500
            buffer_effect = buffer_concentration / (1 + np.power(10, np.clip(p - pKa, -100, 100)))

            dp -= buffer_effect * (p - pH0)

            p_updated = p + dp
            p_clipped = np.clip(p_updated, 2, None)
            dp = p_clipped - p

        return np.concatenate([[dp], dx / x])

    time = []
    pH = []
    Ns = []
    shannon_indices = []
    for day in range(days):
        t_eval = np.linspace(0, 1., 100)  # Specify 100 time points for each day
        sol = solve_ivp(fun=eqs, t_span=(0, 1.), y0=x0, t_eval=t_eval)
        pH.append(sol.y[0])
        populations = np.exp(sol.y[1:].T)
        Ns.append(populations)

        # Calculate Shannon Diversity Index
        otu_data_normalized = populations / populations.sum(axis=1, keepdims=True)
        shannon_index = -np.sum(otu_data_normalized * np.log(otu_data_normalized + EPS), axis=1)
        shannon_indices.append(shannon_index)

        time.append(sol.t + day)
        x0 = sol.y[:, -1].copy()

    pH = np.concatenate(pH)
    Ns = np.concatenate(Ns, axis=0)
    shannon_indices = np.concatenate(shannon_indices)
    time = np.concatenate(time)

    return time[:-1], pH[:-1], Ns[:-1], shannon_indices[:-1]


def single_run(args):
    S, cp, days, pH0, kdeath, kgrowth, acid_lover_index, acid_lover_initial_amount, acid_lover_kgrowth, alpha, beta, movement_factor = args
    return run(S=S, cp=cp, days=days, pH0=pH0, kdeath=kdeath, kgrowth=kgrowth,
               acid_lover_index=acid_lover_index, acid_lover_initial_amount=acid_lover_initial_amount,
               acid_lover_kgrowth=acid_lover_kgrowth, alpha=alpha, beta=beta, movement_factor=movement_factor)


def bootstrap_simulations(S, cp, days, pH0, kdeath, kgrowth, acid_lover_index, acid_lover_initial_amount,
                          acid_lover_kgrowth, alpha, beta, movement_factor, bootstrap_runs=20):
    pH_all = []
    Ns_all = []
    shannon_all = []
    min_length = np.inf

    args = (S, cp, days, pH0, kdeath, kgrowth, acid_lover_index, acid_lover_initial_amount,
            acid_lover_kgrowth, alpha, beta, movement_factor)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(single_run, [args] * bootstrap_runs)

    for idx, result in enumerate(results):
        time, pH, Ns, shannon_indices = result
        min_length = min(min_length, len(pH))
        pH_all.append(pH)
        Ns_all.append(Ns)
        shannon_all.append(shannon_indices)

    pH_all = [pH[:min_length] for pH in pH_all]
    Ns_all = [Ns[:min_length, :] for Ns in Ns_all]
    shannon_all = [shannon[:min_length] for shannon in shannon_all]
    time = time[:min_length]

    pH_all = np.array(pH_all)
    Ns_all = np.array(Ns_all)
    shannon_all = np.array(shannon_all)

    pH_mean = np.mean(pH_all, axis=0)
    pH_ci = 1.96 * np.std(pH_all, axis=0) / np.sqrt(bootstrap_runs)
    Ns_mean = np.mean(Ns_all, axis=0)
    shannon_mean = np.mean(shannon_all, axis=0)
    shannon_ci = 1.96 * np.std(shannon_all, axis=0) / np.sqrt(bootstrap_runs)
    shannon_ci_lower = shannon_mean - shannon_ci
    shannon_ci_upper = shannon_mean + shannon_ci

    return time, pH_mean, pH_ci, Ns_mean, shannon_mean, shannon_ci_lower, shannon_ci_upper


if __name__ == "__main__":
    S_test = 20
    cp_test = 20
    days_test = 20
    pH0_test = 4.1
    kdeath_test = -0.1
    kgrowth_test = 5.
    acid_lover_index = 5
    acid_lover_initial_amount = 0.08
    acid_lover_kgrowth_test = 5.
    alpha_test = 0.5
    beta_test = 0.0
    bootstrap_runs = 20
    movement_factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(movement_factors)))  # Use virdis color map

    for movement_factor, color in zip(movement_factors, colors):
        time, pH_mean, pH_ci, Ns_mean, shannon_mean, shannon_ci95_lower, shannon_ci95_upper = bootstrap_simulations(
            S=S_test, cp=cp_test, days=days_test, pH0=pH0_test, kdeath=kdeath_test,
            kgrowth=kgrowth_test, acid_lover_index=acid_lover_index,
            acid_lover_initial_amount=acid_lover_initial_amount,
            acid_lover_kgrowth=acid_lover_kgrowth_test, alpha=alpha_test, beta=beta_test,
            movement_factor=movement_factor,
            bootstrap_runs=bootstrap_runs)

        plt.fill_between(time, shannon_ci95_lower, shannon_ci95_upper, color=color, alpha=0.3)
        plt.plot(time, shannon_mean, color=color)

    plt.xlabel("Time (days)")
    plt.ylabel("Shannon Diversity Index")
    plt.title('Effect of Movement Factor on Shannon Diversity Index with 95% Confidence Interval')

plt.show()
