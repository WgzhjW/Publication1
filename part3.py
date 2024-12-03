import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.integrate import solve_ivp

EPS = 10 ** -8  # Numerical threshold
dispersal = 10 ** -6  # Species dispersal rate

def run(S=20, cp=100., days=15, pH0=7, kdeath=-0.1, kgrowth=10.,
        acid_lover_index=0, acid_lover_initial_amount=0.1, acid_lover_kgrowth=15.0,
        acid_lover2_index=1, acid_lover2_initial_amount=0.1, acid_lover2_kgrowth=15.0,
        alpha=1000, beta=0, **kwargs):
    ppref = kwargs.get('ppref', np.random.normal(pH0, 1, S))
    ppref[acid_lover_index] = 3.5
    ppref[acid_lover2_index] = 3.5

    cs = np.random.uniform(-1, 1, S) * cp  # Ensuring all cs are negative
    cs[acid_lover_index] = -np.abs(np.random.uniform(2, 4) * cp)
    cs[acid_lover2_index] = -np.abs(np.random.uniform(0, 0) * cp)

    initial_populations = np.random.random(S) * 0.05
    initial_populations[acid_lover_index] = acid_lover_initial_amount
    initial_populations[acid_lover2_index] = acid_lover2_initial_amount
    x0 = kwargs.get('x0', np.concatenate([[pH0], np.log(initial_populations)]))

    ph_ranges = []

    def fun(p):
        ingate = np.zeros(S)
        current_ph_ranges = []
        for i in range(S):
            if i != acid_lover_index and i != acid_lover2_index:
                low_bound = np.random.normal(4, 0.4)
                high_bound = low_bound + 1.0
                ingate[i] = float(low_bound <= p <= high_bound)
                current_ph_ranges.append((low_bound, high_bound))
            else:
                if i == acid_lover_index:
                    ingate[acid_lover_index] = float(3.75 <= p <= 5.0)
                if i == acid_lover2_index:
                    ingate[acid_lover2_index] = float(3.5 <= p <= 5.0)
        ph_ranges.append(current_ph_ranges)
        return ingate

    def eqs(t, x):
        p = x[0]
        x = np.exp(np.clip(x[1:], np.log(EPS), 0))
        dx = np.zeros_like(x)
        for i in range(S):
            growth_rate = kgrowth
            if i == acid_lover_index:
                growth_rate = acid_lover_kgrowth
            elif i == acid_lover2_index:
                growth_rate = acid_lover2_kgrowth
            competition_term = 0
            for j in range(S):
                if i != j:
                    competition_term += (alpha if j == acid_lover_index else beta) * x[j]
            dx[i] = x[i] * (growth_rate * fun(p)[i] * (1 - x[i] - competition_term))

        if kdeath < 0:
            dx += np.log(-kdeath) * x
        dx += dispersal
        dp = np.sum(cs * x)
        if kdeath < 0:

            pKa = 3.5
            buffer_concentration = 400

            exp_input = np.clip((p - pKa) * np.log(10), -100, 10)
            buffer_effect = buffer_concentration / (1 + np.exp(exp_input))
            dp -= buffer_effect * (p - pH0)

            p_updated = p + dp
            p_clipped = np.clip(p_updated, 2, None)
            dp = p_clipped - p

        return np.concatenate([[dp], dx / x])

    time = []
    pH = []
    Ns = []
    for day in range(days):
        t_eval = np.linspace(0, 1., 10)  # Specify 100 time points for each day
        sol = solve_ivp(fun=eqs, t_span=(0, 1.), y0=x0, t_eval=t_eval)
        pH.append(sol.y[0])
        Ns.append(np.exp(sol.y[1:].T))
        time.append(sol.t + day)
        x0 = sol.y[:, -1].copy()
    pH = np.concatenate(pH)
    Ns = np.concatenate(Ns, axis=0)
    time = np.concatenate(time)

    return time, pH, Ns, ph_ranges

# Parameters for testing
S_test = 20
cp_test = 20
days_test = 30
pH0_test = 3.8
kdeath_test = -0.05
kgrowth_test = 5.
acid_lover_index = 18
acid_lover_initial_amount = 0.03
acid_lover_kgrowth_test = 5.
acid_lover2_index = 19
acid_lover2_initial_amount = 0.01
acid_lover2_kgrowth_test = 4.0
alpha_test = 0.0
beta_test = 0.0

# Run simulation with the updated function
time, pH, Ns, _ = run(S=S_test, cp=cp_test, days=days_test, pH0=pH0_test,
                      kdeath=kdeath_test, kgrowth=kgrowth_test,
                      acid_lover_index=acid_lover_index, acid_lover_initial_amount=acid_lover_initial_amount,
                      acid_lover_kgrowth=acid_lover_kgrowth_test, acid_lover2_index=acid_lover2_index,
                      acid_lover2_initial_amount=acid_lover2_initial_amount,
                      acid_lover2_kgrowth=acid_lover2_kgrowth_test, alpha=alpha_test, beta=beta_test)

# Plotting results with optimized color scheme
plt.figure(figsize=(18, 6))  # Increase canvas width
font_prop = FontProperties(family='Arial', weight='bold')

# Plotting pH values over time (left side)
plt.subplot(1, 2, 1)  # Change to 1 row and 3 columns
plt.plot(time, pH, label='pH Value', color='b', linewidth=4)
plt.title('pH Changes Over Time', fontproperties=font_prop, fontsize=24)
plt.xlabel('Time (days)', fontproperties=font_prop, fontsize=24)
plt.ylabel('pH Value', fontproperties=font_prop, fontsize=24)
plt.xlim(0, 30)
plt.gca().set_facecolor('#E6E6E6')  # Set gray background
plt.grid(True, linestyle='--', color='white')

# Adjust tick label fonts
plt.xticks(fontproperties=font_prop, fontsize=22)  # x-axis ticks
plt.yticks(fontproperties=font_prop, fontsize=22)  # y-axis ticks

# Adjust border lines
ax1 = plt.gca()
for spine in ax1.spines.values():
    spine.set_linewidth(3.0)  # Set border line width
    spine.set_color('black')  # Set border line color

# Plotting population dynamics with optimized colors (middle)
plt.subplot(1, 2, 2)
cmap = plt.get_cmap('viridis', Ns.shape[1])  # Get a colormap with the correct number of species

for i in range(Ns.shape[1]):
    if i == acid_lover_index:
        plt.plot(time, Ns[:, i], color='darkred', label=f'Species {i + 1}', linewidth=4)  # Dark red
    elif i == acid_lover2_index:
        plt.plot(time, Ns[:, i], color='red', label=f'Species {i + 1}', linewidth=4)  # Red
    else:
        plt.plot(time, Ns[:, i], color=cmap(i), label=f'Species {i + 1}', linewidth=2)

plt.title('Population Dynamics Over Time', fontproperties=font_prop, fontsize=24)
plt.xlabel('Time (days)', fontproperties=font_prop, fontsize=24)
plt.xlim(0, 30)
plt.yscale('log')
plt.gca().set_facecolor('#E6E6E6')
plt.grid(True, linestyle='--', color='white')

# Adjust tick label fonts
plt.xticks(fontproperties=font_prop, fontsize=22)
plt.yticks(fontproperties=font_prop, fontsize=22)

# Adjust border lines
ax2 = plt.gca()
for spine in ax2.spines.values():
    spine.set_linewidth(3)  # Set border line width
    spine.set_color('black')  # Set border line color

plt.tight_layout()
plt.show()