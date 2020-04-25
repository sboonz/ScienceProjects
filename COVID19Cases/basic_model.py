"""
[1] Estimates of the severity of coronavirus disease 2019: a model-based
analysis, the Lancet, R. Verity, L.C. Okell, I. Dorigatti, P. Winskill,
C. Whittaker, N. Imai, et al, published 30th March 2020, available at:
https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


DEATH_TIME = 17.8
DEATH_TIME_RANGE = [16.9, 19.2]
RECOVERY_TIME = 24.7
RECOVERY_TIME_RANGE = [22.9, 28.1]
CONTAGIOUSNESS = 0.01
P_SYMPTOM = 0.03
P_RECOVERY = 0.05
P_DEATH = 0.01
TOTAL_POPULATION = 65000000


def exposure_ratio_function(exposed_ratio, lockdown_time, time):
    if time < lockdown_time:
        return 1
    return exposed_ratio


def differential_equation(n, er, tl, t):
    fi = exposure_ratio_function(er, tl, t)
    fs = exposure_ratio_function(0, tl, t)
    dnudt = - CONTAGIOUSNESS * n[0] * (fi * n[1] + fs * n[2])
    dnidt = -dnudt - P_SYMPTOM * n[1]
    dnsdt = P_SYMPTOM * n[1] - (P_DEATH + P_RECOVERY) * n[2]
    dnddt = P_DEATH * n[2]
    dnrdt = P_RECOVERY * n[2]
    return [dnudt, dnidt, dnsdt, dnddt, dnrdt]


def cases_by_stage(exposed_ratio, lockdown_time, end_time):
    initial_condition = [TOTAL_POPULATION - 1, 1, 0, 0, 0]
    time = np.arange(0, end_time, 1)
    result = odeint(
        differential_equation,
        initial_condition,
        time,
        args=(exposed_ratio, lockdown_time)
    )
    keys = ["uninfected", "incubating", "symptomatic", "dead", "recovered"]
    return dict(zip(keys, zip(*result)))


def plot_cases(exposed_ratio, lockdown_time, end_time, stage="symptomatic"):
    plt.plot(
        np.arange(0, end_time, 1),
        cases_by_stage(exposed_ratio, lockdown_time, end_time)[stage]
    )
    plt.xlabel(f"Time since outbreak began / hours")
    plt.ylabel("Number of cases")
    plt.title(f"Number of {stage} cases over time")
    plt.show()


if __name__ == "__main__":
    plot_cases(0.2, 120, 960)
