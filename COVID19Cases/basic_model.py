"""
[1] Estimates of the severity of coronavirus disease 2019: a model-based
analysis, the Lancet, R. Verity, L.C. Okell, I. Dorigatti, P. Winskill,
C. Whittaker, N. Imai, et al, published 30th March 2020, available at:
https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext

[2] Mizumoto Kenji, Kagaya Katsushi, Zarebski Alexander, Chowell Gerardo.
Estimating the asymptomatic proportion of coronavirus disease 2019 (COVID-19)
cases on board the Diamond Princess cruise ship, Yokohama, Japan, 2020. Euro
Surveill. 2020;25(10):pii=2000180. https://doi.org/10.2807/1560-7917.
ES.2020.25.10.2000180

[3] COVID-19 Dashboard by the Center for Systems Science and Engineering (CSSE)
at Johns Hopkins University (JHU):
https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
Calculated from data accessed at 22:51, 26th April 2020
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# [1]
DEATH_TIME = 17.8
DEATH_TIME_RANGE = [16.9, 19.2]
RECOVERY_TIME = 24.7
RECOVERY_TIME_RANGE = [22.9, 28.1]

# [2]
ASYMPTOMATIC_RATIO = 328 / 306

# [3]
MORTALITY_RATE = 0.069466
INITIAL_DOUBLE_TIME = 2.5
SYMPTOM_ONSET_TIME = 14

INITIAL_SYMPTOMATIC = 9
INITIAL_INCUBATING = INITIAL_SYMPTOMATIC * ASYMPTOMATIC_RATIO
TOTAL_POPULATION = 66650000


def exposure_ratio(reduced_exposure, lockdown_time, time):
    if time < lockdown_time:
        return 1
    return reduced_exposure


def differential_equation(n, t, re, tl):
    fi = exposure_ratio(re, tl, t)
    dnudt = - (
        (np.log(2) * (fi * n[1] + re * n[2]) * n[0]) /
        (INITIAL_DOUBLE_TIME * TOTAL_POPULATION)
    )
    dnidt = -dnudt - (np.log(2) / SYMPTOM_ONSET_TIME) * n[1]
    dnsdt = (np.log(2) / SYMPTOM_ONSET_TIME) * n[1] - np.log(2) * (
            MORTALITY_RATE / DEATH_TIME + (1 - MORTALITY_RATE) / RECOVERY_TIME
    ) * n[2]
    dnddt = (np.log(2) * MORTALITY_RATE / DEATH_TIME) * n[2]
    dnrdt = (np.log(2) * (1 - MORTALITY_RATE) / RECOVERY_TIME) * n[2]
    return [dnudt, dnidt, dnsdt, dnddt, dnrdt]


def cases_by_stage(exposure, lockdown_time, end_time):
    initial_condition = [TOTAL_POPULATION, INITIAL_INCUBATING, INITIAL_SYMPTOMATIC, 0, 0]
    time = np.arange(0, end_time, 1)
    result = odeint(
        differential_equation,
        initial_condition,
        time,
        args=(exposure, lockdown_time)
    )
    keys = ["uninfected", "incubating", "symptomatic", "dead", "recovered"]
    return dict(zip(keys, zip(*result)))


def plot_cases(exposure, lockdown_time, end_time, stage="symptomatic"):
    plt.plot(
        np.arange(0, end_time, 1),
        cases_by_stage(exposure, lockdown_time, end_time)[stage]
    )
    plt.xlabel(f"Time since outbreak began / days")
    plt.ylabel("Number of cases")
    plt.title(f"Number of {stage} cases over time")
    plt.show()


if __name__ == "__main__":
    plot_cases(0.1, 365, 365, "dead")
