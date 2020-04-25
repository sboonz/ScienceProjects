import numpy as np
from scipy.integrate import odeint


CONTAGIOUSNESS = 0.01
P_SYMPTOM = 0.03
P_RECOVERY = 0.05
P_DEATH = 0.01
TOTAL_POPULATION = 65000000


def exposure_ratio_function(exposed_ratio, lockdown_time, time):
    if time < lockdown_time:
        return 1
    return exposed_ratio


def differential_equation(nu, ni, ns, er, tl, t):
    fi = exposure_ratio_function(er, tl, t)
    fs = exposure_ratio_function(0, tl, t)
    dnudt = - CONTAGIOUSNESS * nu * (fi * ni + fs * ns)
    dnidt = -dnudt - P_SYMPTOM * ni
    dnsdt = P_SYMPTOM * ni - (P_DEATH + P_RECOVERY) * ns
    dnddt = P_DEATH * ns
    dnrdt = P_RECOVERY * ns
    return [dnudt, dnidt, dnsdt, dnddt, dnrdt]


def cases_by_stage(exposed_ratio, lockdown_time, end_time):
    initial_condition = [TOTAL_POPULATION, 0, 0, exposed_ratio, lockdown_time]
    time = np.arange(0, end_time, end_time)
    return odeint(differential_equation, initial_condition, time)


if __name__ == "__main__":
    print(cases_by_stage(0.2, 120, 960))
