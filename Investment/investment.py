import numpy as np
import matplotlib.pyplot as plt


def stock_price(t, seed_share_price, growth_rate, volatility):
    if volatility > 1 or volatility < 0:
        raise Exception(f"Volatility must be between 0 and 1. Got {volatility}")
    return (1 + volatility * (2 * np.random.random() - 1)) * \
        seed_share_price * np.exp(growth_rate * t)


if __name__ == "__main__":
    time = np.arange(0, 200, 1)
    stock_prices = np.array([stock_price(ti, 10, 0.05, 0.4) for ti in time])
    plt.plot(time, stock_prices)
    plt.show()
