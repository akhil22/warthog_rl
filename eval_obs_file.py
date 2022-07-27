import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv(sys.argv[1], header=None)
    df = df.to_numpy()
    num_steps = len(df)
    plt.figure()
    plt.ion()
    t = 0
    for j in df:
        print(j)
        i = 0
        x_way = []
        y_way = []
        for i in range(0,10):
            init_id = i*4
            r = j[init_id]
            th = j[init_id+1]
            x = r*np.cos(th)
            y = r*np.sin(th)
            x_way.append(x)
            y_way.append(y)
        plt.plot(x_way, y_way)
        plt.draw()
        plt.pause(0.001)
        plt.clf()
        t = t+1
if __name__ == "__main__":
    main()
