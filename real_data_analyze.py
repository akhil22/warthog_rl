import pandas as pd
import matplotlib.pyplot as plt

import sys
def main():
    pose_file = sys.argv[1]
    df = pd.read_csv(pose_file)
    vel = df['vel']
    v_cmd = df['v_cmd']
    w = df['w']
    w_cmd = df['w_cmd']
    plt.plot(vel, 'r', label="actual velocity")
    plt.plot(v_cmd, 'g', label="commanded velocity")
    plt.legend()
    plt.title(f"Commanded vs actual velocity")
    plt.figure(2)
    plt.plot(w, 'r', label="Actual angular velocity")
    plt.plot(w_cmd, 'g', label="Commanded angular velocity")
    plt.legend()
    plt.title(f"Commanded vs actual anguarl velocity")
    plt.show()

if __name__=='__main__':
    main()

