import numpy as np
import matplotlib.pyplot as plt

def solve_clock_model_check():
    L = 16
    N = L * L
    Q = 6
    beta = 1.5  # 低温
    J = 1.0
    
    # 角度の定義
    angles = np.array([2 * np.pi * k / Q for k in range(Q)])
    
    # コールドスタート (全員状態0)
    spins = np.zeros((L, L), dtype=int)
    
    # 最近接インデックスの準備 (PBC)
    # 上下左右へのシフト
    
    n_steps = 10000 # 十分なステップ数
    
    energy_hist = []
    I_hist = []
    
    print(f"Start Simulation: L={L}, beta={beta}, Cold Start")
    
    for step in range(n_steps):
        # ランダムにサイトを選ぶ
        i, j = np.random.randint(0, L, 2)
        current_s = spins[i, j]
        
        # 新しい状態候補
        new_s = np.random.randint(0, Q)
        
        if current_s == new_s:
            continue
            
        # 周囲のスピン
        nb_i = [(i+1)%L, (i-1)%L, i, i]
        nb_j = [j, j, (j+1)%L, (j-1)%L]
        
        # 局所エネルギー計算 H = -J sum cos(theta_i - theta_j)
        E_old = 0
        E_new = 0
        
        for k in range(4):
            nb_spin = spins[nb_i[k], nb_j[k]]
            E_old -= J * np.cos(angles[current_s] - angles[nb_spin])
            E_new -= J * np.cos(angles[new_s]     - angles[nb_spin])
            
        dE = E_new - E_old
        
        # Metropolis判定
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = new_s
            
        # 定期的に測定
        if step % 100 == 0:
            # Helicity I の計算 (右隣とのボンド)
            # numpyのロールを使って一括計算
            spins_right = np.roll(spins, -1, axis=1) # 右隣
            diffs = angles[spins] - angles[spins_right]
            I_val = np.sum(np.cos(diffs))
            I_hist.append(I_val)

    print(f"Final I value: {I_hist[-1]:.2f} (Expected ~250)")
    
    plt.plot(I_hist)
    plt.title("Convergence of I (Helicity Cos component)")
    plt.xlabel("Meas Steps")
    plt.ylabel("Sum cos(dTheta)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    solve_clock_model_check()