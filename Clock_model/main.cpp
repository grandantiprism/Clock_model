#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <numeric>

using namespace std;

// シミュレーションパラメータ
const int L = 32;          // 格子サイズ
const int N = L * L;       // サイト数
const int Q = 6;           // 状態数 q
const double beta_min = 1.05;
const double beta_max = 1.55;
const int beta_num = 25;
const int MCS_THERM = 5000; // 熱平衡化のためのスイープ数
const int MCS_MEAS = 5000; // 測定用のスイープ数
const int MEAS_INTERVAL = 10; // 測定間隔

// 乱数生成器
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> dist_q(0, Q - 1);
uniform_int_distribution<int> dist_site(0, N - 1);
uniform_real_distribution<double> dist_01(0.0, 1.0);

// 格子データ
struct Lattice {
    vector<int> spins;
    double angles[Q];

    Lattice() : spins(N) {
        for (int i = 0; i < Q; ++i) {
            angles[i] = 2.0 * M_PI * i / Q;
        }
    }

    // 隣接サイトのインデックス取得 (周期境界条件)
    int get_up(int i) { return (i < L) ? i + N - L : i - L; }
    int get_down(int i) { return (i >= N - L) ? i + L - N : i + L; }
    int get_left(int i) { return (i % L == 0) ? i + L - 1 : i - 1; }
    int get_right(int i) { return (i % L == L - 1) ? i - L + 1 : i + 1; }
};

// メトロポリス・アップデート
void metropolis(Lattice& lat, double beta) {
    for (int i = 0; i < N; ++i) {
        int s = dist_site(gen);
        int old_state = lat.spins[s];
        int new_state = dist_q(gen);
        if (old_state == new_state) continue;

        double dE = 0;
        int neighbors[] = {lat.get_up(s), lat.get_down(s), lat.get_left(s), lat.get_right(s)};
        for (int n : neighbors) {
            dE -= cos(lat.angles[new_state] - lat.angles[lat.spins[n]]) -
                    cos(lat.angles[old_state] - lat.angles[lat.spins[n]]);
        }

        if (dE <= 0 || dist_01(gen) < exp(-beta * dE)) {
            lat.spins[s] = new_state;
        }
    }
}

// ウォルフ・クラスター・アップデート (埋め込みクラスター)
void wolff(Lattice& lat, double beta) {
    int seed = dist_site(gen);
    int r = dist_q(gen); // 反転軸
    
    vector<bool> in_cluster(N, false);
    vector<int> stack;

    auto get_reflected = [&](int s_idx) {
        return (2 * r - lat.spins[s_idx] + Q) % Q;
    };

    stack.push_back(seed);
    in_cluster[seed] = true;
    
    while (!stack.empty()) {
        int curr = stack.back();
        stack.pop_back();

        int old_curr_state = lat.spins[curr];
        int new_curr_state = get_reflected(curr);
        lat.spins[curr] = new_curr_state; // スタックから出た時に反転させる

        int neighbors[] = {lat.get_up(curr), lat.get_down(curr), lat.get_left(curr), lat.get_right(curr)};
        for (int n : neighbors) {
            if (!in_cluster[n]) {
                // 投影成分の計算 (内積)
                double si = cos(lat.angles[old_curr_state] - lat.angles[r]);
                double sj = cos(lat.angles[lat.spins[n]] - lat.angles[r]);
                
                // 同じ向きに投影される成分がある場合のみ結合
                double p = 1.0 - exp(min(0.0, -2.0 * beta * si * sj));
                
                if (dist_01(gen) < p) {
                    stack.push_back(n);
                    in_cluster[n] = true;
                }
            }
        }
    }
}

int main() {
    Lattice lat;
    
    string res_filename = "clock_q" + to_string(Q) + "_L" + to_string(L) + ".csv";
    ofstream ofs(res_filename);
    ofs << "beta,Energy,Mx,My,M2x,M2y,M3x,M3y,I,S" << endl;
    
    auto total_start = chrono::high_resolution_clock::now();

    for (int i = 0; i <= beta_num; ++i) {
        double beta = beta_max - (beta_max - beta_min) * i /beta_num;
        
        // 初期化 (低温側から始める場合は前回の状態を引き継ぐと収束が早い)
        for (int i = 0; i < MCS_THERM; ++i) {
            metropolis(lat, beta);
            wolff(lat, beta);
        }

        for (int i = 0; i < MCS_MEAS; ++i) {
            metropolis(lat, beta);
            wolff(lat, beta);

            if (i % MEAS_INTERVAL == 0) {
                double energy = 0, mx = 0, my = 0, m2x = 0, m2y = 0, m3x = 0, m3y = 0, I = 0, S = 0;
                for (int s = 0; s < N; ++s) {
                    double theta = lat.angles[lat.spins[s]];
                    mx += cos(theta);
                    my += sin(theta);
                    m2x += cos(2 * theta);
                    m2y += sin(2 * theta);
                    m3x += cos(3 * theta);
                    m3y += sin(3 * theta);
                    
                    // エネルギーとヘリシティ係数 (x方向の結合のみ I, S に使用)
                    int r = lat.get_right(s);
                    int d = lat.get_down(s);
                    energy -= cos(theta - lat.angles[lat.spins[r]]);
                    energy -= cos(theta - lat.angles[lat.spins[d]]);
                    
                    I += cos(theta - lat.angles[lat.spins[r]]);
                    S += sin(theta - lat.angles[lat.spins[r]]);
                }
                ofs << beta << "," << energy << "," << mx << "," << my << ","
                    << m2x << "," << m2y << "," << m3x << "," << m3y << "," << I << "," << S << endl;
            }
        }
        cout << "Finished beta = " << beta << endl;
    }

    return 0;
}
