import numpy as np
import math


class RL_FCM:
    def __init__(self, data):
        self.Eps = 10**(-3)
        self.data = data
        self.size, self.dim = self.data.shape
        # self.c = self.size  # Initial number of clusters

    def runAlgorithm(self):
        U, V = self.Iteration()
        clusters = np.argmax(U, axis=1)
        return clusters

    def Iteration(self):
        A, r1, r2, r3, V_last = self.Initial()
        c_last = self.size
        t = 1
        class_num = np.zeros(100)
        flag = 0
        A_nor = A.copy()
        while True:
            # print(c_last,t)
            U = self.calculate_U(A_nor, V_last, r1, r2,c_last)
            # r1 = np.exp(-(t)/100)  # 更新r1
            # r2 = np.exp(-(t) / 1000)  # 更新r2(iris)
            r1 = np.exp(-t / 1000)
            r2 = np.exp(-t / 100)#(seeds)
            A = self.calculate_A(U, A_nor, r3, r1,c_last)  # 更新概率矩阵
            if flag == 0:
                r3 = self.calculate_r3(A, A_nor,U,c_last)  # 更新r3
            A_nor, U_nor,c = self.select_class(A, U,c_last)  # 删除了不应该存在的类后的概率矩阵和隶属度矩阵
            if flag == 0:
                if t >= 100:
                    if class_num[t%100] == c:
                        r3 = 0
                        flag = 1
                class_num[(t-1) % 100] = c
            V = self.calculate_V(U_nor,c)
            max_dist = 0
            for i in range(c):
                if np.linalg.norm(V[i] - V_last[i]) > max_dist:
                    max_dist = np.linalg.norm(V[i] - V_last[i])
            # print(V)
            if max_dist < self.Eps:
                break
            V_last = V.copy()
            t += 1
            c_last = c
        return U, V

    def Initial(self):#初始化中心矩阵，概率矩阵等
        A = [1 / self.size for i in range(self.size)]
        r1 = 1
        r2 = 1
        r3 = 1
        V = []
        for i in range(self.size):
            current_center = []
            for s in range(self.dim):
                current_center.append(self.data[i][s])
            V.append(current_center)
        return A,r1,r2,r3,V

    def calculate_D(self,V,c):
        D = np.zeros((self.size,c))
        for i in range(self.size):
            for j in range(c):
                D[i,j] = np.linalg.norm(self.data[i] - V[j])
        return D

    def calculate_U(self,A,V,r1,r2,c):
        U = []
        D = self.calculate_D(V,c)
        for i in range(self.size):
            current = []
            U_fenmu = 0
            for t in range(c):
                U_fenmu += np.exp((-np.power(D[i,t],2) + r1 * math.log(A[t])) / r2)
            for k in range(c):
                U_fenzi = np.exp((-np.power(D[i,k],2) + r1 * math.log(A[k])) / r2)
                current.append(U_fenzi / U_fenmu)
            U.append(current)
        return U

    def calculate_A(self,U,A_,r3,r1,c):
        A = []
        sum_ = 0
        for t in range(c):
            sum_ += A_[t] * math.log(A_[t])
        for k in range(c):
            sum1 = 0
            for j in range(self.size):
                sum1 += U[j][k]
            one_part = (1 / self.size) * sum1
            two_part = (r3 / r1) * A_[k] * (math.log(A_[k]) - sum_)
            A.append(one_part + two_part)
        return A

    # 存疑
    def calculate_r3(self,A,A_,U,c):
        one_part = 0
        sum_two = 0
        mix_pro = min(1,2 / np.power(self.dim,np.round(self.dim / 2-1)))
        for k in range(c):
            one_part += np.exp(-self.size * mix_pro * abs(A[k] - A_[k]))
        one_part = one_part / c
        for t in range(c):
            sum_two += A_[t] * math.log(A_[t])
        sum_two = sum_two * (-np.max(A_))
        max_d = 0
        for k in range(c):
            sum1 = 0
            for j in range(self.size):
                sum1 += U[j][k]
            sum_U = (1 / self.size) * sum1
            if sum_U > max_d:
                max_d = sum_U
        two_part = (1 - sum_U) / sum_two
        r3 = min(one_part,two_part)
        return r3

    def select_class(self,A,U,c):
        discard_class = []
        for i in range(c):
            if A[i] < 1 / self.size:
                discard_class.append(i)
        discard_length = len(discard_class)
        sum_A = 0
        A_nor = []
        U_nor = []
        for i in range(c):
            if i not in discard_class:
                sum_A += A[i]
        for i in range(c):
            if i not in discard_class:
                A_nor.append(A[i] / sum_A)
        for i in range(self.size):
            sum_Ui = 0
            for k in range(c):
                if k not in discard_class:
                    sum_Ui += U[i][k]
            current = []
            for k in range(c):
                if k not in discard_class:
                    current.append(U[i][k] / sum_Ui)
            U_nor.append(current)
        c = c - discard_length
        return A_nor,U_nor,c

    def calculate_V(self,U,c):
        V = np.zeros((c,self.dim))
        for k in range(c):
            V_fenmu = 0
            for i in range(self.size):
                V_fenmu += U[i][k]
            for s in range(self.dim):
                for i in range(self.size):
                    V[k][s] += U[i][k] * self.data[i][s]
                V[k][s] = V[k][s] / V_fenmu
        return V
