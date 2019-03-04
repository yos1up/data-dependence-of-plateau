import numpy as np
import scipy.special as spsp

def overlap(x, y):
    return np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
def xor(a, b):
    return (a or b) and (not (a and b))
def safe_sqrt(x):
    if x<-1e-6: print('Warning (sqrt):', x)   
    return np.sqrt(x) if x >= 0 else 0.
def safe_arcsin(x):
    if abs(x)>1.000001: print('Warning (arcsin):', x)
    return np.arcsin(min(max(x,-1.),1.))
# 上のwarningがしょっちゅうでる場合は，QRTの初期条件が不正でないか確認してください
# (Qが半正定値でないとか)

class act_erf:
    def g(self,x):
        return spsp.erf(x/np.sqrt(2.))
    def gp(self,x):
        return np.sqrt(2/np.pi) * np.exp(-x*x/2)
    def gx1gx2_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(2), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.g(x[i,0]) * self.g(x[i,1])
        return ret/x.shape[0]
    def gpx1x2gx3_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(3), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * x[i,1] * self.g(x[i,2])
        return ret/x.shape[0]
    def gpx1gpx2gx3gx4_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(4), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * self.gp(x[i,1]) * self.g(x[i,2]) * self.g(x[i,3])
        return ret/x.shape[0]
    def gx1gx2(self,C):
        return 2/np.pi*safe_arcsin(C[0,1]/safe_sqrt((1+C[0,0])*(1+C[1,1])))
    def gpx1x2gx3(self,C):
        return 2/np.pi/safe_sqrt((1+C[0,0])*(1+C[2,2])-C[0,2]**2) * (C[1,2]*(1+C[0,0])-C[0,1]*C[0,2])/(1+C[0,0])
    def gpx1gpx2gx3gx4(self,C):
        L4 = (1+C[0,0])*(1+C[1,1])-C[0,1]**2
        L0 = L4*C[2,3] - C[1,2]*C[1,3]*(1+C[0,0]) - C[0,2]*C[0,3]*(1+C[1,1]) + C[0,1]*C[0,2]*C[1,3] + C[0,1]*C[0,3]*C[1,2]
        L1 = L4*(1+C[2,2]) - C[1,2]**2*(1+C[0,0]) - C[0,2]**2*(1+C[1,1]) + 2*C[0,1]*C[0,2]*C[1,2]
        L2 = L4*(1+C[3,3]) - C[1,3]**2*(1+C[0,0]) - C[0,3]**2*(1+C[1,1]) + 2*C[0,1]*C[0,3]*C[1,3]
        return (2/np.pi)**2 / safe_sqrt(L4) * safe_arcsin(L0/safe_sqrt(L1 * L2))

class act_relu:
    def g(self,x):
        return max(x, 0.)
    def gp(self,x):
        return 1.*(x>0.)
    def gx1_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(1), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.g(x[i,0])
        return ret/x.shape[0]    
    def gx1gx2_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(2), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.g(x[i,0]) * self.g(x[i,1])
        return ret/x.shape[0]
    def gpx1x2gx3_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(3), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * x[i,1] * self.g(x[i,2])
        return ret/x.shape[0]
    def gpx1gpx2gx3gx4_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(4), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * self.gp(x[i,1]) * self.g(x[i,2]) * self.g(x[i,3])
        return ret/x.shape[0]
    def gx1(self,C):
        return safe_sqrt(C[0,0]/2/np.pi)
    def gx1gx2(self,C):
        return C[0,1]*(1./4 + 1./2/np.pi*safe_arcsin(C[0,1]/safe_sqrt(C[0,0]*C[1,1]))) + 1./2/np.pi*safe_sqrt(C[0,0]*C[1,1]-C[0,1]**2)        
    def gpx1x2gx3(self,C):
        return C[1,2]*(1./4 + 1./2/np.pi*safe_arcsin(C[0,2]/safe_sqrt(C[0,0]*C[2,2]))) + C[0,1]/2/np.pi/C[0,0]*safe_sqrt(C[0,0]*C[2,2]-C[0,2]**2)
    def gpx1gpx2gx3gx4(self,C):
        print('cannot determine analytically gpx1gpx2gx3gx4 where g=relu')
        raise ValueError
        return 0.


class act_id:
    def g(self,x):
        return x
    def gp(self,x):
        return 1.
    def gx1gx2_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(2), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.g(x[i,0]) * self.g(x[i,1])
        return ret/x.shape[0]
    def gpx1x2gx3_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(3), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * x[i,1] * self.g(x[i,2])
        return ret/x.shape[0]
    def gpx1gpx2gx3gx4_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(4), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * self.gp(x[i,1]) * self.g(x[i,2]) * self.g(x[i,3])
        return ret/x.shape[0]    
    def gx1gx2(self,C):
        return C[0,1]
    def gpx1x2gx3(self,C):
        return C[1,2]
    def gpx1gpx2gx3gx4(self,C):
        return C[2,3]

class act_exp:
    def g(self,x):
        return np.exp(x)
    def gp(self,x):
        return np.exp(x)
    def gx1gx2_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(2), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.g(x[i,0]) * self.g(x[i,1])
        return ret/x.shape[0]
    def gpx1x2gx3_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(3), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * x[i,1] * self.g(x[i,2])
        return ret/x.shape[0]
    def gpx1gpx2gx3gx4_(self, C=[], x=[]):
        if (x==[]): x = np.random.multivariate_normal(np.zeros(4), C, 1000)
        ret = 0.0
        for i in range(x.shape[0]):
            ret += self.gp(x[i,0]) * self.gp(x[i,1]) * self.g(x[i,2]) * self.g(x[i,3])
        return ret/x.shape[0]    
    def gx1gx2(self,C):
        return np.exp(0.5*C[0,0] + C[0,1] + 0.5*C[1,1])
    def gpx1x2gx3(self,C):
        return np.exp(0.5*C[0,0] + C[0,2] + 0.5*C[2,2])*(C[0,1] + C[1,2])
    def gpx1gpx2gx3gx4(self,C):
        return np.exp(0.5*np.dot(np.dot(np.ones((1,4)), C), np.ones((4,1)))[0,0])



def average_check():
    e = act_erf()
    while 1:
        c1 = np.random.rand(1,1)
        c1 = np.dot(c1, c1.T)
        c2 = np.random.rand(2,2)
        c2 = np.dot(c2, c2.T)
        c3 = np.random.rand(3,3)
        c3 = np.dot(c3, c3.T)
        c4 = np.random.rand(4,4)
        c4 = np.dot(c4, c4.T)
        # print('I1')
        # print(e.gx1(c1))
        # print(sum([e.gx1_(c1) for i in range(1000)])/1000)
        # print('I2')
        # print(e.gx1gx2(c2))
        # print(sum([e.gx1gx2_(c2) for i in range(1000)])/1000)
        print('I3')
        print(e.gpx1x2gx3(c3))
        print(sum([e.gpx1x2gx3_(c3) for i in range(1000)])/1000)
        # print('I4')
        # print(e.gpx1gpx2gx3gx4(c4))
        # print(sum([e.gpx1gpx2gx3gx4_(c4) for i in range(1000)])/1000)
        # print(sum([e.gpx1gpx2gx3gx4_(c4) for i in range(10000)])/10000)
        # print(sum([e.gpx1gpx2gx3gx4_(c4) for i in range(100000)])/100000)




class opd_q1r1d1e1:
    def __init__(self, qrde, eta=0.01, act='erf'):
        self.qrde = qrde.copy()
        self.t = 1.0
        self.f = 1.0
        self.eta = eta
        if act=='erf':
            self.act = act_erf()
        elif act=='relu':
            self.act = act_relu()
        elif act=='id':
            self.act = act_id()
        else:
            raise ValueError
    def I2(self, QRT, i, j):
        return self.act.gx1gx2(QRT[[i,j],:][:,[i,j]])
    def I3(self, QRT, i, j, k):
        return self.act.gpx1x2gx3(QRT[[i,j,k],:][:,[i,j,k]])
    def I4(self, QRT, i, j, k, l):
        return self.act.gpx1gpx2gx3gx4(QRT[[i,j,k,l],:][:,[i,j,k,l]])        
    def update(self):
        # euler
        # dQRT = self.deltaQRT(self.QRT, self.wv)
        # dwv = self.deltawv(self.QRT, self.wv)
        # self.QRT += dQRT
        # self.wv += dwv

        # 2nd runge-kutta
        # dQRT = self.deltaQRT(self.QRT, self.wv)
        # dwv = self.deltawv(self.QRT, self.wv)
        # dQRT2 = self.deltaQRT(self.QRT+dQRT, self.wv+dwv)
        # dwv2 = self.deltawv(self.QRT+dQRT, self.wv+dwv)
        # self.QRT += (dQRT + dQRT2)/2
        # self.wv += (dwv + dwv2)/2

        # 4th runge-kutta
        dqrde = self.delta(self.qrde)
        dqrde2 = self.delta(self.qrde+dqrde/2)
        dqrde3 = self.delta(self.qrde+dqrde2/2)
        dqrde4 = self.delta(self.qrde+dqrde3)
        self.qrde += (dqrde + 2*dqrde2 + 2*dqrde3 + dqrde4)/6
        return
    def delta(self, qrde):
        q, r, d, e = qrde
        QRT = np.array([[q, r], [r, self.t]])
        dq = -2 * self.eta * (d * self.I3(QRT, 0, 0, 0) - e * self.I3(QRT, 0, 0, 1))
        dr = -self.eta * (d * self.I3(QRT, 0, 1, 0) - e * self.I3(QRT, 0, 1, 1))
        dd = -2 * self.eta * (d * self.I2(QRT, 0, 0) - e * self.I2(QRT, 0, 1))
        de = -self.eta * (e * self.I2(QRT, 0, 0) - self.f * self.I2(QRT, 0, 1))
        return np.array([dq, dr, dd, de])
    def calceps(self, qrde):
        return 0
    def eps(self):
        return self.calceps(self.qrde)




class opd_inf_KM_inf:
    '''
    （メモ）
    deltaABC の η^2 の項は，O(1/N) なので，N→∞ で消える．（なので ignore_etasq[1]のデフォが False となっている）
    '''
    def __init__(self, QRT, ABC, N=[], K=[], M=[], O=[], eta=0.01, eta2=[], poserf=False, ignore_etasq=[False,True], etasq_full=True, act='erf',updateperiod=1, calcmode='', QRTcoef=[1, 1], ABCcoef=[1], substep=1):
        assert(poserf==False)
        assert(etasq_full==True)
        assert(len(ignore_etasq)==2)
        assert(QRT.shape[0]==K+M and QRT.shape[1]==K+M)
        assert(ABC.shape[0]==K+M and ABC.shape[1] == K+M)
        assert(substep==1)# or updateperiod==1) # 未実装のため．
        self.QRT = QRT.copy()
        self.ABC = ABC.copy()
        self.N = N
        if (self.N==[]): self.N = 1 # オーダパラメータだけ回す場合とかはこの設定で良い．
        self.K = K
        self.M = M
        self.eta = eta
        self.eta2 = eta2
        if (self.eta2==[]): self.eta2 = eta
        self.poserf = poserf
        self.etasq_full = etasq_full
        # self.etaN = N * eta
        if act=='erf':
            self.act = act_erf()
        elif act=='relu':
            self.act = act_relu()
        elif act=='lin':
            self.act = act_lin()
        elif act=='exp':
            self.act = act_exp()
        elif act=='id':
            self.act = act_id()
        else:
            raise ValueError
        self.deterministic = False
        self.ignore_etasq = ignore_etasq
        self.updateperiod = updateperiod
        self.updateperiodcnt = 0
        self.updatecnt = 0
        self.calcmode = calcmode
        self.QRTcoef = QRTcoef
        self.ABCcoef = ABCcoef
        self.substep = substep
    def I2(self, QRT, i, j):
        if (self.deterministic):
            return self.act.gx1gx2_(x=self.hiddenvoltage[[i,j]].reshape(1,2))
        else:
            return self.act.gx1gx2(QRT[[i,j],:][:,[i,j]])
    def I3(self, QRT, i, j, k):
        if (self.deterministic):
            return self.act.gpx1x2gx3_(x=self.hiddenvoltage[[i,j,k]].reshape(1,3))
        else:        
            return self.act.gpx1x2gx3(QRT[[i,j,k],:][:,[i,j,k]])
    def I4(self, QRT, i, j, k, l):
        if (self.deterministic):
            return self.act.gpx1gpx2gx3gx4_(x=self.hiddenvoltage[[i,j,k,l]].reshape(1,4))
        else:      
            return self.act.gpx1gpx2gx3gx4(QRT[[i,j,k,l],:][:,[i,j,k,l]])        
    def I5(self, QRT, i, j, k, l):
        if (self.deterministic):
            return self.act.gx1gx2gx3gx4(x=self.hiddenvoltage[[i,j,k,l]].reshape(1,4))
        else:      
            return self.act.gx1gx2gx3gx4(QRT[[i,j,k,l],:][:,[i,j,k,l]])        
    def update(self, deterministic=False, hiddenvoltage=[], inputdata=[]):
        if (self.updateperiod==np.inf): return # infの場合は、そもそもアップデートしない。（特殊処理）（これ書かないと、初回の更新幅で未来永劫更新する挙動となる）
        if (self.updateperiodcnt == 0):
            if (deterministic):
                self.deterministic = True
                assert(hiddenvoltage.shape[1]==1)
                self.hiddenvoltage = hiddenvoltage.flatten()
                assert(hiddenvoltage.shape[0]==self.K+self.M)
                self.inputdata = inputdata.flatten()
                # euler
                self.total_dQRT = self.deltaQRT(self.QRT, self.ABC)
                self.total_dABC = self.deltaABC(self.QRT, self.ABC)
                # これらはupdateperiodが2以上のときは再計算されることなく用いられる
            else:
                self.deterministic = False
                # euler
                # dQRT = self.deltaQRT(self.QRT, self.ABC)
                # dABC = self.deltaABC(self.QRT, self.ABC)
                # self.QRT += dQRT
                # self.ABC += dABC
        
                # 2nd runge-kutta
                # dQRT = self.deltaQRT(self.QRT, self.ABC)
                # dABC = self.deltaABC(self.QRT, self.ABC)
                # dQRT2 = self.deltaQRT(self.QRT+dQRT, self.ABC+dABC)
                # dABC2 = self.deltaABC(self.QRT+dQRT, self.ABC+dABC)
                # self.QRT += (dQRT + dQRT2)/2
                # self.ABC += (dABC + dABC2)/2
        
                # 4th runge-kutta
                dQRT = self.deltaQRT(self.QRT, self.ABC, calcmode=self.calcmode)
                dABC = self.deltaABC(self.QRT, self.ABC, calcmode=self.calcmode)
                dQRT2 = self.deltaQRT(self.QRT+dQRT/2, self.ABC+dABC/2, calcmode=self.calcmode)
                dABC2 = self.deltaABC(self.QRT+dQRT/2, self.ABC+dABC/2, calcmode=self.calcmode)
                dQRT3 = self.deltaQRT(self.QRT+dQRT2/2, self.ABC+dABC2/2, calcmode=self.calcmode)
                dABC3 = self.deltaABC(self.QRT+dQRT2/2, self.ABC+dABC2/2, calcmode=self.calcmode)
                dQRT4 = self.deltaQRT(self.QRT+dQRT3, self.ABC+dABC3, calcmode=self.calcmode)
                dABC4 = self.deltaABC(self.QRT+dQRT3, self.ABC+dABC3, calcmode=self.calcmode)
                
                self.total_dQRT = (dQRT + 2*dQRT2 + 2*dQRT3 + dQRT4)/6
                self.total_dABC = (dABC + 2*dABC2 + 2*dABC3 + dABC4)/6
                # これらはupdateperiodが2以上のときは再計算されることなく用いられる

                if isinstance(self.updateperiod, int):
                    self.thisupdateperiod = self.updateperiod
                else:
                    # 上記の更新幅から動的にupdateperiodを決める
                    self.thisupdateperiod = self.updateperiod(self.updatecnt)
        
        self.updateperiodcnt = (self.updateperiodcnt + 1) % self.thisupdateperiod
        self.updatecnt += 1                
        self.QRT += self.total_dQRT
        self.ABC += self.total_dABC
        return
    def deltaABC(self, QRT, ABC, calcmode=''):
        ret = np.zeros((self.K+self.M, self.K+self.M))
        if (calcmode=='RRQQBBAA'): # 8変数系に縮約されている場合の高速な計算(積分項を呼び出す回数が律速のはずなのでそれを減らす)
            for i in range(1):
                for j in range(2):
                    # for p in range(self.M):
                    #     ret[i,j] += self.ABCcoef[0] * self.eta2 * (ABC[j,self.K+p] * self.I2(QRT, i, self.K+p) + ABC[i,self.K+p] * self.I2(QRT, j, self.K+p))
                    if (i==j):
                        ret[i,j] += self.ABCcoef[0] * self.eta2 * 2 * ((self.M-1) * ABC[0,self.K+1] * self.I2(QRT, 0, self.K+1) + ABC[0,self.K] * self.I2(QRT, 0, self.K))
                    else:
                        ret[i,j] += self.ABCcoef[0] * self.eta2 * 2 * (((self.M-2) * ABC[0,self.K+1] + ABC[0,self.K])* self.I2(QRT, 0, self.K+1) + ABC[0,self.K+1] * self.I2(QRT, 0, self.K))
                    # for p in range(self.K):
                    #     ret[i,j] -= self.ABCcoef[0] * self.eta2 * (ABC[j,p] * self.I2(QRT, i, p) + ABC[i,p] * self.I2(QRT, j, p))
                    if (i==j):
                        ret[i,j] -= self.ABCcoef[0] * self.eta2 * 2 * ((self.K-1) * ABC[0,1] * self.I2(QRT, 0, 1) + ABC[0,0] * self.I2(QRT, 0, 0))
                    else:
                        ret[i,j] -= self.ABCcoef[0] * self.eta2 * 2 * (((self.K-2) * ABC[0,1] + ABC[0,0]) * self.I2(QRT, 0, 1) + ABC[0,1] * self.I2(QRT, 0, 0))                   
                    # if (not self.ignore_etasq[1]):
                    #     for p in range(self.K+self.M):
                    #         for q in range(p, self.K+self.M):
                    #             sg = 1 - 2 * xor(p>=self.K, q>=self.K)
                    #             ra = 1 + 1*(p!=q) # こっちはこのように計算を省略できる(p,qについて対称な計算なので)
                    #             ret[i,j] += self.ABCcoef[1] * self.eta2 * self.eta2 * ra * sg * ABC[p, q] * self.I5(QRT, i, j, p, q)
                    assert(self.ignore_etasq[1]) # このケースもう考えないでしょ（実装が面倒なだけですすいません）
                    # if (i==j): ret[i,j] /= 2
                for n in range(2):
                    # for p in range(self.M):
                    #     ret[i,self.K+n] += self.ABCcoef[0] * self.eta2 * ABC[self.K+p, self.K+n] * self.I2(QRT, i, self.K+p)
                    if (i==n):
                        ret[i,self.K+n] += self.ABCcoef[0] * self.eta2 * ((self.M-1) * ABC[self.K, self.K+1] * self.I2(QRT, 0, self.K+1) + ABC[self.K, self.K] * self.I2(QRT, 0, self.K))
                    else:
                        ret[i,self.K+n] += self.ABCcoef[0] * self.eta2 * (((self.M-2) * ABC[self.K, self.K+1] + ABC[self.K, self.K]) * self.I2(QRT, 0, self.K+1) + ABC[self.K, self.K+1] * self.I2(QRT, 0, self.K))
                    # for p in range(self.K):
                    #     ret[i,self.K+n] -= self.ABCcoef[0] * self.eta2 * ABC[p, self.K+n] * self.I2(QRT, i, p)
                    if (i==n):
                        ret[i,self.K+n] -= self.ABCcoef[0] * self.eta2 * ((self.K-1) * ABC[0, self.K+1] * self.I2(QRT, 0, 1) + ABC[0, self.K] * self.I2(QRT, 0, 0))
                    else:
                        ret[i,self.K+n] -= self.ABCcoef[0] * self.eta2 * (((self.K-2) * ABC[0, self.K+1] + ABC[0, self.K]) * self.I2(QRT, 0, 1) + ABC[0, self.K+1] * self.I2(QRT, 0, 0)) 

            dA = np.eye(self.K) * ret[0,0] + (np.ones((self.K,self.K))-np.eye(self.K)) * ret[0,1]
            dB = np.eye(self.K) * ret[0,self.K] + (np.ones((self.K,self.K))-np.eye(self.K)) * ret[0,self.K+1]
            return np.r_[np.c_[dA, dB], np.c_[dB.T, np.zeros((self.M,self.M))]]
        else: # 通常の計算
            for i in range(self.K):
                for j in range(i,self.K):
                    for p in range(self.M):
                        ret[i,j] += self.ABCcoef[0] * self.eta2 * (ABC[j,self.K+p] * self.I2(QRT, i, self.K+p) + ABC[i,self.K+p] * self.I2(QRT, j, self.K+p))
                    for p in range(self.K):
                        ret[i,j] -= self.ABCcoef[0] * self.eta2 * (ABC[j,p] * self.I2(QRT, i, p) + ABC[i,p] * self.I2(QRT, j, p))
                    if (not self.ignore_etasq[1]):
                        for p in range(self.K+self.M):
                            for q in range(p, self.K+self.M):
                                sg = 1 - 2 * xor(p>=self.K, q>=self.K)
                                ra = 1 + 1*(p!=q) # こっちはこのように計算を省略できる(p,qについて対称な計算なので)
                                ret[i,j] += self.ABCcoef[1] * self.eta2 * self.eta2 * ra * sg * ABC[p, q] * self.I5(QRT, i, j, p, q)
                    if (i==j): ret[i,j] /= 2
                for n in range(self.M):
                    for p in range(self.M):
                        ret[i,self.K+n] += self.ABCcoef[0] * self.eta2 * ABC[self.K+p, self.K+n] * self.I2(QRT, i, self.K+p)
                    for p in range(self.K):
                        ret[i,self.K+n] -= self.ABCcoef[0] * self.eta2 * ABC[p, self.K+n] * self.I2(QRT, i, p)
            return ret + ret.T
    def deltaQRT(self, QRT, ABC, calcmode=''):
        ret = np.zeros((self.K+self.M, self.K+self.M))
        if (calcmode=='RRQQBBAA'): # 8変数系に縮約されている場合の高速な計算(積分項を呼び出す回数が律速のはずなのでそれを減らす)
            for i in range(1):
                for k in range(2):
                    # for n in range(self.M):
                    #     ret[i,k] += self.QRTcoef[0] * self.eta * (ABC[i, self.K+n] * self.I3(QRT, i, k, self.K+n) + ABC[k, self.K+n] * self.I3(QRT, k, i, self.K+n))
                    if (i==k):
                        ret[i,k] += self.QRTcoef[0] * self.eta * 2 * ((self.M-1) * ABC[0, self.K+1] * self.I3(QRT, 0, 0, self.K+1) + ABC[0, self.K] * self.I3(QRT, 0, 0, self.K))
                    else:
                        ret[i,k] += self.QRTcoef[0] * self.eta * 2 * ((self.M-2) * ABC[0, self.K+1] * self.I3(QRT, 0, 1, self.K+2) + ABC[0, self.K+1] * self.I3(QRT, 0, 1, self.K+1) + ABC[0, self.K] * self.I3(QRT, 0, 1, self.K))
                        # M==2の場合死にそう
                    # for j in range(self.K):
                    #     ret[i,k] -= self.QRTcoef[0] * self.eta * (ABC[i, j] * self.I3(QRT, i, k, j) + ABC[k, j] * self.I3(QRT, k, i, j))
                    if (i==k):
                        ret[i,k] -= self.QRTcoef[0] * self.eta * 2 * ((self.K-1) * ABC[0, 1] * self.I3(QRT, 0, 0, 1) + ABC[0, 0] * self.I3(QRT, 0, 0, 0))
                    else:
                        ret[i,k] -= self.QRTcoef[0] * self.eta * 2 * ((self.K-2) * ABC[0, 1] * self.I3(QRT, 0, 1, 2) + ABC[0, 1] * self.I3(QRT, 0, 1, 1) + ABC[0, 0] * self.I3(QRT, 0, 1, 0))
                        # K==2の場合死にそう
                    # if (not self.ignore_etasq[0]):
                    #     for p in range(self.K+self.M):
                    #         for q in range(self.K+self.M):
                    #             sg = 1 - 2 * xor(p>=self.K, q>=self.K) # こっちは上のように計算を省略はできない．
                    #             xinorm2 = np.linalg.norm(self.inputdata)**2 if (self.deterministic) else self.N
                    #             ret[i,k] += sg * self.QRTcoef[1] * self.eta * self.eta * xinorm2 * ABC[i,p] * ABC[k,q] * self.I4(QRT, i, k, p, q)
                    assert(self.ignore_etasq[0]) # すいません実装が面倒（このケースは考えうるわけだが・・・）
                    # if (i==k): ret[i,k] /= 2
                for n in range(2):
                    # for m in range(self.M):
                    #     ret[i,self.K+n] += self.QRTcoef[0] * self.eta * ABC[i, self.K+m] * self.I3(QRT, i, self.K+n, self.K+m)
                    if (i==n):
                        ret[i,self.K+n] += self.QRTcoef[0] * self.eta * ((self.M-1) * ABC[0, self.K+1] * self.I3(QRT, 0, self.K, self.K+1) + ABC[0, self.K] * self.I3(QRT, 0, self.K, self.K))
                    else:
                        ret[i,self.K+n] += self.QRTcoef[0] * self.eta * ((self.M-2) * ABC[0, self.K+1] * self.I3(QRT, 0, self.K+1, self.K+2) + ABC[0, self.K+1] * self.I3(QRT, 0, self.K+1, self.K+1) + ABC[0, self.K] * self.I3(QRT, 0, self.K+1, self.K))
                        # M==2の場合死にそう
                    # for j in range(self.K):
                    #     ret[i,self.K+n] -= self.QRTcoef[0] * self.eta * ABC[i, j] * self.I3(QRT, i, self.K+n, j)
                    if (i==n):
                        ret[i,self.K+n] -= self.QRTcoef[0] * self.eta * ((self.K-1) * ABC[0, 1] * self.I3(QRT, 0, self.K, 1) + ABC[0, 0] * self.I3(QRT, 0, self.K, 0))
                    else:
                        ret[i,self.K+n] -= self.QRTcoef[0] * self.eta * ((self.K-2) * ABC[0, 1] * self.I3(QRT, 0, self.K+1, 2) + ABC[0, 1] * self.I3(QRT, 0, self.K+1, 1) + ABC[0, 0] * self.I3(QRT, 0, self.K+1, 0))
                        # K==2の場合死にそう
            dQ = np.eye(self.K) * ret[0,0] + (np.ones((self.K,self.K))-np.eye(self.K)) * ret[0,1]
            dR = np.eye(self.K) * ret[0,self.K] + (np.ones((self.K,self.K))-np.eye(self.K)) * ret[0,self.K+1]
            return np.r_[np.c_[dQ, dR], np.c_[dR.T, np.zeros((self.M,self.M))]]
        else: # 通常の計算
            for i in range(self.K):
                for k in range(i,self.K):
                    for n in range(self.M):
                        ret[i,k] += self.QRTcoef[0] * self.eta * (ABC[i, self.K+n] * self.I3(QRT, i, k, self.K+n) + ABC[k, self.K+n] * self.I3(QRT, k, i, self.K+n))
                    for j in range(self.K):
                        ret[i,k] -= self.QRTcoef[0] * self.eta * (ABC[i, j] * self.I3(QRT, i, k, j) + ABC[k, j] * self.I3(QRT, k, i, j))
                    if (not self.ignore_etasq[0]):
                        for p in range(self.K+self.M):
                            for q in range(self.K+self.M):
                                sg = 1 - 2 * xor(p>=self.K, q>=self.K) # こっちは上のように計算を省略はできない．
                                xinorm2 = np.linalg.norm(self.inputdata)**2 if (self.deterministic) else self.N
                                ret[i,k] += sg * self.QRTcoef[1] * self.eta * self.eta * xinorm2 * ABC[i,p] * ABC[k,q] * self.I4(QRT, i, k, p, q)
                    if (i==k): ret[i,k] /= 2
                for n in range(self.M):
                    for m in range(self.M):
                        ret[i,self.K+n] += self.QRTcoef[0] * self.eta * ABC[i, self.K+m] * self.I3(QRT, i, self.K+n, self.K+m)
                    for j in range(self.K):
                        ret[i,self.K+n] -= self.QRTcoef[0] * self.eta * ABC[i, j] * self.I3(QRT, i, self.K+n, j)
            return ret + ret.T # /self.N
    def Q(self):
        return self.QRT[:self.K,:][:,:self.K]
    def R(self):
        return self.QRT[:self.K,:][:,self.K:]
    def T(self):
        return self.QRT[:,self.K:][:,self.K:]
    def A(self):
        return self.ABC[:self.K,:][:,:self.K]
    def B(self):
        return self.ABC[:self.K,:][:,self.K:]
    def C(self):
        return self.ABC[:,self.K:][:,self.K:]
    def calceps(self, QRT, ABC):
        ret = 0.0
        for p in range(self.K+self.M):
            for q in range(p, self.K+self.M):
                sg = 1 - 2 * xor(p>=self.K, q>=self.K)
                ra = 1 + 1*(p!=q)
                ret += 0.5 * sg * ra * ABC[p, q] * self.I2(QRT, p, q)
        return ret
    def eps(self):
        return self.calceps(self.QRT, self.ABC)


class opd_inf_KM_inf_highorder:
    '''
    2次まで見る．
    
    Σ の固有値 N 個を lambdas で与える．

    '''
    def __init__(self, lambdas, QRT, ABC, K=[], M=[], O=[], eta=0.01, eta2=[], poserf=False, ignore_etasq=[False,True], etasq_full=True, act='erf',updateperiod=1, calcmode='', QRTcoef=[1, 1], ABCcoef=[1], substep=1):
        '''
        Q[e-1] := J Σ^e J
        R[e-1] := J Σ^e B
        T[e-1] := B Σ^e B

        e = 1, 2, 3, ..., deg
        (deg := 相異なる非ゼロ固有値の個数 = 最小多項式の次数)

        A = W W
        B = W V
        C = V V

        eta は N で割ってないものを与えてください．


        ==============================
        args:
            substep (int) : positive integer, 刻み幅 1/substep イテレーションで微分方程式を解きます
        '''
        self.lambdas = lambdas
        self.nonzero_distinct_lambdas = np.unique(lambdas[np.nonzero(lambdas)]) # 非ゼロの相異なる固有値
        self.deg = len(self.nonzero_distinct_lambdas) # その種類数．
        assert(etasq_full == True)
        assert(self.deg <= 2) # いずれ外せるかも．
        assert(np.min(lambdas) >= 0) # Σ は正定値であるべき．
        assert(QRT.shape == (self.deg, K+M, K+M))
        assert(ABC.shape == (K+M, K+M))
        assert(substep==1 or updateperiod==1)

        # Σ の固有多項式（の係数）
        
        if self.deg == 2:
            self.poly = [np.prod(self.nonzero_distinct_lambdas), -np.sum(self.nonzero_distinct_lambdas), 1]
        elif self.deg == 1:
            self.poly = [-self.nonzero_distinct_lambdas[0], 1]
        assert(len(self.poly) == self.deg + 1)
        print('poly deg:', self.deg)

        # 固有値モーメント(e乗平均)の計算
        self.lambda_moment = np.zeros(self.deg + 2)
        for e in range(len(self.lambda_moment)):
            self.lambda_moment[e] = np.mean(self.lambdas ** e)


        self.QRT = QRT.copy()
        self.ABC = ABC.copy()
        self.K = K
        self.M = M
        self.eta = eta
        self.eta2 = eta2
        if (self.eta2==[]): self.eta2 = eta
        self.poserf = poserf
        self.etasq_full = etasq_full
        # self.etaN = N * eta
        if act=='erf':
            self.act = act_erf()
        elif act=='relu':
            self.act = act_relu()
        elif act=='lin':
            self.act = act_lin()
        elif act=='exp':
            self.act = act_exp()
        elif act=='id':
            self.act = act_id()
        else:
            raise ValueError
        self.deterministic = False
        self.ignore_etasq = ignore_etasq
        self.updateperiod = updateperiod
        self.updateperiodcnt = 0
        self.updatecnt = 0
        self.calcmode = calcmode
        self.QRTcoef = QRTcoef
        self.ABCcoef = ABCcoef
        self.substep = substep
    def I2(self, QRT, i, j):
        if (self.deterministic):
            return self.act.gx1gx2_(x=self.hiddenvoltage[[i,j]].reshape(1,2))
        else:
            return self.act.gx1gx2(QRT[[i,j],:][:,[i,j]])
    def I3_one_e_one(self, QRT_one, QRT_eplusone, i, j, k):
        if (self.deterministic):
            raise NotImplementedError
        else:
            mat = np.copy(QRT_eplusone[[i,j,k],:][:,[i,j,k]])
            mat[0,0] = QRT_one[i,i]
            mat[0,2] = QRT_one[i,k]
            mat[2,0] = QRT_one[k,i]
            mat[2,2] = QRT_one[k,k]
            return self.act.gpx1x2gx3(mat)
    def I4(self, QRT, i, j, k, l):
        if (self.deterministic):
            return self.act.gpx1gpx2gx3gx4_(x=self.hiddenvoltage[[i,j,k,l]].reshape(1,4))
        else:      
            return self.act.gpx1gpx2gx3gx4(QRT[[i,j,k,l],:][:,[i,j,k,l]])        
    def update(self, deterministic=False, hiddenvoltage=[], inputdata=[]):
        for ss in range(self.substep):
            if self.updateperiod==np.inf: break # infの場合は、そもそもアップデートしない。（特殊処理）（これ書かないと、初回の更新幅で未来永劫更新する挙動となる）
            if self.updateperiodcnt == 0:
                if deterministic:
                    raise NotImplementedError
                else:
                    self.deterministic = False
            
                    # 4th runge-kutta
                    dQRT = self.deltaQRT(self.QRT, self.ABC, calcmode=self.calcmode)
                    dABC = self.deltaABC(self.QRT, self.ABC, calcmode=self.calcmode)
                    dQRT2 = self.deltaQRT(self.QRT+dQRT/2, self.ABC+dABC/2, calcmode=self.calcmode)
                    dABC2 = self.deltaABC(self.QRT+dQRT/2, self.ABC+dABC/2, calcmode=self.calcmode)
                    dQRT3 = self.deltaQRT(self.QRT+dQRT2/2, self.ABC+dABC2/2, calcmode=self.calcmode)
                    dABC3 = self.deltaABC(self.QRT+dQRT2/2, self.ABC+dABC2/2, calcmode=self.calcmode)
                    dQRT4 = self.deltaQRT(self.QRT+dQRT3, self.ABC+dABC3, calcmode=self.calcmode)
                    dABC4 = self.deltaABC(self.QRT+dQRT3, self.ABC+dABC3, calcmode=self.calcmode)
                    
                    self.total_dQRT = (dQRT + 2*dQRT2 + 2*dQRT3 + dQRT4)/6
                    self.total_dABC = (dABC + 2*dABC2 + 2*dABC3 + dABC4)/6
                    # これらはupdateperiodが2以上のときは再計算されることなく用いられる

                    if isinstance(self.updateperiod, int):
                        self.thisupdateperiod = self.updateperiod
                    else:
                        # 上記の更新幅から動的にupdateperiodを決める
                        self.thisupdateperiod = self.updateperiod(self.updatecnt)
            
            self.updateperiodcnt = (self.updateperiodcnt + 1) % self.thisupdateperiod
            self.updatecnt += 1                
            self.QRT += self.total_dQRT
            self.ABC += self.total_dABC
        return
    def deltaABC(self, QRT, ABC, calcmode=''):
        ret = np.zeros((self.K+self.M, self.K+self.M))
        if (calcmode=='RRQQBBAA'): # 8変数系に縮約されている場合の高速な計算(積分項を呼び出す回数が律速のはずなのでそれを減らす)
            raise NotImplementedError
        else: # 通常の計算
            assert(np.sum(np.array(self.ABCcoef) ** 2) == 0) # 係数が全部ゼロを仮定する（未実装なので・・・）
            return np.zeros(ABC.shape) / self.substep
    def deltaQRT(self, QRT, ABC, calcmode=''):
        ret = np.zeros((self.deg, self.K+self.M, self.K+self.M))
        if (calcmode=='RRQQBBAA'): # 8変数系に縮約されている場合の高速な計算(積分項を呼び出す回数が律速のはずなのでそれを減らす)
            raise NotImplementedError
        else: # 通常の計算
            # まず，固有多項式を使って，QRT[self.deg] を QRT に付け足す．
            QRT_highest = np.zeros((self.K+self.M, self.K+self.M)).astype(np.float)
            for h in range(self.deg):
                QRT_highest += self.poly[h] * QRT[h]
            QRT_highest /= -self.poly[self.deg] # この計算はあってそう．
            # 付け足す！
            QRT = np.r_[QRT, [QRT_highest]]

            for h in range(self.deg):
                e = h + 1 # e次のオーダパラメータ（* Σ^e *）の delta の計算を行う．
                eplusone_idx = e
                one_idx = 0
                # QRTのインデクスは [eplusone_idx]: e+1次, [one_idx]: 1次
                for i in range(self.K):
                    for k in range(i,self.K):
                        for n in range(self.M):
                            ret[h,i,k] += self.QRTcoef[0] * self.eta * (
                                ABC[i, self.K+n] * self.I3_one_e_one(QRT[one_idx], QRT[eplusone_idx], i, k, self.K+n)
                                + ABC[k, self.K+n] * self.I3_one_e_one(QRT[one_idx], QRT[eplusone_idx], k, i, self.K+n)
                            )
                        for j in range(self.K):
                            ret[h,i,k] -= self.QRTcoef[0] * self.eta * (
                                ABC[i, j] * self.I3_one_e_one(QRT[one_idx], QRT[eplusone_idx], i, k, j)
                                + ABC[k, j] * self.I3_one_e_one(QRT[one_idx], QRT[eplusone_idx], k, i, j)
                            )
                        if not self.ignore_etasq[0]:
                            for p in range(self.K+self.M):
                                for q in range(self.K+self.M):
                                    sg = 1 - 2 * xor(p>=self.K, q>=self.K) # 計算を省略はできない．
                                    xinorm2 = self.lambda_moment[e+1]
                                    ret[h,i,k] += sg * self.QRTcoef[1] * self.eta * self.eta * xinorm2 * ABC[i,p] * ABC[k,q] * self.I4(QRT[one_idx], i, k, p, q)
                        if i==k: ret[h,i,k] /= 2
                    for n in range(self.M):
                        for m in range(self.M):
                            ret[h,i,self.K+n] += self.QRTcoef[0] * self.eta * ABC[i, self.K+m] * self.I3_one_e_one(QRT[one_idx], QRT[eplusone_idx], i, self.K+n, self.K+m)
                        for j in range(self.K):
                            ret[h,i,self.K+n] -= self.QRTcoef[0] * self.eta * ABC[i, j] * self.I3_one_e_one(QRT[one_idx], QRT[eplusone_idx], i, self.K+n, j)
            return (ret + ret.swapaxes(1, 2)) / self.substep
    def Q(self, order):
        return self.QRT[order][:self.K,:][:,:self.K]
    def R(self, order):
        return self.QRT[order][:self.K,:][:,self.K:]
    def T(self, order):
        return self.QRT[order][:,self.K:][:,self.K:]
    def A(self, order):
        return self.ABC[order][:self.K,:][:,:self.K]
    def B(self, order):
        return self.ABC[order][:self.K,:][:,self.K:]
    def C(self, order):
        return self.ABC[order][:,self.K:][:,self.K:]
    def calceps(self, QRT_one, ABC): # この計算はあってそう．
        ret = 0.0
        for p in range(self.K+self.M):
            for q in range(p, self.K+self.M):
                sg = 1 - 2 * xor(p>=self.K, q>=self.K)
                ra = 1 + 1*(p!=q)
                ret += 0.5 * sg * ra * ABC[p, q] * self.I2(QRT_one, p, q)
        return ret
    def eps(self):
        return self.calceps(self.QRT[0], self.ABC)        






class opd_inf_1: # 2層非線形，素子数は ∞ - 1
    def __init__(self, QRT, N=[], eta=0.01, ignore_etasq=False, act='erf', updateperiod=1, calcmode='', coef=[1, 1]):
        assert(QRT.shape[0]==2 and QRT.shape[1]==2)
        assert(len(coef) == 2)
        self.QRT = QRT.copy()
        self.N = N
        if self.N==[]: self.N = 1 # オーダパラメータだけ回す場合とかはこの設定で良い．
        self.eta = eta
        # self.etaN = N * eta
        if act=='erf':
            self.act = act_erf()
        elif act=='relu':
            self.act = act_relu()
        elif act=='lin':
            self.act = act_lin()
        elif act=='exp':
            self.act = act_exp()
        elif act=='id':
            self.act = act_id()
        else:
            raise ValueError
        self.deterministic = False
        self.ignore_etasq = ignore_etasq
        self.updateperiod = updateperiod
        self.updateperiodcnt = 0
        self.updatecnt = 0
        self.calcmode = calcmode
        self.coef = coef
    def I2(self, QRT, i, j):
        if (self.deterministic):
            return self.act.gx1gx2_(x=self.hiddenvoltage[[i,j]].reshape(1,2))
        else:
            return self.act.gx1gx2(QRT[[i,j],:][:,[i,j]])
    def I3(self, QRT, i, j, k):
        if (self.deterministic):
            return self.act.gpx1x2gx3_(x=self.hiddenvoltage[[i,j,k]].reshape(1,3))
        else:        
            return self.act.gpx1x2gx3(QRT[[i,j,k],:][:,[i,j,k]])
    def I4(self, QRT, i, j, k, l):
        if (self.deterministic):
            return self.act.gpx1gpx2gx3gx4_(x=self.hiddenvoltage[[i,j,k,l]].reshape(1,4))
        else:      
            return self.act.gpx1gpx2gx3gx4(QRT[[i,j,k,l],:][:,[i,j,k,l]])        
    def I5(self, QRT, i, j, k, l):
        if (self.deterministic):
            return self.act.gx1gx2gx3gx4(x=self.hiddenvoltage[[i,j,k,l]].reshape(1,4))
        else:      
            return self.act.gx1gx2gx3gx4(QRT[[i,j,k,l],:][:,[i,j,k,l]])        
    def update(self, deterministic=False, hiddenvoltage=[], inputdata=[]):
        if self.updateperiod==np.inf: return # infの場合は、そもそもアップデートしない。（特殊処理）（これ書かないと、初回の更新幅で未来永劫更新する挙動となる）
        if self.updateperiodcnt == 0:
            if deterministic:
                raise NotImplementedError
                self.deterministic = True
                assert(hiddenvoltage.shape[1]==1)
                self.hiddenvoltage = hiddenvoltage.flatten()
                assert(hiddenvoltage.shape[0]==self.K+self.M)
                self.inputdata = inputdata.flatten()
                # euler
                self.total_dQRT = self.deltaQRT(self.QRT)
                # これらはupdateperiodが2以上のときは再計算されることなく用いられる
            else:
                self.deterministic = False
                # euler
                # dQRT = self.deltaQRT(self.QRT)
                # self.QRT += dQRT
        
                # 2nd runge-kutta
                # dQRT = self.deltaQRT(self.QRT)
                # dQRT2 = self.deltaQRT(self.QRT+dQRT)
                # self.QRT += (dQRT + dQRT2)/2
        
                # 4th runge-kutta
                dQRT = self.deltaQRT(self.QRT, calcmode=self.calcmode)
                dQRT2 = self.deltaQRT(self.QRT+dQRT/2, calcmode=self.calcmode)
                dQRT3 = self.deltaQRT(self.QRT+dQRT2/2, calcmode=self.calcmode)
                dQRT4 = self.deltaQRT(self.QRT+dQRT3, calcmode=self.calcmode)
                
                self.total_dQRT = (dQRT + 2*dQRT2 + 2*dQRT3 + dQRT4)/6
                # これらはupdateperiodが2以上のときは再計算されることなく用いられる

                if isinstance(self.updateperiod, int):
                    self.thisupdateperiod = self.updateperiod
                else:
                    # 上記の更新幅から動的にupdateperiodを決める
                    self.thisupdateperiod = self.updateperiod(self.updatecnt)
        
        self.updateperiodcnt = (self.updateperiodcnt + 1) % self.thisupdateperiod
        self.updatecnt += 1                
        self.QRT += self.total_dQRT
        return

    def deltaQRT(self, QRT, calcmode=''):
        ret = np.zeros([2, 2])
        ret[0,0] = self.coef[0] * 2 * self.eta * (self.I3(QRT, 0, 0, 1) - self.I3(QRT, 0, 0, 0))
        ret[0,1] = self.coef[0] * self.eta * (self.I3(QRT, 0, 1, 1) - self.I3(QRT, 0, 1, 0))
        if not self.ignore_etasq:
            ret[0,0] += self.coef[1] * self.eta ** 2 * (
                    self.I4(QRT, 0, 0, 0, 0)
                    -2 * self.I4(QRT, 0, 0, 0, 1)
                    + self.I4(QRT, 0, 0, 1, 1)
                )

        ret[0,0] /= 2
        return ret + ret.T # /self.N

    def Q(self):
        return self.QRT[0,0]
    def R(self):
        return self.QRT[0,1]
    def T(self):
        return self.QRT[1,1]
    def calceps(self, QRT):
        return 0.5 * (self.I2(QRT, 0, 0) - 2 * self.I2(QRT, 0, 1) + self.I2(QRT, 1, 1))
    def eps(self):
        return self.calceps(self.QRT)



if __name__=='__main__':
    average_check()
