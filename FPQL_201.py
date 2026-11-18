import sys
import io
import os
import datetime
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from scipy import special, linalg
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
from scipy.integrate import simpson, trapezoid
from scipy.special import legendre
from scipy.linalg import solve_banded
from scipy.optimize import brentq

import math
import time


# from plot_f_velocity_average_pitch_angle_D import plot_f_velocity_average_pitch_angle


# 将标准输出流设置为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ===================================================================
# 模拟外部依赖环境 (Mocking External Modules)
# ===================================================================
class ExternalContext:
    """
    这个类用于模拟原Fortran代码中引用的外部模块变量 (t4_mod_public, t4_mod_equil等).
    在实际使用中，你需要将真实数据填充到这里。
    """
    def __init__(self, nspec, tempi, azi, atm, aconc,dens ):
        # 物理常数
        self.vlight = 2.9979e10
        self.pi = np.pi
        self.rmhme = 1836.15  # 质子电子质量比
        
        # 模拟配置
        self.nspec = nspec  # 离子种类数量
        self.nphi = 0      #n//
        self.baxis = 30.0 # 轴磁场
        self.raxis = 100.0 # 大半径
        self.ukzero = 1.0
        self.xomce = 1.0
        self.ompe2 = 1.0

        # 未知参数
        self.psiedg = 1             # 已追溯
        self.xedg_out = 100.0
        self.ulengt = 100.0
        self.ux = 1.0
        
        # 数组占位符 (需要根据实际网格大小初始化)
        self.n_order = np.array([1])
        self.n_phi_val = np.array([10])
        self.tabpw = np.array([1.0])
        
        
        # 剖面数据占位符
        self.tempi = tempi            #[离子, 离子， 电子]) # 温度
        self.tempec = self.tempi[2]   # 电子中心温度
        self.tempe = self.tempi[2]    # 电子温度
        self.dens = dens              # 电子密度
        
        
        # 种类属性
        self.azi = azi  #np.array([1.0, 2.0, -1.0]) # 电荷数 (最后是电子)
        self.atm = atm  #np.array([2.0, 4.0, 1.0/1836.15]) # 质量数
        self.aconc = aconc  #np.array([0.9, 0.1, 1.0]) # 浓度
        print(f"粒子浓度:{self.aconc}")
        
        # 密度
        self.dens = dens              # 电子密度
        self.xfni = np.array([(dens*aconc[0]), (dens*aconc[1])]) # 离子密度
        print(f"离子密度:{self.xfni}")

        # 功率沉积剖面占位符
        self.pwpsif = np.zeros((self.nspec, 20))  # 基频功率
        self.pwpsih = np.zeros((self.nspec, 20)) # 谐频功率 power
        self.pwpelt = np.zeros(20) # 电子功率
        
        # 共振位置
        self.psfund = np.array([0.5, 0.5])
        self.psharm = np.array([0.5, 0.5])
        self.rcfund = np.array([100.0, 100.0])
        self.rcharm = np.array([100.0, 100.0])
        
        # 回旋频率
        self.xomci = np.array([1.0, 0.5])
        self.ompi2 = np.array([1.0, 1.0])
        self.ohi = np.zeros(2)

    def profstart(self, name):
        print(f"开始记录: {name}")

    def profend(self, name):
        print(f"结束记录: {name}")

# ===================================================================
# 主求解器类
# ===================================================================

class FP:
    def __init__(self, ctx: ExternalContext, nptu, ipheat, legmod, kperp, omega_ci, power, ispec, nbi_enabled, icrh_enabled, Ns, mu_min= -1.0, mu_max=1.0 ):
        self.ctx = ctx

        # 输出文件名字
        self.out = None
        self.file_list = None
        # 当前运行模式的输出目录
        self.current_out_dir = "."  # 默认当前目录，会被主程序覆盖
        # 输出文件路径字典
        self.output_paths = {}  # 默认为空，由主程序覆盖

        # 特定径向位置数组
        self.ipsii = None
        self.ipsiii = None

        # 径向信息
        self.info = None
        self.positions = None
        self.arrays = None

        # 环向捕获效应
        self.enable_trapping = False    # 1：开启 0：关闭
        self.trapping_model = "symmetric"  # 经验共振可达权重；directional 仅用于历史对照
        self.trapping_ucr = 0.4         # 捕获效应 Sigmoid 权重的临界俯仰角 uc
        self.trapping_a = 20.0          # 捕获效应 Sigmoid 权重的陡度参数 a
        self.trapping_model = 'symmetric'
        self.frequency_convention = 'rf_hz'
        self.rf_power_mode = 'absorbed'
        self.power_tolerance = 1e-7
        self.plot_enabled = True

        # 网格设置
        self.r_E_1 = 100.0 # 基频加热模式的速度范围系数，乘以温度就是能量范围
        self.r_E_2 = 30.0 # 基频加热模式的速度范围系数，乘以温度就是能量范围
        
        # 绘图设置
        self.plot_range_k = 0    # 设置绘图范围，1为限定范围，0为全范围
        self.plot_type = 0       # 0为f(v,u)速度分布图，1为f(v,u)热图显示，2为f(v,u1)μ平均速度分布图
        self.plot_range_mu = 0   # 0: 显示完整区间 [-1, 1]，且负区间为虚线 (默认) 1: 仅显示正区间 [0, 1] -1: 仅显示负区间 [-1, 0]
        self.xlim_max = 100      # 设置限定范围的x轴最大值，单位为keV
        self.ylim_max = -6       # 设置限定范围的y轴最大值，单位为log10(f)，是负数
        self.f_max = 1           # 数据最大阈值
        self.f_min = 1e-6        # 数据最小阈值
        self.duu = 0.1           # pitch angle步长
        # 自定义显示的 pitch angle (mu) 列表
        # 如果设为 None，则退回到使用 self.duu 等间距步长的默认逻辑
        self.custom_mu_values = None 

        # 新增：用于存储高能尾特征位置的变量
        self.nbi_tail_E = None
        self.nbi_tail_mu = None
        self.nbi_tail_mu_pos = None
        self.nbi_tail_E_pos = None
        
        # 【新增】：ICRH 单独加热的标记线变量
        self.icrh_tail_E = None
        self.icrh_tail_mu = None
        self.icrh_tail_mu_pos = None
        self.icrh_tail_E_pos = None

        # 用于存储高能尾特征 pitch angle 的变量
        self.nbi_tail_mu_pos = None      # 单独NBI高能尾的特征mu位置
        self.co_tail_mu_pos = None       # 协同加热高能尾的特征mu位置

        self.nbi_f_processed = None  # 用于存储NBI模式下的径向归一化密度分布n_i_NBI(μ，E)
        self.icrh_f_processed = None  # 用于存储ICRH模式下的径向归一化密度分布n_i_ICRH(μ，E)

        # Public variables from module
        self.iex_sumnph = 0
        self.joutql = 0
        self.nrdpsi = 0
        self.npsihf = 0
        self.iwrdql = 1
        self.legmod = legmod
        # Input variables
        self.nptu = nptu
        self.ipheat = ipheat
        self.kperp = kperp
        self.omega_ci = omega_ci
        self.frequency_convention = "rf_hz"
        self.power = power
        self.ispec = ispec
        
        # Internal Mesh and Physics variables
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.nptu = nptu
        self.nmu = 0
        
        self.ksup = 0
        self.kamax = 0
        self.umax = 0.0
        self.du = 0.0
        self.drdpsi = 0.0
        self.nspele = 0
        self.ispres = 0  # 第i种离子
        self.ipheat = ipheat

        self.f_cval = 0.1
        
        # Arrays (Allocatables in Fortran)
        self.tbu = None
        self.tbkev = None
        self.tbfu = None
        self.tbdfu = None
        self.tbdqlk = None
        self.redpsi = None
        self.redxoa = None
        self.redvol = None
        self.pwredl = None
        self.pwredf = None
        self.pwredh = None
        self.zdeni = None
        self.ztempi = None
        
        # Legendre and Matrix specific
        self.pwcoll = np.zeros(2)
        self.qleg00 = None
        self.qleg01 = None
        self.qleg11 = None
        self.fmatrx = None # Dense matrix for Python solver
        self.fb = None
        self.fb_store = None
        
        # Constants
        self.itrmax = 10
        self.accuracy_goal = 1.e-6
        self.fb1_mxw = 0.179587122125
        self.fb1_prv = 0.0
        
        # State control
        self.ibreak = 0
        self.psinxt = 0.0
        self.dpsxnt = 0.0
        self.fnorm = 0.0
        self.zbalpw = 0.0
        self.hrate = 0.0
        self.pwdens = 0.0
        self.qlnorm = 0.0
        self.dqlw = 0.0
        self.xiperp = 0.0
        self.psourc = 0.0   # 临时源项
        self.psource = None # 源项
        self.anuref = 0.0
        self.zconv_nbi = 0.0
        
        # Temporary/Auxiliary arrays
        self.fjbess = None
        self.fctinv = None
        self.auxfun = None
        
        # Output arrays
        self.thrate = None
        self.thtprp = None
        self.thtpar = None
        self.thttot = None
        self.tmtail = None
        self.tavgtx = None
        self.tavgtz = None
        self.tavbkx = None
        self.tavbkz = None
        self.pwgain = None
        self.pwexch = None
        self.tab_pwexch = None
        
        # Matrix construction arrays
        self.nfvrb = 0
        self.nfdim = 0
        self.nfelem = 0
        self.ngaus = 0
        self.zgaus = None
        self.wgaus = None
        self.hl = None
        self.hr = None
        self.fgamm = None
        self.fdiap = None
        self.fdiam = None
        self.fmixl = None
        self.fmixr = None
        self.uvp = None
        self.uvm = None

        # 扩散系数
        self.xstep = np.float64(0)

        self.njbess = np.float64(0)

        self.ttailx = np.float64(0)
        self.ttailz = np.float64(0)
        self.ttblkx = np.float64(0)
        self.ttblkz = np.float64(0)

 
        self.zfb1 = np.zeros(2)
        self.ztpw = np.zeros(2)
        
        # NBI源参数

        self.nbi_enabled = nbi_enabled
        self.icrh_enabled = icrh_enabled
        
        self.Ns = Ns # NBI光束数量
        self.nbi_energies_keV  = None # NBI光束能量，单位keV
        self.nbi_file = None # 保存 NBI 数据文件路径
        self.nbi_range_mu = 1 # NBI 功率分布在 μ 上的区间标志：1: 正半区间 [0,1](默认)，-1: 负半区间 [-1,0]，0: 完整区间 [-1,1] 
        self.nbi_power_file  = None # NBI功率剖面文件路径 (全径向点)，如果提供则覆盖单点文件
        # 【新增】：NBI 物理能量展宽参数
        # 能量高斯展宽的标准差比例，例如 0.01 表示 1% 的能量展宽
        # 对于 90keV 束，1% 意味着 sigma_E = 0.9 keV
        self.nbi_energy_spread_ratio = 0.0  # 默认为 0（无物理展宽，仅数值展宽）

        # 高斯测试分布的控制参数
        self.nbi_mu_center = 0.0  # 高斯中心 (注入角度)
        self.nbi_sigma = 0.0          # 高斯宽度 (束的角向展宽)
        self.nbi_mu_points = 0  # 数据文件的 mu 点数 (用于自适应解析)
        

        # self.vnbi = None # v(i,k) NBI第k束，对应离子i的粒子速度
        self.nbi_in = None # S(i,k,μ,ψ) 功率沉积分布函数

        self.v_nbi_k = np.zeros(self.Ns) # v(k) 粒子种类为i，对应的粒子速度
        self.E_nbi_k = np.zeros(self.Ns) # E(k) 粒子种类为i，对应的粒子能量
        self.S_nbi_ku = None # S(k,u) 磁面位置ψ、粒子种类为i，对应的功率沉积分布函数

        self.sig = None # σ(k,n) 粒子功率沉积分布函数S(k,μ)勒让德展开后整合的函数

        self.source_nbi = None # S(k,n) NBI源项

        # NBI源功率输入与损耗
        self.pwnbi = 0.0    # NBI光束电离引起的功率密度
        self.pwnbi_l = 0.0  # NBI光束电离对应的功率损耗
        # NBI 目标功率密度 (W/cm^3)
        # 这个值必须根据你的实验参数计算：总功率 / 等离子体体积
        # 例如：5 MW = 5e6 W，体积 1e7 cm^3，则密度为 0.5 W/cm^3
        self.pwdens_nbi = 1  # NBI功率密度 (W/cm^3)，根据实际情况调整
        self.dnbi = 1.0        # NBI 源项缩放因子 (归一化系数)
        
    def t_FP(self, power_in, nrdpsi_in, legmod_in, nptu_in, joutql_in=1):
        """
        Main public routine.
        """
        self.power = power_in
        self.nrdpsi = nrdpsi_in
        self.legmod = legmod_in
        self.nptu = nptu_in
        self.joutql = joutql_in
        
        self.ctx.profstart('3Dfpql')

        # 清空文件内容
        if "read" in self.output_paths:
            with open(self.output_paths["read"], 'w', encoding='utf-8') as f: pass
        if "case" in self.output_paths:
            with open(self.output_paths["case"], 'w', encoding='utf-8') as f: pass
            
        # 写入输入参数
        if "read" in self.output_paths:
            append_to_file(self.output_paths["read"], f"==== Read ====\n")
            append_to_file(self.output_paths["read"], f"速度网格点数 nptu = {self.nptu}\n")
            # ... 后续所有的 append_to_file(file_list[0], ...) 都改为 append_to_file(self.output_paths["read"], ...)

        # 写入输入参数
        append_to_file(self.output_paths["read"], f"==== Read ====\n")
        append_to_file(self.output_paths["read"], f"速度网格点数 nptu = {self.nptu}\n")
        append_to_file(self.output_paths["read"], f"勒让德阶数 legmod = {self.legmod}\n")
        append_to_file(self.output_paths["read"], f"运行模式 run_modes = {getattr(self, 'run_modes', '未设置')}\n")
        append_to_file(self.output_paths["read"], f"径向网格点数 nrdpsi = {self.nrdpsi}\n")
        append_to_file(self.output_paths["read"], f"K⊥ kperp = {self.kperp}\n")
        append_to_file(self.output_paths["read"], f"波频率 omega_ci = {self.omega_ci}\n")
        append_to_file(self.output_paths["read"], f"粒子种类数量 nspec = {self.ctx.nspec}\n")
        append_to_file(self.output_paths["read"], f"输出粒子情况 ispec = {self.ispec}\n")
        append_to_file(self.output_paths["read"], f"电荷数组分 azi = {self.ctx.azi}\n")
        append_to_file(self.output_paths["read"], f"质量组分 atm = {self.ctx.atm}\n")

        append_to_file(self.output_paths["read"], f"输出控制 joutql = {self.joutql}\n")
        # append_to_file(self.output_paths["read"], f"iex_sumnph = {self.iex_sumnph}\n")

        append_to_file(self.output_paths["case"],
            f"{' '*2}variables=r(ra),向所有离子碰撞转移功率(MW/m^3),向电子碰撞转移功率(MW/m^3),"
            f"二阶矩温度(keV),二阶矩温度⊥(keV),二阶矩温度∥(keV),高能尾温度⊥(keV)\n"
        )

        # append_to_file(self.output_paths["case"],
        #                  f"{' '*2}variables=r(ra),少子-背景离子_碰撞转移功率,少子-电子_碰撞转移功率,平均能量，平均能量⊥，平均能量∥\n")

        #------------------------确定参数n//--------------------------
        # 根据环向模数的个数，确定n//的取值，以在共振条件下计算k2⊥
        # 若环向模数是1个，则选择具有代表性的n//
        # 若环向模数是多个，则扫描确定最佳n//
        # Determine best n_phi (simplified logic from Fortran)
        # 这里直接在初始化时设置 self.n_phi
        #-------------------------------------------------------------
        if self.iex_sumnph <= 0 :
            self.nphi = self.ctx.nphi
            append_to_file(self.output_paths["read"], f"n// nphi = {self.nphi}\n")

        else:
            zpwmax = 0.0
            # for n in range(2):
            #     self.nphi = self.ctx.n_phi_val[n] # Default
        # Logic to scan n_phi omitted as it requires external data structure
        
        # 初始化MHD模块

        # 初始化
        # Initialization
        self.inifpq()
        self.outfpq(-1)

        if self.ibreak != 0:
            self.ctx.profend('3Dfpql')
            return

        # 计算不同频段（1=基频，2=第一谐频）
        # Loop over harmonics (1=Fundamental, 2=First Harmonic)
        # for jhrm in [1, 2]: 
        
        # 计算指定频段
        for jhrm in [self.ipheat]: 

            self.ipheat = jhrm
            append_to_file(self.output_paths["read"], f"\n{'-'*15}加热模式 ipheat = {self.ipheat}{'-'*15}\n")
            
            # 计算准线性加热项系数参数
            # if self.icrh_enabled:
            # 计算勒让德多项式积分 qleg00 qleg01 qleg11
            # self.coelgd()
            # print("Is Q01 symmetric?", np.allclose(self.qleg01, np.transpose(self.qleg01, (1,0,2))))

            # uc = 0.3  
            # a = 20.0
            # self.coelgd_with_weight_optimized(uc, a)

            # 计算准线性加热项系数参数
            if self.enable_trapping:
                # 捕获效应模式：传入 Sigmoid 参数
                self.coelgd_with_weight_optimized(enable_weight=True, uc=self.trapping_ucr, a=self.trapping_a)
            else:
                # 纯 ICRH 模式：W=1 的纯数值积分
                self.coelgd_with_weight_optimized(enable_weight=False)




            # # 输出 qleg00 qleg01 qleg11
            # with open('01_qleg00.txt', 'w') as f:
            #     f.write(f"=== 01_qleg00 L={self.legmod} ===\n")            
            #     for i in range(self.qleg00.shape[0]):
            #         row_values = []
            #         for j in range(self.qleg00.shape[1]):
            #             for K in range(self.qleg00.shape[2]):
            #                 val = self.qleg00[i, j, K]
            #                 # if val <1e20:
            #                 #     row_values.append(f"[--------]")
            #                 # else:
            #                 row_values.append(f"[{val:.2e}]")
            #             f.write(f"Row {i}: {' '.join(row_values)}\n")

            # 计算不同粒子种类（0=背景离子, 1=少数离子)
            # for isp in range(self.ctx.nspec):  

            # 计算指定粒子种类
            for isp in [self.ispec]:
                print(f"初始化：准线性化算子")

                ipsi = 0
                
                append_to_file(self.output_paths["read"], f"\n{'-'*10}计算第{isp+1}个粒子{'-'*10}\n")
                print(f"计算第{isp+1}个粒子")

                self.iniqln(isp, ipsi)
                
                # self.outfpq(ipsi)
                self.inimat(ipsi)

                if self.ibreak == -99: continue

                # 计算准线性加热项系数参数
                # 【插入诊断】
                # self.diagnose_q01_asymmetry()
            

                # 计算不同半径（ psi ）

                # self.ipsiii = [int(i / 0.02) for i in self.ipsii]
                for ipsi in range(1, self.nrdpsi + 1):
                # for ipsi in self.ipsiii:

                    append_to_file(self.output_paths["read"], f"\n{'-'*5}计算半径 {ipsi}: {self.redpsi[ipsi-1]}处速度分布{'-'*5}\n")
                    append_to_file(self.output_paths["read"], f"粒子温度 ztempi = {self.ztempi[:,ipsi-1]}\n")
                    append_to_file(self.output_paths["read"], f"粒子密度 zdeni = {self.zdeni[:,ipsi-1]}\n")
                    print(f"计算半径 {ipsi}: {self.redpsi[ipsi-1]}处速度分布")                    

                    self.inicol(isp, ipsi)

                    self.iniqln(isp, ipsi)

                    if self.ibreak != 0: continue

                    self.inimat(ipsi)

                    #输出等离子体参数和碰撞系数
                    self.outpsi(ipsi)

                    # Enforce density and the requested RF power in the solve,
                    # without rescaling the physical source afterwards.
                    self.solve_stationary(ipsi)

                    # # 输出带状矩阵文件
                    # with open('11_A.txt', 'w') as f:
                    #     f.write("=== Readable Matrix Format ===\n") 
                    #     f.write(f"fmatrx : {self.fmatrx.shape} \n")            
                    #     for i in range(self.fmatrx.shape[0]):
                    #         row_values = []
                    #         for j in range(self.fmatrx.shape[1]):
                    #             val = self.fmatrx[i, j]
                    #             # if val <1e20:
                    #             #     row_values.append(f"[--------]")
                    #             # else:
                    #             row_values.append(f"[{val:.2e}]")
                    #         f.write(f"Row {i}: {' '.join(row_values)}\n")

                    # 提取带状矩阵，
                    aa = self.fmatrx[self.iwd:self.iwd+self.iwd+self.iwd+1,:]
                    expected_rows = self.iwd + self.iwd + 1
                    print(f"原始矩阵 fmatrx 的形状: {self.fmatrx.shape}")
                    print(f"提取后的矩阵 aa 的形状: {aa.shape}")
                    print(f"使用的带宽: (l={self.iwd}, u={self.iwd})")
                    if aa.shape[0] != expected_rows:
                        raise ValueError(f"提取后的矩阵 aa 行数 {aa.shape[0]} 与期望的 {expected_rows} 不符。")
                    
                    # # 输出矩阵A文件
                    # with open('01_A.txt', 'w') as f:
                    #     f.write("=== Readable Matrix Format ===\n")
                    #     f.write(f"fmatrx : {self.fmatrx.shape} \n") 
                    #     f.write(f"aa : {aa.shape} \n")
                    #     f.write(f"(l={self.iwd}, u={self.iwd})\n")
                    #     for i in range(aa.shape[0]):
                    #         row_values = []
                    #         for j in range(aa.shape[1]):
                    #             val = aa[i, j]
                    #             # if val <1e20:
                    #             #     row_values.append(f"[--------]")
                    #             # else:
                    #             row_values.append(f"[{val:.2e}]")
                    #         f.write(f"Row {i}: {' '.join(row_values)}\n")

                    # # 输出矩阵b文件
                    # with open('01_b.txt', 'w', encoding='utf-8') as f:
                    #     f.write(f"=== 初始向量 b ===\n")
                    #     f.write(f"b : {self.fb.shape}\n") 
                    #     for i in range(self.fb.shape[0]):
                    #         row_values = []
                    #         val = self.fb[i]
                    #         # if val <1e-20:
                    #         #     row_values.append(f"[--------]")
                    #         # else:
                    #         row_values.append(f"[{val:.15e}]")
                    #         f.write(f"Row {i:<4d}: {' '.join(row_values)}\n")

                    # # 导出矩阵aa
                    # try:
                    #     # 也可以保存为文本格式
                    #     np.savetxt('01_out_A.txt', aa)
                    #     print("矩阵也已保存为文本格式: matrix_aa.txt")
                        
                    # except Exception as e:
                    #     print(f"保存矩阵时出错: {e}")

                    # # 导出矩阵bb
                    # bb = self.fb
                    # try:
                    #     # 也可以保存为文本格式
                    #     np.savetxt('01_out_b.txt', bb, fmt='%.15e')
                    #     print("矩阵也已保存为文本格式: matrix_bb.txt")
                        
                    # except Exception as e:
                    #     print(f"保存矩阵时出错: {e}")
                    
                    self.dfparm(ipsi, self.power_iterations)
                    
                    self.outdis(ipsi)
                    self.write_debug_info(ipsi)   # 写入调试信息文件 debug.txt
                    print(f"{' '*10}垂直尾部温度  {self.ttailx:13.4E} keV")

                    # 绘图
                    if self.plot_range_k == 1:
                        plot_range = 'limited'
                    else:
                        plot_range = 'full'

                    if self.redpsi[ipsi-1] in self.ipsii:

                        filename = os.path.join(self.current_out_dir, f'case0_Fi_{self.redpsi[ipsi-1]}.txt')
                        Es = self.tbkev.copy()
                        col_indices = self.xmu
                        print(f"写入维度: nptu={self.nptu}, nmu={self.nmu}")

                        # 保存数据文件
                        row1 = np.array([f"{self.nptu} * {self.nmu}"])
                        row2 = np.concatenate(([r'E\mu'], [f"{x:.6e}" for x in col_indices]))
                        Es_str = [f"{x:.6e}" for x in Es]
                        data_with_index = np.column_stack((Es_str, self.tfecmu))
                        padding_size = data_with_index.shape[1] - row1.shape[0]
                        row1_padded = np.concatenate((row1, np.full(padding_size, '')))
                        full_matrix = np.vstack((row1_padded, row2, data_with_index))
                        np.savetxt(filename, full_matrix, fmt='%s', delimiter=' ')
                        np.savez_compressed(
                            os.path.splitext(filename)[0] + '.npz',
                            raw_coefficients=self.raw_tbfu,
                            raw_derivatives=self.raw_tbdfu,
                            normalized_coefficients=self.tbfu,
                            normalized_derivatives=self.tbdfu,
                            raw_distribution=self.raw_tbfu @ self.plgcmu,
                            normalized_distribution=self.tfecmu,
                            energies_keV=Es,
                            mu=self.xmu,
                            fnorm=self.fnorm,
                            matrix_relative_residual=self.matrix_relative_residual,
                            density_residual=self.density_residual,
                            particle_balance_residual=self.particle_balance_residual,
                            bulk_power_residual=self.bulk_power_residual,
                            nbi_injected_power=self.nbi_injected_power,
                            nbi_sink_power=self.nbi_sink_power,
                            rf_absorbed_power=self.hrate,
                            distribution_checks_passed=self.distribution_checks_passed,
                            mean_energy_keV=self.mean_energy_keV,
                            mean_parallel_velocity_ms=self.mean_parallel_velocity_ms,
                            thermal_parallel_temperature_keV=self.thermal_parallel_temperature_keV,
                        )
                        print(f"✅ 分布函数已保存: {filename}")

                        if not self.plot_enabled:
                            continue

                        idx_pos = np.argmin(np.abs(self.xmu - 0.8))
                        idx_neg = np.argmin(np.abs(self.xmu + 0.8))
                        print(self.xmu[idx_pos], self.xmu[idx_neg])
                        print(self.tfecmu[:, idx_pos].max(), self.tfecmu[:, idx_neg].max())

                        def _plot_heatmap_once():
                            """抽取热图绘制逻辑，避免重复"""
                            print(f"开始绘图...r={self.redpsi[ipsi-1]}a")
                            current_tail_E, current_tail_mu, current_char_mu, current_char_E = self.extract_high_energy_tail_contour()

                            is_synergy = self.icrh_enabled and self.nbi_enabled
                            is_nbi_only = self.nbi_enabled and not self.icrh_enabled
                            is_icrh_only = self.icrh_enabled and not self.nbi_enabled

                            if is_synergy:
                                return self.plot_f_heatmap(
                                    self.ctx, ipsi, plot_range,
                                    nbi_tail_E=self.nbi_tail_E, nbi_tail_mu=self.nbi_tail_mu,
                                    nbi_char_mu=self.nbi_tail_mu_pos, nbi_char_E=self.nbi_tail_E_pos,
                                    icrh_tail_E=self.icrh_tail_E, icrh_tail_mu=self.icrh_tail_mu,
                                    icrh_char_mu=self.icrh_tail_mu_pos, icrh_char_E=self.icrh_tail_E_pos,
                                    co_tail_E=current_tail_E, co_tail_mu=current_tail_mu,
                                    co_char_mu=current_char_mu, co_char_E=current_char_E
                                )
                            elif is_nbi_only:
                                self.nbi_tail_E = current_tail_E
                                self.nbi_tail_mu = current_tail_mu
                                self.nbi_tail_mu_pos = current_char_mu
                                self.nbi_tail_E_pos = current_char_E
                                return self.plot_f_heatmap(
                                    self.ctx, ipsi, plot_range,
                                    nbi_tail_E=current_tail_E, nbi_tail_mu=current_tail_mu,
                                    nbi_char_mu=current_char_mu, nbi_char_E=current_char_E
                                )
                            elif is_icrh_only:
                                self.icrh_tail_E = current_tail_E
                                self.icrh_tail_mu = current_tail_mu
                                self.icrh_tail_mu_pos = current_char_mu
                                self.icrh_tail_E_pos = current_char_E
                                return self.plot_f_heatmap(
                                    self.ctx, ipsi, plot_range,
                                    icrh_tail_E=current_tail_E, icrh_tail_mu=current_tail_mu,
                                    icrh_char_mu=current_char_mu, icrh_char_E=current_char_E
                                )

                        # 根据 plot_type 绘制
                        if self.plot_type == 0:
                            print(f"开始绘图...r={self.redpsi[ipsi-1]}a")
                            plot_result = self.plot_f_velocity_pitch_angle(self.ctx, ipsi, plot_range)

                        elif self.plot_type == 1:
                            plot_result = _plot_heatmap_once()

                        elif self.plot_type == 2:
                            print(f"开始绘图...r={self.redpsi[ipsi-1]}a")
                            plot_result = self.plot_f_velocity_average_pitch_angle(self.ctx, ipsi, plot_range)

                        elif self.plot_type == 3:
                            print(f"开始绘图...r={self.redpsi[ipsi-1]}a")
                            plot_result = self.plot_f_velocity_para_perp(ipsi, plot_range)

                        elif self.plot_type == 4:
                            print(f"开始绘图...r={self.redpsi[ipsi-1]}a")

                            print("  [1/3] 绘制热图...")
                            plot_result_heatmap = _plot_heatmap_once()

                            print("  [2/3] 绘制速度分布图...")
                            plot_result_pitch = self.plot_f_velocity_pitch_angle(self.ctx, ipsi, plot_range)

                            # print("  [3/3] 绘制平行垂直速度分布图...")
                            # plot_result_para_perp = self.plot_f_velocity_para_perp(ipsi, plot_range)

                            saved_files = []
                            # for res in [plot_result_heatmap, plot_result_pitch, plot_result_para_perp]:
                            for res in [plot_result_heatmap, plot_result_pitch]:
                                if isinstance(res, tuple) and len(res) > 1 and res[1]:
                                    saved_files.append(res[1])

                            plot_result = (None, saved_files)

                        # 兼容返回值
                        if isinstance(plot_result, tuple) and len(plot_result) > 1:
                            self.last_saved_figure = plot_result[1]
                        else:
                            self.last_saved_figure = None
                        
                    else:
                        print(f"不在绘图位置，跳过绘图")

                    
                # self.outfpq(1) # End of species loop output

                

        # self.outfpq(2) # Final output
        self.ctx.profend('3Dfpql')

    # ===================================================================
    # Initialization Routines
    # ===================================================================

    def inifpq(self):
        """Initialization of coefficients and memory allocation."""
        self.nspele = self.ctx.nspec + 1
        self.nmu = 101
        self.nibj = 101
        self.kamax = 6  # 仅用于旧版多项式矩诊断；实际 RF 核不截断。
        
        append_to_file(self.output_paths["read"], f"俯仰角空间点数 nmu = {self.nmu}\n")
        append_to_file(self.output_paths["read"], "RF 扩散使用完整 Bessel 核；旧版多项式矩仅供诊断\n")
        append_to_file(self.output_paths["read"], f"电子是第nspele = {self.nspele} 个粒子\n")


        #网络参数
        self.redxoa = np.zeros(self.nrdpsi)

        # 功率参数
        self.zfppmx = np.zeros((self.nspele,2))
        self.ipsplf = np.zeros((self.nspele,2))

        # 勒让德多项式
        self.plg = np.zeros(2*self.legmod+2)
        self.plgcmu = np.zeros((self.legmod+1, self.nmu))  
        self.fbmaxw = np.zeros((self.nptu, self.nmu))  
        

        # Allocate Arrays
        self.fjbess = np.zeros(self.nibj + 1)
        
        # 速度参数
        self.tbu = np.zeros(self.nptu)
        self.tbkev = np.zeros(self.nptu)
        self.tbpsic = np.zeros(self.nptu)
        self.tbpsie = np.zeros(self.nptu)
        self.tbpsit = np.zeros(self.nptu)
        self.tbthec = np.zeros(self.nptu)
        self.auxfun = np.zeros(self.nptu)
        self.tbpl = np.zeros(self.nptu)
        self.ref_maxwell = np.zeros(self.nptu)
        self.tbphif = np.zeros((self.nptu, self.nspele))

        self.tfecmu = np.zeros((self.nptu, self.nmu))
        self.f_log = np.zeros((self.nptu, self.nmu))    # 对数化
        self.f_slope = np.zeros((self.nptu, self.nmu))  # 数据斜率
        self.f_error = np.zeros((self.nptu, self.nmu))  # 数据斜率
        
        
        
        self.nmueff = np.zeros(self.nmu)

        self.tbfu = np.zeros((self.nptu, self.legmod + 1))
        self.tbdfu = np.zeros((self.nptu, self.legmod + 1))


        self.tbdqlk = np.zeros((self.nptu, self.kamax + 1))
        
        self.redpsi = np.zeros(self.nrdpsi)
        self.pwredf = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.pwredh = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.pwredl = np.zeros(self.nrdpsi)
        self.redvol = np.zeros(self.nrdpsi)
        
        self.ztempi = np.zeros((self.nspele, self.nrdpsi))
        self.zdeni = np.zeros((self.nspele, self.nrdpsi))
        
        self.npsihf = self.nrdpsi/2+1
        self.npsihf = int(self.npsihf)  # 将浮点数转换为整数
        print(f"npsihf: {self.npsihf}")
        self.rdhpsi = np.zeros(self.npsihf)
        self.fpsele = np.zeros(self.npsihf)
        self.fpsion = np.zeros(self.npsihf)

        # Output tables
        self.thrate = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.thtprp = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.thtpar = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.thttot = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.tmtail = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.tavgtx = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.tavgtz = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.tavbkx = np.zeros((self.ctx.nspec, self.nrdpsi))
        self.tavbkz = np.zeros((self.ctx.nspec, self.nrdpsi))
        
        self.tab_source = np.zeros(self.nptu)

        self.pwgain = np.zeros((2, self.nspele, self.nrdpsi))
        self.pwexch = np.zeros((2, self.nspele))
        self.tab_pwexch = np.zeros((self.nspele, self.nrdpsi))
        
        # Legendre Integrals
        self.qleg00 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        self.qleg01 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        self.qleg11 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        
        self.dqlu = np.zeros(self.legmod + 1) # Note: Fortran used legmod, but logic implies ksup
        # Actually dqlu size depends on ksup, let's make it safe
        # self.dqlu = np.zeros(100) 
        self.dql00 = np.zeros((self.legmod+1, self.legmod+1))
        self.dql01 = np.zeros((self.legmod+1, self.legmod+1))
        self.dql11 = np.zeros((self.legmod+1, self.legmod+1))

        # 分配 3D 数组，用于保存所有径向点的 f(v, mu, psi)
        # 形状：(速度点数, 俯仰角点数, 径向点数)
        self.f_v_mu_psi = np.zeros((self.nptu, self.nmu, self.nrdpsi))
        
        #一致性测试
        self.ibreak =0

        if not np.isfinite(self.power) or self.power < 0:
            raise ValueError("ICRH 功率必须是有限非负值；0 表示不加热")

        #输出控制
        if self.joutql <= 2 and self.nptu >50 :
            self.jmptu = self.nptu/50
        else:
            self.jmptu = 1

        # Inverse factorials
        jbmax = self.kamax + 1
        self.fctinv = np.zeros(jbmax + 1)
        self.fctinv[0] = 1.0
        self.fctinv[1] = 1.0
        for j in range(2, jbmax + 1):
            self.fctinv[j] = self.fctinv[j-1] / float(j)
            
        
        # Radial Grid Setup (Simplified)
        # 设置径向网络
        # 功率沉积插值
        if self.nrdpsi > 1:
            self.drdpsi = self.ctx.psiedg / float(self.nrdpsi - 1) 
        else:
            self.drdpsi = 0
        append_to_file(self.output_paths["read"], f"径向网络间隔 drdpsi = {self.drdpsi}\n")
        
        # print(f"未知3-psiedg,已设置为 {self.ctx.psiedg}， 改变径向网络 drdpsi")
        self.redpsi[0] = 0                                          #未知：后面循环覆盖
        for isp in range(self.ctx.nspec):
            self.ctx.pwpsif [0:self.ctx.nspec-1 , 0] = 1                               #设置初始为1
            self.ctx.pwpsih [0:self.ctx.nspec-1 , 0] = 1
            self.pwredf[isp,0] = self.power * self.ctx.pwpsif[isp,0]                #吸收功率：后面循环覆盖
            self.pwredh[isp,0] = self.power * self.ctx.pwpsih[isp,0]
            

        self.ctx.pwpelt[0] = 1                  #未知：后面循环覆盖
        self.pwredl[0] = self.power*self.ctx.pwpelt[0]                  #未知：后面循环覆盖
        javg = 0                                                    #未知：影响插值方法
        dpsi = self.ctx.psiedg / 2.0                                #未知： dpsi 
        # print(f"未知4-dpsi,径向网格设置相关，设置为由psiedg得到：{dpsi}, 改变基频、谐频离子加热功率密度 pwredf 、 pwredh 和电子加热功率密度 pwredl")
        if self.drdpsi >= 2.5 * dpsi:
            javg = 1
        theta = 0.0                                                  #未知：影响未知模块 t4_coords(psi, theta)
        kexcpw = 0                                                   #未知：可能影响其他模块
        ibmod = -1                                                   #未知：影响未知模块 t4_surfpl(psi,ibmod)
        # print(f"未知5-未知参数 ibmod ：影响未知模块 t4_surfpl(psi,ibmod) ")

        # 输入参数导入部分
        if self.nrdpsi > 1:
            # 输入径向位置、温度、密度、功率数据
            profiles = {}
            for name in ('P', 'N0', 'N1', 'Ne', 'T0', 'T1', 'Te'):
                values = np.asarray(self.arrays[name], dtype=float)
                if values.shape != (self.nrdpsi,) or not np.isfinite(values).all():
                    raise ValueError(f"径向剖面 {name} 必须与径向网格等长且全部有限")
                profiles[name] = values
            radii = np.asarray(self.positions, dtype=float)
            if (radii.shape != (self.nrdpsi,) or not np.isfinite(radii).all()
                    or np.any(np.diff(radii) <= 0) or np.any(radii < 0) or np.any(radii > 1)):
                raise ValueError("径向剖面位置必须是 [0,1] 内严格递增的 rho")
            if np.any(profiles['P'] < 0):
                raise ValueError("径向 RF 功率剖面不能为负")
            if any(np.any(profiles[name] <= 0) for name in ('T0', 'T1', 'Te', 'Ne')):
                raise ValueError("径向温度和电子密度必须为正")
            if any(np.any(profiles[name] < 0) for name in ('N0', 'N1')):
                raise ValueError("径向离子密度不能为负")
            if np.any(profiles[f'N{self.ispec}'] <= 0):
                raise ValueError("被求解离子的径向密度必须为正")
            charge_density = self.ctx.azi[0] * profiles['N0'] + self.ctx.azi[1] * profiles['N1']
            if not np.allclose(charge_density, profiles['Ne'], rtol=1e-6, atol=0):
                raise ValueError("径向密度剖面必须满足准中性 Σ Zi*ni = ne")
            self.arrays = profiles
            for isp in range(self.ctx.nspec):
                self.pwredf[isp,:] = np.array(self.arrays['P'], dtype=np.float64)
                self.pwredh[isp,:] = np.array(self.arrays['P'], dtype=np.float64)
            self.zdeni[0,:] = np.array(self.arrays['N0'], dtype=np.float64) * 10
            self.zdeni[1,:] = np.array(self.arrays['N1'], dtype=np.float64) * 10
            self.zdeni[2,:] = np.array(self.arrays['Ne'], dtype=np.float64) * 10
            self.ztempi[0,:] = np.array(self.arrays['T0'], dtype=np.float64)
            self.ztempi[1,:] = np.array(self.arrays['T1'], dtype=np.float64) 
            self.ztempi[2,:] = np.array(self.arrays['Te'], dtype=np.float64)
            self.redpsi = self.positions
            self.ctx.aconc[self.nspele-1]  = 1.0
        else :
            for ipsi in range(self.nrdpsi):
                for isp in range(self.ctx.nspec):
                    self.ztempi[isp, ipsi] = self.ctx.tempi[isp]
                    self.zdeni[isp, ipsi] = self.ctx.xfni[isp]
                    self.pwredf[isp,:] = self.power
                    self.pwredh[isp,:] = self.power
                isp = self.nspele-1 
                self.ztempi[isp, ipsi] = self.ctx.tempe
                self.zdeni[isp, ipsi] = self.ctx.dens
                self.pwredl[ipsi] = 0.0
                self.redpsi = np.zeros(1, dtype=np.float64)
                self.redpsi[0] = self.ipsii[0]
                if ipsi == 0 or ipsi == self.nrdpsi-1:                                      # 跳过边界
                    continue

        # for ipsi in range(self.nrdpsi):
        #     psi = float(ipsi) * self.drdpsi
            # self.redpsi[ipsi] = psi                                  #求解器径向网络
            # if psi != 0.0:                                           #未知：未知数据计算，影响未知模块和未知参数
            #     # print(f"未知6-计算参数 redxoa ：与 redpsi 对应的径向网络 (x/a) ")
            #     # self.t4_mhdcoe(psi)                                  #未知：未知模块 t4_mhdcoe(psi)
            #     # self.t4_coords(psi, theta)                           #未知：未知模块 t4_coords(psi, theta)
            #     self.redxoa[ipsi] = self.ctx.ulengt * self.ctx.ux / self.ctx.xedg_out    #未知：都是未知参数
            # else:
            #     self.redxoa[ipsi] = 0.0

            # self.t4_surfpl( psi, ibmod )                                #未知：未知模块 t4_surfpl(psi,ibmod)

            # for isp in range(self.ctx.nspec):
            #     self.ztempi[isp, ipsi] = self.ctx.tempi[isp]
            #     self.zdeni[isp, ipsi] = self.ctx.xfni[isp]
            # isp = self.nspele-1 
            # self.ztempi[isp, ipsi] = self.ctx.tempe
            # self.zdeni[isp, ipsi] = self.ctx.dens

            # # 计算体积比(specific volume)
            # zpsi = psi + 0.5 * self.drdpsi
            # if zpsi < 1.0:                                                   #未知：计算未知参数 self.redvol
            #     print(f"未知7-计算参数 self.redvol ： ipsi 处的体积比")
            #     # jcheb = self.t4_chbint(zpsi, self.volmhd, self.nmhd)       #未知：未知参数，随便设置模块 t4_chbint(zpsi, volmhd, nmhd)
            #     jcheb = 1                                                    #未知：未知参数 jcheb ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)    
            #     # print(f"未知5-jcheb,已设置为 {jcheb} ")
            #     volmhd = 1                                                   #未知：未知参数 volmhd ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)      
            #     nmhd = 1                                                     #未知：未知参数                
            #     if jcheb <= 0:
            #         # dpsmhd = self.volmhd[2] - self.volmhd[1]               #未知：未知参数 dpsmhd ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)
            #         dpsmhd = 1                                               #未知：未知参数 dpsmhd ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)
            #         # print(f"未知6-dpsmhd,已设置为 {dpsmhd} ")

            #         # psimhd = self.volmhd                                   #未知：未知参数 psimhd ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)
            #         psimhd = 1                                               #未知：未知参数 psimhd ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)
            #         # print(f"未知7-psimhd,已设置为 {psimhd} ")

            #         # npt = self.nmhd                                        #未知：未知参数 npt ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)                                    
            #         npt = int(zpsi/dpsmhd) + 1                               #未知：未知参数 npt ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)                                    
            #         # print(f"未知8-npt,已设置为 {npt} ")

            #         zh = zpsi - psimhd[npt]                                  #未知：未知参数 zh ，影响未知模块 self.t4_valspl(zh, volmhd, nmhd, npt, 0)
            #         self.redvol[ipsi] = self.t4_valspl(zh, volmhd, nmhd, npt, 0) * psi
            #     else:
            #         xcap = zpsi - (1.0 - zpsi)
            #         self.redvol[ipsi] = self.t4_chbval(xcap, volmhd, jcheb) * psi
            # else:
            #     self.redvol[ipsi] = self.redvol[ipsi-1]

            # if ipsi == 0 or ipsi == self.nrdpsi-1:                                      #跳过边界
            #     continue

            # #功率线性插值
            # print(f"未知8-计算插值系数 zh, zk, kpsi, tpsi")
            # tpsi = np.array([0.1127016654, 0.5000000000, 0.8872983346])                   #未知：tpsi ，随便设置

            # kpsi = int(psi/dpsi + 0.5)
            # zh = (psi - tpsi[kpsi])/dpsi
            # zk = 1.0 - zh
            # if javg == 0:
            #     for isp in range(self.ctx.nspec):
            #         self.pwredf[isp, ipsi] = self.power*(zk*self.pwpsif[isp, kpsi] + zh*self.pwpsif[isp, kpsi+1])
            #         self.pwredh[isp, ipsi] = self.power*(zk*self.pwpsih[isp, kpsi] + zh*self.pwpsih[isp, kpsi+1])
            #     self.pwredl[ipsi] = self.power*(zk*self.pwpelt[kpsi] + zh*self.pwpelt[kpsi+1])
            # else:
            #     for isp in range(self.ctx.nspec):
            #         self.pwredf[isp, ipsi] = self.power*(zk*self.pwpsif[isp, kpsi-1] + 
            #             self.pwpsif[isp, kpsi] + self.pwpsif[isp, kpsi+1] + 
            #             zh*self.pwpsif[isp, kpsi+2])/3.0
            #         self.pwredh[isp, ipsi] = self.power*(zk*self.pwpsih[isp, kpsi-1] + 
            #             self.pwpsih[isp, kpsi] + self.pwpsih[isp, kpsi+1] + 
            #             zh*self.pwpsih[isp, kpsi+2])/3.0
            #     self.pwredl[ipsi] = self.power*(zk*self.pwpelt[kpsi-1] + self.pwpelt[kpsi] + 
            #         self.pwpelt[kpsi+1] + zh*self.pwpelt[kpsi+2])/3.0
        
        #网格边界设置
        # self.redpsi[self.nrdpsi-1] = self.ctx.psiedg
        append_to_file(self.output_paths["read"], f"径向网格空间 redpsi = \n{self.redpsi}\n") 

        # for isp in range(self.ctx.nspec):
        #     self.pwredf[isp, self.nrdpsi-1] = 0.0
        #     self.pwredh[isp, self.nrdpsi-1] = 0.0

        self.pwredl[self.nrdpsi-1] = 0.0

        #找到粒子功率峰值及谐波值的索引位置，最开始的5个点不精确，排除
        self.zpwmin = 0.0
        for isp in range(self.ctx.nspec):
            self.zfppmx[isp, 0] = 0.0
            self.zfppmx[isp, 1] = 0.0
            self.ipsplf[isp, 0] = -1
            self.ipsplf[isp, 1] = -1
            zpmax1 = 0.0
            zpmax2 = 0.0
            
            for ipsi in range(4, self.nrdpsi):  
                if self.pwredf[isp, ipsi] > zpmax1:
                    zpmax1 = self.pwredf[isp, ipsi]
                    self.zfppmx[isp, 0] = zpmax1
                    self.ipsplf[isp, 0] = ipsi
                    
                if self.pwredh[isp, ipsi] > zpmax2:
                    zpmax2 = self.pwredh[isp, ipsi]
                    self.zfppmx[isp, 1] = zpmax2
                    self.ipsplf[isp, 1] = ipsi
            
            self.ipsplf[isp, 0] = 1
            self.ipsplf[isp, 1] = 1
            
            self.zpwmin = max(self.zpwmin, 0.0001 * zpmax1)
            self.zpwmin = max(self.zpwmin, 0.0001 * zpmax2)
        self.zpwmin = self.power * self.zpwmin
        
        # NBI参数


        # # NBI程序输入假设
        # self.vnbi = np.ones((self.ctx.nspec, self.Ns)) # v(i,k) NBI第k束，对应离子i的粒子速度
        # self.nbi_in = np.ones((self.ctx.nspec, self.Ns, self.nmu, self.nrdpsi)) # S(i,k,μ,ψ) NBI程序得到的粒子功率沉积分布函数
        # # NBI光束 的 速度分量设置
        # column_values = np.ones((self.Ns))
        # E_nbi_keV = column_values*60
        # # E_nbi_keV = np.array([20, 30, 31, 46.5, 60, 93])
        # column_values = np.sqrt(E_nbi_keV / ctx.tempi[self.ispres])
        # self.vnbi[:, :] = column_values 
        
        # ================= 导入全径向 NBI 功率沉积剖面 =================
        if self.nbi_enabled and self.nbi_power_file is not None and os.path.exists(self.nbi_power_file):
            # Both files must use the same radial coordinate as redpsi. Row
            # numbers are not magnetic-surface coordinates.
            power_data = np.loadtxt(self.nbi_power_file, skiprows=1, ndmin=2)
            if (power_data.shape[1] < 2 or not np.all(np.isfinite(power_data[:, :2]))
                    or np.any(power_data[:, 1] < 0)
                    or np.any((power_data[:, 0] < 0) | (power_data[:, 0] > 1))
                    or np.any(np.diff(power_data[:, 0]) <= 0)):
                raise ValueError("NBI power profile requires increasing rho in [0,1] and nonnegative finite powers")
            coordinates = power_data[:, 0]
            if np.any(self.redpsi < coordinates[0]) or np.any(self.redpsi > coordinates[-1]):
                raise ValueError("NBI power profile must cover every requested radial coordinate")
            self.nbi_power_profile = np.interp(self.redpsi, coordinates, power_data[:, 1])
            print(f"✅ NBI 功率剖面已按径向坐标插值到 {len(self.nbi_power_profile)} 个求解磁面")
        elif self.nbi_enabled:
            print(f"⚠️ 未指定 NBI 功率剖面文件或文件不存在，将使用常数 pwdens_nbi={self.pwdens_nbi}")
            self.nbi_power_profile = None
        # ========================================================


        # ============ 统一导入 NBI 功率沉积分布 (带详细诊断) ============
        self.dxmu = (self.mu_max - self.mu_min) / (self.nmu - 1)
        self.xmu = np.array([self.mu_min + imu * self.dxmu for imu in range(self.nmu)])
        
        if self.nbi_enabled and self.nbi_file is not None and os.path.exists(self.nbi_file):
            try:
                print("\n" + "="*60)
                print("【NBI 诊断】开始导入 NBI 数据...")
                print("="*60)
                
                # -----------------------------------------------------------
                # 1. 智能读取：解析表头获取真实的径向坐标 rho
                # -----------------------------------------------------------
                rho_file = None
                with open(self.nbi_file, 'r') as f:
                    header = f.readline().strip()
                    if header.startswith('Mu') or header.startswith('mu'):
                        rho_file = []
                        for item in header.split()[1:]:
                            if 'rho_' in item:
                                rho_file.append(float(item.split('rho_')[1]))
                            else:
                                try:
                                    rho_file.append(float(item)) # 兼容纯数字
                                except ValueError:
                                    rho_file = None 
                                    break
                
                raw_data = np.loadtxt(self.nbi_file, skiprows=1, ndmin=2)
                print(f"[步骤1] 原始数据读取完成，形状: {raw_data.shape}")
                
                # -----------------------------------------------------------
                # 2. 提取并验证 Mu 坐标
                # -----------------------------------------------------------
                if raw_data.shape[0] < 2 or raw_data.shape[1] < 2 or not np.isfinite(raw_data).all():
                    raise ValueError("NBI angular source requires finite mu rows and at least one rho column")
                mu_data = raw_data[:, 0]
                S_data = raw_data[:, 1:] 
                if (np.any(np.diff(mu_data) <= 0) or np.any(np.abs(mu_data) > 1)
                        or np.any(S_data < 0)):
                    raise ValueError("NBI mu must increase within [-1,1] and source values must be nonnegative")
                
                if rho_file is None or len(rho_file) != S_data.shape[1]:
                    raise ValueError("NBI header must name every radial column: Mu rho_0.0 rho_0.5 ...")
                rho_file = np.asarray(rho_file, dtype=float)
                if (not np.isfinite(rho_file).all() or np.any(np.diff(rho_file) <= 0)
                        or np.any((rho_file < 0) | (rho_file > 1))):
                    raise ValueError("NBI radial coordinates must increase within normalized minor radius rho in [0,1]")
                rho_fp = np.asarray(self.redpsi, dtype=float)
                if (rho_fp.shape != (self.nrdpsi,) or not np.isfinite(rho_fp).all()
                        or np.any(rho_fp < rho_file[0]) or np.any(rho_fp > rho_file[-1])):
                    raise ValueError("NBI angular source must cover every requested rho; extrapolation is not supported")
                    
                print(f"[步骤2] 坐标提取完成:")
                print(f"  Mu 范围: [{mu_data.min():.2f}, {mu_data.max():.2f}], 点数: {len(mu_data)}")
                print(f"  Rho 范围: [{rho_file.min():.2f}, {rho_file.max():.2f}], 点数: {len(rho_file)}")
                print(f"  S_data 形状 (Mu x Rho): {S_data.shape}")
                print(f"  S_data 数值范围: [{S_data.min():.2e}, {S_data.max():.2e}]")

                # 导出原始文件读取的二维数据 (方便你画图对比)
                try:
                    header_out = "Mu\t" + "\t".join([f"rho_{r:.2f}" for r in rho_file])
                    np.savetxt('01_NBI_Raw_S_Mu_Rho.txt', np.column_stack((mu_data, S_data)), 
                               header=header_out, fmt='%.6e', comments='')
                    print("  ✅ 原始二维剖面已导出: 01_NBI_Raw_S_Mu_Rho.txt")
                except Exception as e:
                    print(f"  ⚠️ 导出原始数据失败: {e}")

                # -----------------------------------------------------------
                # 3. 径向插值对齐
                # -----------------------------------------------------------
                actual_nrdpsi = S_data.shape[1]
                if actual_nrdpsi != self.nrdpsi or not np.allclose(rho_file, rho_fp):
                    print(f"[步骤3] 开始径向插值: 文件({actual_nrdpsi}点) -> FP代码({self.nrdpsi}点)")
                    print(f"  目标 Rho 范围: [{rho_fp.min():.2f}, {rho_fp.max():.2f}]")
                    
                    S_data_aligned = np.zeros((len(mu_data), self.nrdpsi))
                    for i_mu in range(len(mu_data)):
                        S_data_aligned[i_mu, :] = np.interp(rho_fp, rho_file, S_data[i_mu, :])
                    S_data = S_data_aligned
                    print(f"  径向插值完成，新形状: {S_data.shape}")
                else:
                    print("[步骤3] 径向网格一致，无需插值。")

                # -----------------------------------------------------------
                # 4. Mu 插值对齐
                # -----------------------------------------------------------
                S_nbi_full_mu = np.zeros((self.nrdpsi, self.nmu))
                print(f"[步骤4] 开始 Mu 插值: 文件({len(mu_data)}点) -> FP代码({self.nmu}点), 区间标志: {self.nbi_range_mu}")
                
                for ipsi in range(self.nrdpsi):
                    if self.nbi_range_mu == 0:
                        S_nbi_full_mu[ipsi, :] = np.interp(self.xmu, mu_data, S_data[:, ipsi], left=0.0, right=0.0)
                    elif self.nbi_range_mu == 1:
                        mask_pos = self.xmu >= 0.0
                        xmu_pos = self.xmu[mask_pos]
                        S_nbi_full_mu[ipsi, mask_pos] = np.interp(xmu_pos, mu_data, S_data[:, ipsi], left=0.0, right=0.0)
                    elif self.nbi_range_mu == -1:
                        mask_neg = self.xmu <= 0.0
                        xmu_neg = self.xmu[mask_neg]
                        S_nbi_full_mu[ipsi, mask_neg] = np.interp(xmu_neg, mu_data, S_data[:, ipsi], left=0.0, right=0.0)
                
                print(f"  Mu 插值完成，S_nbi_full_mu 形状 (Rho x Mu): {S_nbi_full_mu.shape}")
                print(f"  S_nbi_full_mu 数值范围: [{S_nbi_full_mu.min():.2e}, {S_nbi_full_mu.max():.2e}]")

                # 导出插值后的二维数据 (与原始数据对比，看是否畸变)
                try:
                    # header_out = "Rho\\Mu\t" + "\t".join([f"{m:.2f}" for m in self.xmu])
                    header_out = r"Rho\Mu\t" + "\t".join([f"{m:.2f}" for m in self.xmu])

                    np.savetxt('02_NBI_Interpolated_S_Rho_Mu.txt', S_nbi_full_mu, 
                               header=header_out, fmt='%.6e', comments='')
                    print("  ✅ 插值后二维剖面已导出: 02_NBI_Interpolated_S_Rho_Mu.txt")
                except Exception as e:
                    print(f"  ⚠️ 导出插值数据失败: {e}")

                # -----------------------------------------------------------
                # 5. 赋值给 nbi_in (区分不同能量的光束 Ns)
                # -----------------------------------------------------------
                if len(self.nbi_energies_keV) != self.Ns:
                    print(f"⚠️ 警告：NBI能量列表长度({len(self.nbi_energies_keV)})与Ns({self.Ns})不符！自动补齐。")
                    while len(self.nbi_energies_keV) < self.Ns:
                        self.nbi_energies_keV.append(self.nbi_energies_keV[-1] / 2.0) 
                
                E_nbi_keV = np.array(self.nbi_energies_keV[:self.Ns])
                # self.vnbi = np.ones((self.ctx.nspec, self.Ns))
                # self.vnbi[:, :] = np.sqrt(E_nbi_keV / self.ctx.tempi[self.ispres])

                self.v_nbi_k = np.zeros(self.Ns)
                self.E_nbi_k = np.zeros(self.Ns)
                self.S_nbi_ku = np.zeros((self.Ns, self.nmu))
                self.sig = np.zeros((self.Ns, self.legmod + 1))
                self.source_nbi = np.zeros((self.Ns, self.legmod + 1))
                
                power_fractions = np.zeros(self.Ns)
                power_fractions[0] = 0.88  
                if self.Ns > 1: power_fractions[1] = 0.1  
                if self.Ns > 2: power_fractions[2] = 0.02  
                power_fractions = power_fractions / np.sum(power_fractions) if np.sum(power_fractions) > 0 else np.ones(self.Ns)/self.Ns

                print(f"[步骤5] 分配多束模型 (Ns={self.Ns}):")
                for k in range(self.Ns):
                    print(f"  光束 {k}: E={E_nbi_keV[k]:.1f} keV, 功率占比={power_fractions[k]*100:.1f}%")

                self.nbi_in = np.zeros((self.ctx.nspec, self.Ns, self.nmu, self.nrdpsi))
                # Preserve the original angular knots for exact integration
                # of the piecewise-linear source, independently of plot nmu.
                self.nbi_native_mu = mu_data.copy()
                self.nbi_native_profiles = power_fractions[:, None, None] * S_data[None, :, :]
                for isp in range(self.ctx.nspec):
                    for k in range(self.Ns):
                        for ipsi in range(self.nrdpsi):
                            self.nbi_in[isp, k, :, ipsi] = S_nbi_full_mu[ipsi, :] * power_fractions[k]
                            
                print(f"  nbi_in 赋值完成，形状: {self.nbi_in.shape}")
                print("="*60)
                print("【NBI 诊断】数据导入流程成功结束！")
                print("="*60 + "\n")
                
            except (OSError, ValueError) as error:
                raise ValueError(f"无法读取外部 NBI 源 {self.nbi_file}: {error}") from error
                
        else:
            if self.nbi_enabled and self.nbi_file is None:
                print(f"⚠️ 警告：NBI 文件 {self.nbi_file} 不存在！回退到测试高斯分布")
                self._generate_test_gaussian_nbi()
        # ========================================================

        # if self.nbi_enabled ：


        # # ============ 导入真实 NBI 功率吸收分布 (标志位控制版) ============
        # # 提前计算 mu 网格
        # self.dxmu = (self.mu_max - self.mu_min) / (self.nmu - 1)
        # self.xmu = np.array([self.mu_min + imu * self.dxmu for imu in range(self.nmu)])
        
        # if self.nbi_enabled and self.nbi_file is not None and os.path.exists(self.nbi_file):
        #     try:
        #         # 1. 读取数据
        #         raw_data = np.loadtxt(self.nbi_file)
                
        #         # 2. 格式分离：判断是否有 mu 坐标列
        #         col0_min, col0_max = np.min(raw_data[:, 0]), np.max(raw_data[:, 0])
        #         is_monotonic = np.all(np.diff(raw_data[:, 0]) > 0) or np.all(np.diff(raw_data[:, 0]) < 0)
                
        #         if -1.1 <= col0_min <= 1.1 and -1.1 <= col0_max <= 1.1 and is_monotonic:
        #             mu_data = raw_data[:, 0]
        #             S_data = raw_data[:, 1:]
        #         else:
        #             # 纯矩阵格式，根据标志位生成对应的虚拟坐标
        #             if self.nbi_range_mu == -1:
        #                 mu_data = np.linspace(-1.0, 0.0, raw_data.shape[0])
        #             elif self.nbi_range_mu == 0:
        #                 mu_data = np.linspace(-1.0, 1.0, raw_data.shape[0])
        #             else: # 默认 1
        #                 mu_data = np.linspace(0.0, 1.0, raw_data.shape[0])
        #             S_data = raw_data
                    
        #         # 3. 动态调整光束数量 Ns
        #         actual_Ns = S_data.shape[1]
        #         if actual_Ns != self.Ns:
        #             print(f"  [NBI 导入提示] 文件的光束数({actual_Ns})与预设({self.Ns})不符，自动调整。")
        #             self.Ns = actual_Ns
        #             self.vnbi = np.ones((self.ctx.nspec, self.Ns))
        #             self.v_nbi_k = np.zeros(self.Ns)
        #             self.E_nbi_k = np.zeros(self.Ns)
        #             self.S_nbi_ku = np.zeros((self.Ns, self.nmu))
        #             self.sig = np.zeros((self.Ns, self.legmod + 1))
        #             self.source_nbi = np.zeros((self.Ns, self.legmod + 1))
        #             E_nbi_keV = np.ones(self.Ns) * 60.0 
        #             self.vnbi[:, :] = np.sqrt(E_nbi_keV / ctx.tempi[self.ispres])

        #         # 4. 基于标志位的安全插值映射
        #         S_nbi_full_mu = np.zeros((self.Ns, self.nmu))
                
        #         if self.nbi_range_mu == 0:
        #             # 完整区间 [-1, 1]
        #             for k in range(self.Ns):
        #                 S_nbi_full_mu[k, :] = np.interp(self.xmu, mu_data, S_data[:, k], left=0.0, right=0.0)
        #         elif self.nbi_range_mu == 1:
        #             # 正半区间 [0, 1]
        #             mask_pos = self.xmu >= 0.0
        #             xmu_pos = self.xmu[mask_pos]
        #             for k in range(self.Ns):
        #                 S_nbi_full_mu[k, mask_pos] = np.interp(xmu_pos, mu_data, S_data[:, k], left=0.0, right=0.0)
        #         elif self.nbi_range_mu == -1:
        #             # 负半区间 [-1, 0]
        #             mask_neg = self.xmu <= 0.0
        #             xmu_neg = self.xmu[mask_neg]
        #             for k in range(self.Ns):
        #                 S_nbi_full_mu[k, mask_neg] = np.interp(xmu_neg, mu_data, S_data[:, k], left=0.0, right=0.0)
        #         else:
        #             raise ValueError(f"未知的 nbi_range_mu 值: {self.nbi_range_mu}，必须是 1, -1 或 0！")

        #         # 5. 广播到所有半径
        #         S_nbi_structured = np.zeros((self.Ns, self.nmu, self.nrdpsi))
        #         for k in range(self.Ns):
        #             for ipsi in range(self.nrdpsi):
        #                 S_nbi_structured[k, :, ipsi] = S_nbi_full_mu[k, :]
                        
        #         print(f"成功从 {self.nbi_file} 导入 NBI 功率分布！(区间标志: {self.nbi_range_mu}, 光束数: {self.Ns})")
                
        #         # 6. 赋值给 nbi_in
        #         self.nbi_in = np.zeros((self.ctx.nspec, self.Ns, self.nmu, self.nrdpsi))
        #         for isp in range(self.ctx.nspec):
        #             self.nbi_in[isp, :, :, :] = S_nbi_structured
                    
        #     except Exception as e:
        #         print(f"导入 NBI 数据失败: {e}，回退到测试高斯分布")
        #         self._generate_test_gaussian_nbi()
                
        # else:
        #     if self.nbi_enabled and self.nbi_file is not None:
        #         print(f"警告：NBI 文件 {self.nbi_file} 不存在！回退到测试高斯分布")
        #     self._generate_test_gaussian_nbi()
        # # ========================================================



        # # NBI光束 在μ上的 吸收功率分布函数设置（以μ=0为中心高斯分布）
        # # 生成高斯分布
        # mu_center = 0.7                # 均值（中心位置）
        # sigma = 0.01             # 标准差  
        # jfz = 4.8                 # 积分值
        # # 在第三个维度（nmu）上创建一个以索引0为中心的高斯分布
        # gaussian = self.f_gaussian(mu_center, sigma, self.nmu, jfz)
        # # 将高斯分布应用到第三个维度
        # self.nbi_in = self.nbi_in * gaussian[np.newaxis, np.newaxis, :, np.newaxis]
        # # 验证结果
        # print("nbi_in形状:", self.nbi_in.shape)  # 应该输出(nspec, Ns, nmu, nrdpsi)
        # print("高斯分布积分值:", np.sum(gaussian)) 

        # # 导入NBI光束 在μ上的 吸收功率分布函数设置
        # self.S_nbi_ku = np.loadtxt('AUG_17245_NBI.txt').T # 导入数据并转置
        # # self.S_nbi_ku = self.S_nbi_ku.T
        # print(f"S_nbi_ku形状: {self.S_nbi_ku.shape}")
        # # self.S_nbi_ku *= 0.58
        # # 广播到其他维度
        # self.nbi_in = self.nbi_in * self.S_nbi_ku[np.newaxis, :, :, np.newaxis]

        # 参数初始化
        self.S_nbi_ku = np.zeros((self.Ns, self.nmu))   # S(k,μ) 磁面位置ψ、粒子种类为i，对应的功率沉积分布函数
        self.sig = np.zeros((self.Ns, self.legmod + 1))    # σ(k,n) 粒子功率沉积分布函数S(k,μ)勒让德展开后整合的函数
        self.source_nbi = np.zeros((self.Ns, self.legmod + 1))    # # NBI源项部分
        
        # 输出初始化信息
        for ipsi in range(self.nrdpsi):
            # 初始化第一组数组
            for isp in range(self.ctx.nspec):
                self.thrate[isp, ipsi] = 0.0
                self.thtprp[isp, ipsi] = 0.0
                self.thtpar[isp, ipsi] = 0.0
                self.thttot[isp, ipsi] = 0.0
                self.tavgtx[isp, ipsi] = 0.0
                self.tavgtz[isp, ipsi] = 0.0
                self.tavbkx[isp, ipsi] = 0.0
                self.tavbkz[isp, ipsi] = 0.0
                self.tmtail[isp, ipsi] = 0.0
            # 初始化第二组数组
            for isp in range(self.ctx.nspec):
                self.pwgain[0, isp, ipsi] = 0.0  
                # self.pwgain[1, isp, ipsi] = self.pwredf[isp, ipsi] + self.pwredh[isp, ipsi]  # 被注释的代码
                self.pwgain[1, isp, ipsi] = 0.0  
                self.tab_pwexch[isp, ipsi] = 0.0
            # 设置nspele位置的值
            self.pwgain[0, self.nspele-1, ipsi] = 0.0  
            self.pwgain[1, self.nspele-1, ipsi] = self.pwredl[ipsi]
            self.tab_pwexch[self.nspele-1, ipsi] = 0.0
        
        #计算勒让德多项式
        self.dxmu = (self.mu_max - self.mu_min) / (self.nmu - 1)
        if self.legmod > 0:
            self.nleg = self.legmod
            for imu in range(self.nmu):
                # 计算xmu值
                xmu = self.mu_min + imu * self.dxmu  
                # 调用 poledg 函数计算勒让德多项式
                self.poledg(xmu, self.nleg)
                # 存储第1阶到第legmod阶的勒让德多项式值
                for n in range(self.legmod+1):
                    if n == 0:
                        self.plgcmu[n, imu] = 1.0  # P_0(mu) = 1
                    else:
                        self.plgcmu[n, imu] = self.plg[n-1] 

        # 保存勒让德系数至txt文件
        plgcmu_new = self.plgcmu.copy()
        plgcmu_new[plgcmu_new > 0] = 0
        with open('01_Pn(u).txt', 'w') as f:
            f.write("=== Pn(u) ===\n")
            f.write(f"u*n = {plgcmu_new.shape[1]}*{plgcmu_new.shape[0]}\n\n")
            
            for i in range(plgcmu_new.shape[1]):
                row_values = []
                for j in range(plgcmu_new.shape[0]):
                    val = plgcmu_new[j, i]
                    if val < 0:
                        row_values.append(f"[{val:.2e}]")
                    else:
                        row_values.append("[---------]")
                f.write(f"Row {i}: {' '.join(row_values)}\n")

        self.xmu = np.array([self.mu_min + imu * self.dxmu for imu in range( self.nmu)])

        # Aux arrays
        self.anucol = np.zeros(self.nspele)
        self.anuisp = np.zeros(self.nspele)
        self.vth = np.zeros(self.nspele)
        self.vthisp = np.zeros(self.nspele)
        self.tauisp = np.zeros(self.nspele)
        self.expuab = np.zeros(self.nspele)
        self.phifun = np.zeros(self.nspele)
        self.thefun = np.zeros(self.nspele)
    
    # def inicol(self, isp, ipsi):
    #     """在每个网格点进行初始化操作
        
    #     Args:
    #         ipsi: 空间网格点索引
    #     """
    #     self.ibreak = 0
        
    #     ipsi = ipsi-1

    #     # 【关键修改】：根据径向位置 ipsi 动态设置 NBI 目标功率
    #     if self.nbi_enabled:
    #         if self.nbi_power_profile is not None:
    #             # 确保索引不越界 (ipsi 是 1-based，数组是 0-based)
    #             idx = ipsi
    #             if 0 <= idx < len(self.nbi_power_profile):
    #                 self.pwdens_nbi = self.nbi_power_profile[idx]
    #             else:
    #                 self.pwdens_nbi = 0.0 # 超出范围设为0
    #         # else: 保持 __init__ 中的默认值

    #     # 每个网格点的初始化
    #     self.ibmod = -1
    #     psi = self.redpsi[ipsi]
    #     self.t4_surfpl(psi,self.ibmod)

    #     # 设置各物种的密度和温度
    #     for isp in range(self.ctx.nspec):
    #         self.ctx.xfni[isp] = self.zdeni[isp, ipsi]
    #         self.ctx.tempi[isp] = self.ztempi[isp, ipsi]
        
    #     isp = self.nspele-1
    #     dens = self.zdeni[self.nspele-1, ipsi]
    #     tempe = self.ztempi[self.nspele-1, ipsi]
    #     self.ctx.dens = dens
    #     self.ctx.tempi[self.nspele-1] = tempe
        
    #     for isp in range(self.ctx.nspec):
    #         self.ctx.aconc[isp] = self.ctx.xfni[isp] / dens
    #     # print(f"粒子温度：{self.ztempi}")
    #     # print(f"粒子密度：{self.zdeni}")

    #     # 速度网格（等间距点）
    #     if self.ipheat == 1:
    #         emax = 100.0 * self.ctx.tempi[self.ispres]
    #     else:
    #         emax = 200.0 * self.ctx.tempi[self.ispres]

    #     # 局部u-网格评估和参考麦克斯韦分布
    #     self.umax = np.sqrt(emax / self.ctx.tempi[self.ispres])
    #     self.du = self.umax / (self.nptu - 1)
    #     print(f"速度步长 du : {self.du}")
    #     for ip in range(self.nptu):
    #         u = ip * self.du
    #         self.tbkev[ip] = u * u * self.ctx.tempi[self.ispres]
    #         self.tbu[ip] = u
        
    #     zn = np.log10(5.56832799683)
    #     zm = 0.4342944819

    #     fe = np.array([-(zm * self.tbu[ip] ** 2 + zn) for ip in range(self.nptu)])

    #     self.ref_maxwell = fe

    #     # 写入速度参数到文件
    #     append_to_file(self.output_paths["read"], f"umax = {self.umax}\n")
    #     append_to_file(self.output_paths["read"], f"du = {self.du}\n")
    #     append_to_file(self.output_paths["read"], f"v = \n {self.tbu}\n")


    #     # 参考麦克斯韦分布
    #     for ip in range(self.nptu):
    #         u = self.tbu[ip]
    #         self.tbfu[ip, 0] = np.exp(-u * u) / 5.56832799683
    #         self.tbdfu[ip, 0] = -2.0 * u * self.tbfu[ip, 0]
    #         # self.tbdfu[ip, 0] = 2.0 * u * self.tbfu[ip, 0]

    #         for n in range(1, self.legmod + 1):
    #             self.tbfu[ip, n] = 0.0
    #             self.tbdfu[ip, n] = 0.0

    #     # # 碰撞频率，速度比
    #     # albcol = 30.9 + np.log(tempe) - 0.5 * np.log(1.0E13 * dens)
    #     # self.zcolli = (1.434E2 * albcol * dens * self.ctx.azi[self.ispres]**2 / 
    #     #             np.sqrt(self.ctx.atm[self.ispres] * self.ctx.tempi[self.ispres]**3))

    #     # for isp in range(self.ctx.nspec):
    #     #     self.anucol[isp] = self.zcolli * self.ctx.aconc[isp] * self.ctx.azi[isp]**2
    #     #     self.vth[isp] = 4.37695E7 * np.sqrt(self.ctx.tempi[isp] / self.ctx.atm[isp])

    #     # self.anucol[self.nspele-1] = self.zcolli

    #     # ------------------------------------------------------
    #     pi = 3.1415926535897931
    #     sqrtwo = 1.4142135623731
    #     ELCHAR = 4.8032E-10  # 库仑
    #     PTMASS = 1.6726E-24  # 千克
    #     PTMCSQ = 938231  # 质子的静止能量，换算为焦耳
    #     VLIGHT = 2.9979E+10  # 米/秒

    #     # 计算表达式
    #     zcolcf = (2 * pi / sqrtwo) * (ELCHAR ** 4 / PTMASS ** 2) * ((np.sqrt(PTMCSQ) / VLIGHT) ** 3) * 1e13

    #     albcol = np.zeros(self.nspele)
    #     for isp in range(self.ctx.nspec):
    #         albcol[isp] = (23.0 - 0.5 * np.log(1.0e13 * dens) -
    #                        np.log(self.ctx.azi[self.ispres] * self.ctx.azi[isp] *
    #                               (self.ctx.atm[self.ispres] + self.ctx.atm[isp]) * 1.0e-3 /
    #                               (self.ctx.atm[self.ispres] * self.ctx.tempi[isp] + self.ctx.atm[isp] * self.ctx.tempi[self.ispres])) -
    #                        0.5 * np.log((self.ctx.aconc[isp] * self.ctx.azi[isp] ** 2 * 1.0e-3 / self.ctx.tempi[isp] +
    #                                      self.ctx.aconc[self.ispres] * self.ctx.azi[self.ispres] ** 2 * 1.0e-3 / self.ctx.tempi[self.ispres])))
    #     albcol[self.nspele - 1] = 30.9 + np.log(self.ctx.tempe) - 0.5 * np.log(1.E13 * dens)
    #     zcolli = zcolcf * dens * self.ctx.azi[self.ispres] ** 2 / np.sqrt(self.ctx.atm[self.ispres] * self.ctx.tempi[self.ispres] ** 3)

    #     for isp in range(self.ctx.nspec):
    #         self.anucol[isp] = zcolli * albcol[isp] * self.ctx.aconc[isp] * self.ctx.azi[isp] ** 2
    #         self.vth[isp] = 4.37695E7 * np.sqrt(self.ctx.tempi[isp] / self.ctx.atm[isp])

    #     self.anucol[self.nspele - 1] = zcolli * albcol[self.nspele - 1]

    #     append_to_file(self.output_paths["read"], f"第ispres = {self.ispres+1}个粒子\n")

    #     append_to_file(self.output_paths["read"], f"zcolcf = {zcolcf}\n")
    #     append_to_file(self.output_paths["read"], f"电子密度dens = {dens}\n")
    #     append_to_file(self.output_paths["read"], f"粒子质量atm = {self.ctx.atm[self.ispres]}\n")
    #     append_to_file(self.output_paths["read"], f"粒子温度tempi = {self.ctx.tempi[self.ispres]}\n")

    #     append_to_file(self.output_paths["read"], f"粒子浓度组分aconc = {self.ctx.aconc}\n")
    #     append_to_file(self.output_paths["read"], f"粒子电荷azi = {self.ctx.azi[self.ispres]}\n")

    #     append_to_file(self.output_paths["read"], f"zcolli = {zcolli}\n")
    #     append_to_file(self.output_paths["read"], f"albcol = {albcol}\n")
    #     append_to_file(self.output_paths["read"], f"anucol = {self.anucol}\n")
        

    #     # ------------------------------------------------------
    #     self.vth[self.nspele-1] = 1.87552E9 * np.sqrt(tempe)

    #     self.anuref = 0.0
    #     for isp in range(self.nspele):
    #         self.anuref += self.anucol[isp]

    #     for isp in range(self.nspele):
    #         self.anuisp[isp] = self.anucol[isp] / self.anuref
    #         self.tauisp[isp] = self.ctx.tempi[self.ispres] / self.ctx.tempi[isp]
    #         self.vthisp[isp] = self.vth[self.ispres] / self.vth[isp]

    #     # 初始化碰撞算子和源项（如果稳态温度不同）
    #     #eqn(26)(27)(29)(30)  vᵢ nᵢ Tᵢ
    #     self.zemxw = 1.609E-3 * dens * self.ctx.aconc[self.ispres] * \
    #                 self.ctx.tempi[self.ispres] * self.anuref
        
    #     append_to_file(self.output_paths["read"], f"anuref = {self.anuref}\n")
    #     append_to_file(self.output_paths["read"], f"zemxw = {self.zemxw}\n")

    #     # # 磁面位置ψ、粒子种类为i，对应的功率沉积分布函数（NBI程序输入参数）
    #     # if self.nbi_enabled:
    #     #     self.S_nbi_ku = np.zeros((self.Ns, self.nmu))
    #     #     for k in range(self.Ns):
    #     #         self.v_nbi_k[k] = self.vnbi[self.ispres, k]
    #     #         # self.E_nbi_k[k] = 0.5*self.ctx.atm[self.ispres]*(self.v_nbi_k[k])**2   # E=(1/2)mv^2
    #     #         self.E_nbi_k[k] = (self.v_nbi_k[k])**2 * self.ctx.tempi[self.ispres]          # E=kT*(v/v_th)^2，直接用温度和速度比计算能量
    #     #         for u in range(self.nmu):
    #     #             self.S_nbi_ku[k,u] = self.nbi_in[self.ispres, k, u, ipsi]

            
    #     #     self.plot_S_nbi_ku()

    #     append_to_file(self.output_paths["read"], f"粒子温度  = {self.ctx.tempi}\n")
    #     for ip in range(self.nptu):
    #         u = self.tbu[ip]

    #         # 调用福克-普朗克算子
    #         self.fokker(u)

    #         self.tbpsic[ip] = self.psicol
    #         self.tbpsit[ip] = self.psitau
    #         self.tbthec[ip] = self.thecol
    #         self.tbpsie[ip] = self.psiele
    #         for isp in range(self.nspele):
    #             self.tbphif[ip, isp] = self.anuisp[isp] * self.phifun[isp]

    #         # 调用源项
    #         sa = self.nbi_enabled
    #         self.nbi_enabled = 0     
    #         self.source(u)
    #         self.nbi_enabled = sa
    #         self.tab_source[ip] = self.psourc

    #     k = -2
    #     self.zdotn = 4.0 * np.pi * self.zemxw * self.qlmomu(self.tab_source, self.du, self.nptu, k)
    #     k = 0
    #     self.pwsrce = 4.0 * np.pi * self.zemxw * self.qlmomu(self.tab_source, self.du, self.nptu, k)

    #     # 在没有高频加热的情况下向其他物种的功率转移
    #     for isp in range(self.nspele):
    #         if self.ctx.tempi[isp] != self.ctx.tempi[self.ispres]:
    #             self.pwexch[0, isp] = -self.zemxw * self.exchcl(isp)
    #             self.pwgain[0, isp, ipsi] -= self.pwexch[0, isp]
    #         else:
    #             self.pwexch[0, isp] = 0.0
    #     # append_to_file(self.output_paths["read"], f"没有高频加热的情况下加热曲线 pwexch = {self.pwexch}\n")
    #     # append_to_file(self.output_paths["read"], f"没有高频加热的情况下加热曲线 pwgain = {self.pwgain}\n")

        

    #     isp = -1
    #     self.pwcoll[0] = self.zemxw * self.exchcl(isp)

        
    #     # NBI参数
    #     if self.nbi_enabled:
    #         self.S_nbi_ku = np.zeros((self.Ns, self.nmu))
    #         for k in range(self.Ns):
    #             # ================= 【动态计算归一化速度】 =================
    #             # 1. 使用当前半径的局部温度，重新计算归一化注入速度 u_inj = sqrt(E_inj / T_local)
    #             self.v_nbi_k[k] = np.sqrt(self.nbi_energies_keV[k] / self.ctx.tempi[self.ispres])
    #             # self.E_nbi_k[k] = 0.5*self.ctx.atm[self.ispres]*(self.v_nbi_k[k])**2   # E=(1/2)mv^2
                
    #             # 2. 反推能量用于诊断和后续功率计算 (E = u^2 * T_local，完美闭环等于设定值)
    #             self.E_nbi_k[k] = (self.v_nbi_k[k])**2 * self.ctx.tempi[self.ispres]
    #             # ==================================================================
                
    #             for u in range(self.nmu):
    #                 self.S_nbi_ku[k,u] = self.nbi_in[self.ispres, k, u, ipsi]
            
    #         # 【画图】：在数据计算完成后再画图，此时 E_nbi_k 是准确的
    #         # self.plot_S_nbi_ku()

    #         # 计算 sig
    #         self.sig_nbi()
            
            
    #         # ================= 【修正：使用物理注入速率进行归一化（推荐） =================
    #         # 1. 计算主束 (k=0) 的物理能量 (单位: J)
    #         E_beam_J = self.nbi_energies_keV[0] * 1.602E-16 
    #         if E_beam_J > 1e-30:
    #             self.n_dot_nbi = self.pwdens_nbi / E_beam_J
    #         else:
    #             self.n_dot_nbi = 0.0
            
    #         # 2. 计算量纲转换系数 
    #         zconv_nbi = 1.609E-3 * self.zdeni[self.nspele-1, ipsi] * \
    #                     self.ctx.aconc[self.ispres] * self.ctx.tempi[self.ispres]
            
    #         # 3. 计算原始物理功率
    #         # raw_phys_pwdens = zconv_nbi * sum(self.E_nbi_k[k] * self.sig[k, 0] for k in range(self.Ns))
    #         raw_phys_pwdens = self.zemxw
            
    #         # 4. 计算归一化系数
    #         if abs(raw_phys_pwdens) > 1.0e-20:
    #             norm_coeff = self.pwdens_nbi / raw_phys_pwdens
    #         else:
    #             norm_coeff = 1.0
                
    #         # 5. 将 sig 缩放到目标功率
    #         self.sig = self.sig * norm_coeff
            
    #         # 6. 将 dnbi 永远设为 1
    #         self.dnbi = 1.0
            
    #         print(f"  [NBI 归一化] 注入速率: {self.n_dot_nbi:.4e} /s/cm^3, 原始功率: {raw_phys_pwdens:.4e}, 目标功率: {self.pwdens_nbi:.4e}, 缩放系数: {norm_coeff:.4e}")
    #         # =================================================================

    #         # 计算NBI源功率
    #     if self.nbi_enabled:
    #         for k in range(self.Ns):
    #             # self.n_i_nbi += self.sig[k,0]
    #             self.pwnbi +=  self.E_nbi_k[k]*self.sig[k,0]
    #             self.pwnbi_l -= 1.5*self.sig[k,0]*self.ctx.tempi[self.ispres]
    #     # Init collision operator tables (tbpsic, etc would be arrays in full implementation)
    #     # Here we just calculate source for now
    #     # Note: In full code tbpsic etc are arrays. I will implement them as needed in fokker/mastfp
        
    #     # Source term calculation
    #     # ... (Simplified: assuming source logic handled in matrix construction)
    
    def inicol(self, isp, ipsi):
        """在每个网格点进行初始化操作，确保NBI能量自洽"""
        self.ibreak = 0
        ipsi = ipsi - 1
        
        # 【关键修改】：根据径向位置 ipsi 动态设置 NBI 目标功率
        if self.nbi_enabled:
            if self.nbi_power_profile is not None:
                idx = ipsi
                if 0 <= idx < len(self.nbi_power_profile):
                    self.pwdens_nbi = self.nbi_power_profile[idx]
                else:
                    self.pwdens_nbi = 0.0
        
        # 每个网格点的初始化
        self.ibmod = -1
        psi = self.redpsi[ipsi]
        self.t4_surfpl(psi, self.ibmod)
        
        # 设置各物种的密度和温度
        for isp in range(self.ctx.nspec):
            self.ctx.xfni[isp] = self.zdeni[isp, ipsi]
            self.ctx.tempi[isp] = self.ztempi[isp, ipsi]
        
        # 设置电子参数
        isp = self.nspele - 1
        dens = self.zdeni[self.nspele-1, ipsi]
        tempe = self.ztempi[self.nspele-1, ipsi]
        self.ctx.dens = dens
        self.ctx.tempi[self.nspele-1] = tempe
        self.ctx.tempe = tempe
        
        # 设置浓度
        for isp in range(self.ctx.nspec):
            self.ctx.aconc[isp] = self.ctx.xfni[isp] / dens
        
        # 速度网格初始化
        if self.ipheat == 1:
            emax = self.r_E_1 * self.ctx.tempi[self.ispres]
        else:
            emax = self.r_E_2 * self.ctx.tempi[self.ispres]
        
        self.umax = np.sqrt(emax / self.ctx.tempi[self.ispres])
        self.du = self.umax / (self.nptu - 1)
        
        for ip in range(self.nptu):
            u = ip * self.du
            self.tbkev[ip] = u * u * self.ctx.tempi[self.ispres]
            self.tbu[ip] = u
        
        zn = np.log10(5.56832799683)
        zm = 0.4342944819

        fe = np.array([-(zm * self.tbu[ip] ** 2 + zn) for ip in range(self.nptu)])

        self.ref_maxwell = fe

        # 参考麦克斯韦分布
        for ip in range(self.nptu):
            u = self.tbu[ip]
            self.tbfu[ip, 0] = np.exp(-u * u) / 5.56832799683
            self.tbdfu[ip, 0] = -2.0 * u * self.tbfu[ip, 0]
            for n in range(1, self.legmod + 1):
                self.tbfu[ip, n] = 0.0
                self.tbdfu[ip, n] = 0.0
        
        # 计算碰撞参数
        pi = 3.1415926535897931
        sqrtwo = 1.4142135623731
        ELCHAR = 4.8032E-10
        PTMASS = 1.6726E-24
        PTMCSQ = 938231
        VLIGHT = 2.9979E+10
        
        zcolcf = (2 * pi / sqrtwo) * (ELCHAR ** 4 / PTMASS ** 2) * ((np.sqrt(PTMCSQ) / VLIGHT) ** 3) * 1e13
        
        albcol = np.zeros(self.nspele)
        for isp in range(self.ctx.nspec):
            albcol[isp] = (23.0 - 0.5 * np.log(1.0e13 * dens) -
                        np.log(self.ctx.azi[self.ispres] * self.ctx.azi[isp] *
                                (self.ctx.atm[self.ispres] + self.ctx.atm[isp]) * 1.0e-3 /
                                (self.ctx.atm[self.ispres] * self.ctx.tempi[isp] +
                                self.ctx.atm[isp] * self.ctx.tempi[self.ispres])) -
                        0.5 * np.log((self.ctx.aconc[isp] * self.ctx.azi[isp] ** 2 * 1.0e-3 / self.ctx.tempi[isp] +
                                    self.ctx.aconc[self.ispres] * self.ctx.azi[self.ispres] ** 2 * 1.0e-3 / self.ctx.tempi[self.ispres])))
        albcol[self.nspele - 1] = 30.9 + np.log(self.ctx.tempe) - 0.5 * np.log(1.E13 * dens)
        zcolli = zcolcf * dens * self.ctx.azi[self.ispres] ** 2 / np.sqrt(self.ctx.atm[self.ispres] * self.ctx.tempi[self.ispres] ** 3)
        
        for isp in range(self.ctx.nspec):
            self.anucol[isp] = zcolli * albcol[isp] * self.ctx.aconc[isp] * self.ctx.azi[isp] ** 2
            self.vth[isp] = 4.37695E7 * np.sqrt(self.ctx.tempi[isp] / self.ctx.atm[isp])
        
        self.anucol[self.nspele - 1] = zcolli * albcol[self.nspele - 1]
        self.vth[self.nspele - 1] = 1.87552E9 * np.sqrt(tempe)
        
        self.anuref = 0.0
        for isp in range(self.nspele):
            self.anuref += self.anucol[isp]
        
        for isp in range(self.nspele):
            self.anuisp[isp] = self.anucol[isp] / self.anuref
            self.tauisp[isp] = self.ctx.tempi[self.ispres] / self.ctx.tempi[isp]
            self.vthisp[isp] = self.vth[self.ispres] / self.vth[isp]
        
        # 初始化碰撞算子
        self.zemxw = 1.602176634E-3 * dens * self.ctx.aconc[self.ispres] * \
                    self.ctx.tempi[self.ispres] * self.anuref
        
        # 计算福克-普朗克算子和源项
        for ip in range(self.nptu):
            u = self.tbu[ip]
            self.fokker(u)
            self.tbpsic[ip] = self.psicol
            self.tbpsit[ip] = self.psitau
            self.tbthec[ip] = self.thecol
            self.tbpsie[ip] = self.psiele
            for isp in range(self.nspele):
                self.tbphif[ip, isp] = self.anuisp[isp] * self.phifun[isp]
            
            # 计算源项
            sa = self.nbi_enabled
            self.nbi_enabled = 0
            self.source(u)
            self.nbi_enabled = sa
            self.tab_source[ip] = self.psourc
        
        # Use the same source samples as the weak-form RHS. zdotn is the
        # dimensionless particle-rate moment; pwsrce is an energy rate.
        particle_source = 0.0
        energy_source = 0.0
        nbi_enabled = self.nbi_enabled
        self.nbi_enabled = 0
        for element in range(self.nfelem):
            for point, weight in zip(self.zgaus, self.wgaus):
                u = self.tbu[element] + self.du * point
                self.fokker(u)
                self.source(u)
                particle_source += self.du * weight * self.psourc
                energy_source += self.du * weight * u**2 * self.psourc
        self.nbi_enabled = nbi_enabled
        self.zdotn = 4 * np.pi * particle_source
        self.pwsrce = 4 * np.pi * self.zemxw * energy_source
        
        # 计算能量交换
        for isp in range(self.nspele):
            if self.ctx.tempi[isp] != self.ctx.tempi[self.ispres]:
                self.pwexch[0, isp] = -self.zemxw * self.exchcl(isp)
                self.pwgain[0, isp, ipsi] -= self.pwexch[0, isp]
            else:
                self.pwexch[0, isp] = 0.0
        
        isp = -1
        self.pwcoll[0] = self.zemxw * self.exchcl(isp)
        
        # sig[k,n] has units W cm^-3 keV^-1. Its zeroth coefficient is
        # P_k/E_k; higher coefficients preserve the supplied angular shape.
        self.pwnbi = 0.0
        self.pwnbi_l = 0.0
        if self.nbi_enabled:
            if not np.isfinite(self.pwdens_nbi) or self.pwdens_nbi < 0:
                raise ValueError("NBI power density must be finite and nonnegative")
            energies = np.asarray(self.nbi_energies_keV, dtype=float)
            if energies.shape != (self.Ns,) or np.any(~np.isfinite(energies)) or np.any(energies <= 0):
                raise ValueError("Each NBI beam requires one positive finite energy in keV")
            self.E_nbi_k = energies.copy()
            self.v_nbi_k = np.sqrt(energies / self.ctx.tempi[self.ispres])
            self.S_nbi_ku = self.nbi_in[self.ispres, :, :, ipsi].copy()
            if self.nbi_file is not None:
                self.S_nbi_native = self.nbi_native_profiles[:, :, ipsi]
            if np.any(~np.isfinite(self.S_nbi_ku)) or np.any(self.S_nbi_ku < 0):
                raise ValueError("NBI angular power profiles must be finite and nonnegative")
            self.sig_nbi()
            relative_power = self.sig[:, 0].sum()
            if self.pwdens_nbi > 0 and relative_power <= 0:
                raise ValueError("Positive NBI power requires a nonzero angular source profile")
            if relative_power > 0:
                self.sig *= self.pwdens_nbi / (relative_power * energies[:, None])
            else:
                self.sig.fill(0.0)
            self.dnbi = 1.0
            self.pwnbi = float(np.dot(energies, self.sig[:, 0]))
        self._prepare_nbi_source()
        append_to_file(
            self.output_paths["read"],
            f"NBI 物理输入功率 = {self.pwnbi:.12e} W/cm³\n"
            f"NBI 注入粒子率 = {self.nbi_particle_injection_rate:.12e} cm^-3 s^-1\n"
            f"NBI Maxwell 热粒子汇功率 = {-self.nbi_sink_power:.12e} W/cm³\n"
        )
            
            # # 计算物理注入速率
            # E_beam_J = self.nbi_energies_keV[0] * 1.602E-16
            # if E_beam_J > 1e-30:
            #     self.n_dot_nbi = self.pwdens_nbi / E_beam_J
            # else:
            #     self.n_dot_nbi = 0.0
            
            # # 计算归一化系数
            # self.zconv_nbi = 1.609E-3 * self.zdeni[self.nspele-1, ipsi] * \
            #             self.ctx.aconc[self.ispres] * self.ctx.tempi[self.ispres]
            
            # # self.zconv_nbi = self.zemxw
            # raw_power = self.zconv_nbi * sum(self.E_nbi_k[k] * self.sig[k, 0] for k in range(self.Ns))
            # # self.pwsrce = 4.0 * np.pi * self.zemxw * self.qlmomu(self.tab_source, self.du, self.nptu, k)
            
            # if abs(raw_power) > 1.0e-20:
            #     norm_coeff = self.pwdens_nbi / raw_power
            # else:
            #     norm_coeff = 1.0
            
            # # 应用归一化
            # self.sig = self.sig * norm_coeff
            # self.dnbi = 1.0
            
            # # 计算NBI功率
            # self.pwnbi = 0.0
            # self.pwnbi_l = 0.0
            # for k in range(self.Ns):
            #     self.pwnbi += self.E_nbi_k[k] * self.sig[k, 0]
            #     self.pwnbi_l -= 1.5 * self.sig[k, 0] * self.ctx.tempi[self.ispres]
            
            # # 验证能量守恒
            # expected_power = self.pwdens_nbi
            # actual_power = self.pwnbi * self.zconv_nbi
            # energy_error = abs(actual_power - expected_power) / expected_power
            
            # if energy_error > 1e-6:
            #     print(f"警告：NBI能量不守恒，误差: {energy_error:.2e}")
                # 可以在这里添加额外的修正措施

            # # ================= 【新增：初始化预平衡机制】 =================
            # # 此时 pwcoll 是基于纯麦氏分布算出的(很小)，而 pwnbi 很大。
            # # 为了防止第一次求解矩阵时源项过大导致数值爆炸，
            # # 我们需要预估稳态下的碰撞功率，并调整初始的 fb (分布函数边界条件)。
            
            # # 稳态下，NBI功率主要通过与电子和离子的碰撞耗散：
            # # P_nbi ≈ P_coll_e + P_coll_i
            # # 我们可以简单地认为，初始时刻的碰撞耗散能力等于NBI注入功率
            # # （这是一种一阶近似，帮助迭代起步）
            
            # # 将NBI功率转化为无量纲单位，与 pwcoll 同量纲
            # pwnbi_dimensionless = self.pwnbi * (self.zconv_nbi / self.zemxw) if self.zemxw > 0 else 0.0
            
            # # 计算初始功率不平衡量
            # initial_imbalance = pwnbi_dimensionless + self.pwsrce - self.pwcoll[0]
            
            # # 将不平衡量反馈给源项平衡变量，供 boundv 迭代使用
            # self.zbalpw = initial_imbalance
            
            # # 诊断输出
            # print(f"  [初始化预平衡] 麦氏碰撞功率: {self.pwcoll[0]:.4e} (无量纲)")
            # print(f"  [初始化预平衡] NBI注入功率: {pwnbi_dimensionless:.4e} (无量纲)")
            # print(f"  [初始化预平衡] 初始不平衡量: {initial_imbalance:.4e} (将在迭代中由快离子碰撞耗散)")
            # # =================================================================


    def plot_S_nbi_ku(self):
        """绘制 S_nbi_ku 的 2D 热力图和 1D 剖面图"""
        if not hasattr(self, 'S_nbi_ku') or self.S_nbi_ku is None:
            print("⚠️ S_nbi_ku 尚未计算，无法绘图。")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.2, 1]})
        
        mu_values = np.linspace(self.mu_min, self.mu_max, self.nmu)
        k_values = np.arange(self.Ns)
        
        # ==========================================
        # 子图 1: 2D 热力图 (k vs mu)
        # ==========================================
        # 注意：pcolormesh 要求 X, Y 是二维网格，Z 是对应的数据
        MU_GRID, K_GRID = np.meshgrid(mu_values, k_values)
        
        pcm = ax1.pcolormesh(MU_GRID, K_GRID, self.S_nbi_ku, cmap='hot', shading='auto')
        fig.colorbar(pcm, ax=ax1, label=r'$S_{nbi}(k, \mu)$')

        
        ax1.set_xlabel(r'Pitch Angle $\mu$')
        ax1.set_ylabel(r'Beam Index $k$')
        ax1.set_title(r'NBI Power Deposition Profile $S_{nbi}(k, \mu)$')
        ax1.set_yticks(k_values)  # 确保纵轴只显示整数光束索引
        
        # ==========================================
        # 子图 2: 1D 剖面图 (沿 mu 方向展开)
        # ==========================================
        colors = plt.cm.viridis(np.linspace(0, 0.9, self.Ns)) # 为不同光束分配颜色
        
        for k in range(self.Ns):
            ax2.plot(mu_values, self.S_nbi_ku[k, :], color=colors[k], linewidth=2, 
                    label=f'Beam k={k} (E={self.E_nbi_k[k]:.2f})')
            
        ax2.set_xlabel(r'Pitch Angle $\mu$')
        ax2.set_ylabel(r'$S_{nbi}(k, \mu)$')
        ax2.set_title(r'Distribution along $\mu$ for each beam')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 如果之前设置了负数中心点，可以通过添加竖线标出 0 点位置
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label=r'$\mu=0$')
        
        plt.tight_layout()
        plt.show()
    def iniqln(self, ispec, ipsi):
        """初始化准线性算子
        
        Args:
            ispec: 物种索引
            ipsi: 空间网格点索引
        """
        # 初始化局部变量
        isp = ip = k = legsup = 0
        znpar = znpar2 = bmodx = ohe = 0.0
        ar = al = ass = xiumax = zdqlw = xiu = u = 0.0

        # 设置第一个网格点
        if ipsi <= 0:
            # 检查ispec物种的离子是否存在共振
            self.ibreak = 0
            self.ispres = ispec

            if self.ipheat == 1:
                if self.ctx.psfund[self.ispres] < -1.0:
                    self.ibreak = -99
                    print(f"粒子{self.ispres}无基频吸收")
            else:
                if self.ctx.psharm[self.ispres] < -1.0:
                    self.ibreak = -99
                    print(f"粒子{self.ispres:3d}无谐波吸收")

            # 局部u-网格评估和参考麦克斯韦分布
            if self.ipheat == 1:
                self.xqres = self.ctx.rcfund[self.ispres]
            else:
                self.xqres = self.ctx.rcharm[self.ispres]

            self.fb1_prv = 0.0
            return

        # 每个网格点的初始化
        ipsi = ipsi-1

        # 设置功率密度
        if self.ipheat == 1:
            # 如果开启ICRH，使用吸收功率；否则设为0
            self.pwdens = self.pwredf[self.ispres, ipsi] if self.icrh_enabled else 0.0
        if self.ipheat == 2:
            # 如果开启ICRH，使用吸收功率；否则设为0
            self.pwdens = self.pwredh[self.ispres, ipsi] if self.icrh_enabled else 0.0

        # # 设置功率密度
        # if self.ipheat == 1:
        #     # 吸收功率设置为0.7839
        #     self.pwdens = self.pwredf[self.ispres, ipsi]
        # if self.ipheat == 2:
        #     # 吸收功率设置为0.7839
        #     self.pwdens = self.pwredh[self.ispres, ipsi]

        if ispec != self.nspele:
            self.htemp = self.ctx.tempi[self.ispres]
            self.htperp = self.ctx.tempi[self.ispres]
            self.htpar = self.ctx.tempi[self.ispres]
            self.ttailx = self.ctx.tempi[self.ispres]
            self.ttailz = self.ctx.tempi[self.ispres]
            self.ttblkx = self.ctx.tempi[self.ispres]
            self.ttblkz = self.ctx.tempi[self.ispres]
            self.thtprp[self.ispres, ipsi] = self.ctx.tempi[self.ispres]
            self.thtpar[self.ispres, ipsi] = self.ctx.tempi[self.ispres]
            self.thttot[self.ispres, ipsi] = self.ctx.tempi[self.ispres]
            self.tavgtx[self.ispres, ipsi] = self.ctx.tempi[self.ispres]
            self.tavgtz[self.ispres, ipsi] = self.ctx.tempi[self.ispres]
            self.tmtail[self.ispres, ipsi] = 0.0

            if self.ipheat == 1:
                self.emax = 50.0 * self.ctx.tempi[self.ispres]
                self.xqres = self.ctx.rcfund[self.ispres]
            else:
                self.emax = 30.0 * self.ctx.tempi[self.ispres]
                self.xqres = self.ctx.rcharm[self.ispres]

        # 计算共振处的垂直指数,只考虑一个n//，不进行m_th修正
        print(f"计算共振处的垂直指数 ")
        self.znpar = self.nphi / (self.ctx.ukzero * self.xqres)
        self.znpar2 = self.znpar * self.znpar
        bmodx = self.ctx.baxis * self.ctx.raxis / self.xqres

        ohe = self.ctx.xomce * bmodx
        ar = 1.0 - self.ctx.ompe2 / (1.0 - ohe)
        al = 1.0 - self.ctx.ompe2 / (1.0 + ohe)
        
        for isp in range(self.ctx.nspec):
            self.ctx.ohi[isp] = self.ctx.xomci[isp] * bmodx
            ar = ar - self.ctx.ompi2[isp] / (1.0 + self.ctx.ohi[isp])
            al = al - self.ctx.ompi2[isp] / (1.0 - self.ctx.ohi[isp])
        
        ass = 0.5 * (ar + al)

        self.qnfsq = -(self.znpar2 - ar) * (self.znpar2 - al) / (self.znpar2 - ass)

        if self.qnfsq <= 0.0:
            print(f"{' '*5}Warning: resonance in evanescent region for psi ={self.redpsi[ipsi]:8.3f}")
            self.qnfsq = -self.qnfsq

        # xiperp = np.sqrt(self.qnfsq) * self.ipheat * self.vth[self.ispres] / self.ctx.vlight


        # Bessel 参数使用离子回旋角频率；vth 为 cm/s，kperp 为 m^-1。
        if self.frequency_convention == "rf_hz":
            cyclotron_angular_frequency = 2 * np.pi * self.omega_ci / self.ipheat
        elif self.frequency_convention == "cyclotron_hz":
            cyclotron_angular_frequency = 2 * np.pi * self.omega_ci
        elif self.frequency_convention == "cyclotron_rad_s":
            cyclotron_angular_frequency = self.omega_ci
        else:
            raise ValueError(f"Unknown frequency_convention: {self.frequency_convention}")
        xiperp = self.kperp * self.vth[self.ispres] * 1e-2 / cyclotron_angular_frequency
        
        self.xiperp = xiperp
        # self.xiperp = 73.7
        xiumax = self.xiperp * self.umax
        self.xiumax = xiumax
        print(f"垂直参数 K⊥ = {self.kperp}")
        print(f"垂直参数 热速度 = {self.vth[self.ispres]}")
        print(f"垂直参数 omega_ci = {self.omega_ci}")
        print(f"垂直参数 xi = {self.xiperp}")
        
        # 直接积分完整非负 Bessel 平方核，不再截断为 kamax 项。
        self.ksup = 0
        self.coelgd_with_weight_optimized(
            enable_weight=self.enable_trapping, uc=self.trapping_ucr, a=self.trapping_a,
            max_bessel_argument=abs(xiumax),
        )
        append_to_file(self.output_paths["read"],
                       f"QL kernel: full Bessel square; angular Gauss points = {len(self.ql_mu)}\n")
        if not self.icrh_enabled or self.pwdens == 0.0:
            self.dqlw = 0.0
            self.tbdqlk.fill(0.0)
            return
        

        # 初始化扩散系数
        if self.ibreak == 0:
            zdqlw = self.pwdens / self.zemxw
            self.dqlw = 1.0

            # 计算D0和系数
            legsup = 0
            qlnorm = self.qlrate(legsup)

            if qlnorm <= 0.0:
                raise ValueError("所选波数和谐波没有正的 RF 吸收，无法匹配正输入功率")

            # D0
            self.dqlw = zdqlw / qlnorm

            if self.icrh_enabled == 0:
                self.dqlw = 0.0

            # 【插入诊断】
            # print(f"【诊断 D_ql】zdqlw (目标功率密度参数): {zdqlw:.4e}")
            # print(f"【诊断 D_ql】qlnorm (归一化分母): {qlnorm:.4e}")
            # print(f"【诊断 D_ql】dqlw (最终扩散系数): {self.dqlw:.4e}")
            if abs(self.dqlw) < 1e-15 and self.icrh_enabled == 1:
                append_to_file(self.output_paths["read"], f"警告：D_ql 几乎为 0！ICRH 加热项将无效，分布必为麦氏分布。\n")
                print("❌ 警告：D_ql 几乎为 0！ICRH 加热项将无效，分布必为麦氏分布。")

            append_to_file(self.output_paths["read"], f"D0 xiperp = {self.xiperp}\n")
            append_to_file(self.output_paths["read"], f"D0 dqlw = {zdqlw} / {qlnorm} = {self.dqlw}\n")

            # append_to_file(self.output_paths["read"], f"qlnorm = {qlnorm}\n")
            
            # 仅用于旧版逐速度输出：mu=0 处的完整核，不参与功率或矩阵。
            self.tbdqlk.fill(0.0)
            self.tbdqlk[:, 0] = self.dqlw * special.jv(
                self.ipheat - 1, self.xiperp * self.tbu
            )**2
        else:
            for k in range(self.ksup + 1):
                for ip in range(self.nptu):
                    self.tbdqlk[ip, k] = 0.0

    def herm1(self, z):
        """一阶埃尔米特函数"""
        return (1.0 + 2.0 * abs(z)) * (1.0 - abs(z))**2

    def herm2(self, z):
        """二阶埃尔米特函数"""
        return z * (1.0 - abs(z))**2

    def dherm1(self, z):
        """一阶埃尔米特函数的导数"""
        return -6.0 * (1.0 - abs(z)) * z

    def dherm2(self, z):
        """二阶埃尔米特函数的导数"""
        return (1.0 - abs(z)) * (1.0 - 3.0 * abs(z))

    def inimat(self, ipsi):
        """初始化高斯积分点和相关参数
        
        Args:
            ipsi: 空间网格点索引
        """
        if ipsi == 0:
            # 初始化3点和5点高斯积分的节点和权重
            points, weights = special.roots_legendre(3)
            self.zg3 = (points + 1.0) / 2.0
            self.wg3 = weights / 2.0
            
            points, weights = special.roots_legendre(5)
            self.zg5 = (points + 1.0) / 2.0
            self.wg5 = weights / 2.0

            self.ngaus = 3
            self.lgvr = self.legmod + 1
            self.iwd = 4 * self.lgvr - 1
            self.iwdmtr = 3 * self.iwd + 1
            self.idimax = 2 * self.lgvr * self.nptu
            self.nfvrb = self.legmod + 1
            self.nfdim = 2 * self.nfvrb * self.nptu
            self.nfelem = self.nptu - 1
            self.nfshft = 2 * self.iwd + 1

            
            # 初始化Gamma矩阵为零
            self.fgamm = np.zeros((self.nfvrb, 2, self.nfvrb, 2))

            self.hel1 = np.zeros(self.ngaus)
            self.hel2 = np.zeros(self.ngaus)
            self.dhel1 = np.zeros(self.ngaus)
            self.dhel2 = np.zeros(self.ngaus)

            self.her1 = np.zeros(self.ngaus)
            self.her2 = np.zeros(self.ngaus)
            self.dher1 = np.zeros(self.ngaus)
            self.dher2 = np.zeros(self.ngaus)

            self.zgaus = np.zeros(self.ngaus)
            self.wgaus = np.zeros(self.ngaus)

            # 形函数维度含义：
            # - 第一个维度(2)：节点自由度 (0=函数值G, 1=导数dG)
            # - 第二个维度(2)：形函数类型 (0=函数值F, 1=导数dF)
            # - 第三个维度(ngaus)：高斯点索引
            self.hl = np.zeros((2, 2, self.ngaus))
            self.hr = np.zeros((2, 2, self.ngaus))


            self.x = np.zeros(self.idimax)
            self.dx = np.zeros(self.idimax)

            # 选择高斯积分点数
            if self.ngaus == 3:
                self.zgaus = self.zg3.copy()
                self.wgaus = self.wg3.copy()
            else:
                self.ngaus = 5
                self.zgaus = self.zg5.copy()
                self.wgaus = self.wg5.copy()

            # 计算埃尔米特函数值
            for i in range(self.ngaus):
                z = self.zgaus[i]
                self.hel1[i] = self.herm1(z)
                self.hel2[i] = self.herm2(z)
                self.dhel1[i] = self.dherm1(z)
                self.dhel2[i] = self.dherm2(z)

                z1 = -1.0 + z
                self.her1[i] = self.herm1(z1)
                self.her2[i] = self.herm2(z1)
                self.dher1[i] = self.dherm1(z1)
                self.dher2[i] = self.dherm2(z1)
            return

        # 将元素调整到ipsi点的速度网格
        xstep = self.du
        self.xstep = xstep
        for ip in range(self.nptu):
            self.x[ip] = self.tbu[ip]
            self.dx[ip] = self.du

        # 计算左右边界处的矩阵元素
        for i in range(self.ngaus):
            self.hl[0, 0, i] = self.hel1[i]
            self.hl[1, 0, i] = self.hel2[i] * xstep
            self.hl[0, 1, i] = self.dhel1[i] / xstep
            self.hl[1, 1, i] = self.dhel2[i]

            self.hr[0, 0, i] = self.her1[i]
            self.hr[1, 0, i] = self.her2[i] * xstep
            self.hr[0, 1, i] = self.dher1[i] / xstep
            self.hr[1, 1, i] = self.dher2[i]

        # append_to_file(self.output_paths["read"], f"高斯相关系数 hl = \n{self.hl}\n")
        # append_to_file(self.output_paths["read"], f"高斯相关系数 hr = \n{self.hr}\n")
    

    # ===================================================================
    # Matrix Construction & Solving
    # ===================================================================
    def matrix(self):
        """组装有限元矩阵"""
        # 初始化矩阵和向量
        self.fb = np.zeros(self.nfdim)
        self.fmatrx = np.zeros((self.iwdmtr, self.nfdim))
        self.psource = np.zeros(self.nfvrb) # 源项
        self._prepare_nbi_source()
        # self.pwnbi = 0.0
        # self.pwnbi_l = 0.0

        # self.sig_nbi()
        append_to_file(self.output_paths["read"], f"NBI功率沉积 S(k,u) S(0,0.4) = {self.S_nbi_ku[0,20]}\n")
        append_to_file(self.output_paths["read"], f"σ(k,n) sig(0,0) = {self.sig[0,0]}\n")
        

        # 循环遍历单元
        for ip in range(self.nfelem):
            # 初始化单元矩阵
            self.uvp = np.zeros((self.nfvrb, 2))
            self.uvm = np.zeros((self.nfvrb, 2))
            self.fdiap = np.zeros((self.nfvrb, 2, self.nfvrb, 2))
            self.fdiam = np.zeros((self.nfvrb, 2, self.nfvrb, 2))
            self.fmixl = np.zeros((self.nfvrb, 2, self.nfvrb, 2))
            self.fmixr = np.zeros((self.nfvrb, 2, self.nfvrb, 2))

            # 在单元IP内循环高斯积分点
            for i in range(self.ngaus):
                # 计算单元IP在Gauss点I的GAMMA值
                self.mastfp(ip, i)
                # if ip<1 and i<1:
                    # append_to_file(self.output_paths["read"], f"系数 fgamm = \n{self.fgamm}\n")

                w = self.xstep * self.wgaus[i]

                # Contract the derivative indices while retaining the nodal
                # value/derivative ordering used by the banded solver.
                left, right = self.hl[:, :, i], self.hr[:, :, i]
                self.fdiap += w * np.einsum('ldme,ad,be->lamb', self.fgamm, left, left)
                self.fdiam += w * np.einsum('ldme,ad,be->lamb', self.fgamm, right, right)
                self.fmixl += w * np.einsum('ldme,ad,be->lamb', self.fgamm, left, right)
                self.fmixr += w * np.einsum('ldme,ad,be->lamb', self.fgamm, right, left)
                                # if ip<2 and i<2 and ipla<2 and iplb<2:
                                #     append_to_file(self.output_paths["read"], f"源项高斯系数 fw = {fw}\n")

                # # 源项（这里只有ipla=0，即各向同性部分，ida=0）
                # # ∫ S(v) * φ(v) dv 
                # # 使用高斯积分近似：Σ w_i * S(v_i) * φ(v_i)
                # ipla = 0  # 勒让德阶数n=0
                # ida = 0   # 有G(v)的项
                # sw = w * self.psourc
                # for ia in range(2):
                #     self.uvp[ipla, ia] += sw * self.hl[ia, ida, i]
                #     self.uvm[ipla, ia] += sw * self.hr[ia, ida, i]

                # 源项 
                # ∫ S(v) * φ(v) dv 
                # 使用高斯积分近似：Σ w_i * S(v_i) * φ(v_i)
                for ipla in range(self.nfvrb):      # 勒让德阶数n
                    ida = 0                         # 有G(v)的项
                    # NBI has its own exact/adaptive Hermite projection below;
                    # collision quadrature only integrates the smooth background.
                    sw = w * self.psourc if ipla == 0 else 0.0
                    for ia in range(2):
                        self.uvp[ipla, ia] += sw * self.hl[ia, ida, i]
                        self.uvm[ipla, ia] += sw * self.hr[ia, ida, i]

                # append_to_file(self.output_paths["read"], f"源项高斯系数 w = {w}\n")


            # append_to_file(self.output_paths["read"], f"源项1  = \n{self.uvp:.15e}\n")
            # append_to_file(self.output_paths["read"], f"源项2  = \n{self.uvm}\n")

            
            self.uvp += self.nbi_element_source_left[ip]
            self.uvm += self.nbi_element_source_right[ip]

            # 边界条件（这里只在速度u=0处,即ip=0时）
            if ip == 0:
                self.boundf()

            # 将单元ip和ip-1的块矩阵属性添加到ip的刚度矩阵中
            # 点ip处的对角块，右部
            self.addblk(ip, ip, self.fdiap)
            # 点ip处的对角块，左部
            self.addblk(ip+1, ip+1, self.fdiam)
            # 非对角块(ip,ip+1)
            self.addblk(ip, ip+1, self.fmixl)
            # 非对角块(ip+1,ip)
            self.addblk(ip+1, ip, self.fmixr)

            # 点ip处的源项
            self.addrhs(ip, self.uvp)
            # 点ip+1处的源项
            self.addrhs(ip+1, self.uvm)

        append_to_file(
            self.output_paths["read"],
            f"NBI 弱形式注入粒子矩 = {self.nbi_particle_source_moment:.12e}\n"
            f"NBI 弱形式热粒子汇矩 = {self.nbi_particle_sink_moment:.12e}\n"
            f"NBI 净粒子源矩 = {self.nbi_net_particle_source_moment:.12e}\n"
            f"NBI 注入功率 = {self.nbi_injected_power:.12e} W/cm³\n"
            f"NBI 热粒子汇功率 = {-self.nbi_sink_power:.12e} W/cm³\n"
            f"NBI 方程净源能量矩 = {self.nbi_source_power:.12e} W/cm³\n"
        )
        
        # # 计算NBI源功率
        # if self.nbi_enabled:
        #     for k in range(self.Ns):
        #         # self.n_i_nbi += self.sig[k,0]
        #         self.pwnbi +=  self.E_nbi_k[k]*self.sig[k,0]
        #         self.pwnbi_l -= 1.5*self.sig[k,0]*self.ctx.tempi[self.ispres]

            # # 计算 NBI 功率归一化因子
            # if abs(self.pwnbi) > 1.0e-20:
            #     # 缩放因子 = 目标功率密度 / 当前计算出的原始功率密度
            #     pwdens_nbi = self.pwdens_nbi
            #     self.dnbi = pwdens_nbi / self.pwnbi
            #     # 打印诊断信息，观察归一化效果
            #     append_to_file(self.output_paths["read"], f"  [NBI 归一化] 原始功率: {self.pwnbi:.4e} W/cm^3, 目标功率: {self.pwdens_nbi:.4e} W/cm^3, 缩放因子 dnbi: {self.dnbi:.4e}")
            # else:
            #     self.dnbi = 0.0
            #     append_to_file(self.output_paths["read"], "  [NBI 归一化] 警告：原始 NBI 功率为 0，无法归一化！")
        

    def addblk(self, ip, jp, fblok):
        """添加块矩阵到全局矩阵
        
        Args:
            ip: 行索引
            jp: 列索引
            fblok: 块矩阵
            lgvr: 勒让德多项式阶数+1
        """
        size = 2 * self.nfvrb
        rows = size * ip + np.arange(size)[:, None]
        columns = size * jp + np.arange(size)[None, :]
        self.fmatrx[rows - columns + self.nfshft - 1, columns] += fblok.reshape(size, size)

    def addrhs(self, ip, uvblok):
        """Adds to RHS vector."""
        idx_start = 2 * self.nfvrb * ip
        for ipla in range(self.nfvrb):
            for ia in range(2):
                idx = idx_start + 2 * ipla + ia
                self.fb[idx] += uvblok[ipla, ia]

    def mastfp(self, ip, l):
        """Evaluates coefficients at Gauss point."""
        u = self.tbu[ip] + self.du * self.zgaus[l]
        uu = u * u
        
        # 计算准线性算子
        self.qlcoef(u)            
            
        # 计算碰撞算子和源项
        self.fokker(u)
        self.source(u)
        
        
        # append_to_file(self.output_paths["read"], f"θ(uβ) thefun({u}) = {self.thefun[1]}\n")
        
        for ln in range(self.nfvrb):
            n = ln # Leg order
            # ann = float(2*n)
            ann = float(n)
            annhf = ann + 0.5
            
            for lm in range(self.nfvrb):
                m = lm
                self.fgamm[ln, 0, lm, 0] = annhf * self.dql11[n, m]
                self.fgamm[ln, 1, lm, 0] = -annhf * u * self.dql01[n, m]
                self.fgamm[ln, 0, lm, 1] = -annhf * u * self.dql01[m, n]
                self.fgamm[ln, 1, lm, 1] = annhf * uu * self.dql00[n, m]
                
            self.fgamm[ln, 0, ln, 0] += ann * (ann + 1.0) * self.thecol / (2.0 * u if u > 0 else 1.0)
            self.fgamm[ln, 1, ln, 0] += uu * self.psitau
            self.fgamm[ln, 1, ln, 1] += u * self.psicol / 2.0

        # if ip<1 and l<1:
            # append_to_file(self.output_paths["read"], f"速度 v = {u}\n")

            # # append_to_file(self.output_paths["read"], f"θc(v) thecol = {self.thecol}\n")
            # append_to_file(self.output_paths["read"], f"(vi/β)/vi anuisp = {self.anuisp}\n")
            # append_to_file(self.output_paths["read"], f"θ(uβ) thefun = {self.thefun}\n")
            
            # append_to_file(self.output_paths["read"], f"ψc(v) psicol = {self.psicol}\n")
            # append_to_file(self.output_paths["read"], f"ψ(uβ) phifun = {self.phifun}\n")


            # append_to_file(self.output_paths["read"], f"ψτ(v) psitau = {self.psitau}\n")
            # append_to_file(self.output_paths["read"], f"Ti/Tβ tauisp = {self.tauisp}\n")

            # append_to_file(self.output_paths["read"], f"Q qleg00 = {self.qleg00}\n")

            # append_to_file(self.output_paths["read"], f"系数 dql00 = \n{self.dql00}\n")
            # append_to_file(self.output_paths["read"], f"系数 dql01 = \n{self.dql01}\n")
            # append_to_file(self.output_paths["read"], f"系数 dql11 = \n{self.dql11}\n")
            
            # append_to_file(self.output_paths["read"], f"源项 S = \n{self.psource[0]}\n")
            # append_to_file(self.output_paths["read"], f"NBI源项 Snbi(0) = \n{self.source_nbi[0]}\n")
            # append_to_file(self.output_paths["read"], f"NBI源项 Snbi(1) = \n{self.source_nbi[1]}\n")

    # def boundf(self):
    #     """
    #     u=0 处的边界条件处理 (严格自由度版)
    #     物理要求：
    #     1. f_n(0) = 0 (n>=1) -> 保证各向同性，消除 mu 倾斜
    #     2. f_n'(0) = 0 (所有n) -> 保证速度空间原点无梯度
    #     3. f_0(0) = val -> 保留为归一化迭代锚点，保证数值幅值正确
    #     """
    #     ia = 0  # ia=0 代表函数值 G(v)
    #     ib = 1  # ib=1 代表导数值 dG/dv
        
    #     for ipla in range(self.nfvrb):
    #         n = ipla  # 勒让德阶数
            
    #         # 1. 彻底清零矩阵块和源项（切断污染源，这是防止倾斜的关键！）
    #         for iplb in range(self.nfvrb):
    #             for idb in [0, 1]:
    #                 self.fdiap[ipla, ia, iplb, idb] = 0.0  # 清零函数值对应的矩阵行
    #                 self.fdiap[ipla, ib, iplb, idb] = 0.0  # 清零导数值对应的矩阵行
    #                 self.fmixl[ipla, ia, iplb, idb] = 0.0
    #                 self.fmixl[ipla, ib, iplb, idb] = 0.0
            
    #         # 矩阵对角设为 1 (施加 Dirichlet 条件)
    #         self.fdiap[ipla, ia, ipla, ia] = 1.0
    #         self.fdiap[ipla, ib, ipla, ib] = 1.0
            
    #         # 清零源项，防止边界条件被高斯积分的残差污染
    #         self.uvp[ipla, ia] = 0.0
    #         self.uvp[ipla, ib] = 0.0
    #         self.uvm[ipla, ia] = 0.0
    #         self.uvm[ipla, ib] = 0.0
            
    #         # 2. 严格按照自由度索引，分类处理 RHS (self.fb)
    #         # 在节点 ip=0 处，第 n 阶的函数值索引为 2*n，导数值索引为 2*n + 1
    #         if n == 0:
    #             # f_0(0) 保留为迭代锚点
    #             self.fb[2*n] = self.fb1_mxw if self.fb1_prv == 0.0 else self.fb1_prv
    #             # f_0'(0) 强制为 0
    #             self.fb[2*n + 1] = 0.0
    #         else:
    #             # f_n(0) 强制为 0 (消除不对称性的根本手段)
    #             self.fb[2*n] = 0.0
    #             # f_n'(0) 强制为 0
    #             self.fb[2*n + 1] = 0.0

    def boundf(self):
        """矩阵中 u=0 时的边界条件."""
        # Zero out the block corresponding to u=0 (node 0)
        # Then set diagonal to 1 for Dirichlet condition
        # In addblk logic, this modifies fdiap for ip=0
        # self.fdiap.fill(0.0)
        # self.fmixl.fill(0.0)
        
        # # Set F(0) = const condition
        # for ipla in range(self.nfvrb):
        #     self.fdiap[ipla, 0, ipla, 0] = 1.0

        # Keep the omitted weak row so particle balance can be checked after
        # imposing the independent density constraint.
        self.origin_weak_row = (
            self.fdiap[0, 0].ravel().copy(),
            self.fmixl[0, 0].ravel().copy(),
            float(self.uvp[0, 0]),
        )
        ia = 0
        for ipla in range(self.nfvrb):
            for iplb in range(self.nfvrb):
                for ib in [0, 1]:
                    self.fdiap[ipla, ia, iplb, ib] = 0.0
                    self.fmixl[ipla, ia, iplb, ib] = 0.0

            self.fdiap[ipla, 0, ipla, 0] = 1.0

        # for ipla in range(self.nfvrb):
        #     self.fdiap[ipla, 0, ipla, 0] = 1.0
        #     self.fdiap[ipla, 1, ipla, 1] = 1.0
        
        # RHS for boundary
        self.fb[0] = self.fb1_mxw
        self.uvp[:, 0] = 0.0

        # Zero out rest of node 0 RHS
        for iplb in range(1, self.nfvrb):
            self.fb[2 * iplb] = 0.0

    # def boundf(self):
    #     """矩阵中 u=0 时的边界条件."""
    #     # Zero out the block corresponding to u=0 (node 0)
    #     # Then set diagonal to 1 for Dirichlet condition
    #     # In addblk logic, this modifies fdiap for ip=0
    #     # self.fdiap.fill(0.0)
    #     # self.fmixl.fill(0.0)
        
    #     # # Set F(0) = const condition
    #     # for ipla in range(self.nfvrb):
    #     #     self.fdiap[ipla, 0, ipla, 0] = 1.0

    #     ia = 0
    #     for ipla in range(self.nfvrb):
    #         for iplb in range(self.nfvrb):
    #             for ib in [0, 1]:
    #                 self.fdiap[ipla, ia, iplb, ib] = 0.0
    #                 self.fmixl[ipla, ia, iplb, ib] = 0.0

    #         self.fdiap[ipla, 0, ipla, 0] = 1.0

    #         self.uvp[ipla, ia] = 0.0
    #         self.uvm[ipla, ia] = 0.0

    #     # for ipla in range(self.nfvrb):
    #     #     self.fdiap[ipla, 0, ipla, 0] = 1.0
    #     #     self.fdiap[ipla, 1, ipla, 1] = 1.0

        
    #     # RHS for boundary
    #     val = self.fb1_mxw if self.fb1_prv == 0.0 else self.fb1_prv
    #     self.fb[0] = val

    #     # Zero out rest of node 0 RHS
    #     for iplb in range(1, self.nfvrb):
    #         self.fb[iplb] = 0.0

    def boundv(self, itry):
        """处理边界条件和右端向量
        
        Args:
            itry: 尝试次数标记
                1: 存储右端向量
                2: 第一次更新边界条件
                >2: 后续更新边界条件
        """

        if itry == 1:
            # 存储求解前的右端向量
            self.fb_store = self.fb.copy()
        else:
            # 恢复右端向量
            self.fb = self.fb_store.copy()
            
            # 更新u=0处的边界条件
            if itry == 2:
                self.zfb1[1] = self.fb[0]  # Python索引从0开始
                self.ztpw[1] = self.zbalpw
                self.fb[0] = self.fb[0] / self.fnorm
            # 
            else:
                self.zfb1[0] = self.zfb1[1]
                self.ztpw[0] = self.ztpw[1]
                self.zfb1[1] = self.fb[0]
                self.ztpw[1] = self.zbalpw
                self.fb[0] = self.fb[0] - self.zbalpw * \
                            (self.zfb1[1] - self.zfb1[0]) / (self.ztpw[1] - self.ztpw[0])

            self.fb_store[0] = self.fb[0]

    # ===================================================================
    # Physics Coefficients
    # ===================================================================
    def fokker(self, u):
        """计算碰撞项和源项
        
        Args:
            u: 热速度参数 v/vth
        """
        self.psicol = 0.0       # Psi(g*u) 以1为权重
        self.psitau = 0.0       # Psi(g*u) 以温度为权重
        self.thecol = 0.0       # Theta(g*u) 以1为权重
        self.psiele = 0.0

        if u != 0.0:
            for isp in range(self.nspele):
                uab = u * self.vthisp[isp]
                uabsq = uab * uab
                self.expuab[isp] = 1.128379167 * np.exp(-uabsq)
        
                self.phifun[isp] = self.fperfu(uab) / uabsq
                self.thefun[isp] = uab * self.expuab[isp] + (uabsq - 0.5) * self.phifun[isp]

                self.psicol += self.anuisp[isp] * self.phifun[isp]
                self.psitau += self.anuisp[isp] * self.tauisp[isp] * self.phifun[isp]
                self.thecol += self.anuisp[isp] * self.thefun[isp]
            
            # 电子贡献
            self.psiele = self.anuisp[self.nspele-1] * self.phifun[self.nspele-1]
        else:
            self.expuab[:] = 1.128379167
            self.phifun[:] = 0.0
            self.thefun[:] = 0.0
    
    

    def _prepare_nbi_source(self):
        """Project physical injection and a number-balanced thermal sink onto Hermite elements.

        F integrates to one in d³u and time is normalized to anuref^-1.
        sig[k,0] = P_k/E_k; consequently T*sig/zemxw is the
        dimensionless particle-rate Legendre coefficient. Source integration is
        separate from collision quadrature so a narrow beam is never widened.
        """
        shape = (self.nfelem, self.nfvrb, 2)
        self.nbi_injection_left = np.zeros(shape)
        self.nbi_injection_right = np.zeros(shape)
        self.nbi_sink_left = np.zeros((self.nfelem, 2))
        self.nbi_sink_right = np.zeros((self.nfelem, 2))
        self.nbi_element_source_left = np.zeros(shape)
        self.nbi_element_source_right = np.zeros(shape)
        self.nbi_particle_source_moment = 0.0
        self.nbi_particle_sink_moment = 0.0
        self.nbi_net_particle_source_moment = 0.0
        self.nbi_injected_power = 0.0
        self.nbi_sink_power = 0.0
        self.nbi_source_power = 0.0
        self.pwnbi_l = 0.0
        self.nbi_particle_injection_rate = 0.0
        self.nbi_beam_particle_moments = np.zeros(self.Ns)
        self.nbi_beam_source_powers = np.zeros(self.Ns)
        self._nbi_sink_norm = 1.0
        self._nbi_energy_norm = float(erf(8.0 / np.sqrt(2.0)))
        if not self.nbi_enabled:
            return
        if not np.isfinite(self.nbi_energy_spread_ratio) or self.nbi_energy_spread_ratio < 0:
            raise ValueError("NBI energy_spread_ratio must be finite and nonnegative")
        if not np.isfinite(self.dnbi) or self.dnbi < 0:
            raise ValueError("NBI source multiplier must be finite and nonnegative")

        temperature = self.ctx.tempi[self.ispres]
        nodes, weights = np.polynomial.legendre.leggauss(12)
        left_energy = np.column_stack((self.tbu[:-1]**2, 2 * self.tbu[:-1]))
        right_energy = np.column_stack((self.tbu[1:]**2, 2 * self.tbu[1:]))
        for beam in range(self.Ns):
            coefficients = temperature * self.sig[beam] * self.dnbi / (4 * np.pi * self.zemxw)
            if coefficients[0] == 0:
                continue
            energy = self.E_nbi_k[beam]
            spread = self.nbi_energy_spread_ratio * energy
            if energy > self.tbkev[-1]:
                raise ValueError("NBI beam energy exceeds the velocity-domain energy; increase energy_range")
            beam_left = np.zeros((self.nfelem, 2))
            beam_right = np.zeros((self.nfelem, 2))
            if spread == 0:
                speed = self.v_nbi_k[beam]
                element = min(int(speed / self.du), self.nfelem - 1)
                z = (speed - self.tbu[element]) / self.du
                beam_left[element] = [self.herm1(z), self.du * self.herm2(z)]
                beam_right[element] = [self.herm1(z - 1), self.du * self.herm2(z - 1)]
            else:
                if energy - 8 * spread < 0 or energy + 8 * spread > self.tbkev[-1]:
                    raise ValueError(
                        "NBI energy Gaussian ±8 sigma must lie within [0, Emax]; "
                        "increase energy_range or reduce the physical energy spread"
                    )
                # Integrate in standardized energy; subdivision prevents a very
                # narrow Gaussian from being missed by velocity Gauss points.
                for element in range(self.nfelem):
                    lo = max(-8.0, (self.tbkev[element] - energy) / spread)
                    hi = min(8.0, (self.tbkev[element + 1] - energy) / spread)
                    if lo >= hi:
                        continue
                    edges = np.linspace(lo, hi, max(1, int(np.ceil(hi - lo))) + 1)
                    for lower, upper in zip(edges[:-1], edges[1:]):
                        t = (lower + upper) / 2 + (upper - lower) * nodes / 2
                        probability = ((upper - lower) * weights / 2
                                       * np.exp(-t*t / 2) / np.sqrt(2 * np.pi))
                        speeds = np.sqrt((energy + spread * t) / temperature)
                        z = (speeds - self.tbu[element]) / self.du
                        beam_left[element] += [
                            np.dot(probability, self.herm1(z)),
                            self.du * np.dot(probability, self.herm2(z)),
                        ]
                        beam_right[element] += [
                            np.dot(probability, self.herm1(z - 1)),
                            self.du * np.dot(probability, self.herm2(z - 1)),
                        ]
                mass = beam_left[:, 0].sum() + beam_right[:, 0].sum()
                beam_left /= mass
                beam_right /= mass

            projected_mass = beam_left[:, 0].sum() + beam_right[:, 0].sum()
            projected_energy = temperature * (
                np.sum(beam_left * left_energy) + np.sum(beam_right * right_energy)
            )
            if not np.isclose(projected_energy / projected_mass, energy, rtol=1e-11, atol=0):
                raise RuntimeError("NBI source projection did not preserve the specified mean beam energy")
            self.nbi_injection_left += beam_left[:, None, :] * coefficients[None, :, None]
            self.nbi_injection_right += beam_right[:, None, :] * coefficients[None, :, None]
            self.nbi_beam_particle_moments[beam] = 4 * np.pi * coefficients[0] * projected_mass
            self.nbi_beam_source_powers[beam] = (
                4 * np.pi * self.zemxw * coefficients[0] * projected_energy / temperature
            )

        # The sink is a Maxwellian particle population, so its weak source
        # contains u². Normalize over the actual finite velocity domain.
        for element in range(self.nfelem):
            edges = np.linspace(self.tbu[element], self.tbu[element + 1],
                                max(1, int(np.ceil(self.du))) + 1)
            for lower, upper in zip(edges[:-1], edges[1:]):
                speeds = (lower + upper) / 2 + (upper - lower) * nodes / 2
                probability = ((upper - lower) * weights / 2
                               * 4 / np.sqrt(np.pi) * speeds**2 * np.exp(-speeds**2))
                z = (speeds - self.tbu[element]) / self.du
                self.nbi_sink_left[element] += [
                    np.dot(probability, self.herm1(z)),
                    self.du * np.dot(probability, self.herm2(z)),
                ]
                self.nbi_sink_right[element] += [
                    np.dot(probability, self.herm1(z - 1)),
                    self.du * np.dot(probability, self.herm2(z - 1)),
                ]
        self._nbi_sink_norm = self.nbi_sink_left[:, 0].sum() + self.nbi_sink_right[:, 0].sum()
        injection_mass = (self.nbi_injection_left[:, 0, 0].sum()
                          + self.nbi_injection_right[:, 0, 0].sum())
        sink_scale = injection_mass / self._nbi_sink_norm
        self.nbi_sink_left *= sink_scale
        self.nbi_sink_right *= sink_scale
        self.nbi_element_source_left = self.nbi_injection_left.copy()
        self.nbi_element_source_right = self.nbi_injection_right.copy()
        self.nbi_element_source_left[:, 0, :] -= self.nbi_sink_left
        self.nbi_element_source_right[:, 0, :] -= self.nbi_sink_right
        self.nbi_particle_source_moment = 4 * np.pi * injection_mass
        self.nbi_particle_sink_moment = -4 * np.pi * (
            self.nbi_sink_left[:, 0].sum() + self.nbi_sink_right[:, 0].sum()
        )
        self.nbi_net_particle_source_moment = (
            self.nbi_particle_source_moment + self.nbi_particle_sink_moment
        )
        self.nbi_injected_power = 4 * np.pi * self.zemxw * (
            np.sum(self.nbi_injection_left[:, 0, :] * left_energy)
            + np.sum(self.nbi_injection_right[:, 0, :] * right_energy)
        )
        self.nbi_sink_power = 4 * np.pi * self.zemxw * (
            np.sum(self.nbi_sink_left * left_energy) + np.sum(self.nbi_sink_right * right_energy)
        )
        self.nbi_source_power = self.nbi_injected_power - self.nbi_sink_power
        self.nbi_particle_injection_rate = (
            self.nbi_particle_source_moment * self.anuref
            * self.ctx.dens * self.ctx.aconc[self.ispres] * 1e13
        )
        self.pwnbi_l = -self.nbi_sink_power

    def source(self, u):
        """Continuous weak source u² S/(n_i nu_ref), with v_th³ absorbed in S.

        Zero-width beam deltas are represented by their exact element
        projections in _prepare_nbi_source, not by pointwise samples here.
        """
        self.psourc = 0.0
        self.psource = np.zeros(self.nfvrb)
        self.source_nbi = np.zeros(self.nfvrb)
        uu = u * u
        for isp in range(self.nspele):
            if self.ctx.tempi[isp] != self.ctx.tempi[self.ispres]:
                self.psourc -= (
                    2 * self.anuisp[isp] * (self.tauisp[isp] - 1)
                    * (self.vthisp[isp] * self.expuab[isp] - u * self.phifun[isp])
                    * uu * np.exp(-uu) / 5.56832799683
                )
        self.psource[0] = self.psourc
        if not self.nbi_enabled:
            return

        temperature = self.ctx.tempi[self.ispres]
        for beam in range(self.Ns):
            spread = self.nbi_energy_spread_ratio * self.E_nbi_k[beam]
            if spread == 0:
                continue
            t = (temperature * uu - self.E_nbi_k[beam]) / spread
            if abs(t) <= 8:
                radial_probability = (
                    2 * temperature * u * np.exp(-t*t / 2)
                    / (spread * np.sqrt(2 * np.pi) * self._nbi_energy_norm)
                )
                self.source_nbi += (
                    temperature * self.sig[beam] / (4 * np.pi * self.zemxw) * radial_probability
                )
        rate = temperature * self.sig[:, 0].sum() / self.zemxw
        self.source_nbi[0] -= rate * uu * np.exp(-uu) / (np.pi**1.5 * self._nbi_sink_norm)
        self.psource += self.dnbi * self.source_nbi

            # # 诊断打印
            # if abs(u - self.v_nbi_k[0]) < 2.0 * self.du:
            #     print(f"【修复后 NBI 源】u={u:.2f}, 净源项={self.source_nbi[0]:.4e}, dnbi={self.dnbi:.4e}, 最终源项={self.psource[0]:.4e}")
                
            # self.psource[0] = self.psourc + self.source_nbi[0]
            # for n in range(1, self.legmod+1):
            #     self.psource[n] = self.source_nbi[n]
            # append_to_file(self.output_paths["read"], f"NBI源项 Snbi(0) = \n{self.source_nbi[0]}\n")
            # append_to_file(self.output_paths["read"], f"NBI源项 Snbi(1) = \n{self.source_nbi[1]}\n")
    
    def sig_nbi(self):
        """Project the physical angular source, independently of plotting nodes."""
        orders = np.arange(self.legmod + 1)
        self.sig = np.zeros((self.Ns, self.legmod + 1))
        nodes, weights = special.roots_legendre(max(16, (self.legmod + 3) // 2))
        if self.nbi_file is None:
            # Work in Gaussian standard deviations so even a source narrower
            # than the 101-point display grid is integrated without widening.
            center = self.nbi_mu_center
            width = self.nbi_sigma
            lo = max(-10.0, (-1.0 - center) / width)
            hi = min(10.0, (1.0 - center) / width)
            edges = np.linspace(lo, hi, max(1, int(np.ceil(hi - lo))) + 1)
            integrals = np.zeros(self.legmod + 1)
            for lower, upper in zip(edges[:-1], edges[1:]):
                t = (lower + upper) / 2 + (upper - lower) * nodes / 2
                mu = center + width * t
                quadrature = width * (upper - lower) * weights / 2 * np.exp(-t*t / 2)
                polynomials = special.eval_legendre(orders[:, None], mu[None, :])
                integrals += polynomials @ quadrature
            self.sig = self.nbi_gaussian_beam_weights[:, None] * integrals[None, :]
        else:
            # Linear interpolation is the external source's declared angular
            # representation. Degree L+1 is integrated exactly on each knot span.
            for interval in range(len(self.nbi_native_mu) - 1):
                lower = self.nbi_native_mu[interval]
                upper = self.nbi_native_mu[interval + 1]
                lo, hi = lower, upper
                if self.nbi_range_mu == 1:
                    lo = max(lo, 0.0)
                elif self.nbi_range_mu == -1:
                    hi = min(hi, 0.0)
                if lo >= hi:
                    continue
                mu = (lo + hi) / 2 + (hi - lo) * nodes / 2
                fraction = (mu - lower) / (upper - lower)
                source = (self.S_nbi_native[:, interval, None] * (1 - fraction)
                          + self.S_nbi_native[:, interval + 1, None] * fraction)
                polynomials = special.eval_legendre(orders[:, None], mu[None, :])
                self.sig += (source * ((hi - lo) * weights / 2)) @ polynomials.T
        self.sig *= (2 * orders + 1)[None, :] / 2
        if self.nbi_enabled:
            for beam in range(self.Ns):
                print(f"NBI beam {beam}: angular Legendre coefficients {self.sig[beam, :3]}")
        # ============ 诊断代码结束 ============

    def fperfu(self, u):
        """计算误差函数
        
        Args:
            u: 输入参数
        Returns:
            fperfu: 函数值
        """
        # 定义系数A
        self.A = np.array([0.254829592, -0.284496736, 
                        1.421413741, -1.453152027, 
                        1.061405429])
        
        p = 0.3275911
        q = 1.128379167
        
        fperfu = 0.0
        if u == 0.0:
            return fperfu

        usq = u * u
        if u >= 0.5:
            # 误差函数的多项式插值
            t = 1.0 / (1.0 + p * u)
            sum_val = self.A[4] * t
            for i in range(1, 5):
                k = 4 - i
                sum_val = (self.A[k] + sum_val) * t
            fperfu = 1.0 - np.exp(-usq) * (sum_val + q * u)
        else:
            # 幂级数展开
            k = 0
            tf = 1.0
            sum_val = 1.0 / 3.0
            
            while True:
                k += 1
                tf = -tf * usq / k
                term = tf / (2.0 * k + 3.0)
                sum_val += term
                
                if abs(term / sum_val) <= 1.0E-9:
                    break
                    
            fperfu = 2.0 * q * usq * u * sum_val
        
        # part1 = -2/np.sqrt(np.pi) * u * np.exp(-u**2)
        # part2 = erf(u)
        # fperfu = part1+part2
        
        return fperfu


    def qlcoef(self, u):
        """Galerkin projection of the complete, nonnegative perpendicular RF kernel."""
        if self.dqlw == 0.0:
            self.dql00.fill(0.0)
            self.dql01.fill(0.0)
            self.dql11.fill(0.0)
            return
        argument = abs(float(self.xiperp * u))
        if argument > self._ql_max_argument:
            self.coelgd_with_weight_optimized(
                enable_weight=self.use_weighted_coelgd, uc=self._ql_uc, a=self._ql_a,
                max_bessel_argument=max(1.0, 2.0 * argument),
            )
        if self._ql_cache_xiperp != self.xiperp:
            self._ql_coefficient_cache.clear()
            self._ql_cache_xiperp = self.xiperp
        key = float(u)
        coefficients = self._ql_coefficient_cache.get(key)
        if coefficients is None:
            weights = self.ql_weights * (1.0 - self.ql_mu**2) * special.jv(
                self.ipheat - 1, argument * np.sqrt(1.0 - self.ql_mu**2)
            )**2
            polynomials = self.ql_polynomials
            derivatives = self.ql_mu_derivatives
            coefficients = (
                (polynomials * weights) @ polynomials.T,
                (polynomials * weights) @ derivatives.T,
                (derivatives * weights) @ derivatives.T,
            )
            if self._ql_symmetric:
                for coefficient in coefficients:
                    coefficient[self._ql_odd_pairs] = 0.0
            self._ql_coefficient_cache[key] = coefficients
        self.dql00[:] = self.dqlw * coefficients[0]
        self.dql01[:] = self.dqlw * coefficients[1]
        self.dql11[:] = self.dqlw * coefficients[2]

    def qldcoe(self, u):
        """计算准线性扩散系数中的贝塞尔函数
            仅保留旧版展开诊断；求解矩阵与功率使用 qlcoef 的完整核。
            eqn(20)
            计算贝塞尔函数
            输入： u //速度 ipheat //谐波次数p xiperp //垂直 ksup //阶数 legmod //
            返回贝塞尔函数： dqlu 
        Args:
            u: 速度参数
        """
        ib = self.ipheat - 1
        xiu = self.xiperp * u
        self.fjbess[ib] = 0.0
        self.njbess = self.ksup + 1
        
        # 计算贝塞尔函数
        self.rjbess(xiu)
        self.dqlu[0] = self.fjbess[ib] * self.fjbess[ib]
        
        txk = 1.0
        for k in range(1, self.ksup + 1):
            zjbs = 0.0
            for kp in range(k + 1):
                kd = k - kp
                zjbs = (zjbs + 
                        self.fctinv[kp] * self.fctinv[kd] * 
                        self.fjbess[ib + kp] * self.fjbess[ib + kd])
            
            txk = txk * xiu / 2.0
            self.dqlu[k] = txk * zjbs
        
        return

    def qlrate(self, legsup):
        """RF energy moment using the same Hermite elements and Gauss rule as matrix()."""
        if self.dqlw == 0.0:
            return 0.0
        rate = 0.0
        for z, weight in zip(self.zgaus, self.wgaus):
            # Compute the shapes directly: iniqln also calls this before the
            # velocity-dependent tables in inimat(1) have been initialized.
            values = (
                self.herm1(z) * self.tbfu[:-1, :legsup + 1]
                + self.du * self.herm2(z) * self.tbdfu[:-1, :legsup + 1]
                + self.herm1(z - 1.0) * self.tbfu[1:, :legsup + 1]
                + self.du * self.herm2(z - 1.0) * self.tbdfu[1:, :legsup + 1]
            )
            derivatives = (
                self.dherm1(z) / self.du * self.tbfu[:-1, :legsup + 1]
                + self.dherm2(z) * self.tbdfu[:-1, :legsup + 1]
                + self.dherm1(z - 1.0) / self.du * self.tbfu[1:, :legsup + 1]
                + self.dherm2(z - 1.0) * self.tbdfu[1:, :legsup + 1]
            )
            for ip, u in enumerate(self.tbu[:-1] + self.du * z):
                self.qlcoef(u)
                rate += self.du * weight * u**2 * (
                    self.dql01[0, :legsup + 1] @ values[ip]
                    - u * self.dql00[0, :legsup + 1] @ derivatives[ip]
                )
        return 4.0 * np.pi * rate
    
    def diagnose_q01_asymmetry(self):
        """
        诊断函数：对比三种模式下 Q01 的计算结果
        模式1：原解析递推 coelgd
        模式2：纯数值积分 (W=1)
        模式3：解析基准 + 扰动 (W=1)  <-- 这是我们最终采用的方案
        """
        print("\n" + "="*80)
        print(" 诊断测试：Q01 对称性分析 (纯 ICRH, W=1)")
        print("="*80)
        
        # ---------------------------------------------------------
        # 模式1：原解析递推
        # ---------------------------------------------------------
        self.coelgd()
        ana_qleg01 = self.qleg01.copy()
        
        # ---------------------------------------------------------
        # 模式2：纯数值积分 (W=1，直接设 weight=1)
        # ---------------------------------------------------------
        num_qleg01 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        
        num_points = 500
        u_vals = np.linspace(-1.0, 1.0, num_points)
        du = u_vals[1] - u_vals[0]
        
        weight_1 = np.ones(num_points) # 强制 W=1
        p_power = 0 if self.ipheat == 1 else 1
        geo_base = (1.0 - u_vals**2)**(1 + p_power)
        k_indices = np.arange(self.kamax + 1).reshape(-1, 1)
        u_powers = u_vals.reshape(1, -1) ** (2 * k_indices)
        base_weight = weight_1 * geo_base * u_powers
        
        P_vals = np.zeros((self.legmod + 1, num_points))
        dP_vals = np.zeros((self.legmod + 1, num_points))
        for n in range(self.legmod + 1):
            Pn = legendre(n)
            P_vals[n, :] = Pn(u_vals)
            dP_vals[n, :] = Pn.deriv()(u_vals)
            
        u_dP_vals = np.zeros_like(P_vals)
        for n in range(self.legmod + 1):
            u_dP_vals[n, :] = n * P_vals[n, :]
            k_rec = n - 1
            while k_rec >= 0:
                u_dP_vals[n, :] += (2*k_rec + 1) * P_vals[k_rec, :]
                k_rec -= 1
                
        simpson_w = np.ones(num_points)
        simpson_w[1:-1:2] = 4
        simpson_w[2:-1:2] = 2
        simpson_w = simpson_w * (du / 3.0)
        base_weight_s = base_weight * simpson_w
        
        num_qleg01 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, u_dP_vals, optimize=True)
        
        # ---------------------------------------------------------
        # 模式3：解析基准 + 扰动 (W-1=0，即最终方案在 W=1 时的行为)
        # ---------------------------------------------------------
        # 调用我们设计的函数，传入 enable_weight=False
        self.coelgd_with_weight_optimized(enable_weight=False)
        final_qleg01 = self.qleg01.copy()
        
        # ---------------------------------------------------------
        # 打印对比结果
        # ---------------------------------------------------------
        print(f"\n{'Q01(n,m,k=0) 对比':<25} | {'解析递推':<15} | {'纯数值(W=1)':<15} | {'最终方案(W=1)':<15}")
        print("-" * 75)
        
        for n in range(min(4, self.legmod+1)):
            for m in range(min(4, self.legmod+1)):
                ana_val = ana_qleg01[n, m, 0]
                num_val = num_qleg01[n, m, 0]
                fin_val = final_qleg01[n, m, 0]
                
                # 只打印非零或显著的项
                if abs(ana_val) > 1e-20 or abs(num_val) > 1e-20:
                    print(f"n={n}, m={m:<15} | {ana_val:<15.4e} | {num_val:<15.4e} | {fin_val:<15.4e}")
                    
        print("\n【核心结论】")
        print("1. 解析递推的 Q01 不为 0 (约 1e-15 量级)，这是导致你看到倾斜的元凶。")
        print("2. 纯数值积分 (W=1) 的 Q01 也不为 0 (约 1e-10 量级)，仍有网格离散误差。")
        print("3. 最终方案 (W=1) 的 Q01 严格为 0，因为底层调用了解析递推，并在 qlcoef 中被强制清零！")
        print("="*80 + "\n")

    def coelgd_with_weight_optimized(self, enable_weight=False, uc=0.0, a=0.0,
                                     force_pure_numerical=False, max_bessel_argument=0.0):
        """Build stable angular quadrature and the legacy polynomial diagnostics.

        The empirical trapping weight is symmetric by default. It is not a
        bounce-averaged orbit model. ``directional`` preserves the old signed
        pitch-angle experiment. The production operator integrates J² directly.
        """
        self.use_weighted_coelgd = enable_weight
        self._ql_uc = uc
        self._ql_a = a
        self._ql_max_argument = abs(float(max_bessel_argument))
        if self.trapping_model not in ("symmetric", "directional"):
            raise ValueError(f"Unknown trapping_model: {self.trapping_model}")
        self._ql_symmetric = not enable_weight or self.trapping_model == "symmetric"
        orders = np.arange(self.legmod + 1)
        self._ql_odd_pairs = (orders[:, None] + orders[None, :]) % 2 != 0

        def integrate(order):
            nodes, quadrature_weights = special.roots_legendre(order)
            # Split at zero to resolve the absolute-value corner of the
            # symmetric weight, retaining positive integration weights.
            mu = np.concatenate(((nodes - 1.0) / 2.0, (nodes + 1.0) / 2.0))
            weights = np.tile(quadrature_weights / 2.0, 2)
            if enable_weight:
                pitch = np.abs(mu) if self.trapping_model == "symmetric" else mu
                weights *= special.expit(a * (pitch - uc))
            polynomials = np.array([special.eval_legendre(n, mu) for n in orders])
            derivatives = np.zeros_like(polynomials)
            # Gauss nodes exclude the endpoints; this identity avoids the
            # ill-conditioned monomial derivative used by poly1d.deriv().
            for n in range(1, self.legmod + 1):
                derivatives[n] = n * (mu * polynomials[n] - polynomials[n - 1]) / (mu**2 - 1.0)
            mu_derivatives = mu * derivatives
            basis_weights = weights * (1.0 - mu**2)**self.ipheat
            powers = mu[None, :]**(2 * np.arange(self.kamax + 1)[:, None])
            weighted_powers = basis_weights * powers
            q00 = np.einsum('ki,li,mi->lmk', weighted_powers, polynomials, polynomials, optimize=True)
            q01 = np.einsum('ki,li,mi->lmk', weighted_powers, polynomials, mu_derivatives, optimize=True)
            q11 = np.einsum('ki,li,mi->lmk', weighted_powers, mu_derivatives, mu_derivatives, optimize=True)
            full_weight = weights * (1.0 - mu**2) * special.jv(
                self.ipheat - 1, self._ql_max_argument * np.sqrt(1.0 - mu**2)
            )**2
            full = (
                (polynomials * full_weight) @ polynomials.T,
                (polynomials * full_weight) @ mu_derivatives.T,
                (mu_derivatives * full_weight) @ mu_derivatives.T,
            )
            return mu, weights, polynomials, mu_derivatives, (q00, q01, q11), full

        order = max(32, self.legmod + 8, int(np.ceil(2 * self._ql_max_argument)) + 16,
                    int(np.ceil(abs(a) / 2)) if enable_weight else 0)
        previous = integrate(order)
        for _ in range(5):
            order *= 2
            current = integrate(order)
            if all(np.allclose(left, right, rtol=1e-10, atol=1e-11)
                   for left, right in zip(previous[4] + previous[5], current[4] + current[5])):
                break
            previous = current
        else:
            raise RuntimeError("RF angular quadrature did not converge; check legmod and trapping parameters")
        self.ql_mu, self.ql_weights, self.ql_polynomials, self.ql_mu_derivatives = current[:4]
        self.qleg00, self.qleg01, self.qleg11 = current[4]
        if self._ql_symmetric:
            for coefficient in (self.qleg00, self.qleg01, self.qleg11):
                coefficient[self._ql_odd_pairs, :] = 0.0
        self._ql_cache_xiperp = self.xiperp
        self._ql_coefficient_cache = {}

    # def coelgd_with_weight_optimized(self, enable_weight=False, uc=0.0, a=0.0):
    #     """
    #     优化版：使用向量化运算和矩阵乘法加速数值积分
    #     增加控制开关：enable_weight 控制是否使用捕获效应权重
    #     采用 "解析基准 + 数值扰动" 策略，确保 W=1 时绝对对称
    #     """
    #     # 【关键】：根据开关设置标记，供 qlcoef 函数判断
    #     self.use_weighted_coelgd = enable_weight
        
    #     # 1. 初始化输出数组
    #     self.qleg00 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg01 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg11 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        
    #     # 2. 始终先计算 W=1 的解析基准 (调用原 coelgd)
    #     # 这保证了纯 ICRH 模式下的绝对对称性
    #     self.coelgd()
        
    #     # 如果不启用权重，直接返回解析结果
    #     if not enable_weight:
    #         return
            
    #     # --- 以下是启用权重 (捕获效应) 的数值扰动逻辑 ---
        
    #     # 3. 设置数值积分网格
    #     num_points = 500
    #     u_vals = np.linspace(-1.0, 1.0, num_points)
    #     du = u_vals[1] - u_vals[0]
        
    #     # 4. 预计算所有不依赖 n, m, k 的项 (向量化)
        
    #     # 【关键修改】：计算权重扰动项 Delta_W = W - 1
    #     # 这样数值积分只计算由捕获效应引起的偏离对称的部分
    #     weight = 1.0 / (1.0 + np.exp(-a * (u_vals - uc)))
    #     delta_weight = weight - 1.0
        
    #     # 几何因子 (1-u^2)^{1+p}: Shape (N,)
    #     p_power = 0 if self.ipheat == 1 else 1
    #     geo_base = (1.0 - u_vals**2)**(1 + p_power)
        
    #     # u^{2k} 算子: Shape (Kmax+1, N)
    #     k_indices = np.arange(self.kamax + 1).reshape(-1, 1)
    #     u_powers = u_vals.reshape(1, -1) ** (2 * k_indices)
        
    #     # 组合基础权重因子: Shape (Kmax+1, N)
    #     base_weight = delta_weight * geo_base * u_powers
        
    #     # 5. 预计算勒让德多项式矩阵 (向量化)
    #     P_vals = np.zeros((self.legmod + 1, num_points))
    #     dP_vals = np.zeros((self.legmod + 1, num_points))
        
    #     for n in range(self.legmod + 1):
    #         Pn = legendre(n)
    #         P_vals[n, :] = Pn(u_vals)
    #         dP_vals[n, :] = Pn.deriv()(u_vals)
            
    #     # 【关键修复】：使用递推关系计算 u * P_n'(u)
    #     # 数学公式: u * P_n'(u) = n * P_n(u) + (2n-1) * P_{n-1}(u) + (2n-5) * P_{n-2}(u) + ...
    #     u_dP_vals = np.zeros_like(P_vals)
    #     for n in range(self.legmod + 1):
    #         u_dP_vals[n, :] = n * P_vals[n, :]
    #         k_rec = n - 1
    #         while k_rec >= 0:
    #             u_dP_vals[n, :] += (2*k_rec + 1) * P_vals[k_rec, :]
    #             k_rec -= 1
        
    #     # 6. 计算辛普森权重系数
    #     simpson_w = np.ones(num_points)
    #     simpson_w[1:-1:2] = 4
    #     simpson_w[2:-1:2] = 2
    #     simpson_w = simpson_w * (du / 3.0)
        
    #     base_weight_s = base_weight * simpson_w
        
    #     # 7. 计算扰动项的积分
    #     delta_qleg00 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, P_vals, optimize=True)
    #     delta_qleg01 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, u_dP_vals, optimize=True)
    #     delta_qleg11 = np.einsum('ki,li,mi->lmk', base_weight_s, u_dP_vals, u_dP_vals, optimize=True)
        
    #     # 8. 叠加：最终结果 = 解析基准 + 数值扰动
    #     self.qleg00 += delta_qleg00
    #     self.qleg01 += delta_qleg01
    #     self.qleg11 += delta_qleg11

    # def coelgd_with_weight_optimized(self, enable_weight=False, uc=0.0, a=0.0):
    #     """
    #     优化版：使用向量化运算和矩阵乘法加速数值积分
    #     增加控制开关：enable_weight 控制是否使用捕获效应权重
    #     """
    #     # 【关键】：根据开关设置标记，供 qlcoef 函数判断
    #     self.use_weighted_coelgd = enable_weight
        
    #     # 1. 初始化输出数组
    #     self.qleg00 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg01 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg11 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        
    #     # 如果不启用权重，直接调用原始的解析递推算法 (保证纯 ICRH 绝对对称)
    #     if not enable_weight:
    #         self.coelgd()
    #         return
            
    #     # --- 以下是启用权重 (捕获效应) 的数值积分逻辑 ---
        
    #     # 2. 设置数值积分网格
    #     num_points = 500
    #     u_vals = np.linspace(-1.0, 1.0, num_points)
    #     du = u_vals[1] - u_vals[0]
        
    #     # 3. 预计算所有不依赖 n, m, k 的项 (向量化)
        
    #     # 权重因子 W: Shape (N,)
    #     weight = 1.0 / (1.0 + np.exp(-a * (u_vals - uc)))
        
    #     # 几何因子 (1-u^2)^{1+p}: Shape (N,)
    #     p_power = 0 if self.ipheat == 1 else 1
    #     geo_base = (1.0 - u_vals**2)**(1 + p_power)
        
    #     # u^{2k} 矩阵: Shape (Kmax+1, N)
    #     k_indices = np.arange(self.kamax + 1).reshape(-1, 1)
    #     u_powers = u_vals.reshape(1, -1) ** (2 * k_indices)
        
    #     # 组合基础权重因子: Shape (Kmax+1, N)
    #     base_weight = weight * geo_base * u_powers
        
    #     # 4. 预计算勒让德多项式矩阵 (向量化)
    #     P_vals = np.zeros((self.legmod + 1, num_points))
    #     dP_vals = np.zeros((self.legmod + 1, num_points))
        
    #     for n in range(self.legmod + 1):
    #         Pn = legendre(n)
    #         P_vals[n, :] = Pn(u_vals)
    #         dP_vals[n, :] = Pn.deriv()(u_vals)
            
    #     # 【关键修复】：使用递推关系计算 u * P_n'(u)
    #     # 数学公式: u * P_n'(u) = n * P_n(u) + (2n-1) * P_{n-1}(u) + (2n-5) * P_{n-2}(u) + ...
    #     u_dP_vals = np.zeros_like(P_vals)
    #     for n in range(self.legmod + 1):
    #         u_dP_vals[n, :] = n * P_vals[n, :]
    #         k_rec = n - 1
    #         while k_rec >= 0:
    #             u_dP_vals[n, :] += (2*k_rec + 1) * P_vals[k_rec, :]
    #             k_rec -= 1
        
    #     # 5. 计算辛普森权重系数
    #     simpson_w = np.ones(num_points)
    #     simpson_w[1:-1:2] = 4
    #     simpson_w[2:-1:2] = 2
    #     simpson_w = simpson_w * (du / 3.0)
        
    #     base_weight_s = base_weight * simpson_w
        
    #     # --- 计算 Qleg00 ---
    #     self.qleg00 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, P_vals, optimize=True)
        
    #     # --- 计算 Qleg01 ---
    #     self.qleg01 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, u_dP_vals, optimize=True)
        
    #     # --- 计算 Qleg11 ---
    #     self.qleg11 = np.einsum('ki,li,mi->lmk', base_weight_s, u_dP_vals, u_dP_vals, optimize=True)


    # def coelgd_with_weight_optimized(self, uc, a):
    #     """
    #     优化版：使用向量化运算和矩阵乘法加速数值积分 (最终修正版)
    #     修复了 u * dP 的计算逻辑，确保不对称性物理自洽
    #     """
    #     # 标记当前使用的是带权重的计算，不要在 qlcoef 中强制置零
    #     self.use_weighted_coelgd = True
        
    #     # 1. 初始化输出数组
    #     self.qleg00 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg01 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg11 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        
    #     # 2. 设置数值积分网格
    #     num_points = 500
    #     u_vals = np.linspace(-1.0, 1.0, num_points)
    #     du = u_vals[1] - u_vals[0]
        
    #     # 3. 预计算所有不依赖 n, m, k 的项 (向量化)
    #     weight = 1.0 / (1.0 + np.exp(-a * (u_vals - uc)))
    #     p_power = 0 if self.ipheat == 1 else 1
    #     geo_base = (1.0 - u_vals**2)**(1 + p_power)
        
    #     k_indices = np.arange(self.kamax + 1).reshape(-1, 1)
    #     u_powers = u_vals.reshape(1, -1) ** (2 * k_indices)
    #     base_weight = weight * geo_base * u_powers
        
    #     # 4. 预计算勒让德多项式矩阵 (向量化)
    #     P_vals = np.zeros((self.legmod + 1, num_points))
    #     dP_vals = np.zeros((self.legmod + 1, num_points))
        
    #     for n in range(self.legmod + 1):
    #         Pn = legendre(n)
    #         P_vals[n, :] = Pn(u_vals)
    #         dP_vals[n, :] = Pn.deriv()(u_vals)
            
    #     # 【关键修复】：使用递推关系计算 u * P_n'(u)
    #     # 数学公式: u * P_n'(u) = n * P_n(u) + (2n-1) * P_{n-1}(u) + (2n-5) * P_{n-2}(u) + ...
    #     u_dP_vals = np.zeros_like(P_vals)
    #     for n in range(self.legmod + 1):
    #         u_dP_vals[n, :] = n * P_vals[n, :]
    #         k_rec = n - 1
    #         while k_rec >= 0:
    #             u_dP_vals[n, :] += (2*k_rec + 1) * P_vals[k_rec, :]
    #             k_rec -= 1
        
    #     # 5. 预计算辛普森权重系数
    #     simpson_w = np.ones(num_points)
    #     simpson_w[1:-1:2] = 4
    #     simpson_w[2:-1:2] = 2
    #     simpson_w = simpson_w * (du / 3.0)
        
    #     base_weight_s = base_weight * simpson_w
        
    #     # --- 计算 Qleg00 ---
    #     self.qleg00 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, P_vals, optimize=True)
        
    #     # --- 计算 Qleg01 ---
    #     # 使用修复后的 u_dP_vals
    #     self.qleg01 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, u_dP_vals, optimize=True)
        
    #     # --- 计算 Qleg11 ---
    #     self.qleg11 = np.einsum('ki,li,mi->lmk', base_weight_s, u_dP_vals, u_dP_vals, optimize=True)


    
    # def coelgd_with_weight_optimized(self, uc, a):
    #     """
    #     优化版：使用向量化运算和矩阵乘法加速数值积分
    #     """
    #     # 1. 初始化输出数组
    #     self.qleg00 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg01 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
    #     self.qleg11 = np.zeros((self.legmod+1, self.legmod+1, self.kamax+1))
        
    #     # 2. 设置数值积分网格
    #     num_points = 500
    #     u_vals = np.linspace(-1.0, 1.0, num_points)
    #     du = u_vals[1] - u_vals[0]
        
    #     # 3. 预计算所有不依赖 n, m, k 的项 (向量化)
        
    #     # 权重因子 W: Shape (N,)
    #     weight = 1.0 / (1.0 + np.exp(-a * (u_vals - uc)))
        
    #     # 几何因子 (1-u^2)^{1+p}: Shape (N,)
    #     p_power = 0 if self.ipheat == 1 else 1
    #     geo_base = (1.0 - u_vals**2)**(1 + p_power)
        
    #     # u^{2k} 矩阵: Shape (Kmax+1, N)
    #     # 使用广播机制: (Kmax+1, 1) * (1, N) -> (Kmax+1, N)
    #     k_indices = np.arange(self.kamax + 1).reshape(-1, 1)
    #     u_powers = u_vals.reshape(1, -1) ** (2 * k_indices)
        
    #     # 组合基础权重因子: Shape (Kmax+1, N)
    #     # base_weight[k, i] = W[i] * Geo[i] * u[i]^(2k)
    #     base_weight = weight * geo_base * u_powers
        
    #     # 4. 预计算勒让德多项式矩阵 (向量化)
    #     # P_vals[n, i] = P_n(u_i), Shape (Lmax+1, N)
    #     P_vals = np.zeros((self.legmod + 1, num_points))
    #     dP_vals = np.zeros((self.legmod + 1, num_points))
        
    #     for n in range(self.legmod + 1):
    #         Pn = legendre(n)
    #         P_vals[n, :] = Pn(u_vals)
    #         dP_vals[n, :] = Pn.deriv()(u_vals)
            
    #     # 5. 利用 einsum 进行张量积分
    #     # 积分本质上是: result = sum(integrand * weights * du)
    #     # simpson 只是改变了 weights (1, 4, 2, 4, ..., 1)，这里我们简化处理，
    #     # 假设步长 du 足够小，或者直接使用 simpson 函数对最后一维积分。
    #     # 为了演示优化，我们使用矩阵乘法思路。
        
    #     # 预计算辛普森权重系数 (Simpson's Rule Weights)
    #     simpson_w = np.ones(num_points)
    #     simpson_w[1:-1:2] = 4
    #     simpson_w[2:-1:2] = 2
    #     simpson_w = simpson_w * (du / 3.0) # Shape (N,)
        
    #     # 将辛普森权重合并到 base_weight 中
    #     # base_weight_s[k, i] = base_weight[k, i] * simpson_w[i]
    #     base_weight_s = base_weight * simpson_w # Shape (Kmax+1, N)
        
    #     # --- 计算 Qleg00 ---
    #     # 积分式: Sum_k ( base_weight * P_n * P_m )
    #     # 张量形状: (K, N) * (L, N) * (L, N) -> 我们需要通过 einsum 处理维度对齐
    #     # 目标: qleg00[n, m, k] = Sum_i ( base_weight[k, i] * P_vals[n, i] * P_vals[m, i] )
        
    #     # 优化：将 N 维放在最后进行求和
    #     # base_weight_s: (K, N) -> (K, N)
    #     # P_vals: (L, N) -> (L, N)
    #     # 结果: (L, L, K)
        
    #     # 这里的关键是利用 np.einsum 自动处理广播和求和
    #     # 'kn,ln,lm->lmk' 表示:
    #     # k: kmax维度, n: legmod维度, l: legmod维度, m: legmod维度
    #     # 输入1 (base_weight_s): 索引 k, n (这里n其实是grid点，用i表示更好)
    #     # 输入2 (P_vals): 索引 l, i
    #     # 输入3 (P_vals): 索引 m, i
    #     # 输出: l, m, k (对应 n, m, k)
        
    #     # 修正维度命名以便理解:
    #     # base_weight_s: (K, I)  I代表积分点
    #     # P_vals: (L, I)
        
    #     # Q00 = Sum_I ( base[K, I] * P[L, I] * P[M, I] )
    #     self.qleg00 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, P_vals, optimize=True)
        
    #     # --- 计算 Qleg01 ---
    #     # 积分式: Sum_k ( base_weight * P_n * u * dP_m )
    #     # u_vals: (I,)
    #     # dP_vals: (L, I)
    #     # u_dP: (L, I) element-wise multiply
        
    #     u_dP_vals = u_vals * dP_vals # Shape (L, I)
        
    #     # Q01 = Sum_I ( base[K, I] * P[L, I] * u_dP[M, I] )
    #     self.qleg01 = np.einsum('ki,li,mi->lmk', base_weight_s, P_vals, u_dP_vals, optimize=True)
        
    #     # --- 计算 Qleg11 ---
    #     # 积分式: Sum_k ( base_weight * u * dP_n * u * dP_m )
    #     # u_dP_vals 已经计算好了: u * dP
        
    #     # Q11 = Sum_I ( base[K, I] * u_dP[L, I] * u_dP[M, I] )
    #     self.qleg11 = np.einsum('ki,li,mi->lmk', base_weight_s, u_dP_vals, u_dP_vals, optimize=True)

    # ===================================================================
    # Legendre Polynomial Integrals (Recursive generation)
    # ===================================================================

    def coelgd(self):
        """"
        初始化勒让德多项式积分
        利用勒让德函数的递推关系进行积分计算
        Q00=∫_{-1}^{1} ((1-μ^2)^(1+p)*(μ^2k))P_n(μ) *P_m(μ)dμ
        Q01=∫_{-1}^{1} ((1-μ^2)^(1+p)*(μ^2k))P_n(μ) *μ (P_m(μ))'dμ
        Q11=∫_{-1}^{1} ((1-μ^2)^(1+p)*(μ^2k))μ (P_n(μ))' *μ P_m(μ))'dμ
        """

        self.kadim = self.kamax + 4 * self.legmod
        self.qauxil = np.zeros((self.legmod+1, self.legmod+1, self.kadim+1))
        self.xauxil = np.zeros((self.legmod+1, self.legmod+1, self.kadim+1))
        self.zauxil = np.zeros((self.legmod+1, self.legmod+1, self.kadim+1))
        self.uauxil = np.zeros((self.legmod+1, self.legmod+1, self.kadim+1))
        self.tauxil = np.zeros((self.legmod+1, self.legmod+1, self.kadim+1))
        self.wauxil = np.zeros((self.legmod+1, self.legmod+1, self.kadim+1))
         
        # 初始化数组
        for n in range(self.legmod + 1):
            for m in range(self.legmod + 1):
                for k in range(self.kamax + 1):
                    self.qleg00[n, m, k] = 0.0
                    self.qleg01[n, m, k] = 0.0
                    self.qleg11[n, m, k] = 0.0
                
                for k in range(self.kadim + 1):
                    self.qauxil[n, m, k] = 0.0
                    self.tauxil[n, m, k] = 0.0
                    self.uauxil[n, m, k] = 0.0
                    self.wauxil[n, m, k] = 0.0
                    self.xauxil[n, m, k] = 0.0
                    self.zauxil[n, m, k] = 0.0

        # 根据ipheat条件初始化qauxil
        if self.ipheat == 1:
            for k in range(self.kadim + 1):
                akk = 2.0 * k
                self.qauxil[0, 0, k] = 4.0 / ((1.0 + akk) * (3.0 + akk))
        else:
            for k in range(self.kadim + 1):
                akk = 2.0 * k
                self.qauxil[0, 0, k] = 16.0 / ((1.0 + akk) * (3.0 + akk) * (5.0 + akk))

        # 计算边界值
        kn1 = self.kadim - 1
        kn2 = self.kadim - 2

        for k in range(kn1 + 1):
            self.xauxil[0, 0, k] = self.qauxil[0, 0, k + 1]

        for k in range(kn2 + 1):
            self.zauxil[0, 0, k] = self.qauxil[0, 0, k + 2]

        for k in range(self.kamax + 1):
            self.qleg00[0, 0, k] = self.qauxil[0, 0, k]

        # 计算边界值，所有n对应m=0和所有m对应n=0
        for n in range(1, self.legmod + 1):
            ann = 2.0 * n
            bnn = 4.0 * n - 1.0

            for k in range(kn1 + 1):
                self.qauxil[n, 0, k] = 2.0 * self.xauxil[n-1, 0, k] - self.qauxil[n-1, 0, k] - \
                                    (self.xauxil[n-1, 0, k] - self.qauxil[n-1, 0, k]) / ann
                self.qauxil[0, n, k] = self.qauxil[n, 0, k]
                self.uauxil[n, 0, k] = 0.0
                self.uauxil[0, n, k] = self.uauxil[0, n-1, k] + bnn * self.zauxil[0, n-1, k]
                self.tauxil[n, 0, k] = 0.0
                self.tauxil[0, n, k] = self.tauxil[0, n-1, k] + bnn * self.xauxil[n-1, 0, k]
                self.wauxil[n, 0, k] = 0.0
                self.wauxil[0, n, k] = 0.0

            for k in range(self.kamax + 1):
                self.qleg00[n, 0, k] = self.qauxil[n, 0, k]
                self.qleg00[0, n, k] = self.qauxil[0, n, k]
                self.qleg01[0, n, k] = self.tauxil[0, n, k]

            km1 = kn1
            km2 = km1 - 1
            kn1 = kn2
            kn2 = kn1 - 1

            for k in range(kn1 + 1):
                self.xauxil[n, 0, k] = 2.0 * self.qauxil[n, 0, k+1] - self.xauxil[n-1, 0, k] - \
                                    (self.qauxil[n, 0, k+1] - self.xauxil[n-1, 0, k]) / (ann + 1.0)
                self.xauxil[0, n, k] = 2.0 * self.zauxil[n-1, 0, k] - self.xauxil[0, n-1, k] - \
                                    (self.zauxil[n-1, 0, k] - self.xauxil[0, n-1, k]) / ann

            for k in range(kn2 + 1):
                self.zauxil[n, 0, k] = 2.0 * self.xauxil[0, n, k+1] - self.zauxil[n-1, 0, k] - \
                                    (self.xauxil[0, n, k+1] - self.zauxil[n-1, 0, k]) / (ann + 1.0)
                self.zauxil[0, n, k] = self.zauxil[n, 0, k]

            # 循环第二个索引
            for m in range(1, n + 1):
                amm = 2.0 * m
                bmm = 4.0 * m - 1.0

                for k in range(km1 + 1):
                    self.qauxil[n, m, k] = 2.0 * self.xauxil[m-1, n, k] - self.qauxil[n, m-1, k] - \
                                        (self.xauxil[m-1, n, k] - self.qauxil[n, m-1, k]) / amm
                    self.qauxil[m, n, k] = self.qauxil[n, m, k]

                for k in range(km2 + 1):
                    self.xauxil[n, m, k] = 2.0 * self.zauxil[n, m-1, k] - self.xauxil[n, m-1, k] - \
                                        (self.zauxil[n, m-1, k] - self.xauxil[n, m-1, k]) / amm
                    self.xauxil[m, n, k] = 2.0 * self.qauxil[n, m, k+1] - self.xauxil[m-1, n, k] - \
                                        (self.qauxil[n, m, k+1] - self.xauxil[m-1, n, k]) / (amm + 1.0)
                    self.tauxil[n, m, k] = self.tauxil[n, m-1, k] + bmm * self.xauxil[m-1, n, k]
                    self.tauxil[m, n, k] = self.tauxil[m, n-1, k] + bnn * self.xauxil[n-1, m, k]
                    self.uauxil[n, m, k] = self.uauxil[n, m-1, k] + bmm * self.zauxil[n, m-1, k]
                    self.uauxil[m, n, k] = self.uauxil[m, n-1, k] + bnn * self.zauxil[n-1, m, k]
                    self.wauxil[n, m, k] = self.wauxil[n-1, m-1, k] + \
                                        bnn * self.uauxil[n-1, m-1, k] + bmm * self.uauxil[m-1, n-1, k] + \
                                        bnn * bmm * self.zauxil[n-1, m-1, k]
                    self.wauxil[m, n, k] = self.wauxil[n, m, k]

                km1 = km2
                km2 = km1 - 1

                for k in range(km2 + 1):
                    self.zauxil[n, m, k] = 2.0 * self.xauxil[n, m, k+1] - self.zauxil[n, m-1, k] - \
                                        (self.xauxil[n, m, k+1] - self.zauxil[n, m-1, k]) / (amm + 1.0)
                    self.zauxil[m, n, k] = self.zauxil[n, m, k]

                for k in range(self.kamax + 1):
                    self.qleg00[n, m, k] = self.qauxil[n, m, k]
                    self.qleg00[m, n, k] = self.qauxil[n, m, k]
                    self.qleg01[n, m, k] = self.tauxil[n, m, k]
                    self.qleg01[m, n, k] = self.tauxil[m, n, k]
                    self.qleg11[n, m, k] = self.wauxil[n, m, k]
                    self.qleg11[m, n, k] = self.wauxil[n, m, k]

    # ===================================================================
    # Post-Processing & Output
    # ===================================================================

    def velocity_moment(self, values, derivatives, power):
        """Integrate u**power times the actual cubic Hermite FE function."""
        points, weights = special.roots_legendre(5)
        z = (points + 1.0) / 2.0
        u = self.du * (np.arange(len(values) - 1)[:, None] + z)
        interpolated = (
            values[:-1, None] * self.herm1(z)
            + self.du * derivatives[:-1, None] * self.herm2(z)
            + values[1:, None] * self.herm1(z - 1.0)
            + self.du * derivatives[1:, None] * self.herm2(z - 1.0)
        )
        return float(self.du * np.sum(interpolated * u**power * weights / 2.0))

    def solve_density_constrained(self, banded, rhs):
        """Fix integral F d³u=1 by the homogeneous solution, keeping S unchanged.

        A constant-density source has zero net particle injection. The origin
        value is then a free amplitude, not a particle sink. Two banded RHSs
        impose the density without introducing a dense normalization row.
        """
        source_rhs = rhs.copy()
        source_rhs[0] = 0.0
        boundary_rhs = np.zeros_like(rhs)
        boundary_rhs[0] = 1.0
        solutions = solve_banded(
            (self.iwd, self.iwd), banded, np.column_stack((source_rhs, boundary_rhs))
        )
        coefficients = solutions.reshape(self.nptu, 2 * self.nfvrb, 2)
        masses = [4.0 * np.pi * self.velocity_moment(
            coefficients[:, 0, j], coefficients[:, 1, j], 2
        ) for j in range(2)]
        if not np.isfinite(masses).all() or abs(masses[1]) < np.finfo(float).tiny:
            raise RuntimeError("无法施加粒子数约束：齐次解的粒子矩无效")
        origin_value = (1.0 - masses[0]) / masses[1]
        self.fb = solutions[:, 0] + origin_value * solutions[:, 1]
        if not np.isfinite(self.fb).all():
            raise RuntimeError("稳态求解产生非有限分布")
        self.fb_store = source_rhs + origin_value * boundary_rhs
        self.density_constrained = True
        residual = -self.fb_store.copy()
        for offset in range(-self.iwd, self.iwd + 1):
            start = max(0, -offset)
            stop = min(self.nfdim, self.nfdim - offset)
            residual[start + offset:stop + offset] += (
                banded[self.iwd + offset, start:stop] * self.fb[start:stop]
            )
        self.matrix_relative_residual = float(
            np.linalg.norm(residual, np.inf)
            / max(np.linalg.norm(self.fb_store, np.inf), np.finfo(float).tiny)
        )
        return self.fb

    def solve_stationary(self, ipsi):
        """Solve at fixed density, optionally matching final absorbed RF power."""
        self.matrix()
        seed_matrix = self.fmatrx[self.iwd:3 * self.iwd + 1].copy()
        rhs = self.fb.copy()
        seed_origin = self.origin_weak_row
        seed_diffusion = self.dqlw
        self.power_iterations = 0
        self.power_relative_residual = 0.0
        self.density_constrained = False

        if self.rf_power_mode == 'absorbed' and self.icrh_enabled and self.pwdens > 0:
            if seed_diffusion <= 0 or not np.isfinite(seed_diffusion):
                raise RuntimeError("正吸收功率需要正的准线性扩散系数")
            self.dqlw = 0.0
            self.matrix()
            collision_matrix = self.fmatrx[self.iwd:3 * self.iwd + 1].copy()
            collision_origin = self.origin_weak_row
            ql_matrix = seed_matrix - collision_matrix

            def power_error(scale):
                self.power_iterations += 1
                self.dqlw = scale * seed_diffusion
                self.solve_density_constrained(collision_matrix + scale * ql_matrix, rhs)
                self.disfun(ipsi)
                power = self.zemxw * self.qlrate(self.legmod)
                if not np.isfinite(power):
                    raise RuntimeError("RF 功率自洽计算产生非有限值")
                return power / self.pwdens - 1.0

            upper = 1.0
            error = power_error(upper)
            for _ in range(20):
                if error >= 0:
                    break
                upper *= 2.0
                error = power_error(upper)
            else:
                raise RuntimeError("当前能域与模型无法达到请求的 RF 吸收功率")
            scale = brentq(power_error, 0.0, upper, xtol=1e-12,
                           rtol=self.power_tolerance * 0.1, maxiter=100)
            self.power_relative_residual = abs(power_error(scale))
            if self.power_relative_residual > self.power_tolerance:
                raise RuntimeError("RF 吸收功率迭代未达到要求精度")
            final_matrix = collision_matrix + scale * ql_matrix
            self.origin_weak_row = tuple(
                base + scale * (reference - base)
                for base, reference in zip(collision_origin, seed_origin)
            )
        else:
            self.dqlw = seed_diffusion
            self.solve_density_constrained(seed_matrix, rhs)
            self.disfun(ipsi)
            self.power_iterations = 1
            final_matrix = seed_matrix
            self.origin_weak_row = seed_origin

        self.fmatrx[self.iwd:3 * self.iwd + 1] = final_matrix
        left, right, origin_rhs = self.origin_weak_row
        stride = 2 * self.nfvrb
        self.particle_balance_residual = float(4 * np.pi * (
            left @ self.fb[:stride] + right @ self.fb[stride:2 * stride] - origin_rhs
        ))
        self.density_residual = abs(self.fnorm - 1.0)
        if np.min(self.tbfu[:, 0]) < -1e-12:
            raise ValueError(
                "当前参数和离散设置没有得到非负稳态（各向同性分布为负）；"
                "请检查网格及恒密度 Maxwell 替换模型的适用性，必要时采用密度演化/粒子损失模型"
            )
        append_to_file(
            self.output_paths['read'],
            f"密度约束残差 = {self.density_residual:.6e}\n"
            f"原点弱方程粒子平衡残差 = {self.particle_balance_residual:.6e}\n"
            f"线性方程相对残差 ||Ax-b||∞/||b||∞ = {self.matrix_relative_residual:.6e}\n"
            f"RF 功率模式 = {self.rf_power_mode}; 相对误差 = {self.power_relative_residual:.6e}\n"
        )

    def disfun(self, ipsi):
        """Reconstruct distribution function from solution vector fb."""
        print(" fb ", self.fb[:3])
        for ip in range(self.nptu):
            ibase = 2 * self.nfvrb * ip
            for n in range(self.legmod + 1):
                i = ibase + 2*n
                self.tbfu[ip, n] = self.fb[i]
                self.tbdfu[ip, n] = self.fb[i+1]

        # 保留求解器原始幅值，便于区分源项响应与归一化后的分布形状。
        self.raw_tbfu = self.tbfu.copy()
        self.raw_tbdfu = self.tbdfu.copy()

        # 所有阶使用同一归一化因子，保持角向结构不变。
        self.fnorm = 4.0 * np.pi * self.velocity_moment(self.tbfu[:, 0], self.tbdfu[:, 0], 2)
        if not getattr(self, 'density_constrained', False):
            if not np.isfinite(self.fnorm) or self.fnorm <= 0:
                raise ValueError("分布的粒子数必须为正且有限")
            self.tbfu /= self.fnorm
            self.tbdfu /= self.fnorm

        # print(f"保存F(v)至11_Fv.txt")
        # tbfu_new = self.tbfu.copy()
        # # tbfu_new[tbfu_new > 0] = 0
        # with open('11_Fv.txt', 'w') as f:
        #     f.write("=== Readable Matrix Format ===\n")
        #     f.write("Format: [value] (negative) or [MARKED] (non-negative)\n\n")
            
        #     for i in range(tbfu_new.shape[0]):
        #         row_values = []
        #         for j in range(tbfu_new.shape[1]):
        #             val = tbfu_new[i, j]
        #             if val < 0:
        #                 row_values.append(f"[{val:.2e}]")
        #             else:
        #                 row_values.append("[---------]")
        #         f.write(f"Row {i}: {' '.join(row_values)}\n")
        
        
        append_to_file(self.output_paths["read"], f"Fn(v)归一化参数 fnorm = {self.fnorm}\n")

    def dfparm(self, ipsi, itry):
        """检查收敛性，并计算加热率和碰撞损失
            收敛iok=1,否则为0
        
        Args:
            ipsi: 空间网格点索引
        """
        self.rteffx = np.zeros(5)  # 对应real(r8), dimension(5)
        self.rteffz = np.zeros(5)
        self.istp = np.zeros(2)
        pwnbi_dimensionless = 0.0
        pwnbi_l_dimensionless = 0.0

        if self.nbi_enabled:
            zconv_nbi = 1.609E-3 * self.zdeni[self.nspele-1, ipsi-1] * \
                        self.ctx.aconc[self.ispres] * self.ctx.tempi[self.ispres]
            
            if self.zemxw > 1e-20:
                # 1. 快离子动能注入：只有当前求解的物种是 NBI 注入物种时才加
                if self.ispres == self.ispec: # 假设 ispec 是 NBI 注入的少数离子索引
                    pwnbi_dimensionless = self.pwnbi * (self.zconv_nbi / self.zemxw)
                    # pwnbi_dimensionless = self.pwnbi
                    
                # 2. 电离损失：只有当前求解的物种是电子时才加（作为电子的损失）
                if self.ispres == self.nspele - 1: # 假设 nspele-1 是电子
                    pwnbi_l_dimensionless = self.pwnbi_l * (self.zconv_nbi / self.zemxw)


        # # ================== 插入硬核诊断代码 V2 ==================
        # if self.nbi_enabled and itry == 1: # 只在第一次迭代时打印
        #     print("\n" + "="*70)
        #     print(f"【硬核诊断 V2】半径索引 {ipsi}, 迭代 {itry} : 源项代数守恒性检查")
        #     print("="*70)
            
        #     # 1. 测量：背景碰撞源项 (保证粒子数守恒的项)
        #     int_psourc = 4.0 * np.pi * self.qlmomu(self.tab_source, self.du, self.nptu, 0)
            
        #     # 2. 直接读取 NBI 源项的零阶系数 (这代表了归一化空间中的粒子注入率)
        #     sum_sig0 = np.sum(self.sig[:, 0])
            
        #     # 3. 读取当前计算出的 NBI 功率相关参数
        #     print(f"  [背景源] ∫S_back (应接近0)    = {int_psourc:.6e}")
        #     print(f"  [NBI系数] Σ sig(k,0)          = {sum_sig0:.6e}")
        #     print(f"  [NBI功率] 原始 pwnbi          = {self.pwnbi:.6e}")
        #     print(f"  [NBI缩放] dnbi                = {self.dnbi:.6e}")
        #     print(f"  [NBI目标] pwdens_nbi          = {self.pwdens_nbi:.6e}")
            
        #     # 4. 核心诊断：检查 source 函数中的代数逻辑
        #     # 在你的 source 函数中，最终源项是：self.source_nbi[0] * self.dnbi
        #     # 我们来反推这个操作带来的实际物理注入率
        #     actual_nbi_particle_rate = sum_sig0 * self.dnbi * 100
        #     print(f"  >>> [关键] 实际 NBI 粒子注入率 (sig0) = {actual_nbi_particle_rate:.6e}")
            
        #     # 5. 诊断判断
        #     if abs(int_psourc) > 1e-3:
        #         print("  ❌ 警告：背景源项 ∫S_back 不为0！碰撞平衡被打破！")
            
        #     if self.dnbi < 1e-6 or self.dnbi > 1e6:
        #         print("  ❌ 警告：dnbi 缩放因子极端异常！功率归一化逻辑可能崩溃！")
                
        #     if actual_nbi_particle_rate < 0:
        #         print("  ❌ 警告：实际 NBI 粒子注入率为负！源项在抽取粒子！")
                
        #     print("="*70 + "\n")
        # # ================== 插入硬核诊断代码 V2 结束 ==================

        # 扩散功率，加热率和碰撞损失
        legsup = self.legmod
        hrate = self.zemxw * self.qlrate(legsup)
        self.hrate = hrate
        # 测试功率平衡
        isp = -1
        self.pwcoll[1] = -self.zemxw * self.exchcl(isp)

        # 计算功率平衡
        # 在 dfparm 中计算功率平衡残差时，必须包含 NBI
        # self.zbalpw = hrate + self.pwsrce + pwnbi_dimensionless + self.pwcoll[1]

        # self.zbalpw = hrate + self.pwcoll[1] + self.pwsrce + self.pwnbi + self.pwnbi_l # Wql + Wcoll + Wsrc + Wnbi + Wnbiloss = 0
        self.zbalpw = hrate + self.pwcoll[1] + self.pwsrce / self.fnorm
        # f_raw / fnorm 满足的方程，其固定源也应除以 fnorm。
        # 这是体积能量诊断，未计边界通量，不能作为线性或物理收敛判据。
        self.bulk_power_residual = (
            hrate + self.pwcoll[1] + (self.pwsrce + self.nbi_source_power) / self.fnorm
        )

        append_to_file(self.output_paths["read"], f"pwcoll = {self.pwcoll}\n")
        append_to_file(self.output_paths["read"], f"部分功率残差（未含NBI源和边界通量） zbalpw = {self.zbalpw}= {hrate} + {self.pwcoll[1]} + {self.pwsrce / self.fnorm}\n")
        append_to_file(
            self.output_paths["read"],
            f"归一化解体积功率残差（未计边界通量，含积分离散误差） = {self.bulk_power_residual:.6e} W/cm³\n"
        )

        # if abs(self.zbalpw) > self.accuracy_goal:
        #     self.iok = 0
        #     if itry < self.itrmax:
        #         return
        
        # Linear, density, particle and energy checks are independent of the
        # angular/domain convergence checks below.
        self.iok = int(
            getattr(self, 'matrix_relative_residual', np.inf) < 1e-8
            and getattr(self, 'density_residual', np.inf) < 1e-10
            and abs(getattr(self, 'particle_balance_residual', np.inf)) < 1e-8
            and abs(self.bulk_power_residual) < 1e-7 * max(1.0, abs(hrate), abs(self.nbi_source_power))
        )
        self.fb1_prv = self.fb[0]
        
        # 碰撞功率与 hrate 必须使用同一份解，不能再按参考输入功率缩放。
        self.zpnorm = 1.0
        
        # 向其他物种的碰撞转移
        for isp in range(self.nspele):
            self.pwexch[1, isp] = -self.zemxw * self.zpnorm * self.exchcl(isp)
            self.tab_pwexch[isp, ipsi-1] = self.pwexch[0, isp] - self.pwexch[1, isp]
            self.pwgain[1, isp, ipsi-1] += self.tab_pwexch[isp, ipsi-1]
        
        isp = -1
        self.pwcoll[1] = -self.zemxw * self.zpnorm * self.exchcl(isp)

        
        # 平行和垂直温度
        temperature = self.ctx.tempi[self.ispres]
        etp0 = 4.0 * np.pi * self.velocity_moment(self.tbfu[:, 0], self.tbdfu[:, 0], 4) * temperature
        etp2 = 0.0
        mean_u_parallel = 0.0
        if self.legmod >= 1:
            etp2 = 4.0 * np.pi * self.velocity_moment(self.tbfu[:, 1], self.tbdfu[:, 1], 4) * temperature
            mean_u_parallel = 4.0 * np.pi / 3.0 * self.velocity_moment(
                self.tbfu[:, 1], self.tbdfu[:, 1], 3
            )
        
        etp22 = 0.0
        if self.legmod >= 2:
            etp22 = 4.0 * np.pi * self.velocity_moment(self.tbfu[:, 2], self.tbdfu[:, 2], 4) * temperature
        
        print(f"勒让德分量积分: etp0={etp0:.4e}, etp1={etp2:.4e}, etp2={etp22:.4e}")
        
        self.htemp = 2.0 * etp0 / 3.0
        self.htperp = (2.0 / 3.0) * (etp0 - etp22 / 5.0)  # 使用 etp22 (f_2)
        self.htpar = (2.0 / 3.0) * (etp0 + 2.0 * etp22 / 5.0) # 使用 etp22 (f_2)
        self.mean_energy_keV = etp0
        self.mean_parallel_energy_keV = self.htpar / 2.0
        self.mean_perpendicular_energy_keV = self.htperp
        self.mean_parallel_velocity_ms = mean_u_parallel * self.vth[self.ispres] * 1e-2
        self.thermal_parallel_temperature_keV = self.htpar - 2.0 * temperature * mean_u_parallel**2

        # self.htemp = 2.0 * etp0 / 3.0
        # self.htperp = (2.0 / 3.0) * (etp0 - etp2 / 5.0)
        # self.htpar = (2.0 / 3.0) * (etp0 + 2.0 * etp2 / 5.0)

        # 常数mu处的分布函数
        self.tfecmu = self.tbfu @ self.plgcmu
        self.quality_warnings = []
        if self.tfecmu.min() < -max(1e-12, self.f_min * 0.01):
            self.quality_warnings.append("分布有显著负值；需要提高角向/速度分辨率并检查模型适用性")
        if np.max(self.tfecmu[-1]) >= self.f_min:
            self.quality_warnings.append("高能尾触及能域上界；需要扩大能域并保持速度分辨率")
        if not self.iok:
            self.quality_warnings.append("离散方程、粒子数或功率平衡检验未通过")
        for warning in self.quality_warnings:
            append_to_file(self.output_paths['read'], f"物理验证未通过：{warning}\n")
        self.distribution_checks_passed = not self.quality_warnings
        
        # 平滑处理
        # self._postprocess_tfecmu_tail()


        # # ================= 消除吉布斯震荡 =================
        # # 1. 物理截断：分布函数不能为负值，将负值强制设为极小正数
        # f_floor = 1.0e-30
        # self.tfecmu = np.maximum(self.tfecmu, f_floor)
        
        # # 2. 高斯滤波：在 mu 方向 (axis=1) 进行轻微平滑，消除高频毛刺
        # # sigma=1 表示平滑宽度约为 1-2 个网格点，足以消除勒让德震荡，同时保留 NBI 的宏观偏心特征
        # from scipy.ndimage import gaussian_filter1d
        # self.tfecmu = gaussian_filter1d(self.tfecmu, sigma=1.0, axis=1)
        # # ==============================================================

        # 将当前径向点的 2D 分布函数存入 3D 历史数组
        # 注意：ipsi 在此函数中是 1-based 索引（从 1 开始），数组需转为 0-based
        ipsi_idx = ipsi - 1
        if 0 <= ipsi_idx < self.nrdpsi:
            self.f_v_mu_psi[:, :, ipsi_idx] = self.tfecmu

        
        # if "Fi" in self.output_paths:
        #     with open(self.output_paths["Fi"], 'w', encoding='utf-8') as f: 
        #         np.savetxt(self.output_paths["Fi"], self.tfecmu)

        # 防止大平行速度时的不收敛
        ip2 = int(np.floor(self.nmu / 2.0))
        print(f"ip2 = {ip2} =({self.nmu} / 2.0)")
        # self.nmueff[0] = self.nptu-1
        ztc = 0.25 * self.tfecmu[self.nptu-1, ip2]
        for imu in range(0, self.nmu):
            self.nmueff[imu] = self.nptu
            # ztc = self.tfecmu[self.nptu-1, imu]
            for ip in range(self.nptu):
                if self.tfecmu[ip, imu] <= ztc :
                    self.nmueff[imu] = ip
                    break
        
        # 有效尾部温度
        print(f"最大能量值= {self.tbkev[self.nptu-1]}")
        self.istp[0] = (22 * self.nmueff[self.nmu-1]) // 100
        self.istp[1] = (22 * self.nmueff[ip2]) // 100
        print(f"有效尾部温度:{self.istp}")
        min_fE = self.f_min/10
        
        for k in range(5):
            i1 = (k) * int(self.istp[0])
            i2 = i1 + 2

            print(f"索引: i1={i1}, i2={i2}")

            zdepar = self.tbkev[i2] - self.tbkev[i1]
            if self.tfecmu[i2, self.nmu-1] <= 0 or self.tfecmu[i1, self.nmu-1] <= 0:
                self.rteffz[k] = self.ctx.tempi[self.ispres]
            else:
                zdflog = np.log(self.tfecmu[i2, self.nmu-1]) - np.log(self.tfecmu[i1, self.nmu-1])
                # 防止除以零，如果对数差极小，说明尾部极其平坦，温度趋于无穷大，这里截断为最大能量
                if abs(zdflog) < 1.0e-10:
                    self.rteffz[k] = self.tbkev[self.nptu-1] # 赋予最大能量值
                    print(f"相差过小，rteffz赋最大能量值= {self.tbkev[self.nptu-1]}")
                else:
                    self.rteffz[k] = -zdepar / zdflog
                    print(f"rteffz[{k}] = {self.rteffz[k]} = {-zdepar} / {zdflog} = {-zdepar} / ({np.log(self.tfecmu[i2, self.nmu-1])} - {np.log(self.tfecmu[i1, self.nmu-1])})")  # 调试打印
            print(f"rteffz[{k}] = {self.rteffz[k]}")  # 调试打印
            
            i1 = (k) * int(self.istp[1])
            i2 = i1 + 2

            print(f"索引: i1={i1}, i2={i2}")

            zdepar = self.tbkev[i2] - self.tbkev[i1]
            if self.tfecmu[i2, ip2] <= 1e-15 or self.tfecmu[i1, ip2] <= 1e-15:
                self.rteffx[k] = self.ctx.tempi[self.ispres]
            else:
                zdflog = np.log(self.tfecmu[i2, ip2]) - np.log(self.tfecmu[i1, ip2])
                if abs(zdflog) < 1.0e-10:                       # 防止除以零
                    self.rteffx[k] = self.tbkev[self.nptu-1]
                    print(f"相差过小，rteffx赋最大能量值= {self.tbkev[self.nptu-1]}")
                else:
                    self.rteffx[k] = -zdepar / zdflog
                    print(f"rteffx[{k}] = {self.rteffx[k]} = {-zdepar} / {zdflog} = {-zdepar} / ({np.log(self.tfecmu[i2, ip2])} - {np.log(self.tfecmu[i1, ip2])})")  # 调试打印
                # if abs(zdflog) < 1e-6:
                #     self.rteffx[k] = self.ctx.tempi[self.ispres]
                # else:
                #     self.rteffx[k] = -zdepar / zdflog
                #     # 防止非有限值或负值
                #     if not np.isfinite(self.rteffx[k]) or self.rteffx[k] < 0.0:
                #         self.rteffx[k] = self.ctx.tempi[self.ispres]
            print(f"rteffx[{k}] = {self.rteffx[k]}")  # 调试打印

                # if abs(zdflog) < 1e-10:
                #     self.rteffx[k] = self.ctx.tempi[self.ispres]
                # else:
                #     self.rteffx[k] = -zdepar / zdflog
                #     # 防止非有限值
                #     if not np.isfinite(self.rteffx[k]) or self.rteffx[k] < 0.0:
                #         self.rteffx[k] = self.ctx.tempi[self.ispres]

        # 计算平均有效温度
        ztz = 0.0
        ztx = 0.0
        itz = 0
        itx = 0
        for k in range(2):
            if self.rteffz[k] > 0.0:
                ztz += self.rteffz[k]
                itz += 1
            if self.rteffx[k] > 0.0:
                ztx += self.rteffx[k]
                itx += 1    
        
        if itz != 0:
            self.ttblkz = ztz / itz
        if itx != 0:
            self.ttblkx = ztx / itx

        ztz = 0.0
        ztx = 0.0
        itz = 0
        itx = 0             
        for k in range(2, 5):
            if self.rteffz[k] > 0.0:
                ztz += self.rteffz[k]
                itz += 1
            if self.rteffx[k] > 0.0:
                ztx += self.rteffx[k]
                itx += 1
        
        if itz != 0:
            self.ttailz = ztz / itz
        if itx != 0:
            self.ttailx = ztx / itx

        # fmin = 1E-12
        # # 计算ip1
        # ip1 = int(np.floor(2.0 / self.du)) + 1
        # ip2 = int(np.floor(self.nmu / 2.0))
        # print(f"ip1 = {ip1} =(2.0 / {self.du}) + 1, ip2 = {ip2} =({self.nmu} / 2.0)")

        # # ttail的计算
        # self.ttail = 0.0
        # f_avg = np.mean(self.tfecmu, axis=1)      #pitch angle平均后的速度分布，一维
        # for ip in range(ip1, self.nptu):
        #     if f_avg[ip] <= fmin:
        #         break
        #     zdepar = self.tbkev[ip] - self.tbkev[ip - 1]
        #     zdflog = np.log(f_avg[ip]) - np.log(f_avg[ip - 1])
        #     rteffx = -zdepar / zdflog
        #     if rteffx > self.ttail:
        #         self.ttail = rteffx
        #     else:
        #         break

        # # ttailx的计算
        # self.ttailx = 0.0
        # for ip in range(ip1, self.nptu):
        #     if self.tfecmu[ip, 0] <= fmin:
        #         break
        #     zdepar = self.tbkev[ip] - self.tbkev[ip - 1]
        #     zdflog = np.log(self.tfecmu[ip, ip2]) - np.log(self.tfecmu[ip - 1, ip2])
        #     rteffx = -zdepar / zdflog
        #     if rteffx > self.ttailx:
        #         self.ttailx = rteffx
        #     else:
        #         break

        # # ttailz的计算
        # ztc = 0.25 * self.tfecmu[self.nptu - 1, 0]
        # self.ttailz = 0.0
        # for ip in range(ip1, self.nptu):
        #     if self.tfecmu[ip, self.nmu - 1] <= fmin or self.tfecmu[ip, self.nmu - 1] < ztc:
        #         break
        #     zdepar = self.tbkev[ip] - self.tbkev[ip - 1]
        #     zdflog = np.log(self.tfecmu[ip, self.nmu - 1]) - np.log(self.tfecmu[ip - 1,self.nmu - 1])
        #     rteffz = -zdepar / zdflog
        #     if rteffz > self.ttailz:
        #         self.ttailz = rteffz
        #     else:
        #         break

        # # Effective bulk temperatures的计算
        # ip = int(np.floor(1.0 / self.du)) + 1

        # zdepar = 0

        # # ttblkx = self.tempi[self.ispec]
        # # ttblkz = self.tempi[self.ispec]
        # if self.tfecmu[ip, 0] > fmin:
        #     zdepar = self.tbkev[ip]
        #     zdflog = np.log(self.tfecmu[ip, ip2]) - np.log(self.tfecmu[0, ip2])
        #     self.ttblkx = -zdepar / zdflog

        # if self.tfecmu[ip, self.nmu - 1] > fmin:
        #     zdflog = np.log(self.tfecmu[ip, self.nmu - 1]) - np.log(self.tfecmu[0, self.nmu - 1])
        #     self.ttblkz = -zdepar / zdflog


    def _postprocess_tfecmu_tail(self):
        """
        处理尾部震荡
        Stabilize tfecmu in the high-energy tail.

        When tfecmu becomes very small, the truncated Legendre reconstruction can
        introduce spurious sign changes / oscillations. These show up as gaps
        (NaN after thresholding) and non-physical "re-appearing" tails on log plots.

        Strategy:
        - enforce positivity (clip negatives to 0) and remove the u=0 spike by
          setting f[0]=f[1] (u=0 has zero weight in moments anyway);
        - detect a “bump” (df/du>0) at sufficiently large E/Ti where f is already
          small, and from that point on replace the tail by a monotone exponential
          continuation fitted from nearby points in log-space.
        """
        if self.tfecmu is None or self.tbkev is None:
            return
        if self.tfecmu.shape[0] != self.nptu or self.tfecmu.shape[1] != self.nmu:
            return

        ti = float(self.ctx.tempi[self.ispres]) if self.ispres < len(self.ctx.tempi) else None
        if ti is None or not np.isfinite(ti) or ti <= 0.0:
            return

        e_over_t = self.tbkev / ti
        umax2 = float(e_over_t[-1]) if e_over_t.size else None
        if umax2 is None or not np.isfinite(umax2) or umax2 <= 0.0:
            return

        # Avoid near-origin numerical artifacts (u=0 has zero weight in moments, but
        # can create a visible vertical segment on log plots).
        min_e_over_t = 2.0

        # Default "tail" start as a fraction of max E/Ti.
        # Fundamental: umax2=400 -> start ~ 100
        # Harmonic:    umax2=30  -> start ~ 7.5
        tail_start = 0.3 * umax2
        default_start_idx = int(np.argmax(e_over_t >= tail_start)) if np.any(e_over_t >= tail_start) else self.nptu - 1
        if default_start_idx <= 1:
            default_start_idx = 2

        ref_tail0 = float(self.tfecmu[self.nptu - 1, 0])
        tiny = np.finfo(np.float64).tiny
        if not np.isfinite(ref_tail0) or ref_tail0 <= 0.0:
            ref_tail0 = tiny
        floor = max(tiny, ref_tail0 * 1.0e-12)

        for imu in range(self.nmu):
            f_raw = self.tfecmu[:, imu]
            f_raw = np.where(np.isfinite(f_raw), f_raw, 0.0)
            f = np.maximum(f_raw, 0.0)
            if f.size >= 2:
                f[0] = f[1]

            f_max = float(np.max(f)) if f.size else 0.0
            small_threshold = max(f_max * 2.0e-3, floor)

            # If we see spurious "bump" (increasing f with energy) once the
            # distribution is already small, smooth from that point onward.
            df = np.diff(f)
            bump = np.where(
                (df > 0.0)
                & (e_over_t[:-1] >= min_e_over_t)
                & (f[:-1] <= small_threshold)
            )[0]
            if bump.size:
                start_idx = max(2, int(bump[0]))
            else:
                start_idx = default_start_idx

            tail = f[start_idx:]
            if tail.size < 3:
                self.tfecmu[:, imu] = f
                continue
            if not np.any(np.diff(tail) > 0.0):
                self.tfecmu[:, imu] = f
                continue

            # Smooth the tail with a monotone exponential decay anchored at the
            # stitching point. This avoids oscillations and avoids the long
            # flat segments produced by pure monotone-envelope methods.
            fit_window = 60
            fit_lo = max(1, start_idx - fit_window)
            fit_hi = start_idx + 1
            x = e_over_t[fit_lo:fit_hi]
            y = np.full_like(x, np.nan, dtype=np.float64)
            raw_slice = f_raw[fit_lo:fit_hi]
            pos = raw_slice > floor
            if np.any(pos):
                y[pos] = np.log(raw_slice[pos])
            good = np.isfinite(x) & np.isfinite(y)

            slope = np.nan
            if np.count_nonzero(good) >= 3:
                xg = x[good]
                yg = y[good]
                # # 使用加权最小二乘，给近端点更高权重
                # weights = np.exp(-0.1 * (xg - xg[0]))  # 指数衰减权重
                # A = np.column_stack([np.ones_like(xg), xg])
                # W = np.diag(weights)
                # coef, *_ = np.linalg.lstsq(W @ A, W @ yg, rcond=None)
                
                # # A = np.column_stack([np.ones_like(xg), xg])
                # # coef, *_ = np.linalg.lstsq(A, yg, rcond=None)
                # slope = float(coef[1])
                # 使用RANSAC-like方法进行鲁棒拟合
                best_slope = np.nan
                best_score = np.inf
                for _ in range(10):  # 多次随机采样
                    sample_idx = np.random.choice(len(xg), size=min(3, len(xg)), replace=False)
                    x_sample = xg[sample_idx]
                    y_sample = yg[sample_idx]
                    A = np.column_stack([np.ones_like(x_sample), x_sample])
                    coef, *_ = np.linalg.lstsq(A, y_sample, rcond=None)
                    trial_slope = float(coef[1])
                    # 计算拟合分数
                    residuals = yg - (coef[0] + trial_slope * xg)
                    score = np.median(np.abs(residuals))
                    if score < best_score:
                        best_score = score
                        best_slope = trial_slope
                slope = best_slope

            if not np.isfinite(slope) or slope >= -1e-8:
                # Fallback: take a robust (median) negative local slope if possible.
                y2 = np.log(np.maximum(f[fit_lo:fit_hi], floor))
                dx = np.diff(x)
                dy = np.diff(y2)
                local = np.where(dx > 0, dy / dx, np.nan)
                local = local[np.isfinite(local) & (local < 0.0)]
                
                if local.size > 0:
                    # 使用分位数方法获得更稳定的斜率估计
                    slope = float(np.percentile(local, 25))  # 使用25%分位数
                else:
                    slope = -1e-6

            # 确保斜率在合理范围内
            slope = np.clip(slope, -1e-4, -1e-7)  # 限制斜率范围

            #     # 使用加权中位数，更关注近端点
            #     weights = np.exp(-0.1 * np.arange(len(local)))
            #     slope = float(np.average(local, weights=weights)) if local.size else -1e-6
            #     # slope = float(np.median(local)) if local.size else -1e-6

            # # Ensure strictly decreasing (in log-space) beyond start_idx.
            # slope = min(slope, -5e-7)

            x0 = float(e_over_t[start_idx - 1])
            y0 = float(np.log(max(f[start_idx - 1], floor)))
            
            y_tail = y0 + slope * (e_over_t[start_idx:] - x0)
            tail_fit = np.exp(y_tail)

            # 使用平滑的过渡而不是硬性的最小值累积
            alpha = 0.5  # 平滑因子
            
            # tail_fit = alpha * tail_fit + (1 - alpha) * np.minimum.accumulate(tail_fit)
            # tail_fit = np.minimum(tail_fit, f[start_idx - 1])

            # tail_fit = np.minimum.accumulate(np.minimum(tail_fit, f[start_idx - 1]))
            
            # 确保单调性
            tail_fit = np.minimum.accumulate(tail_fit)
            tail_fit = np.minimum(tail_fit, f[start_idx - 1])
            
            # 应用平滑
            for i in range(1, len(tail_fit)):
                tail_fit[i] = alpha * tail_fit[i] + (1 - alpha) * tail_fit[i-1]

            f[start_idx:] = tail_fit
            self.tfecmu[:, imu] = f

            self.tfecmu = np.maximum(self.tfecmu, 1e-30)
    
    def regrid_to_uniform_energy(self):
        """Build dn/(dE dmu) on a shared keV grid without modifying f(u,mu,psi).

        The stored velocity distribution satisfies integral f d^3u = 1.
        Since E = T_i u^2, d^3u/(dE dmu) = pi sqrt(E)/T_i^(3/2).
        Values beyond a local solved energy domain are zero-filled; this remapping
        does not establish that the physical high-energy tail is converged.
        """
        distribution = np.asarray(self.f_v_mu_psi)
        temperatures = np.asarray(self.ztempi[self.ispres, :], dtype=float)
        densities = np.asarray(self.zdeni[self.ispres, :], dtype=float) * 1e19
        if distribution.shape != (self.nptu, self.nmu, self.nrdpsi):
            raise ValueError("Velocity distribution must have shape (nptu, nmu, nrdpsi)")
        if (temperatures.shape != (self.nrdpsi,) or densities.shape != (self.nrdpsi,)
                or not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0)
                or not np.all(np.isfinite(densities)) or np.any(densities < 0)):
            raise ValueError("Selected-species temperatures must be positive and densities nonnegative")

        energy_ratio = self.r_E_1 if self.ipheat == 1 else self.r_E_2
        velocity = np.linspace(0, np.sqrt(energy_ratio), self.nptu)
        energy_grid = np.linspace(0, energy_ratio * temperatures.max(), self.nptu)
        spectrum = np.zeros(distribution.shape, dtype=float)
        for ipsi, temperature in enumerate(temperatures):
            local_velocity = np.sqrt(energy_grid / temperature)
            jacobian = densities[ipsi] * np.pi * np.sqrt(energy_grid) / temperature**1.5
            for imu in range(self.nmu):
                spectrum[:, imu, ipsi] = jacobian * np.interp(
                    local_velocity, velocity, distribution[:, imu, ipsi], left=0.0, right=0.0
                )

        self.energy_grid = energy_grid
        self.energy_spectrum_mu_psi = spectrum
        self.local_energy_limits_keV = energy_ratio * temperatures
        self.E_max = energy_grid[-1]
        return energy_grid
    # def integrate_over_radius(self):
    #     """
    #     计算全局快离子绝对数密度的速度空间分布 (体积加权平均)
    #     单位: 个 m^-3
        
    #     公式: N_fast(v, mu) = ∫ [ f(v, mu, psi) * n_e(psi) * 1e19 ] dV(psi) / ∫ dV(psi)
    #     """
    #     if self.f_v_mu_psi is None:
    #         print("警告：未找到 3D 分布函数数据，无法进行径向积分。")
    #         return None
            
    #     print("\n" + "="*60)
    #     print("开始执行径向积分，计算全局实际密度分布 (单位: m^-3)...")
    #     print("="*60)
        
    #     # 1. 确定径向体积权重 dV(psi)
    #     if hasattr(self, 'redvol') and np.sum(self.redvol) > 0:
    #         weights_vol = self.redvol
    #         print("使用物理体积权重 进行径向积分。")
    #     else:
    #         # 退化情况：如果没有体积，假设柱坐标体积正比于 r*dr
    #         if hasattr(self, 'redpsi') and len(self.redpsi) == self.nrdpsi:
    #             weights_vol = np.maximum(self.redpsi, 1e-6) 
    #         else:
    #             weights_vol = np.ones(self.nrdpsi)
    #         print("警告：缺少物理体积权重，使用径向坐标近似权重。")
            
    #     # 2. 获取电子密度剖面 n_e(psi) 并转换单位
    #     if hasattr(self, 'zdeni') and self.zdeni.shape[1] == self.nrdpsi:
    #         # 提取电子密度 (原始单位: 10^19 m^-3)
    #         ne_profile_normalized = self.zdeni[self.nspele-1, :] 
    #         # 【关键】：转换为真实的物理密度 (单位: m^-3)
    #         ne_profile_physical = ne_profile_normalized * 1.0e19
    #         print(f"成功提取电子密度剖面，范围: [{ne_profile_physical.min():.2e}, {ne_profile_physical.max():.2e}] m^-3")
    #     else:
    #         print("警告：未找到电子密度剖面，将退化为纯体积积分！")
    #         ne_profile_physical = np.ones(self.nrdpsi)
            
    #     # 3. 计算总体积 (用于归一化，确保结果仍是密度单位 m^-3)
    #     from scipy.integrate import simpson
    #     total_volume = simpson(weights_vol, x=self.redpsi)
    #     if total_volume <= 0:
    #         print("错误：计算出的总体积非正，无法归一化。")
    #         return None
            
    #     # 4. 构造带密度的被积函数: f(v, mu, psi) * n_e(psi) * dV(psi)
    #     # f_v_mu_psi 形状: (nptu, nmu, nrdpsi)
    #     # ne_profile_physical * weights_vol 形状: (nrdpsi,) -> 需要广播到 (1, 1, nrdpsi)
    #     integrand = self.f_v_mu_psi * (ne_profile_physical * weights_vol).reshape(1, 1, -1)
        
    #     # 5. 沿径向 (axis=2) 积分
    #     numerator = simpson(integrand, x=self.redpsi, axis=2)
        
    #     # 6. 除以总体积，得到体积平均的实际密度分布
    #     N_fast_v_mu = numerator / total_volume
        
    #     print(f"径向积分完成。结果形状: {N_fast_v_mu.shape}")
    #     print(f"✅ 结果的物理单位为: m^-3 (真实的快离子数密度)")
        
    #     # 7. 保存结果
    #     try:
    #         np.savetxt('12_N_fast_v_mu.txt', N_fast_v_mu)
    #         print("✅ 全局快离子密度分布已保存为: 12_N_fast_v_mu.txt")
    #     except Exception as e:
    #         print(f"⚠️ 保存失败: {e}")
            
    #     return N_fast_v_mu

    import numpy as np

    # def regrid_to_max_temp(self):
    #     """
    #     基于最大温度点重新整理速度网格，并计算能量网格
        
    #     返回:
    #         energy_grid: 能量网格，形状为 (nptu,)
    #     """
    #     # 找到最大温度点
    #     max_temp_idx = np.argmax(self.ztempi[self.ispres, :])
    #     max_temp = self.ztempi[self.ispres, max_temp_idx]
        
    #     # 计算最大温度点的速度范围
    #     if self.ipheat == 1:
    #         emax = self.r_E_1 * max_temp
    #     else:
    #         emax = self.r_E_2 * max_temp
        
    #     umax = np.sqrt(emax / max_temp)
    #     du = umax / (self.nptu - 1)
        
    #     # 创建新的速度网格
    #     v_grid = np.linspace(0, umax, self.nptu)
        
    #     # 创建新的分布函数数组
    #     new_f_v_mu_psi = np.zeros_like(self.f_v_mu_psi)
        
    #     # 对每个点进行处理
    #     for ipsi_idx in range(self.nrdpsi):
    #         # 计算当前点的速度网格
    #         current_temp = self.ztempi[self.ispres, ipsi_idx]
            
    #         if self.ipheat == 1:
    #             current_emax = self.r_E_1 * current_temp
    #         else:
    #             current_emax = self.r_E_2 * current_temp
            
    #         current_umax = np.sqrt(current_emax / current_temp)
    #         current_du = current_umax / (self.nptu - 1)
            
    #         # 当前点的速度网格
    #         current_v_grid = np.linspace(0, current_umax, self.nptu)
            
    #         # 对每个pitch angle进行插值
    #         for imu in range(self.nmu):
    #             # 使用numpy的interp进行线性插值
    #             interp_values = np.interp(
    #                 v_grid,                      # 新的速度网格
    #                 current_v_grid,              # 当前点的速度网格
    #                 self.f_v_mu_psi[:, imu, ipsi_idx],  # 当前点的分布函数
    #                 left=0,                      # 低于最小值时的填充值
    #                 right=0                      # 高于最大值时的填充值
    #             )
    #             new_f_v_mu_psi[:, imu, ipsi_idx] = interp_values
        
    #     # 用新的分布函数替换原来的
    #     self.f_v_mu_psi = new_f_v_mu_psi
        
    #     # 保存新的速度网格信息
    #     self.umax = umax
    #     self.du = du
    #     self.v_grid = v_grid
        
    #     # 计算能量网格 E = T * v^2
    #     energy_grid = max_temp * v_grid**2
        
    #     return energy_grid

    # def integrate_over_radius(self):
    #     """
    #     计算全局快离子绝对数密度的速度空间分布 (体积加权平均)
    #     单位: 个 m^-3
        
    #     公式: N_fast(v, mu) = ∫ [ f(v, mu, psi) * n_e(psi) * 1e19 ] dV(psi) / ∫ dV(psi)
        
    #     使用eqdsk文件中的磁场数据来计算体积元
    #     """
    #     import eqdsk
    #     from pathlib import Path
    #     from scipy.integrate import simpson
    #     from scipy.interpolate import interp1d
    #     import numpy as np
        
    #     if self.f_v_mu_psi is None:
    #         print("警告：未找到 3D 分布函数数据，无法进行径向积分。")
    #         return None
            
    #     energy_grid = self.regrid_to_uniform_energy()

    #     print("\n" + "="*60)
    #     print("开始执行径向积分，计算全局实际密度分布 (单位: m^-3)...")
    #     print("="*60)
        
    #     # 读取eqdsk文件
    #     try:
    #         file_path = Path('./gfile/99896M10_ps.geqdsk')
    #         data = eqdsk.file._read_eqdsk(file_path)
    #     except Exception as e:
    #         print(f"错误：无法读取eqdsk文件: {e}")
    #         return None
        
    #     # 检查是单径向点模式还是多径向点模式
    #     is_single_point = (self.nrdpsi == 1)
        
    #     if is_single_point:
    #         print("检测到单径向点模式，直接返回该点的分布函数。")
    #         # 单径向点模式：直接返回该点的分布函数，乘以电子密度
    #         if hasattr(self, 'zdeni') and self.zdeni.shape[1] >= 1:
    #             # 提取电子密度 (原始单位: 10^19 m^-3)
    #             ne_profile_normalized = self.zdeni[self.nspele-1, 0] 
    #             # 转换为真实的物理密度 (单位: m^-3)
    #             ne_profile_physical = ne_profile_normalized * 1.0e19
    #         else:
    #             print("警告：未找到电子密度剖面，假设为1.0e19 m^-3")
    #             ne_profile_physical = 1.0e19
            
    #         # 直接返回该点的分布函数乘以电子密度
    #         N_fast_v_mu = self.f_v_mu_psi[:, :, 0] * ne_profile_physical
            
    #         print(f"单径向点模式，结果形状: {N_fast_v_mu.shape}")
    #         print(f"✅ 结果的物理单位为: m^-3 (真实的快离子数密度)")
            
    #         # 保存结果
    #         try:
    #             np.savetxt('12_N_fast_v_mu.txt', N_fast_v_mu)
    #             print("✅ 全局快离子密度分布已保存为: 12_N_fast_v_mu.txt")
    #         except Exception as e:
    #             print(f"⚠️ 保存失败: {e}")
                
    #         return N_fast_v_mu, energy_grid
        
    #     # 多径向点模式
    #     print("检测到多径向点模式，执行径向积分。")
        
    #     # 1. 使用eqdsk数据计算体积权重
    #     try:
    #         # 获取eqdsk数据
    #         nx = data['nx']
    #         nz = data['nz']
    #         x = data['x']
    #         z = data['z']
    #         psi = data['psi']
    #         psimag = data['psimag']
    #         psibdry = data['psibdry']
            
    #         # 计算归一化的磁面坐标
    #         psinorm = (psi - psimag) / (psibdry - psimag)
            
    #         # 计算体积元
    #         dx = x[1] - x[0]
    #         dz = z[1] - z[0]
    #         R, Z = np.meshgrid(x, z)
    #         dV = 2 * np.pi * R * dx * dz
            
    #         # 计算磁面梯度
    #         dpsi_dx = np.gradient(psi, axis=1)
    #         dpsi_dz = np.gradient(psi, axis=0)
    #         grad_psi = np.sqrt(dpsi_dx**2 + dpsi_dz**2)
            
    #         # 归一化体积元
    #         dV_normalized = dV / grad_psi
    #         dV_normalized /= np.sum(dV_normalized)
            
    #         # 将体积元映射到径向网格
    #         psi_bins = np.linspace(psimag, psibdry, self.nrdpsi+1)
    #         weights_vol = np.zeros(self.nrdpsi)
            
    #         for i in range(self.nrdpsi):
    #             # 找到当前psi范围对应的网格点
    #             mask = (psi >= psi_bins[i]) & (psi < psi_bins[i+1])
    #             weights_vol[i] = np.sum(dV_normalized[mask])
                
    #         print("使用eqdsk数据计算的体积权重。")
    #     except Exception as e:
    #         print(f"警告：使用eqdsk计算体积权重失败: {e}")
    #         # 退化情况：如果没有体积，假设柱坐标体积正比于 r*dr
    #         if hasattr(self, 'redpsi') and len(self.redpsi) == self.nrdpsi:
    #             weights_vol = np.maximum(self.redpsi, 1e-6)
    #         else:
    #             weights_vol = np.ones(self.nrdpsi)
    #         print("使用径向坐标近似权重。")
        
    #     # 2. 获取电子密度剖面 n_e(psi) 并转换单位
    #     if hasattr(self, 'zdeni') and self.zdeni.shape[1] == self.nrdpsi:
    #         # 提取电子密度 (原始单位: 10^19 m^-3)
    #         ne_profile_normalized = self.zdeni[self.nspele-1, :] 
    #         # 转换为真实的物理密度 (单位: m^-3)
    #         ne_profile_physical = ne_profile_normalized * 1.0e19
    #         print(f"成功提取电子密度剖面，范围: [{ne_profile_physical.min():.2e}, {ne_profile_physical.max():.2e}] m^-3")
    #     else:
    #         print("警告：未找到电子密度剖面，将退化为纯体积积分！")
    #         ne_profile_physical = np.ones(self.nrdpsi)
        
    #     # 3. 计算总体积 (用于归一化，确保结果仍是密度单位 m^-3)
    #     # 确保 weights_vol 和 redpsi 长度一致
    #     if len(weights_vol) != len(self.redpsi):
    #         print(f"警告：weights_vol ({len(weights_vol)}) 和 redpsi ({len(self.redpsi)}) 长度不一致，将进行插值对齐。")
    #         interp_func = interp1d(self.redpsi, weights_vol, kind='linear', fill_value="extrapolate")
    #         weights_vol = interp_func(self.redpsi)
        
    #     total_volume = simpson(weights_vol, x=self.redpsi)
    #     if total_volume <= 0:
    #         print("错误：计算出的总体积非正，无法归一化。")
    #         return None
        
    #     # 4. 构造带密度的被积函数: f(v, mu, psi) * n_e(psi) * dV(psi)
    #     # f_v_mu_psi 形状: (nptu, nmu, nrdpsi)
    #     # ne_profile_physical * weights_vol 形状: (nrdpsi,) -> 需要广播到 (1, 1, nrdpsi)
    #     integrand = self.f_v_mu_psi * (ne_profile_physical * weights_vol).reshape(1, 1, -1)
        
    #     # 5. 沿径向 (axis=2) 积分
    #     numerator = simpson(integrand, x=self.redpsi, axis=2)
        
    #     # 6. 除以总体积，得到体积平均的实际密度分布
    #     N_fast_v_mu = numerator / total_volume
        
    #     print(f"径向积分完成。结果形状: {N_fast_v_mu.shape}")
    #     print(f"✅ 结果的物理单位为: m^-3 (真实的快离子数密度)")
        
    #     # 7. 保存结果
    #     try:
    #         np.savetxt('12_N_fast_v_mu.txt', 
    #                     np.column_stack([energy_grid, N_fast_v_mu]), 
    #                     header='Energy_grid(eV)  N_fast(v,mu1)  N_fast(v,mu2) ...')
    #         # np.savetxt('12_N_fast_v_mu.txt', N_fast_v_mu)
    #         # np.savetxt('12_energy_grid.txt', energy_grid)
    #         print("✅ 全局快离子密度分布已保存为: 12_N_fast_v_mu.txt")
    #         # print("✅ 能量网格已保存为: 12_energy_grid.txt")
    #     except Exception as e:
    #         print(f"⚠️ 保存失败: {e}")
            
    #     return N_fast_v_mu, energy_grid

    def integrate_over_radius(self, dvolume_dpsi=None):
        """Return a volume-averaged dn/(dE dmu) spectrum and its energy grid.

        dvolume_dpsi supplies the physical dV/dpsi at the redpsi nodes. With no
        weights, redpsi is explicitly treated as normalized minor radius and the
        cylindrical approximation dV proportional to r dr is used. Only the
        sampled radial interval is averaged; no unsampled volume is extrapolated.
        """
        if self.f_v_mu_psi is None:
            return None, None
        energy_grid = self.regrid_to_uniform_energy()
        spectra = self.energy_spectrum_mu_psi
        if self.nrdpsi == 1:
            averaged = spectra[:, :, 0].copy()
            self.radial_weight_model = "single radial point; no volume average"
        else:
            radius = np.asarray(self.redpsi, dtype=float)
            if (radius.shape != (self.nrdpsi,) or not np.all(np.isfinite(radius))
                    or np.any(np.diff(radius) <= 0)):
                raise ValueError("Radial coordinates must be finite and strictly increasing")
            if dvolume_dpsi is None:
                if np.any(radius < 0):
                    raise ValueError("Cylindrical radial coordinates must be nonnegative")
                weights = radius
                self.radial_weight_model = "cylindrical approximation: dV proportional to r dr"
            else:
                weights = np.asarray(dvolume_dpsi, dtype=float)
                self.radial_weight_model = "explicit dV/dpsi weights"
            if (weights.shape != (self.nrdpsi,) or not np.all(np.isfinite(weights))
                    or np.any(weights < 0)):
                raise ValueError("dV/dpsi weights must be finite, nonnegative, and match redpsi")
            # Trapezoidal weights remain nonnegative on a nonuniform radial mesh.
            volume = trapezoid(weights, x=radius)
            if volume <= 0:
                raise ValueError("Integrated radial volume weight must be positive")
            averaged = trapezoid(spectra * weights[None, None, :], x=radius, axis=2) / volume

        self.radial_energy_spectrum = averaged
        print(f"径向能量谱 dn/(dE dmu) [m^-3 keV^-1]；{self.radial_weight_model}")
        output = os.path.join(self.current_out_dir, "12_energy_spectrum.txt")
        np.savetxt(
            output, np.column_stack((energy_grid, averaged)),
            header=("Energy_keV  dn_dE_dmu [m^-3 keV^-1] for each mu column\n"
                    f"selected_species_index={self.ispres}; {self.radial_weight_model}\n"
                    "Beyond each local solved energy domain: zero-filled, not a convergence claim"),
        )
        return averaged, energy_grid

    def outfpq(self, ientry):
        """输出处理子程序
        
        Args:
            ientry: 控制参数
                -1: 为每个离子物种绘图
                -2: 为电子绘图
                0: 初始化输出
                1: 每个离子物种的最终输出
                2: 所有物种的最终输出
        """
        # 初始化局部变量
        nr = i = j = k = isp = ipsi  = jpheat = 0
        ztcle = ztclm = ztt = psi = 0.0
        zfpi = np.zeros(3)
        zfpe = np.zeros(3)
        self.idlout = 1
        self.igsmhd = 1

        if ientry == -1:
            # 一般输出
            print(f"\n{' '*10}FOKKER-PLANCK QUASILINEAR SOLVER\n")
            print(f"{' '*10}Total coupled power       = {self.power:13.3f} MW")
            # print(f"{' '*10}Representative n_phi      = {self.nphi:13d}")
            print(f"{' '*10}NOTE: Species {self.nspele:3d} are the electrons.")

            # IDL输出
            # if self.idlout != 0:
                # if self.igsmhd != 0:
                #     print(" Auxiliary tables for rho_tor, rho_pol, x/a")
                #     print(" Radial mesh, num. of points")
                #     print(f"{self.nrdpsi:5d}")
                #     print(" rho_pol, FP mesh")
                #     print(" ".join(f"{x:11.4E}" for x in self.redpsi))
                #     print(" X/a, FP mesh")
                #     print(" ".join(f"{x:11.4E}" for x in self.redxoa))
                # else:
                #     print("analytic")

                # print("离子种类")
                # print(f"{self.ctx.nspec:5d}")
                # for isp in range(self.ctx.nspec):
                #     print(f"hf power, 基频 {isp+1}")
                #     print(" ".join(f"{x:11.4E}" for x in self.pwredf[isp]))
                #     print(f"hf power, 谐频 {isp+1}")
                #     print(" ".join(f"{x:11.4E}" for x in self.pwredh[isp]))
                # print("hf power, 电子")
                # print(" ".join(f"{x:11.4E}" for x in self.pwredl))

        elif ientry == 0:
            # 打开输出
            if self.idlout != 0:
                jpheat = self.ipheat
                if self.ibreak != 0:
                    jpheat = -1
                print(" IC harmonic number")
                print(f"{jpheat:5d}")
                if self.ibreak == 0:
                    print(f" *** Ion species: {self.ispres}")
                    print(f"{self.ispres:5} {self.ipsplf[self.ispres, self.ipheat-1]:5}")

            if self.ibreak != 0:
                return

            if self.ipheat == 1:
                print("\n" + " "*10 + "-"*40)
                print(f"{' '*10}Species {self.ispres:2d} heated at the fundamental")
                print(" "*10 + "-"*40)
            else:
                print("\n" + " "*10 + "-"*43)
                print(f"{' '*10}Species {self.ispres:2d} heated at the first harmonic")
                print(" "*10 + "-"*43)

            print("\n" + " "*8 + "Mesh information:")
            print(f"{' '*10}速度空间点数 ={self.nptu:13d}")
            print(f"{' '*10}勒让德多项式数量     ={self.legmod:13d}")
            print(f"{' '*10}径向网格点数   ={self.nrdpsi:13d}")

            self.psinxt = 0.0
            self.dpsxnt = 0.05 if self.joutql >= 1 else 0.25

        elif ientry == 1:
            # 每个离子物种的最终输出
            if self.joutql >= 2:
                print(f"\n{' '*10}Linear and quasilinear heating profiles for species{self.ispres:2d}")
                print(f"{' '*10}A ={self.ctx.atm[self.ispres]:3.1f} Z ={self.ctx.azi[self.ispres]:3.1f} n_i/n_e ={self.ctx.aconc[self.ispres]:7.3f}")

                if self.ipheat == 1:
                    for ipsi in range(self.nrdpsi):
                        print(f" Psi ={self.redpsi[ipsi]:8.3f} P_lin ={self.pwredf[self.ispres, ipsi]:11.3E} "
                            f"P_ql ={self.thrate[self.ispres, ipsi]:11.3E} "
                            f"to elec. ={self.pwexch[1, self.nspele]:11.3E} W/cm3")
                else:
                    for ipsi in range(self.nrdpsi):
                        print(f" Psi ={self.redpsi[ipsi]:8.3f} P_lin ={self.pwredh[self.ispres, ipsi]:11.3E} "
                            f"P_ql ={self.thrate[self.ispres, ipsi]:11.3E} "
                            f"to elec. ={self.pwexch[1, self.nspele]:11.3E} W/cm3")

            if self.joutql >= 2:
                print(f"\n{' '*10}垂直和平行尾部温度")
                if self.ipheat == 1:
                    for ipsi in range(self.nrdpsi):
                        print(f" Psi ={self.redpsi[ipsi]:8.3f} T_i ={self.ztempi[self.ispres, ipsi]:11.3E} "
                            f"T_perp ={self.tavgtx[self.ispres, ipsi]:11.3E} "
                            f"T_par ={self.tavgtz[self.ispres, ipsi]:11.3E} "
                            f"frac. ={self.tmtail[self.ispres, ipsi]:11.3E}")
                else:
                    for ipsi in range(self.nrdpsi):
                        print(f" Psi ={self.redpsi[ipsi]:8.3f} T_i ={self.ztempi[self.ispres, ipsi]:11.3E} "
                            f"T_perp ={self.tavgtx[self.ispres, ipsi]:11.3E} "
                            f"T_par ={self.tavgtz[self.ispres, ipsi]:11.3E}")

                print(f"\n{' '*10}垂直和平行二阶矩等效温度")
                for ipsi in range(self.nrdpsi):
                    print(f" Psi ={self.redpsi[ipsi]:8.3f} T_i ={self.ztempi[self.ispres, ipsi]:11.3E} "
                        f"T2_perp ={self.thtprp[self.ispres, ipsi]:11.3E} "
                        f"T2_par ={self.thtpar[self.ispres, ipsi]:11.3E} "
                        f"T2_tot ={self.thttot[self.ispres, ipsi]:11.3E} keV")

            # IDL输出
            if self.idlout != 0:
                print("Density")
                print(" ".join(f"{x*1.0E13:11.4E}" for x in self.zdeni[self.ispres]))
                print("Temperature")
                print(" ".join(f"{x:11.4E}" for x in self.ztempi[self.ispres]))
                print("T_tail, perp")
                print(" ".join(f"{x:11.4E}" for x in self.tavgtx[self.ispres]))
                print("T_tail, parallel")
                print(" ".join(f"{x:11.4E}" for x in self.tavgtz[self.ispres]))
                print("Second-moment equivalent temperature, perp [keV]")
                print(" ".join(f"{x:11.4E}" for x in self.thtprp[self.ispres]))
                print("Second-moment equivalent temperature, parallel [keV]")
                print(" ".join(f"{x:11.4E}" for x in self.thtpar[self.ispres]))
                print("Avg energy, total")
                print(" ".join(f"{x:11.4E}" for x in self.thttot[self.ispres]))

                if self.ipheat == 1:
                    print("Fraction of min. ions in the tail")
                    print(" ".join(f"{x:11.4E}" for x in self.tmtail[self.ispres]))

                for isp in range(self.nspele):
                    print(f" 损失到物种{isp+1}")
                    print(" ".join(f"{x:11.4E}" for x in self.tab_pwexch[isp]))

            # TORIC波方程中少数族加热的数据文件
            if self.ipheat == 1 and self.iwrdql != 0:
                print(f"{self.ispres:5d}{self.nrdpsi:5d}")
                print(" ".join(f"{x:11.4E}" for x in self.redpsi))
                print(" ".join(f"{x:11.4E}" for x in self.tavbkx[self.ispres]))
                print(" ".join(f"{x:11.4E}" for x in self.tavbkz[self.ispres]))
                print(" ".join(f"{x:11.4E}" for x in self.tavgtx[self.ispres]))
                print(" ".join(f"{x:11.4E}" for x in self.tavgtx[self.ispres]))
                print(" ".join(f"{x:11.4E}" for x in self.tmtail[self.ispres]))

        elif ientry == 2:
            # 所有物种的最终输出
            self.rdhpsi[0] = 0.0
            self.fpsion[0] = 0.0
            self.fpsele[0] = 0.0

            zfpi[2] = 0.0
            zfpe[2] = 0.0
            nr = self.nrdpsi - 2
            ipsi = 1
            
            for i in range(0, nr, 2):
                j = ipsi
                ipsi = j + 1
                self.rdhpsi[ipsi] = self.redpsi[i+2]

                zfpi[0] = zfpi[2]
                zfpi[1] = 0.0
                zfpi[2] = 0.0
                for isp in range(self.ctx.nspec):
                    zfpi[1] += self.pwgain[1, isp, i+1]
                    zfpi[2] += self.pwgain[1, isp, i+2]
                self.fpsion[ipsi] = self.fpsion[j] + zfpi[0]*self.redvol[i] + \
                                4.0*zfpi[1]*self.redvol[i+1] + zfpi[2]*self.redvol[i+2]

                zfpe[0] = zfpe[2]
                zfpe[1] = self.pwgain[1, self.nspele-1, i+1]
                zfpe[2] = self.pwgain[1, self.nspele-1, i+2]
                self.fpsele[ipsi] = self.fpsele[j] + zfpe[0]*self.redvol[i] + \
                                4.0*zfpe[1]*self.redvol[i+1] + zfpe[2]*self.redvol[i+2]

            ztcle = self.fpsele[self.npsihf-1]
            ztclm = self.fpsion[self.npsihf-1]

            ztt = ztcle + ztclm
            if ztt == 0.0:
                ztt = 1.0
            ztcle = 100.0 * ztcle / ztt
            ztclm = 100.0 * ztclm / ztt

            for i in range(1, self.npsihf):
                self.fpsele[i] /= ztt
                self.fpsion[i] /= ztt

            append_to_file(self.output_paths["read"], f"碰撞转移给电子的功率比例 {ztcle:7.2f} %\n")
            append_to_file(self.output_paths["read"], f"碰撞转移给离子的功率比例 {ztclm:7.2f} %\n")

            if self.joutql >= 2:
                print("\n In the radial profiles:")
                print("  P_hf  is the linear power absorbed from the waves")
                print("  P_mxw is the collisional power from all other species")
                print("        in the absence of hf heating")
                print("  P_chf is the hf power transferred from all other species due to collisions")

                for isp in range(self.ctx.nspec):
                    print(f"\n{' '*10}Linear and collisional heating profiles of species{isp+1}")
                    for ipsi in range(self.nrdpsi):
                        print(f" Psi ={self.redpsi[ipsi]:8.3f} P_hf ={self.pwredf[isp, ipsi]+self.pwredh[isp, ipsi]:11.3E} "
                            f"P_mxw ={self.pwgain[0, self.nspele-1, ipsi]:11.3E} "
                            f"P_hf ={self.pwgain[1, self.nspele-1, ipsi]:11.3E} W/cm3")

                print(f"\n{' '*10}Linear and collisional heating profiles of the electrons")
                for ipsi in range(self.nrdpsi):
                    print(f" Psi ={self.redpsi[ipsi]:8.3f} P_hf ={self.pwredl[ipsi]:11.3E} "
                        f"P_mxw ={self.pwgain[0, self.nspele-1, ipsi]:11.3E} "
                        f"P_hf ={self.pwgain[1, self.nspele-1, ipsi]:11.3E} W/cm3")

                print(f"\n{' '*10}Integrated collisional fractions")
                for ipsi in range(self.npsihf):
                    print(f" psi ={self.rdhpsi[ipsi]:6.3f} to e{self.fpsele[ipsi]:11.3E} to i{self.fpsion[ipsi]:11.3E}")

            # IDL输出
            if self.idlout != 0:
                for k in range(2):
                    if k == 0:
                        print("  Coll. exchanges, without hf")
                    else:
                        print("  Coll. exchanges, rf only")
                    for isp in range(self.nspele):
                        print(f" Gains of species{isp+1}")
                        print(" ".join(f"{x:11.4E}" for x in self.pwgain[k, isp, :self.nrdpsi]))

                print("Radial mesh for integrated fractions")
                print(f"{self.npsihf:5d}")
                print(" ".join(f"{x:11.4E}" for x in self.rdhpsi[:self.npsihf]))
                print("Integrated fraction to electrons")
                print(" ".join(f"{x:11.4E}" for x in self.fpsele[:self.npsihf]))
                print("Integrated fraction to bkgr ions")
                print(" ".join(f"{x:11.4E}" for x in self.fpsion[:self.npsihf]))

    def outpsi(self, ipsi):
        """输出等离子体参数和碰撞系数
        
        Args:
            ipsi: 空间网格点索引
        """
        psi = self.redpsi[ipsi-1]
        if psi < self.psinxt:
            return

        if self.joutql >= 1:
            print(f"\n{' '*8}Plasma parameters at psi ={psi:8.3f}")
            zdens = 1.0E13 * self.ctx.dens
            print(f"{' '*10}电子密度            {zdens:13.4E} cm-3")
            print(f"{' '*10}电子温度        {self.tempe:13.3f} keV")
            print(f"{' '*10}加热下温度    {self.ctx.tempi[self.ispres]:13.3f} keV")
            print(f"{' '*10}参考碰撞频率 {self.anuref:13.4E} hz")

            if self.joutql >= 2:
                print(f"{' '*20}nu(i)  ={self.anucol[self.ispres]:13.4E} "
                    f"nu(i)/nu(ref)  ={self.anuisp[self.ispres]:13.4E}")
                print(f"{' '*20}tau(i) ={self.tauisp[self.ispres]:13.4E} "
                    f"vth(i)/vth(ir) ={self.vthisp[self.ispres]:13.4E}")
                
                print(f"{' '*10}Electrons:")
                print(f"{' '*20}nu(i)  ={self.anucol[self.nspele]:13.4E} "
                    f"nu(i)/nu(ref)  ={self.anuisp[self.nspele]:13.4E}")
                print(f"{' '*20}tau(i) ={self.tauisp[self.nspele]:13.4E} "
                    f"vth(i)/vth(ir) ={self.vthisp[self.nspele]:13.4E}")
                
                if self.ibreak != 0:
                    xiumax = self.xiperp * self.umax
                    print(f"\n{' '*10}Nfs**2                   ={self.qnfsq:13.2f}")
                    print(f"{' '*10}最大能量显示              ={self.emax:13.3f} keV")
                    print(f"{' '*10}xiperp                   ={self.xiperp:13.4E}")
                    print(f"{' '*10}upper limit on xiperp*u  ={xiumax:13.4E}")
                    print(f"{' '*10}Dqlw                     ={self.dqlw:13.4E}")

        if self.joutql >= 3 and ipsi == self.ipsplf[self.ispres, self.ipheat-1]:
            # 详细测试输出。为避免过多输出，仅在功率密度最大的点进行
            print(f"\n{' '*10}Collision coefficients at psi ={psi:9.2f}")
            for ip in range(0, self.nptu, self.jmptu):
                print(f" u ={self.tbu[ip]:7.3f} Psi_c ={self.tbpsic[ip]:13.4E} "
                    f"Psi_e ={self.tbpsie[ip]:13.4E} "
                    f"Psi_tau ={self.tbpsit[ip]:13.4E} "
                    f"Theta ={self.tbthec[ip]:13.4E}")

            print(f"\n{' '*10}Quasilinear radial diffusion projection D00[0,0]")
            for ip in range(0, self.nptu, self.jmptu):
                self.qlcoef(self.tbu[ip])
                print(f" u ={self.tbu[ip]:7.3f} D00[0,0] ={self.dql00[0, 0]:13.4E}")


    def outdis(self, ipsi):
        """传输到输出表格
        
        Args:
            ipsi: 空间网格点索引
        """
        # 传输到输出表格
        self.thrate[self.ispres, ipsi-1] = self.hrate
        self.thtprp[self.ispres, ipsi-1] = self.htperp
        self.thtpar[self.ispres, ipsi-1] = self.htpar
        self.thttot[self.ispres, ipsi-1] = self.htemp
        self.tavgtx[self.ispres, ipsi-1] = self.ttailx
        self.tavgtz[self.ispres, ipsi-1] = self.ttailz
        self.tavbkx[self.ispres, ipsi-1] = self.ttblkx
        self.tavbkz[self.ispres, ipsi-1] = self.ttblkz

        # 测试源项
        if abs(self.zdotn) > 1.0E-6:
            print(f"\n 警告：对于物种{self.ispres:3d}，在psi ={self.psi:8.3f}处")
            print(f" 源项仅保持粒子守恒精度为{self.zdotn:13.4E}")

        if self.joutql >= 2:
            print(f"\n{' '*8}物种 {self.ispres:3d} 积分源项 ={self.zdotn:15.5E}")
            print(f"{' '*8}总估计损失（麦克斯韦） ={self.pwcoll[1]:15.5E}")
            print(f"{' '*8}来自临时源的加热率 ={self.pwsrce:15.5E}")

        # 检查收敛是否足够（避免在极低加热率时的不准确性）
        zt1 = (2.0 * self.htperp + self.htpar) / (3.0 * self.ctx.tempi[self.ispres])
        zt2 = self.ttailx / self.htperp
        zt0 = min(zt1, zt2)

        # 尾部快离子比例（估计）
        if self.ipheat == 1:
            if zt0 > 1.001:
                zft = (2.0 * self.ttailx + self.ttailz) / (3.0 * self.htemp)
                zfb = (2.0 * self.ttblkx + self.ttblkz) / (3.0 * self.htemp)
                self.ficonc = max(0.0, (1.0 - zfb) / (zft - zfb))
            else:
                self.ficonc = 0.0
            self.tmtail[self.ispres, ipsi-1] = self.ficonc

        # 详细输出
        append_to_file(self.output_paths["read"], f"详细输出\n")
        psi = self.redpsi[ipsi-1]

        # 在最大功率密度点绘制常数mu处的分布函数
        if ipsi == self.ipsplf[self.ispres, self.ipheat-1]:
            if self.ipheat == 1:
                self.bimaxw()

            for ip in range(self.nptu):
                if self.tfecmu[ip, 0] > 0.0:
                    self.tbpl[ip] = np.log10(self.tfecmu[ip, 0])
                else:
                    self.tbpl[ip] = -999.0

            # IDL输出
            if self.idlout != 0:
                # print("分布函数网格")
                # print(f"{self.nptu:5d}{self.nmu:5d}")
                # print(" ".join(f"{x:13.4E}" for x in self.tbkev))

                # if self.ipheat == 1:
                #     print("F(E), 基频")
                # else:
                #     print("F(E), 谐波")
                # print(" ".join(f"{x:13.4E}" for x in self.tbpl))

                for imu in range(1, self.nmu):
                    # 防止可能的收敛失败
                    ymin = self.tbpl[self.nptu-1] - 2.0
                    fmin = 10.0 ** ymin
                    mptp = 0
                    for ip in range(self.nptu):
                        fe = self.tfecmu[ip, imu]
                        if fe > 0.0 and fe >= fmin:
                            self.tbpl[ip] = np.log10(fe)
                            mptp = ip
                        else:
                            self.tbpl[ip] = -999.0
                        

                    # if mptp >= 5:
                    #     print(f"{mptp:5d}")
                    #     print(" ".join(f"{x:13.4E}" for x in self.tbpl))

                # 参考麦克斯韦分布
                print("参考麦克斯韦分布")
                zn = np.log10(5.56832799683)
                zm = 0.4342944819 / self.ctx.tempi[self.ispres]
                mptp = 0
                # for ip in range(self.nptu):
                #     fe = -(zm * self.tbkev[ip] + zn)
                #     if fe < ymin:
                #         self.tbpl[ip] = np.nan
                #         break
                #     self.tbpl[ip] = fe
                #     mptp = ip
                # self.ref_maxwell = self.tbpl
                for ip in range(self.nptu):
                    fe = -(zm * self.tbkev[ip] + zn)
                    if fe < ymin:
                        self.ref_maxwell[ip] = np.nan
                        break
                    self.ref_maxwell[ip] = fe
                    mptp = ip
                

                # if mptp >= 5:
                #     print(f"{mptp:5d}")
                #     print(" ".join(f"{x:13.4E}" for x in self.tbpl[:mptp]))

        # 输出部分
        if psi < self.psinxt:
            return

        self.psinxt = self.psinxt + self.dpsxnt
        if self.hrate == 0.0:
            print("\n ICRH加热率在psi =", psi, "处为零")
        # if self.ibreak != 0:
        #     return

        
        append_to_file(self.output_paths["read"], f"\n{' '*8}在psi ={psi:12.3f}处的稳态结果\n")
        append_to_file(self.output_paths["read"], f"{' '*10}ICRH输入功率      {self.pwdens:13.6E} W/cm3\n")
        append_to_file(self.output_paths["read"], f"{' '*10}准线性加热率      {self.hrate:13.6E} W/cm3\n")
        append_to_file(self.output_paths["read"], f"{' '*10}碰撞损失           {self.pwcoll[1]:13.5E} W/cm3\n")
        append_to_file(self.output_paths["read"], f"{' '*10}          （在麦克斯韦极限下  {self.pwsrce:13.5E} W/cm3）\n")
        append_to_file(self.output_paths["read"], f"{' '*10}NBI束系数标定功率   {self.pwnbi:13.4E} W/cm3\n")
        append_to_file(self.output_paths["read"], f"{' '*10}NBI热粒子替换汇功率 {self.pwnbi_l:13.4E} W/cm3\n")
        append_to_file(self.output_paths["read"], f"{' '*10}部分残差（未含NBI源和边界通量） {self.zbalpw:13.4E} W/cm3\n")

        append_to_file(self.output_paths["read"], f"{self.pwexch[1, :]}\n")
        append_to_file(self.output_paths["read"], "\n")
        for isp in range(self.nspele):
            if abs(self.ctx.tempi[isp] - self.ctx.tempi[self.ispres]) == 0.0:
                if abs(self.pwcoll[1]) > 1.0e-20:
                    frhtel = 100.0 * self.pwexch[1, isp] / self.pwcoll[1]
                else:
                    frhtel = 0.0  # 如果没有碰撞损失，比例设为0
                # frhtel = 100.0 * self.pwexch[1, isp] / self.pwcoll[1]
                append_to_file(self.output_paths["read"], f"{' '*10}在物种{isp:3d}上的损失 :{self.pwexch[1, isp]:13.4E} W/cm3 (= {frhtel:10.2f} %)\n")
                
            else:
                append_to_file(self.output_paths["read"], f"{' '*10}在物种{isp:3d}上的损失 :{self.pwexch[1, isp]:13.4E} W/cm3\n")
                append_to_file(self.output_paths["read"], f"{' '*15}没有高频（麦克斯韦）     {self.pwexch[0, isp]:13.4E} W/cm3\n")
                difference = self.pwexch[1, isp] - self.pwexch[0, isp]
                if self.pwdens != 0.0:
                    fraction = f"相对ICRH参考输入 {100.0 * difference / self.pwdens:.2f} %"
                else:
                    fraction = "ICRH参考输入为零，不计算比例"
                append_to_file(self.output_paths["read"], f"{' '*15}相对麦克斯韦的差值 {difference:13.4E} W/cm3 ({fraction})\n")

        ze = 2.0 * self.emax / self.ttailx
        zcut = 0.5 * (ze + 1.0) * np.exp(-ze)
        append_to_file(self.output_paths["read"], f"\n{' '*10}二阶矩等效温度：总 {self.htemp:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}    垂直          {self.htperp:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}    平行          {self.htpar:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}垂直体温度  {self.ttblkx:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}平行体温度  {self.ttblkz:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}垂直尾部温度  {self.ttailx:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}平行尾部温度  {self.ttailz:13.4E} keV\n")
        append_to_file(self.output_paths["read"], f"{' '*10}估计截断误差     {zcut:13.4E}\n")

        if self.ipheat == 1:
            # Stix分布中的快离子比例
            zavgtz = (1.0 - self.ficonc) * self.ttblkz + self.ficonc * self.ttailz
            print(f"\n{' '*10}快离子比例     {self.ficonc:13.3f}")
            print(f"{' '*10}估计平行温度    {zavgtz:13.3f} keV")

        if self.joutql >= 3 and ipsi == self.ipsplf[self.ispres, self.ipheat-1]:
            # 详细测试。为避免过多输出，仅在最大功率密度点进行
            print(f"\n{' '*10}常数mu处的分布函数")
            for ip in range(0, self.nptu, self.jmptu):
                u = self.tbu[ip]
                e = self.tbkev[ip]
                # print(f" E ={e:7.3f} u ={u:7.3f} f =",
                #     " ".join(f"{x:11.3E}" for x in self.tfecmu[ip, :self.nmu]))

            n0 = 0
            n1 = min(self.legmod, 3)
            print(f"\n{' '*10}在psi ={psi:8.3f}处的勒让德展开系数")
            print(f"{' '*20}（znorm ={self.fnorm:13.4E}）")
            print(f"{' '*20}阶数{n0:3d}到{n1:3d}的系数")

            while True:
                for ip in range(self.nptu):
                    print(f" u ={self.tbu[ip]:7.3f} F_0 ... =",
                        " ".join(f"{x:13.4E}" for x in self.tbfu[ip, n0:n1+1]))
                if n1 >= self.legmod:
                    break
                n0 = n1 + 1
                n1 = n0 + 4
                print(f"\n{' '*20}阶数{n0:3d}到{n1:3d}的系数")

        append_to_file(self.output_paths["case"], 
            f"{' '*2}{psi:.15f}"
            f"{' '*7}{np.sum(self.pwexch[1, :self.nspele - 1]):.15E}"
            f"{' '*7}{self.pwexch[1, self.nspele - 1]:.15E}"

            f"{' '*7}{self.htemp:.15E}"
            f"{' '*7}{self.htperp:.15E}"
            f"{' '*7}{self.htpar:.15E}"

            f"{' '*7}{self.ttailx:.15E}{' '*5}\n"
        )

        

    def bimaxw(self):
        """计算分布函数"""
        self.zgsqx = np.zeros(2)
        self.zgsqz = np.zeros(2)
        self.znorm = np.zeros(2)
        self.pithf = 5.568327996
        
        # 计算各向同性部分的参数
        self.zgsqx[0] = self.ttblkx / self.ctx.tempi[self.ispres]
        self.zgsqz[0] = self.ttblkz / self.ctx.tempi[self.ispres]
        self.znorm[0] = (1.0 - self.ficonc) / (self.pithf * self.zgsqx[0] * np.sqrt(self.zgsqz[0]))
        self.zgsqx[1] = self.ttailx / self.ctx.tempi[self.ispres]
        self.zgsqz[1] = self.ttailz / self.ctx.tempi[self.ispres]
        self.znorm[1] = self.ficonc / (self.pithf * self.zgsqx[1] * np.sqrt(self.zgsqz[1]))

        # 计算mu方向的分布
        dxmu = 1.0 / self.nmu
        for imu in range(self.nmu):
            xmu = imu+1 * dxmu  # Python索引从0开始
            xmusq = xmu * xmu
            
            for ip in range(self.nptu):
                u = self.tbu[ip]
                uu = u * u
                self.fbmaxw[ip, imu] = 0.0
                
                for ik in range(2):  # 对应ik=1,2
                    zarg = (xmusq / self.zgsqz[ik] + 
                        (1.0 - xmusq) / self.zgsqx[ik]) * uu
                    self.fbmaxw[ip, imu] += self.znorm[ik] * np.exp(-zarg)
            
            # 详细输出
            if self.joutql >= 3:
                print("\n 3DFPQL和bimaxw在mu =", xmu, "处的分布")
                for ip in range(self.nptu):
                    zarg = self.tfecmu[ip, imu] / self.fbmaxw[ip, imu]
                    # print(f" u = {self.tbu[ip]:8.3f} E = {self.tbkev[ip]:8.3f} "
                    #     f"f_ql = {self.tfecmu[ip, imu]:12.4e} "
                    #     f"f_bmx = {self.fbmaxw[ip, imu]:12.4e}")

    def rjbess(self, z):
        """Populate the legacy Bessel table with the stable library implementation."""
        self.njbess = max(int(self.njbess), 2)
        required = max(self.njbess + 2, self.ipheat + self.ksup + 1)
        if self.fjbess is None or len(self.fjbess) < required:
            self.fjbess = np.zeros(required)
        self.fjbess[:] = special.jv(np.arange(len(self.fjbess)), z)

    def zjbsum(self, zhsq, n):
        """计算贝塞尔函数的级数和
        
        Args:
            zhsq: 输入参数
            n: 级数项数
        
        Returns:
            zjbsum: 级数和
        """
        self.zepsil = 1.0E-9
        zjbsum = 1.0
        zterm = 1.0
        m = 0

        while True:
            m += 1
            zterm = -zterm * zhsq / (m * (m + n))
            zjbsum += zterm
            
            if abs(zterm / zjbsum) <= self.zepsil:
                break
        
        return zjbsum
    
    def poledg(self, arg, nleg):
        """Calculate poloidal edge quantities."""
        self.ierr = 0
        
        # 检查参数范围
        if abs(arg) > 1.0:
            if nleg * np.log10(abs(arg)) > 30.0:
                self.ierr = 1
                return self.ierr
        
        # 初始化结果数组
        self.polg = np.zeros(nleg + 1)  # Python索引从0开始，所以需要nleg+1
        
        # # 计算勒让德多项式
        # self.polg[0] = arg  # 对应Fortran的polg(1)
        # if nleg <= 1:
        #     return self.ierr
        
        # self.polg[1] = 1.5 * arg * arg - 0.5  # 对应Fortran的polg(2)
        # if nleg == 2:
        #     return self.ierr
        
        # # 递推计算更高阶的多项式
        # for n in range(2, nleg ):  # Python中从2开始，对应Fortran的n=3
        #     self.polg[n] = 2.0 * arg * self.polg[n-1] - self.polg[n-2] - \
        #                 (arg * self.polg[n-1] - self.polg[n-2]) / n
        
        # self.plg = self.polg

        self.plg = [legendre(i)(arg) for i in range(1, nleg+1)]

        return
    
    def qlmomu(self, tbf,du,ndu,ipw):
        """
        辛普森积分法计算积分：∫v^(ipw+2) f(u) du
        使用Simpson方法,高效的数值积分计算, 特别适合计算带有幂函数权重的积分
        输入：tbf: 存储函数 f(u) 值的数组
            du: 积分步长
            ndu: 积分区间分割的段数
            ipw: 幂函数的指数偏移量
        返回积分值：qlmomu
        """
        # =====================================================================
        #  Program FP
        #  Evaluates: integral u**(ipw+2) f(u) du, where
        #      f(u) is stored in the array TBF
        #  with Simpson method.
        # =====================================================================
        u = np.arange(ndu, dtype=float) * du
        return float(simpson(np.asarray(tbf)[:ndu] * u**(ipw + 2), x=u))

    def exchcl(self, isp):
        """Collisional energy transfer from the assembled Hermite weak form."""
        integral = 0.0
        for element in range(self.nfelem):
            for z, weight in zip(self.zgaus, self.wgaus):
                u = self.tbu[element] + self.du * z
                value = (
                    self.herm1(z) * self.tbfu[element, 0]
                    + self.du * self.herm2(z) * self.tbdfu[element, 0]
                    + self.herm1(z - 1) * self.tbfu[element + 1, 0]
                    + self.du * self.herm2(z - 1) * self.tbdfu[element + 1, 0]
                )
                derivative = (
                    self.dherm1(z) / self.du * self.tbfu[element, 0]
                    + self.dherm2(z) * self.tbdfu[element, 0]
                    + self.dherm1(z - 1) / self.du * self.tbfu[element + 1, 0]
                    + self.dherm2(z - 1) * self.tbdfu[element + 1, 0]
                )
                self.fokker(u)
                if isp >= 0:
                    diffusion = self.anuisp[isp] * self.phifun[isp]
                    drag = self.tauisp[isp] * diffusion
                else:
                    diffusion = self.psicol
                    drag = self.psitau
                integral += self.du * weight * u**2 * (
                    derivative * diffusion + 2 * u * value * drag
                )
        return 4 * np.pi * integral


    def t4_surfpl(self,psi,ibmod):
        """
        读取径向位置
        """
        # 重置迭代状态变量
        self.fb1_prv = 0.0  # 上一次迭代的 f(0) 值，必须重置否则会继承上一个半径的边界条件
        self.ibreak = 0     # 中断标志，必须清零

        pass

        return
    
    # ===================================================================
    # 辅助函数，生成高斯分布
    # ===================================================================
    def _generate_test_gaussian_nbi(self):
        """生成测试用的单向高斯 NBI 注入分布"""
        self.nbi_in = np.ones((self.ctx.nspec, self.Ns, self.nmu, self.nrdpsi))
        self.nbi_gaussian_beam_weights = np.ones(self.Ns)
        
        # 【修改】：使用主程序传入的参数
        mu_center = self.nbi_mu_center
        sigma = self.nbi_sigma
        
        mu_values = np.linspace(self.mu_min, self.mu_max, self.nmu)
        gaussian = np.exp(-((mu_values - mu_center) ** 2) / (2 * sigma ** 2))
        # gaussian[mu_values < 0] = 0.0  # 强制单向注入，只保留正mu值
        self.nbi_in = self.nbi_in * gaussian[np.newaxis, np.newaxis, :, np.newaxis]
        print(f"已生成测试高斯 NBI 分布 (中心={mu_center}, 宽度={sigma})。")
        
        # 绘制高斯分布gaussian
        aa = 0  # 画图开关
        if aa:
            plt.figure(figsize=(12, 8))
            # 1. 基本线图
            plt.subplot(2, 2, 1)
            plt.plot(mu_values, gaussian, 'b-', linewidth=2)
            plt.title('高斯分布曲线')
            plt.xlabel('索引')
            plt.ylabel('值')
            plt.grid(True, alpha=0.3)
            # 2. 填充区域图
            plt.subplot(2, 2, 2)
            plt.fill_between(mu_values, gaussian, alpha=0.5, color='green')
            plt.title('高斯分布填充图')
            plt.xlabel('索引')
            plt.ylabel('值')
            plt.grid(True, alpha=0.3)

            # 3. 柱状图
            plt.subplot(2, 2, 3)
            plt.bar(mu_values, gaussian, width=0.8, alpha=0.7, color='red')
            plt.title('高斯分布柱状图')
            plt.xlabel('索引')
            plt.ylabel('值')
            plt.grid(True, alpha=0.3)

            # 4. 对数尺度图
            plt.subplot(2, 2, 4)
            plt.semilogy(mu_values, gaussian, 'purple', linewidth=2)
            plt.title('高斯分布（对数尺度）')
            plt.xlabel('索引')
            plt.ylabel('值（对数）')
            plt.grid(True, alpha=0.3)

            # 调整子图间距
            plt.tight_layout()

            # 显示图形
            plt.show()

    # ===================================================================
    # 统一绘图格式配置
    # ===================================================================
    def get_origin_style_config(self):
        """
        获取 Origin 风格的统一绘图配置
        返回配置参数字典
        """
        config = {
            # 页面大小（英寸）
            'PAGE_WIDTH': 6,
            'PAGE_HEIGHT': 6,
            
            # 图层位置和大小（英寸）
            'LAYER_LEFT': 1,
            'LAYER_TOP': 1,
            'LAYER_WIDTH': 3,
            'LAYER_HEIGHT': 3,
            
            # 字体设置
            'FONT_FAMILY': 'serif',
            'FONT_SERIF': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
            'MATH_FONTSET': 'stix',
            
            # 字体大小
            'FONT_SIZE': 16,
            'LABEL_SIZE': 16,
            'TICK_SIZE': 12,
            'COLORBAR_SIZE': 12,
            'LEGEND_SIZE': 10,
            
            # 字体粗细
            'COLORBAR_LABEL_WEIGHT': 'bold',
            'LABEL_WEIGHT': 'bold',
            'TICK_WEIGHT': 'bold',
            'LEGEND_WEIGHT': 'bold',
            
            # 线宽
            'AXIS_LINE_WIDTH': 1.2,
            'GRID_LINEWIDTH': 1,
            'GRID_ALPHA': 0.5,
            
            # 分辨率
            'DPI': 300,
        }
        return config


    def apply_origin_font_settings(self, config):
        """
        应用 Origin 风格的全局字体设置
        """
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            'font.family': config['FONT_FAMILY'],
            'font.serif': config['FONT_SERIF'],
            'mathtext.fontset': config['MATH_FONTSET'],
            'font.size': config['FONT_SIZE'],
            'axes.linewidth': config['AXIS_LINE_WIDTH'],
        })


    def create_origin_axes(self, fig, config, subplot_idx=0, n_subplots=1):
        """
        创建 Origin 风格的坐标轴
        支持单图和双图布局
        """
        if n_subplots == 2:
            # 双图模式
            layer_width_each = (config['LAYER_WIDTH'] - 0.3) / 2
            if subplot_idx == 0:
                left = config['LAYER_LEFT']
            else:
                left = config['LAYER_LEFT'] + layer_width_each + 0.3
            
            ax = fig.add_axes([
                left / config['PAGE_WIDTH'],
                1 - (config['LAYER_TOP'] + config['LAYER_HEIGHT']) / config['PAGE_HEIGHT'],
                layer_width_each / config['PAGE_WIDTH'],
                config['LAYER_HEIGHT'] / config['PAGE_HEIGHT']
            ])
        else:
            # 单图模式
            ax = fig.add_axes([
                config['LAYER_LEFT'] / config['PAGE_WIDTH'],
                1 - (config['LAYER_TOP'] + config['LAYER_HEIGHT']) / config['PAGE_HEIGHT'],
                config['LAYER_WIDTH'] / config['PAGE_WIDTH'],
                config['LAYER_HEIGHT'] / config['PAGE_HEIGHT']
            ])
        
        return ax


    def create_origin_colorbar(self, fig, ax, im, config, is_double=False):
        """
        创建 Origin 风格的色标
        """
        font_height_inch = config['COLORBAR_SIZE'] / 72
        cbar_width_inch = font_height_inch * 0.9
        label_offset_inch = font_height_inch * 0.3
        
        if is_double:
            # 双图模式：色标放在子图右侧
            ax_pos = ax.get_position()
            cbar_x = ax_pos.x1 + 0.02
            cbar_y = ax_pos.y0
            cbar_height = ax_pos.height
            cbar_width = cbar_width_inch / config['PAGE_WIDTH']
        else:
            # 单图模式
            cbar_x = (config['LAYER_LEFT'] + config['LAYER_WIDTH'] + label_offset_inch) / config['PAGE_WIDTH']
            cbar_y = 1 - (config['LAYER_TOP'] + config['LAYER_HEIGHT']) / config['PAGE_HEIGHT']
            cbar_height = config['LAYER_HEIGHT'] / config['PAGE_HEIGHT']
            cbar_width = cbar_width_inch / config['PAGE_WIDTH']
        
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        
        # 设置色标标签和刻度
        cbar.set_label('logf', fontsize=config['COLORBAR_SIZE'], 
                    fontweight=config['COLORBAR_LABEL_WEIGHT'])
        cbar.ax.tick_params(labelsize=config['COLORBAR_SIZE'], 
                            width=config['AXIS_LINE_WIDTH'])
        
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight(config['TICK_WEIGHT'])
            label.set_fontsize(config['COLORBAR_SIZE'])
        
        cbar.outline.set_linewidth(config['AXIS_LINE_WIDTH'])
        
        return cbar


    def setup_origin_axes(self, ax, config, xlabel, ylabel, title=None):
        """
        设置 Origin 风格的坐标轴标签、刻度和边框
        """
        # 坐标轴标签
        ax.set_xlabel(xlabel, fontsize=config['LABEL_SIZE'], 
                    fontweight=config['LABEL_WEIGHT'])
        ax.set_ylabel(ylabel, fontsize=config['LABEL_SIZE'], 
                    fontweight=config['LABEL_WEIGHT'])
        
        # 标题（可选）
        if title:
            ax.set_title(title, fontsize=config['LABEL_SIZE'], 
                        fontweight=config['LABEL_WEIGHT'])
        
        # 刻度设置
        ax.tick_params(axis='both', which='major',
                    labelsize=config['TICK_SIZE'],
                    width=config['AXIS_LINE_WIDTH'],
                    length=8)
        
        # 刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight(config['TICK_WEIGHT'])
            label.set_fontsize(config['TICK_SIZE'])
        
        # 轴线条宽
        for spine in ax.spines.values():
            spine.set_linewidth(config['AXIS_LINE_WIDTH'])
            spine.set_edgecolor('black')
        
        # 网格线
        # ax.grid(True, linestyle='--', alpha=config['GRID_ALPHA'], 
        #         linewidth=config['GRID_LINEWIDTH'], color='gray')


    # ===================================================================
    # 修改后的三个绘图函数
    # ===================================================================
    def plot_f_heatmap(self, ctx, ipsi, plot_range='both', 
                    nbi_tail_E=None, nbi_tail_mu=None, nbi_char_mu=None, nbi_char_E=None, 
                    icrh_tail_E=None, icrh_tail_mu=None, icrh_char_mu=None, icrh_char_E=None,
                    co_tail_E=None, co_tail_mu=None, co_char_mu=None, co_char_E=None):
        """
        绘制粒子速度分布函数的热图 - Origin格式
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        import os
        
        # 验证输入
        if not self:
            print("警告：没有提供输入数据")
            return
        if not plot_range in ['limited', 'full', 'both']:
            plot_range = 'both'
        
        # 获取统一配置
        config = self.get_origin_style_config()
        self.apply_origin_font_settings(config)
        
        # 计算能量
        Es = self.tbu**2 * ctx.tempi[self.ispres]
        
        f = self.tfecmu.copy()
        min_cutoff_ip = 1
        f[:min_cutoff_ip, :] = np.nan
        fmax = self.f_max
        f[f > fmax] = np.nan
        fmin = self.f_min
        f[f < fmin] = np.nan
        
        # 转换为 log 显示
        f[f <= 0] = np.nan
        f_log = np.log10(f)
        
        # 创建颜色映射
        colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0)] 
        cm = LinearSegmentedColormap.from_list('wy_rk', colors, N=100)
        
        # 创建图形
        fig = plt.figure(figsize=(config['PAGE_WIDTH'], config['PAGE_HEIGHT']))
        
        # 创建图层
        n_subplots = 2 if plot_range == 'both' else 1
        if plot_range == 'both':
            axes = [self.create_origin_axes(fig, config, 0, 2),
                    self.create_origin_axes(fig, config, 1, 2)]
        else:
            axes = [self.create_origin_axes(fig, config, 0, 1)]
        
        x_lim_max1 = None
        
        for idx, (ax, range_type) in enumerate(zip(axes, ['limited', 'full'] if plot_range == 'both' else [plot_range])):
            # 设置坐标轴范围
            if range_type == 'limited':
                x_lim_max = min(self.xlim_max, Es.max())
                ax.set_xlim(Es[min_cutoff_ip], x_lim_max)
                x_lim_max1 = x_lim_max
            else:
                x_lim_max1 = Es.max()
                ax.set_xlim(Es[min_cutoff_ip], x_lim_max1)
            
            ax.set_ylim(self.mu_min, self.mu_max)
            
            # 数据范围
            valid_data = f_log[~np.isnan(f_log)]
            vmin = np.min(valid_data) if len(valid_data) > 0 else -10
            vmax = np.max(valid_data) if len(valid_data) > 0 else 0
            
            # 绘制热图
            im = ax.pcolormesh(Es, self.xmu, f_log.T, cmap=cm, shading='gouraud', 
                            norm=Normalize(vmin=vmin, vmax=vmax))
            
            # # ============ 绘制高能尾轮廓标记线 ============
            # if nbi_tail_E is not None and len(nbi_tail_E) > 0:
            #     ax.plot(nbi_tail_E, nbi_tail_mu, 'k--', linewidth=2.5, label='NBI tail', zorder=5)

            # if icrh_tail_E is not None and len(icrh_tail_E) > 0:
            #     ax.plot(icrh_tail_E, icrh_tail_mu, color='gray', linestyle='--', linewidth=2.5, label='ICRH tail', zorder=5)

            # if co_tail_E is not None and len(co_tail_E) > 0:
            #     ax.plot(co_tail_E, co_tail_mu, 'b--', linewidth=2.5, label='Synergy tail', zorder=5)


            # # ============ 绘制特征 pitch angle 横线 ============
            # if nbi_char_mu is not None:
            #     ax.axhline(y=nbi_char_mu, color='black', linestyle='-', linewidth=2,
            #             label=f'NBI $\\mu$={nbi_char_mu:.2f}', zorder=4)

            # if icrh_char_mu is not None:
            #     ax.axhline(y=icrh_char_mu, color='gray', linestyle='-', linewidth=2,
            #             label=f'ICRH $\\mu$={icrh_char_mu:.2f}', zorder=4)

            # if co_char_mu is not None:
            #     ax.axhline(y=co_char_mu, color='blue', linestyle='-', linewidth=2,
            #             label=f'Synergy $\\mu$={co_char_mu:.2f}', zorder=4)


            # # ============ 绘制特征能量竖线 ============
            # if nbi_char_E is not None:
            #     ax.axvline(x=nbi_char_E, color='black', linestyle='--', linewidth=2,
            #             label=f'NBI E={nbi_char_E:.1f}keV', zorder=4)

            # if icrh_char_E is not None:
            #     ax.axvline(x=icrh_char_E, color='gray', linestyle='--', linewidth=2,
            #             label=f'ICRH E={icrh_char_E:.1f}keV', zorder=4)

            # if co_char_E is not None:
            #     ax.axvline(x=co_char_E, color='blue', linestyle='--', linewidth=2,
            #             label=f'Synergy E={co_char_E:.1f}keV', zorder=4)
            
            # 创建色标
            cbar = self.create_origin_colorbar(fig, ax, im, config, is_double=(plot_range == 'both'))
            
            # 设置坐标轴
            self.setup_origin_axes(ax, config, 'E [keV]', r'$\mu$')
            
            # 图例设置
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                leg = ax.legend(handles, labels,
                            loc='upper right',
                            bbox_to_anchor=(0.98, 0.98),
                            borderaxespad=0,
                            framealpha=0.9,
                            fontsize=config['LEGEND_SIZE'],
                            borderpad=0.4,
                            labelspacing=0.3,
                            frameon=True,
                            facecolor='white',
                            edgecolor='black')
                
                for text in leg.get_texts():
                    text.set_fontname('Times New Roman')
                    text.set_fontsize(config['LEGEND_SIZE'])
                    text.set_fontweight(config['LEGEND_WEIGHT'])
                
                leg.get_frame().set_linewidth(1.5)
        
        # 保存图片
        saved_filename = None
        base_name = "result"
        if "name" in self.output_paths:
            base_name = os.path.basename(self.output_paths["name"])
        
        suffix = '_limited' if plot_range == 'limited' else '_full' if plot_range == 'full' else '_both'
        bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}'
        
        filename = f'{base_name}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_E_{x_lim_max1}_dv_{self.du}_{bh}{suffix}_heatmap.png'
        
        save_dir = self.current_out_dir if hasattr(self, 'current_out_dir') and self.current_out_dir else "."
        full_path = os.path.join(save_dir, filename)
            
        try:
            plt.savefig(full_path, dpi=config['DPI'], bbox_inches='tight')
            print(f"图形已保存为: {full_path}")
            saved_filename = full_path
        except Exception as e:
            print(f"保存图片失败: {e}")

        return fig, saved_filename


    def plot_f_velocity_para_perp(self, ipsi, plot_range='full'):
        """
        绘制 f(v_perp, v_parallel) 速度空间图 - Origin格式
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        from matplotlib.ticker import MaxNLocator
        
        # 获取统一配置
        config = self.get_origin_style_config()
        self.apply_origin_font_settings(config)
        
        # 计算速度网格
        v = self.tbu
        
        # 转换为 (v_par, v_perp) 网格
        V, MU = np.meshgrid(v, self.xmu, indexing='ij')
        V_par = V * MU
        V_perp = V * np.sqrt(1 - MU**2)
        
        # 处理分布函数数据
        f = self.tfecmu.copy()
        fmin = self.f_min
        f[f < fmin] = np.nan
        f_log = np.log10(f)
        
        # 计算有效数据范围
        valid_mask = ~np.isnan(f)
        if np.any(valid_mask):
            max_v_par = np.max(np.abs(V_par[valid_mask]))
            max_v_perp = np.max(V_perp[valid_mask])
        else:
            max_v_par = np.max(np.abs(V_par))
            max_v_perp = np.max(V_perp)
        
        # 创建图形
        fig = plt.figure(figsize=(config['PAGE_WIDTH'], config['PAGE_HEIGHT']))
        ax = self.create_origin_axes(fig, config, 0, 1)
        
        # 绘制热图
        pcm = ax.pcolormesh(V_par, V_perp, f_log, cmap='hot_r', shading='gouraud')
        
        # 创建色标
        cbar = self.create_origin_colorbar(fig, ax, pcm, config, is_double=False)
        
        # 设置坐标轴
        self.setup_origin_axes(ax, config, r'$v_{//}$', r'$v_{\perp}$')
        
        ax.set_aspect('equal')
        
        # 计算坐标轴范围
        if plot_range == 'limited':
            max_v_par = min(5.0, max_v_par)
            max_v_perp = min(5.0, max_v_perp)
        
        x_span = 2 * max_v_par
        y_span = max_v_perp
        max_span = max(x_span, y_span)
        max_span = int(np.ceil(max_span))
        
        ax.set_xlim(-max_span/2, max_span/2)
        ax.set_ylim(0, max_span)
        
        # 强制整数刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 保存图片
        saved_filename = None
        base_name = "result"
        if "name" in self.output_paths:
            base_name = os.path.basename(self.output_paths["name"])
        
        suffix = '_limited' if plot_range == 'limited' else '_full' if plot_range == 'full' else '_both'
        bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}'
        
        filename = f'{base_name}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_dv_{self.du}_{bh}{suffix}_vpar_vperp_.png'
        
        save_dir = self.current_out_dir if hasattr(self, 'current_out_dir') and self.current_out_dir else "."
        full_path = os.path.join(save_dir, filename)
            
        try:
            plt.savefig(full_path, dpi=config['DPI'], bbox_inches='tight')
            print(f"图形已保存为: {full_path}")
            saved_filename = full_path
        except Exception as e:
            print(f"保存图片失败: {e}")

        return fig, saved_filename


    def plot_f_velocity_pitch_angle(self, ctx, ipsi, plot_range='both', save_path=None):
        """
        绘制粒子速度分布函数 - Origin格式
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        
        # 验证输入
        if not self:
            print("警告：没有提供输入数据")
            return
        if not plot_range in ['limited', 'full', 'both']:
            plot_range = 'both'
        
        # 获取统一配置
        config = self.get_origin_style_config()
        self.apply_origin_font_settings(config)
        
        # 计算能量
        Es = self.tbu**2 * ctx.tempi[self.ispres]

        f = self.tfecmu.copy()

        fmax = self.f_max
        fmin = self.f_min
        f[f > fmax] = np.nan
        f[f < fmin] = np.nan
        f = np.log10(f)
        
        # 创建图形
        fig = plt.figure(figsize=(config['PAGE_WIDTH'], config['PAGE_HEIGHT']))
        
        # 创建图层
        n_subplots = 2 if plot_range == 'both' else 1
        if plot_range == 'both':
            axes = [self.create_origin_axes(fig, config, 0, 2),
                    self.create_origin_axes(fig, config, 1, 2)]
        else:
            axes = [self.create_origin_axes(fig, config, 0, 1)]
        
        # 获取要绘制的 mu 索引
        plot_indices = []
        if self.custom_mu_values is not None:
            target_mus = np.array(self.custom_mu_values)
            for target_mu in target_mus:
                closest_idx = np.argmin(np.abs(self.xmu - target_mu))
                if closest_idx not in plot_indices:
                    plot_indices.append(closest_idx)
        else:
            duu = self.duu
            nduu = (self.tfecmu.shape[1]-1)*duu
            for ii in range(self.tfecmu.shape[1]):
                if ii % nduu != 0:
                    continue
                plot_indices.append(ii)
        
        # 计算全局 y 轴范围
        all_y_values = []
        for ii in plot_indices:
            mu_val = np.round(self.xmu[ii], 2)
            if self.plot_range_mu == 1 and mu_val < 0: continue
            elif self.plot_range_mu == -1 and mu_val > 0: continue
            y_values = f[:, ii]
            valid_values = y_values[~np.isnan(y_values)]
            all_y_values.extend(valid_values)
        
        if all_y_values:
            global_y_min = min(all_y_values)
        else:
            global_y_min = f[1,0]-5 if not np.isnan(f[1,0]) else -10
        
        x_lim_max = Es.max()
        
        for ax, range_type in zip(axes, ['limited', 'full'] if plot_range == 'both' else [plot_range]):
            # 绘制不同 pitch 角的分布
            for ii in plot_indices:
                mu_val = np.round(self.xmu[ii], 2)
                
                if self.plot_range_mu == 1 and mu_val < 0:
                    continue
                elif self.plot_range_mu == -1 and mu_val > 0:
                    continue
                elif self.plot_range_mu == 0 and mu_val < 0:
                    line_style = '--'
                else:
                    line_style = '-'
                
                y_values = f[:, ii]
                ax.plot(Es, y_values, label=fr'$\mu$ = {mu_val}', linestyle=line_style)
            
            # 绘制 Maxwell 分布
            ax.plot(Es, self.ref_maxwell, label='Maxwell.', linestyle='--')
            
            # 设置坐标轴范围
            if range_type == 'limited':
                x_lim_max = min(self.xlim_max, Es.max())
                ax.set_xlim(0, x_lim_max)
                idx_max = np.argmin(np.abs(Es - x_lim_max))
                y_values_at_max = []
                for ii in plot_indices:
                    mu_val = np.round(self.xmu[ii], 2)
                    if self.plot_range_mu == 1 and mu_val < 0: continue
                    elif self.plot_range_mu == -1 and mu_val > 0: continue
                    y_val = f[idx_max, ii]
                    if not np.isnan(y_val):
                        y_values_at_max.append(y_val)
                if y_values_at_max:
                    y_min = min(y_values_at_max)
                    ax.set_ylim((y_min - 0.5, 0))
                else:
                    ax.set_ylim((-5, 0))
                ax.set_ylim(self.ylim_max, 0)
            else:
                x_lim_max = Es.max()
                ax.set_xlim(0, x_lim_max)
                ax.set_ylim((global_y_min - 0.5, 0))
                # ax.set_title(f'r={self.redpsi[ipsi-1]}a P:{self.pwdens}W/cm^3 kperp:{self.kperp:.0f}')
            
            # 设置坐标轴
            self.setup_origin_axes(ax, config, 'E', r'$\log_{10}f$')
            
            # 图例设置
            legend = ax.legend(loc='upper right',
                            bbox_to_anchor=(0.98, 1),
                            framealpha=0.9,
                            ncol=2,
                            fontsize=config['LEGEND_SIZE'])
            legend.get_frame().set_linewidth(config['AXIS_LINE_WIDTH'])
            legend.get_frame().set_edgecolor('black')
            for text in legend.get_texts():
                text.set_fontname('Times New Roman')
                text.set_fontsize(config['LEGEND_SIZE'])
        
        # 保存图片
        saved_filename = None
        base_name = "result"
        if "name" in self.output_paths:
            base_name = os.path.basename(self.output_paths["name"])
        
        suffix = '_limited' if plot_range == 'limited' else '_full' if plot_range == 'full' else '_both'
        bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}'
        
        filename = f'{base_name}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_E_{x_lim_max}_dv_{self.du}_{bh}{suffix}_Fvu.png'
        
        save_dir = self.current_out_dir if hasattr(self, 'current_out_dir') and self.current_out_dir else "."
        full_path = os.path.join(save_dir, filename)
            
        try:
            plt.savefig(full_path, dpi=config['DPI'], bbox_inches='tight')
            print(f"图形已保存为: {full_path}")
            saved_filename = full_path
        except Exception as e:
            print(f"保存图片失败: {e}")

        return fig, saved_filename




    # # ===================================================================
    # # 绘图函数，绘制不同pitch angle下的粒子速度分布函数
    # # ===================================================================
    # def plot_f_velocity_pitch_angle(self, ctx, ipsi, plot_range='both', save_path=None):
    #     """
    #     绘制粒子速度分布函数
        
    #     参数:
    #         self: FP实例，包含计算结果
    #         plot_range: str, 可选 'limited', 'full', 或 'both'
    #         save_path: str, 可选，保存图片的路径
        
    #     返回:
    #         fig: matplotlib图形对象
    #     """


    #     # 验证输入
    #     if not self:
    #         print("警告：没有提供输入数据")
    #         return
    #     if not plot_range in ['limited', 'full', 'both']:
    #         print("警告：没有提供范围类型，默认绘制全范围")

    #     # 设置字体大小
    #     plt.rcParams.update({
    #         'font.size': 20,           # 设置全局字体大小
    #         'axes.titlesize': 20,      # 设置标题字体大小
    #         'axes.labelsize': 30,      # 设置坐标轴标签字体大小
    #         'xtick.labelsize': 20,     # 设置刻度字体大小
    #         'ytick.labelsize': 20,     # 设置刻度字体大小
    #         'legend.fontsize': 16,     # 设置图例字体大小
    #         'figure.titlesize': 16,    # 设置标题字体大小
    #         'axes.linewidth': 2,       # 设置坐标轴线宽
    #         'axes.edgecolor': 'black',  # 设置坐标轴颜色
    #         'xtick.major.width': 2,    # 设置x轴主刻度线宽度
    #         'ytick.major.width': 2,    # 设置y轴主刻度线宽度
    #         'xtick.minor.width': 1,    # 设置x轴次刻度线宽度
    #         'ytick.minor.width': 1,    # 设置y轴次刻度线宽度
    #         'xtick.major.size': 8,     # 设置x轴主刻度线长度
    #         'ytick.major.size': 8,     # 设置y轴主刻度线长度
    #         'xtick.minor.size': 4,     # 设置x轴次刻度线长度
    #         'ytick.minor.size': 4      # 设置y轴次刻度线长度
    #     })

    #     # 计算能量
    #     Es = self.tbu**2 * ctx.tempi[self.ispres]
    #     # Es = Es/ctx.tempi[0]
        
    #     fmax = self.f_max   # 最大阈值
    #     fmin = self.f_min   # 最小阈值
    #     # fmin = 10 ** (np.log10(self.tfecmu[-1,0])-3)   #计算一个最小阈值

    #     self.tfecmu[self.tfecmu>fmax] = np.nan          #将小于最小阈值的数置0，从而忽略较小的值  
    #     self.tfecmu[self.tfecmu<fmin] = np.nan          #将小于最小阈值的数置0，从而忽略较小的值

    #     f =  np.log10(self.tfecmu) # 对ssp对象取对数
    #     # f = f+0.3

    #     # # 设置pitch angle步长
    #     # duu = self.duu # pitch angle步长
    #     # nduu = (self.tfecmu.shape[1]-1)*duu # 系数

    #     # 创建图形
    #     if plot_range == 'both':
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    #         axes = [ax1, ax2]
    #     else:
    #         fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    #         axes = [ax]
        
    #     # 计算所有分布函数数据的最小值
    #     # all_y_values = []
    #     # for ii in range(self.tfecmu.shape[1]):
    #     #     if ii % nduu != 0:
    #     #     # if ii != 0 and ii != 13 and ii !=25 and ii != 50:
    #     #         continue
    #     #     y_values = f[:,ii]
    #     #     valid_values = y_values[~np.isnan(y_values)]
    #     #     all_y_values.extend(valid_values)
        
    #     # # 使用所有数据的最小值作为full图的y轴下限
    #     # if all_y_values:
    #     #     global_y_min = min(all_y_values)
    #     #     print(f"y轴下限 {global_y_min}" )
    #     # else:
    #     #     global_y_min = f[1,0]-5  # 如果没有有效值，使用默认值

    #     # ============ 自定义 mu 值的智能提取 ============
    #     # 确定需要绘制的 mu 索引列表
    #     plot_indices = []
    #     if self.custom_mu_values is not None:
    #         # 模式1：使用自定义指定的 mu 列表 (寻找最近的网格点索引)
    #         target_mus = np.array(self.custom_mu_values)
    #         for target_mu in target_mus:
    #             # 找到 xmu 中最接近 target_mu 的索引
    #             closest_idx = np.argmin(np.abs(self.xmu - target_mu))
    #             if closest_idx not in plot_indices:
    #                 plot_indices.append(closest_idx)
    #     else:
    #         # 模式2：默认的等间距步长逻辑 (保留你的原逻辑作为后备)
    #         duu = self.duu
    #         nduu = (self.tfecmu.shape[1]-1)*duu
    #         for ii in range(self.tfecmu.shape[1]):
    #             if ii % nduu != 0:
    #                 continue
    #             plot_indices.append(ii)
    #     # ========================================================

    #     # 先遍历一次所有需要绘制的曲线，收集 y 值以计算全局 y 轴范围
    #     all_y_values = []
    #     for ii in plot_indices:
    #         mu_val = np.round(self.xmu[ii], 2)
    #         # 过滤不需要显示的区间
    #         if self.plot_range_mu == 1 and mu_val < 0: continue
    #         elif self.plot_range_mu == -1 and mu_val > 0: continue
            
    #         y_values = f[:, ii]
    #         valid_values = y_values[~np.isnan(y_values)]
    #         all_y_values.extend(valid_values)
            
    #     # 计算全局 y 轴下限 (移到循环外部，避免重复计算覆盖)
    #     if all_y_values:
    #         global_y_min = min(all_y_values)
    #         print(f"y轴下限 {global_y_min}" )
    #     else:
    #         global_y_min = f[1,0]-5  # 如果没有有效值，使用默认值

    #     x_lim_max = Es.max()

    #     for ax, range_type in zip(axes, ['limited', 'full'] if plot_range == 'both' else [plot_range]):
    #         # 绘制不同pitch角的分布
    #         for ii in plot_indices:
    #             # 获取实际的 mu 值 (保留2位小数避免浮点尾巴)
    #             mu_val = np.round(self.xmu[ii], 2)
                
    #             # 根据显示标志位过滤和设置样式
    #             if self.plot_range_mu == 1 and mu_val < 0:
    #                 continue  # 仅显示正区间，跳过负值
    #             elif self.plot_range_mu == -1 and mu_val > 0:
    #                 continue  # 仅显示负区间，跳过正值
    #             elif self.plot_range_mu == 0 and mu_val < 0:
    #                 line_style = '--'  # 完整区间模式，负区间用虚线
    #             else:
    #                 line_style = '-'   # 默认实线
                
    #             # 提取 y 数据并收集用于计算全局 y 轴范围
    #             y_values = f[:, ii]
    #             valid_values = y_values[~np.isnan(y_values)]
    #             all_y_values.extend(valid_values)
                
    #             ax.plot(Es, y_values, 
    #                 label=fr'$\mu$ = {mu_val}', 
    #                 linestyle=line_style)
        
                
    #     # # 绘制分布函数
    #     # for ax, range_type in zip(axes, ['limited', 'full'] if plot_range == 'both' else [plot_range]):
    #     #     # 绘制不同pitch角的分布
    #     #     for ii in range(self.tfecmu.shape[1]):
    #     #         if ii % nduu != 0:
    #     #         # if ii != 0 and ii != 13 and ii !=25 and ii != 50:
    #     #             continue
                
    #     #         mu_val = np.round(self.xmu[ii], 2)
                
    #     #         # 【新增】：根据显示标志位过滤和设置样式
    #     #         if self.plot_range_mu == 1 and mu_val < 0:
    #     #             continue  # 仅显示正区间，跳过负值
    #     #         elif self.plot_range_mu == -1 and mu_val > 0:
    #     #             continue  # 仅显示负区间，跳过正值
    #     #         elif self.plot_range_mu == 0 and mu_val < 0:
    #     #             # 完整区间模式，负区间用虚线
    #     #             line_style = '--'
    #     #         else:
    #     #             # 默认实线
    #     #             line_style = '-'
                
    #     #         ax.plot(Es, f[:,ii], 
    #     #             label=fr'$\mu$ = {mu_val}', 
    #     #             linestyle=line_style)  # 应用线条样式
            
    #         # # 使用所有数据的最小值作为full图的y轴下限
    #         # if all_y_values:
    #         #     global_y_min = min(all_y_values)
    #         #     print(f"y轴下限 {global_y_min}" )
    #         # else:
    #         #     global_y_min = f[1,0]-5  # 如果没有有效值，使用默认值

    #         # 绘制Maxwell分布
    #         ax.plot(Es, self.ref_maxwell, label='Maxwell.', linestyle='--')
            
    #         # 设置中文字体
    #         plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    #         plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


    #         # 设置坐标轴范围
    #         if range_type == 'limited':
    #             x_lim_max = min(self.xlim_max, Es.max())
    #             x_lim = (0, x_lim_max)
    #             ax.set_xlim(x_lim)
                
    #             # 找到x_lim[1]（最大x值）对应的y值
    #             idx_max = np.argmin(np.abs(Es - x_lim[1]))
    #             y_values_at_max = []
                
    #             # # 收集所有曲线在最大x值处的y值
    #             # for ii in range(self.tfecmu.shape[1]):
    #             #     if ii % nduu != 0:
    #             #     # if ii != 0 and ii != 13 and ii !=25 and ii != 50:
    #             #         continue
    #             #     y_val = f[idx_max, ii]
    #             #     if not np.isnan(y_val):
    #             #         y_values_at_max.append(y_val)
                
    #             # 使用 plot_indices 替换原来的 ii % nduu
    #             for ii in plot_indices:
    #                 mu_val = np.round(self.xmu[ii], 2)
    #                 if self.plot_range_mu == 1 and mu_val < 0: continue
    #                 elif self.plot_range_mu == -1 and mu_val > 0: continue
                    
    #                 y_val = f[idx_max, ii]
    #                 if not np.isnan(y_val):
    #                     y_values_at_max.append(y_val)

    #             # 使用最小值作为y轴下限，留出一些边距
    #             if y_values_at_max:
    #                 y_min = min(y_values_at_max)
    #                 ax.set_ylim((y_min - 0.5, 0))  # 留出0.5的边距
    #             else:
    #                 ax.set_ylim((-5, 0))  # 如果没有有效值，使用默认范围
    #             ax.set_ylim(self.ylim_max, 0)
    #             # ax.set_title(f'P={self.pwdens}W/cm^3 勒让德阶数:{self.legmod} kperp={self.kperp}')

    #         else:
    #             x_lim_max = Es.max()
    #             ax.set_ylim((global_y_min - 0.5, 0))
    #             # ax.set_ylim((-5, 0))
    #             ax.set_title(f'r={self.redpsi[ipsi-1]}a P:{self.pwdens}W/cm^3 kperp:{self.kperp:.0f}')    
            
    #         # 设置标签
    #         # ax.set_xlabel('$E$ [keV]')
    #         ax.set_xlabel('E')
    #         ax.set_ylabel(r'$\log_{10}f$')
            
    #         # 设置刻度样式
    #         ax.tick_params(axis='both', which='major', width=2, length=8)  # 设置主刻度的宽度和长度
    #         ax.tick_params(axis='both', which='minor', width=1, length=4)  # 设置次刻度的宽度和长度

    #         # 设置边框粗细
    #         for spine in ax.spines.values():
    #             spine.set_linewidth(2)  # 设置边框线宽为2
    #             spine.set_edgecolor('black')  # 设置边框颜色为黑色

    #         # 设置图例
    #         legend = ax.legend(loc='upper right',
    #                         bbox_to_anchor=(0.98, 1),
    #                         framealpha=0.9,
    #                         ncol=2
    #                         )
    #         # 设置图例边框
    #         legend.get_frame().set_linewidth(2)  # 设置图例边框线宽为2
    #         legend.get_frame().set_edgecolor('black')  # 设置图例边框颜色为黑色
    #         # for text in legend.get_texts():
    #         #     text.set_fontsize(22)

    #         ax.grid(True)
        
    #     # ============ 保存图片 ============
    #     saved_filename = None
        
    #     # 1. 确定基础前缀：优先使用字典里定义的，否则使用默认编号
    #     base_name = "result"
    #     if "name" in self.output_paths:
    #         # 提取字典里路径的文件名部分作为基础前缀
    #         base_name = os.path.basename(self.output_paths["name"])
        
    #     # 2. 动态拼接物理参数后缀
    #     suffix = '_limited' if plot_range == 'limited' else '_full' if plot_range == 'full' else '_both'
    #     bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}'
        
    #     # 组合完整的文件名: 前缀 + 物理参数 + 范围后缀 + .png
    #     filename = f'{base_name}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_E_{x_lim_max}_dv_{self.du}_{bh}{suffix}.png'
        
    #     # 3. 确定保存的文件夹路径：强制保存到当前模式的输出目录 (self.current_out_dir)
    #     save_dir = self.current_out_dir if hasattr(self, 'current_out_dir') and self.current_out_dir else "."
        
    #     # 拼接最终的完整路径：子文件夹 + 拼接后的完整文件名
    #     full_path = os.path.join(save_dir, filename)
            
    #     # 4. 执行保存
    #     try:
    #         plt.savefig(full_path, dpi=300, bbox_inches='tight')
    #         print(f"图形已保存为: {full_path}")
    #         saved_filename = full_path
    #     except Exception as e:
    #         print(f"保存图片失败: {e}")

    #     # plt.show()  # 在批量运行时注释掉，避免弹窗阻断程序
    #     return fig, saved_filename
        
    #     # plt.tight_layout()
        
    #     # # 保存图片
    #     # if save_path:
    #     #     suffix = '_f(v_u)_limited' if plot_range == 'limited' else '_f(v_u)_full' if plot_range == 'full' else '_f(v_u)_both'
    #     #     bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}' # ICRH模式 NBI模式 加热模式 离子种类  
    #     #     plt.savefig(f'{save_path}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_E_{x_lim_max}_dv_{self.du}_{bh}{suffix}.png', dpi=300, bbox_inches='tight')
    #     #     print(f"图形已保存为: {save_path}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_E_{x_lim_max}_dv_{self.du}_{bh}{suffix}.png")
    #     #     # plt.savefig(f'{save_path}{suffix}_P_{self.pwdens}_Wcm3_L_{self.legmod}_kperp_{self.kperp}_Xmax_{x_lim_max1}_dv_{self.du}.png', dpi=300, bbox_inches='tight')
    #     #     # print(f"图形已保存为: {save_path}{suffix}_P_{self.pwdens}_Wcm3_L_{self.legmod}_kperp_{self.kperp}_Xmax_{x_lim_max1}_dv_{self.du}.png")
        
    #     # # 图片显示开关-速度分布
    #     # plt.show()
    #     # # plt.close()
    #     # return figplot_f_heatmap
    
    # # ===================================================================
    # # 绘图函数，绘制粒子速度分布函数，热图形式
    # # ===================================================================
    # def plot_f_heatmap(self, ctx, ipsi, plot_range='both', 
    #                 nbi_tail_E=None, nbi_tail_mu=None, nbi_char_mu=None, nbi_char_E=None, 
    #                 icrh_tail_E=None, icrh_tail_mu=None, icrh_char_mu=None, icrh_char_E=None,
    #                 co_tail_E=None, co_tail_mu=None, co_char_mu=None, co_char_E=None):
    #     """
    #     绘制粒子速度分布函数的热图 - Origin格式
    #     """
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import LinearSegmentedColormap, Normalize
    #     from matplotlib.ticker import FuncFormatter
    #     import os
        
    #     # 验证输入
    #     if not self:
    #         print("警告：没有提供输入数据")
    #         return
    #     if not plot_range in ['limited', 'full', 'both']:
    #         plot_range = 'both'
        
    #     # ==================== Origin 格式配置 ====================
    #     PAGE_WIDTH = 6
    #     PAGE_HEIGHT = 6
        
    #     LAYER_LEFT = 1
    #     LAYER_TOP = 1
    #     LAYER_WIDTH = 3
    #     LAYER_HEIGHT = 3
        
    #     FONT_FAMILY = 'serif'
    #     FONT_SERIF = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
    #     MATH_FONTSET = 'stix'

    #     FONT_SIZE = 16
    #     LABEL_SIZE = 16
    #     TICK_SIZE = 12
    #     COLORBAR_SIZE = 12
    #     LEGEND_SIZE = 10

    #     COLORBAR_LABEL_WEIGHT = 'bold'
    #     LABEL_WEIGHT = 'bold'
    #     TICK_WEIGHT = 'bold'
    #     LEGEND_WEIGHT = 'bold'
        
    #     AXIS_LINE_WIDTH = 1.2
    #     GRID_LINEWIDTH = 1
    #     GRID_ALPHA = 0.5
        
    #     DPI = 300
        
    #     plt.rcParams.update({
    #         'font.family': FONT_FAMILY,
    #         'font.serif': FONT_SERIF,
    #         'mathtext.fontset': MATH_FONTSET,
    #         'font.size': FONT_SIZE,
    #         'axes.linewidth': AXIS_LINE_WIDTH,
    #     })
        
    #     # 计算能量
    #     Es = self.tbu**2 * ctx.tempi[self.ispres]
        
    #     min_cutoff_ip = 1
    #     if min_cutoff_ip > 0:
    #         self.tfecmu[:min_cutoff_ip, :] = np.nan

    #     fmax = self.f_max
    #     self.tfecmu[self.tfecmu > fmax] = np.nan
    #     fmin = self.f_min
    #     self.tfecmu[self.tfecmu < fmin] = np.nan
    #     f = self.tfecmu.copy()
        
    #     # ========== 新增：转换为 log 显示 ==========
    #     f[f <= 0] = np.nan
    #     f_log = np.log10(f)
        
    #     # 创建自定义颜色映射
    #     colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0)] 
    #     cm = LinearSegmentedColormap.from_list('wy_rk', colors, N=100)
        
    #     # 创建图形 - Origin格式
    #     fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        
    #     # 创建图层（单图或双图）
    #     if plot_range == 'both':
    #         layer_width_each = (LAYER_WIDTH - 0.3) / 2
    #         ax1 = fig.add_axes([
    #             LAYER_LEFT / PAGE_WIDTH,
    #             1 - (LAYER_TOP + LAYER_HEIGHT) / PAGE_HEIGHT,
    #             layer_width_each / PAGE_WIDTH,
    #             LAYER_HEIGHT / PAGE_HEIGHT
    #         ])
    #         ax2 = fig.add_axes([
    #             (LAYER_LEFT + layer_width_each + 0.3) / PAGE_WIDTH,
    #             1 - (LAYER_TOP + LAYER_HEIGHT) / PAGE_HEIGHT,
    #             layer_width_each / PAGE_WIDTH,
    #             LAYER_HEIGHT / PAGE_HEIGHT
    #         ])
    #         axes = [ax1, ax2]
    #     else:
    #         ax = fig.add_axes([
    #             LAYER_LEFT / PAGE_WIDTH,
    #             1 - (LAYER_TOP + LAYER_HEIGHT) / PAGE_HEIGHT,
    #             LAYER_WIDTH / PAGE_WIDTH,
    #             LAYER_HEIGHT / PAGE_HEIGHT
    #         ])
    #         axes = [ax]
        
    #     x_lim_max1 = None
        
    #     for idx, (ax, range_type) in enumerate(zip(axes, ['limited', 'full'] if plot_range == 'both' else [plot_range])):
    #         # 设置坐标轴范围
    #         if range_type == 'limited':
    #             x_lim_max = min(self.xlim_max, Es.max())
    #             ax.set_xlim(Es[min_cutoff_ip], x_lim_max)
    #             x_lim_max1 = x_lim_max
    #         else:
    #             x_lim_max1 = Es.max()
    #             ax.set_xlim(Es[min_cutoff_ip], x_lim_max1)
            
    #         ax.set_ylim(self.mu_min, self.mu_max)
            
    #         # 数据范围（基于 log 数据）
    #         valid_data = f_log[~np.isnan(f_log)]
    #         vmin = np.min(valid_data) if len(valid_data) > 0 else -10
    #         vmax = np.max(valid_data) if len(valid_data) > 0 else 0
            
    #         # 绘制热图 - 使用 log 数据和线性 Normalize
    #         im = ax.pcolormesh(Es, self.xmu, f_log.T, cmap=cm, shading='gouraud', 
    #                         norm=Normalize(vmin=vmin, vmax=vmax))
            
    #         # ==================== 色标设置（参考 plot_f_velocity_para_perp） ====================
    #         font_height_inch = COLORBAR_SIZE / 72
    #         cbar_width_inch = font_height_inch * 0.9
    #         label_offset_inch = font_height_inch * 0.3
            
    #         # 计算色标位置
    #         if plot_range == 'both':
    #             ax_pos = ax.get_position()
    #             cbar_x = ax_pos.x1 + 0.02
    #             cbar_y = ax_pos.y0
    #             cbar_height = ax_pos.height
    #             cbar_width = cbar_width_inch / PAGE_WIDTH
    #         else:
    #             cbar_x = (LAYER_LEFT + LAYER_WIDTH + label_offset_inch) / PAGE_WIDTH
    #             cbar_y = 1 - (LAYER_TOP + LAYER_HEIGHT) / PAGE_HEIGHT
    #             cbar_height = LAYER_HEIGHT / PAGE_HEIGHT
    #             cbar_width = cbar_width_inch / PAGE_WIDTH
            
    #         # 创建色标
    #         cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    #         cbar = fig.colorbar(im, cax=cbar_ax)
            
    #         # 设置色标标签 - 使用 logf
    #         cbar.set_label('logf', fontsize=COLORBAR_SIZE, fontweight=COLORBAR_LABEL_WEIGHT)
    #         cbar.ax.tick_params(labelsize=COLORBAR_SIZE, width=AXIS_LINE_WIDTH)
            
    #         # 设置色标刻度字体
    #         for label in cbar.ax.get_yticklabels():
    #             label.set_fontname('Times New Roman')
    #             label.set_fontweight(TICK_WEIGHT)
    #             label.set_fontsize(COLORBAR_SIZE)
            
    #         cbar.outline.set_linewidth(AXIS_LINE_WIDTH)
            
    #         # ==================== 坐标轴标签 ====================
    #         ax.set_xlabel('E [keV]', fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
    #         ax.set_ylabel(r'$\mu$', fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
            
    #         # 刻度设置
    #         ax.tick_params(axis='both', which='major', 
    #                     labelsize=TICK_SIZE, 
    #                     width=AXIS_LINE_WIDTH, 
    #                     length=8)
            
    #         for label in ax.get_xticklabels() + ax.get_yticklabels():
    #             label.set_fontname('Times New Roman')
    #             label.set_fontweight(TICK_WEIGHT)
    #             label.set_fontsize(TICK_SIZE)
            
    #         for spine in ax.spines.values():
    #             spine.set_linewidth(AXIS_LINE_WIDTH)
    #             spine.set_edgecolor('black')
            
    #         # 网格线
    #         ax.grid(True, linestyle='--', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, color='gray')
            
    #         # ==================== 图例设置（图内） ====================
    #         handles, labels = ax.get_legend_handles_labels()
    #         if handles:
    #             leg = ax.legend(handles, labels, 
    #                         loc='upper right',
    #                         bbox_to_anchor=(0.98, 0.98),
    #                         borderaxespad=0, 
    #                         framealpha=0.9,
    #                         fontsize=LEGEND_SIZE, 
    #                         borderpad=0.4, 
    #                         labelspacing=0.3,
    #                         frameon=True,
    #                         facecolor='white',
    #                         edgecolor='black')
                
    #             for text in leg.get_texts():
    #                 text.set_fontname('Times New Roman')
    #                 text.set_fontsize(LEGEND_SIZE)
    #                 text.set_fontweight(LEGEND_WEIGHT)
                
    #             leg.get_frame().set_linewidth(1.5)
        
    #     # ============ 保存图片 ============
    #     saved_filename = None
        
    #     base_name = "result"
    #     if "name" in self.output_paths:
    #         base_name = os.path.basename(self.output_paths["name"])
        
    #     suffix = '_limited' if plot_range == 'limited' else '_full' if plot_range == 'full' else '_both'
    #     bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}'
        
    #     filename = f'{base_name}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_E_{x_lim_max1}_dv_{self.du}_{bh}{suffix}.png'
        
    #     save_dir = self.current_out_dir if hasattr(self, 'current_out_dir') and self.current_out_dir else "."
    #     full_path = os.path.join(save_dir, filename)
            
    #     try:
    #         plt.savefig(full_path, dpi=DPI, bbox_inches='tight')
    #         print(f"图形已保存为: {full_path}")
    #         saved_filename = full_path
    #     except Exception as e:
    #         print(f"保存图片失败: {e}")

    #     return fig, saved_filename

    # # ===================================================================
    # # 绘图函数，绘制粒子速度分布函数，F(V∥,V⊥)形式
    # # ===================================================================
    # def plot_f_velocity_para_perp(self, ipsi, plot_range='full'):
    #     """
    #     绘制 f(v_perp, v_parallel) 速度空间图 - Origin格式
    #     """
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import LogNorm
    #     from matplotlib.ticker import FuncFormatter
    #     import os
        
    #     # ==================== Origin 格式配置 ====================
    #     PAGE_WIDTH = 6
    #     PAGE_HEIGHT = 6
        
    #     LAYER_LEFT = 1
    #     LAYER_TOP = 1
    #     LAYER_WIDTH = 3
    #     LAYER_HEIGHT = 3
        
    #     FONT_FAMILY = 'serif'
    #     FONT_SERIF = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
    #     MATH_FONTSET = 'stix'

    #     FONT_SIZE = 16
    #     LABEL_SIZE = 16
    #     TICK_SIZE = 12
    #     COLORBAR_SIZE = 12
    #     LEGEND_SIZE = 10

    #     COLORBAR_LABEL_WEIGHT = 'bold'
    #     LABEL_WEIGHT = 'bold'
    #     TICK_WEIGHT = 'bold'
    #     LEGEND_WEIGHT = 'bold'
        
    #     AXIS_LINE_WIDTH = 1.2
    #     GRID_LINEWIDTH = 1
    #     GRID_ALPHA = 0.5
        
    #     DPI = 300
        
    #     plt.rcParams.update({
    #         'font.family': FONT_FAMILY,
    #         'font.serif': FONT_SERIF,
    #         'mathtext.fontset': MATH_FONTSET,
    #         'font.size': FONT_SIZE,
    #         'axes.linewidth': AXIS_LINE_WIDTH,
    #     })
        
    #     # 1. 计算速度网格
    #     v = self.tbu
        
    #     # 2. 将 (v, mu) 网格转换为 (v_par, v_perp) 网格
    #     V, MU = np.meshgrid(v, self.xmu, indexing='ij')
    #     V_par = V * MU
    #     V_perp = V * np.sqrt(1 - MU**2)
        
    #     # 3. 处理分布函数数据（先截断）
    #     f = self.tfecmu.copy()
    #     fmin = self.f_min
    #     f[f < fmin] = np.nan  # 截断低值数据
    #     f_log = np.log10(f)
        
    #     # ========== 新增：基于截断后的数据计算有效范围 ==========
    #     # 找到有效数据（非 NaN）对应的 v_par 和 v_perp 范围
    #     valid_mask = ~np.isnan(f)
        
    #     if np.any(valid_mask):
    #         # 计算有效数据点的最大 |v_par| 和 v_perp
    #         v_par_valid = V_par[valid_mask]
    #         v_perp_valid = V_perp[valid_mask]
            
    #         max_v_par = np.max(np.abs(v_par_valid))  # 最大 |v_par|
    #         max_v_perp = np.max(v_perp_valid)         # 最大 v_perp
            
    #         print(f"截断后数据范围: |v_par| ≤ {max_v_par:.2f}, v_perp ≤ {max_v_perp:.2f}")
    #     else:
    #         # 如果没有有效数据，使用原始网格范围
    #         max_v_par = np.max(np.abs(V_par))
    #         max_v_perp = np.max(V_perp)
    #         print("警告：没有有效数据，使用原始网格范围")
        
    #     # 4. 创建图形
    #     fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        
    #     ax = fig.add_axes([
    #         LAYER_LEFT / PAGE_WIDTH,
    #         1 - (LAYER_TOP + LAYER_HEIGHT) / PAGE_HEIGHT,
    #         LAYER_WIDTH / PAGE_WIDTH,
    #         LAYER_HEIGHT / PAGE_HEIGHT
    #     ])
        
    #     pcm = ax.pcolormesh(V_par, V_perp, f_log, 
    #                         cmap='hot_r', shading='gouraud')
        
    #     # ==================== 色标设置 ====================
    #     font_height_inch = COLORBAR_SIZE / 72
    #     cbar_width_inch = font_height_inch * 0.9
    #     label_offset_inch = font_height_inch * 0.3
        
    #     cbar_x = (LAYER_LEFT + LAYER_WIDTH + label_offset_inch) / PAGE_WIDTH
    #     cbar_y = 1 - (LAYER_TOP + LAYER_HEIGHT) / PAGE_HEIGHT
    #     cbar_height = LAYER_HEIGHT / PAGE_HEIGHT
    #     cbar_width = cbar_width_inch / PAGE_WIDTH
        
    #     cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    #     cbar = fig.colorbar(pcm, cax=cbar_ax)
        
    #     cbar.set_label('logf', fontsize=COLORBAR_SIZE, fontweight=COLORBAR_LABEL_WEIGHT)
    #     cbar.ax.tick_params(labelsize=COLORBAR_SIZE, width=AXIS_LINE_WIDTH)
        
    #     for label in cbar.ax.get_yticklabels():
    #         label.set_fontname('Times New Roman')
    #         label.set_fontweight(TICK_WEIGHT)
    #         label.set_fontsize(COLORBAR_SIZE)
        
    #     cbar.outline.set_linewidth(AXIS_LINE_WIDTH)
        
    #     # ==================== 坐标轴设置 ====================
    #     ax.set_xlabel(r'$v_{//}$', fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
    #     ax.set_ylabel(r'$v_{\perp}$', fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
        
    #     # ax.set_title(f'ICRH Heating (r={self.redpsi[ipsi-1]}a)', 
    #     #             fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT)
        
    #     ax.set_aspect('equal')
        
    #     # 基于截断后数据计算范围
    #     if plot_range == 'limited':
    #         max_v_par = min(5.0, max_v_par)
    #         max_v_perp = min(5.0, max_v_perp)
        
    #     # 计算跨度：x 是正负范围所以乘 2，y 是单边范围
    #     x_span = 2 * max_v_par      # x 总跨度
    #     y_span = max_v_perp         # y 总跨度
        
    #     # 取较大值作为统一跨度标准
    #     max_span = max(x_span, y_span)
        
    #     # 取整到整数，确保刻度显示整数
    #     max_span = int(np.ceil(max_span))
        
    #     # 设置坐标轴范围
    #     # x: -max_span/2 到 max_span/2，跨度 max_span
    #     # y: 0 到 max_span，跨度 max_span
    #     ax.set_xlim(-max_span/2, max_span/2)
    #     ax.set_ylim(0, max_span)

    #     # 强制整数刻度
    #     from matplotlib.ticker import MaxNLocator
    #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    #     print(f"数据范围: |v_par| ≤ {max_v_par:.2f}, v_perp ≤ {max_v_perp:.2f}")
    #     print(f"跨度比较: x_span = {x_span:.2f}, y_span = {y_span:.2f}")
    #     print(f"坐标轴范围: x ∈ [-{max_span/2}, {max_span/2}], y ∈ [0, {max_span}], 统一跨度 {max_span}")
        
    #     ax.tick_params(axis='both', which='major', 
    #                 labelsize=TICK_SIZE, 
    #                 width=AXIS_LINE_WIDTH, 
    #                 length=8)
        
    #     for label in ax.get_xticklabels() + ax.get_yticklabels():
    #         label.set_fontname('Times New Roman')
    #         label.set_fontweight(TICK_WEIGHT)
    #         label.set_fontsize(TICK_SIZE)
        
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(AXIS_LINE_WIDTH)
    #         spine.set_edgecolor('black')
        
    #     # ax.grid(True, linestyle='--', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, color='gray')

    #     # ============ 保存图片 ============
    #     saved_filename = None
        
    #     base_name = "result"
    #     if "name" in self.output_paths:
    #         base_name = os.path.basename(self.output_paths["name"])
        
    #     suffix = '_limited' if plot_range == 'limited' else '_full' if plot_range == 'full' else '_both'
    #     bh = f'{self.icrh_enabled}{self.nbi_enabled}{self.ipheat}{self.ispres+1}'
        
    #     filename = f'{base_name}_P_{self.pwdens}_Wcm3_kperp_{self.kperp}_L_{self.legmod}_dv_{self.du}_{bh}{suffix}_vpar_vperp_.png'
        
    #     save_dir = self.current_out_dir if hasattr(self, 'current_out_dir') and self.current_out_dir else "."
    #     full_path = os.path.join(save_dir, filename)
            
    #     try:
    #         plt.savefig(full_path, dpi=DPI, bbox_inches='tight')
    #         print(f"图形已保存为: {full_path}")
    #         saved_filename = full_path
    #     except Exception as e:
    #         print(f"保存图片失败: {e}")

    #     return fig, saved_filename

    def extract_high_energy_tail_contour(self, absolute_threshold=None):
        """
        提取高能尾等值线，返回其向高能方向延伸最远处的 μ 和能量。
        默认阈值为 f_min；达到能域末点仍未降至阈值的曲线保留末点。
        """
        if self.tfecmu is None or self.tbkev is None:
            return None, None, None, None
            
        # 直接使用 f_min 作为阈值
        if absolute_threshold is None:
            absolute_threshold = self.f_min
            
        tail_E = []
        tail_mu = []
        
        for imu in range(self.nmu):
            f_profile = self.tfecmu[:, imu]
            
            valid_f = f_profile[~np.isnan(f_profile)]
            if len(valid_f) == 0 or np.max(valid_f) <= 0:
                continue
            
            # 找到分布函数的峰值位置
            peak_idx = np.nanargmax(f_profile)
            peak_value = f_profile[peak_idx]
            
            # 如果峰值低于阈值，跳过
            if peak_value < absolute_threshold:
                continue
                
            # 从峰值向高能方向搜索，找到分布函数下降到阈值的位置
            tail_idx = peak_idx
            while tail_idx < len(f_profile) - 1 and f_profile[tail_idx] >= absolute_threshold:
                tail_idx += 1
                
            # 如果搜索到末点，使用最后一个有效点。
            if tail_idx >= len(f_profile) - 1:
                # 使用最后一个有效点作为尾部位置
                valid_indices = np.where(~np.isnan(f_profile))[0]
                if len(valid_indices) > 0:
                    tail_idx = valid_indices[-1]
                else:
                    continue
                    
            # 在阈值附近进行插值，提高精度
            if tail_idx > 0 and f_profile[tail_idx] < absolute_threshold:
                # 使用线性插值
                f1 = f_profile[tail_idx - 1]
                f2 = f_profile[tail_idx]
                
                if not np.isnan(f1) and not np.isnan(f2) and f1 != f2:
                    # 计算插值系数
                    ratio = (f1 - absolute_threshold) / (f1 - f2)
                    # 插值后的精确能量
                    E_refined = self.tbkev[tail_idx - 1] + ratio * (self.tbkev[tail_idx] - self.tbkev[tail_idx - 1])
                else:
                    E_refined = self.tbkev[tail_idx]
            else:
                E_refined = self.tbkev[tail_idx]
            
            tail_E.append(E_refined)
            tail_mu.append(self.xmu[imu])
                
        tail_E = np.array(tail_E)
        tail_mu = np.array(tail_mu)
        
        characteristic_E = None
        characteristic_mu = None
        
        if len(tail_E) > 0:
            # 寻找延伸到最大能量时的 mu 和 E
            # max_E_idx = np.argmax(tail_E)
            # characteristic_mu = tail_mu[max_E_idx]
            # characteristic_E = tail_E[max_E_idx]

            max_E = np.max(tail_E)
            max_E_indices = np.where(tail_E == max_E)[0]
            max_E_idx = max_E_indices[-1]  # 最大值的索引
            characteristic_mu = tail_mu[max_E_idx]
            characteristic_E = tail_E[max_E_idx]
            
        return tail_E, tail_mu, characteristic_mu, characteristic_E

    # ===================================================================
    # 调试信息输出
    # ===================================================================
    def write_debug_info(self, ipsi):
        """
        将调试信息写入结果根目录的 debug.txt 汇总文件，按模式分区记录，
        同一模式下的输入信息和输出信息放在同一分区内。

        记录内容：
        - 输入信息：NBI 功率 pwdens_nbi、束能量 energies_keV（NBI 开启时）；
          ICRH 输入功率 pwdens、trapping 参数（ICRH 开启时）
        - 输出信息：尾端等值线的位置、阈值及能域截断情况；
          原始/归一化后的 NBI 束附近峰值和分布非负性诊断
        每个径向点追加一段记录。
        """
        # 汇总文件在结果根目录（模式文件夹的上一级）
        debug_path = os.path.join(os.path.dirname(self.current_out_dir), 'debug.txt')

        # 当前加热模式名称
        if self.nbi_enabled and self.icrh_enabled:
            mode_name = "协同加热"
        elif self.nbi_enabled:
            mode_name = "NBI"
        else:
            mode_name = "ICRH"

        lines = []

        # ---- 模式分区头 + 输入信息（每个模式只写一次） ----
        section_title = f"========== {mode_name}模式 =========="
        if os.path.exists(debug_path):
            with open(debug_path, encoding='utf-8') as f:
                existing = f.read()
        else:
            existing = ''
        if section_title not in existing:
            lines.append(f"\n{section_title}\n")
            if self.nbi_enabled:
                lines.append("[输入] NBI 设置\n")
                lines.append(f"  NBI功率密度 pwdens_nbi = {self.pwdens_nbi} W/cm^3\n")
                lines.append(f"  束能量 energies_keV = {list(self.nbi_energies_keV)} keV\n")
            if self.icrh_enabled:
                lines.append("[输入] ICRH 设置\n")
                lines.append(f"  ICRH输入功率 pwdens = {self.power} W/cm^3\n")
                lines.append(f"  环向捕获 trapping: enabled={int(self.enable_trapping)}, "
                             f"ucr={self.trapping_ucr}, a={self.trapping_a}\n")
            lines.append("[输出]\n")

        # ---- 输出信息（每个径向点一段） ----
        lines.append(f"{'-'*10} 径向位置 psi = {self.redpsi[ipsi-1]} {'-'*10}\n")
        lines.append(f"  整体归一化因子 fnorm = {self.fnorm:.12e}\n")
        lines.append(f"  带状矩阵相对残差 = {self.matrix_relative_residual:.6e}\n")
        if hasattr(self, 'density_residual'):
            lines.append(f"  密度约束残差 = {self.density_residual:.6e}\n")
            lines.append(f"  原点弱方程粒子平衡残差 = {self.particle_balance_residual:.6e}\n")
            lines.append(f"  总功率残差 = {self.bulk_power_residual:.6e} W/cm^3\n")
            lines.append(f"  实际RF吸收 = {self.hrate:.6e} W/cm^3, 模式 = {self.rf_power_mode}\n")
            lines.append(f"  分布非负性及能域检查通过 = {self.distribution_checks_passed}\n")
            for warning in self.quality_warnings:
                lines.append(f"  需处理：{warning}\n")
            lines.append(f"  平均总能量 = {self.mean_energy_keV:.6e} keV\n")
            lines.append(f"  平均平行速度 = {self.mean_parallel_velocity_ms:.6e} m/s\n")
            lines.append(f"  扣除定向流能的平行温度 = {self.thermal_parallel_temperature_keV:.6e} keV\n")
        min_iE, min_imu = np.unravel_index(np.nanargmin(self.tfecmu), self.tfecmu.shape)
        lines.append(f"  归一化分布最小值 = {self.tfecmu[min_iE, min_imu]:.6e}, "
                     f"E = {self.tbkev[min_iE]:.4f} keV, mu = {self.xmu[min_imu]:.4f}, "
                     f"负值网格点数 = {np.count_nonzero(self.tfecmu < 0)}\n")

        tail_E, tail_mu, char_mu, char_E = self.extract_high_energy_tail_contour()
        lines.append("尾端等值线特征（extract_high_energy_tail_contour）\n")
        lines.append(f"  等值线阈值 f_min = {self.f_min:.6e}\n")
        lines.append(f"  特征能量 char_E = {char_E} keV\n")
        lines.append(f"  特征俯仰角 char_mu = {char_mu}\n")
        boundary_mu = tail_mu[tail_E == self.tbkev[-1]]
        lines.append(f"  能域边界截断 = {bool(len(boundary_mu))}, "
                     f"E_max = {self.tbkev[-1]:.4f} keV, "
                     f"触及边界的 mu = {boundary_mu.tolist()}\n")

        # NBI 模式：束能量附近（±5% 窗口）tfecmu 的最大值及对应 E 和 μ
        if self.nbi_enabled:
            lines.append(f"  NBI实际注入功率 = {self.nbi_injected_power:.6e} W/cm^3\n")
            lines.append(f"  NBI热粒子替换带走功率 = {self.nbi_sink_power:.6e} W/cm^3\n")
            lines.append(f"  NBI净源功率 = {self.nbi_source_power:.6e} W/cm^3\n")
            lines.append(f"  NBI净粒子源矩 = {self.nbi_net_particle_source_moment:.6e}\n")
            raw_distribution = self.raw_tbfu @ self.plgcmu
            for E_beam in np.atleast_1d(self.nbi_energies_keV):
                idx_E = np.where(np.abs(self.tbkev - E_beam) <= 0.05 * E_beam)[0]
                if len(idx_E) == 0:
                    continue
                sub = self.tfecmu[idx_E, :]
                iE, imu = np.unravel_index(np.nanargmax(sub), sub.shape)
                lines.append(f"束能量 {E_beam} keV 附近 tfecmu 最大值\n")
                lines.append(f"  max(tfecmu) = {sub[iE, imu]:.6e},  "
                             f"E = {self.tbkev[idx_E[iE]]:.4f} keV,  mu = {self.xmu[imu]:.4f}\n")
                lines.append(f"  归一化前原始峰值 = {np.nanmax(raw_distribution[idx_E, :]):.6e}\n")

        with open(debug_path, 'a', encoding='utf-8') as f:
            f.writelines(lines)

        
    # # 标记高能尾位置
    # def extract_high_energy_tail_contour(self, absolute_threshold=None):
    #     """
    #     从当前计算结果中提取高能尾的等值轮廓线，并确定特征 pitch angle
    #     使用绝对阈值截断法，并引入亚网格插值提高尖端位置精度
    #     """
    #     if self.tfecmu is None or self.tbkev is None:
    #         return None, None, None, None
            
    #     # 如果没有传入阈值，使用 self.f_min
    #     if absolute_threshold is None:
    #         absolute_threshold = self.f_min
            
    #     tail_E = []
    #     tail_mu = []
    #     tail_f_vals = []  
        
    #     for imu in range(self.nmu):
    #         f_profile = self.tfecmu[:, imu]
            
    #         valid_f = f_profile[~np.isnan(f_profile)]
    #         if len(valid_f) == 0 or np.max(valid_f) <= 0:
    #             continue
            
    #         f_threshold = absolute_threshold
            
    #         # 反向搜索：找到第一个大于等于阈值的点 (粗略边界)
    #         tail_idx = -1
    #         for ip in range(self.nptu - 1, -1, -1):
    #             if not np.isnan(f_profile[ip]) and f_profile[ip] >= f_threshold:
    #                 tail_idx = ip
    #                 break
            
    #         if tail_idx > 0:
    #             # 检查边界条件，确保tail_idx+1不会越界
    #             if tail_idx >= self.nptu - 1:
    #                 # 如果tail_idx已经是最后一个点，直接使用该点的能量
    #                 E_refined = self.tbkev[tail_idx]
    #                 tail_E.append(E_refined)
    #                 tail_mu.append(self.xmu[imu])
    #                 tail_f_vals.append(f_profile[tail_idx])
    #                 continue
                    
    #             # 【新增】：亚网格插值精修边界位置
    #             # tail_idx 是刚跨过阈值的点，tail_idx+1 是低于阈值的点
    #             f1 = f_profile[tail_idx]       # 略大于阈值
    #             f2 = f_profile[tail_idx + 1]   # 略小于阈值
                
    #             # 确保插值有效
    #             if not np.isnan(f2) and f1 != f2:
    #                 # 线性插值系数：f 在 f1 和 f2 之间线性变化，计算达到 f_threshold 的比例
    #                 ratio = (f1 - f_threshold) / (f1 - f2)
    #                 # 插值后的精确能量 = 粗略点能量 + 比例 * 步长
    #                 E_refined = self.tbkev[tail_idx] + ratio * (self.tbkev[tail_idx + 1] - self.tbkev[tail_idx])
    #             else:
    #                 E_refined = self.tbkev[tail_idx]
                
    #             tail_E.append(E_refined)
    #             tail_mu.append(self.xmu[imu])
    #             tail_f_vals.append(f_profile[tail_idx])
                
    #     tail_E = np.array(tail_E)
    #     tail_mu = np.array(tail_mu)
    #     tail_f_vals = np.array(tail_f_vals)
        
    #     characteristic_E = None
    #     characteristic_mu = None
        
    #     if len(tail_E) > 0:
    #         # 寻找延伸到最大能量时的 mu 和 E
    #         max_E_idx = np.argmax(tail_E)
    #         characteristic_mu = tail_mu[max_E_idx]
    #         characteristic_E = tail_E[max_E_idx]
            
    #     return tail_E, tail_mu, characteristic_mu, characteristic_E


    # def extract_high_energy_tail_contour(self, threshold_factor=1e-7):
    #     """
    #     从当前计算结果中提取高能尾的最大能量轮廓线，并确定特征 pitch angle
    #     """
    #     if self.tfecmu is None or self.tbkev is None:
    #         return None, None, None
            
    #     tail_E = []
    #     tail_mu = []
    #     tail_f_vals = []
        
    #     for imu in range(self.nmu):
    #         f_profile = self.tfecmu[:, imu]
    #         f_max = np.nanmax(f_profile)
    #         if f_max <= 0 or np.isnan(f_max):
    #             continue
                
    #         f_threshold = f_max * threshold_factor
            
    #         tail_idx = -1
    #         for ip in range(self.nptu - 1, -1, -1):
    #             if f_profile[ip] >= f_threshold:
    #                 tail_idx = ip
    #                 break
            
    #         if tail_idx > 0:
    #             tail_E.append(self.tbkev[tail_idx])
    #             tail_mu.append(self.xmu[imu])
    #             tail_f_vals.append(f_profile[tail_idx])
                
    #     tail_E = np.array(tail_E)
    #     tail_mu = np.array(tail_mu)
    #     tail_f_vals = np.array(tail_f_vals)
        
    #     # 计算特征 pitch angle：高能尾延伸到最大能量时的 mu 值
    #     characteristic_mu = None
    #     if len(tail_E) > 0:
    #         max_E_idx = np.argmax(tail_E)
    #         characteristic_mu = tail_mu[max_E_idx]
    #         # max_f_idx = np.argmax(tail_f_vals)
    #         # characteristic_mu = tail_mu[max_f_idx]
            
    #     return tail_E, tail_mu, characteristic_mu









# ===================================================================
# 配置输入文件
# ===================================================================
def parse_command_line_args():
    """
    解析命令行参数并返回配置文件、CSV文件和标签列表
    
    Returns:
        tuple: (config_files, csv_files, labels)
    """
    # 初始化变量
    plot = False
    custom_labels = None
    
    # 导入必要的模块
    import sys
    import os
    import io
    import json
    
    # 定义默认配置
    DEFAULT_CONFIG_FILES = ['int_1.json']
    DEFAULT_CSV_FILES = ['data_GYCAVA_RF.csv', 
                        'Stix_data_P_0.98_ksi_69.11.csv', 
                        'data_Stix.csv', 
                        'Stix_data_P_0_ksi_0.00.csv']
    DEFAULT_LABELS = ['FP', 'GYCAVA-RF', 
                     'Stix,analytic（南华）', 
                     'Stix,analytic', 
                     'Maxwellian']
    
    # 显示使用说明
    if len(sys.argv) < 2:
        print("Usage: python 3Dfpql2.py [config_file1] [config_file2] ...")
        print("  or: python 3Dfpql2.py to run with default configs")
        print(f"  Default configs: {DEFAULT_CONFIG_FILES}")
        print(f"  Default CSV files: {DEFAULT_CSV_FILES}")
        print(f"  Default labels: {DEFAULT_LABELS}")
    
    print("输入文件开始处理1")
    
    # 初始化列表
    config_files = []
    csv_files = []
    labels = []
    
    # 解析命令行参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # 根据文件扩展名分类
        if arg.endswith('.json'):
            config_files.append(arg)
        elif arg.endswith('.csv'):
            csv_files.append(arg)
        else:
            labels.append(arg)
        i += 1
    
    # 使用默认值（如果未提供）
    config_files = config_files if config_files else DEFAULT_CONFIG_FILES
    csv_files = csv_files if csv_files else DEFAULT_CSV_FILES
    labels = labels if labels else DEFAULT_LABELS
    
    return config_files, csv_files, labels

# ===================================================================
# 验证配置文件
# ===================================================================
def validate_files(config_files, csv_files):
    """
    验证文件是否存在
    
    Args:
        config_files: 配置文件列表
        csv_files: CSV文件列表
    """
    import os
    
    print("\n验证文件存在性:")
    
    # 检查配置文件
    print("配置文件:")
    for config in config_files:
        if os.path.exists(config):
            print(f"   {config}")
        else:
            print(f"   {config} (不存在)")
    
    # 检查CSV文件
    print("\nCSV文件:")
    for csv in csv_files:
        if os.path.exists(csv):
            print(f"   {csv}")
        else:
            print(f"   {csv} (不存在)")

# ===================================================================
# 配置并验证输入文件
# ===================================================================
def config_input_files():
    """
    配置并验证输入文件
    """
    # 解析命令行参数
    config_files, csv_files, labels = parse_command_line_args()
    
    # 验证文件
    validate_files(config_files, csv_files)
    
    # 打印解析结果
    print("\n解析结果:")
    print(f"配置文件: {config_files}")
    print(f"CSV文件: {csv_files}")
    print(f"标签: {labels}")
    
    return config_files, csv_files, labels

# ===================================================================
# 读取json配置文件
# ===================================================================
def validate_config(para):
    """Validate physical input at the JSON boundary before mutating species arrays."""
    if para['nspec'] != 2:
        raise ValueError("当前剖面接口支持恰好两种离子（nspec=2）")
    for name, minimum in [('nptu', 3), ('legmod', 2)]:
        if not isinstance(para[name], int) or isinstance(para[name], bool) or para[name] < minimum:
            raise ValueError(f"{name} 必须是至少 {minimum} 的整数")
    if para['ipheat'] not in (1, 2) or para['ispec'] not in (1, 2):
        raise ValueError("ipheat 和 ispec 只支持 1 或 2")
    for name in ('ne', 'tempe', 'omega_ci'):
        if not np.isfinite(para[name]) or para[name] <= 0:
            raise ValueError(f"{name} 必须是有限正数")
    for name in ('pwdens', 'kperp'):
        if not np.isfinite(para[name]) or para[name] < 0:
            raise ValueError(f"{name} 必须是有限非负数")
    for name in ('azi', 'atm', 'aconc', 'tempi'):
        values = np.asarray(para[name], dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError(f"{name} 必须包含两种离子的有限值")
        if np.any(values < 0) or (name != 'aconc' and np.any(values == 0)):
            raise ValueError(f"{name} 含非物理负值或零值")
    if para['aconc'][para['ispec'] - 1] <= 0:
        raise ValueError("被求解物种的浓度必须大于零")
    if not np.isclose(np.dot(para['azi'], para['aconc']), 1.0, rtol=0, atol=1e-6):
        raise ValueError("aconc 定义为 ni/ne，必须满足准中性 Σ Zi*aconc_i = 1")
    nbi = para.get('nbi', {})
    count = nbi.get('Ns', para['Ns'])
    if not isinstance(count, int) or count < 1:
        raise ValueError("Ns 必须是正整数")
    energies = nbi.get('energies_keV')
    if energies is None:
        energies = [nbi.get('energy_base_keV', 90.0)]
    energies = np.asarray(energies, dtype=float)
    if energies.shape != (count,) or not np.isfinite(energies).all() or np.any(energies <= 0):
        raise ValueError("nbi.energies_keV 必须与 Ns 等长并全部为有限正数")
    for name, default in [('pwdens', 1.0), ('energy_spread_ratio', 0.01)]:
        value = nbi.get(name, default)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"nbi.{name} 必须是有限非负数")
    gaussian = nbi.get('gaussian', {})
    if (not -1 <= gaussian.get('mu_center', 0.7) <= 1
            or not np.isfinite(gaussian.get('sigma', 0.07)) or gaussian.get('sigma', 0.07) <= 0):
        raise ValueError("NBI mu_center 必须在 [-1,1]，sigma 必须大于零")
    for name in ('fnbi', 'ficrh'):
        if para[name] not in (0, 1):
            raise ValueError(f"{name} 必须为 0 或 1")
    if para.get('frequency_convention', 'rf_hz') not in ('rf_hz', 'cyclotron_hz', 'cyclotron_rad_s'):
        raise ValueError("frequency_convention 必须是 rf_hz、cyclotron_hz 或 cyclotron_rad_s")
    if para.get('rf_power_mode', 'absorbed') not in ('absorbed', 'maxwell_reference'):
        raise ValueError("rf_power_mode 必须是 absorbed 或 maxwell_reference")
    trapping = para.get('trapping', {})
    if trapping.get('model', 'symmetric') not in ('symmetric', 'directional'):
        raise ValueError("trapping.model 必须是 symmetric 或 directional")
    critical = trapping.get('ucr', 0.4)
    steepness = trapping.get('a', 20.0)
    lower = 0 if trapping.get('model', 'symmetric') == 'symmetric' else -1
    if not lower <= critical <= 1 or not np.isfinite(steepness) or steepness < 0:
        raise ValueError("trapping 的 ucr 或 a 不在支持范围内")
    for name, default in [('r_E_1', 200.0), ('r_E_2', 100.0)]:
        value = para.get('energy_range', {}).get(name, default)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"energy_range.{name} 必须为有限正数")
    if para.get('plot_settings', {}).get('type', 4) not in (0, 1, 3, 4):
        raise ValueError("plot_settings.type 支持 0、1、3、4；μ平均图尚未实现")
    if para.get('input_mode', {}).get('case_mode', 0) not in (0, 1):
        raise ValueError("input_mode.case_mode 必须是 0 或 1")
    radii = np.asarray(para.get('input_mode', {}).get('ipsii', [0.2667]), dtype=float)
    if (radii.ndim != 1 or len(radii) == 0 or not np.isfinite(radii).all()
            or np.any(radii < 0) or np.any(radii > 1) or np.any(np.diff(radii) <= 0)):
        raise ValueError("input_mode.ipsii 必须为 [0,1] 内递增的归一化小半径 rho")
    if para.get('input_mode', {}).get('case_mode', 0) == 0 and len(radii) != 1:
        raise ValueError("单径向点模式必须恰好提供一个 ipsii")
    if nbi.get('use_external_file', False):
        source = nbi.get('source_file')
        if not source or not os.path.isfile(source):
            raise ValueError("启用外部 NBI 源时必须提供存在的 source_file")


def read_config_file(config_file):
    """
    读取配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        dict: 配置文件内容
    """
    import json

    with open(config_file, 'r') as f:
        para = json.load(f)

    # External files are relative to the configuration, not the invocation cwd.
    if para.get('nbi', {}).get('use_external_file', False):
        for key in ('source_file', 'power_profile_file'):
            value = para['nbi'].get(key)
            if value:
                para['nbi'][key] = os.path.abspath(os.path.join(os.path.dirname(config_file), value))
    validate_config(para)

    # # 添加电子参数
    rmhme = 1836.15
    para['azi'].append(-1)
    para['atm'].append(1./rmhme)
    para['aconc'].append(1)
    para['tempi'].append(para['tempe'])
    
    # 提取参数
    nspec = para['nspec']
    nptu = para['nptu']
    ipheat = para['ipheat']
    legmod = para['legmod']
    tempi = para['tempi']
    azi = para['azi']
    atm = para['atm']
    aconc = para['aconc']
    dens = para['ne']
    kperp = para['kperp']
    omega_ci = para['omega_ci']
    power = para['pwdens']
    ispec = para['ispec'] - 1
    out = para['out']
    plot = para['plot']  # 设置绘图标志

    nbi_enabled = para['fnbi']      # NBI功能启用标志，1为开启
    icrh_enabled = para['ficrh']    # ICRH功能启用标志，1为开启
    Ns = para.get('nbi', {}).get('Ns', para['Ns'])
    # nbi_file = para.get('nbi_file', None)   # 从配置文件读取 NBI 文件路径，如果没写则默认为 None

    ctx = ExternalContext( nspec, tempi, azi, atm, aconc,dens )
    ssp = FP( ctx, nptu, ipheat, legmod, kperp, omega_ci, power, ispec, nbi_enabled, icrh_enabled, Ns )

    return ctx, ssp, out, plot, para

# ===================================================================
# 从 JSON 配置应用运行设置
# ===================================================================
def apply_config_settings(ssp, para):
    """
    根据 JSON 配置字典设置求解器运行参数（NBI / 环向捕获 / 能量范围 / 绘图 / 输入模式）。

    说明：所有 .get(key, default) 中的 default 仅在 JSON 缺少该键时生效，
    JSON 中写了什么值就使用什么值。

    Args:
        ssp: FP 求解器实例
        para: read_config_file 读取的 JSON 配置字典

    Returns:
        case_mode (int): 0 单径向点，1 多径向点
        ipsii (list): 径向位置列表（多径向点模式下使用）
        run_modes (list): 运行的加热模式列表
    """
    ssp.frequency_convention = para.get('frequency_convention', 'rf_hz')
    ssp.rf_power_mode = para.get('rf_power_mode', 'absorbed')
    ssp.plot_enabled = bool(para.get('plot', True))
    # ---- 1. NBI 设置 ----
    nbi_cfg = para.get('nbi', {})
    ssp.pwdens_nbi = nbi_cfg.get('pwdens', 1.0)          # NBI功率密度，单位W/cm^3
    ssp.Ns = nbi_cfg.get('Ns', ssp.Ns)                   # NBI模型光束数量

    energies = nbi_cfg.get('energies_keV', None)         # 显式能量列表，null 时用基础能量
    if energies is None:
        energies = [nbi_cfg.get('energy_base_keV', 90.0)]
    ssp.nbi_energies_keV = energies                      # NBI光束能量，单位keV
    ssp.nbi_energy_spread_ratio = nbi_cfg.get('energy_spread_ratio', 0.01)  # 物理能量展宽
    ssp.nbi_range_mu = 0                                       # 固定为完整区间 [-1,1]
    ssp.nbi_mu_points = nbi_cfg.get('mu_points', 50)     # 仅当使用真实数据文件时生效

    if nbi_cfg.get('use_external_file', False):
        # 开启外部输入文件：使用外部 NBI 源分布文件
        ssp.nbi_file = nbi_cfg.get('source_file', None)          # NBI功率沉积文件
        ssp.nbi_power_file = nbi_cfg.get('power_profile_file', None)  # 功率沉积剖面文件(全径向点)
        print(f"NBI源：外部输入文件 {ssp.nbi_file}")
    else:
        # 关闭外部输入文件：使用高斯分布设置
        ssp.nbi_file = None
        ssp.nbi_power_file = None
        gauss = nbi_cfg.get('gaussian', {})
        ssp.nbi_mu_center = gauss.get('mu_center', 0.7)  # 中心俯仰角 (0纯垂直, 接近1纯平行)
        ssp.nbi_sigma = gauss.get('sigma', 0.07)         # 束的角向宽度
        print(f"NBI源：高斯分布 mu_center={ssp.nbi_mu_center}, sigma={ssp.nbi_sigma}")

    # ---- 2. 环向捕获效应 ----
    trapping_cfg = para.get('trapping', {})
    ssp.enable_trapping = trapping_cfg.get('enabled', 1)  # 1：开启 0：关闭
    ssp.trapping_ucr = trapping_cfg.get('ucr', 0.4)       # Sigmoid 权重的临界俯仰角 uc
    ssp.trapping_a = trapping_cfg.get('a', 20.0)          # Sigmoid 权重的陡度参数 a
    ssp.trapping_model = trapping_cfg.get('model', 'symmetric')

    # ---- 3. 能量范围系数 ----
    er_cfg = para.get('energy_range', {})
    ssp.r_E_1 = er_cfg.get('r_E_1', 200.0)   # 基频能量范围系数
    ssp.r_E_2 = er_cfg.get('r_E_2', 100.0)   # 谐频能量范围系数

    # ---- 4. 绘图设置 ----
    plot_cfg = para.get('plot_settings', {})
    ssp.plot_range_k = plot_cfg.get('range_k', 0)        # 1为限定范围，0为全范围
    ssp.plot_type = plot_cfg.get('type', 4)              # 0:f(v,u) 1:热图 2:μ平均 3:f(v∥,v⊥) 4:绘制3个图
    ssp.plot_range_mu = plot_cfg.get('range_mu', 0)      # 0: 完整区间(负区间虚线) 1: 仅正区间 -1: 仅负区间
    ssp.xlim_max = plot_cfg.get('xlim_max', 200)         # 限定范围x轴最大值，单位keV
    ssp.ylim_max = plot_cfg.get('ylim_max', -10)         # 限定范围y轴最大值，单位log10(f)
    ssp.f_max = plot_cfg.get('f_max', 1e-1)              # 数据最大阈值
    ssp.f_min = plot_cfg.get('f_min', 1e-7)              # 数据最小阈值
    ssp.custom_mu_values = plot_cfg.get('custom_mu_values', None)  # f(v,u)图自定义pitch angle列表，null时用duu步长
    ssp.duu = plot_cfg.get('duu', 0.1)                   # pitch angle步长，custom_mu_values为null时生效

    # ---- 5. 输入模式设置 ----
    mode_cfg = para.get('input_mode', {})
    case_mode = mode_cfg.get('case_mode', 0)             # 0为单径向点，1为多径向点
    ssp.case_mode = case_mode
    ipsii = mode_cfg.get('ipsii', [0.2667])              # 特定径向位置列表

    # ---- 6. 运行的加热模式（根据 fnbi / ficrh 开关自动派生） ----
    # fnbi=1 且 ficrh=1 -> NBI、ICRH、Synergy 三种都运行
    # 仅 fnbi=1        -> 只运行 NBI；仅 ficrh=1 -> 只运行 ICRH
    run_modes = []
    if ssp.nbi_enabled:
        run_modes.append('NBI')
    if ssp.icrh_enabled:
        run_modes.append('ICRH')
    if ssp.nbi_enabled and ssp.icrh_enabled:
        run_modes.append('Synergy')
    if not run_modes:
        print("⚠️ 警告：fnbi 和 ficrh 均为 0，没有可运行的加热模式！")
    mode_msg = f"运行模式（由 fnbi={ssp.nbi_enabled}, ficrh={ssp.icrh_enabled} 派生）: {run_modes}"
    print(mode_msg)
    ssp.run_modes = run_modes  # 保存到求解器实例，t_FP 会写入输出日志文件

    return case_mode, ipsii, run_modes

# def read_config_files(config_files, csv_files, labels):
#     """
#     读取配置文件和CSV文件

#     Args:
#         config_files: 配置文件列表
#         csv_files: CSV文件列表
#         labels: 标签列表
#     """
#     # 读取配置文件
#     for config_file in config_files:
#         ctx, ssp, out, plot = read_config_file(config_file)

# ===================================================================
# 读取txt文件
# ===================================================================
def read_txt_file(file_path, skip_rows=1):
    """读取文件，返回 {位置(浮点): 数据} 的字典"""
    data_map = {}
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return None

    def process_lines(lines):
        for line in lines[skip_rows:]:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data_map[float(parts[0])] = parts[1]
                except ValueError:
                    continue

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            process_lines(f.readlines())
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                process_lines(f.readlines())
        except Exception as e:
            print(f"读取失败: {e}")
            return None
    return data_map

def check_uniform_interval(data_map):
    """检查是否等间隔，返回 (是否等间隔, 间隔大小)"""
    if not data_map or len(data_map) < 2:
        return False, 0.0
    positions = sorted(data_map.keys())
    first_interval = positions[1] - positions[0]
    for i in range(len(positions) - 1):
        if abs((positions[i+1] - positions[i]) - first_interval) > 1e-4:
            return False, 0.0
    return True, first_interval

# ===================================================================
# 确定径向位置
# ===================================================================
def process_files_to_arrays(input_paths, custom_names):
    """
    批量处理文件，以间隔最大的为基准，返回基准信息和自定义命名的数据数组。
    静默模式，不打印中间过程。
    """
    
    if len(input_paths) != len(custom_names):
        print("错误：输入文件列表和输出名字列表长度不一致！")
        return None, None

    file_data_cache = {} 
    max_interval = -1.0
    base_index = -1
    
    # --- 第一步：静默扫描 ---
    for i, path in enumerate(input_paths):
        data = read_txt_file(path)
        if not data:
            continue
            
        is_uniform, interval = check_uniform_interval(data)
        if not is_uniform:
            continue
            
        file_data_cache[i] = (data, interval)
        
        if interval > max_interval:
            max_interval = interval
            base_index = i

    if base_index == -1:
        print("错误：未找到有效的等间隔文件。")
        return None, None, None

    # --- 第二步：构建基准信息 ---
    base_data, _ = file_data_cache[base_index]
    base_positions = sorted(base_data.keys())
    base_values = [base_data[pos] for pos in base_positions]
    
    base_info = {
        "filename": os.path.basename(input_paths[base_index]),
        "min_pos": base_positions[0],
        "max_pos": base_positions[-1],
        "count": len(base_positions),
        "interval": max_interval
    }
    
    # --- 第三步：对齐数据 ---
    data_arrays = {}
    
    for i, path in enumerate(input_paths):
        if i not in file_data_cache:
            continue
            
        name = custom_names[i]
        target_data, _ = file_data_cache[i]
        
        aligned_values = []
        for pos in base_positions:
            val = target_data.get(pos)
            aligned_values.append(val if val is not None else None)
            
        data_arrays[name] = aligned_values

    print(f"\n[基准文件]: {base_info['filename']}")
    print(f"  范围: {base_info['min_pos']} 到 {base_info['max_pos']}")
    print(f"  总数: {base_info['count']}")
    print(f"  间隔: {base_info['interval']}")

    return base_info, base_positions, data_arrays

# ===================================================================
# 输出信息到文件
# ===================================================================
def append_to_file(filename, content):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(content)


# ===================================================================
# 输出径向归一化密度分布图
# ===================================================================
def plot_radial_distribution_with_contours(ssp, mode='NBI',
                                         show_nbi_contour=True,
                                         show_nbi_marker=True,
                                         show_icrh_contour=True,
                                         show_icrh_marker=True,
                                         highlight_nbi_peak=True):
    """Plot the physical dn/(dE dmu) spectrum without changing its amplitude.

    highlight_nbi_peak adds an energy band only; it never multiplies the
    spectrum. Negative values are retained in data and masked only for LogNorm.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    spectrum, energy = ssp.integrate_over_radius()
    if spectrum is None:
        return fig, ax
    positive = spectrum[np.isfinite(spectrum) & (spectrum > 0)]
    vmax = float(positive.max()) if positive.size else 1.0
    vmin = vmax / 1e8
    display = np.ma.masked_where(~np.isfinite(spectrum) | (spectrum <= 0), spectrum)
    pcm = ax.pcolormesh(energy, ssp.xmu, display.T, cmap='inferno', shading='auto',
                        norm=LogNorm(vmin=vmin, vmax=vmax))
    colorbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    colorbar.set_label(r'$dn/(dE\,d\mu)$ [m$^{-3}$ keV$^{-1}$]', fontsize=14)
    ax.set(xlabel='E [keV]', ylabel=r'$\mu$', ylim=(ssp.mu_min, ssp.mu_max))
    ax.set_title(f'{mode}: {ssp.radial_weight_model}', fontsize=11)

    if not hasattr(ssp, 'radial_spectra_by_mode'):
        ssp.radial_spectra_by_mode = {}
    ssp.radial_spectra_by_mode[mode] = (energy.copy(), ssp.xmu.copy(), spectrum.copy())
    if mode in ('NBI', 'ICRH'):
        setattr(ssp, f'{mode.lower()}_f_processed', spectrum.copy())

    beams = getattr(ssp, 'nbi_energies_keV', None)
    if beams is not None:
        for beam in beams:
            ax.axvline(beam, color='cyan', linestyle='--', alpha=0.4, linewidth=1)
        if highlight_nbi_peak and len(beams):
            beam = max(beams)
            ax.axvspan(0.8 * beam, 1.2 * beam, facecolor='none', edgecolor='cyan',
                       linewidth=1, linestyle=':', label='NBI energy band')

    overlay_options = [('NBI', show_nbi_contour, show_nbi_marker, 'cyan', 'o'),
                       ('ICRH', show_icrh_contour, show_icrh_marker, 'lime', '*')]
    for overlay_mode, show_contour, show_marker, color, marker in overlay_options:
        if mode != 'Synergy' and mode != overlay_mode:
            continue
        stored = ssp.radial_spectra_by_mode.get(overlay_mode)
        if stored is None or not (show_contour or show_marker):
            continue
        overlay_energy, overlay_mu, overlay = stored
        if not np.nanmin(overlay) < vmin < np.nanmax(overlay):
            continue
        contours = ax.contour(overlay_energy, overlay_mu, overlay.T, levels=[vmin],
                              colors=[color], linewidths=1.5,
                              alpha=0.7 if show_contour else 0.0)
        vertices = [path.vertices for path in contours.get_paths() if path.vertices.size]
        if show_marker and vertices:
            points = np.concatenate(vertices)
            point = points[np.argmax(points[:, 0])]
            ax.plot(point[0], point[1], marker=marker, color=color, markeredgecolor='white',
                    markersize=10, label=f'{overlay_mode} spectrum contour end')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=10)
    path = os.path.join(ssp.current_out_dir, f'{mode}_Integrated_2D_Heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


# ===================================================================
# 在主程序中调用这个函数，绘制不同模式下径向归一化的密度分布图
# ===================================================================
def plot_all_modes(ssp, run_folder, show_nbi_in_synergy=True, show_icrh_in_synergy=True):
    """Reject the legacy placeholder loader rather than relabeling one solve three times."""
    raise NotImplementedError(
        "plot_all_modes has no implemented result loader. Use run_selected_modes "
        "to solve and plot each mode from its own distribution."
    )

# ===================================================================
# 模式选择
# ===================================================================
def run_selected_modes(ssp, run_folder, base_filenames, paths, nrdpsi, initial_power,
                       active_modes=None,
                       timing_file=None,
                       start_time=None):
    """
    自主选择并运行 NBI / ICRH / Synergy 模式。
    
    参数:
        active_modes: 列表，如 ['NBI', 'Synergy']。默认 None 运行全部三种。
        timing_file: 外部传入的计时文件路径。为 None 时内部自动创建。
        start_time: 外部传入的起始时间戳。为 None 时以函数被调用时刻为准。
    """
    if active_modes is None:
        active_modes = ['NBI', 'ICRH', 'Synergy']

    MODE_CONFIG = {
        'NBI': {
            'title': 'NBI Mode Timing Log',
            'print_msg': '运行单独NBI模式',
            'folder': '1_NBI_only',
            'prefix': 'NBI',
            'plot_mode': 'NBI',
            'enable_icrh': 0,
            'enable_nbi': 1,
            'backup_nbi': True,
            'plot_kwargs': {
                'show_nbi_contour': False,
                'show_nbi_marker': False,
                'show_icrh_contour': False,
                'show_icrh_marker': False,
                'highlight_nbi_peak': False,
            }
        },
        'ICRH': {
            'title': 'ICRH Mode Timing Log',
            'print_msg': '运行单独ICRH模式',
            'folder': '2_ICRH_only',
            'prefix': 'ICRH',
            'plot_mode': 'ICRH',
            'enable_icrh': 1,
            'enable_nbi': 0,
            'backup_nbi': False,
            'plot_kwargs': {
                'show_nbi_contour': False,
                'show_nbi_marker': False,
                'show_icrh_contour': False,
                'show_icrh_marker': False,
                'highlight_nbi_peak': False,
            }
        },
        'Synergy': {
            'title': 'Synergy Mode Timing Log',
            'print_msg': '运行NBI+ICRF协同模式',
            'folder': '3_Synergy',
            'prefix': 'Synergy',
            'plot_mode': 'Synergy',
            'enable_icrh': 1,
            'enable_nbi': 1,
            'backup_nbi': True,
            'plot_kwargs': {
                'show_nbi_contour': False,
                'show_nbi_marker': False,
                'show_icrh_contour': False,
                'show_icrh_marker': False,
                'highlight_nbi_peak': False,
            }
        },
    }

    # 若外部未传入，则内部创建
    if timing_file is None:
        timing_file = os.path.join(run_folder, "timing_log.txt")
        with open(timing_file, 'w') as f:
            f.write("Timing Log\n")
            f.write("===================\n")

    saved_figures = []
    last_time = time.time()

    # 初始化调试信息汇总文件（结果根目录，按模式分区记录）
    debug_file = os.path.join(run_folder, 'debug.txt')
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*20} 调试信息汇总 {'='*20}\n")

    for mode_name in active_modes:
        if mode_name not in MODE_CONFIG:
            raise ValueError(f"未知模式: {mode_name}，可选: {list(MODE_CONFIG.keys())}")

        cfg = MODE_CONFIG[mode_name]

        with open(timing_file, 'a') as f:
            f.write(f"\n{cfg['title']}\n")
            f.write("=" * 20 + "\n")

        print("\n" + "=" * 20 + f" {cfg['print_msg']} " + "=" * 20)

        # 重置状态
        ssp.icrh_enabled = cfg['enable_icrh']
        ssp.nbi_enabled = cfg['enable_nbi']
        ssp.power = initial_power
        ssp.ibreak = 0
        ssp.fb1_prv = 0.0
        ssp.last_saved_figure = None

        # 创建子文件夹
        mode_folder = os.path.join(run_folder, cfg['folder'])
        os.makedirs(mode_folder, exist_ok=True)

        ssp.output_paths = {
            k: os.path.join(mode_folder, v)
            for k, v in base_filenames.items()
        }
        ssp.current_out_dir = mode_folder
        ssp.out = os.path.join(mode_folder, cfg['prefix'])

        # 备份 NBI 输入文件
        if cfg['backup_nbi'] and ssp.nbi_enabled:
            for path in paths:
                if os.path.exists(path):
                    shutil.copy2(path, os.path.join(mode_folder, os.path.basename(path)))

        # 求解器
        solver_start = time.time()
        ssp.t_FP(ssp.power, nrdpsi, ssp.legmod, ssp.nptu, joutql_in=0)
        solver_time = time.time() - solver_start

        with open(timing_file, 'a') as f:
            f.write(f"{mode_name}模式求解器耗时: {solver_time:.4f} 秒\n")

        if ssp.last_saved_figure:
            saved_figures.append(ssp.last_saved_figure)

        # 绘图
        if ssp.case_mode:
            plot_start = time.time()
            plot_radial_distribution_with_contours(
                ssp,
                mode=cfg['plot_mode'],
                **cfg['plot_kwargs']
            )
            plot_time = time.time() - plot_start

            with open(timing_file, 'a') as f:
                f.write(f"{mode_name}模式计算径向归一化密度分布耗时: {plot_time:.4f} 秒\n")

        total_time = time.time() - last_time
        with open(timing_file, 'a') as f:
            f.write(f"{mode_name}模式任务执行耗时: {total_time:.4f} 秒\n")

        last_time = time.time()

    # ================= 最终汇总输出 =================
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"🎉 所有模拟运行完毕！结果已保存在文件夹: {run_folder}")
    print("=" * 60)
    print("📊 生成的结果图片汇总如下：")
    if saved_figures:
        for i, fig_name in enumerate(saved_figures, 1):
            print(f"  [{i}] {fig_name}")
    else:
        print("  未生成任何图片。")
    print("=" * 60)

    with open(timing_file, 'a') as f:
        f.write("\n执行时间总结\n")
        f.write("======================\n")
        f.write(f"总执行时间: {total_elapsed:.4f} 秒\n")

    print(f"Execution finished in {total_elapsed:.4f} seconds.")

    return saved_figures, total_elapsed

# ===================================================================
# 主程序
# ===================================================================

if __name__ == "__main__":
    start_time = time.time()
    print("main start")

    # ============ 创建带时间戳的结果文件夹 ============
    # 获取当前时间，格式化为 YYYY-MM-DD_HH-MM-SS
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 文件夹名加上运行模式标识，更清晰
    run_folder = f"FPQL_Results_{current_time}"
    
    # 如果文件夹不存在，则创建
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        print(f"🎉 已创建本次运行的结果文件夹: {run_folder}")
    else:
        print(f"文件夹已存在: {run_folder}")
    # ======================================================

    # ============ 2. 配置文件并读取 ============
    print(f"开始配置文件...")
    config_files, csv_files, labels = config_input_files()
    
    # 将输入配置文件原封不动地复制到结果文件夹中
    for config_file in config_files:
        if os.path.exists(config_file):
            # 构建目标路径：保存到结果文件夹的根目录
            dest_path = os.path.join(run_folder, f"{os.path.basename(config_file)}")
            try:
                shutil.copy2(config_file, dest_path)
                print(f"✅ 输入配置已备份: {dest_path}")
            except Exception as e:
                print(f"⚠️ 备份配置文件 {config_file} 失败: {e}")
        else:
            print(f"⚠️ 配置文件 {config_file} 不存在，无法备份")


    n = 1  # 第n个配置文件
    config_file = config_files[n-1]  # 获取第n个配置文件
    
    # 2. 读取输入文件，实例化求解器
    print(f"开始读取输入文件...")
    ctx, ssp, out, plot, para = read_config_file(config_file)

    # 准备径向点数、温度、密度、功率数据
    paths = [
        'case.txt',
    'Init_N_i1.txt', 
    'Init_N_e.txt', 
    'Init_N_i2.txt', 
    'Init_T_i1.txt', 
    'Init_T_e.txt', 
    'Init_T_i2.txt'

    ]

    names = [
    'P',
    'N0',
    'Ne',
    'N1',
    'T0',
    'Te',
    'T1'
    ]

    # 输出文件列表
    file_list = [
        os.path.join(run_folder, '01_read.txt'),
        # os.path.join(run_folder, 'case2_read.txt'),
        # os.path.join(run_folder, 'case1(2)_read.txt'),

        os.path.join(run_folder, '01_case.txt')
        # os.path.join(run_folder, 'case2.txt'),
        # os.path.join(run_folder, 'case1(2).txt')

    ]
    
    # ========================================================
    # 从 JSON 配置文件读取运行设置（NBI / 环向捕获 / 绘图 / 输入模式）
    # ========================================================
    case_mode, ipsii, run_modes = apply_config_settings(ssp, para)

    if case_mode == 0:
        ipsiii = [1]  # 单点数组索引与输入的物理 rho 分开。
        nrdpsi = 1
        ssp.nrdpsi = nrdpsi
        ssp.info = 1
        ssp.positions = np.array([ipsii[0]], dtype=np.float64)
        ssp.arrays = 1
        ssp.file_list = file_list
        ssp.ipsii = ipsii
        ssp.ipsiii = ipsiii
        print(f"单径向点模式，ipsii: {ipsii}")
    elif case_mode == 1:
        # 调用处理函数
        info, positions, arrays = process_files_to_arrays(paths, names)
        interval = info['interval']
        min_pos = info['min_pos']
        max_pos = info['max_pos']
        ipsiii = []
        for i in ipsii:
            # 越界检查
            if not (min_pos <= i <= max_pos):
                ipsiii.append(None)
                continue
            
            # 先归一化到 0-based，再加 1 转为 1-based，加上 interval / 2 是为了消除浮点数精度误差带来的整除偏移问题
            idx_0based = (i - min_pos + interval / 2) // interval
            idx_1based = idx_0based + 1
            ipsiii.append(int(idx_1based))
            print(f'{ipsiii}={idx_0based}+1=({i} - {min_pos}) / {interval} + 1')    
        # ipsiii = [int(i / 0.02) for i in ipsii]
        # ipsiii = [int(i / info['interval']) for i in ipsii]
        nrdpsi = info['count']
        ssp.nrdpsi = nrdpsi
        ssp.info = info
        ssp.positions = positions
        ssp.arrays = arrays
        ssp.file_list = file_list
        ssp.ipsii = ipsii
        ssp.ipsiii = ipsiii
        print(f"多径向点模式，ipsii: {ipsii}")

    # 3. 运行求解器求解
    print(f"开始运行求解器...")
    # start_time = time.time()
    # 在主程序中
    print(f"参考碰撞频率， anuref = {ssp.anuref:.4e}")

    
    # 确保文件夹存在
    os.makedirs(run_folder, exist_ok=True)

    base_prefix = os.path.basename(out) if out else "result"
            
    # 定义一个基础的文件名字典 (只定义文件名，不包含路径)
    base_filenames = {
        "read": "01_read.txt",
        "case": "01_case.txt",
        "qleg00": "matrix_qleg00.txt",
        "A_full": "matrix_A_full.txt",
        "A_banded": "matrix_A_banded.txt",
        "b_vector": "vector_b.txt",
        "A_out": "matrix_A_out.txt",
        "b_out": "vector_b_out.txt",
        "Pn": "legendre_Pn.txt",
        "Fv": "distribution_Fv.txt",
        "Fi": "distribution_Fi.txt",
        "name": base_prefix
    }

    initial_power = ssp.power 

    saved_figs, total_time = run_selected_modes(
        ssp, run_folder, base_filenames, paths, nrdpsi, initial_power,
        active_modes=run_modes,                        # ← 在JSON的run_modes中勾选模式
        timing_file=os.path.join(run_folder, "timing_log.txt"),
        start_time=start_time                      # ← 继承主程序开头的 start_time
    )
