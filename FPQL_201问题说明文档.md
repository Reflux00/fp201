# FPQL_201 问题说明文档

---
## 问题
问题 1：NBI 功率增加但 tfecmu 影响较小
问题 2：固定 μcr 但峰值位置随功率移动

## 排查工具

每次运行结果根目录生成 `debug.txt`（与 `int_1.json` 备份、`timing_log.txt` 同目录），按模式分区记录：

- **输入信息**：NBI 功率 pwdens_nbi、束能量 energies_keV；ICRH 输入功率 pwdens、trapping 参数（enabled / ucr / a）
- **输出信息**：每个径向点的分布能量峰值特征（char_E、char_mu）；NBI 模式下束能量 ±5% 窗口内 tfecmu 的最大值及对应 E、μ

对应关系：

- **问题 1** → 用"束能量附近 tfecmu 最大值 vs pwdens_nbi"定量追踪峰值响应
- **问题 2** → 用"char_mu vs pwdens（ICRH/协同）"追踪峰值位置移动，配合 `trapping.ucr` 做对照扫描

## 参数入口（int_1.json）

| 问题 | 相关 JSON 参数 |
| --- | --- |
| 问题 1 | `nbi.pwdens`、`nbi.energy_base_keV`、`nbi.energy_spread_ratio`、`nbi.gaussian.mu_center/sigma`、`nptu`（网格分辨率） |
| 问题 2 | `pwdens`（ICRH 功率）、`trapping.enabled`、`trapping.ucr`、`trapping.a`、`plot_settings.f_min`（尾端阈值） |

---





## 问题 1：NBI 功率增加但 tfecmu 影响较小

### 问题说明

- **观察方式**：`plot_settings.type = 0`，绘制 f(E, μ) 曲线图
- **理论预期**：pwdens_nbi ↑ → norm_coeff ↑ → sig ↑ → psource ↑ → 矩阵右端项 ↑ → tfecmu 在 E_nbi 处峰值显著增大
- **实际现象**：pwdens_nbi ↑，但 f(E, μ) 曲线在 E_nbi 处峰值变化不明显

### 逻辑链条

```
pwdens_nbi ↑
└──→ norm_coeff ↑（线性）→ sig ↑（线性）→ psource[n] ↑（线性）→ 矩阵右端项 ↑ → f(E, μ)
```

### 相关代码逻辑

**功率归一化（inicol）**：

```
pwnbi = Σ E_nbi_k[k] × sig[k,0]

norm_coeff = pwdens_nbi / pwnbi
sig = sig × norm_coeff
dnbi = 1.0
```

参数说明：

- `E_nbi_k[k]`：第 k 束的束能量，等于 `nbi_energies_keV[k]`，为固定值
- `sig[k,0]`：(1/2) ∫ S(μ) dμ，由源分布决定，与 pwdens_nbi 无关
- `pwdens_nbi`：NBI 目标功率密度，对应 JSON `nbi.pwdens`
- `norm_coeff`：归一化系数，pwnbi 固定 → norm_coeff ∝ pwdens_nbi → sig ∝ pwdens_nbi，链路严格线性
- `dnbi`：NBI 源项缩放因子，固定为 1.0

**源项构成（source）**：

```
psource[0] = psourc + source_nbi[0] × dnbi
```

参数说明：

- `psourc`：背景碰撞项
- `source_nbi[0]`：NBI 注入项（勒让德 0 阶分量）

**束能量高斯展宽（source）**：

```
σ_v = max(σ_v_physical, 1.5 × du)
σ_v_physical = σ_E / (2 × v_nbi × T_local)
delta_v = exp(-(u-v_nbi)² / (2σ_v²)) / (σ_v √(2π))
```

参数说明：

- `σ_E`：物理能量展宽，来自 JSON `nbi.energy_spread_ratio`（比例 × 束能量）
- `du`：速度网格步长，由 `nptu` 决定
- `v_nbi`：束能量对应的归一化速度
- `T_local`：当地离子温度

---

## 问题 2：固定 μcr 但峰值位置随功率移动

### 问题说明

- **观察方式**：`plot_settings.type = 1`，绘制 f(E, μ) 热力图
- **理论预期**：固定 ucr → W(μ) 捕获边界不变 → 功率 ↑ 只增强扩散幅度 → 热力图峰值 μ 位置不变
- **实际现象**：功率 ↑ → 峰值 μ 位置向 μ = 0 移动

### 逻辑链条

```
固定 ucr
│
├──→ W(μ) 形状固定 → qleg 各向异性固定 → 捕获边界固定
│
└──→ pwdens ↑ → dqlw ↑ → 整体加热增强 → 热力图幅度增大 → 峰值 μ 位置（理论不变，实际向 μ=0 移动）
```

### 相关代码逻辑

**捕获效应权重（t_FP → coelgd_with_weight_optimized）**：

```python
if self.enable_trapping:
    self.coelgd_with_weight_optimized(enable_weight=True, uc=self.trapping_ucr, a=self.trapping_a)
```

```
W(μ) = 1 / (1 + exp(-a × (μ - ucr)))
```

参数说明：

- `enable_trapping`：环向捕获效应开关，对应 JSON `trapping.enabled`
- `ucr`：Sigmoid 权重的临界俯仰角，对应 JSON `trapping.ucr`
- `a`：Sigmoid 权重的陡度参数，对应 JSON `trapping.a`
- `W(μ)`：捕获效应权重函数，形状由 ucr、a 决定，不随功率变化

**功率对扩散的影响（iniqln）**：

```
zdqlw = pwdens / zemxw
dqlw  = zdqlw / qlnorm
```

参数说明：

- `pwdens`：ICRH 吸收功率，取自 pwredf / pwredh（JSON 顶层 `pwdens` 为输入值）
- `zemxw`：归一化系数
- `qlnorm`：准线性算子归一化分母
- `dqlw`：准线性扩散系数幅度，只随功率变化；`icrh_enabled = 0` 时强制为 0

**扩散系数合成（qlcoef）**：

```
dql00/01/11 = dqlw × Σ qleg × dqlu
```

参数说明：

- `dqlw`：功率幅度因子
- `qleg`：勒让德积分（含捕获权重 W(μ)，ucr 固定 → 各向异性结构固定）
- `dqlu`：μ 方向网格权重

---
