# FPQL_201 程序使用说明

Fokker-Planck 准线性求解程序（NBI / ICRH 及协同加热版）

## 使用方法

定位到程序所在目录，执行 `python FPQL_201.py`，程序默认读取当前目录下的 `int_1.json`。运行结果保存在自动创建的时间戳文件夹 `FPQL_Results_YYYY-MM-DD_HH-MM-SS` 中，输入配置会自动备份。

## 输入参数说明

> 注意：以下注释仅用于说明，实际 JSON 文件中不能包含 `//` 注释。

```jsonc
{
    "nptu": 301,                // 速度网格点数
    "legmod": 61,               // 勒让德展开阶数
    "energy_range": {
        "r_E_1": 200.0,         // 基频能量范围系数（乘以温度得到能量范围）
        "r_E_2": 135.0          // 谐频能量范围系数；Ti=4 keV时Emax=540 keV
    },

    "ipheat": 2,                // 加热模式：1 基频加热，2 谐频加热
    "pwdens": 0.2,              // ICRH 最终吸收功率密度（单位：W/cm³）
    "rf_power_mode": "absorbed", // absorbed：固定最终吸收；maxwell_reference：参考Maxwell标定
    "omega_ci": 36.5e6,         // 波频率（单位：Hz）
    "frequency_convention": "rf_hz", // 也支持cyclotron_hz、cyclotron_rad_s
    "kperp": 10,                // 垂直波数 k⊥，单位m^-1

    "ne": 5.5,                  // 电子密度（单位：10¹⁹ m⁻³）
    "tempe": 4.0,               // 电子温度（单位：keV）
    "nspec": 2,                 // 当前剖面接口固定支持两种离子
    "ispec": 1,                 // 被求解及注入粒子：1 第一种离子，2 第二种离子
    "azi": [1.0, 1.0],          // 离子电荷数（以电子电荷归一化）
    "atm": [2, 1],              // 离子质量数（以质子质量归一化）
    "aconc": [0.97, 0.03],      // ni/ne；须满足ΣZi*aconc_i=1
    "tempi": [4.0, 4.0],        // 离子温度（单位：keV）
    "out": "ASDEX_NBI_D_H_17245",  // 输出文件名称前缀
    "plot": true,               // 是否输出分布函数图像

    "fnbi": 1,                  // NBI 功能开关：1 开启，0 关闭
    "ficrh": 1,                 // ICRH 功能开关：1 开启，0 关闭
    // 运行模式由 fnbi/ficrh 自动派生：都开= NBI+ICRH+协同；只开一个=只运行该模式
    // 派生结果写入结果文件夹的 01_read.txt 日志

    "Ns": 1,                    // NBI 模型光束数量

    "nbi": {
        "pwdens": 1.0,          // NBI 功率密度（单位：W/cm³）
        "energy_base_keV": 90.0,   // NBI 基础束能量（单位：keV）
        "energies_keV": null,   // 显式束能量列表；null 时自动使用 [energy_base_keV]
        "use_external_file": false,  // NBI 源选择：true 用外部文件，false 用高斯分布
        "source_file": "AUG_17245_NBI.txt",  // NBI 功率沉积文件（use_external_file 为 true 时生效）
        "power_profile_file": "TRANSO_Init_P_NBI_Envelope.txt",  // 功率沉积剖面文件，全径向点（可选）
        "energy_spread_ratio": 0.01,  // 物理能量展宽比例（0.01 即 1%）
        "mu_points": 50,        // 俯仰角方向点数（仅使用真实数据文件时生效）

        "gaussian": {           // 高斯分布设置（use_external_file 为 false 时生效）
            "mu_center": 0.7,   // 中心俯仰角：0 纯垂直注入，接近 1 纯平行注入
            "sigma": 0.07       // 角向宽度：越大越弥散，越小越集中
        }
    },

    "trapping": {
        "enabled": 1,           // 经验角向权重开关；不代表完整反弹轨道模型
        "model": "symmetric",  // symmetric依赖|mu|；directional显式保留旧单侧权重
        "ucr": 0.4,             // 捕获效应 Sigmoid 权重的临界俯仰角
        "a": 20.0               // 捕获效应 Sigmoid 权重的陡度参数，越大过渡越陡
    },

    "plot_settings": {
        "range_k": 0,           // 绘图范围：1 限定范围，0 全范围
        "type": 4,              // 0分布曲线，1热图，3平行/垂直速度图，4热图+曲线；2未实现且会报错
        "range_mu": 0,          // 俯仰角显示区间：0 完整区间（负区间虚线），1 仅正区间，-1 仅负区间
        "f_max": 1e-1,          // 分布函数数据最大阈值
        "f_min": 1e-7,          // 分布函数数据最小阈值
        "custom_mu_values": [0.0, 0.2, 0.5, 0.7, 1.0],  // f(v,u) 图自定义俯仰角列表；null 时按 duu 步长自动生成
        "duu": 0.1,             // 俯仰角步长（custom_mu_values 为 null 时生效）
        "xlim_max": 200,        // 限定范围 x 轴最大值（单位：keV）
        "ylim_max": -10         // 限定范围 y 轴最大值（单位：log₁₀(f)，负数）
    },

    "input_mode": {
        "case_mode": 0,         // 输入模式：0 单径向点，1 多径向点
        "ipsii": [0.2667]       // rho=r/a；单径向点使用唯一位置，多径向点为选定输出位置
    }
}
```

基础网格、物种和运行开关等字段仍为必填项；部分嵌套设置支持默认值。请以仓库中的 `int_1.json` 为完整配置示例。

所有径向输入使用同一个坐标 `rho=r/a∈[0,1]`（归一化小半径），包括 `input_mode.ipsii`、等离子体温度/密度/RF 功率剖面的首列、NBI 角向源的列坐标及 NBI 功率剖面的首列。不能混用归一化磁通、磁通平方根与小半径；需要使用实际平衡预先转换。程序不会再除以当前输入的最大半径重新归一化。

外部 NBI 角向源文件第一列是严格递增的 `mu∈[-1,1]`，其余列的表头必须显式给出严格递增的径向坐标，例如 `Mu rho_0.0 rho_0.5 rho_1.0`；数据应有限且非负。NBI 功率剖面第一列为严格递增的 rho、第二列为非负功率密度 W/cm³，首行为表头。两类文件均按实际 rho 插值，并须覆盖请求的位置；不会用数组行号代替位置、隐式外推或在输入失败后改用高斯源。外部文件路径相对于 JSON 配置文件所在目录解析。

## 修复后的计算与输出

采用恒密度总分布模型：NBI 注入物理粒子率为 `P/Ebeam`，并从 Maxwell 热背景移除相同数量的粒子。注入能量和热汇能量分别输出，例如默认 90 keV 束、Ti=4 keV、注入1 W/cm³，对应热汇0.0666667 W/cm³、净加热源0.9333333 W/cm³。热汇是维持固定物种密度的模型假设，不是原子电离势能损失；它不能替代真实轨道损失或密度演化。

求解通过齐次解幅度施加 `∫F d³u=1`，保持物理源强不变；不再固定原点值后缩放整个解。主流程中的 `fnorm` 应接近1。高功率/低密度时若各向同性分布显著为负，程序明确拒绝该解，提示检查分辨率与固定背景模型适用性。

NBI 能量高斯使用输入的真实标准差，独立投影到 Hermite 有限元，不再强制加宽到1.5个网格。零展宽采用精确δ源投影；非零展宽要求±8σ包含在能域内，否则明确报错。角高斯及外部角分布也使用稳定 Gauss 投影，不依赖绘图的101个μ点。

QL 使用完整非负 Bessel 平方核、稳定勒让德导数和收敛检查的 Gauss 角积分，消除固定6项截断。默认 `W(μ)=sigmoid(a*(|μ|-ucr))` 是对称经验权重；`trapping.model="directional"` 可保留旧单侧定义。两种模型的矩阵与加热诊断均使用同一权重，固定ucr仍不保证分布特征位置不变。

默认 `frequency_convention="rf_hz"` 表示 `omega_ci` 字段实际存RF频率Hz，回旋角频率按 `2πfRF/ipheat` 计算。`cyclotron_hz` 和 `cyclotron_rad_s` 分别表示离子回旋频率Hz和角频率rad/s。`rf_power_mode="absorbed"` 迭代扩散强度，使最终分布吸收功率匹配输入；`maxwell_reference` 仅在参考Maxwell分布上标定，此时最终吸收可以不同。功率为0保持0，负值会被拒绝。

分布与诊断输出：

- `case0_Fi_*.txt`：未经裁零或绘图处理的归一化分布，保留负值用于检查离散误差。
- 同名 `.npz`：保留 `raw_*` 和 `normalized_*` 兼容名称；主流程两者相同，均为满足单位粒子数约束的无量纲速度分布。另存密度、原点粒子、功率残差，NBI注入/热汇功率、实际RF吸收、平均能量、平均流速和扣流动后的平行温度。
- `debug.txt`：原始/归一化 NBI 束附近峰值、负值统计、尾端等值线阈值及能域截断标记。`char_mu` 是等值线延伸最远处的俯仰角，不是固定能量截面的分布峰位；若截断为 True，需要扩大能域并检查网格收敛后再比较位置。
- `01_read.txt`：实际矩阵源项的粒子矩、能量矩、线性方程残差和功率残差。碰撞与QL功率使用和矩阵相同的单元Gauss积分及Hermite分布。当前高速度端为自然零通量边界；小平衡残差不证明能域足够。
- `01_case.txt`：向所有离子、电子的碰撞转移功率及二阶矩等效温度；不再将 `2<E>/3` 错标为平均能量。
- 多径向输出 `12_energy_spectrum.txt`：所选物种的 `dn/(dE dμ)`，单位m⁻³ keV⁻¹／单位μ，使用 `π ni sqrt(E)/Ti^(3/2)` Jacobian。它包含热背景，不自动等于快离子分量。重映射保留原始速度数组，重复调用幂等。

径向积分可向 `integrate_over_radius(dvolume_dpsi=...)` 传入网格上的实际dV/dρ。未提供时明确使用柱坐标近似 `dV∝ρdρ`，不再自动读取硬编码eqdsk文件。等离子体密度剖面文件延续原输入单位10²⁰m⁻³，读入后乘10转换为内部10¹⁹m⁻³；温度为keV，RF功率密度为W/cm³。外部温度、密度、功率和准中性均在接收边界检查。

`distribution_checks_passed` 只报告显著负值、尾端截断及离散守恒检查，不能替代对每个新物理工况做独立网格/能域收敛。明显各向同性负值直接报错；角谱产生的小负值保留并诊断，不裁零。背景碰撞仍采用固定Maxwell的test-particle近似，没有完整波场、轨道或实验磁平衡自洽。

## 回归验证与 NBI 功率扫描

在已安装 NumPy、SciPy、Matplotlib 的 Python 环境中，从仓库根目录运行：

```sh
python -m unittest discover -s tests -v
python analysis/nbi_power_scan_probe.py --output-dir FPQL_Results_nbi_scan
```

扫描脚本也支持 `--config path/to/int_1.json`。扫描功率固定为 0.1、0.5、1、2、5、10 W/cm³，覆盖配置中的 NBI 功率和径向功率剖面；束能量、网格及物种参数来自所选配置，非等温背景源保持固定。输出目录中的 `scan.json` 和 `P-*.npz` 保存结果；不指定目录时使用系统临时目录。

扫描结果增加 `nonnegative_isotropic_component` 和 `model_failure`。高功率下出现负核心的线性解会被明确标为模型失效，不能用于物理比较；常规主流程直接拒绝此类结果。

`analysis/nbi_power_scan_results.json`、`analysis/physics_*_results.json` 中旧审计数据保留为历史证据，行号及数值对应修复前版本。修复后的验证见 [物理修复验证](FPQL_201物理修复验证.md)。旧功率目录没有覆盖，需要用新模型重新计算。
