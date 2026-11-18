# FPQL_201 程序使用说明

Fokker-Planck 准线性求解程序（NBI / ICRH 及协同加热版）

## 使用方法

定位到程序所在目录，执行 `python FPQL_201.py`，程序默认读取当前目录下的 `int_1.json`。运行结果保存在自动创建的时间戳文件夹 `FPQL_Results_YYYY-MM-DD_HH-MM-SS` 中，输入配置会自动备份。

## 输入参数说明

> 注意：以下注释仅用于说明，实际 JSON 文件中不能包含 `//` 注释。

```jsonc
{
    "nptu": 201,                // 速度网格点数
    "legmod": 21,               // 勒让德展开阶数
    "energy_range": {
        "r_E_1": 200.0,         // 基频能量范围系数（乘以温度得到能量范围）
        "r_E_2": 100.0          // 谐频能量范围系数
    },

    "ipheat": 2,                // 加热模式：1 基频加热，2 谐频加热
    "pwdens": 2.4,              // ICRH 功率密度（单位：W/cm³）
    "omega_ci": 36.5e6,         // 波频率（单位：Hz）
    "kperp": 10,                // 垂直波数 k⊥

    "ne": 5.5,                  // 电子密度（单位：10¹⁹ m⁻³）
    "tempe": 4.0,               // 电子温度（单位：keV）
    "nspec": 2,                 // 离子组分数
    "ispec": 1,                 // 输出粒子：1 第一种离子，2 第二种离子……
    "azi": [1.0, 1.0],          // 离子电荷数（以电子电荷归一化）
    "atm": [2, 1],              // 离子质量数（以质子质量归一化）
    "aconc": [0.97, 0.03],      // 离子浓度组分
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
        "enabled": 1,           // 环向捕获效应开关：1 开启，0 关闭
        "ucr": 0.4,             // 捕获效应 Sigmoid 权重的临界俯仰角
        "a": 20.0               // 捕获效应 Sigmoid 权重的陡度参数，越大过渡越陡
    },

    "plot_settings": {
        "range_k": 0,           // 绘图范围：1 限定范围，0 全范围
        "type": 4,              // 图型：0 f(v,u) 分布图；1 热图；2 μ 平均分布图；3 f(v∥,v⊥) 图；4 同时绘制 3 种（除 μ 平均图）
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
        "ipsii": [0.2667]       // 径向位置列表（仅多径向点模式生效；单径向点模式固定使用位置 1）
    }
}
```

基础网格、物种和运行开关等字段仍为必填项；部分嵌套设置支持默认值。请以仓库中的 `int_1.json` 为完整配置示例。

## 修复后的计算与输出

角向积分使用正确的 `μ·P′ₙ(μ)` 和 501 点 Simpson 规则。捕获权重沿用配置声明的单侧 `W(μ) = sigmoid(a × (μ - ucr))`，其奇偶阶耦合同时进入求解矩阵和功率诊断；未启用权重时才按奇偶性清除数值舍入误差。固定 `ucr` 不保证尾端等值线位置固定。

分布仍对所有勒让德分量使用同一个 `fnorm` 归一化。新增输出用于区分原始幅值与归一化形状：

- `case0_Fi_*.txt`：未经裁零或绘图处理的归一化分布，保留负值用于检查离散误差。
- 同名 `.npz`：`raw_coefficients`、`raw_derivatives`、`raw_distribution`，以及对应的 `normalized_*` 数组、`energies_keV`、`mu`、`fnorm` 和 `matrix_relative_residual`。原始解尚未换算为物理粒子密度。
- `debug.txt`：原始/归一化 NBI 束附近峰值、负值统计、尾端等值线阈值及能域截断标记。`char_mu` 是等值线延伸最远处的俯仰角，不是固定能量截面的分布峰位；若截断为 True，需要扩大能域并检查网格收敛后再比较位置。
- `01_read.txt`：实际矩阵源项的粒子矩、能量矩、线性方程残差，以及归一化解的体积功率残差。碰撞功率不再按输入功率二次缩放，与准线性加热率对应同一份解。

输入 NBI 功率的代数标定与实际方程源能量矩仍有差异，日志分别列出。体积功率残差含积分离散误差，未计边界通量；当前仍执行一次线性求解，不能据此宣称物理守恒或网格收敛已经通过。

## 回归验证与 NBI 功率扫描

在已安装 NumPy、SciPy、Matplotlib 的 Python 环境中，从仓库根目录运行：

```sh
python -m unittest discover -s tests -v
python analysis/nbi_power_scan_probe.py --output-dir FPQL_Results_nbi_scan
```

扫描脚本也支持 `--config path/to/int_1.json`。扫描功率固定为 0.1、0.5、1、2、5、10 W/cm³，覆盖配置中的 NBI 功率和径向功率剖面；束能量、网格及物种参数来自所选配置，非等温背景源保持固定。输出目录中的 `scan.json` 和 `P-*.npz` 保存结果；不指定目录时使用系统临时目录。

`analysis/nbi_power_scan_results.json` 保留修复前的历史扫描证据。修复后的默认 NBI-only 扫描仍复现这六组数值；ICRH 和协同模式需要重新计算。

本次修复验证了 14 项回归测试、默认配置的三模式完整运行，以及统一为 400 keV 能域、201 个速度点、21 阶展开、阈值 `1e-7` 的六档 ICRH 功率（0.06、0.2、0.4、0.8、2.0、3.6 W/cm³）和对应协同工况。12 组线性方程相对残差均小于 `6e-14`，其中 11 组尾端触及能域上界，高功率结果仍有负值。该检查确认了实现与诊断链路，没有完成能域、网格和展开阶数的物理收敛验证。
