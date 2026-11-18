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

所有参数都支持缺省：不写的键自动使用程序内置默认值，旧格式配置文件可直接运行。
