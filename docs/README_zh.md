# Log-SNR 采样可视化工具

> **说明**：本仓库为个人/教学目的而编写的 **非官方实现**，与论文 *Style-Friendly SNR Sampler for Style-Driven Generation* 的官方代码无关。

本项目提供了**独立的工具脚本**，用于在 log-SNR（λ）空间中可视化概率分布。这些分布被用于诸如 SD3 / FLUX 以及 *Style-Friendly* 等最新扩散模型。

项目 **不包含** 任何扩散模型的训练或推理代码，仅提供两个轻量级 Python 脚本，帮助您：

* 验证在 log-SNR 空间中服从高斯分布采样器的统计特性；
* 复现论文中的目标 log-SNR 分布曲线（如 Fig.2）。

---
## 1. 目录结构

| 文件 | 功能 |
|------|------|
| `fake_loop_snr.py` | “伪训练”循环，从自定义的 Normal 分布中采样 λ，映射到时间步 \(t = \sigma(-\lambda/2)\)，并通过 `diffusers.schedulers.FlowMatchEulerDiscreteScheduler` 注册自定义网格。脚本累计数万样本并生成三张诊断图（λ 直方图、t 直方图、目标 λ PDF）。 |
| `plot_style_friendly_logsnr.py` | 纯绘图库脚本，生成多个 Normal 分布的高质量曲线图，可复现 *Style-Friendly* 论文 Fig.2 中的四条曲线，并支持通过 JSON 参数自定义曲线。 |

上述脚本与模型权重无关，仅依赖 NumPy、Matplotlib，以及（仅 `fake_loop_snr.py` 需要）*diffusers* 与 *PyTorch* **≥1.13**。默认运行于 CPU，GPU 可选。

---
## 2. 环境安装

1. **克隆仓库**（或直接复制两个脚本）

   ```bash
   git clone https://github.com/your-name/LogSNRVis.git
   cd LogSNRVis
   ```

2. **创建虚拟环境**（可选但推荐）

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **安装依赖**

   ```bash
   pip install torch diffusers matplotlib numpy
   ```

   • 只有在运行 `fake_loop_snr.py` 时才需要 *torch* ≥1.13 以及 *diffusers*。<br/>
   • 若仅使用绘图脚本，可跳过这两项依赖。

---
## 3. 使用方法

### 3.1 `fake_loop_snr.py`

该脚本用于快速检查高斯 log-SNR 采样器是否符合预期。

核心流程：

1. 从 \(\mathcal{N}(\mu, \sigma^2)\) 采样一批 λ；
2. 可选：应用分辨率平移（SD3 规则）\(\lambda \leftarrow \lambda - 2\log\alpha\)，其中 \(\alpha = \sqrt{m/n}\)；
3. 映射至扩散时间步 \(t = \sigma(-\lambda/2)\)；
4. 将排序后的 t 传入 `scheduler.set_timesteps()`；
5. 累积样本，用于绘制统计图。

#### 默认配置

```python
SAVE_NAME    = "style_friendly_"   # 输出图片前缀
MU_LAMBDA    = -6.0                # λ 分布均值
SIGMA_LAMBDA = 2.0                 # λ 分布标准差
BATCH_SIZE   = 256                 # 每次迭代样本数
NUM_ITERS    = 1000                # 迭代次数（约 7.7e4 样本）
ALPHA        = 1.0                 # 分辨率缩放（α>1 时 λ 向低 SNR 区移动）
DEVICE       = torch.device("cpu") # 或 "cuda"
```

运行脚本：

```bash
python fake_loop_snr.py
```

脚本将输出三张 PNG：

* `<SAVE_NAME>lambda_hist_verify.png` — λ 直方图与目标正态 PDF；
* `<SAVE_NAME>t_hist_verify.png` — t 直方图与解析 *p*(t)；
* `<SAVE_NAME>logsnr_pdf.png` — 目标 λ PDF 单独图。

您可在脚本顶部修改全局常量以改变分布、批大小、分辨率缩放等。

### 3.2 `plot_style_friendly_logsnr.py`

此脚本是一个 Matplotlib CLI 封装，默认绘制四条 Normal 分布曲线：

1. SD3 / FLUX 训练分布：\(\mathcal{N}(\mu = -2\log3, \sigma = 2)\)
2. Style-friendly: \(\mathcal{N}(\mu = -6, \sigma = 1)\)
3. Style-friendly: \(\mathcal{N}(\mu = -6, \sigma = 2)\)
4. Style-friendly: \(\mathcal{N}(\mu = -6, \sigma = 3)\)

基础用法：

```bash
python plot_style_friendly_logsnr.py --out style_friendly_curves.png
```

通过 CLI 参数自定义 x 轴范围、采样点数、阴影区及曲线：

```bash
python plot_style_friendly_logsnr.py \
  --xmin -12.5 --xmax 7.5 --shade -5 \
  --curves '[{"mu":-2*math.log(3),"sigma":2,"label":"SD3 and Flux"}, {"mu":-6,"sigma":1,"label":"μ=-6 σ=1"}]' \
  --out custom_curves.png
```

`--curves` 接受一个 JSON 数组，每项包含：

* `mu` — λ 分布均值（字符串，可写 Python 表达式，例如 `-2*math.log(3)`）；
* `sigma` — 标准差；
* `label` — 图例标签（可选）。

---
## 4. 分辨率平移（SD3 规则）

多分辨率训练时，SD3 使用固定的 log-SNR 平移：

\[ \Delta\lambda = -2\log\alpha, \qquad \alpha = \sqrt{m/n} \]

在 `fake_loop_snr.py` 中将 `ALPHA>1` 即可观察到这一常数平移并打印对应值。

---
## 5. 参考文献

> **Style-Friendly SNR Sampler for Style-Driven Generation**.<br/>
> Xi Chen 等, 2024, arXiv:2411.14793

---
## 6. 示例输出

仓库已附带两组共 **六张 PNG**，分别对应 *Style-friendly* 与 *SD3 / Flux* 分布：

| 文件 | 内容 | 解释 |
|------|------|------|
| `style_friendly_lambda_hist_verify.png` | λ 直方图与目标 Normal PDF 对比 | 验证采样器是否遵循 \(\mathcal{N}(\mu=-6,\sigma=2)\) |
| `style_friendly_t_hist_verify.png` | t 直方图与解析 *p*(t) 对比 | 验证换元推导的正确性 |
| `style_friendly_logsnr_pdf.png` | 目标 λ PDF，阴影标注高噪声区 (λ ≤ −5) | 高噪声区通常出现风格化特征 |
| `sd3_flux_lambda_hist_verify.png` | SD3 / Flux 的 λ 直方图 | 确认 SD3 训练采样器形状 |
| `sd3_flux_t_hist_verify.png` | SD3 / Flux 的 t 直方图 | 与对应解析密度对齐 |
| `sd3_flux_logsnr_pdf.png` | SD3 / Flux 的 λ PDF | 相比 Style-friendly，将更多概率质量分配于高 SNR |

---
## 7. 作者

- **Tian YE** — 香港科技大学（广州）博士生
- **ChatGPT O3** — 大语言模型协助代码与文档

---
## 8. 许可证

本项目采用 MIT License，详见 `LICENSE` 文件。 