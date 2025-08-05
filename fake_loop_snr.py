#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake training loop to validate a log-SNR sampler with FlowMatchEulerDiscreteScheduler.

What it does:
  1) Sample λ ~ N(mu=-6, sigma=2) in each iteration.
  2) Map to t = sigmoid(-λ/2).
  3) Call scheduler.set_timesteps(timesteps=sorted(t, desc=True)) each iter.
  4) Accumulate many samples and plot:
       (a) Histogram of λ vs. target Normal PDF.
       (b) Histogram of t vs. induced p(t) from Normal(λ).
  5) (Optional) Resolution shift: α = sqrt(m/n). Apply λ <- λ - 2 log α.

Requires: torch, diffusers >= 0.34, matplotlib, numpy
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from typing import Union, Optional  # 新增：类型注解支持

# ---------------------------
# Config
# ---------------------------
# ----- 中文说明 -----
# 以下配置控制采样分布与迭代规模：
# 1) MU_LAMBDA、SIGMA_LAMBDA 定义 log-SNR λ 的目标正态分布 N(μ, σ²)。
# 2) BATCH_SIZE 指一次迭代中要采样的样本数量。
# 3) NUM_ITERS 控制总迭代次数，用来累积足够统计量绘制直方图。
# 4) ALPHA 是 SD3 的分辨率缩放系数，α>1 时意味着高分辨率，λ 会整体平移 −2logα。
# 5) DEVICE 指明计算应运行于 CPU 还是 GPU。
SAVE_NAME = "compute_style_friendly_"
#Style Friendly
MU_LAMBDA   = -6.0
SIGMA_LAMBDA = 2.0
#SD3&Flux
# SAVE_NAME = "sd3_flux_"
# #SD3/FLUX training sampler in λ-space:  N(μ = -2*log(3), σ = 2)
# MU_LAMBDA   = -2*math.log(3)
# SIGMA_LAMBDA = 2.0



BATCH_SIZE  = 256
NUM_ITERS   = 1000                 # total samples = 76,800
ALPHA       = 1.0                 # √(m/n). If >1, shifts λ toward lower logSNR by -2logα (SD3)
DEVICE      = torch.device("cpu") # or "cuda" if available

# ---- 派生 logit-normal 参数（连接 diffusers 标准采样函数） ----
LOGIT_MEAN  = -(MU_LAMBDA - 2.0 * math.log(ALPHA)) / 2.0
LOGIT_STD   = SIGMA_LAMBDA / 2.0
WEIGHTING_SCHEME = "logit_normal"
MODE_SCALE  = None  # 仅在 weighting_scheme=="mode" 时使用


# ---------------------------
# Init scheduler
# ---------------------------
# ----- 中文说明 -----
# 构造扩散调度器 FlowMatchEulerDiscreteScheduler。
# shift 为基础平移量；use_dynamic_shifting=True 时调度器会根据分辨率自动调整 λ，
# 本示例手动平移 λ，因此保持 False。
# Docs: set custom timesteps or use shift/dynamic shifting; num_train_timesteps is irrelevant for this fake loop.
# https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete
scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=1.0,                   # base shift; resolution-related shifts can also be handled internally if desired
    use_dynamic_shifting=False,  # set True if you want internal dynamic shifting; here we demonstrate manual λ-shift
)

# ---------------------------
# Helpers
# ---------------------------
# ----- 中文说明 -----
# 工具函数：
# 1) sample_lambda_and_t：采样 λ 并映射到 t。
# 2) normal_pdf：一维高斯 PDF。
# 3) lam_of_t：由 t 反算 λ。
# 4) p_t_from_normal_lambda：利用换元公式计算由 Normal(λ) 推导出的 t 分布。
def sample_lambda_and_t(B, mu, sigma, alpha=1.0, device="cpu"):
    """λ ~ N(mu, sigma^2); optional SD3 resolution shift; then t = sigmoid(-λ/2)"""
    lam = torch.normal(mean=torch.tensor(mu, device=device),
                       std=torch.tensor(sigma, device=device),
                       size=(B,))
    if alpha != 1.0:
        lam = lam - 2.0 * math.log(alpha)   # SD3 log-SNR constant shift
    t = torch.sigmoid(-lam / 2.0)
    return lam, t

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def lam_of_t(t):
    return 2.0 * np.log((1.0 - t) / t)

def p_t_from_normal_lambda(t, mu, sigma):
    # Change-of-variables:  p_t(t) = NormalPDF(λ(t); μ, σ) * |dλ/dt|, with |dλ/dt| = 2/(t(1-t))
    lam = lam_of_t(t)
    jac = 2.0 / (t * (1.0 - t))
    return normal_pdf(lam, mu, sigma) * jac

# ----- 新增函数: compute_density_for_timestep_sampling -----
def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    """
    计算 SD3 训练中 time step 采样密度。

    源自 diffusers PR https://github.com/huggingface/diffusers/pull/8528。
    论文参考：https://huggingface.co/papers/2403.03206v1。
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u

# ---------------------------
# Fake loop (no model, no loss)
# ---------------------------
# ----- 中文说明 -----
# 伪训练循环：重复 NUM_ITERS 次采样并将 t 注册到调度器，以收集 λ / t 样本用于后续可视化。
torch.manual_seed(0)
all_lambda = []
all_t = []

for it in range(NUM_ITERS):
    # 使用 diffusers 标准采样密度函数得到 t
    t = compute_density_for_timestep_sampling(
        weighting_scheme=WEIGHTING_SCHEME,
        batch_size=BATCH_SIZE,
        logit_mean=LOGIT_MEAN,
        logit_std=LOGIT_STD,
        mode_scale=MODE_SCALE,
        device=DEVICE,
    )

    # 由 t 反算 λ
    lam = 2.0 * torch.log((1.0 - t) / t)

    # 注册本 batch 的 timestep 网格（降序）
    t_steps = torch.sort(t, descending=True).values
    scheduler.set_timesteps(timesteps=t_steps.detach().cpu().tolist())  # noqa

    all_lambda.append(lam.cpu())
    all_t.append(t.cpu())

all_lambda = torch.cat(all_lambda).numpy()
all_t = torch.cat(all_t).numpy()

print(f"Empirical λ mean/std: {all_lambda.mean():.4f} / {all_lambda.std(ddof=0):.4f}  "
      f"(target {-6.0 - 2.0*math.log(ALPHA):.4f} / {SIGMA_LAMBDA:.4f})")

# ---------------------------
# Plots: λ histogram vs. Normal, and t histogram vs. induced density
# ---------------------------
# ----- 中文说明 -----
# 绘制两张图验证采样分布：
# 1) λ 直方图与目标高斯曲线对比；
# 2) t 直方图与由 λ 分布推导的解析 p(t) 对比。
# λ-plot
x = np.linspace(-16, 6, 2000)
pdf = normal_pdf(x, MU_LAMBDA - 2.0*np.log(ALPHA), SIGMA_LAMBDA)

plt.figure(figsize=(8, 4.5))
plt.hist(all_lambda, bins=140, density=True, alpha=0.6, label="Empirical λ")
plt.plot(x, pdf, linewidth=2, label=r"Target $\mathcal{N}(\mu,\sigma^2)$")
plt.title(r"Empirical $\lambda$ (logSNR) vs. target Normal")
plt.xlabel(r"$\lambda$")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.xlim(-12.5, 7.5)
plt.savefig(SAVE_NAME + "lambda_hist_verify.png", dpi=200)

# t-plot
tgrid = np.linspace(1e-5, 1-1e-5, 2000)
pt = p_t_from_normal_lambda(tgrid, MU_LAMBDA - 2.0*np.log(ALPHA), SIGMA_LAMBDA)

plt.figure(figsize=(8, 4.5))
plt.hist(all_t, bins=140, density=True, alpha=0.6, label="Empirical t")
plt.plot(tgrid, pt, linewidth=2, label=r"Analytic $p(t)$ from Normal($\lambda$)")
plt.title("Empirical t vs. induced p(t)")
plt.xlabel("t")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(SAVE_NAME + "t_hist_verify.png", dpi=200)

# (Optional) show that Δλ = λ(t_m) − λ(t_n) = −2 log α is constant if you also remap t by SD3's resolution rule.
# Here we just confirm the constant shift property on λ directly:
if ALPHA != 1.0:
    delta = -2.0 * math.log(ALPHA)
    print(f"With α={ALPHA}, expected constant log-SNR shift Δλ = {delta:.6f}.")
    # You can further verify by sampling tn, mapping to tm with tm = (α tn)/(1+(α−1) tn), then eval λ(tm)−λ(tn).

print("Saved: " + SAVE_NAME + "lambda_hist_verify.png, " + SAVE_NAME + "t_hist_verify.png")


# --- Plot & save the log-SNR probability distribution (PDF) ---
# ----- 中文说明 -----
# 额外绘制目标 log-SNR 概率密度函数，横轴范围固定为 [−12.5, 7.5]，
# 并用 axvspan 将 λ≤−5 区域着色以提醒高噪声区。
# Target: λ ~ Normal( μ = MU_LAMBDA - 2*log(ALPHA), σ = SIGMA_LAMBDA )
x = np.linspace(-12.5, 7.5, 2000)
pdf = normal_pdf(x, MU_LAMBDA - 2.0*np.log(ALPHA), SIGMA_LAMBDA)

plt.figure(figsize=(8, 4.5))
# （可选）阴影高噪声区：λ <= -5
plt.axvspan(x.min(), -5.0, alpha=0.20)
plt.plot(x, pdf, linewidth=2)
plt.title("Probability Distribution of logSNR")
plt.xlabel(r"logSNR  $\lambda$")
plt.ylabel("Probability Density")
plt.xlim(-12.5, 7.5)
plt.tight_layout()
plt.savefig(SAVE_NAME + "logsnr_pdf.png", dpi=200)
print("Saved: " + SAVE_NAME + "logsnr_pdf.png")
