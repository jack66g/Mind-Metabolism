import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

# 强制 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 配置路径与底层环境 ----------------
OUTPUT_DIR = r"D:\MindSystem_Env\code\mainTest"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] 正在初始化偏好库(PrefPool)压测环境，计算设备: {device}")

# ---------------- 严谨设定张量与方程超参数 ----------------
TIME_STEPS = 1000             # 模拟连续运行 1000 个时间步 (高频交互)
VECTOR_DIM = 4096             # 严格对齐 7B 模型特征向量维度 (高维张量)
GAMMA = 0.015                 # 遗忘曲线衰减率 \gamma
EPSILON = 0.05                # 偏好权重生存阈值 \epsilon，低于此值执行 torch 内存释放
ZETA_INIT = 1.0               # 偏好初始强度 \zeta_j

def run_prefpool_stress_test(enable_forgetting=True):
    """
    偏好库内存防爆压测：
    使用真实的 PyTorch Tensor 拼接与掩码切片，绝不使用 Python List 妥协。
    """
    # 初始化动态张量池 (初始维度为 0)
    # 采用 float16 模拟真实的端侧半精度量化推理环境
    pref_vectors = torch.empty((0, VECTOR_DIM), device=device, dtype=torch.float16)
    pref_times = torch.empty((0,), device=device, dtype=torch.float32)
    pref_zetas = torch.empty((0,), device=device, dtype=torch.float32)
    
    history_tensor_size = []
    history_vram_mb = []

    for t in range(TIME_STEPS):
        # 模拟高频且碎片化的用户交互：每步随机生成 1~3 个新偏好特征
        num_new = torch.randint(1, 4, (1,)).item()
        
        # 真实分配 GPU 显存创建新张量
        new_vecs = torch.randn((num_new, VECTOR_DIM), device=device, dtype=torch.float16)
        new_times = torch.full((num_new,), t, device=device, dtype=torch.float32)
        new_zetas = torch.full((num_new,), ZETA_INIT, device=device, dtype=torch.float32)
        
        # ---------------- 张量动态累积 ----------------
        pref_vectors = torch.cat([pref_vectors, new_vecs], dim=0)
        pref_times = torch.cat([pref_times, new_times], dim=0)
        pref_zetas = torch.cat([pref_zetas, new_zetas], dim=0)
        
        # ---------------- 核心大一统方程：偏好防爆机制 ----------------
        if enable_forgetting:
            # 计算当前所有偏好的时序衰减权重：w = \zeta * exp(-\gamma * (t - t_j))
            current_weights = pref_zetas * torch.exp(-GAMMA * (t - pref_times))
            
            # 生成存活掩码 (Boolean Mask Tensor)
            keep_mask = current_weights >= EPSILON
            
            # 触发底层的张量切片与内存释放 (剔除低于阈值的无价值历史偏好)
            pref_vectors = pref_vectors[keep_mask]
            pref_times = pref_times[keep_mask]
            pref_zetas = pref_zetas[keep_mask]
            
            # 强制清空无用缓存，防止 PyTorch 显存碎片化
            if device.type == 'cuda' and t % 50 == 0:
                torch.cuda.empty_cache()

        # ---------------- 严密监控指标采集 ----------------
        current_size = pref_vectors.shape[0]
        # 精确计算显存占用 (MB)：
        # float16 占 2 Bytes, float32 占 4 Bytes
        bytes_allocated = (current_size * VECTOR_DIM * 2) + (current_size * 4) + (current_size * 4)
        vram_mb = bytes_allocated / (1024 * 1024)
        
        history_tensor_size.append(current_size)
        history_vram_mb.append(vram_mb)

    # 运行完毕后清理内存
    del pref_vectors, pref_times, pref_zetas
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return history_tensor_size, history_vram_mb

# ---------------- 执行对照组与实验组 ----------------
print("[!] 启动对照组压测 (静态累积，无遗忘机制) - 准备迎接显存爆炸...")
size_base, vram_base = run_prefpool_stress_test(enable_forgetting=False)

print("[*] 启动实验组压测 (激活 exp(-\gamma*(t-t_j)) 指数衰减) - 监控动态平衡...")
size_ours, vram_ours = run_prefpool_stress_test(enable_forgetting=True)

# ---------------- 绘制学术级高对比度图表 ----------------
print("[*] 正在渲染偏好库显存压测图谱...")
time_axis = np.arange(TIME_STEPS)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(r"偏好库遗忘曲线防爆压测 (Ablation on $e^{-\gamma(t-t_j)}$)", fontsize=18, fontweight='bold', y=1.05)

# 图 1：张量池条目规模比对 (Tensor Size)
axes[0].set_title("PrefPool 张量维度演进 (条目数 $N$)", fontsize=14)
axes[0].plot(time_axis, size_base, label="对照组 (静态累积，触发 OOM 风险)", color='red', linewidth=3, linestyle='--')
axes[0].plot(time_axis, size_ours, label=r"实验组 (激活 $e^{-\gamma(t-t_j)}$ 动态释放)", color='green', linewidth=3)
axes[0].fill_between(time_axis, size_ours, color='green', alpha=0.1)
axes[0].set_xlabel("时间步长 $t$", fontsize=12)
axes[0].set_ylabel("偏好张量池大小 (条)", fontsize=12)
axes[0].legend(loc="upper left")
axes[0].grid(True, linestyle=':', alpha=0.7)

# 图 2：真实显存占用比对 (VRAM Usage)
axes[1].set_title("本地 VRAM 显存占用演进 (MB)", fontsize=14)
axes[1].plot(time_axis, vram_base, label="对照组 (显存线性爆炸)", color='darkred', linewidth=3, linestyle='-.')
axes[1].plot(time_axis, vram_ours, label=r"实验组 (达到显存动态平衡)", color='teal', linewidth=3)
axes[1].fill_between(time_axis, vram_ours, color='teal', alpha=0.1)
axes[1].set_xlabel("时间步长 $t$", fontsize=12)
axes[1].set_ylabel("理论显存占用量 (MB)", fontsize=12)
axes[1].legend(loc="upper left")
axes[1].grid(True, linestyle=':', alpha=0.7)

# 标注动态平衡线
equilibrium_vram = np.mean(vram_ours[-200:]) # 取最后200步的平均显存作为平衡点
axes[1].axhline(y=equilibrium_vram, color='black', linestyle=':', linewidth=2, alpha=0.6)
axes[1].text(TIME_STEPS*0.6, equilibrium_vram*1.2, f'动态显存平衡线: ~{equilibrium_vram:.1f} MB', fontsize=11, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "exp2_forgetting_ablation.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[+] 压测完毕！极度硬核的张量防爆图谱已生成至: {save_path}")
plt.show()