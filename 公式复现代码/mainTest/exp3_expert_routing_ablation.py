import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# 强制 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r"D:\MindSystem_Env\code\mainTest"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] 正在初始化论文级多态路由压测 (基于 UnifiedMindSystem 真实规则)...")

# ---------------- 严谨对齐源码的参数 ----------------
VECTOR_DIM = 4096             
MAX_EXPERTS = 5               # 对齐你源码中的 if self.active_experts < 5
TIME_STEPS = 100              

# 对齐你源码 check_satiety 的阈值
THRESH_DROP_HIGH = 0.95       # >0.95 绝对冗余 Drop
THRESH_MERGE = 0.75           # 0.75~0.95 同化融合 Merge
EXPAND_HIT_LIMIT = 5          # 命中 5 次触发 Expand

def generate_test_sequence():
    """使用正交基与三角函数，严格控制高维张量的余弦相似度"""
    seq = []
    anchor_A = F.normalize(torch.randn(VECTOR_DIM, device=device), p=2, dim=0)
    anchor_B = F.normalize(torch.randn(VECTOR_DIM, device=device), p=2, dim=0)
    
    base_w = anchor_A.clone()
    
    for t in range(TIME_STEPS):
        if t < 25:
            # Phase 1: 价值同化 (相似度控制在 0.85 左右) -> 触发 MERGE，满5次触发 EXPAND
            z_in = F.normalize(0.85 * anchor_A + 0.52 * anchor_B, p=2, dim=0)
            stage = "价值同化区 (Phase 1)"
        elif t < 50:
            # Phase 2: 绝对冗余 (相似度 > 0.95) -> 触发 DROP (太熟悉了不学)
            noise = F.normalize(torch.randn(VECTOR_DIM, device=device), p=2, dim=0)
            z_in = F.normalize(0.98 * anchor_A + 0.19 * noise, p=2, dim=0)
            stage = "绝对冗余区 (Phase 2)"
        elif t < 75:
            # Phase 3: 跨界开荒 (相似度 < 0.75) -> 触发 SPAWN (填满槽位)
            # 每次都生成一个全新的正交方向
            new_domain = F.normalize(torch.randn(VECTOR_DIM, device=device), p=2, dim=0)
            z_in = F.normalize(new_domain, p=2, dim=0)
            stage = "异构开荒区 (Phase 3)"
        else:
            # Phase 4: 槽位耗尽后的噪音 -> 触发 DROP (保护机制)
            z_in = F.normalize(torch.randn(VECTOR_DIM, device=device), p=2, dim=0)
            stage = "算力保护区 (Phase 4)"
            
        seq.append((t, z_in, stage))
    return seq, base_w

def run_routing_simulation():
    seq, base_w = generate_test_sequence()
    
    # 模拟底层的专家矩阵质心
    expert_centroids = [base_w]
    expert_hits = [0]
    expert_dims = [16] # 模拟专家网络维度
    
    history_sim = []
    history_actions = []
    history_exp_count = []
    history_exp_dims = []

    action_map = {"Merge": 3, "Expand": 2, "Spawn": 1, "Drop": 0}

    for t, z_in, stage in seq:
        if len(expert_centroids) == 0:
            best_sim, best_idx = 0.0, -1
        else:
            sims = [F.cosine_similarity(z_in, w, dim=0).item() for w in expert_centroids]
            best_sim = max(sims)
            best_idx = sims.index(best_sim)
            
        history_sim.append(best_sim)
        action = None
        
        # ---------------- 完美复刻你的 check_satiety 逻辑 ----------------
        if best_sim > THRESH_DROP_HIGH:
            action = "Drop" # 绝对冗余
        elif best_sim > THRESH_MERGE:
            expert_hits[best_idx] += 1
            if expert_hits[best_idx] >= EXPAND_HIT_LIMIT:
                action = "Expand"
                expert_dims[best_idx] += 4 # 模拟网络扩容
                expert_hits[best_idx] = 0  # 重置 hit
            else:
                action = "Merge"
        else:
            if len(expert_centroids) < MAX_EXPERTS:
                action = "Spawn"
                expert_centroids.append(z_in) # 记录新质心
                expert_hits.append(0)
                expert_dims.append(16)
            else:
                action = "Drop" # 槽位满，拒绝学习
                
        history_actions.append(action_map[action])
        history_exp_count.append(len(expert_centroids))
        history_exp_dims.append(sum(expert_dims)) # 记录总网络容量

    return history_sim, history_actions, history_exp_count, history_exp_dims

sims, actions, counts, dims = run_routing_simulation()

# ---------------- 绘制高精度学术图谱 ----------------
print("[*] 正在渲染多态专家路由图谱...")
time_axis = np.arange(TIME_STEPS)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1.5, 1.5]})
fig.suptitle(r"资源受限下的多态专家路由压测 (基于 UnifiedMindSystem 源码)", fontsize=18, fontweight='bold', y=0.96)

stages = [(0, 25, '价值同化 (Phase 1)', '#e6f2ff'), 
          (25, 50, '绝对冗余 (Phase 2)', '#ffe6e6'), 
          (50, 75, '异构开荒 (Phase 3)', '#e6ffe6'), 
          (75, 100, '防御保护 (Phase 4)', '#eeeeee')]

# 图 1：知识投影打分
axes[0].plot(time_axis, sims, color='black', linewidth=2)
axes[0].axhline(y=THRESH_DROP_HIGH, color='red', linestyle='--', label=f'绝对冗余阈值 ({THRESH_DROP_HIGH})')
axes[0].axhline(y=THRESH_MERGE, color='blue', linestyle='--', label=f'同化融合阈值 ({THRESH_MERGE})')
for start, end, label, color in stages:
    axes[0].axvspan(start, end, facecolor=color, alpha=0.5)
    axes[0].text(start + 2, 1.05, label, fontsize=12, fontweight='bold')
axes[0].set_ylabel("余弦相似度 (Cosine Sim)", fontsize=12)
axes[0].legend(loc="lower right")
axes[0].grid(True, linestyle=':', alpha=0.7)
axes[0].set_ylim(0, 1.15)

# 图 2：系统容量监控 (专家数量与网络维度)
color1 = 'purple'
axes[1].set_ylabel("已激活专家数量 (个)", color=color1, fontsize=12)
axes[1].step(time_axis, counts, color=color1, linewidth=3, where='post')
axes[1].tick_params(axis='y', labelcolor=color1)
axes[1].set_yticks(range(1, MAX_EXPERTS + 2))
axes[1].axhline(y=MAX_EXPERTS, color='red', linestyle='-.', alpha=0.5, label='矩阵槽位上限')

ax2 = axes[1].twinx()  
color2 = 'teal'
ax2.set_ylabel("专家网络总隐层维度", color=color2, fontsize=12)
ax2.plot(time_axis, dims, color=color2, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

for start, end, label, color in stages:
    axes[1].axvspan(start, end, facecolor=color, alpha=0.5)
axes[1].grid(True, linestyle=':', alpha=0.7)

# 图 3：动作决策输出
scatter_colors = ['red' if a == 0 else 'green' if a == 1 else 'orange' if a == 2 else 'blue' for a in actions]
axes[2].scatter(time_axis, actions, c=scatter_colors, s=80, alpha=0.8, edgecolors='black')
for start, end, label, color in stages:
    axes[2].axvspan(start, end, facecolor=color, alpha=0.5)
axes[2].set_yticks([0, 1, 2, 3])
axes[2].set_yticklabels(["Drop (丢弃)", "Spawn (开荒)", "Expand (扩容)", "Merge (同化)"], fontsize=12)
axes[2].set_xlabel("时间步长 $t$", fontsize=12)
axes[2].grid(True, axis='x', linestyle=':', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = os.path.join(OUTPUT_DIR, "exp3_routing_paper_version.png")
plt.savefig(save_path, dpi=300)
print(f"[+] 压测完毕！完全吻合源码逻辑的图谱已生成至: {save_path}")
plt.show()