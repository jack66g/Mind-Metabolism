import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 强制 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 配置路径与环境 ----------------
MODEL_PATH = r"D:\MindSystem_Env\code\mode"
OUTPUT_DIR = r"D:\MindSystem_Env\code\mainTest"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] 正在初始化认知调度中心，当前计算设备: {device}")

# 严谨加载 7B 底座 (使用 4-bit 量化防止显存炸裂，保留张量计算空间)
# 注：如果只是纯量测时序公式，可注释掉底座加载以极速运行。这里保留以示架构完整性。
try:
    print("[*] 正在挂载通义 7B 模型权重...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        load_in_4bit=True, # 强行压缩至 ~5GB VRAM
        trust_remote_code=True
    )
    print("[+] 底座模型挂载完成，认知层处于待机状态。")
except Exception as e:
    print(f"[!] 模型加载跳过或失败，转为纯张量引擎测算模式。原因: {e}")

# ---------------- 定义张量动作空间 ----------------
# 动作索引: 0=高强度同化文档, 1=深度思考/整理, 2=闲聊, 3=系统休眠/发呆
ACTION_NAMES = ["高强度同化文档", "深度思考", "闲聊", "系统休眠"]
NUM_ACTIONS = len(ACTION_NAMES)

# 初始化大一统公式的核心张量参数 (模拟认知层下发的初始状态)
# 设定动作 0 (同化文档) 具有极其诱人的基础效用
U_base = torch.tensor([0.90, 0.40, 0.20, 0.10], device=device, dtype=torch.float32)
Lambda_short = torch.tensor([0.80, 0.50, 0.30, 0.00], device=device, dtype=torch.float32)
Psi_env = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float32)

# 超参数权重 (对应公式中的 \kappa, \alpha, \beta)
kappa, alpha, beta = 0.5, 0.3, 1.2 

# 疲劳衰减与恢复系数
DECAY_RATE = 0.08      # 每次执行动作造成的疲劳磨损
RECOVERY_RATE = 0.03   # 休息时其他动作的疲劳恢复速度

def run_cognitive_flow(enable_fatigue=True, time_steps=120):
    """
    运行时间步长循环，测算 Q_i(t) 决策流
    """
    # 疲劳张量初始化 f_fatigue(t=0) = [1.0, 1.0, 1.0, 1.0]
    f_fatigue = torch.ones(NUM_ACTIONS, device=device, dtype=torch.float32)
    
    history_actions = []
    history_Q_values = {i: [] for i in range(NUM_ACTIONS)}

    for t in range(time_steps):
        # 核心：大一统方程的张量计算 (暂时冻结偏好库与专家路由项)
        if enable_fatigue:
            # Q_i(t) = \kappa U_i(t) + \alpha \Lambda_i(t) + \beta \Psi_i(t) * f_fatigue(t)
            Q_t = (kappa * U_base) + (alpha * Lambda_short) + (beta * Psi_env * f_fatigue)
        else:
            # Baseline 退化模型：没有疲劳惩罚，f_fatigue 恒等于 1
            Q_t = (kappa * U_base) + (alpha * Lambda_short) + (beta * Psi_env * 1.0)
            
        # 决策层：非随机人格（确定性） a_t = \arg\max_i Q_i(t)
        action_t = torch.argmax(Q_t).item()
        
        # 记录数据
        history_actions.append(action_t)
        for i in range(NUM_ACTIONS):
            history_Q_values[i].append(Q_t[i].item())
            
        # 张量状态更新：应用疲劳机制
        if enable_fatigue:
            # 执行的动作疲劳度暴跌 (模拟注意力消耗)
            f_fatigue[action_t] -= DECAY_RATE
            
            # 其他未执行的动作缓慢恢复 (模拟注意力转移)
            mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
            mask[action_t] = False
            f_fatigue[mask] += RECOVERY_RATE
            
            # 边界截断夹角，确保 f_fatigue 在 [0, 1] 之间
            f_fatigue = torch.clamp(f_fatigue, min=0.0, max=1.0)

    return history_actions, history_Q_values

# ---------------- 执行对照实验 ----------------
print("[*] 开始执行 Baseline 测算 (禁用 f_fatigue) ...")
actions_base, q_base = run_cognitive_flow(enable_fatigue=False)

print("[*] 开始执行 Ours 测算 (激活 f_fatigue 时序衰减) ...")
actions_ours, q_ours = run_cognitive_flow(enable_fatigue=True)

# ---------------- 绘制学术级验证图表 ----------------
print("[*] 正在渲染时序动作分布图...")
time_axis = np.arange(120)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle(r"时序疲劳消融实验 (Ablation on $f_{fatigue}$)", fontsize=18, fontweight='bold')

# 图 1：Baseline 的 Q 值走势
axes[0, 0].set_title("对照组 (无疲劳机制): $Q_i(t)$ 演进", fontsize=14)
for i in range(NUM_ACTIONS):
    axes[0, 0].plot(time_axis, q_base[i], label=ACTION_NAMES[i], linewidth=2.5)
axes[0, 0].set_ylabel("决策意愿得分 $Q(t)$", fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# 图 2：Ours 的 Q 值走势 (动态轮换)
axes[0, 1].set_title("实验组 (激活 $f_{fatigue}$): $Q_i(t)$ 演进", fontsize=14)
for i in range(NUM_ACTIONS):
    axes[0, 1].plot(time_axis, q_ours[i], label=ACTION_NAMES[i], linewidth=2.5)
axes[0, 1].set_ylabel("决策意愿得分 $Q(t)$", fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# 图 3：Baseline 动作分布甘特图/散点图
axes[1, 0].set_title("对照组：动作执行序列 (死循环坍缩)", fontsize=14)
axes[1, 0].scatter(time_axis, actions_base, c='red', s=50, alpha=0.7)
axes[1, 0].set_yticks(range(NUM_ACTIONS))
axes[1, 0].set_yticklabels(ACTION_NAMES)
axes[1, 0].set_xlabel("时间步长 $t$", fontsize=12)

# 图 4：Ours 动作分布甘特图/散点图
axes[1, 1].set_title("实验组：动作执行序列 (健康波峰波谷)", fontsize=14)
axes[1, 1].scatter(time_axis, actions_ours, c='green', s=50, alpha=0.7)
axes[1, 1].set_yticks(range(NUM_ACTIONS))
axes[1, 1].set_yticklabels(ACTION_NAMES)
axes[1, 1].set_xlabel("时间步长 $t$", fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = os.path.join(OUTPUT_DIR, "exp1_fatigue_result.png")
plt.savefig(save_path, dpi=300)
print(f"[+] 实验完毕！高清验证图谱已生成至: {save_path}")
plt.show()