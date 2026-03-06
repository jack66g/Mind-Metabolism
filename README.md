# Mind-Metabolism: 动态认知系统 (PyTorch 实验版)

这是一个基于 PyTorch 构建的纯本地大模型动态认知系统实验项目。本项目不依赖任何复杂的云端 API 或向量数据库，而是通过原生张量计算，直接为大语言模型（如 Qwen2.5-AWQ）外挂一套具备“疲劳感知”、“动态扩容”和“记忆代谢”的神经网络模块。

## 💡 核心特性

本项目完全由底层代码驱动，包含以下几个核心模块：
* **认知与过滤底座**：借助 Qwen2.5-AWQ 模型提取文本的高维特征向量，并内置认知过滤器，自动拒绝低质量或无价值的噪音语料。
* **时序疲劳与动作惩罚**：告别传统 Agent 的死循环死锁。系统会记录高频动作，短时间内重复执行同一动作会触发“疲劳值”累积，强制系统转移注意力。
* **多态专家动态路由 (Spawn/Expand/Merge/Drop)**：当系统学习新语料（如通过 Hitokoto API 抓取）时，如果遇到全新领域，会自动在 GPU 上初始化（Spawn）一个新的前馈神经网络（专家）；如果是相似领域，则进行同化（Merge）或矩阵扩容（Expand），从物理层面避免模型微调带来的灾难性遗忘。
* **活性记忆代谢**：系统的记忆偏好会随系统运行时间自动衰减，优先释放过期或低权重的特征向量，完美控制本地显存占用。

---

## 📂 目录结构要求

在运行代码前，请务必保证你的本地文件夹结构严格按照以下方式排列，否则代码中的相对路径会导致模型加载失败：

```text
Project_Root/
│
├── mode/                  # 放置下载好的 Qwen2.5-AWQ 权重文件
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json ...
│
└── code/                  # 你的代码执行目录
    ├── main.py            # 本项目提供的核心代码
    └── mind_state.pth     # 运行后自动生成的记忆状态与专家权重持久化文件
🛠️ Windows 虚拟环境与依赖部署
本项目主要针对 Windows 10/11 本地开发环境设计。请使用 Python 3.10 或 3.11。

1. 创建并激活虚拟环境
在命令行（PowerShell 或 CMD）中执行：

Bash
python -m venv mind_env
.\mind_env\Scripts\activate
2. 安装核心深度学习依赖 (注意 CUDA 版本适配)
本项目重度依赖 torch.cuda。请确保你的电脑装有 NVIDIA 显卡。

Bash
# 安装带 CUDA 加速的 PyTorch (以 CUDA 12.1 为例)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
3. 安装大模型与量化推理依赖
代码底层使用 Transformers 库及 AWQ 量化框架加速：

Bash
pip install transformers accelerate requests
pip install autoawq
4. 安装画图与数据分析依赖 (用于压测复现)
如果你需要复现本项目的各项动态压测曲线（如显存动态平衡、决策多样性等），请补充安装以下可视化库：

Bash
pip install matplotlib seaborn pandas numpy
🚀 启动与交互指南
确保终端已经激活 mind_env 虚拟环境。

将工作目录切换至 code/ 文件夹下。

执行主程序：

Bash
python main.py
系统唤醒 Qwen2.5-AWQ 认知底座后，会进入交互菜单：

1 [学习]：系统将自动通过网络接口抓取语料，自主判断价值并进行反向传播 (loss.backward()) 更新专家网络，伴随突触更新日志。

2 [响应]：输入你的问题，系统会根据当前心智状态、疲劳度和历史记忆提取意图，并调用对应专家网络生成回复。

3 [自发]：无需输入，系统根据当前疲劳度与内在动机，自发完成一个动作或想法。

exit：安全退出并保存当前专家权重到 mind_state.pth。

⚠️ 实验性提示
本项目包含 requires_grad=True 的本地反向传播过程。如果在学习模式下遇到显存溢出 (OOM) 报错，请尝试降低基础模型的上下文长度或缩减专家的初始维度 (initial_dim)。
