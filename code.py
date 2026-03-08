import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
from datetime import datetime # [新增] 用于获取真实系统时间以驱动生物钟

# ==========================================
# 1. 环境初始化与文件夹准备
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "mode")
save_path = os.path.join(current_dir, "mind_state.pth")

corpus_path = os.path.join(current_dir, "training_corpus.txt") 

knw_corpus_dir = os.path.join(current_dir, "knowledge_corpus")
if not os.path.exists(knw_corpus_dir):
    os.makedirs(knw_corpus_dir)
    print(f"📁 已自动创建知识库文件夹：{knw_corpus_dir}")

persona_path = os.path.join(current_dir, "persona.txt")

print("正在唤醒认知底座 (Qwen2.5-AWQ)...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_llm = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16
).eval()
for param in base_llm.parameters(): param.requires_grad = False
HIDDEN_DIM = base_llm.config.hidden_size 

# ==========================================
# 2. 核心系统 (双轨制：泛化层 + 知识层)
# ==========================================
class UnifiedMindSystem(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.structure_changed = False 
        
        # ---------------------------------------------
        # [彻底解耦] 代码里不再硬编码任何冗长性格
        # ---------------------------------------------
        if not os.path.exists(persona_path):
            with open(persona_path, 'w', encoding='utf-8') as f:
                f.write("你是一个温柔、情绪稳定的女生。请用自然、亲密的语气聊天。\n")
                
        with open(persona_path, 'r', encoding='utf-8') as f:
            core_traits = [line.strip() for line in f if line.strip()]
            
        if not core_traits: core_traits = ["系统默认AI"] # 防崩保底
        self.persona_text = "\n".join(core_traits)
        fixed_actions = ["休眠", "发呆", "伸懒腰"] 

        traits_embs = [self._get_emb(t) for t in core_traits]
        self.C_matrix = nn.Parameter(torch.cat(traits_embs, dim=0), requires_grad=False)
        self.C_t = nn.Parameter(self.C_matrix.mean(dim=0, keepdim=True), requires_grad=False)
        
        print("⏳ 正在根据人格钢印，为你随机抽取今日专属潜意识动作...")
        action_prompt = (
            f"<|im_start|>system\n你是一个心理学与行为设计专家。请根据以下角色的【思想钢印】，"
            f"为TA设计3个简短的、极具个性的【日常小动作/微意图】（例如：想喝奶茶、整理桌面、叹气、看窗外等）。"
            f"只需直接输出这3个词，用逗号隔开，不要包含任何多余解释！<|im_end|>\n"
            f"<|im_start|>user\n【思想钢印】：\n" + self.persona_text + "<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(action_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = base_llm.generate(**inputs, max_new_tokens=20, temperature=0.7)
        res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        dynamic_actions = [a.strip() for a in re.split(r'[,，、\n]', res) if a.strip()]
        if len(dynamic_actions) > 3: dynamic_actions = dynamic_actions[:3]
        if not dynamic_actions: dynamic_actions = ["回想往事", "轻哼歌曲"]
        print(f"✨ 抽取完成！提取到的动态潜意识：{dynamic_actions}")
        
        self.immediate_goals = []      
        self.active_preferences = []   
        self.current_action_vectors = None 
        self.action_labels = [] 
        
        self.active_actions = fixed_actions + dynamic_actions 
        
        self.episodic_buffer = [] 
        self.last_Q = None        
        self.rho = 0.8            
        self.forget_rate = 0.004
        self.day_start_time = time.time()
        
        # ====== [新增] 多巴胺与疲劳的基础底层参数 ======
        self.dopamine_level = 5           # 默认活跃度 5
        self.cycle_seconds = 144          # 动态触发周期（映射多巴胺）
        self.fatigue_rate = 0.35          # 疲劳度单次积攒速度
        # ================================================

        self.goal_fatigue = None
        self.action_counts = None
        self.tau = 1.3
        
        self.kappa = nn.Parameter(torch.tensor(1.2))
        self.zeta = nn.Parameter(torch.tensor(1.1))
        self.eta = nn.Parameter(torch.tensor(0.5))    
        self.lambda_meta = nn.Parameter(torch.tensor(0.5))
        self.phi_meta = nn.Parameter(torch.tensor(0.3))
        self.omega_z = nn.Parameter(torch.tensor(1.6))
        self.alpha_residual = nn.Parameter(torch.tensor(0.1))

        # ---------------------------------------------
        # 右脑：泛化层 (Gen-Ex)
        # ---------------------------------------------
        self.active_experts = 0
        self.experts = nn.ModuleList([]) 
        self.expert_centroids = None     
        self.expert_anchors = []
        self.expert_hits = []            

        # ---------------------------------------------
        # 左脑：知识层 (Knw-Ex)
        # ---------------------------------------------
        self.max_knw_experts = 5  
        self.knw_active_experts = 0
        self.knw_experts = nn.ModuleList([]) 
        self.knw_centroids = None     
        self.knw_anchors = []
        self.knw_locked_dims = [] 

    def _get_emb(self, text):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return base_llm(**inputs, output_hidden_states=True).hidden_states[-1][0, -1, :].unsqueeze(0).to(torch.float32)

    def cognitive_judge(self, text):
        prompt = f"<|im_start|>system\n你是一个严格的认知过滤器。请判断以下文本是否包含有价值的知识、细腻的情感或高质量的对话语料。只回答“YES”或“NO”。\n<|im_end|>\n<|im_start|>user\n文本：{text}\n<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = base_llm.generate(**inputs, max_new_tokens=5, temperature=0.1)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return "YES" in response.upper()

    def learn_from_text(self, text, expert_idx):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        with torch.no_grad():
            outputs = base_llm(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1] 
        expert_out = self.experts[expert_idx](hidden_states.to(torch.float32))
        modified_hidden = hidden_states + self.alpha_residual * expert_out.to(hidden_states.dtype)
        logits = base_llm.lm_head(modified_hidden).to(torch.float32)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def spawn_expert(self, H_base, anchor_text, initial_dim=4):
        new_expert = nn.Sequential(
            nn.Linear(self.hidden_dim, initial_dim), nn.GELU(),
            nn.Linear(initial_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim)
        ).cuda()
        self.experts.append(new_expert)
        self.expert_anchors.append(anchor_text[:15])
        self.expert_hits.append(0)
        self.expert_centroids = H_base.detach().clone() if self.expert_centroids is None else torch.cat([self.expert_centroids, H_base.detach()], dim=0)
        self.active_experts += 1
        print(f"🎇 [泛化层 Spawn] 专家 E{self.active_experts} 诞生！")

    def expand_expert(self, expert_idx, expansion_size=4):
        old_expert = self.experts[expert_idx]
        old_dim = old_expert[0].out_features
        new_dim = old_dim + expansion_size
        new_l1 = nn.Linear(self.hidden_dim, new_dim).cuda()
        new_l2 = nn.Linear(new_dim, self.hidden_dim).cuda()
        new_l1.weight.data[:old_dim, :] = old_expert[0].weight.data
        new_l1.bias.data[:old_dim] = old_expert[0].bias.data
        new_l2.weight.data[:, :old_dim] = old_expert[2].weight.data
        new_l2.bias.data = old_expert[2].bias.data
        self.experts[expert_idx] = nn.Sequential(new_l1, nn.GELU(), new_l2, nn.LayerNorm(self.hidden_dim).cuda())
        print(f"📈 [泛化层 Expand] 专家 E{expert_idx+1} 扩容: {old_dim} -> {new_dim}")

    def check_satiety(self, H_base):
        if self.active_experts == 0: return "SPAWN", 0, 0.0
        sims = F.cosine_similarity(H_base, self.expert_centroids)
        max_sim, idx = torch.max(sims, dim=0)
        if max_sim.item() > 0.95: return "DROP", idx.item(), max_sim.item()
        elif max_sim.item() > 0.75: return "MERGE", idx.item(), max_sim.item()
        else: 
            if self.active_experts < 5: return "SPAWN", idx.item(), max_sim.item()
            else: return "DROP", idx.item(), max_sim.item()

    def chi_function(self, mode, H_in=None, W_m=None):
        if mode == "DROP": return 0.0
        elif mode == "MERGE": return F.cosine_similarity(H_in, W_m.unsqueeze(0)).item()
        elif mode == "EXPAND": return 0.6 + 0.4 * F.cosine_similarity(H_in, W_m.unsqueeze(0)).item()
        elif mode == "SPAWN": 
            dist = 1.0 - F.cosine_similarity(H_in, W_m.unsqueeze(0)).item() if W_m is not None else 1.0
            return 0.8 + 0.4 * dist
        return 0.0

    def learn_from_knw_text(self, text, expert_idx):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        with torch.no_grad():
            outputs = base_llm(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1] 
        expert_out = self.knw_experts[expert_idx](hidden_states.to(torch.float32))
        modified_hidden = hidden_states + self.alpha_residual * expert_out.to(hidden_states.dtype)
        logits = base_llm.lm_head(modified_hidden).to(torch.float32)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def check_knw_satiety(self, H_base):
        if self.knw_active_experts == 0: return "SPAWN", 0, 0.0
        sims = F.cosine_similarity(H_base, self.knw_centroids)
        max_sim, idx = torch.max(sims, dim=0)
        
        if max_sim.item() > 0.95: return "DROP", idx.item(), max_sim.item()
        elif max_sim.item() > 0.75: return "EXPAND", idx.item(), max_sim.item()
        else:
            if self.knw_active_experts < self.max_knw_experts:
                return "SPAWN", idx.item(), max_sim.item()
            else:
                return "EXPAND", idx.item(), max_sim.item() 

    def spawn_knw_expert(self, H_base, anchor_text, initial_dim=4):
        new_expert = nn.Sequential(
            nn.Linear(self.hidden_dim, initial_dim), nn.GELU(),
            nn.Linear(initial_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim)
        ).cuda()
        self.knw_experts.append(new_expert)
        self.knw_anchors.append(anchor_text[:15])
        self.knw_locked_dims.append(0) 
        self.knw_centroids = H_base.detach().clone() if self.knw_centroids is None else torch.cat([self.knw_centroids, H_base.detach()], dim=0)
        self.knw_active_experts += 1
        print(f"📖 [知识库 Spawn] K{self.knw_active_experts} 领域确立！")

    def expand_knw_expert(self, expert_idx, expansion_size=4):
        old_expert = self.knw_experts[expert_idx]
        old_dim = old_expert[0].out_features
        new_dim = old_dim + expansion_size
        new_l1 = nn.Linear(self.hidden_dim, new_dim).cuda()
        new_l2 = nn.Linear(new_dim, self.hidden_dim).cuda()
        new_l1.weight.data[:old_dim, :] = old_expert[0].weight.data
        new_l1.bias.data[:old_dim] = old_expert[0].bias.data
        new_l2.weight.data[:, :old_dim] = old_expert[2].weight.data
        new_l2.bias.data = old_expert[2].bias.data
        self.knw_experts[expert_idx] = nn.Sequential(new_l1, nn.GELU(), new_l2, nn.LayerNorm(self.hidden_dim).cuda())
        
        self.knw_locked_dims[expert_idx] = old_dim 
        print(f"📏 [知识库 Expand] K{expert_idx+1} 知识补丁增长: {old_dim} -> {new_dim} (前 {old_dim} 维已被加锁保护)")

    def apply_knw_gradient_mask(self, expert_idx):
        locked_dim = self.knw_locked_dims[expert_idx]
        if locked_dim > 0:
            with torch.no_grad():
                expert = self.knw_experts[expert_idx]
                if expert[0].weight.grad is not None:
                    expert[0].weight.grad[:locked_dim, :] = 0
                    expert[0].bias.grad[:locked_dim] = 0
                if expert[2].weight.grad is not None:
                    expert[2].weight.grad[:, :locked_dim] = 0

    def generate_combined(self, prompt, H_base, max_new_tokens=60):
        alpha, best_gen_exp = 0.0, -1
        if self.active_experts > 0:
            sims_e = [F.cosine_similarity(H_base, self.expert_centroids[i].unsqueeze(0)).item() for i in range(self.active_experts)]
            best_gen_exp, alpha = sims_e.index(max(sims_e)), max(max(sims_e), 0.0)
            
        beta, best_knw_exp = 0.0, -1
        if self.knw_active_experts > 0:
            sims_k = [F.cosine_similarity(H_base, self.knw_centroids[i].unsqueeze(0)).item() for i in range(self.knw_active_experts)]
            best_knw_exp, beta = sims_k.index(max(sims_k)), max(max(sims_k), 0.0)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        past_key_values = None
        generated_tokens = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = base_llm(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
                past_key_values = outputs.past_key_values
                next_token_hidden = outputs.hidden_states[-1][:, -1:, :]
            
            H_res_gen = self.experts[best_gen_exp](next_token_hidden.to(torch.float32)) if best_gen_exp != -1 else 0
            H_res_knw = self.knw_experts[best_knw_exp](next_token_hidden.to(torch.float32)) if best_knw_exp != -1 else 0
            
            residual = alpha * H_res_gen + beta * H_res_knw
            if isinstance(residual, torch.Tensor):
                modified_hidden = next_token_hidden + self.alpha_residual * residual.to(next_token_hidden.dtype)
            else:
                modified_hidden = next_token_hidden
            
            with torch.no_grad():
                logits = base_llm.lm_head(modified_hidden)
                probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id: break
            input_ids = next_token
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def differentiate_cycle(self):
        self.action_labels = []
        self.immediate_goals = []
        for i in range(2):
            self.immediate_goals.append(self.C_t.detach() + torch.randn_like(self.C_t) * 0.15)
            self.action_labels.append(f"聊天分身_{i+1}")
        self.active_preferences = []
        now = time.time()
        if len(self.episodic_buffer) > 0:
            scored = [(p['vec'], torch.exp(torch.tensor(-self.forget_rate * (now - p['timestamp']))), p['text'])
                      for p in self.episodic_buffer]
            scored.sort(key=lambda x: x[1], reverse=True)
            for v, weight, txt in scored[:2]:
                self.active_preferences.append(v * weight.cuda())
                self.action_labels.append(f"记忆偏好_{txt[:6]}")
        for act in self.active_actions:
            self.immediate_goals.append(self._get_emb(act))
            self.action_labels.append(f"主动_{act}")
        all_vecs = self.immediate_goals + self.active_preferences
        self.current_action_vectors = torch.cat(all_vecs, dim=0).cuda()
        num_actions = self.current_action_vectors.size(0)
        self.goal_fatigue = torch.zeros(num_actions).cuda()
        self.action_counts = torch.zeros(num_actions).cuda()
        self.last_Q = torch.zeros(num_actions).cuda()
        self.day_start_time = time.time()

    def _update_memory(self, H_base, text, raw_Q):
        now = time.time()
        new_L = min(max(raw_Q.max().item(), 0.0), 5.0)
        if len(self.episodic_buffer) > 0:
            mem_vecs = torch.cat([m['vec'] for m in self.episodic_buffer], dim=0)
            sims = F.cosine_similarity(H_base, mem_vecs)
            max_sim, max_idx = torch.max(sims, dim=0)
            if max_sim.item() > 0.85:
                self.episodic_buffer[max_idx.item()]['timestamp'] = now
                old_L = self.episodic_buffer[max_idx.item()]['L']
                self.episodic_buffer[max_idx.item()]['L'] = min(old_L + 0.5 * new_L, 5.0)
            else:
                self.episodic_buffer.append({"vec": H_base.detach(), "timestamp": now, "text": text, "L": new_L})
        else:
            self.episodic_buffer.append({"vec": H_base.detach(), "timestamp": now, "text": text, "L": new_L})
        survivors = []
        for p in self.episodic_buffer:
            weight = torch.exp(torch.tensor(-self.forget_rate * (now - p['timestamp']))).item()
            if weight >= 0.001: survivors.append(p)
        if len(survivors) > 2000:
            survivors.sort(key=lambda x: torch.exp(torch.tensor(-self.forget_rate * (now - x['timestamp']))).item() * x['L'], reverse=True)
            survivors = survivors[:2000]
        self.episodic_buffer = survivors

    def forward(self, text):
        if time.time() - self.day_start_time > self.cycle_seconds or self.last_Q is None: self.differentiate_cycle()
        H_base = self._get_emb(text)
        scale_factor = (self.hidden_dim ** 0.5)
        U_i = torch.matmul(H_base, self.current_action_vectors.T) / scale_factor
        
        topk_texts, Q_assoc = [], 0
        if len(self.episodic_buffer) > 0:
            mem_vecs = torch.cat([m['vec'] for m in self.episodic_buffer], dim=0)
            sims = F.cosine_similarity(H_base, mem_vecs)
            k = min(3, len(self.episodic_buffer))
            top_sims, top_idx = torch.topk(sims, k)
            assoc_energy = 0
            for i in range(k):
                idx = top_idx[i].item()
                assoc_energy += top_sims[i] * self.episodic_buffer[idx]['L']
                topk_texts.append(self.episodic_buffer[idx]['text'])
            Q_assoc = self.eta * assoc_energy * U_i
            
        mode, e_idx, m_sim = self.check_satiety(H_base)
        chi_score = self.chi_function(mode, H_base, self.expert_centroids[e_idx] if e_idx >= 0 and self.expert_centroids is not None else None)
        Q_chi = chi_score * U_i
        f_fatigue = torch.exp(-self.goal_fatigue)
        D_penalty = 1.0 - torch.clamp(self.action_counts * 0.1, max=0.4)
        
        Q_pref = torch.zeros_like(U_i) 
        if len(self.active_preferences) > 0:
            pref_mat = torch.cat(self.active_preferences, dim=0)
            pref_scores = torch.matmul(H_base, pref_mat.T) / scale_factor * self.zeta
            Q_pref[0, len(self.immediate_goals):len(self.immediate_goals)+len(self.active_preferences)] = pref_scores[0]
            
        Q_meta = self.lambda_meta * U_i - self.phi_meta * self.goal_fatigue.unsqueeze(0)
        novelty_gate = 1.6 if mode != "DROP" else 0.1
        Q_structure = self.omega_z * U_i * novelty_gate
        Q_now = self.kappa * U_i * f_fatigue * D_penalty + Q_pref + Q_meta + Q_structure + Q_assoc + Q_chi
        
        # =========================================================
        # [修改点：植入真实生物钟] 在算分完毕后进行物理加权！
        # =========================================================
        current_hour = datetime.now().hour
        for i, label in enumerate(self.action_labels):
            if "休眠" in label and 0 <= current_hour < 6:
                Q_now[0, i] += 10.0  # 极其暴力的凌晨拉升
            elif "发呆" in label and 14 <= current_hour < 16:
                Q_now[0, i] += 10.0  # 极其暴力的午后拉升
        # =========================================================

        if self.active_experts > 0:
            sims_e = [F.cosine_similarity(H_base, self.expert_centroids[i].unsqueeze(0)) for i in range(self.active_experts)]
            best_exp = sims_e.index(max(sims_e))
            expert_out = self.experts[best_exp](H_base)
            H_residual = self.alpha_residual * expert_out
            Q_now = Q_now + torch.matmul(H_residual, self.current_action_vectors.T) / scale_factor
            
        Q = self.rho * self.last_Q + (1.0 - self.rho) * Q_now
        self.last_Q = Q.detach()  
        P_A = F.softmax(Q / self.tau, dim=-1)
        return P_A, H_base, (mode, e_idx, m_sim, topk_texts), Q

# ==========================================
# 3. 持久化与辅助函数
# ==========================================
def save_mind(mind, path):
    torch.save({
        'expert_configs': [exp[0].out_features for exp in mind.experts],
        'knw_configs': [exp[0].out_features for exp in mind.knw_experts],
        'state_dict': mind.state_dict(),
        'buffer': mind.episodic_buffer[-100:], 
        'anchors': mind.expert_anchors,
        'centroids': mind.expert_centroids,
        'knw_anchors': mind.knw_anchors,
        'knw_centroids': mind.knw_centroids,
        'knw_locked_dims': mind.knw_locked_dims
    }, path)

def load_mind(mind, path):
    cp = torch.load(path)
    for dim in cp['expert_configs']:
        mind.experts.append(nn.Sequential(nn.Linear(mind.hidden_dim, dim), nn.GELU(), nn.Linear(dim, mind.hidden_dim), nn.LayerNorm(mind.hidden_dim)).cuda())
    for dim in cp.get('knw_configs', []):
        mind.knw_experts.append(nn.Sequential(nn.Linear(mind.hidden_dim, dim), nn.GELU(), nn.Linear(dim, mind.hidden_dim), nn.LayerNorm(mind.hidden_dim)).cuda())
    mind.load_state_dict(cp['state_dict'])
    mind.episodic_buffer, mind.expert_anchors, mind.expert_centroids = cp['buffer'], cp['anchors'], cp['centroids']
    mind.knw_anchors, mind.knw_centroids, mind.knw_locked_dims = cp.get('knw_anchors', []), cp.get('knw_centroids', None), cp.get('knw_locked_dims', [])
    mind.active_experts = len(mind.experts)
    mind.knw_active_experts = len(mind.knw_experts)

def get_optimizer(mind):
    return torch.optim.Adam([p for p in mind.parameters() if p.requires_grad], lr=5e-5, weight_decay=1e-4)

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    mind = UnifiedMindSystem(HIDDEN_DIM).cuda()
    if os.path.exists(save_path): load_mind(mind, save_path)
    mind.differentiate_cycle()
    optimizer = get_optimizer(mind)

    while True:
        # [修改点：菜单栏增加了选项5]
        print("\n" + "="*80 + "\n1. [离线学泛化] | 2. [响应] | 3. [自发流] | 4. [刻录硬知识] | 5. [调整活跃度] | exit\n" + "="*80)
        cmd = input(">> ")
        
        if cmd == '1':
            if not os.path.exists(corpus_path):
                print("⚠️ 找不到 training_corpus.txt")
                continue
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_lines = [line.strip() for line in f if line.strip()]
            for idx, text in enumerate(corpus_lines):
                try:
                    if not mind.cognitive_judge(text): continue
                    if F.cosine_similarity(mind._get_emb(text), mind.C_matrix).max().item() < 0.45: continue
                    optimizer.zero_grad()
                    probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind(text)
                    if mode == "SPAWN": 
                        mind.spawn_expert(H_base, text)
                        optimizer = get_optimizer(mind)
                    elif mode == "MERGE":
                        mind.expert_hits[e_idx] += 1
                        if mind.expert_hits[e_idx] >= 5:
                            mind.expand_expert(e_idx)
                            optimizer = get_optimizer(mind)
                            mind.expert_hits[e_idx] = 0
                    if mode != "DROP":
                        loss = mind.learn_from_text(text, e_idx if e_idx != -1 else mind.active_experts-1)
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(mind.parameters(), max_norm=1.0)
                            optimizer.step()
                            print(f"[{idx+1}/{len(corpus_lines)}] 🎭 [泛化塑形] Gen-Ex Loss: {loss.item():.4f}")
                    save_mind(mind, save_path)
                    time.sleep(0.5) 
                except KeyboardInterrupt: break
                except Exception: continue
            print("\n🏁 泛化层学习完毕！")

        # =========================================================
        # [修改点：彻底分流 cmd=2(单次被动)]
        # 1. 保留底层演算与记忆提取 
        # 2. 阻断抽卡与动作污染
        # 3. 纯净 Prompt 注入
        # 4. 照常更新记忆库
        # 5. 不增加特定动作的疲劳度
        # =========================================================
        elif cmd == '2':
            with torch.no_grad():
                text = input("[用户问询]: ")
                
                # 1. 保留底层演算与记忆提取 (计算过程内部不变，状态继续演进)
                probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind(text)
                
                # 2. 阻断抽卡：移除 torch.multinomial 和意图获取
                print(f"🧠 联想记忆: {topk}")
                
                memory_context = "\n".join(topk) if len(topk) > 0 else "无"
                
                # 3. 纯净 Prompt 注入：只有人设、记忆联想、提问，彻底删除动作驱动 intent
                prompt = (
                    f"<|im_start|>system\n"
                    f"{mind.persona_text}\n"
                    f"【近期记忆联想】：\n{memory_context}<|im_end|>\n"
                    f"<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                
                # 核心保留：通过泛化专家层+知识专家层生成
                response_text = mind.generate_combined(prompt, H_base)
                print(f"🗣️ AI: {response_text}")
                
                # 4. 照常更新记忆库
                mind._update_memory(H_base, text, raw_Q)
                
                # 5. 不增加特定动作的疲劳度：删除了 mind.goal_fatigue 和 mind.action_counts 累加逻辑

        elif cmd == '3':
            print("🌊 [进入自发行为流] 系统不再等待输入，根据生物钟与意图抽卡 (输入 'q' 退出闭环)...")
            current_trigger_text = "主动触发"
            
            while True:
                with torch.no_grad():
                    probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind(current_trigger_text)
                    idx = torch.multinomial(probs[0], 1).item()
                    intent = mind.action_labels[idx]
                    print(f"\n🧠 联想记忆: {topk}\n📊 自发意图:【{intent}】")
                    
                    clean_intent = intent.split('_')[-1] if '_' in intent else intent
                    memory_context = "\n".join(topk) if len(topk) > 0 else "无"
                    
                    prompt = (
                        f"<|im_start|>system\n"
                        f"{mind.persona_text}\n"
                        f"【近期记忆联想】：\n{memory_context}\n"
                        f"【你此刻的真实状态驱动】：{clean_intent}<|im_end|>\n"
                        f"<|im_start|>user\n{current_trigger_text}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    
                    response_text = mind.generate_combined(prompt, H_base)
                    print(f"🗣️ AI (自发): {response_text}")
                    
                    mind._update_memory(H_base, current_trigger_text, raw_Q)
                    mind.goal_fatigue[idx] += mind.fatigue_rate
                    mind.action_counts[idx] += 1.0

                # 形成用户回复闭环
                user_reply = input("\n[你的回复 / 输入 q 退出]: ")
                if user_reply.strip().lower() == 'q':
                    print("💤 满足退出条件，退出自发流，回归基态。")
                    break
                current_trigger_text = user_reply
        # =========================================================

        elif cmd == '4':
            knw_files = [os.path.join(knw_corpus_dir, f) for f in os.listdir(knw_corpus_dir) if f.endswith('.txt')]
            if not knw_files:
                print(f"⚠️ 文件夹 {knw_corpus_dir} 为空！请放入专业知识/API文档的 txt 文件后重试。")
                continue
                
            print(f"📚 发现 {len(knw_files)} 个知识文件，开启纯净左脑刻录...")
            for file_path in knw_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if len(line.strip()) > 5] 
                
                for idx, text in enumerate(lines):
                    try:
                        optimizer.zero_grad()
                        H_base = mind._get_emb(text)
                        
                        mode, knw_e_idx, m_sim = mind.check_knw_satiety(H_base)
                        
                        if mode == "SPAWN": 
                            mind.spawn_knw_expert(H_base, text)
                            optimizer = get_optimizer(mind)
                            knw_e_idx = mind.knw_active_experts - 1
                        elif mode == "EXPAND":
                            mind.expand_knw_expert(knw_e_idx)
                            optimizer = get_optimizer(mind)
                            
                        if mode != "DROP":
                            loss = mind.learn_from_knw_text(text, knw_e_idx)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                loss.backward()
                                mind.apply_knw_gradient_mask(knw_e_idx)
                                torch.nn.utils.clip_grad_norm_(mind.parameters(), max_norm=1.0)
                                optimizer.step()
                                print(f"[{idx+1}/{len(lines)}] 💽 [硬知识刻录] Knw-Ex{knw_e_idx+1} Loss: {loss.item():.4f} (模式:{mode})")
                        
                        save_mind(mind, save_path)
                        time.sleep(0.3)
                    except KeyboardInterrupt: break
                    except Exception as e: continue
            print("\n🏁 左脑硬核知识刻录完毕！物理锁已封死旧年轮。")

        # =========================================================
        # [新增：多巴胺阈值映射]
        # =========================================================
        elif cmd == '5':
            print("\n请设置 AI 的自发活跃度/多巴胺阈值 (1: 极度高冷，可能几天不说话 ~ 10: 极度粘人，是个话痨)")
            val_str = input(">> ")
            if val_str.isdigit():
                val = int(val_str)
                val = max(1, min(10, val)) # 限制在 1-10 范围
                mind.dopamine_level = val
                
                # 底层数学映射
                # 触发周期 Cycle Seconds：高冷时长，话痨时短
                mind.cycle_seconds = int(3600 / (val ** 2)) 
                
                # 疲劳衰减率 Decay Rate：高冷累积快，话痨累积慢
                mind.fatigue_rate = 0.35 * (11 - val) / 5.0 
                
                print(f"⚙️ 底层参数映射成功！")
                print(f"   ► 多巴胺阈值: {mind.dopamine_level}")
                print(f"   ► 自发触发周期: 每 {mind.cycle_seconds} 秒")
                print(f"   ► 疲劳累积系数: 每次 {mind.fatigue_rate:.2f}")
            else:
                print("⚠️ 输入无效，请保持原状态。")
        # =========================================================

        elif cmd == 'exit': break

if __name__ == "__main__": main()