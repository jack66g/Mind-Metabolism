import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import time

# ==========================================
# 1. 环境初始化
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "mode")
save_path = os.path.join(current_dir, "mind_state.pth")

print("正在唤醒认知底座 (Qwen2.5-AWQ)...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_llm = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16
).eval()
for param in base_llm.parameters(): param.requires_grad = False
HIDDEN_DIM = base_llm.config.hidden_size 

# ==========================================
# 2. 核心系统 (动力学全满 + 物理装甲全满)
# ==========================================
class UnifiedMindSystem(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.structure_changed = False 
        
        core_traits = [
            "性格底色：温柔亲密，情绪稳定，带有轻松的幽默感。",
            "爱情观：渴望长久稳定的陪伴，对感情极其细腻且专一。",
            "生活观：喜欢在平凡的日常中寻找小确幸，热爱大自然。",
            "价值观：极度排斥暴力、粗俗、负能量暴言和虚无主义。"
        ]
        traits_embs = [self._get_emb(t) for t in core_traits]
        self.C_matrix = nn.Parameter(torch.cat(traits_embs, dim=0), requires_grad=False)
        self.C_t = nn.Parameter(self.C_matrix.mean(dim=0, keepdim=True), requires_grad=False)
        
        self.immediate_goals = []      
        self.active_preferences = []   
        self.current_action_vectors = None 
        self.action_labels = [] 
        self.active_actions = ["主动问候", "主动闲聊", "发呆", "分享日常"]
        
        self.active_experts = 0
        self.experts = nn.ModuleList([]) 
        self.expert_centroids = None     
        self.expert_anchors = []
        self.expert_hits = []            
        
        self.episodic_buffer = [] 
        self.last_Q = None        
        self.rho = 0.8            
        
        self.forget_rate = 0.004
        self.day_start_time = time.time()
        self.cycle_seconds = 60  
        self.goal_fatigue = None
        self.action_counts = None
        self.tau = 1.3
        
        self.kappa = nn.Parameter(torch.tensor(1.2))
        self.zeta = nn.Parameter(torch.tensor(1.1))
        self.eta = nn.Parameter(torch.tensor(0.5))    
        self.lambda_meta = nn.Parameter(torch.tensor(0.5))
        self.phi_meta = nn.Parameter(torch.tensor(0.3))
        self.omega_z = nn.Parameter(torch.tensor(1.6))
        self.alpha_residual = nn.Parameter(torch.tensor(0.8)) #可调，但是现在不支持整数

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
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def generate_with_expert(self, prompt, expert_idx, max_new_tokens=50):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        past_key_values = None
        generated_tokens = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = base_llm(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
                past_key_values = outputs.past_key_values
                next_token_hidden = outputs.hidden_states[-1][:, -1:, :]
            if expert_idx != -1 and self.active_experts > 0:
                expert_net = self.experts[expert_idx]
                residual = expert_net(next_token_hidden.to(torch.float32))
                modified_hidden = next_token_hidden + self.alpha_residual * residual.to(next_token_hidden.dtype)
            else:
                modified_hidden = next_token_hidden
            with torch.no_grad():
                logits = base_llm.lm_head(modified_hidden)
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id: break
            input_ids = next_token
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def spawn_expert(self, H_base, anchor_text, initial_dim=4):
        new_expert = nn.Sequential(
            nn.Linear(self.hidden_dim, initial_dim),
            nn.GELU(),
            nn.Linear(initial_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).cuda()
        self.experts.append(new_expert)
        self.expert_anchors.append(anchor_text[:15])
        self.expert_hits.append(0)
        if self.expert_centroids is None:
            self.expert_centroids = H_base.detach().clone()
        else:
            self.expert_centroids = torch.cat([self.expert_centroids, H_base.detach()], dim=0)
        self.active_experts += 1
        self.structure_changed = True 
        print(f"🎇 [异构开荒 Spawn] 专家 E{self.active_experts} 诞生！初始矩阵: {self.hidden_dim}x{initial_dim}")

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
        self.structure_changed = True 
        print(f"📈 [领域拓展 Matrix Expansion] 专家 E{expert_idx+1} 知识扩容！规模: {old_dim} -> {new_dim}")

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
        elif mode == "MERGE":
            return F.cosine_similarity(H_in, W_m.unsqueeze(0)).item()
        elif mode == "EXPAND":
            return 0.6 + 0.4 * F.cosine_similarity(H_in, W_m.unsqueeze(0)).item()
        elif mode == "SPAWN":
            dist = 1.0 - F.cosine_similarity(H_in, W_m.unsqueeze(0)).item() if W_m is not None else 1.0
            return 0.8 + 0.4 * dist
        return 0.0

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

    def forward(self, text):
        if time.time() - self.day_start_time > self.cycle_seconds or self.last_Q is None:
            self.differentiate_cycle()
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
        # Q_chi 专家行为贡献
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
        
        # Q_now 整合 Q_chi
        Q_now = self.kappa * U_i * f_fatigue * D_penalty + Q_pref + Q_meta + Q_structure + Q_assoc + Q_chi
        
        best_exp = -1
        if self.active_experts > 0:
            sims_e = [F.cosine_similarity(H_base, self.expert_centroids[i].unsqueeze(0)) for i in range(self.active_experts)]
            best_exp = sims_e.index(max(sims_e))
            expert_out = self.experts[best_exp](H_base)
            # 专家残差影响
            H_residual = self.alpha_residual * expert_out
            Q_now = Q_now + torch.matmul(H_residual, self.current_action_vectors.T) / scale_factor
            
        Q = self.rho * self.last_Q + (1.0 - self.rho) * Q_now
        self.last_Q = Q.detach()  
        P_A = F.softmax(Q / self.tau, dim=-1)
        return P_A, H_base, (mode, e_idx, m_sim, best_exp, chi_score, topk_texts), Q

# ==========================================
# 3. 持久化与辅助函数
# ==========================================
def save_mind(mind, path):
    torch.save({
        'expert_configs': [exp[0].out_features for exp in mind.experts],
        'state_dict': mind.state_dict(),
        'buffer': mind.episodic_buffer[-100:], 
        'anchors': mind.expert_anchors,
        'hits': mind.expert_hits,
        'centroids': mind.expert_centroids
    }, path)

def load_mind(mind, path):
    cp = torch.load(path)
    for dim in cp['expert_configs']:
        mind.experts.append(nn.Sequential(nn.Linear(mind.hidden_dim, dim), nn.GELU(), nn.Linear(dim, mind.hidden_dim), nn.LayerNorm(mind.hidden_dim)).cuda())
    mind.load_state_dict(cp['state_dict'])
    mind.episodic_buffer, mind.expert_anchors, mind.expert_hits, mind.expert_centroids = cp['buffer'], cp['anchors'], cp['hits'], cp['centroids']
    mind.active_experts = len(mind.experts)

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
        print("\n" + "="*70 + "\n1. [学习] | 2. [响应] | 3. [自发] | exit\n" + "="*70)
        cmd = input(">> ")
        if cmd == '1':
            try:
                while True:
                    try:
                        text = requests.get("https://v1.hitokoto.cn/", timeout=3).json()['hitokoto']
                        if not mind.cognitive_judge(text): continue
                        if F.cosine_similarity(mind._get_emb(text), mind.C_matrix).max().item() < 0.45: continue
                    except: time.sleep(2); continue
                    optimizer.zero_grad()
                    probs, H_base, (mode, e_idx, m_sim, b_exp, chi, topk), raw_Q = mind(text)
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
                        target = b_exp if b_exp != -1 else (mind.active_experts - 1)
                        if target != -1:
                            loss = mind.learn_from_text(text, target)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(mind.parameters(), max_norm=1.0)
                                optimizer.step()
                                print(f"⚡ [突触更新] E{target+1} Loss: {loss.item():.4f}")
                    mind.episodic_buffer.append({"vec": H_base.detach(), "timestamp": time.time(), "text": text, "L": min(max(raw_Q.max().item(), 0.0), 5.0)})
                    save_mind(mind, save_path); time.sleep(1)
            except KeyboardInterrupt: pass
        elif cmd in ['2', '3']:
            with torch.no_grad():
                text = input("[用户问询]: ") if cmd == '2' else "主动触发"
                probs, H_base, (mode, e_idx, m_sim, b_exp, chi, topk), raw_Q = mind(text)
                idx = torch.multinomial(probs[0], 1).item()
                intent = mind.action_labels[idx]
                print(f"🧠 联想: {topk}\n📊 意图:【{intent}】 | 专家: E{b_exp+1}")
                prompt = f"<|im_start|>system\n你是一个温柔的女生。根据意图回复：{intent}。<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
                print(f"🗣️ AI: {mind.generate_with_expert(prompt, b_exp)}")
                mind.episodic_buffer.append({"vec": H_base.detach(), "timestamp": time.time(), "text": text, "L": min(max(raw_Q.max().item(), 0.0), 5.0)})
                mind.goal_fatigue[idx] += 0.35
                mind.action_counts[idx] += 1.0
        elif cmd == 'exit': break

if __name__ == "__main__": main()