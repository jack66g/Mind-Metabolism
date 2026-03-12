import os
import sys           # [新增] 用于终端的安全输出控制
import threading     # [新增] 用于真正的后台独立心跳线程
import random        # [新增] 用于多巴胺掷骰子
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
from datetime import datetime # 用于获取真实系统时间以驱动生物钟
import asyncio

# ==========================================
# [新增] FastAPI 现代化服务端依赖
# ==========================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ==========================================
# 0. 全局状态锁与系统状态 (防并发/防精神分裂)
# ==========================================
SYSTEM_STATE = "IDLE"           # 可选值: IDLE, USER_BUSY, WAITING_FOR_REPLY, GENERATING, TRAINING
AUTONOMOUS_ACTIVE = False       # 控制自发引擎是否开启的全局开关
LAST_AUTO_MSG_TIME = time.time()# 记录AI最后一次主动说话的时间，用于“冷落生气”机制

# [新增] WebSocket 客户端池
connected_clients = set()

def safe_print(msg):
    """防止后台线程打印时，切断或顶乱主线程输入框的防撕裂函数"""
    sys.stdout.write('\r' + ' '*80 + '\r') # 清空当前行
    print(msg)
    sys.stdout.flush()

# [新增] 异步推送消息给所有前端
async def broadcast_ws(msg_dict):
    for client in list(connected_clients):
        try:
            await client.send_json(msg_dict)
        except Exception:
            connected_clients.remove(client)

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
# 2. 核心系统 (双轨制：泛化层 + 知识层) -> 绝对未动
# ==========================================
class UnifiedMindSystem(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.structure_changed = False 
        
        if not os.path.exists(persona_path):
            with open(persona_path, 'w', encoding='utf-8') as f:
                f.write("你是一个温柔、情绪稳定的女生。请用自然、亲密的语气聊天。\n")
                
        with open(persona_path, 'r', encoding='utf-8') as f:
            core_traits = [line.strip() for line in f if line.strip()]
            
        if not core_traits: core_traits = ["系统默认AI"] 
        self.persona_text = "\n".join(core_traits)
        fixed_actions = ["发呆", "伸懒腰"] 

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
        
        self.dopamine_level = 5           
        self.cycle_seconds = 144          
        self.fatigue_rate = 0.35          

        self.goal_fatigue = None
        self.action_counts = None
        self.tau = 1.3
        
        self.kappa = nn.Parameter(torch.tensor(1.2))
        self.zeta = nn.Parameter(torch.tensor(1.1))
        self.eta = nn.Parameter(torch.tensor(0.5))    
        self.lambda_meta = nn.Parameter(torch.tensor(0.5))
        self.phi_meta = nn.Parameter(torch.tensor(0.3))
        self.omega_z = nn.Parameter(torch.tensor(1.6))
        self.alpha_residual = nn.Parameter(torch.tensor(0.9))

        self.active_experts = 0
        self.experts = nn.ModuleList([]) 
        self.expert_centroids = None     
        self.expert_anchors = []
        self.expert_hits = []            

        self.max_knw_experts = 5  
        self.knw_active_experts = 0
        self.knw_experts = nn.ModuleList([]) 
        self.knw_centroids = None     
        self.knw_anchors = []
        self.knw_locked_dims = [] 
        self.knw_buffers = [] 

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
        print(f"📏 [知识库 Expand] K{expert_idx+1} 知识补丁增长: {old_dim} -> {new_dim}")

    def defrag_knw_expert(self, expert_idx, base_optimizer):
        buffer = self.knw_buffers[expert_idx]
        if not buffer: return
        print(f"🌀 [小乐协议触发] 正在解冻 K{expert_idx+1} 矩阵，执行特征空间对齐与碎片整理...")
        temp_lock = self.knw_locked_dims[expert_idx]
        self.knw_locked_dims[expert_idx] = 0 
        for param_group in base_optimizer.param_groups:
            param_group['lr'] = 1e-6
        recent_memories = buffer[-16:] 
        history_pool = buffer[:-16]
        history_memories = random.sample(history_pool, min(len(history_pool), 16))
        replay_batch = recent_memories + history_memories
        for past_text in replay_batch: 
            base_optimizer.zero_grad()
            loss = self.learn_from_knw_text(past_text, expert_idx)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                self.apply_knw_gradient_mask(expert_idx) 
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                base_optimizer.step()
        self.knw_locked_dims[expert_idx] = temp_lock
        for param_group in base_optimizer.param_groups:
            param_group['lr'] = 5e-5
        print(f"✨ [重算完毕] K{expert_idx+1} 旧知识压缩重组完成！")

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

    def generate_combined(self, prompt, H_base, max_new_tokens=500):
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
        
        current_hour = datetime.now().hour
        for i, label in enumerate(self.action_labels):
            if "休眠" in label and 0 <= current_hour < 6:
                Q_now[0, i] += 10.0  
            elif "发呆" in label and 14 <= current_hour < 16:
                Q_now[0, i] += 10.0  

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
        'knw_locked_dims': mind.knw_locked_dims,
        'knw_buffers': mind.knw_buffers 
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
    mind.knw_buffers = cp.get('knw_buffers', [[] for _ in range(mind.knw_active_experts)]) 

def get_optimizer(mind):
    return torch.optim.Adam([p for p in mind.parameters() if p.requires_grad], lr=5e-5, weight_decay=1e-4)

# ==========================================
# 4. 核心心跳守护线程 (增加 WebSocket 推送)
# ==========================================
def autonomous_heartbeat(mind, event_loop):
    global SYSTEM_STATE, AUTONOMOUS_ACTIVE, LAST_AUTO_MSG_TIME
    
    while True:
        time.sleep(10) 
        
        if not AUTONOMOUS_ACTIVE:
            continue
            
        if SYSTEM_STATE in ["USER_BUSY", "GENERATING", "TRAINING"]:
            continue 

        now_time = time.time()
        current_hour_str = datetime.now().strftime("%H:%M")
        trigger_text = mind.persona_text[:20] if len(mind.persona_text) > 0 else "自我意识"
        
        # 机制 A：被冷落机制
        if SYSTEM_STATE == "WAITING_FOR_REPLY" and (now_time - LAST_AUTO_MSG_TIME > 1800):
            SYSTEM_STATE = "GENERATING" 
            with torch.no_grad():
                probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind(trigger_text)
                prompt = (
                    f"<|im_start|>system\n"
                    f"{mind.persona_text}\n"
                    f"【当前系统真实时间】：{current_hour_str}\n"
                    f"【最高优先级指令】：你半小时前主动找用户聊天，但用户完全无视了你。请结合你的人设，发泄一点小脾气或进行抱怨。绝对不允许使用客服语气！<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                response_text = mind.generate_combined(prompt, H_base)
                safe_print(f"\n💢 [超时未回复触发] \n🗣️ AI (生气): {response_text}")
                mind._update_memory(H_base, f"我因为被冷落而生气抱怨：{response_text}", raw_Q)
                
                # [新增] 异步推送到前端主动弹窗
                asyncio.run_coroutine_threadsafe(
                    broadcast_ws({"type": "proactive_msg", "text": response_text}), event_loop
                )
            
            LAST_AUTO_MSG_TIME = time.time()
            SYSTEM_STATE = "IDLE" 
            continue

        # 机制 B：空闲多巴胺触发
        if SYSTEM_STATE == "IDLE":
            prob = 10.0 / max(10.0, float(mind.cycle_seconds)) 
            if random.random() < prob:
                SYSTEM_STATE = "GENERATING" 
                with torch.no_grad():
                    probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind(trigger_text)
                    idx = torch.multinomial(probs[0], 1).item()
                    intent = mind.action_labels[idx]
                    clean_intent = intent.split('_')[-1] if '_' in intent else intent
                    memory_context = "\n".join(topk) if len(topk) > 0 else "无"
                    
                    prompt = (
                        f"<|im_start|>system\n"
                        f"{mind.persona_text}\n"
                        f"【近期记忆联想】：\n{memory_context}\n"
                        f"【当前系统真实时间】：{current_hour_str}\n"
                        f"【最高优先级指令】：用户当前处于沉默状态。你此刻的身体本能和核心驱动力是【{clean_intent}】。请务必、立刻在接下来的回答中，用符合你性格的语气表现出这种生理/心理状态！绝对不允许像客服一样嘘寒问暖！<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    
                    response_text = mind.generate_combined(prompt, H_base)
                    safe_print(f"\n🧠 [后台掷骰成功! 当前权重最高意图]: {intent}\n🗣️ AI (主动): {response_text}")
                    mind._update_memory(H_base, f"我主动对你说：{response_text}", raw_Q)
                    
                    mind.goal_fatigue[idx] += mind.fatigue_rate
                    mind.action_counts[idx] += 1.0
                    
                    # [新增] 异步推送到前端主动弹窗
                    asyncio.run_coroutine_threadsafe(
                        broadcast_ws({"type": "proactive_msg", "text": response_text}), event_loop
                    )
                    
                LAST_AUTO_MSG_TIME = time.time()
                SYSTEM_STATE = "WAITING_FOR_REPLY"


# ==========================================
# 5. [重构核心] FastAPI Web 服务
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

mind_instance = None
optimizer_instance = None

# ---- 请求模型 ----
class ChatRequest(BaseModel): text: str
class TrainRequest(BaseModel): type: str
class SettingsRequest(BaseModel): 
    autonomous: bool = None
    dopamine: int = None

@app.on_event("startup")
def startup_event():
    global mind_instance, optimizer_instance
    mind_instance = UnifiedMindSystem(HIDDEN_DIM).cuda()
    if os.path.exists(save_path): load_mind(mind_instance, save_path)
    mind_instance.differentiate_cycle()
    optimizer_instance = get_optimizer(mind_instance)

    loop = asyncio.get_event_loop()
    heartbeat_thread = threading.Thread(target=autonomous_heartbeat, args=(mind_instance, loop), daemon=True)
    heartbeat_thread.start()
    print("\n🚀 CogniCore OS 后端已启动！等待前端连接 (ws://localhost:8000/ws) ...\n")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    global SYSTEM_STATE, LAST_AUTO_MSG_TIME
    SYSTEM_STATE = "USER_BUSY"
    
    text = req.text
    # 后门指令拦截
    if text.strip().lower() in ['q', '滚', '退下', '结束']:
        SYSTEM_STATE = "IDLE"
        LAST_AUTO_MSG_TIME = time.time()
        safe_print("🔇 已强制结束话题，AI 回归安静挂机状态。")
        return {"reply": "[已退下，AI 回归安静挂机状态]", "type": "system"}

    with torch.no_grad():
        probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind_instance(text)
        safe_print(f"🧠 联想记忆: {topk}")
        
        memory_context = "\n".join(topk) if len(topk) > 0 else "无"
        current_hour_str = datetime.now().strftime("%H:%M") 
        
        prompt = (
            f"<|im_start|>system\n"
            f"{mind_instance.persona_text}\n"
            f"【近期记忆联想】：\n{memory_context}\n"
            f"【当前系统真实时间】：{current_hour_str}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        response_text = mind_instance.generate_combined(prompt, H_base)
        safe_print(f"🗣️ AI: {response_text}")
        
        # [核心保留] 完美闭环的上下文刻录
        mind_instance._update_memory(H_base, f"用户对我说：{text}\n我的回答是：{response_text}", raw_Q)

    LAST_AUTO_MSG_TIME = time.time()
    SYSTEM_STATE = "WAITING_FOR_REPLY"
    return {"reply": response_text, "type": "chat"}

@app.post("/settings")
def update_settings(req: SettingsRequest):
    global AUTONOMOUS_ACTIVE, SYSTEM_STATE
    if req.autonomous is not None:
        AUTONOMOUS_ACTIVE = req.autonomous
        if AUTONOMOUS_ACTIVE:
            safe_print("🟢 [系统提示]: AI 自主意识已开启！")
            SYSTEM_STATE = "IDLE"
        else:
            safe_print("🔴 [系统提示]: AI 自主意识已关闭！")
            
    if req.dopamine is not None:
        SYSTEM_STATE = "USER_BUSY"
        mind_instance.dopamine_level = req.dopamine
        mind_instance.cycle_seconds = int(3600 / (req.dopamine ** 2)) 
        mind_instance.fatigue_rate = 0.35 * (11 - req.dopamine) / 5.0 
        safe_print(f"⚙️ 多巴胺更新: {mind_instance.dopamine_level}")
        SYSTEM_STATE = "IDLE"
        
    return {"status": "ok"}

@app.post("/train")
def train_endpoint(req: TrainRequest):
    global SYSTEM_STATE, optimizer_instance
    
    if req.type == 'gen':
        if not os.path.exists(corpus_path): return {"error": "无泛化语料"}
        SYSTEM_STATE = "TRAINING"
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        for idx, text in enumerate(lines):
            try:
                if not mind_instance.cognitive_judge(text): continue
                if F.cosine_similarity(mind_instance._get_emb(text), mind_instance.C_matrix).max().item() < 0.45: continue
                optimizer_instance.zero_grad()
                probs, H_base, (mode, e_idx, m_sim, topk), raw_Q = mind_instance(text)
                if mode == "SPAWN": 
                    mind_instance.spawn_expert(H_base, text)
                    optimizer_instance = get_optimizer(mind_instance)
                elif mode == "MERGE":
                    mind_instance.expert_hits[e_idx] += 1
                    if mind_instance.expert_hits[e_idx] >= 5:
                        mind_instance.expand_expert(e_idx)
                        optimizer_instance = get_optimizer(mind_instance)
                        mind_instance.expert_hits[e_idx] = 0
                if mode != "DROP":
                    loss = mind_instance.learn_from_text(text, e_idx if e_idx != -1 else mind_instance.active_experts-1)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(mind_instance.parameters(), max_norm=1.0)
                        optimizer_instance.step()
                        safe_print(f"[{idx+1}/{len(lines)}] 🎭 Gen-Ex Loss: {loss.item():.4f}")
                save_mind(mind_instance, save_path)
            except Exception: continue
        safe_print("\n🏁 泛化层学习完毕！")
        SYSTEM_STATE = "IDLE"
        return {"status": "done"}
        
    elif req.type == 'knw':
        knw_files = [os.path.join(knw_corpus_dir, f) for f in os.listdir(knw_corpus_dir) if f.endswith('.txt')]
        if not knw_files: return {"error": "无知识语料"}
        SYSTEM_STATE = "TRAINING"
        for file_path in knw_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if len(line.strip()) > 5] 
            for idx, text in enumerate(lines):
                try:
                    optimizer_instance.zero_grad()
                    H_base = mind_instance._get_emb(text)
                    mode, knw_e_idx, m_sim = mind_instance.check_knw_satiety(H_base)
                    
                    if mode == "SPAWN": 
                        mind_instance.spawn_knw_expert(H_base, text)
                        optimizer_instance = get_optimizer(mind_instance)
                        knw_e_idx = mind_instance.knw_active_experts - 1
                        mind_instance.knw_buffers.append([text])
                    elif mode == "EXPAND":
                        mind_instance.defrag_knw_expert(knw_e_idx, optimizer_instance)
                        mind_instance.expand_knw_expert(knw_e_idx)
                        optimizer_instance = get_optimizer(mind_instance)
                        mind_instance.knw_buffers[knw_e_idx].append(text)
                    else:
                        if knw_e_idx != -1 and len(mind_instance.knw_buffers) > knw_e_idx:
                            mind_instance.knw_buffers[knw_e_idx].append(text)

                    if mode != "DROP":
                        loss = mind_instance.learn_from_knw_text(text, knw_e_idx)
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            loss.backward()
                            mind_instance.apply_knw_gradient_mask(knw_e_idx)
                            torch.nn.utils.clip_grad_norm_(mind_instance.parameters(), max_norm=1.0)
                            optimizer_instance.step()
                            safe_print(f"[{idx+1}/{len(lines)}] 💽 Knw-Ex{knw_e_idx+1} Loss: {loss.item():.4f}")
                    save_mind(mind_instance, save_path)
                except Exception: continue
        safe_print("\n🏁 左脑硬核知识刻录完毕！物理锁已封死旧年轮。")
        SYSTEM_STATE = "IDLE"
        return {"status": "done"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)