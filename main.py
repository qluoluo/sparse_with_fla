from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, 
    Unpack, 
    ALL_ATTENTION_FUNCTIONS, 
    FlashAttentionKwargs, 
    eager_attention_forward,
    logger, 
    repeat_kv
)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat



def load_finetuned_model(base_model_name: str, adapter_path: str) -> PeftModel:
    """
    加载微调后的LLaMA模型
    
    Args:
        base_model_name: 基础模型路径/名称
        adapter_path: LoRA适配器路径
        
    Returns:
        PeftModel: 加载了适配器的微调模型
    """
    config = AutoConfig.from_pretrained(base_model_name)
    # print(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        config=config,
        trust_remote_code=True,
    )

    print(base_model)

    return PeftModel.from_pretrained(base_model, adapter_path)


def analyze_llama_attention(
    base_model_name: str,
    input_text: str,
    sparse_num: int = 4,
    num_layers_to_use: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分析LLaMA模型的注意力机制并返回重要token索引
    
    Args:
        base_model_name: 基础模型路径/名称
        input_text: 要分析的输入文本
        sparse_num: 要提取的top注意力token数量
        num_layers_to_use: 用于分析的前N层transformer层数
        
    Returns:
        tuple: (注意力权重, 重要token索引)
    """
    # 初始化全局变量存储注意力状态
    q = k = v = None
    
    def attn_forward_with_saving_satates(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        nonlocal q, k, v
        
        # 原始前向传播逻辑
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 存储当前注意力状态
        q, k, v = query_states, key_states, value_states

        # 选择注意力实现方式
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "Using eager attention instead of SDPA for attention weights"
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 计算注意力
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    


    # 加载基础模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 设置模型用于分析（只使用前几层）
    model.model.layers = model.model.layers[:num_layers_to_use]
    model.model.layers[-1].self_attn.forward = partial(
        attn_forward_with_saving_satates, 
        model.model.layers[-1].self_attn
    )
    
    # 处理输入文本
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    # 计算最后一个token的注意力权重
    def get_last_attn_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = q[..., -1:, :]
        num_key_value_groups = q.shape[1] // k.shape[1]
        k = repeat_kv(k, num_key_value_groups)
        return torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    
    attn_weights = get_last_attn_weights(q, k)
    attn_weights = attn_weights.squeeze(2).squeeze(0)  # 移除批次和头维度
    summed_weights = torch.sum(attn_weights, dim=0)
    sparse_indices = torch.topk(summed_weights, sparse_num, dim=-1)[1]
    
    return sparse_indices


# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        "base_model_name": "models/Llama-3.2-3B-fla-base2",
        "adapter_path": "models/base2-lora/checkpoint-992",
        "input_text": "User: Please write a short story about a robot learning to love.\nAssistant:",
        "sparse_num": 4,
        "num_layers_to_use": 2
    }
    
    # 运行分析
    # sparse_indices = analyze_llama_attention(
    #     base_model_name=config["base_model_name"],
    #     input_text=config["input_text"],
    #     sparse_num=config["sparse_num"],
    #     num_layers_to_use=config["num_layers_to_use"]
    # )

    sparse_indices = torch.tensor([ 0, 15, 14, 16], device='cuda:0')
    sparse_indices, _ = torch.sort(sparse_indices)

    print(sparse_indices)

    model = load_finetuned_model(config["base_model_name"], config["adapter_path"])

    def scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0):
        """
        计算缩放点积注意力
        
        参数:
            Q: Query 张量，形状为 [batch_size, num_heads, seq_len_q, head_dim]
            K: Key 张量，形状为 [batch_size, num_heads, seq_len_k, head_dim]
            V: Value 张量，形状为 [batch_size, num_heads, seq_len_k, head_dim]
            attn_mask: 注意力掩码，形状为 [batch_size, num_heads, seq_len_q, seq_len_k]
                    True 表示需要被掩蔽的位置（设置为 -inf）
            dropout_p: dropout 概率
        
        返回:
            注意力输出和注意力权重
        """
        print(f"{Q.shape=}, {K.shape=}, {V.shape=}")
        rep_nums = Q.shape[1] // K.shape[1]
        K = repeat_kv(K, rep_nums)
        V = repeat_kv(V, rep_nums)
        # 计算 QK^T / sqrt(d_k)
        d_k = Q.size(-1)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) #/ torch.sqrt(torch.tensor(d_k))
        print(f"{attn_scores.shape=}")
        
        # 应用 attention mask
        if attn_mask is not None:
            attn_scores += attn_mask
        
        # 计算 softmax 获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用 dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        
        # 计算加权和
        output = torch.matmul(attn_weights, V)
        
        # return output, attn_weights
        return output

    def sparse_attn_with_fla_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sparse_indices: Optional[torch.Tensor] = None,        # Add for sparse attention
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        ############################################################################
        # past_key_value 处理机制
        assert type(past_key_value) == DynamicCache or past_key_value is None, f"{type(past_key_value)=}"
        final_state_up, final_state_down = None, None
        key_cache, value_cache = None, None

        if past_key_value is not None:
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append([])
                past_key_value.value_cache.append([])
                # print(f"{self.layer_idx=} fla create new cache")
            else:
                final_state_up = past_key_value.key_cache[self.layer_idx]['final_state_up']
                final_state_down = past_key_value.key_cache[self.layer_idx]['final_state_down']
                key_cache = past_key_value.key_cache[self.layer_idx]['key_cache']
                value_cache = past_key_value.value_cache[self.layer_idx]['value_cache']
                # print(f"{self.layer_idx=} fla use old cache, {final_state_up.shape=}, {final_state_down.shape=}")
        ############################################################################

        if not hasattr(self, "head_dim"):
            self.head_dim = self.head_k_dim
        if not hasattr(self, "num_key_value_groups"):
            self.num_key_value_groups = self.num_kv_groups
        self.is_causal = True

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 用于sparse_attn的未训练的qkv
        sparse_q = self.q_proj.base_layer.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        sparse_k = self.k_proj.base_layer.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        sparse_v = self.v_proj.base_layer.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        key_cache_len = key_cache.shape[-2] if key_cache is not None else 0
        sparse_q_position = torch.arange(key_cache_len, key_cache_len + sparse_q.shape[-2]).to(sparse_q.device)
        # sparse_k_position = torch.cat([sparse_indices, sparse_q_position], dim=0)
        sparse_k_position = sparse_indices.to(sparse_q.device)

        def create_attn_mask(q_pos, k_pos):
            # 确保 q_pos 和 k_pos 是 1D 张量
            assert q_pos.ndim == 1, "q_pos 应该是 1D 张量"
            assert k_pos.ndim == 1, "k_pos 应该是 1D 张量"
            
            # 获取查询和键的数量
            num_queries = q_pos.size(0)
            num_keys = k_pos.size(0)
            
            # 创建一个广播后的布尔掩码，表示哪些位置需要被屏蔽
            mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)  # Shape: (num_queries, num_keys)
            
            # 将布尔掩码转换为数值掩码，True -> -inf, False -> 0
            causal_mask = torch.zeros((num_queries, num_keys), dtype=float).to(mask.device)
            causal_mask.masked_fill_(mask, float('-inf'))
            causal_mask = causal_mask[None, None, :, :]
            
            return causal_mask


        cos, sin = position_embeddings
        sparse_q, sparse_k = apply_rotary_pos_emb(sparse_q, sparse_k, cos, sin)
        key_cache = sparse_k if key_cache is None else torch.cat([key_cache, sparse_k], dim=-2)
        value_cache = sparse_v if value_cache is None else torch.cat([value_cache, sparse_v], dim=-2)


        sparse_k = key_cache[..., sparse_indices, :]
        sparse_v = value_cache[..., sparse_indices, :]
        from transformers.modeling_utils import flash_attention_forward
        sparse_attn_mask = create_attn_mask(sparse_q_position, sparse_k_position).to(sparse_q)
        # sparse_attn_output = flash_attention_forward(
        #     self, sparse_q, sparse_k, sparse_v, sparse_attn_mask, dropout_p=0)
        
        # sparse_attn_output = flash_attention_forward(
        #     self, sparse_q, sparse_k, sparse_v, attention_mask=sparse_attn_mask, dropout_p=0)[0]

        # import ipdb; ipdb.set_trace()
        
        sparse_attn_output = scaled_dot_product_attention(
            sparse_q, sparse_k, sparse_v, sparse_attn_mask, dropout_p=0)
        
        # sparse_attn_output3 = eager_attention_forward(
        #     self, sparse_q, sparse_k, sparse_v, attention_mask=sparse_attn_mask, dropout_p=0, scaling=1)[0]
        
        # import ipdb; ipdb.set_trace()

        # 用于fla的微调后的qkv
        fla_q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        fla_k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        fla_v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        mode = self.mode
        fla_q = rearrange(fla_q, '... (h d) -> ... h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            fla_k = repeat(fla_k, '... (h d) -> ... (h g) d', d=self.head_k_dim, g=self.num_kv_groups)
            fla_v = repeat(fla_v, '... (h d) -> ... (h g) d', d=self.head_v_dim, g=self.num_kv_groups)
        else:
            fla_k = rearrange(fla_k, '... (h d) -> ... h d', d=self.head_k_dim)
            fla_v = rearrange(fla_v, '... (h d) -> ... h d', d=self.head_v_dim)

        fla_q = self.feature_map_q(fla_q)
        fla_k = self.feature_map_k(fla_k)

        if self.norm_q:     # FIXME: 如果效果不好，记得尝试normalize
            fla_q = fla_q / (fla_q.sum(-1, True) + 1e-4)
        if self.norm_k:
            fla_k = fla_k / (fla_k.sum(-1, True) + 1e-4)

        assert mode == 'fused_chunk', 'Only fused_chunk is supported'
        from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn

        # FIXME: past_key_values
        o, final_state_up = fused_chunk_linear_attn(
            q=fla_q,
            k=fla_k,
            v=fla_v,
            normalize=self.do_feature_map_norm,
            initial_state=final_state_up,
            output_final_state=True,
            head_first=False
        )

        n, final_state_down = fused_chunk_linear_attn(
            q=fla_q,
            k=fla_k,
            # v=torch.ones((v.shape[0], v.shape[1], v.shape[2], 1)).to(q.device).to(q.dtype),
            v=torch.ones_like(fla_v).to(fla_q.device).to(fla_q.dtype),
            normalize=self.do_feature_map_norm,
            initial_state=final_state_down,
            output_final_state=True,
            head_first=False
        )
        n = n[..., :1]
        o = o / n

        return o, None

    for i in range(2, len(model.base_model.model.model.layers)):
        from fla.layers.linear_attn import LinearAttention
        if type(model.base_model.model.model.layers[i].self_attn) == LinearAttention:
            model.base_model.model.model.layers[i].self_attn.forward = partial(
                sparse_attn_with_fla_forward, 
                model.base_model.model.model.layers[i].self_attn,
                sparse_indices=sparse_indices,
            )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
    input_text = "User: Please write a short story about a robot learning to love.\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)