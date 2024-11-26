"""
    自定义模型 与 配置
"""

from transformers import PretrainedConfig, GPT2LMHeadModel


class MyGPTConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=512,
        embd_pdrop=0.1,
        resid_pdrop=0.1,  # 新增的 resid_pdrop 参数
        n_inner=None,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        attn_pdrop=0.1,
        initializer_range=0.02,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        use_cache=True,
        activation_function="gelu",  # 新增的 activation_function 参数
        n_layer=None,  # 新增的 n_layer 参数
        n_embd=None,  # 新增的 n_embd 参数
        **kwargs,
    ):

        super().__init__(**kwargs)

        # 词汇表大小
        self.vocab_size = 30000

        # 必需的模型超参数
        self.hidden_size = hidden_size  # Transformer 的隐藏层大小
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.num_hidden_layers = num_hidden_layers  # Transformer 的层数

        # 最大位置编码数量（最大序列长度）
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入
        # 嵌入层丢弃率（dropout）
        self.embd_pdrop = embd_pdrop  # 嵌入层的丢弃率

        self.resid_pdrop = resid_pdrop  # 新增的 resid_pdrop 参数，控制残差连接丢弃率

        # `n_inner` 是前馈网络的维度，通常为 `hidden_size * 4`
        self.n_inner = (
            n_inner if n_inner is not None else hidden_size * 4
        )  # 默认值是 hidden_size 的 4 倍
        # LayerNorm 超参数
        self.layer_norm_epsilon = layer_norm_epsilon  # 默认值通常是1e-5

        # 是否缩放注意力权重
        self.scale_attn_weights = scale_attn_weights  # 通常设为 True

        # 注意力层丢弃率
        self.attn_pdrop = attn_pdrop  # 注意力层的丢弃率

        # 权重初始化范围
        self.initializer_range = initializer_range  # 通常设为 0.02

        # 隐藏层丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层的丢弃率

        # 注意力概率丢弃率
        self.attention_probs_dropout_prob = (
            attention_probs_dropout_prob  # 注意力概率的丢弃率
        )

        # 是否按逆层级缩放注意力权重
        self.scale_attn_by_inverse_layer_idx = (
            scale_attn_by_inverse_layer_idx  # 是否按逆层级缩放
        )
        self.reorder_and_upcast_attn = reorder_and_upcast_attn  # 是否在计算注意力时重新排序并提升精度。这个配置项通常设置为 False。如果设为 True，则在多头注意力计算中会对注意力权重进行优化和调整
        self.use_cache = use_cache  # 是否在推理时缓存计算过的 attention 输出
        # 激活函数的类型（如 "gelu" 或 "relu"）
        self.activation_function = activation_function  # 激活函数，通常是 "gelu"

        # 新增的 n_layer 和 num_hidden_layers 兼容
        self.n_layer = n_layer if n_layer is not None else num_hidden_layers
        # 新增的 n_embd，默认为 None 使用 hidden_size
        self.n_embd = n_embd if n_embd is not None else hidden_size

        # 自定义的特殊符号 ID
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.unk_token_id = kwargs.get("unk_token_id", 3)
        self.mask_token_id = kwargs.get("mask_token_id", 4)
