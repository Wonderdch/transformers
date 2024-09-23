from src.transformers.models.qwen2 import Qwen2Config, Qwen2Model
import torch


def run_qwen2():
    # 下面用到的配置基本上是 Qwen2-7B-Instruct 的配置，但是把一些参数缩小了一半
    qwen2_config = Qwen2Config(vocab_size=151936,
                               hidden_size=4096 // 2,
                               intermediate_size=22016 // 2,
                               num_hidden_layers=32 // 2,
                               num_attention_heads=32,
                               max_position_embeddings=32768 // 2,
                               attn_implementation="eager",
                               )

    qwen2_model = Qwen2Model(qwen2_config)

    # 模拟 tokenizer 的输入
    ids = torch.randint(0, qwen2_config.vocab_size, (4, 30))

    res = qwen2_model(ids)
    print(type(res))
    print(res.last_hidden_state.shape)



if __name__ == '__main__':
    run_qwen2()
