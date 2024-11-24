from transformers import AutoTokenizer

if __name__ == "__main__":
    
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    
    encoded = tokenizer("大学生真是好")
    print(f"编码结果: {encoded}")

    # 获取 input_ids
    input_ids = encoded["input_ids"]

    # 方法1: 使用分词器提供的解码方法
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"解码结果: {decoded_text}")

    # 方法2: 手动转换为字符串
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"对应的 tokens: {tokens}")

    # 将 tokens 转为文本
    text = tokenizer.convert_tokens_to_string(tokens)
    print(f"手动解码结果: {text}")