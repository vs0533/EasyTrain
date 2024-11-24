from transformers import AutoTokenizer,PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel




if __name__ == "__main__":
    
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer2")
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer_Checkpoint/interim_1.json")
    if tokenizer.backend_tokenizer.decoder is None:
        tokenizer.backend_tokenizer.decoder = ByteLevel(add_prefix_space=False)
        print("已修复分词器的解码器。")
    # tokenizer.add_special_tokens({
    #     "bos_token": "<s>",
    #     "eos_token": "</s>",
    #     "unk_token": "<unk>",
    #     "pad_token": "<pad>",
    #     "mask_token": "<mask>"
    # })
    # tokenizer.save_pretrained("tokenizer2")
    
    # 手动为 backend_tokenizer 设置解码器
    # if tokenizer.backend_tokenizer.decoder is None:
    #     tokenizer.backend_tokenizer.decoder = ByteLevel()
    #     print("已修复分词器的解码器。")
    # # 调试分词器的后端设置
    # print("backend_tokenizer:", tokenizer.backend_tokenizer)
    # print("decoder:", getattr(tokenizer.backend_tokenizer, "decoder", None))
    # 测试编码
    encoded = tokenizer.encode_plus("大学生真是好")
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
    
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 利用WordPiece特殊标记重新组合词语
    words = []
    current_word = ""
    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
                current_word = ""
            current_word += token

    if current_word:
        words.append(current_word)

    print(words)
