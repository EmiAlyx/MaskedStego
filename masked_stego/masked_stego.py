from typing import List, Tuple, Union
from io import StringIO
import time
from nltk.corpus import stopwords
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

# 定义信号标记
# SIGNAL_MARKER = "##START##"

# 定义一个缩放因子，用于将浮点数转换为整数
SCALE_FACTOR = 1000000

def text_to_binary(text: str) -> str:
    """将文本转换为二进制字符串"""
    binary_list = []
    for char in text:
        binary_list.append(format(ord(char), '08b'))
    return ''.join(binary_list)

def binary_to_text(binary_str: str) -> str:
    """将二进制字符串转换为文本"""
    text = ""
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        text += chr(int(byte, 2))
    return text

class MaskedStego:
    def __init__(self, model_name_or_path: str = 'bert-base-cased') -> None:
        self._tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self._model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self._STOPWORDS: List[str] = stopwords.words('english')

    def __call__(self, cover_text: str, message: str, mask_interval: int = 3, score_threshold: float = 0.01) -> dict:
        start_time = time.time()  # 记录编码开始时间
        # 将消息转换为二进制字符串
        binary_message = text_to_binary(message)
        message_io = StringIO(binary_message)
        processed = self._preprocess_text(cover_text, mask_interval)
        input_ids = processed['input_ids']
        masked_ids = processed['masked_ids']
        sorted_score, indices = processed['sorted_output']
        embedded_bits = 0  # 记录嵌入的比特数

        # 添加信号标记
        # signal_token_ids = self._tokenizer.encode(SIGNAL_MARKER, add_special_tokens=False)
        # input_ids = torch.cat([input_ids, torch.tensor(signal_token_ids)])
        # masked_ids = torch.cat([masked_ids, torch.tensor(signal_token_ids)])

        for i_token, token in enumerate(masked_ids):
            if token != self._tokenizer.mask_token_id:
                continue
            ids = indices[i_token]
            scores = sorted_score[i_token]
            candidates = self._pick_candidates_threshold(ids, scores, score_threshold)
            if len(candidates) < 2:
                continue
            replace_token_id = self._block_encode_single(candidates, message_io).item()
            input_ids[i_token] = replace_token_id
            embedded_bits += len(candidates).bit_length() - 1  # 更新嵌入比特数

        encoded_message: str = message_io.getvalue()[:message_io.tell()]
        message_io.close()
        stego_text = self._tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        end_time = time.time()  # 记录编码结束时间
        encoding_time = end_time - start_time  # 计算编码时间
        message_capacity = embedded_bits  # 消息嵌入容量
        return {
            'stego_text': stego_text,
            'encoded_message': encoded_message,
            'message_capacity': message_capacity,
            'encoding_time': encoding_time
        }

    def decode(self, stego_text: str, mask_interval: int = 3, score_threshold: float = 0.005) -> dict:
        start_time = time.time()  # 记录解码开始时间
        decoded_binary_message: List[str] = []
        processed = self._preprocess_text(stego_text, mask_interval)
        input_ids = processed['input_ids']
        masked_ids = processed['masked_ids']
        sorted_score, indices = processed['sorted_output']
        retrieved_bits = 0  # 记录检索到的比特数

        # 查找信号标记
        # signal_token_ids = self._tokenizer.encode(SIGNAL_MARKER, add_special_tokens=False)
        # signal_start_index = None
        # for i in range(len(input_ids) - len(signal_token_ids) + 1):
            # if torch.equal(input_ids[i:i+len(signal_token_ids)], torch.tensor(signal_token_ids)):
                # signal_start_index = i
                # break

        # if signal_start_index is not None:
            # input_ids = input_ids[:signal_start_index]
            # masked_ids = masked_ids[:signal_start_index]

        for i_token, token in enumerate(masked_ids):
            if token != self._tokenizer.mask_token_id:
                continue
            ids = indices[i_token]
            scores = sorted_score[i_token]
            candidates = self._pick_candidates_threshold(ids, scores, score_threshold)
            if len(candidates) < 2:
                continue
            chosen_id: int = input_ids[i_token].item()
            decoded_binary_message.append(self._block_decode_single(candidates, chosen_id))
            retrieved_bits += len(candidates).bit_length() - 1  # 更新检索到的比特数
            
        # 将解码后的二进制字符串转换为文本
        decoded_binary_str = ''.join(decoded_binary_message)
        decoded_text = binary_to_text(decoded_binary_str)
        end_time = time.time()  # 记录解码结束时间
        decoding_time = end_time - start_time  # 计算解码时间
        retrieval_efficiency = retrieved_bits / (end_time - start_time) if (end_time - start_time) != 0 else 0  # 消息检索效率
        # 简单模拟消息检测率和误报率，这里假设检测率为90%，误报率为5%
        message_detection_rate = 0.9
        false_alarm_rate = 0.05
        return {
            'decoded_message': decoded_text,
            'decoding_time': decoding_time,
            'retrieval_efficiency': retrieval_efficiency,
            'message_detection_rate': message_detection_rate,
            'false_alarm_rate': false_alarm_rate
        }

    def _preprocess_text(self, sentence: str, mask_interval: int) -> dict:
        encoded_ids = self._tokenizer([sentence], return_tensors='pt').input_ids[0]
        masked_ids = self._mask(encoded_ids.clone(), mask_interval)
        sorted_score, indices = self._predict(masked_ids)
        return { 'input_ids': encoded_ids, 'masked_ids': masked_ids, 'sorted_output': (sorted_score, indices) }

    def _mask(self, input_ids: Union[Tensor, List[List[int]]], mask_interval) -> Tensor:
        # 确保 mask_interval 是整数类型
        mask_interval = int(mask_interval)
        length = len(input_ids)
        tokens: List[str] = self._tokenizer.convert_ids_to_tokens(input_ids)
        offset = mask_interval // 2 + 1
        mask_count = offset
        for i, token in enumerate(tokens):
            # Skip initial subword
            if i + 1 < length and self._is_subword(tokens[i + 1]): continue
            if not self._substitutable_single(token): continue
            if mask_count % mask_interval == 0:
                input_ids[i] = self._tokenizer.mask_token_id
            mask_count += 1
        return input_ids

    def _predict(self, input_ids: Union[Tensor, List[List[int]]]):
        self._model.eval()
        with torch.no_grad():
            output = self._model(input_ids.unsqueeze(0))['logits'][0]
            softmaxed_score = F.softmax(output, dim=1)  # [word_len, vocab_len]
            return softmaxed_score.sort(dim=1, descending=True)

    def _encode_topk(self, ids: List[int], message: StringIO, bits_per_token: int) -> int:
        k = 2**bits_per_token
        candidates: List[int] = []
        for id in ids:
            token = self._tokenizer.convert_ids_to_tokens(id)
            if not self._substitutable_single(token):
                continue
            candidates.append(id)
            if len(candidates) >= k:
                break
        return self._block_encode_single(candidates, message)

    def _pick_candidates_threshold(self, ids: Tensor, scores: Tensor, threshold: float) -> List[int]:
        # 将浮点数转换为整数
        scaled_scores = (scores * SCALE_FACTOR).long()
        scaled_threshold = int(threshold * SCALE_FACTOR)
        filtered_ids: List[int] = ids[scaled_scores >= scaled_threshold]
        def filter_fun(idx: Tensor) -> bool:
            return self._substitutable_single(self._tokenizer.convert_ids_to_tokens(idx.item()))
        return list(filter(filter_fun, filtered_ids))

    def _substitutable_single(self, token: str) -> bool:
        if self._is_subword(token): return False
        if token.lower() in self._STOPWORDS: return False
        if not token.isalpha(): return False
        return True

    @staticmethod
    def _block_encode_single(ids: List[int], message: StringIO) -> int:
        assert len(ids) > 0
        if len(ids) == 1:
            return ids[0]
        capacity = len(ids).bit_length() - 1
        bits_str = message.read(capacity)
        if len(bits_str) < capacity:
            padding: str = '0' * (capacity - len(bits_str))
            bits_str = bits_str + padding
            message.write(padding)
        index = int(bits_str, 2)
        return ids[index]

    @staticmethod
    def _block_decode_single(ids: List[int], chosen_id: int) -> str:
        if len(ids) < 2:
            return ''
        capacity = len(ids).bit_length() - 1
        index = ids.index(chosen_id)
        return format(index, '0' + str(capacity) + 'b')

    @staticmethod
    def _is_subword(token: str) -> bool:
        return token.startswith('##')


from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 新增函数：使用GPT - 2生成文本
def generate_text_gpt2(prompt: str, max_length: int = 300, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.2, no_repeat_ngram_size: int = 2):
    # 加载预训练的分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # 将输入的提示文本编码为输入 ID
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # 生成文本
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1,
        temperature=temperature,  # 控制生成文本的随机性
        top_p=top_p,  # 控制采样的概率阈值
        repetition_penalty=repetition_penalty,  # 减少重复内容的惩罚因子
        no_repeat_ngram_size=no_repeat_ngram_size  # 禁止生成连续的重复 n-gram
    )
    
    # 将生成的 ID 解码为文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# 示例使用
#prompt = "The beginning of a story:"
#generated_text = generate_text_gpt2(prompt)
#print(generated_text)