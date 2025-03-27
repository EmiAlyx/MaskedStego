### 项目名称
生成式隐蔽通信系统

### 项目描述
本项目旨在开发一个基于自然语言处理技术的生成式隐蔽通信系统，借助预训练语言模型达成消息的隐蔽嵌入与提取。系统整合了 BERT 和 GPT - 2 模型，具备文本生成、消息编码和解码等核心功能，可在正常文本交流里实现隐蔽信息传递，适用于对信息安全和隐蔽性要求较高的场景。

### 项目实现

#### 1. 命令行脚本开发与代码实现
- **参数解析框架搭建**：在 `main.py` 中，运用 Python 的 `argparse` 库精心搭建命令行参数解析框架。以下是核心代码：
```python
from argparse import ArgumentParser
psr = ArgumentParser()
psr.add_argument('text', type=str, help='Text to encode or decode message.')
psr.add_argument('-d', '--decode', action='store_true', help='If this flag is set, decodes from the text.')
psr.add_argument('-m', '--message', type=str, help='Binary message to encode consisting of 0s or 1s.')
psr.add_argument('-i', '--mask_interval', type=int, default=3)
psr.add_argument('-s', '--score_threshold', type=float, default=0.01)
args = psr.parse_args()
```
此代码定义了多个关键参数，包括待处理的文本、编码/解码模式选择、待编码的二进制消息、掩码间隔以及分数阈值，使得系统能够通过命令行灵活控制，以满足不同的使用场景需求。
- **编码函数封装**：实现了 `encode` 函数，将文本编码过程进行封装，具体代码如下：
```python
def encode(cover_text: str, message: str, mask_interval: int, score_threshold: float):
    print(masked_stego(cover_text, message, mask_interval, score_threshold))
```
该函数接收文本、消息、掩码间隔和分数阈值作为参数，调用 `MaskedStego` 类的实例对输入内容进行编码操作，并将编码结果打印输出。
- **编码解码逻辑控制**：根据命令行参数中的 `--decode` 标志判断执行编码还是解码操作，代码如下：
```python
masked_stego = MaskedStego()
if args.decode:
    print(masked_stego.decode(args.text, args.mask_interval, args.score_threshold))
else:
    print(masked_stego(args.text, args.message, args.mask_interval, args.score_threshold))
```
此逻辑确保系统能够依据用户的命令行输入，准确执行相应的操作。

#### 2. 核心功能模块开发与代码优化
- **`MaskedStego` 类实现**：在 `masked_stego.py` 中开发了 `MaskedStego` 类，该类实现了消息的编码和解码核心逻辑。以下是部分关键代码：
```python
class MaskedStego:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def __call__(self, cover_text, message, mask_interval, score_threshold):
        # 编码逻辑实现
        pass

    def decode(self, encoded_text, mask_interval, score_threshold):
        # 解码逻辑实现
        pass

    def _preprocess_text(self, text):
        # 文本预处理逻辑
        pass

    def _mask(self, text, mask_interval):
        # 掩码操作逻辑
        pass

    def _predict(self, text):
        # 模型预测逻辑
        pass
```
在编码过程中，将消息转换为二进制字符串，通过掩码操作和模型预测选择合适的候选词嵌入消息，同时记录编码时间和消息嵌入容量；解码时，从编码文本中提取二进制消息并转换为文本，记录解码时间和消息检索效率。
- **辅助方法优化**：对 `_preprocess_text`、`_mask`、`_predict` 等辅助方法进行了优化，提高了文本预处理和模型预测的效率和准确性。例如，在 `_preprocess_text` 方法中，采用更高效的分词和清洗策略，减少不必要的计算开销。

#### 3. Web 服务与前端交互代码开发
- **Flask 框架搭建 Web 服务**：使用 Flask 框架在 `app.py` 中搭建 Web 服务，设计了 `/generate_text`、`/encode` 和 `/decode` 三个 API 接口，以下是部分代码示例：
```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt')
    # 调用生成文本的函数
    result = generate_text_gpt2(prompt)
    return jsonify({'text': result})

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    text = data.get('text')
    message = data.get('message')
    # 调用编码函数
    encoded_text = masked_stego(text, message)
    return jsonify({'encoded_text': encoded_text})

@app.route('/decode', methods=['POST'])
def decode():
    data = request.get_json()
    encoded_text = data.get('encoded_text')
    # 调用解码函数
    decoded_message = masked_stego.decode(encoded_text)
    return jsonify({'decoded_message': decoded_message})

if __name__ == '__main__':
    app.run(debug=True)
```
这些接口分别处理文本生成、编码和解码请求，实现了前后端的数据交互。
- **前端页面与交互代码实现**：开发前端页面 `index.html`，使用 HTML、CSS 和 JavaScript 设计用户界面，提供文本输入框、按钮和结果显示区域，实现文件上传和保存功能。以下是部分 JavaScript 代码示例，用于与后端 API 交互：
```javascript
function generateText() {
    const prompt = document.getElementById('prompt').value;
    fetch('/generate_text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: prompt })
    })
   .then(response => response.json())
   .then(data => {
        document.getElementById('generated-text').innerHTML = data.text;
    });
}

function encodeMessage() {
    const text = document.getElementById('text').value;
    const message = document.getElementById('message').value;
    fetch('/encode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text, message: message })
    })
   .then(response => response.json())
   .then(data => {
        document.getElementById('encoded-text').innerHTML = data.encoded_text;
    });
}

function decodeMessage() {
    const encodedText = document.getElementById('encoded-text').value;
    fetch('/decode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ encoded_text: encodedText })
    })
   .then(response => response.json())
   .then(data => {
        document.getElementById('decoded-message').innerHTML = data.decoded_message;
    });
}
```
通过这些代码，实现了前端页面与后端 API 的交互，用户可以方便地进行文本生成、编码和解码操作。

##注意事项
在使用编码功能时，要确保输入的二进制消息由 0 和 1 组成。
对于掩码间隔和分数阈值等参数，需要根据实际情况进行调整，以达到最佳的编码和解码效果。


