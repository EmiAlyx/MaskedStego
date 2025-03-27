## 一、项目概述
生成式隐蔽通信系统旨在利用自然语言处理技术，实现消息的隐蔽嵌入和提取。通过预训练语言模型，在正常文本中隐藏秘密信息，同时保证文本的自然流畅性，达到隐蔽通信的目的。本项目主要包含命令行交互脚本和前端Web界面两部分，下面主要介绍命令行交互脚本 `main.py` 的功能与实现。

## 二、代码功能分析

### （一）命令行参数解析
在 `main.py` 中，使用 `argparse` 库进行命令行参数的解析，允许用户灵活控制编码和解码操作。

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

#### 参数说明
- `text`：必需参数，用于指定要进行编码或解码操作的文本。
- `-d` 或 `--decode`：可选标志参数，若设置该标志，则执行解码操作；否则执行编码操作。
- `-m` 或 `--message`：可选参数，用于指定要编码的二进制消息（由 0 和 1 组成）。
- `-i` 或 `--mask_interval`：可选整数参数，默认值为 3，用于指定掩码间隔。
- `-s` 或 `--score_threshold`：可选浮点数参数，默认值为 0.01，用于指定分数阈值。

### （二）编码函数实现
定义了 `encode` 函数，用于将文本和消息进行编码操作。

```python
def encode(cover_text: str, message: str, mask_interval: int, score_threshold: float):
    print(masked_stego(cover_text, message, mask_interval, score_threshold))
```

#### 功能说明
该函数接收四个参数：`cover_text`（待编码的文本）、`message`（要嵌入的二进制消息）、`mask_interval`（掩码间隔）和 `score_threshold`（分数阈值）。函数内部调用 `MaskedStego` 类的实例进行编码操作，并打印编码结果。

### （三）编码解码逻辑控制
根据命令行参数中的 `--decode` 标志，判断执行编码还是解码操作。

```python
masked_stego = MaskedStego()
if args.decode:
    print(masked_stego.decode(args.text, args.mask_interval, args.score_threshold))
else:
    print(masked_stego(args.text, args.message, args.mask_interval, args.score_threshold))
```

#### 逻辑说明
- 首先实例化 `MaskedStego` 类，得到 `masked_stego` 对象。
- 若 `args.decode` 为 `True`，则调用 `masked_stego` 对象的 `decode` 方法进行解码操作，并打印解码结果。
- 若 `args.decode` 为 `False`，则调用 `masked_stego` 对象进行编码操作，并打印编码结果。

## 三、使用示例

### （一）编码操作
假设要对文本 “This is a test.” 进行编码，嵌入二进制消息 “1010”，掩码间隔为 2，分数阈值为 0.02，可以使用以下命令：
```bash
python main.py "This is a test." -m "1010" -i 2 -s 0.02
```

### （二）解码操作
假设要对编码后的文本 “This is a masked test.” 进行解码，掩码间隔为 2，分数阈值为 0.02，可以使用以下命令：
```bash
python main.py "This is a masked test." -d -i 2 -s 0.02
```

## 四、项目成果
- 成功实现了一个可通过命令行灵活控制的编码解码脚本，能够根据用户输入的文本和参数准确完成消息的编码和解码任务，为项目的整体功能提供了重要的命令行交互支持。
- 通过模块化设计和参数化配置，提高了代码的可维护性和扩展性，方便后续对系统进行功能扩展和优化。

## 五、注意事项
- 在使用编码功能时，要确保输入的二进制消息由 0 和 1 组成。
- 对于掩码间隔和分数阈值等参数，需要根据实际情况进行调整，以达到最佳的编码和解码效果。 
