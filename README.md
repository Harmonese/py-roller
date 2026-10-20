# py-roller

把 plain lyrics 对齐到歌曲，输出 LRC。提供英文 `en`、普通话 `zh` 和通用音素 `mul` 配置。默认 ordered 模式要求完整顺序和所有重复行；unordered 模式允许歌词集合不按演唱顺序排列。

**0.10.3**：新增实验性普通话和多语言音素后端，支持简繁匹配及 Unicode 歌词；英文流程独立保留。Whisper large-v3 发现歌词实例，CTC 细化时间，提供面向 Rolling Pebble 的 **Protocol v2**。0.10 系列重写不兼容 0.9 CLI、配置或协议。

从 0.10.0 升级后先运行 `py-roller models migrate`，将已有模型导入统一目录，避免重新下载。当前默认采用完整有序歌词对齐；需要原来的自由匹配行为时显式指定 `--alignment-mode unordered`。

## 安装

需要 Python 3.10–3.12、FFmpeg。源码安装还需要 C++17 编译器；macOS 可使用 Xcode Command Line Tools，Windows 使用 MSVC C++ Build Tools。神经依赖的实际可用平台以对应 PyTorch/CTranslate2 版本为准。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'py-roller==0.10.3'
py-roller models download
py-roller doctor
```

可选人声分离：`python -m pip install 'py-roller[vocals]==0.10.3'`；多语言音素：`python -m pip install 'py-roller[mul]==0.10.3'`。核心模型需数 GiB 磁盘空间，安装 Python 包不包含模型。源码开发安装使用 `python -m pip install -e '.[test]'`。

## 对齐一首歌

| 配置 | 用途 | CTC 后端 | 状态 |
|---|---|---|---|
| `en` | 英文 | 英文 wav2vec2 | 既有特化流程 |
| `zh` | 普通话 | 中文 XLSR | 实验性，已做两首验证 |
| `mul` | 其他语言及通用回退 | XLSR-53-espeak 音素 | 实验性，已测英/中/俄三首 |

未指定语言时使用 `auto` 按歌词文字选择；需要让英中文也使用通用后端时显式选择 `mul`。中英自动选择仍使用原有 `zh-en` 有限配置，真实混唱质量尚待验收。

```bash
py-roller align song.flac lyrics.txt -o output/song
py-roller align song.flac lyrics.txt -o output/song-vocals --audio-mode vocals
py-roller align song.flac complete-lyrics.txt -o output/song-ordered --alignment-mode ordered
py-roller models download --kind ctc --language zh
py-roller align 中文歌曲.flac 歌词.txt -o output/中文歌曲 --language zh
# 同样支持
python -m pyroller align song.flac lyrics.txt -o output/song
```

`lyrics.txt` 为 UTF-8 无时间戳歌词，一行一句；忽略 `[Chorus]` 等段落标题。输出目录必须不存在，防止覆盖已有结果。失败后用新的输出目录重试，已完成的模型窗口会复用缓存。

默认使用原声。Demucs 可能把机器人声或特殊唱法当作伴奏移除，应由调用者明确选择。输出为行级 LRC，不承诺精确逐词卡拉 OK 时间。普通话和 mul 处于实验阶段；mul 已对英文、中文和一首俄文歌进行验证，不代表所有语言、方言或真实混唱均通过验收。`zh` 仅表示普通话。

`--language auto`（默认）根据有效歌词文字选择 `en`、`zh` 或 `zh-en` 配置，不是从音频辨认方言。英文继续复用原模型与缓存，中文和混唱使用固定中文 CTC。简繁转换仅用于内部匹配和模型编码，输出保留原文。无法编码的字或数字不会静默删除，将保留粗时间或未定位诊断。

其他文字会由 `auto` 选择实验性 `mul`；也可显式指定，让英文或中文使用同一个通用后端。`mul` 使用 Whisper 多语言转录和固定的 XLSR-53-espeak 音素 CTC。它与 `en`、`zh` 是独立配置，覆盖范围更广，不承诺达到特化配置的准确率。原有 `zh-en` 仍是使用中文 CTC 的有限配置，不等于通用多语言支持。

```bash
python -m pip install 'py-roller[mul]==0.10.3'
brew install espeak-ng  # macOS；Linux 需安装 libespeak-ng1
py-roller models download --kind ctc --language mul
py-roller align song.mp3 lyrics.txt -o output/mul --language mul --alignment-mode ordered
```

`mul` 的发音按歌词文字推断，保留原文和词/字对应；不能仅凭汉字判定普通话、粤语或日语。暂不保证所有 eSpeak 语言、方言及真实混唱的质量。无法转换或词表外音素会明确诊断并回退，不能用它们生成假时间。

默认 `ordered` 模式要求输入完整且顺序、重复次数准确，保留每次输入行并按顺序定位。`unordered` 模式将输入当作歌词集合，允许重排和重复。可用 `PYROLLER_ALIGNMENT_MODE=unordered` 切回自由匹配，显式 `--alignment-mode` 优先。未能定位的输入行保存在 `alignment.json` 的 `unaligned_lines` 中，不生成虚假的 LRC 时间。

## 输出

| 文件 | 用途 |
|---|---|
| `result.lrc` | 最终歌词起点 |
| `alignment.json` | 稳定 Protocol v2 结果，供应用接入 |
| `events.jsonl` | 进度、完整预览快照和终态 |
| `preview.lrc` | 最近一次中间预览，可能不存在 |
| `coarse.lrc` | CTC 修正前的粗时间 |
| `result.json` | 内部候选与声学诊断，不属于稳定接口 |
| `run.json` | 配置、模型/音频身份、缓存路径、耗时及状态 |

`final` 只表示计算完成；所有行仍需人工确认。`end` 用于展示，不是精确的人声结束点。

## Rolling Pebble / Python 接入

```bash
python -m pyroller run --request request.json
```

标准输出为纯 JSONL，日志写标准错误。应用应按 `request_id` 和 `sequence` 接收事件，每次 `snapshot` 替换整份歌词列表，在 `completed` 后读取 `alignment.json` 或 LRC。

```python
from pyroller.engine import EngineConfig, align

result = align("song.flac", "lyrics.txt", "output/song",
               EngineConfig(offline=True), request_id="job-001")
for line in result["lines"]:
    print(line["start"], line["text"])
```

## 文档

- [CLI 与模型管理](docs/CLI.md)
- [Protocol v2 接入规范](docs/PROTOCOL_V2.md)及[请求示例](docs/protocol-v2/request.example.json)
- [架构和边界](docs/ARCHITECTURE.md)
- [从 0.9 / 实验版迁移](docs/MIGRATION_0.10.md)
- [发布检查](docs/RELEASE.md)、[Changelog](CHANGELOG.md)
- [研究结论索引](docs/research/README.md)，历史文档不代表当前默认配置

## 开发验证

```bash
python -m pytest
python -m pip install -e '.[release]'
python -m build
python scripts/check_release.py dist
python -m twine check dist/*
```

运行时代码位于 `pyroller/`，测试位于 `tests/`，离线研究工具位于 `research/`。歌曲、模型和历史缓存不进入发布包。模型依次加载以控制内存；第一份预览需等待完整 CTC 处理和第一个 Whisper 窗口，这不是实时麦克风识别服务。
