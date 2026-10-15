# py-roller

把英文 plain lyrics 对齐到歌曲，输出 LRC。歌词文本应包含可能出现的完整行，但不必按实际演唱顺序排列，也不必写全重复次数。

**0.10.0 发布准备版**：Whisper large-v3 发现歌词实例，CTC 细化时间；提供面向 Rolling Pebble 的 **Protocol v2**。这次重写不兼容 0.9 CLI、配置或协议。

## 安装

需要 Python 3.10–3.12、FFmpeg。源码安装还需要 C++17 编译器；macOS 可使用 Xcode Command Line Tools，Windows 使用 MSVC C++ Build Tools。神经依赖的实际可用平台以对应 PyTorch/CTranslate2 版本为准。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
py-roller models download
py-roller doctor
```

可选人声分离：`python -m pip install -e '.[vocals]'`。核心模型需数 GiB 磁盘空间，安装 Python 包不包含模型。源码发布准备完成不等于 PyPI 已发布。

## 对齐一首歌

```bash
py-roller align song.flac lyrics.txt -o output/song
py-roller align song.flac lyrics.txt -o output/song-vocals --audio-mode vocals
# 同样支持
python -m pyroller align song.flac lyrics.txt -o output/song
```

`lyrics.txt` 为 UTF-8 无时间戳歌词，一行一句；忽略 `[Chorus]` 等段落标题。输出目录必须不存在，防止覆盖已有结果。失败后用新的输出目录重试，已完成的模型窗口会复用缓存。

默认使用原声。Demucs 可能把机器人声或特殊唱法当作伴奏移除，应由调用者明确选择。当前仅支持英文、行级 LRC，不承诺精确逐词卡拉 OK 时间。

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
