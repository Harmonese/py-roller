# CLI

所有入口等价：`py-roller ...` 或 `python -m pyroller ...`。版本检查：`py-roller --version`。

## align：人工调用

```bash
py-roller align song.flac lyrics.txt -o output/song
py-roller align song.flac lyrics.txt -o output/song --offline --no-previews
py-roller align song.flac lyrics.txt -o output/song --audio-mode vocals
```

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--audio-mode` | `mix` | `mix` 原声，`vocals` 显式 Demucs |
| `--offline` | 关闭 | 不下载模型；人声模式还要求已完成的分离缓存 |
| `--cache-dir` | `~/.cache/py-roller/v1` | 模型与推理证据缓存 |
| `--whisper-model` | 固定 large-v3 | 显式本地 faster-whisper 检查点目录 |
| `--ctc-model` | 固定 wav2vec2 | 显式本地 Transformers CTC 目录 |
| `--ctc-device` | `auto` | `auto` 优先 MPS，否则 CPU；也可选 `cpu` / `mps` |
| `--cpu-threads` | `4` | CPU 推理线程预算 |
| `--no-previews` | 关闭 | 只计算最终匹配结果；仍有进度事件 |
| `--json-events` | 关闭 | stdout 输出 Protocol v2 JSONL |

普通模式进度写 stderr，成功后 stdout 输出最终 LRC 绝对路径。输出目录必须是新目录，不能用输出文件名代替目录。不要传入已有 LRC；重复歌词可仅写一次，顺序无需和音频一致。

## run：应用工作进程

```bash
python -m pyroller run --request /absolute/path/request.json
```

只接受协议请求文件，不混合命令行覆盖参数。stdout 始终是 JSONL，stderr 是人类可读日志；不再使用 `PYROLLER_EVENT` 前缀。详见 [Protocol v2](PROTOCOL_V2.md)。

## models：模型准备

```bash
py-roller models list --json
py-roller models download
py-roller models download --kind whisper --cache-dir /path/to/cache
```

`list` 不访问网络、不加载权重。`download` 准备缺失的固定版本，可指定 `whisper`、`ctc` 或默认 `all`。Whisper large-v3 是 CTranslate2 检查点，不是 OpenAI 的 `.pt` 权重。模型可用性只检查必要文件，不表示推理质量或文件完整哈希已验证。

Demucs 权重由人声分离后端首次运行时取得，`models download` 只准备 Whisper 与 CTC。离线人声任务需要先做过相同配置的分离，或把已分离音频以 `mix` 输入。

## doctor：只读环境检查

```bash
py-roller doctor --json
py-roller doctor --audio-mode vocals
```

检查 Python、FFmpeg、依赖可导入性、C++ 扩展及模型文件是否存在；不下载、不加载神经权重。`ready` 要求基本运行依赖及两个模型都可用。`network.https_available` 单独检查 Python SSL 是否可导入，失败时离线缓存仍可运行；它不检查远端连通性或证书链。人声模式额外检查 Demucs/torchaudio，但无法在未指定音频时确认其分离缓存或 Demucs 权重是否已准备好。`ready` 不保证设备内存、网络证书或每个输入都可处理。

## 退出和取消

- `0`：命令成功；`doctor` 表示检查就绪。
- `1`：运行或请求错误；`doctor` 表示尚未就绪。
- `2`：命令行语法错误，可能没有 JSONL。
- `130`：Ctrl+C、SIGINT 或 SIGTERM 被正常捕获后的取消。

强制杀进程、断电和管道断开不能保证写出终态。SIGKILL 可能留下 Demucs `working.lock`；确认没有相应工作进程后才可删除该锁目录并重试。不要根据文件存在就判定成功，应等待 `completed` 和退出码 0。
