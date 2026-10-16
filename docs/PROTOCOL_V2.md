# Protocol v2：Rolling Pebble 接入

适用 py-roller 0.10.x（当前 0.10.1）。版本号是 `protocol_version: 2`，与 Python 包版本、缓存目录和内部诊断 schema 无关。旧 protocol v1 不兼容；此文档不表示相邻 Rolling Pebble 代码已迁移。

新增可选 `options.alignment_mode: "unordered" | "ordered"`，显式值覆盖 `PYROLLER_ALIGNMENT_MODE`，未设置时默认 ordered。ordered 要求输入完整并保留所有重复次数。

ordered 的结果与每次 snapshot 增加 `alignment_mode: "ordered"` 及 `unaligned_lines`。`lines` 继续只包含具有数值时间的已定位行；`unaligned_lines` 包含 `occurrence_id`、`line_id`、`input_line_index`、`text`、`reason`，没有虚构时间。两者通过从零开始的原文件物理行号 `input_line_index` 合并还原输入；空行和 `[Chorus]` 等标题不产生条目。重复文本共享 `line_id`，但 ordered 的 `occurrence_id` 由输入行号生成，在同任务预览间保持稳定。每次快照需同时替换这两个数组。LRC 仅写入 `lines`。

未定位原因包括 `pending_audio`（预览尚无后续证据）、`region_budget_or_empty`、`insufficient_evidence` 和 `overlapping_assignment`。完成表示算法已结束，不表示所有输入行均已定位。unordered 模式保持既有结果形状。

## 启动与请求

```bash
/path/to/venv/bin/python -m pyroller run --request /path/to/request.json
```

请求结构见 [JSON Schema](protocol-v2/request.schema.json) 和 [完整示例](protocol-v2/request.example.json)。

```json
{
  "protocol_version": 2,
  "request_id": "job-001",
  "audio": "song.flac",
  "lyrics": "lyrics.txt",
  "output": "runs/job-001",
  "options": {"audio_mode": "mix", "offline": true, "previews": true}
}
```

路径相对于请求文件所在目录解析；展开 `~`，输出事件中的路径为绝对路径。`request_id` 由调用者提供，只用于关联，不能当路径使用。音频及歌词是本机文件，不接受 URL。歌词文件为无时间戳 UTF-8 文本。输出目录必须不存在。

未知字段、重复 JSON key、未知选项、非有限 JSON 数字、错误类型或协议版本均拒绝。不要将 `false` 写成字符串。`options` 只包含 CLI 文档列出的运行选项（`previews` 为正向布尔值）。缓存目录和模型目录可以显式配置，算法开关不属于协议。

## 输出通道与公共信封

stdout 每行一个 JSON 对象，UTF-8，无前缀；stderr 单独收集日志。不要把两个流合并再解析。每条事件包含：

```json
{"protocol_version":2,"engine":"py-roller","engine_version":"0.10.1","request_id":"job-001","sequence":1,"type":"started"}
```

`sequence` 从 1 开始严格递增。客户端检查版本、任务 ID，忽略已经处理过的 sequence。允许未来增加输出字段；不要依赖 JSON key 顺序或 stderr 文本。

| type | 额外字段 | 客户端含义 |
|---|---|---|
| `started` | 无 | 任务开始，尚未证明输入有效 |
| `progress` | `stage`；可选 `duration`、`audio_available_until`、`cached` | 阶段信息；不是全任务百分比 |
| `snapshot` | `revision`、`final`、`audio_available_until`、`lines` | 替换当前整份歌词列表 |
| `completed` | `revision`、`artifacts`、`summary`、`metrics` | 成功终态，文件均已写入 |
| `failed` | `code`、`message` | 失败终态，不应用最终结果 |
| `cancelled` | `code: cancelled`、`message` | 取消终态，缓存可复用 |

事件 Schema：[event.schema.json](protocol-v2/event.schema.json)。阶段目前为 separation、decode、models、ctc、whisper；未来可能增加，客户端应显示未知阶段而不是报错。

正常事件顺序：started → 若干 progress/snapshot → 最终 snapshot → completed。失败/取消可发生在任意未完成阶段。每次正常可捕获调用恰有一个终态，且终态后不再发送事件。

`audio_available_until` 是当前阶段的音频时钟。CTC 整曲完成后才开始 Whisper，因此不能跨阶段把它当单调总进度。snapshot 时钟则对应当时可用的 Whisper 及 CTC 证据。

## 快照与歌词实例

`lines` 字段与 [alignment.schema.json](protocol-v2/alignment.schema.json) 中的行定义一致：

```json
{"occurrence_id":"occ-000001","line_id":"line-example","text":"Come on","start":12.3,"end":14.1,"status":"needs_review","finalized":true,"content_status":"text_matched_unverified","timing_status":"ctc_supported_unverified","timing_method":"local_ordered_ctc"}
```

时间为秒。`start` 是 LRC 起点；`end` 是展示边界，不能当精确发声结束时间。`occurrence_id` 只在当前 revision 内唯一，不能跨 revision 合并。`line_id` 表示归一化文本身份，同一句重复演唱共享身份。

中间行 `status=tentative`；最终行 `status=needs_review`。`finalized=true` 或 `final=true` 只表示计算完成，**不表示人工确认或识别正确**。预览可能撤回、改变文本身份或移动时间。应用每次收到 snapshot 必须替换整份列表，不是追加。

`content_status`、`timing_status`、`timing_method` 是诊断提示字符串，不应作为成功/失败枚举或准确率阈值。未知值应保留或显示一般“待核实”。空 `lines` 也是合法成功结果。

## 文件与成功条件

`completed.artifacts` 包含 `lrc`、`alignment`、`diagnostics`、`run`、`events` 的绝对路径。稳定结果为 `alignment.json`：公共信封（无 sequence/type）、`status: complete`、`duration`、`lines`。

内部 `result.json`、`run.json` 仅用于诊断，不要求前端解析候选、字符路径或打分字段。`summary.lines` 是输出行数，不是正确率；`metrics` 提供模型推理、加载、匹配和总耗时，单位秒，cache_hits 为计数。

客户端必须同时确认 **completed + 进程退出码 0**。最终 snapshot 不是提交信号；磁盘写入/管道错误仍可能导致失败。非零退出或 EOF 没有终态都视为异常，保留日志，不擅自应用部分结果。

请求尚未通过校验时，失败事件使用引擎生成的 request_id，不能声称它属于未经验证的请求 ID；调用者仍可用进程所属 job 关联。命令行语法错误（退出 2）可能只输出 stderr。

## 取消、重试和并发

SIGINT/SIGTERM 正常捕获后发 cancelled 并退出 130；神经原生调用期间可能延迟响应。Rolling Pebble 可先发信号，超时后终止整个进程组；强制杀死不能保证终态和日志落盘。Windows 终止进程组的实现由宿主负责。

每次重试创建新输出目录、新 request_id；完整神经窗口按内容哈希复用。协议暂不支持续写同一输出目录、批处理请求、修改运行中歌词或双向控制消息。

不同任务应有独立输出目录；共享缓存可以复用原子写入的模型证据。同一音源/配置首次 Demucs 分离存在排他锁，遇到 busy 应稍后重试。SIGKILL 遗留锁需人工确认工作进程已退出后清理。

## Rolling Pebble 迁移步骤

1. 将独立运行环境版本范围更新到 `py-roller>=0.10,<0.11`，安装可选 `vocals` 依赖（如需要）。
2. 改用 `python -m pyroller doctor --json` 和 `models download`，不再调用旧 install 命令。
3. 每个任务保存 plain lyrics 与 v2 request.json，调用 `run --request`。
4. 分开读取 stdout JSONL 与 stderr，处理 sequence、整份快照、终态和退出码。
5. completed 后读取 alignment/LRC，并核对任务起源的音频及歌词版本仍匹配当前编辑器状态，避免覆盖用户的新编辑。

这里不提供 protocol v1 适配层，也不让引擎承担 UI 文档版本合并。
