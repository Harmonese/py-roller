# 迁移到 0.10.0

这是一次明确的破坏性重写，版本是 **0.10.0**，不是此前实验文档中的“v1/1.0.0”。旧文档里的 v1 指那一阶段的重构引擎，不能用来推断当前协议版本。

| 原有入口或约定 | 0.10.0 |
|---|---|
| `python -m pyroller.cli.main ...` | `python -m pyroller ...` |
| 实验版 `py-roller audio lyrics -o out` | `py-roller align audio lyrics -o out` |
| protocol v1、`PYROLLER_EVENT` 前缀 | Protocol v2，纯 JSONL，无前缀 |
| install/config/batch 等旧 CLI | Python 环境由宿主管理；`models download`、`doctor`、单任务 `run` |
| 多语言音素/TTS、ASS | 当前仅英文 plain lyrics → LRC |
| 旧配置对象、模型注册表 | `pyroller.engine.EngineConfig` 的运行选项 |
| `align()` 返回内部 occurrences/candidates | 返回稳定 `alignment.json` 对象，歌词为 `lines` |
| `snapshot.occurrences` | `snapshot.lines`，完整替换，不追加 |
| 以最后快照判断完成 | `completed` 加进程退出码 0 |
| 默认 large-v2 | 固定 large-v3，不自动回退 v2 |

算法研究开关从正式 CLI 移除。内部 `pyroller.hybrid` 保留算法回归与离线研究接口，不能作为产品 API 依赖。

缓存仍可读取既有默认 CTC 路径和本机已验证的 v3 权重；音频哈希与模型哈希避免错误复用。原来 v2 的转录结果保留在实验产物中，旧 v2 权重可删除，需要重跑 v2 时必须自行重新下载并显式指定。

Rolling Pebble 需要按 [Protocol v2](PROTOCOL_V2.md) 更新请求生成、事件读取、终态处理和版本范围。本次没有修改相邻项目，也没有兼容适配层。

旧引擎与原型源码已从活动目录移除。历史评估和结论位于 `docs/research/`、`docs/archive/`；本机整理前源码快照位于被 Git 忽略的 `py-roller-artifacts/pre-0.10-source-20261015.tar.gz`。歌曲与证据缓存没有删除或打包发布。
