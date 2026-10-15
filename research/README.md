# 离线研究工具

正式运行请使用 `py-roller` CLI。此目录不打包、不构成兼容性承诺。

`scripts/` 保留与当前证据格式有关的评分、覆盖统计、v3 对照、中文报告和发布回放工具。它们使用工作区已有缓存和本机音频；不是可下载的公开数据集，也不是 CI 前置条件。

```bash
python research/scripts/check_release_replay.py
python research/scripts/report_whisper_v3.py py-roller-artifacts/whisper-v3-expanded
```

此前原型和旧实验驱动已退役，不保留一套看似仍受支持的重复引擎。原始结论保存在 `docs/research/`；本机完整源码快照见迁移文档。
