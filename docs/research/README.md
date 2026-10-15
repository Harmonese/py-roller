# 研究结论归档

这些文档描述探索过程及当时的默认配置，**不是 0.10.0 的使用说明**。当前行为以主 README、CLI、Protocol v2 和架构文档为准。

优先阅读：

- [v3 扩展验证](WHISPER_V3_EXPANDED_ZH.md)：六首十组的整体覆盖与时间取舍。其“暂不切默认”是当时的建议，随后用户决定以 v3 固定为基线。
- [v3 初轮测试](WHISPER_V3_VALIDATION.md)。
- [小词扣分与统一评分](SCORING_VALIDATION.md)、[窗口证据保全](WINDOW_EVIDENCE_VALIDATION.md)：正式默认算法的历史依据。
- [CTC 问题审视](CTC_REPAIR_PLAN.md)、[条件定时根因](CONDITIONAL_ROOT_CAUSE_AUDIT.md)：未采用的实验及局限。
- [POWER 验证](POWER_VALIDATION.md)：保留研究结果，非正式默认。
- [重构迁移记录](V1_MIGRATION.md)：历史“v1”命名不等于当前 Protocol v2。

其余文件按原有题名保留。旧报告可能引用已经退役的源码位置、旧命令或本机绝对路径；这些是历史记录，不保证其原命令仍可执行。相关源码已保存到本机整理前快照，不随发行包分发。

`src-reconstruct/` 现在仅保留被忽略的历史 artifacts，目的是保留歌曲、人声分离和证据缓存中的原始路径，避免移动多 GiB 数据并破坏来源记录。`py-roller-artifacts/` 存放新引擎实验与发布验证产物。两者都不进入 sdist/wheel。
