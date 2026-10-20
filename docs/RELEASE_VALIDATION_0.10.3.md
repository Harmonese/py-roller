# v0.10.3 发布验证

日期：2026-10-20。该版本包含普通话和多语言音素功能；它们此前在版本号仍为 0.10.2 的工作目录中完成实验，本次统一整理发布，不改变已验证的模型或评分参数。

## 功能范围

- 保留英文 en 与默认 ordered/unordered 选择；新增实验性普通话 zh、有限 zh-en 以及通用 mul 配置，扩展 CLI、Python、doctor、模型管理与 Protocol v2。
- 固定中文 XLSR 与多语言 XLSR-53-espeak 检查点；语言单位、简繁规范化、发音、编码与模型时间网格分层，原文和来源关系保留。
- mul 为可选安装项，另需本机 espeak-ng。不引入 TTS 或重新训练，不将测试过三种语言等同于所有语言/混唱通过验收。
- 不包含音频、歌词数据集、神经权重或历史缓存。源码和 wheel 的模型下载仍由模型管理完成。

## 算法及歌曲检查

功能实现阶段 167 项测试通过；10 组既有英文 v3 缓存结果不变，2 首中文原声 ordered 回归一致。三首 mul 原声 ordered 的缓存回放一致，354 个事件和三个公开结果符合协议 Schema；音频与参考 LRC 哈希不变。

|mul 样本|参考行|未定位|已定位但 >0.5 秒|其中 >2 秒|
|---|---:|---:|---:|---:|
|The Veldt|28|0|16|8|
|流水线|32|5|10|0|
|Песня без слов|24|0|10|2|

同名 LRC 仅作参考，未逐句人工确认。参考时间不参与推理。中文漏行最终受到局部窗口预算阻塞，英文/俄文有句首偏早，不能把有时间的行全部称为正确实例。详细过程见仓库 [中文实验](research/CHINESE_VALIDATION.md) 与 [mul 实验](research/MULTILINGUAL_VALIDATION.md)；研究报告只随 Git 仓库保留，不进入发行包。

## 发布检查

发布前已完成：

- 源码测试与源码目录外安装 wheel 的测试均为 167 项通过；CLI 报告 `0.10.3 (protocol 2)`。
- sdist→wheel 构建、内容/版本检查与两份产物的 Twine 检查通过；wheel 45 项、sdist 90 项，未混入模型、音频或缓存。
- 最终 wheel 在源码目录之外运行俄文 mul ordered 缓存任务：73 个 CTC 块、12 个 Whisper 窗口命中，模型推理耗时为零，24 行与之前结果一致。90 个 stdout 事件与磁盘日志一致，事件版本均为 0.10.3。
- wheel 的 en/zh/mul doctor 均报告就绪；当前活动环境中的发音依赖和模型可供离线使用。

本机 Python 3.12 缺失 Homebrew OpenSSL 动态库，在线 Python HTTPS 不可用；离线模型推理与检查已完成，网络状态单独报告。Git/gh 及系统 Python HTTPS 独立可用；不通过修改用户系统环境掩盖该限制。PyPI 上传沿用 GitHub Release 的 Trusted Publishing 工作流。

本地 wheel 仅供当前 macOS arm64/Python 3.12 验证；发行工作流发布 sdist，不把本地 wheel 当作跨平台发行包。远端发布完成与否以 GitHub Actions 和 PyPI 实际结果为准。
