# 0.10.0 本机发布验证

日期：2026-10-15。状态：发布准备；没有创建 tag、提交远端或上传 PyPI。

## 已验证

- 整理前 97 项算法/运行测试通过；整理后保留适用的算法测试并新增协议、配置、缓存完整性、排他锁和事件日志写入失败测试，当前 120 项通过。
- Whisper 默认固定为 Systran large-v3 的 `edaa852ec7e145841d8ffdb056a99866b5f0a478`；没有新增解码参数优化。
- 六首十组既有 v3 证据回放：忽略耗时字段，其余结果（包含候选、路径选择和最终实例）与整理前逐项一致。
- sdist 和本机 CPython 3.12 / macOS ARM64 wheel 构建成功；wheel 由 sdist 构建。
- 包内容检查通过：无模型、音频、实验 artifacts、0.9 引擎或原型源码；源码包包含协议 JSON Schema、测试和 C++ 源码。
- Twine 对 sdist 和 wheel 的元数据检查通过。
- wheel 安装到独立目录后，在源码目录之外运行保留测试，验证模块来自安装包。
- 从安装包运行 Lily 原声的真实 `run --request`，退出 0；49 个 CTC 块和 8 个 Whisper 窗口均命中缓存，神经推理耗时均为 0。输出 46 行，与此前 v3 结果完全一致。
- 该任务产生 62 条合法 Protocol v2 事件，stdout 与磁盘事件文件相同，最终结果及所有事件通过 JSON Schema 校验。
- 基本依赖和本机两个模型可用；没有重跑 Demucs，也没有新增歌曲效果实验。

## 环境与发布限制

本机虚拟环境 Python 的 OpenSSL 动态库缺失，HTTPS 下载不能据此环境宣称通过。doctor 将 SSL 可用性作为独立 network 字段报告；已有本地模型的离线任务正常。包元数据校验使用独立的系统 Python 工具目录，不改系统 Python 或用户依赖。

本机 wheel 标签为 `cp312-cp312-macosx_27_0_arm64`，仅供当前环境验证，不是广泛平台覆盖声明。发布工作流默认只发布 sdist；未验证 Linux/Windows 的真实模型运行，也未触发远端 CI 或检查 PyPI Trusted Publishing 账户配置。

Rolling Pebble 项目未修改，仍需按 Protocol v2 文档迁移后才能使用此次版本。旧调用入口不兼容。

## 清理

删除活动目录中的 0.9 引擎、旧测试、原型源码、废弃驱动脚本和旧发行产物；历史文档归档。整理前约 644 KiB 源码快照保存在本机 `py-roller-artifacts/pre-0.10-source-20261015.tar.gz`，不进入发布包。

本轮另删不再作为基线的 large-v2 权重约 2.9 GiB。v3、正式 CTC、Demucs、音频、人工歌词和神经证据缓存保留。`src-reconstruct/artifacts/` 暂保留原位置以维持历史来源路径。

## 产物

- `dist/0.10.0/`：本机 sdist 和 wheel。
- `py-roller-artifacts/release-0.10.0/replay.json`：十组回放一致性结果。
- `py-roller-artifacts/release-0.10.0/lily/`：安装包完整任务产物。
- `py-roller-artifacts/release-0.10.0/lily-stdout.jsonl`、`lily-stderr.log`：独立工作进程输出。

这些本机产物和音频路径不适用于外部用户，也不是发布包内容。
