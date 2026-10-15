# 0.10.0 发布准备

本文件是准备清单，不代表已经发布。版本已设为 0.10.0；Changelog 仍标记 Unreleased。发布标签为 `v0.10.0`，需维护者在实际发布时确认。

## 本地检查

```bash
python -m pip install -e '.[test,release]'
python -m pytest
python -m pyroller --version
python -m pyroller align --help
python -m pyroller doctor --json
python -m build --outdir dist/0.10.0
python scripts/check_release.py dist/0.10.0
python -m twine check dist/0.10.0/*
```

`doctor` 的模型就绪状态取决于本机缓存；CI 不下载神经模型。源码包包含公开文档、协议 Schema、测试与 C++ 源码，不包含歌曲、模型、研究缓存或旧引擎。

必须从 sdist 再构建 wheel，并在源码目录之外安装该 wheel 验证 CLI 和协议测试，避免只验证可编辑安装。C++ 扩展使 wheel 与平台/Python 版本相关；本地 macOS wheel 不能安装到 Linux 或 Windows。

## CI 与发布

测试工作流在 Python 3.10、3.11、3.12 验证核心算法、CLI 和协议。发布准备工作流可手动触发构建、测试、包内容检查和 wheel 安装冒烟测试。

发布工作流只在 GitHub Release published 后上传 **sdist** 到 PyPI，并使用受保护的 `pypi` environment / Trusted Publishing。不把单台 Ubuntu 构建的本地 Linux wheel 当作跨 Linux 通用 wheel 发布。用户安装 sdist 需 C++17 编译器；多平台预编译 wheel 的 CI 发布矩阵不在本轮交付范围。

维护者需在发布前确认：

- GitHub 的 PyPI Trusted Publishing 与 `pypi` environment 已配置；本轮未检查远端账号配置。
- 标签、pyproject、包 `__version__` 和 Changelog 一致；补上实际发布日期。
- Rolling Pebble 已按 Protocol v2 迁移后，再更新其生产依赖版本；旧版不可直接升级。
- 实际支持系统的神经依赖可安装；本机验证不能代替所有平台的真实模型测试。

## 数据与研究

`research/scripts/check_release_replay.py` 可在本机现有证据齐全时比较十组 v3 缓存结果，忽略耗时，逐项比较内部结果内容；不运行模型，不将人工时间输入推理。离线工具与本机数据不属于发布包。

详细本轮执行结果记录在 [发布验证记录](RELEASE_VALIDATION_0.10.md)。若工具或环境检查未通过，记录实际限制，不能只因代码版本已改为 0.10.0 就宣称发布完成。
