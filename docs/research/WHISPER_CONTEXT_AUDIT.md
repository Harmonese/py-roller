# Whisper 上下文使用审计

本轮只检索文献、阅读本地实现、检查已保存窗口，没有再次运行Whisper，也没有改变提示流程。环境为 faster-whisper 1.2.1 / CTranslate2 4.8.1，large-v2。

## 已证实的接口语义

1. `initial_prompt` 放入 `all_tokens`，随后作为 `previous_tokens` 传给 `get_prompt`，位于 `<|startofprev|>` 后。它是前文条件，不是一个只改变词汇概率的独立约束器。Whisper作者说明了这一训练用途，也说明它可被用于词汇提示；因此不能简单断言用它提供歌词属于API误用。
2. 当前检索将最多6条完整歌词按匹配得分排列，以换行拼接。它既不是已确认的前文，也不保证与当前演唱顺序一致，重复次数也不保留。这是可能影响生成结构的设计假设，尚未通过提示格式消融证明是唯一根因。
3. `hotwords` 在该版本也进入同一个前文token区域。对本次47个非空提示实际调用 `get_prompt`，把相同字符串从initial_prompt改传hotwords，首轮解码得到的token序列47/47完全相同。仅改参数名不会解决首轮问题。两者在后续内部解码的携带行为不同，不能声称全程等价。
4. `prefix` 是当前音频已有转录的前缀，并非“可能出现的歌词清单”，不应拿整个库存替代initial_prompt。
5. `condition_on_previous_text=False` 不会禁用首轮initial_prompt；在一次transcribe调用内部的后续解码会重置前文。外部每26秒核心调用一次，因此每个外部窗口重新获得提示，但内部第二次解码未必仍带提示。

## 本地失败窗口的证据

Within原声窗口4（102–132秒）与窗口7（180–210秒）的提示只有31个token，远小于192的检索预算。分别从无提示的6段、9段压缩到一段，提示输出只包含两个短句；最后一段跨越约21.26秒。遗漏在原始Whisper输出中已经存在，不能归咎于后续CTC或跨窗口拼接。

这些输出的token数也远低于224的生成上限，不能把这两处解释成token预算耗尽。我们关闭了VAD、no_speech跳过、hallucination_silence_threshold，没有配置禁止重复n-gram，因此这些过滤器不是已知解释。

窗口3还出现了词粘连（例如一个多余字符串直接接到本来可识别的词前），这会进一步破坏P1匹配。单纯整窗替换无提示结果，会把新得到的内容和这些退步一起引入。

## 时间推进机制

`_split_segments_by_timestamps` 会根据生成的timestamp token决定本次推进位置；在没有成对时间戳或单时间戳结尾等分支，可能直接推进整个音频块。启用word timestamps时，另有分支按最后一个词结束时间修正seek。因此提示改变文本/分段以后，也可能改变是否重新处理余下音频。

维护者已明确指出initial_prompt可以同时改变文本和分段时长。我们的保存结果与这个现象一致。不过当前缓存保存的是最终segments，而非每次原始decoder输出和所有seek决策；不能仅凭缓存严格证明漏句全部由seek引起，也不能声称模型“以为歌词已经唱完”。

## 研究如何指导下一次实验

研究表明预期答案/短语可以帮助Whisper，但效果依赖内容和音频条件；短语提示不是本质上无效。语音研究中的收益不能直接移植到重复歌声，也不应照搬其静音裁剪到歌曲时间轴。

最小可辨别实验应只选上述失败窗口及一处成功难句，保持音频和所有解码参数不变：

- 无提示；当前完整歌词提示；短小去重词汇/短语提示。
- 为每次内部decode记录输入prompt token、原始生成token、timestamp切分和seek变化。
- 先比较原始文字覆盖和重复次数，再比较词时间与下游匹配。
- 不把“原句不再出现在结果”自动解释成提示错误，需定位是生成遗漏、分段/推进还是词粘连。

若短词汇提示仍压缩重复，再单独研究固定音频范围解码与时间戳推进解耦；不要同时切换提示格式、窗口长度、VAD和模型。保留无提示结果作独立证据，暂不直接覆盖它。

## 来源

- [Whisper作者解释prompt与prefix](https://github.com/openai/whisper/discussions/117)：前文条件、当前音频前缀的区别。
- [Whisper词汇提示讨论](https://github.com/openai/whisper/discussions/328)：可以用initial_prompt引入相关词汇，非保证性约束。
- [faster-whisper维护者回应提示改变分段时长](https://github.com/SYSTRAN/faster-whisper/issues/474)：文字和分段都可能改变；用户示例仅作为旁证。
- [faster-whisper v1.2.1实现](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)：get_prompt、generate_segments、_split_segments_by_timestamps；本地安装文件已核对并保存哈希。
- [Cheng, Interspeech 2024: Context-Aware Speech Recognition Using Prompts for Language Learners](https://www.isca-archive.org/interspeech_2024/cheng24b_interspeech.pdf)：相关短语提示在语言学习语音上的收益及局限，不能视为歌声结果。

复现检查：`scripts/audit_whisper_context.py`。产物：`py-roller-artifacts/conditional-v1/whisper-audit.json`。这份结果区分了源码可证实机制、本地可观察现象和待验证因果关系。
