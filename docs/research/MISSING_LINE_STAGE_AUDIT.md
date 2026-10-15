# 缺句的分阶段归因：重点Fly

基于 `posterior-three-song-v1` 的文本次数缺失清单，回看无提示Whisper原始窗口、拼接词流、候选、选句与CTC长短版本复核。没有新模型推理，没有修改默认流程。参考只定位事后诊断；未重新人工核听。50项包括Within人声的32项，不能把这些视图相加当成不同歌曲的漏句总数。

额外做了24次**原始单窗口的匹配回放**：同一套P1设置、完整歌词身份清单、原始逐词时间，未加入参考词或参考时间。这能证明原窗口的线索足以进入现有算法，但不等于拼接修改后整曲一定恢复，也不证明时间准确。

## 三类问题的边界

1. 原始窗口已经遗漏/替换：说明本次模型输出不够，不能证明模型或原音频永久没有可用信息。CTC是否能单独重新发现这些实例，本次没有另做参考引导的声学搜索。
2. 后处理损失：原始窗口有线索，拼接删掉/覆盖；或拼接保留线索，但候选准入/身份竞争失败。
3. 最后定时：当前legacy与posterior只调整已选实例的时间，不删减身份和次数。故本次“同文本次数不足”的主要原因不在这一步。CTC长短版本复核是另一个模块，不能与末端定时混为一谈。

## Fly原声：9项逐例

| 参考时间 | 歌词 | 原始证据与实际损失点 | 归因 |
|---|---|---|---|
| 23.07 | Fly for your life, fly for your life | 原窗口只有前一句Take it tonight之后的一次fly for your life，未输出所需的额外两次；人声版原窗口却有三次fight for your life | 本次原声ASR重复遗漏 |
| 43.12 | There's one thing I know for sure | 原窗口直接跳到47秒You've gotta fly for your life，没有该句可辨认文本 | 本次ASR遗漏 |
| 56.74 | A world I've never seen before | 原文What I've never seen before；后四词完整，最高相关整句相似度0.667，低于0.72，候选未入场 | ASR前缀错误＋匹配容错不足 |
| 60.46 | A silhouette into the sun | 原文Your silhouette in the sun；silhouette/the/sun线索明确，相似度0.60，候选未入场 | ASR小词错误＋匹配容错不足 |
| 147.41 | You've got those dangerous | 原文多了eyes；短版完整候选相似度1.0，仍输给覆盖更多词的长版；短版3.5分、完整长版4.5分 | 长短身份竞争，非整句消失 |
| 201.81 | Fly for your life, fly for your life | 原始输出Take it tonight, fight for your life，然后Just fight for your life；中间额外重复未输出 | 本次ASR重复遗漏 |
| 206.10 | Just fly for your life | 原窗Just fight for your life完整；208秒无共用词切点丢弃208.10–208.50的life；直接回放原窗可选出目标句，相似度0.80 | 拼接造成可证实的后处理损失 |
| 214.93 | We'll keep falling in love | 原窗只输出208秒这一次，之后直接进入225秒Oh it's a miracle；人声版有215秒另一次 | 本次原声ASR重复遗漏 |
| 230.19 | I got spirit, I got wings | 原窗包含229.62起的完整目标词序；234秒切点丢掉后两个got/wings。原窗回放目标候选相似度1.0且被选中 | 拼接造成可证实的后处理损失 |

按主要损失点归类：2项拼接、2项候选容错、1项长短身份、4项原始输出缺少所需实例。不是对“修复成功率”的预测。

## Fly分离人声：6项逐例

| 参考时间 | 歌词 | 原始证据与实际损失点 | 归因 |
|---|---|---|---|
| 23.07 | Fly for your life, fly for your life | 原窗Take it tonight后有三次fight for your life；第三次在26.36–27.98，下一窗未输出它。26秒无共用词切点删掉第三次。剩余双重复候选与前句抢词而落败。原窗回放23.44秒候选相似度0.75且被选中 | 先拼接损失，再候选竞争；强后处理证据 |
| 54.56 | You've got those dangerous | 原窗输出第二个dangerous eyes；短版完整候选入场但长版获胜。CTC复核更偏好短版并标alternative_pending_review，政策不自动改身份 | 长短身份问题；不是没有听见整句 |
| 147.41 | You've got those dangerous | 原窗只有一份长版，随后直接We'll keep falling in love；原声窗能看到额外的第二份。不能把给前一句匹配的短候选冒充第二次实例已经发现 | 本次人声ASR未区分额外短句；兼有长短身份歧义 |
| 201.81 | Fly for your life, fly for your life | 同原声，原始窗口缺少额外双重复，只有前句尾部及Just fight... | 本次ASR重复遗漏 |
| 206.10 | Just fly for your life | 前窗完整Just fight for your life；后窗误成Just like falling in love。拼接以Just作锚，保留前窗Just后全接后窗，把正确后缀覆盖。原窗回放目标相似度0.80且被选中，但粗起点202.06本身偏早 | 拼接错误选择，且ASR时间需后续修正 |
| 230.19 | I got spirit, I got wings | 原窗I got spirit / Yes! I got wings!；234秒切点删去got/wings。原窗回放可容忍Yes，目标相似度0.857且被选中 | 拼接损失 |

按主要损失点：3项拼接、1项长短身份、2项本次ASR输出遗漏/合并。不能据此说剩余两项永远无法由模型重跑或其他证据恢复。

## 拼接的两个结构问题

`stitch_windows` 没有共用词时，在固定core边界两边按词时间中点裁切。它隐含“后窗会提供边界后的内容”；Fly 234秒处后窗恰好漏掉整段，因此把前窗已经识别的got/wings扔掉。前窗原本覆盖到236秒，有足够上下文，但不被保留。

有共用词时，使用一个锚点决定整段切换，没有比较锚点之后哪个版本更支持完整短语。Fly人声Just例子中，一次错误切换把fight for your life换成like falling in love。由于Just的时间区间很长，它能通过当前按词中点的时间门槛；不是简单把词相似度换成发音相似度就自然解决。

本次发现5个音频视图/实例具备“原窗可被同一匹配器选中、拼接后失败”的直接证据：Fly人声23秒，原声与人声206秒，原声与人声230秒。这对应3处歌曲位置，并非5个不同位置。

## Blossom与Within

- Blossom原声无同文本次数缺失，不代表所有时间正确。
- Blossom人声35.67秒：原窗为I'm now wishing for a life, a picture you，happy/oh缺失，well-wishing、I也被替换。拼接保留这些词，没有额外截断。完整候选最高相关相似度约0.636，未达0.72。它有很明显的部分身份线索，属于**部分ASR错误＋整句准入不足的混合问题**，不能归成纯粹“没有模型证据”；原窗回放仍未通过当前匹配。
- Within原声147.88和153.32秒：原窗被写成I am lost, I can't find my way back to the way I used to be，分别缺失Many rooms...、Where are the locks...的可辨认词序。拼接没有把完整正确版本扔掉。是本次无提示ASR输出的问题；此前提示实验局部可恢复的事实也说明不能把它称为绝对模型能力上限。
- Within人声32项缺失：用户已核实Demucs把机器人声当伴奏移走；应优先归入输入音频选择/分离损伤，不宜用这份音频评判定时或匹配算法的能力。原声仅缺2项，对照差异明显。

## CTC到底参与了什么

当前CTC没有独立从空隙补充新歌词实例的入口。若拼接或候选阶段没生成这一句，最后定时根本不会问CTC“这里是否还唱了这句”。所以不能因为最终缺句就推出CTC没有能力识别它。

长短版复核是例外：Fly人声54.56秒附近CTC短版得分-37.41、长版-62.77，均通过前缀支持；代码选择标记短版候选、保留原身份，原因是“弱句尾证据不能证明eyes不存在”。该保守策略可能留下参考中的长短差异，但不能把负分更高直接当成短版必然正确。原声147秒附近两版都没通过前缀支持，CTC没能提供足够可靠的纠正。

最后定时确实存在重复实例错位和跨句借音，但那是此前报告中的时间/实例定位问题，并不是这些文本次数缺口的直接来源。对于漏句优先的目标，继续只修改末端定时收益有限。

## 建议优先级

1. **先保全重叠窗口中已经存在的短语证据。** 不再在无法确认重复的边界，直接丢弃前窗剩余词；保留有来源的替代词段，或先窗口内生成歌词候选再做跨窗去重与竞争。必须区别同一次演唱的两个窗口拷贝与真实重复，不能简单把所有窗口文字拼接起来。
2. **再处理有充分局部身份线索的候选。** 对What I've never seen before、Your silhouette in the sun等，不只放宽全局0.72阈值；需要允许部分支持提名候选，保留缺词/替换位置，并以局部音频验证。直接把所有类似串填成歌词会增加误识别。
3. **长短版作为身份歧义单列。** 不和整句缺失混为一个优先级；保留上下文和CTC诊断，由独立验证判断是否可自动纠正。
4. 无可辨认原始线索的段落，再考虑原声/人声互补或有预算的局部模型重跑，而非要求字符串匹配凭空造出歌词。

优先修复拼接的证据最强：不需要换模型、放宽整曲匹配或依赖CTC从零猜句，就能保住当前模型已经提供的信息。本次仅完成归因和诊断回放，未改推理流程。

## 产物

- [逐阶段原文与单窗口回放](../../py-roller-artifacts/posterior-three-song-v1/missing-stage-audit.md)
- [完整候选、拒绝原因、原始逐词信息](../../py-roller-artifacts/posterior-three-song-v1/missing-stage-audit.json)
- 重现：`python scripts/audit_missing_stages.py py-roller-artifacts/posterior-three-song-v1`。
