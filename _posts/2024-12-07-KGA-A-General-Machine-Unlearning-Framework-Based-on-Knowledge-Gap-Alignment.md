---
title: 知识间隔对齐论文笔记
description: notes for “KGA-A General Machine Unlearning Framework Based on Knowledge Gap Alignment”
tags:
  - ai-explanation
  - llm-unlearning
category: AI
date: 2024-12-07
---
最近关于“被遗忘权”的立法引发了人们对机器反学习的兴趣，即学习到的模型被赋予忘记特定训练实例信息的功能，就好像它们从未存在于训练集中。以前的工作主要集中在计算机视觉场景中，在很大程度上忽略了 NLP 领域遗忘的本质，因为文本数据比图像包含更多明确和敏感的个人信息。本文提出了一个称为 KGA 的通用遗忘框架来诱导遗忘。与试图恢复梯度或强制模型接近某一特定分布的工作不同，KGA 保持了分布差异（即知识差距）。此外，本文首先将遗忘方法应用于各种 NLP 任务（即分类、翻译、响应生成），并提出了几个有针对性的反学习评估指标。在大规模数据集上的实验表明，KGA 比基线有了全面的改进，其中广泛的分析进一步验证了 KGA 的有效性并为 NLP 任务的反学习提供了见解。

## 本文的背景
机器学习模型通常使用从个人用户收集的大量数据进行训练。个人数据本质上是敏感的，因为它可能包含个人地址和医疗记录等信息。在不知情的情况下， 训练后的模型可能会侵犯用户的隐私，因为其参数会永久编码个人信息及其衍生信息。因此，机器遗忘引起了研究和工业界越来越多的兴趣，旨在帮助模型忘记训练集中的某些特定数据，同时保持现有模型的性能。除了隐私优势外，机器遗忘还可以解决忘记有毒和脏数据的问题。

## 本文要解决的问题
直接重训对于大模型来说成本过高难以完成，实际情况中数据删除请求频繁，不断进行重训不切实际。深度学习模型是黑盒模型，由于模型权重和数据之间的关系不明确，很难知道遗忘时应该修改权重的哪些部分。

## 已有方法为什么不行
（当时的）已有工作主要集中在计算机视觉领域，对于NLP领域关注较少。而且当时的方法只能有效处理少量数据删除请求，而NLP应用中的删除请求可能有数百个。此外由于NLP生成模型通常基于Seq2Seq框架，且有复杂的注意力机制，（当时）基于梯度计算的遗忘方法难以直接应用。
+ 精确遗忘的方法难以扩展到大模型上：SISA方法在遗忘请求规模很大时效率很低，并且需要时刻保持所有训练数据，而这并不现实。
+ 近似方法需要计算训练数据上的Hessian，依然非常耗时。或者假设了在遗忘数据上的表现应当和随机模型一样，而这与实际不符，实际应当是如同没见过数据一样。此外，已有方法或者需要很强的假设，或者难以扩展到神经网络模型上。

## 本文的方法
KGA 的灵感来自一项通用知识适应工作，采用权重和函数空间先验来重建模型的梯度。本文的知识差距定义为使用不同数据训练的两个结构相同的模型的预测分布之间的距离。

定义类似，遗忘数据集$$D_f$$依然来自于训练数据集。然后还有一个小型的不在训练数据中的额外数据集$$D_n$$，以及（可选的）剩余训练数据里面的一小部分构成的$$D_r$$

KGA方法首先在$$D_f$$和$$D_n$$上分别训练两个相同结构的模型$$A_f$$和$$A_n$$(两边$$D_r$$都可以参与)，或者根据某些预训练模型微调而来。

然后结合遗忘前的模型$$A_D$$（因为是在整个训练集$$D$$上训练的），本文将知识差距（knowledge gap）定义为*原模型$$A_D$$与普通模型$$A_n$$在额外数据集上的分布差距*和*遗忘后模型$$A$$与遗忘数据集上的模型$$A_f$$在遗忘数据集上的分布差距*间的距离：

$$
\mathcal{L}_a=\sum_{(y,z)\in(D_f,D_n)}\vert dis_{(D_n)}(A_D,A_n)-dis_{(D_f)}(A,A_f) \vert
$$

基本的思路是$$A_n$$见过$$A_D$$没见过的$$D_n$$，而$$A_f$$见过$$A$$（理应）没见过的$$D_f$$，那么$$A_f$$和$$A$$在$$D_f$$上的表现差异应当和$$A_n$$与$$A_D$$在$$D_n$$上的表现差异类似。

> 那为什么不用$$D_n$$直接微调$$A_D$$得到$$A_n$$呢？这样$$A_D$$可以当作一个重训模型。

知识差距的优化目标如上式，其中$$dis(c\cdot,\cdot)$$采用了KL散度。在剩余数据集上训练以保留模型可用性的训练目标使用了和原模型分布的KL散度：

$$
\mathcal{L}_r=\sum_{x\in D_r}KL[Pr_{(A^*)}(x)\Vert Pr_{(A_D)}(x)]
$$

最后联合优化这两个损失得到遗忘后模型$$A^*$$：

$$
\mathcal{L}=\mathcal{L}_a+\alpha\cdot\mathcal{L}_r
$$

## 如何说明的效果
使用不同的NLP生成任务的经典指标（Micro F1， BLEU4，PPL）衡量可用性，使用JSD，LPD和PDLP衡量遗忘效果
+ Jensen–Shannon Divergence (JSD)：两个分布分别为pred和true分布，计算两个KL散度，然后0.5加权
+ Language model Probability Distance (LPD)：pred和true模型的结果困惑度的距离除以true的困惑度
+ Proportion of instances with Decreased Language model Probability (PDLP)：计算遗忘后模型结果的困惑度下降了的样本比例

与重训模型和SISA模型进行比较，还有使用hessian的LCODEC方法（选择了一部分参数来降低了计算成本），BADT方法（让遗忘样本上的表现和随机模型相似）。使用了DistilBERT，encoder-decoder transformer架构模型作为基础模型。

主要实验结论包括：
+ KGA在测试集上可以有效保持模型可用性
+ KGA在遗忘集上的性能以及预测分布与重训模型更加接近
+ 忘记原始模型中的数据并不意味着遗忘后的模型根本无法处理这些样本：重训模型在遗忘样本上的表现相较原模型下降了，但依然与测试集的结果接近，也就是说遗忘并不代表着要跟随机模型一样。
+ 时间比重训类方法下降明显，成员推理攻击成功率比重训高，但没有优于BADT，应该是因为BADT接近随机模型的思想让推理攻击更难生效。
+ 某些极端简单的样本的遗忘导致了模型可用性的上升：遗忘简单样本使BLEU分数上升了。
+ 在遗忘包含某个特定单词的所有样本之后，模型无法再生成对应的单词，但会找到其他跟这个单词相近的词来代替。
## 可能的未来方向

+ 遗忘哪些样本会对模型的效果有提升？
+ 知识蒸馏方法可以对得到的模型效果给出上限保证吗？
+ 知识蒸馏方法应该效率提升在于使用较小的数据集微调代替了使用较大数据集预训练，但代价就是需要多个模型，这个模型成本可以降下来吗，比如用Adapters的方法等？（感觉应该有人做过了）

