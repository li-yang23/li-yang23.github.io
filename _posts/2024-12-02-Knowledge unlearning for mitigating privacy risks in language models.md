---
title: Knowledge unlearning for mitigating privacy risks in language models阅读笔记
description: notes for Knowledge unlearning for mitigating privacy risks in language models
tags:
  - llm-unlearning
category: AI llm-unlearning
date: 2024-12-02
---
预训练语言模型（LMs）在初始预训练过程中会记忆大量的知识，包括可能侵犯个人生活和身份隐私的信息。以往针对LMs隐私问题的工作主要集中在*数据预处理*和*差分隐私*方法上，这两者都需要重新训练底层的LM。本文提出知识遗忘作为一种替代方法，以减少LMs事后的隐私风险。本文通过简单地对目标token序列执行梯度上升，在不降低大型LMs的一般语言建模性能的情况下忘记目标序列。本文发现顺序遗忘比一次性尝试遗忘所有数据更有效，并且遗忘高度依赖于被遗忘的数据类型（领域）。通过与已知可以减轻LMs隐私风险的先前方法进行比较，本文方法可以在已知数据容易受到提取攻击的场景中提供更强的经验隐私保证，同时更加高效和稳健。

## 本文的背景

攻击者可以通过数据提取攻击从预训练模型中获取包括个人身份信息在内的各种隐私信息（AI聊天机器人Iruda在2021年成为了第一个因为违反个人信息保护条例而被起诉的AI系统）。随着模型规模扩大，提取训练数据变得更加容易。目前行业做法是发布包含几十亿个参数的超大模型供公众使用，因此让大模型提供隐私保护保证就显得愈发重要。

## 本文要解决的问题

法律法规的被遗忘权要求企业在收到个人要求后从模型中删除对应的个人信息。

## 原来的方法为什么不行

原来的一般做法是在训练之前就从数据中删除个人隐私信息，或者设计满足差分隐私的算法。但这两种方法在收到个人遗忘要求后都需要重新训练模型，导致这两种方法无法有效应用于训练成本超高的大模型上。此外，数据预处理方法假设隐私信息易于识别、指定和删除，而DP算法只能保证对具有明确隐私边界的信息的保护，这使得它们在现实场景中存在不足，因为每个人的隐私标准可能不同。

## 本文的方法是什么

使用*知识遗忘*作为替代方法，对少量的参数进行微调，而不是重新预训练整个模型。具体的遗忘方法是反向优化，即将梯度下降方向逆向，最大化损失函数。
$$
\mathcal{L}_{UL}(f_{\theta}, \boldsymbol{x})=-\sum_{t=1}^T\log(p_{\theta}(x_t\vert x_{<t}))
$$

提出了两种度量指标来衡量语言模型的隐私泄露风险：提取相似性和记忆准确度。

**提取相似性：** 根据给定的token序列和语言模型，将提取相似性定义为生成的后缀和真实后缀的n-gram平均重合程度。
$$
\begin{aligned}
EL_n(\boldsymbol{x})=\frac{\sum_{t=1}^{T-n}OVERLAP_n(f_{\theta}(x_{<t}),x_{\leq t})}{T-n} \\
OVERLAP_n(\boldsymbol{a},\boldsymbol{b})=\frac{\sum_{c\in ng(\boldsymbol{a})}\mathbb{1}\{c\in ng(\boldsymbol{b})\}}{ng(\boldsymbol{a})}
\end{aligned}
$$
**记忆准确度**：记忆准确度定义为使用不同长度前缀提示模型后，模型最有可能生成的是真实的next token的概率。
$$
MA(\boldsymbol{x})=\frac{\sum_{t=1}^{T-1}\mathbb{1}\{arg\max(p_{\theta}(\cdot\vert x_{<t}))=x_t\}}{T-1}
$$
最后将这两个指标合起来，定义了遗忘的度量方式：如果指定的序列在这两个指标上的结果都不大于在一个陌生的token序列集合上的平均值，就可以认为这个序列达到了遗忘要求，即
$$
\begin{aligned}
EL_n(\boldsymbol{x})&\leq\frac{1}{\vert D'\vert}\sum_{\boldsymbol{x'}\in D'}EL_n(\boldsymbol{x'}) \\
MA(\boldsymbol{x})&\leq\frac{1}{\vert D'\vert}\sum_{\boldsymbol{x'}\in D'}MA(\boldsymbol{x'})
\end{aligned}
$$
## 本文怎么说明效果的

本文使用了在整个Pile预料数据集上预训练的GPT-Neo-125M/1.3B/2.7B和在部分去重的Pile数据集和一些其他领域数据集上预训练的OPT-125M/1.3B/2.7B作为基座模型。从[Training Data Extraction Challenge](https://github.com/google-research/lm-extraction-benchmark)中采样了一些样本作为目标数据来度量隐私风险，这个数据中包含Pile预料数据集的16个领域的15,000个样本（每个的token序列长度都是200）。然后使用了9个不同的数据集来评估遗忘之后的模型可用性，包括语言推理能力（Hellaswag，Lambada）、常识推理能力（Winagrande，COPA），科学推理能力（ARC-Easy，ARC-Challenge，Piqa，MathQA，PubmedQA）。还是用了四个对话数据集来衡量模型的泛化能力（Wizard of Wikipedia，Empathetic Dialogues，Blended Skill Talk，Wizard of Internet）。Lmabada使用了验证集，其他的用了测试集。

**主要实验结果**

遗忘效果实验结论主要包括：
1. OPT在两个指标上的值都低于GPT-NEO，说明数据去重确实有助于减轻隐私风险
2. 差分隐私解码两个指标都最低，可以有效抵御提取攻击，但也严重降低了对话生成任务的表现
3. 本文遗忘方法在小模型上导致分类和对话任务的性能退化，1.3B模型上对话任务退化，而2.7B模型上能够保留大部分之前的性能。
4. 随着模型规模扩大，目标序列被遗忘所需的周期更少，与上一条结合说明更大模型接受遗忘的能力更强
5. 遗忘方法相较重新训练可以有更高的计算效率

序列遗忘比批量遗忘的稳定性更强：每次只遗忘一部分样本，然后执行多次遗忘来忘掉更大批次的样本比一次性遗忘所有样本的表现下降的程度更低。

尽管经历了相同数量的token更新（10个遗忘周期），不同领域却导致了截然不同的结果：ENRON EMAILS导致平均LM性能仅下降了-0.4%，而USPTO BACKGROUNDS导致了-4.5%的性能下降。此外，最终的提取相似性根据领域不同而有所变化，这表明某些领域（例如，FREELAW）比其他领域更难被遗忘。

最后，那些结构化程度更高的领域，即数据包含某种模式的领域，如电子邮件列表（ENRON EMAILS）或代码（GITHUB (CODE)），似乎**与结构化程度较低的领域相比，导致LM性能的下降较少**，后者指的是数据主要由原始英文文本组成的领域，如期刊提交的评论（PUBMED CENTRAL）。
