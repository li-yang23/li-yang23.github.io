---
title: Mechanistic Unlearning-Robust Knowledge Unlearning and Editing via Mechanistic Localization论文阅读笔记
description: notes for llm unlearning paper ‘Mechanistic Unlearning-Robust Knowledge Unlearning and Editing via Mechanistic Localization’
tags:
  - ai-explanation
  - llm-unlearning
category: AI-safety llm
date: 2024-12-10
---
大型语言模型中的知识编辑和反学习方法寻求编辑或删除不需要的知识或能力，而不会损害一般语言建模性能。本文研究了**机械可解释性（部分旨在识别与构成模型能力的特定可解释机制相关的模型组件（电路））如何提高编辑和反学习的精度和有效性。** 本文发现在训练通过不同方法定位的组件时，反学习和编辑稳健性存在明显差异。本文强调了主要基于保留输出来定位组件的方法与寻找具有可预测中间状态的高级机制的方法之间的重要区别。特别是，将编辑/遗忘定位到与事实回忆查找表机制相关的组件 1) 会导致跨不同输入/输出格式的更稳健的编辑/遗忘，并且 2) 抵制重新学习不需要的信息的尝试，同时与基线相比减少了意外的副作用，在体育事实数据集和多个模型的 CounterFact数据集上都是如此。某些局部编辑比任何其他基线都更能破坏模型中的潜在知识，从而使反学习对各种攻击更具鲁棒性。

> 所以更好的鲁棒性意味着更全面和彻底地破坏模型中的潜在知识？我怎么知道彻底不彻底呢？

## 本文的背景
大型语言模型 (LLM) 通常会学习编码不良知识。选择性编辑或遗忘此类知识被视为确保人工智能的准确性、公平性和控制力的重中之重。然而，编辑和遗忘这些模型中的知识仍然具有挑战性。常见的编辑和取消学习方法通​​常以影响模型内其他一般或间接知识或能力为代价。此外，通过这些方法实现的编辑可能不够稳健——例如，提示表述中的轻微变化通常仍可引出原始事实或能力，或者在白盒访问的情况下原始答案仍然存在/可提取。
## 本文要解决的问题

如何实现更加鲁棒的遗忘效果，让遗忘后的模型在保留可用性的同时，更加难以提取到遗忘的知识，并且对微调的抵抗性更强。

## 已有方法为什么不行

最近的一些研究探索了依赖于机械可解释性的编辑或遗忘技术，试图追踪网络中的哪些组件存储了特定事实。这些方法（例如因果追踪或归因修补）侧重于测量当干净/损坏的输入被修补到特定组件时输出或任务准确性会受到怎样的影响。Hase 等人 (2023) 质疑了因果跟踪等输出跟踪 (OT) 技术在编辑方面的有效性。本文发现发现基于几种现有 OT 方法的局部编辑和事实遗忘通常与简单地更新整个模型的效果相同或更差。在评估编辑对即时变化和重新学习的稳健性时，以及在探索剩余的潜在知识时，这一点尤其明显。

局部和非局部更新方法都显示无法达成遗忘和编辑的目标。Patil 等人 (2023) 通过中间残差流和改述prompt提取对已编辑事实的正确答案。Yong 等人 (2024) 表明，低资源语言越狱模型会输出不安全的内容，Lo 等人 (2024)；Lermen 等人 (2023)；Deeb & Roger (2024) 证明，使用少量计算/数据进行重新学习会导致模型重新获得不良的知识。Xhonneux 等人 (2024) 表明，仅靠上下文学习就足以在没有明确微调的前提下重新引入不良知识，尽管模型被设计为拒绝输出此类知识。Lee 等人(2024) 表明，即使应用对齐技术使模型无毒，毒性表示仍然存在，只是没有被触发 - 他们认为这是模型缺乏稳健性的原因，并且仍然可以越狱以触发这种不良行为。

Hong 等人 (2024) 通过测量内部激活中的残留知识来评估遗忘，并证明当前的方法无法消除这些残留知识，因此可以被利用。他们尝试通过针对这些残留知识痕迹所在的 MLP 来进行遗忘，但未能找到成功消除残留信息的非预言机遗忘方法。
## 本文的方法
本文的方法在localize-and-edit这个框架里面，首先通过机制可解释性方法定位需要修改的组件，然后使用梯度提升方法修改这部分参数。本文考虑了两种组件定位方法，一种是输出追踪定位方法，一种是事实查找定位。

**组件定位方法（localization methods）**

给定将token序列$$X$$映射到logit$$L\in\mathbb{R}^{\vert V\vert}$$的模型$$M:X\rightarrow L$$，本文将其看作一个有向无环图$$(C,E)$$，其中$$C$$指模型中的组件，$$E$$表示组件间的连接关系。本文的目标是**找到组件在给定任务上的重要性的映射**$$S:C\rightarrow R$$，并定义一个localization $$C_{\tau}$$是一组重要性超过阈值的组件。

$$
C_{\tau}=\{c\vert c\in C, |S(c)|>\tau\}
$$

（在具体实现中固定了阈值，从而保证不同的可解释性方法得到了相同数量的localization组件）

本文首先尝试了输出追踪定位方法，包含因果追踪（causal tracing）和归因修补（attribution patching）两种。因果追踪方法找到对事实关联具有最高直接因果重要性的组件，归因修补方法被视作因果追踪的一种高效近似方法，自动找到具有高直接和间接重要性的组件。*本文假设这些基于输出的技术将优先考虑共享提取组件和其他机制，而不是更扩散的事实查找组件，以重新格式化预测。因此虽然看起来更精确，但将潜在的隐式信息留在了模型中。*

本文随后尝试了事实查找定位（FLU）方法，对MLP层使用了手动派生的定位。

对于能够获得明确的预测范围的任务，本文通过训练逻辑回归模型（探针），从不同层获取的模型内部激活状态来预测正确的结果。本文定义FLU的localization组件是探测准确率迅速增加的MLP层，因为这些层反映了编码的信息足够丰富从而得以得到正确预测。

对于没有明确预测范围的任务，本文首先使用路径修补技术来衡量注意力头和MLP对正确答案和错误答案之间的最终logit差异的直接重要性。路径修补是一种可以发现从发送方组件到接收方组件的单个路径损坏对模型最终 logit 的影响的技术，在没有其他组件介入的情况下显著改变了输出的组件会直接影响logit，因此本文认为是这些组件负责将表征中编码的事实提取到输出logit中，并将这些组件称为属性提取机制。随后本文再次使用路径修补机制来修补MLP和属性提取机制之间的路径，找到通过提取机制传导的对logit差异贡献最大的组件，这些组件通过适当的事实丰富了token表征，然后将其提取了出来。本文将其用作FLU定位。

事实查找定位与输出追踪定位技术的不同之处在于，它关注消融对事实召回机制使用的中间表示的因果影响，而不仅仅是对输出的影响。本文假设稳健编辑的最佳位置是在事实查找阶段，而不是属性提取阶段，因为如果知识仍留在潜在流中，对手可能会设计出提取方法。因此，本文的模型编辑完全集中在事实查找 MLP 上。

**参数更新方法**

在获得了localization集合后，本文执行遗忘或编辑方法，仅更新localization中的组件权重。本文使用了类似[[2024-12-07-large-language-model-unlearning]]的三个权重的联合优化

$$
L=\lambda_1L_{injection}+\lambda_2L_{retain}+\lambda_3L_{SFT}
$$

分别是要做梯度上升的遗忘集的损失，做梯度下降的保留集的损失以及保持泛化的第三方数据集（Pile）的损失。
## 本文方法的效果

本文考虑了两个数据集上的遗忘任务，第一个是Sports Facts，包含1567个运动员和3个运动门类的关联数据，第二个是CounterFact数据集，包含提示以及对应的事实和反事实回答的数据。

对第一个数据集，本文构建了三个编辑/遗忘任务，sports-athlete-editing任务将运动员和原来对应的运动门类打乱，full-sports-editing任务将与某个运动关联的所有运动员的运动门类全部改成了原来类别里没有的高尔夫，sports-unlearning任务将某个运动门类中所有的运动员关联关系全部遗忘。

对第二个数据集，本文构建了两个任务。counterfact-editing任务中，本文首先过滤除了模型能够以超过50%的概率回答正确的事实，然后将一个子集中的事实回答换成另一个虚假回答，保留集为其他没有修改的事实数据。sequential-counterfact-editing任务中，本文逐步编辑了随机选择的16个事实的子集。（第一个一次性打乱，第二个逐渐打乱，想看看不同的事实是否出现在不同的组件中）

本文使用的模型包括Gemma-7B和Gemma-2-9B，Llama-3-8B。

本文的评估方法包括提示补全和对抗重训（relearning）。**提示补全方法**评估编辑方法在保留不相关知识的同时忘记或编辑特定信息的能力。这通过评估遗忘后的模型如何完成来自遗忘集的提示来衡量：首先评估模型回忆不想要的遗忘答案的准确度（遗忘准确度），以及回忆新的想要的编辑答案的准确度（编辑准确度）。此外还评估了在遗忘集以外的事实上的回忆的准确度。

本文使用多项选择题格式进行鲁棒性检查。在遗忘集上，将模型回答基本事实的准确率称为 MCQ 遗忘准确率（更强大的方法应该会降低 MCQ 遗忘准确率），将模型回答新编辑答案的准确率称为 MCQ 编辑准确率（更强大的方法应该会提高 MCQ 编辑准确率）。最后还评估了模型在 MMLU上遗忘后的准确率，将其作为一般语言理解的代理，以衡量任何意外的副作用。

**对抗重训方法**衡量模型承受对抗性再学习的能力，主要是考虑到对手可能有微调访问权限，页为了衡量遗忘的质量。本文将遗忘集分成两个独立的部分，用其中的一半进行再训练，另一半进行评估。本文使用LoRA进行再训练，关注sportd-athlete-editing任务，模型应该合理的从对一半篮球运动员的在学习中推广到对另一半篮球运动员用篮球回答。

**提示补全评估结果**：在sports任务中，在不同的提示格式中，使用事实查找定位方法的遗忘在忘记原始信息和采用替代信息方面表现出色。在counterfact任务中，使用事实查找定位方法的遗忘更多实用新的替代答案进行回答。此外，邻域编辑错误强调其他本地化方法表现出稍微更明显的副作用，无意中编辑了非预期的、语义相似的事实。顺序编辑比 MCQ 中的非顺序编辑显示出略高的稳健性。这一观察结果支持一种方法，即通过对集合进行分区并按顺序应用遗忘，可以更有效地遗忘大量事实。

**对抗重训评估效果**：使用因果追踪等输出定位方法时，对原始“遗忘”数据的子集进行再训练可以恢复很大一部分（高达 63%）据称被遗忘的信息。这表明这些方法可能只是掩盖了这些信息的直接检索，使模型容易通过再训练恢复这些信息。相比之下，事实查找定位对这种对抗性再学习表现出更大的弹性，只有大约 20% 的遗忘信息被恢复。这表明事实查找定位可能更有效地定位和删除底层知识，从而更难通过再训练恢复。

**潜在知识分析**：本文在模型的每一层的提示激活上训练逻辑回归模型（探针）来预测运动员的正确运动门类。本文为每种运动训练二元分类探针并进行最大分类。主要是为了发现模型认为答案是真的真实运动的内部表示，然后将这些探针应用于遗忘集的运动员的提示激活上。事实查找定位遗忘后的模型上的探针始终比其他任何定位方法得到的模型都更少预测应当遗忘的答案，而预测替代答案更多。事实查找探测器分类大部分从非零起始准确度单调收敛到 0（对于忘记准确度）和 1（对于编辑准确度）。在早期层中，其他每个定位的峰值探测器分类忘记准确度都高得多，尤其是输出追踪定位，其峰值分类忘记准确度几乎达到 100%。这强烈表明**这些模型在早期层中仍然明显编码了基本事实答案，而不是编辑答案**。
## 可能的未来方向

想不到......接着做下去就是更全面的组件定位？或者更深刻的遗忘效果评估方法？在MLaaS这种场景下如何评估遗忘效果，验证遗忘过程？机制可解释性目前必须是在白盒场景去思考的，但是MLaaS这种场景下，数据方会要求把模型部署在独立的服务方，自己是没有机会也没有资源去重新训练，或者部署白盒去检测是否遗忘的。