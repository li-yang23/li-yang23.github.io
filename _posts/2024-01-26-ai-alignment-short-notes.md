---
layout: post
title: AI对齐论文阅读杂记
description: notes for ai-alignment papers do not need to write a whole blog
tags: ai-alignment
categories: AI
giscus_comments: true
date: 2024-01-26
---
## arxiv

### 2023

1. Trustworthy LLMs: A Survey and Guideline for Evaluating Large Language Models’ Alignment
   1. LLM对齐的目标是构建一个可信任的LLM，文章将对齐的角度分为可靠性、安全性、公平性、误用的抵抗力、可解释性、社会规范和鲁棒性等七类，对每一类的对齐内容进行描述，相关工作提了一点。最后文章提出了自己设计的对齐程度的测量方法，主要思路两点：使用选择题代替问答题从而让答案可以被量化统计，使用被良好对齐的LLM来为不可量化的回答打标记，避免人类参与，提升测量效率。带来的问题就是测量方法高度依赖LLM对于输出格式的理解和处理能力，以及对齐程度受做测量工具使用的LLM的对齐程度限制。*本质上只能算二级度量，毕竟做测量工具使用的那个LLM肯定不会是用这个方法度量出来的。*
2. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
   1. 针对原来的训练方法都需要额外的偏好标注数据集的问题，设计了新的训练方法，利用sft阶段的数据集进行对齐
   2. 基本逻辑其实有点对抗优化：首先训练评分模型来区分sft回答与LLM生成回答，然后训练LLM来生成评分模型更加难以分辨的回答。但最后使用LLM来表示了评分模型，将两个环节统一，做成了self-play形式（中间的推理过程和分析看明白了扩展成一篇博客）
   3. 最终的损失函数与DPO基本一致，只是reference model是上一轮训练的模型，好回答使用sft数据，坏回答由训练的LLM自行生成。
   4. > 不知道这样的reference会不会导致分布偏移问题，以及如果sft数据更强调专业性而不强调人类意图不就歇逼了？

### 2024

3. Self-Rewarding Language Models
   1. 尝试解决需要超人类反馈提供训练信号的超人类智能体的对齐问题。提出了自奖励LM，通过LLM-as-a-Judge prompting使用LLM自身作为奖励模型。自己生成对齐的数据进行训练。
   2. 方法旨在能够同时拥有指令遵循和自指令生成两个能力的模型，可以在给定用户请求后生成高质量的无害回答，同时可以生成并评估新的指令遵循样本并添加到训练集中。
   3. 方法使用一组指令微调数据（指令问答对）和一组评估微调数据（评估指令问答对）进行监督微调，然后使用AI反馈（自反馈）数据进行增强，迭代完成训练。每次迭代通过IFT和EFT进行SFT后进行两轮的AIFT数据生成和微调训练。
   4. AIFT的生成：首先从IFT数据中采样prompt，使用few-shot prompting生成新的prompt，然后利用新的prompt生成若干个回答，最后使用LLM的LLM-as-a-Judge能力对每个回答进行评分。AIFT数据可以是分数最高和最低的回答构成的好坏问答对，也可以是满分回答构成的gold问答数据。
   5. 每轮训练流程包括一轮SFT和两轮DPO，每一轮都使用前一轮训练得到的模型作为基座模型，SFT和DPO训练完成后都用模型生成一版AIFT供下一轮训练
   6. 迭代次数的缩放效应，奖励入侵问题，安全性评估以及安全训练方法，以及这种训练方式是否可以用于提升模型安全性
