---
layout: post
title: 论文阅读杂记
description: notes for papers do not need to write a whole blog
tags: ethical-value-alignment
categories: AI
giscus_comments: true
date: 2024-01-16
---
## AI-Alignment

1. Trustworthy LLMs: A Survey and Guideline for Evaluating Large Language Models’ Alignment
   LLM对齐的目标是构建一个可信任的LLM，文章将对齐的角度分为可靠性、安全性、公平性、误用的抵抗力、可解释性、社会规范和鲁棒性等七类，对每一类的对齐内容进行描述，相关工作提了一点。最后文章提出了自己设计的对齐程度的测量方法，主要思路两点：使用选择题代替问答题从而让答案可以被量化统计，使用被良好对齐的LLM来为不可量化的回答打标记，避免人类参与，提升测量效率。带来的问题就是测量方法高度依赖LLM对于输出格式的理解和处理能力，以及对齐程度受做测量工具使用的LLM的对齐程度限制。本质上只能算二级度量，毕竟做测量工具使用的那个LLM肯定不会是用这个方法度量出来的。