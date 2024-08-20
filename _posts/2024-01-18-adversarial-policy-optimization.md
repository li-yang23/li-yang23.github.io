---
title: APO背景知识
description: notes for "Adversarial Policy Optimization"
tags: ai-alignment llm
category: AI
date: 2024-01-18

# bibliography: 2023-11-02-xai.bib
---
## RLHF相关方法回顾

人类偏好对齐的目的是在一个偏好数据集$$\mathcal{D}_P=\{(x,y^{good},y^{bad})\}$$上微调策略模型$$\pi_{\theta}(y\vert x)$$，以使其输出与人类偏好一致，从而提升人机交互质量。一般会用一个奖励模型$$r_{\phi}(x,y)$$来度量策略模型的生成质量，奖励模型的损失一般采用Bradley-Terry ranking loss

$$
\mathcal{L}_{Ranking}(r_{\phi};\mathcal{D}_P)=-\mathbb{E}_{x,y^{good},y^{bad}\sim\mathcal{D}_P}\left[\log\sigma(r_{\phi}(x,y^{good})-r_{\phi}(x,y^{bad}))\right]
$$

奖励模型用于描述某个潜在的人类偏好的分布，输出的评分可以反应这个人类偏好对数据集中好回答质量优于坏回答的认同程度（概率），可以描述为

$$
Q_{r_{\phi}}(y\prec y'\vert x)=\frac{\exp(r_{\phi}(x,y))}{\exp(r_{\phi}(x,y))+\exp(r_{\phi}(x,y'))}
$$

可以看到$$\mathcal{L}_{Ranking}(r_{\phi};\mathcal{D}_P)$$中的$$\sigma(\cdot)$$里面的东西就是$$Q_{r_{\phi}}(y\prec y'\vert x)$$，所以对于ranking loss的优化可以解释为对于$$Q_{r_{\phi}}$$的最大对数似然：$$\mathcal{L}_{Ranking}(r_{\phi;\mathcal{D}_P})=-\mathbb{E}_{\mathcal{D}_P}\left[\log Q_{r_{\phi}}(y^{good}\prec y^{bad}\vert x)\right]$$

有了一个训练好的奖励模型之后，对齐方法会以最大化生成回答的奖励期望为目标对策略模型进行训练，一般会加一个KL散度的正则化项来避免策略模型衰退到重复奖励值最高的回答，从而保护生成的多样性。

$$
\max_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}}\left[r_{\phi}(x,y)-\beta KL\left[\pi_{\theta}(y\vert x)\vert\vert\pi_{ref}(y\vert x)\right]\right]
$$

为了向策略模型准确传达奖励模型的反馈，RLHF阶段会用PPO和DPO，RRHF等方法进行模型更新。DPO找到了一个奖励模型和LLM的最优解之间的关系，然后用policy model跟reference model的相似性比例替代了奖励模型。

$$
\mathcal{L}_{DPO}(\pi_{\theta};\pi_{ref})=-\mathbb{E}_{(x,y^{good},y^{bad})\sim D}\left[\log\sigma(\beta\log\frac{\pi_{\theta}(y^{good}\vert x)}{\pi_{ref}(y^{good}\Vert x)}-\beta\log\frac{\pi_{\theta}(y^{bad}\vert x)}{\pi_{ref}(y^{bad}\vert x)})\right]
$$

RRHF使用了最优回答进行对比学习

$$
\mathcal{L}_{RRHF}=-\mathbb{E}_{(x,y^{good},y^{bad}\sim\mathcal{D})}\left[ReLU(\log\pi_{\theta}(y^{bad}\vert x)-\log\pi_{\theta}(y^{good}\vert x))-\lambda\log\pi_{\theta}(y^{best}\vert x)\right]
$$

拒绝采样方法更进一步，直接在最优回答上进行监督微调来简化对齐过程

$$
\mathcal{L}_{RJS}(\pi_{\theta})=-\mathbb{E}_{x\sim\mathcal{D},(y^1,y^2,...,y^S)\sim\pi_{\theta}(y\vert x)}\left[\log\pi_{\theta}(y^{best}\vert x)\right]
$$

其中最优解指这采样到的$$S$$个样本中的奖励分最高的回答。

## APO方法

APO方法用一个对抗方式的优化目标来进行对齐

$$
\min_{r_{\phi}}\max_{\phi_{\theta}}\mathcal{E}_{(x,y)\sim P_{\theta}(x,y)}\left[r_{\phi}(x,y)\right]-\mathcal{E}_{(x,y)\sim P_{gold}(x,y)}\left[r_{\phi}(x,y)\right] \\
s.t. KL\left[\pi_{\theta}(y\vert x)\vert\vert\pi_{ref}(y\vert x)\right]<\eta_1 \\
KL\left[P(y\prec y'\vert x)\vert\vert Q_{r_{\phi}}(y\prec y'\vert x)\right]<\eta_2
$$

在奖励模型优化和策略模型优化的时候分别使用一个正则项，通过对应的超参数$$\beta_1, \beta_2$$加到优化目标里面。

在奖励模型优化的时候，使用策略模型给最优回答的问题也生成一个回答，作为对最优回答的对比，这样将生成的回答数据集和最优回答数据集合并，作为APO数据集$$\mathcal{D}_{APO}$$。于是对于奖励模型的优化就成为了一个在APO数据集上的奖励模型优化过程。在正则项上，为了保证奖励模型优化的稳定性，在优化目标后面使用了一个单独的人类偏好数据集，并用这个数据集的ranking loss作为正则项。文章说是需要把超参数设大一些，因为人类偏好数据集相比最优回答数据集要小很多，而他们是要整批训练的，数据量过小会导致正则项作用变小。

策略模型优化的时候使用一个问题数据集，生成回答后最大化奖励模型对于问答的分数，正则项使用策略模型和参考模型的KL散度。

两个优化阶段都要求固定另一个模型的参数，且都是全batch训练，要把所有数据都跑过一轮之后才进入到另一个模型的优化阶段。