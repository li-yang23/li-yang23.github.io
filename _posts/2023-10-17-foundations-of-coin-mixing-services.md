---
title: Foundations of Coin Mixing Services
description: notes for the paper
tags: coin-mixing-service unfinished
category: bitcoin
date: 2023-10-17
---

# 混币服务的基础

## 内容概括

主要是Tairi等人在S&P2021年对混币服务做了形式化定义并且提出了一种高效且互操作性强的新混币加密协议$$A^2L$$。本文发现了上文正式模型的问题并用两个实例加以证实，展示了如何构造两个满足上文定义但导致完全不安全系统的加密策略。

为了修正问题，本文探索了混币服务的安全构造，提出了盲条件加密（BCS）的概念，作为混币服务的核心密码原理。然后提出了盲条件加密的基于博弈的安全性定义，并设计了一个满足定义的$$A^2L$$协议的改进版本，$$A^2L^+$$。我们在理想模型（类似代数群模型）中进行了分析，假设问题与one-more离散对数问题难度相当。最后，本文设计了一个满足更强的UC-security概念，但计算成本显著增加的BCS实现。这可能说明构造在组合下安全的混淆服务协议需要比预想中更复杂的加密机制。

- 比特币（及核心原则相同的其他密码币）基于假名的交易无关联性已被学术研究驳倒，且被区块链监视机构验证。因此有大量工作开始通过混币协议的方式向比特币提供隐私保护，保持交易的不可关联性。
- 已有混币协议面临引导问题，即如何选择执行协议的参与人集合。大规模的参与人集合要求更高的隐私保证，且可延展性很差
- 第三方混币服务应运而生，第三方收集密码币，混合后再分发给用户，从而提供无关联性。
- 第三方混币机构会跑路（exit scam），促使了加密协议的研究，目的是脱离对于第三方服务的信任。Heilman等人为核心加密原语奠定了基础，称为同步难题（Synchronization Puzzles）协议，使第三方角度的无关联性成为现实。但提出的协议实例TumbleBit依赖于哈希时间锁合约（HTLCs），难以兼容其他密码币系统。
- Tairi等人在通用组合框架下提出了同步难题的形式化定义，并提出了一个实例$$A^2L$$，仅需要时间锁和数字签名验证就能达到比TumbleBit更高的效率和更强的兼容性。

### 背景知识

定义$$n\in\mathcal{N}$$为安全参数，$$x\leftarrow\mathcal{A}(in;r)$$为算法$$A$$根据输入$$in$$和随机变量$$r\leftarrow\$\{0,1\}^*$$下的输出。

- 数字签名：数字签名策略$$\Pi_{DS}:=(KGen, Sign, Vf)$$包含一个验签钥生成方法$$KGen$$，一个签名方法$$Sign$$和一个验证方法$$Vf$$，KGen生成验证签名密钥对$$(vk,sk)\leftarrow KGen(1^n)$$，用签名密钥$$sk$$生成信息$$m$$的签名$$\sigma\leftarrow Sign(sk,m)$$，然后用验证密钥$$vk$$进行验证$$Vf(vk,m,\sigma)$$。
- 严格关系：使用证明/目击对$$(Y,y)$$来描述关系$$R$$，关联语言$$\mathcal{L}_R$$定义为$$\mathcal{L}_R:=\{Y\verb\exist y, (Y,y)\in R\}$$。如果（1）存在一个概率线性时间采样算法$$GenR(1^n)$$可以输出证明/目击对$$(Y,y)\in R$$；（2）关系时线性时间可定的；（3）所有概率线性时间对手关系$$\mathcal{A}$$在输入$$Y$$的情况下输出$$y$$的概率是可忽略不计的，那么称关系是严格关系。本文使用离散对数语言$$\mathcal{L}_{DL}:=\{Y\verb\exist y\in Z_p, Y=g^y\}$$来定义严格关系
- 适配器签名（adaptor signature）：为信息预生成一个预签名，随后可以根据某些隐私知识适配为一个有效签名，包含$$\Pi_{ADP}:=(KGen, PreSig, PreVf, Adapt, Vf, Ext)$$，$$PreSig(sk, m, Y)$$为信息生成一个预签名$$\widetilde{\sigma}$$，$$PreVf(vk, m, Y, \widetilde{\sigma})$$验证预签名被成功生成，$$Adapt(\widetilde{\sigma},y)$$将预签名根据严格关系$$\mathcal{L}_R$$下实例$$Y$$的目击$$y$$转换为正式签名$$\sigma$$，$$Ext(\widetilde{\sigma}, \sigma, Y)$$根据预签名，正式签名和实例，输出目击$$y$$。
- 仅线性同态加密：如果存在某些有效的计算操作$$\circ$$满足$$Enc(ek, m_0)\circ Enc(ek, m_1)=Enc(ek, m_0+m_1)$$，则称加密操作是仅线性同态加密的
- 非接触零知识（Non-Interactive Zero-Knowledge）：大概理解就是可以通过证明（statement）的信息验证目击（witness）的有效性。具体得再看，现在看不懂密码学的东西。
- one-more离散对数：看不懂，得再补

### 反例

因为$$\Pi_{ADP}.Ext(\widetilde{\sigma},\sigma,h)=\widetilde{x}$$，后面的没看懂，感觉在描述的时候换了字符。

第一种攻击，通过查询hub向alice提供的预言机（oracle）$$n$$次，可以完全发现hub的解密密钥$$dk$$。

## 优缺点分析

## 感悟
