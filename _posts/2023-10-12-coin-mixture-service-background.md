---
layout: post
title: Backgrounds of Coin Mixing Service
description: some backgounds of coin mixing service
tags: coin-mixing-service unfinished
categories: bitcoin
giscus_comments: true
date: 2023-10-12
featured: true
toc:
    begining: true

authors:
  - name: Yang Li
    <!-- url: "https://en.wikipedia.org/wiki/Albert_Einstein" -->
    affiliations:
      name: CS, Tsinghua
---

# 混币服务

## 比特币基本知识

比特币协议用地址(address)描述交易参与方，每笔交易包括输入方和输出方，输入方地址的比特币按照指定比例发给输出方地址。

因此在比特币交易视角看来是多个地址之间在进行交易。一个地址对应一个用户，但一个用户可以拥有多个地址。比特币管理软件可以让用户创建和操作任意数量的地址。因为用户控制地址进行交易，地址又不包括用户信息无法准确逆映射，因此在比特币交易中不会保留参与者的身份信息。这种用户与地址间一对多的关系是比特币匿名性的基础，也被称为伪匿名性。

> 但好像匿名性的基础是无法通过地址反向追踪到用户吧，无法逆映射好像才更重要一点。

## 混币服务
