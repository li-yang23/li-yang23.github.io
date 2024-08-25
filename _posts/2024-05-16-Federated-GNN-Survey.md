---
title: 联邦图神经网络综述
description: notes for Federated Graph Neural Networks： Overview, Techniques and Challenges
tags: machine-unlearning federated-learning short-notes
category: AI-safety
date: 2024-05-27
---
## 术语和分类

联邦学习是一种协作机器学习范式，在不交换原始数据的情况下在多个数据拥有者之间训练模型。联邦学习中，进行协调的中心实体称为*服务器*，受服务器协调的敏感本地数据的数据拥有者称为*客户端*。联邦学习主要分为两种类型：**水平联邦学习**和**垂直联邦学习**。在水平联邦学习中，不同客户端上的数据集在特征空间有较大重合，但在样本空间上重合较少（特征基本一致，样本在不同客户端上）。根据通信架构，水平联邦学习分为中心式联邦学习去中心式联邦学习，中心式联邦学习下服务器协调客户端来共同训练模型，去中心式联邦学习下客户端在没有中心服务器的情况下各自与其他客户端通信来共同训练模型。和在垂直联邦学习中，客户端上的数据在特征空间上重合较少，但在样本空间上重合较多（统一批数据，特征在不同客户端上）。

联邦学习同样包括一个聚合操作，在服务器上根据客户端上传的本地模型参数来更新服务器上的模型参数，方法可以是取平均等等，看FedAvg的操作是平均每个客户端上每个样本的损失平均作为整体损失来训练服务器上的模型。

![联邦图神经网络的两层分类法，上面是整合方式，下面是数据异质性](/images/MyImages/FedGNN-taxonomy.png "联邦图神经网络的两层分类法，上面是整合方式，下面是数据异质性")

如上图所示，在本文中分布式图神经网络遵循二维分类法。首先根据FL和GNN的整合方式分为**受GNN支持的FL**和**受FL支持的GNN**。然后再根据FL对不同级别的图数据异构性的聚合操作进行细化。
1. 受GNN支持的FL：这一类方法主要关注结构化的客户端下的联邦学习训练，GNN方法用于帮助联邦学习的模型训练过程。GNN可以利用在客户端之间存在的图结构来提升已有联邦学习算法的表现。这类方法可以根据中心服务器是否存在细化为*中心化联邦GNN*和*去中心化联邦GNN*。通常假设中心服务器有一个客户端之间图结构的整体视野，服务器可以使用这个事业来训练一个图神经网络模型来改进联邦学习聚合效果或者帮助客户端根据内部训练的GNN模型来更新本地模型。没有中心服务器的情况下，客户端之间的图结构必须提前给定，从而让每个客户端保存一个子图来寻找它们的邻居客户端。
2. 受FL支持的GNN：这类方法主要关注有独立图数据的图神经网络训练，通常假设图数据分布式保存，并且客户端只能访问本地的图数据，所以使用联邦学习方法来训练一个全局GNN模型。这类方法可以分为*水平联邦GNN*和*垂直联邦GNN*。在水平联邦GNN中，客户端保存有节点大概率不重叠的图数据，联邦学习在通常情况下有效，但有可能存在客户端之间丢失边的情况。在垂直联邦GNN中，客户端共享同一个节点IF集合，但拥有不同的特征。

二级细化划分侧重于处理FL客户端之间的异质性。它们可以分为三类：**具有相同ID的节点的客户端**；**具有不同节点但相同网络结构的客户端**，以及**采用不同网络结构的客户端**。不同的中间信息应用于不同类别的FL聚合操作。

+ 对于有相同节点的客户端，将节点嵌入特征上传到服务器进行FL聚合。垂直联邦GNN和一些处理重叠节点的水平联邦GNN采用此方法。
+ 对于有不同节点但使用相同网络结构进行训练的客户端，使用模型权重和梯度进行FL聚合。受GNN支持的FL和一些没有重叠节点的水平联邦GNN采用此方法。
+ 对于训练具有不同网络结构的本地模型的客户端，可以首先将网络结构建模为图，然后在其上使用GNN模型，GNN模型权重或梯度可用于FL聚合。目前只有中心化的联邦GNN采用此方法。

<!-- 遗忘算法应该要研究哪个？ -->

## 受GNN支持的FL

某些场景下客户端之间具有联系，从而可以表示为图结构。因为客户端之间存在的图关系，GNN模型可以用于帮助FL的训练过程。在图中紧密相连的客户端之间倾向于共享相同的数据分布的假设下，GNN模型可以使用FL系统中的图结构来解决客户端之间的Non-IID（非独立同分布）问题。此外，在本地模型结构不同时，GNN还可以通过将神经网络建模为一个图来帮助FL系统。

### 中心化联邦GNN

中心化联邦GNN有一个中心服务器来根据客户端之间的图关系来协调客户端。客户端的本地数据不需要一定是图数据，根据图结构在何处保存，GNN的训练可以分为服务器侧和客户端侧。

#### 服务器侧GNN训练

服务器上根据客户端间的图关系训练一个GNN模型。它假设相邻客户端倾向于拥有相似的本地模型或特征嵌入。服务器首先从客户端收集模型参数，得到的模型参数会作为客户端节点的节点特征。然后服务器以此训练一个GNN模型来改进FL聚合操作。最后更新的模型参数会返回给客户端。客户端之间的图关系可以提前给出或者根据在训练中使用自注意力模块提取。因为服务器有一个独立的GNN模型，如何同时训练本地模型和GNN模型具有挑战性。

**双向模型训练**：相关工作(23,24) 设计了双向优化策略来训练本地模型和GNN模型，包含两种目标函数：客户端训练本地模型的本地任务目标函数 $$g_c(\cdot)$$和服务器训练GNN模型的目标函数 $$f(\cdot)$$：

$$
\begin{aligned}
    min_{\phi}\quad & f(\phi, w_c^*(\phi)\vert c\in1,...,C) \\
    s.t.\quad & w_c^*(\phi)\in argmin_{w_c}g_c(\phi,w_c)
\end{aligned}
$$

其中$$w_c^*$$是本地最优模型参数，$$\phi$$是GNN模型参数。Big-Fed[^23]和SFL[^24]使用了不同的GNN目标函数来满足相邻客户端的本地模型相似这一假设。Big-Fed提出了一种无监督对比学习损失函数，SFL提出了一种使用了图平滑正则项的有监督损失函数来同时训练本地和全局模型。

[^23]:[BiG-Fed: Bilevel optimization enhanced graph-aided federated learning](https://ieeexplore.ieee.org/document/9832778)
[^24]:[Personalized federated learning with graph](https://arxiv.org/abs/2203.00829)

**序列模型训练**：这部分工作不同时训练本地模型和GNN模型，而是通过独立的目标函数依序训练。Lee[^26]等人让客户端根据不同的本地任务训练本地模型，然后让服务器训练GNN模型来融合多任务本地估计。通过使用图正则化项最小化数据重建误差，本地估计可以根据客户端的相似性进行细化。PDGNet[^25]在服务器中使用GNN模型对功率分配策略进行建模，以找到最优的分配策略。目标函数时最小化所有FL客户端的传输错误概率。GNN模型采用原始对偶迭代方法进行训练。CNNFGNN[^27]和MLFGL[^22]仅使用本地目标函数训练本地模型和GNN模型，用固定的GNN模型权重更新本地模型权重，然后再用固定的本地模型权重更新GNN模型权重，交替优化多轮。

[^22]:[Multi-level federated graph learning and self-attention based personalized wi-fi indoor fingerprint localization](https://ieeexplore.ieee.org/document/9734052)
[^25]:[Power allocation for wireless federated learning using graph neural networks](https://arxiv.org/abs/2111.07480)
[^26]:[Privacy-preserving federated multi-task linear regression: A one-shot linear mixing ap- proach inspired by graph regularization](https://ieeexplore.ieee.org/document/9746007)
[^27]:[Cross-node federated graph neural network for spatio-temporal data modeling](https://arxiv.org/abs/2106.05223)

#### 客户端侧GNN训练

客户端侧训练GNN模型用于解决**数据异构性**和**模型异构性**两个问题，训练流程与通常联邦学习方式一样：客户端训练本地GNN模型，将权重参数上传至服务器，服务器进行聚合操作并将更新后的参数分发至客户端做下一轮训练。在客户端构建不同的图可以解决不同的问题。（应该是仍有一个本地模型一个GNN模型，客户端完成这俩模型的训练，服务器端完成这俩模型的更新）

**数据异构性问题**：这个设定下每个客户端都有客户端之间图结构的全局信息。客户端不止训练本地模型，还要根据全局图结构训练GNN模型来从其他客户端获取全局知识以解决数据分布异构问题。FedCG[^28]根据客户端之间的模型参数或模式特征的相似性构建了一个全连接图。一个客户端使用全连接图训练一个GNN来得到全局嵌入，然后将本地模型嵌入和全局嵌入组合为可训练权重。

**模型异构性问题**：客户端的本地模型异构时标准的联邦学习无法适配。HAFL-GHN[^29]使用GNN提出了一个方案：将客户端中的网络结构建模为图，其中节点表示每个参数层，边表示层之间的计算流。节点特征初始化为类别（one-hot）特征来表示层的类型。以最小化所有客户端的实际误差为目标训练一个基于GNN的图超网络来处理模型结构的图表示。GNN输出的隐层节点特征会映射回层参数一训练原来的网络。通过将神经网络转换为图并使用GHN模型训练，异构模型权重可以在联邦学习的客户端间通过上传本地GHN权重到服务器聚合来间接地聚合。

[^28]:[Cluster-driven graph federated learning over multiple domains](https://arxiv.org/abs/2104.14628)
[^29]:[Federated Learning with Heterogeneous Architectures using Graph HyperNetworks](https://arxiv.org/abs/2201.08459)

### 去中心化联邦GNN

关键在于没有服务器后如何进行联邦模型聚合。去中心化的联邦学习假设相邻客户端之间可以通信，已有工作一般使用两类方法解决以图结构关联的客户端节点之间的去中心化FL聚合问题：通过加权聚合邻域内更新的模型来更新FL模型权重，或者通过图正则化来更新FL模型权重。

**FL模型参数的加权聚合**：这种方法中客户端根据图结构与相邻客户端通信，并通过聚合相邻客户端的本地模型参数来更新自己的本地模型：

$$
w_c^{r+1}=\sum_{j\in\mathcal{N}(c)}a_{cj}\cdot[w_j^r]
$$

其中$$w_c^{r+1}\in\mathbb{R}^p$$表示客户端$$c$$在第$$r+1$$轮的本地模型参数，$$[\cdot]$$代表加密操作，用于数据隐私保护（比如Diffie-Hellman key exchange或者secret sharing）。$$a_{cj}$$表示图的邻接矩阵中的$$c$$行$$j$$列元素，用于表示客户端$$c$$和$$j$$之间的本地数据分布相似性。

以上方法根据不同的侧重有不同的变体，DSGT[^33]使用去中心化随机梯度追踪。为了处理大规模图数据，Rizk等人[^32]提出了一个多服务器联邦GNN结构来提升通信效率。其假设网络中有通过固定图结构关联的多个服务器，并且服务器网络中没有协调服务器的服务器。每个服务器下的客户端使用正常的中心化FL协议训练FL模型。当所有服务器都聚合了自己的客户端的模型更新后，服务器根据上式进行服务器之间的模型聚合。PSO-GFML[^34]通过与服务器只交换一部分本地模型参数的方式来提升通信效率。不同于提前知道邻接矩阵，图可以通过类似图注意力网络的方式在训练中习得。客户端之间图结构上的权重可以通过未标记的图嵌入[^36]或者对应客户端的隐层参数[^37]的相似性进行计算。

**FL模型参数的图正则**：


[^32]:[A graph federated architecture with privacy preserving learning](https://arxiv.org/abs/2104.13215)
[^33]:[Decentralized federated learning for electronic health records](https://ieeexplore.ieee.org/document/9086196)
[^34]:[Decentralized graph federated multitask learning for streaming data](https://ieeexplore.ieee.org/document/9751160)
[^36]:[SemiGraphFL: Semi-supervised Graph Federated Learning for Graph Classification](https://link.springer.com/chapter/10.1007/978-3-031-14714-2_33)
[^37]:[Fedstn: Graph representation driven federated learning for edge computing enabled urban traffic flow prediction](https://dl.acm.org/doi/10.1109/TITS.2022.3157056)

## 受FL支持的GNN

某些应用中图数据是孤立并存储在不同的客户端上的，关键问题就是如何在保护数据隐私的同时使用孤立的图数据训练GNN模型。另一个重要问题是遵循不同的数据分布的独立图数据可能导致的非独立同分布问题。联邦学习方法主要用于解决数据孤立问题，在保护数据隐私的前提下训练GNN模型，个性化FL可以用于缓解Non-IID问题。

#### 水平联邦GNN

水平FedGNN处理客户端共享节点特征空间但拥有不同的节点ID的场景。每个客户端至少有一个图或者图的集合。不同客户端存储的图节点之间可能存在连边，水平FedGNN由此分两个子场景，第一种假设这些边由客户端持有，或者这些边不存在，称为“无遗失边”场景。第二种假设这些边连接了不同客户端中的节点，但是遗失了，称为“有遗失边”场景。

**无遗失边场景**：通常策略是首先在客户端训练本地GNN模型来学习本地图表示或节点表示，然后使用FL算法来聚合本地模型参数或梯度，并将更新后的参数发回给客户端来进行下一轮训练。与GNN相关的研究主要关注图数据的non-IID问题，分布式时间-空间或大规模图数据嵌入，以及分布式神经结构搜索问题。

1. 图数据的非独立同分布(non-IID)问题：客户端中来自不同领域的图数据遵循不同的数据分布，如何使用non-IID分布的图数据来训练一个全局GNN模型变成了主要问题。主要方法是借鉴个性化FL方案，包括基于模型的方法和基于数据的方法。
   1. *基于模型的方法* 改进本地模型的适应表现或者学习一个有力的全局FL模型来为每个客户端进行未来的个性化调整，包括模型插入，正则化本地损失，元学习等等。ASFGNN[^44]和FedEgo[^45]在客户端使用模型插入方法，最终的模型是全局模型和本地模型的结合，更新过程中的本地模型占比由一个混合权重控制。FedAlign[^46]在损失函数中添加了一个本地和全局模型之间的基于最优转换距离的正则项来最小化FedProx[^47]中的模型散度。GraphFL[^48]使用了一个元学习策略来缓解non-IID问题，寻找一个可以在少量本地更新后快速适配到客户端的良好初始模型。
   2. *基于数据的方法* 旨在通过采样重新赋权，聚类和流式学习等方法来降低客户端数据分布的统计异构性。FLIT[^50]通过根据采样的预测可信度为样本重新赋权的方式解决non-IID问题，并为本地模型对于预测结果可信度低于全局模型的样本赋予更高的权重来保证各个客户端的本地训练更一致，以及避免过拟合本地数据。GCFL+[^51]通过根据每个客户端的GIN模型的梯度聚类客户端来解决non-IID问题。FMTGL[^53]通过使用在客户端之间共享的混合模块得到统一的任务表征来缓解non-IID问题。计算统一任务表征的关键在于在一个包含多个可学习支持向量的共同的混合平面上处理多源表征度量。
2. 分布式时间空间图数据嵌入：
3. 分布式大规模图数据嵌入：高计算成本阻碍了GNN在大规模图上的训练。为了降低GNN模型训练的计算代价，FedGraph[^57]设计了一种基于强化学习的服务器采样策略。在每一轮中，服务器为客户端细化采样策略(即待采样的节点数)和GNN模型参数。
4. 分布式神经结构搜索：
5. FL聚合改进：
6. 隐私保护：通过共享模型参数和图拓扑，FedGNN具有很大的攻击面。有些工作更加注重隐私保护。FeSoG[^65]使用动态本地差分隐私（LDP）将加密梯度上传到服务器以进行FL聚合。FedGraph[^57]中的客户端使用可信执行环境（TEE）加密本地训练，服务器使用安全多方计算（MPC）或同态加密（HE）加密全局模型聚合。FedEgo[^45]通过在客户端构建混合的自我图来保护图隐私。通过对中心节点固定大小的相邻节点进行采样来构建客户端中的特征图来保护全局结构。本地图嵌入通过在上传到服务器之前平均池化一批自我图（混合或捣碎）来保持匿名性。

**有遗失边场景**：这种场景假设连接不同客户端中的节点的一些边遗失。 遗失的边可以分为两种类型：具有不同节点ID的节点之间的边，以及不同客户端中对齐的节点之间的边。 对于第一类，常见的策略是通过重建缺失的边来修改本地图，因为完整的本地图可以确保高质量的图表示，并且客户端之间的边可以在一定程度上缓解非独立同分布（non-IID）问题。 对于第二种类型，知识图谱（KG）补全是一种重要应用场景。关键策略是在跨客户端的对齐节点之间转换信息，以帮助完成本地知识图谱嵌入。一旦本地图完成修正，FL算法就会被用于辅助GNN的训练，就像“无遗失边”场景的工作方式一样。

1. 本地图修正：
2. 知识图谱补全：

[^44]:[Asfgnn: Automated separated-federated graph neural network](https://arxiv.org/abs/2011.03248)
[^45]:[Fedego: Privacy preserving personalized federated graph learning with ego-graphs](https://arxiv.org/abs/2208.13685)
[^46]:[Improving federated relational data modeling via basis alignment and weight penalty](https://arxiv.org/abs/2011.11369)
[^47]:[Federated optimization in heterogeneous networks](https://arxiv.org/abs/1812.06127)
[^48]:[Graphfl: A federated learning framework for semi-supervised node classification on graphs](https://arxiv.org/abs/2012.04187)
[^50]:[Federated learning of molecular properties in a heterogeneous setting](https://arxiv.org/abs/2109.07258)
[^51]:[Federated graph classification over non-iid graphs](https://arxiv.org/abs/2106.13423)
[^53]:[Federated multi-task graph learning](https://dl.acm.org/doi/10.1145/3527622)
[^57]:[Fedgraph: Federated graph learning with intelligent sampling](https://arxiv.org/abs/2111.01370)
[^65]:[Federated social recommendation with graph neural network](https://arxiv.org/abs/2111.10778)

#### 垂直联邦GNN

垂直联邦GNN假设客户端保存了相同的节点，但是在不同的特征空间。客户端在FL的帮助下使用不同客户端的特征训练一个全局GNN模型。垂直联邦GNN有两种不同的子场景，第一种假设图结构，节点特征和节点标签归属于不同的客户端，集客户端没有完整的图数据。第二种假设只有节点特征空间归属于不同的客户端，图结构所有客户端都有。

**客户端没有完整的图数据**：在这个设定中，不同的客户端包含部分图数据。系统中有三个客户端，一个客户端拥有节点特征，一个客户端拥有图拓扑，一个客户端拥有节点标签。 或者当系统中只有两个客户端时，则一个获得节点特征，另一个拥有其他数据。 如何在保护客户端隐私的同时让客户端协同工作是关键挑战。SGNN[^82]不共享图的原始邻接矩阵，而是计算基于动态时间规整（DTW）算法的相似性矩阵来传达相同的图拓扑且隐藏原始结构。为了保护节点特征的隐私，采用one-hot编码将原始特征映射到矩阵。然后，不同客户端的信息上传到服务器，以训练用于节点分类任务的全局GNN模型。FedSGC[^83]假设只有两个客户端而没有中央服务器。图拓扑和节点特征由两个客户端拥有。拥有节点标签的客户端是创建加密密钥对的主动方。客户端使用加法同态加密（AHE）对敏感信息进行加密，然后将其发送给对方进行GNN模型参数更新。

**客户端有完整的图数据**：在此设置中，客户端包含完整的图数据，包括图结构和节点特征，但不同客户端的节点特征类型不同。拼接节点特征是关键策略。FedVGCN[^84]假设只有两个客户端和一个服务器。 对于每次迭代，两个客户端在同态加密下相互传输中间结果。服务器负责为客户端创建加密密钥对并为模型进行FL聚合。VFGNN[^85]中的客户端首先使用差分隐私（DP）对节点嵌入进行加密，然后通过均值、串联或回归计算将节点特征集成到半诚实服务器中进行FL聚合。训练完成后，拥有节点标签的客户端从服务器接收更新的节点嵌入以执行节点预测。FMLST[^86]假设存在一个有包含相同节点的所有客户端共享的全局模式图，使用多层感知器(MLP)并以串联模式作为输入，融合局部时空(ST)模式和全局ST模式。客户端通过评估全局模式图和本地模式图之间的差异，利用全局模式来个性化自己的本地模式图。Graph-Fraudster[^87]研究了对本地原始数据和节点嵌入的对抗性攻击，证明差分隐私（DP）机制和top-k机制是针对该攻击的两种可能的防御方法。

[^82]:[Sgnn: A graph neural network based federated learning approach by hiding structure](https://ieeexplore.ieee.org/document/9005983)
[^83]:[Fedsgc: Federated simple graph convolution for node classification](https://federated-learning.org/fl-ijcai-2021/FTL-IJCAI21_paper_5.pdf)
[^84]:[A vertical federated learning framework for graph convolutional network](https://arxiv.org/abs/2106.11593)
[^85]:[Vertically federated graph neural network for privacy-preserving node classification](https://www.ijcai.org/proceedings/2022/0272.pdf)
[^86]:[Federated meta-learning for spatial-temporal prediction](https://link.springer.com/article/10.1007/s00521-021-06861-3)
[^87]:[Graphfraudster: Adversarial attacks on graph neural network-based vertical federated learning](https://arxiv.org/abs/2110.06468)

## 相关文献