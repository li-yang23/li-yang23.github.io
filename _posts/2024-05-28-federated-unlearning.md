---
title: 联邦遗忘综述
description: notes for Federated Unlearning：A Survey on Methods, Design Guidelines, and Evaluation Metrics
tags: machine-unlearning federated-learning short-notes
category: AI-safety
date: 2024-05-27
---
## 联邦遗忘

在联盟中使用客户端$$u$$训练的全局模型可能会泄露有关客户端$$u$$私有数据的信息，使$$u$$至少容易受到成员资格攻击。

假设在经过$$t$$轮的联邦学习后，一个给定的客户端$$u$$提交了一个针对其隐私数据$$D_u$$的一个子集$$S_u$$的遗忘请求，需要执行一个遗忘算法$$\mathcal{U}$$来满足这个请求。遗忘算法$$\mathcal{U}$$应用于（使用包含客户端$$u$$在内的客户池中设备训练的）全局模型$$w_t$$上，生成一个遗忘后的全局模型$$w^{\overline{u}}_t$$，即 $$w^{\overline{u}}_t=\mathcal{U}(\mathcal{w_t})$$ 。$$\mathcal{U(\cdot)}$$可以被定义为一个确保$$w^{\overline{u}}$$ 表现出和没有$$S_u$$贡献的全局模型 $$w_t^{\overline{u}}$$ 性能无法区分或近似无法区分的函数。

> 所以遗忘请求应该发给服务器，然后让服务器根据遗忘任务来安排遗忘算法？

### 对象（遗忘任务）

**样本遗忘**：样本遗忘旨在从训练模型中删除特定数据样本的贡献。请求遗忘的客户端需要删除其数据中一个子集的贡献

**类遗忘**：类遗忘旨在删除客户端之间属于某个类$$c$$的所有数据样本的贡献，即遗忘$$C=\bigcup_{k\in K}S_k$$，其中$$S_k=\{x_i,c\}_i^{n_k^c}$$，$$n_k^c$$是客户端$$k$$中类别为$$c$$的本地样本的数量。遗忘后的模型如果接收属于已删除类的样本，则应在其余类之间随机产生结果。

**客户端遗忘**：客户端遗忘是联邦学习场景下的特殊任务，并且与客户端的遗忘权相关。客户端遗忘旨在删除特定客户端的整个本地数据集的贡献，即$$S_u=D_u$$。

### 主要挑战

主要包括协作学习的迭代性质、每轮参与客户的随机选择以及本地数据的分散性和不可预测性。

**迭代学习过程**：联邦学习算法通过迭代轮次进行，在轮次$$t$$聚合的全局模型$$w_t$$是轮次$$t+1$$的起始状态。假设在轮次$$t_i$$时，一个已被包含在轮次$$t_u(t_u<t_i)$$中的客户端$$u$$需要删除其贡献（例如，客户端遗忘）。对于客户端遗忘的简单解决方案是恢复轮次$$t_u$$的全局模型，并从轮次$$t_u$$的聚合中排除客户端$$u$$的更新。然而，联邦学习算法的迭代性质意味着在轮次$$t_u$$之后的所有全局聚合都应该被无效化（就是恢复后需要从$$t_u$$轮开始重新训练，原来之后训练的轮次结果都要作废）。

**非静态的训练过程**：参与联邦学习轮次的客户端选择受到各种动态因素的影响。例如，有资格参与某一轮次的客户端可能会因为连接问题、电源问题或不愿意参与而在未来变得不可用。联邦学习的这一特性使得在$$t_u$$之后完全重现轮次变得不现实。同时，抛弃在轮次$$t_u$$的聚合和轮次$$t_i$$（即最新轮次）的聚合之间获取的所有知识（可能时间相隔较远，$$t_u<<t_i$$），会浪费联邦学习的效率。从这个意义上说，最好是设计能嵌入这些知识的方法。

**模型更新和最终存储的不可预测性**：从原则上讲，联邦学习的聚合操作或服务器不应该或不能检查单个客户端的模型更新（例如，因为更新可能通过多方计算进行聚合）。这使得回滚到轮次$$t_u$$的全局聚合的简单解决方案变得不可行，因为它需要访问单个客户端的更新。在另一方面，轮次回滚需要联邦学习的中心聚合操作存储每一轮聚合模型的整个历史记录以及收集的更新，这将导致无法承受的存储容量成本。

**本地数据的不可预测性**：在集中式机器学习中，所有用于训练的数据都应该是直接可访问的，并且可被用于模型更新。相反，在联邦学习环境中，数据集保持私密性是设计之初的要求。这阻止了使用需要直接访问待遗忘/保留数据的中心化的机器遗忘方法。

**数据异构性**：在集中式机器学习中，数据被假定为符合独立同分布，因为数据集中在同一位置，并且在分布式训练中可以在机器之间进行打乱和任意划分。相反，联邦学习设置的一个基本特征是客户端之间存在数据的异质性，即客户端持有的数据可能是非IID的。虽然这可能会阻碍全局模型的收敛，但是联邦学习设置的这一特征对于遗忘算法可能是方便的，因为可能更容易删除针对数据的倾斜子集学习到的表征。然而，由于数据不是直接可访问的，因此可能无法事先评估数据的异质性水平。

### 执行主体

执行主体可以是服务器，目标客户端或者其他客户端。服务器有更强的算力和存储能力，可以保存历史全局模型并且追踪客户端的更新。但服务器需要能够接触某几个特定的客户端的特定历史更新来进行遗忘，这可能会导致其他的隐私问题。目标客户端可以直接接触待遗忘的数据，可以在本地执行遗忘过程。但考虑到恶意客户端的情况，本地的遗忘可能带来额外的安全问题，比如通过发起本地遗忘请求来想模型植入后门，或者故意损害训练过程。其他客户端有需要保留的数据，通常用于在遗忘时恢复遗忘模型的性能。

## 已有方法

### 客户端遗忘

**历史更新的重新校正**

FedEraser[^54]是最早提出的客户端遗忘方法之一，其通过重新训练来实现，利用存储在服务器端的历史参数更新加速恢复阶段。假设当前轮次为$$t_i$$（请求遗忘时），且客户端$$u$$参与了轮次$$t_u$$。FedEraser仅在校准阶段包括保留的客户端，通过迭代地对在客户端$$u$$参与后产生的更新进行清理，而不是将其丢弃。FedEraser执行$$t_i−t_u$$个校准轮次，每轮包括校准更新的本地训练和服务器端对这些更新的聚合，以生成清理后的全局模型，以在下一次重新校准轮次进行广播，直到所有过去的更新都被恢复为止。Yuan等人引入了FRU[^57]，这是一种基于FedEraser设计的用于联邦推荐系统的遗忘算法。为了最小化所需的存储空间，FRU建议使用用户-物品混合半硬负采样（semi-hard negative sampling）组件来减少物品嵌入更新的数量，并使用基于重要性的更新选择组件来仅保留关键的模型更新。

曹等人提出了FedRecover[^58]，这是一种利用历史信息恢复被毒化模型的方法。FedRecover利用来自客户端的保留更新来估计重新从头开始训练期间产生的更新，利用柯西均值定理和L-BFGS算法[^78]对整合的Hessian矩阵进行高效近似。为了增强模型的性能，它实施了各种策略来微调恢复的模型。这些策略包括热身、定期校正、异常修复和最终调优。在热身阶段，客户端计算精确的更新并将其发送到服务器，以创建由L-BFGS算法用于构建Hessian矩阵的缓冲区。这些缓冲区在定期校正阶段定期更新，服务器请求客户端发送精确的更新。这些更新也用于校正恢复的全局模型。当客户端的估计模型更新量较大时，服务器会要求客户端发送精确的更新，以减轻潜在的估计不准确的大型模型更新的影响，构成异常修复阶段。最后，在最终调优阶段，客户端发送精确的更新来微调恢复的全局模型，这一步被证明能够提升模型的性能。

高等人提出了一个统一的框架VeriFi[^27]来执行遗忘和验证。这是第一个授予用户验证权利（RTV）的工作，在这个框架下，参与者可以主动检查遗忘效果。VeriFi包括三个部分：一个遗忘模块，一个验证模块，以及一个将遗忘模块和验证模块链接成一个集成的管道的遗忘验证机制。虽然遗忘模块可以是任何遗忘方法，但作者还提出了一个更高效的一步式遗忘方法——scale-to-unlearn（S2U）。只有当参与者在联邦学习过程的后期阶段离开时，即模型已经稳定下来时，才会触发遗忘过程。S2U提升了留在联邦中的客户端的贡献，并降低了离开客户端的贡献。这种调整旨在使全局模型与留下的客户端的本地模型更加接近，同时与离开的客户端拉开距离。验证过程由离开的客户端在本地执行。它包括两个步骤：标记和检查。标记步骤涉及利用标记数据（称为标记）来微调本地模型。随后，检查步骤验证全局模型已经遗忘了多少标记。

郭等人提出了FAST[^59]，这是一个旨在消除拜占庭参与者对全局模型的影响的协议。服务器保留所有客户端的更新，利用这些信息直接调整模型参数并得出遗忘模型。具体而言，遗忘模型是通过从最终全局模型中减去每一轮训练中恶意客户端的贡献而获得的。对于每次迭代，服务器比较当前遗忘模型与先前模型的准确性。如果当前的遗忘模型始终优于其前任，表明恶意客户端的历史影响仍然存在，促使继续在下一轮中删除贡献的步骤。然而，他们的方法最终可能导致对不需要遗忘的其他数据的预测能力下降。为了解决这一问题，服务器维护一个额外的小型基准数据集，用于补充训练遗忘模型，从而提高整体准确性。

**知识蒸馏**

吴等人提出了一种基于知识蒸馏的联邦遗忘机制[^60]，利用知识蒸馏来快速恢复在移除遗忘客户端的历史模型更新后的全局模型性能。遗忘算法在服务器端进行两个步骤：（1）从当前全局模型中删除已遗忘客户端的历史更新，以生成一个经过消毒的模型；（2）先前的全局模型作为教师，经过消毒的模型作为学生。消毒的全局模型在代理未标记数据上模仿先前全局模型的输出，以恢复步骤（1）导致的性能下降。客户端不必直接参与遗忘阶段。但需要服务器端的（未标记的）数据，这对于特定任务可能不容易获得。此外，服务器应保留从客户端收集的更新的完整历史记录，并且服务器应能够识别客户端的更新，然后响应遗忘请求。

赵等人提出了一种名为动量降解（MoDe）[^56]的知识消除策略。所提出的方法将联邦遗忘过程与训练解耦，使其适用于任何已完成联邦学习训练的模型架构，而不改变训练过程。遗忘包括两个阶段：知识消除和记忆引导。在第一阶段，预训练的模型参数被调整以减少对目标数据的可辨识性，使其与重新训练的模型参数对齐。服务器初始化一个降解模型，将其发送到客户端进行联邦学习训练，并使用其权重作为更新预训练模型的指导方向。在动量降解之后，预训练模型对所有数据点的可辨识性都降低了。记忆引导阶段通过将预训练模型发送到所有客户端和将降解模型发送到目标客户端来恢复性能。目标客户端使用降解模型在其本地数据集上的输出作为伪标签，引导预训练模型的本地训练。这确保了遗忘模型的输出与重新训练模型的输出密切相似。所有客户端都有助于聚合预训练模型。知识消除和记忆引导步骤会多次迭代。

叶等人提出了一个名为HDUS[^61]的分布式遗忘框架，采用蒸馏（较小）模型为每个客户端构建可擦除的集成。HDUS在完全分布式的学习场景中运行，客户端之间交换信息而无需中央服务器。除了其本地主要数据集和模型之外，每个客户端还拥有一个蒸馏模型和在所有客户端之间共享的参考数据集。客户端的目标是使用特定的损失函数和参考数据集优化蒸馏模型，尽可能地将其蒸馏模型与主模型对齐。蒸馏模型与相邻的客户端进行交换。在推断阶段，客户端通过加权其主模型的输出和所有邻居蒸馏模型的输出来生成集成模型输出。如果客户端$$u$$请求遗忘数据，那么其邻居将简单地从集成中排除客户端$$u$$的蒸馏模型。

**梯度修改**

Halimi等人介绍了一种在目标客户端上直接执行遗忘的方法[^50]。该方法采用投影梯度上升来最大化参考模型的损失，参考模型定义为剩余客户端模型的平均值。一个半径为$$\delta$$的$$\ell_2$$范数球体围绕着参考模型，限制了无界损失。为了改善遗忘模型的性能，剩余客户端进行额外的联邦学习轮次，在几轮内获得良好的性能。同样，Li等人提出了一种基于子空间的联邦遗忘方法（SFU）[^62]，利用梯度上升，在其他客户端形成的输入梯度空间的正交空间约束下，消除目标客户端的贡献。SFU不需要在客户端或服务器上存储中间更新。该过程包括三个步骤：（1）除目标客户端外的每个客户端使用其本地数据的一部分创建一个表示矩阵，并将其发送到服务器。神经网络中每一层的表示矩阵由通过网络的随机样本的正向传递输出组合而成，每列表示不同样本的输出。通过DP算法保护表示矩阵的隐私。为了防止个体客户端数据信息的泄漏，对每个客户端的层表示添加随机因子。由于矩阵的正交性质，这种添加不会影响子空间搜索过程。(2) 目标客户端执行一些轮次的梯度上升，并将更新后的梯度发送到服务器。(3) 然后，服务器对表示矩阵集合进行奇异值分解（SVD），以获得输入梯度子空间集合。它得到目标梯度在此子空间中的正交投影，并通过更新全局模型来消除目标客户端的贡献。
****
Alam等人[^63]尝试通过联邦遗忘消除对敌人有利的后门。一旦攻击者实现了他们的预期目标或怀疑可能被检测到，他们可能希望消除他们先前注入的后门，并抹去他们存在的任何痕迹。所提出的方法利用目标客户端上的梯度上升来忘记其贡献。然而，为了应对像灾难性遗忘、在遗忘过程中生成偏离模型以及个别参数的变化重要性等挑战，作者提出了两种策略：记忆保留和动态惩罚。第一种策略利用目标客户端的良性数据集来防止灾难性遗忘。第二种策略实施了一个动态惩罚机制，旨在惩罚与全局模型的偏离。此外，权重也被纳入惩罚项中，根据其重要性为每个参数分配唯一的权重。对所有参数引入非均匀惩罚可能会产生更好的结果。损失函数根据所描述的策略修改。

**聚类方法**

KNOT[^51]在异步联邦学习中进行客户端聚类以加速重训练。当要求进行遗忘时，只在客户端$$u$$的集群内进行重训，而其余客户端不受影响。

**贝叶斯联邦学习**

贝叶斯联邦学习将贝叶斯方法与联邦学习相结合，引入了模型分布以捕捉不确定性，而不是一个固定的模型。龚等人[^55]在贝叶斯框架内首次介绍了分布式网络中的第一个联邦遗忘方法。它基于指数族参数化，并利用本地流言协议驱动的通信。当客户端请求遗忘时，它会从当前全局模型中删除其本地变分参数，然后将结果转发给下一个客户端。龚等人还提出了一种非参数的联邦贝叶斯遗忘方法，称为Forget-Stein变分梯度下降（Forget-SVGD）[^64]。Forget-SVGD是SVG的扩展，SVG是一种基于粒子的近似贝叶斯推断方法，利用基于梯度的确定性更新，其联邦变体称为分布式SVG（DSVGD）。在联邦学习过程结束后，一个或多个客户端可能会请求忘记他们的数据。从变分后验中遗忘数据集被形式化为最小化遗忘自由能。这涉及确定一个与当前变分后验密切对齐的分布，同时最大化指定要遗忘的数据集的平均训练损失。在[^65]中提出了一种针对速率受限信道的优化版本。它跨多个粒子应用量化和稀疏化。

BFU[^66]是另一种包含参数自共享的贝叶斯联邦遗忘方法。所提出的方案引入了一个遗忘率，平衡了遗忘擦除的数据和记住原始全局模型之间的权衡。为了限制潜在的性能下降，数据擦除和保持学习精度被认为是两个不同的目标。

**差分隐私**

张等人提出了FedRecovery[^67]，这是一种在他们的遗忘机制中嵌入差分隐私（DP）的算法。事实上，当客户端$$u$$请求遗忘时，首先服务器通过消除其所有梯度残差来消除其影响，这些残差是由客户端的历史提交生成的。然后，服务器向结果模型注入高斯噪声，使得这种遗忘模型在统计上无法与重新训练的模型区分开来。噪声根据两个模型之间距离的上界进行校准，该距离是通过客户端$$u$$损失函数的平滑度估计得出的。通过利用差分隐私，观察者无法确定模型是由$$K$$个客户端还是$$K-1$$个客户端训练得到的。

**其他方法**

陶等人[^70]提出了一个利用总变差（TV）稳定性概念的精确FU框架，该概念衡量两个分布之间的差异。如果使用整个数据集产生的模型与使用其子集产生的模型之间的差异最多为ρ，则学习算法是ρ-TV稳定的。作者介绍了FATS，这是一种基于FedAvg的快速重新训练的新型算法，并被证明是ρ-TV稳定的。RevFRF[^68]是一个用于联邦随机森林的框架，支持安全的参与者撤销。作者从两个独特的角度探讨了撤销过程：（1）确保从联邦随机森林中删除目标数据，（2）预防被删除的客户端与服务器串通以非法地继续使用过时的联邦随机森林。为了实现第一级的撤销，RevFRF遍历了训练的联邦随机森林中的随机决策树，并删除了由寻求被遗忘的参与者贡献的所有分裂。随后，联邦随机森林经历重建，只考虑剩余的随机决策树s。由于它们彼此隔离，一个随机决策树的重建不会影响其他随机决策树。然而，需要注意的是，这种方法不容易扩展到其他ML模型，并且最坏情况下与从头开始重建模型一致。第二级的撤销涉及执行额外的计算来使用随机值刷新已撤销的分裂，使它们对任何潜在串通的服务器都不可访问。

潘等人介绍了一种新颖的联邦遗忘算法[^69]，用于执行联邦K-means++。简而言之，客户端维护在其本地数据上计算的质心向量。然后，这些向量被传输到服务器，服务器使用它们来获取一个质心集。如果一个客户端希望其数据被遗忘，它会将其质心向量重置为零，促使服务器在剩余客户端的向量上重新执行聚类。

### 类遗忘

在CNN模型中，通过可视化不同通道激活的特征图，可以揭示它们对图像分类中不同图像类别的不同贡献。王等人[^53]利用这一特性，通过有选择地修剪对目标类别具有高区分度的通道来引入联邦遗忘。类别区分度是使用TF-IDF算法确定的。每个通道的输出可以看作是一个词，表示不同类别的特征图可以看作是文档，而TF-IDF评估了通道和类别之间的相关性。一旦区分性通道被修剪，就会应用微调操作来恢复模型的性能。Zaho等人提出的MoDe方法[^56]也可以作为一种用于类别移除的解决方案。

### 样本遗忘

因为样本遗忘可以看作是一种精确化的客户端遗忘，这些方法内在保持一样的原则，因此使用类似的划分方法。

**梯度修改**

刘等人[^52]提出了一种快速重训策略，利用低成本的Hessian近似方法，并在遗忘数据删除后将其应用于客户端的本地训练。具体来说，机制的核心是使用对角经验费舍尔信息矩阵（FIM），遗忘模型的更新规则利用了一阶和二阶矩来计算海森动量，类似于Adam优化器。然而，当处理复杂模型时，这种方法可能会导致性能显著降低。

Dhasade等人提出了QuickDrop[^71]，一种利用数据集精炼（DD）执行遗忘和恢复的联邦遗忘方法。DD是一种将大型训练数据集压缩成紧凑且较小的合成数据集的技术[^84]。然而，生成高质量的合成数据需要许多本地更新步骤。客户端重复利用在FL训练期间计算的梯度更新，以减少计算开销。当收到遗忘请求时，每个客户端在其本地精炼数据集上执行随机梯度上升（SGA）。而在恢复阶段，网络执行恢复轮次，期间还使用了未从样本中提取出来的精炼数据进行遗忘。值得注意的是，使用精炼数据会略微降低性能。为了避免性能下降，在恢复阶段，作者在精炼数据集中包含了一些原始样本，以抵消潜在的性能下降。

FedFilter[^72]是一种边缘缓存方案，利用联邦遗忘从全局模型中移除无效和过时的数据。遗忘由服务器发起，服务器选择要遗忘的内容并生成一个反向梯度。反向梯度应用于本地模型，以在本地擦除选定的内容。此外，为了最大化遗忘效果的同时最大程度地减少对模型准确性的影响，参数通过SGD进行迭代。最后，对于基本层参数，联邦遗忘通过聚合和广播模型参数实现。

金等人提出了可遗忘的联邦线性学习（2F2L）框架[^73]。2F2L引入了一种利用深度神经网络（DNN）的一阶泰勒展开式的线性近似的联邦线性学习策略。用于执行线性近似的权重的第一个初始值是使用服务器的数据集构建的。有了服务器上合理数量的数据，这个值被认为接近基于整个数据集的最优模型权重。通过线性近似，可以使用二次遗忘损失函数进行梯度上升来执行数据移除。然而，为了实现移除，需要计算Hessian矩阵的逆，这是一项复杂的任务。因此，2F2L利用基于服务器数据的数值逼近。

Forsaken[^41]是一种遗忘算法，利用虚拟梯度来刺激机器学习模型的神经元，擦除特定数据的知识。客户端$$u$$拥有一个可训练的生成器，用于产生梯度。在每个遗忘周期中，生成器基于一个目标函数进行优化，该目标函数旨在减小当前置信向量与完全遗忘样本规则的置信向量之间的距离。然后，客户端产生一个新的虚拟梯度并将其发送到服务器。

夏等人提出了Fed$$ME^2$$[^74]，这是一个针对移动网络数字孪生的联邦遗忘框架。Fed$$ME^2$$包括两个模块：MEval和MErase。第一个模块旨在构建一个内存评估模型，以检测待遗忘数据是否被记住以及记忆程度。MErase模块利用模型损失、MEval损失和一个惩罚项优化本地模型，该惩罚项限制了数据擦除前后全局模型的相似性。然后，服务器从客户端重新聚合本地模型，并完成不含隐私信息的全局模型的构建。

**量化**

熊等人提出了Exact-Fun[^75]，这是一种量化的联邦遗忘算法，旨在从全局模型中消除目标客户端的数据子集的影响。在某些情况下，当客户端请求删除其数据的特定部分时，该算法会使用客户端训练模型直至该迭代来计算一个本地模型。如果生成的量化模型与先前的全局模型匹配，则删除该数据子集不会对整体模型产生影响。然而，如果量化导致不匹配，则需要重新训练，以有效消除未学习数据的影响。

车等人将他们之前关于Prompt Certified Machine Unlearning (PCMU) [^85]的工作扩展到在联邦环境中有效运行[^76]。PCMU利用随机梯度平滑和量化同时执行训练和遗忘操作，提高了整体效率。使用梯度的随机平滑意味着机器遗忘模型直接在整个数据集上进行训练，与仅在剩余数据上重新训练的模型共享相同的梯度（和参数）。这发生在特定的认证半径内，涉及数据移除前后梯度变化，并且有一个数据移除的认证预算。为了在联邦环境中实现PCMU，作者提出了一种方法，该方法涉及使用PCMU算法在客户端上创建机器遗忘模型。然后，这些模型被重新构造为Nemytskii算子的输出函数，诱导出一个Frechet可微的平滑流形。该流形具有全局Lipschitz常数，该常数界定了两个本地机器遗忘模型之间的差异。在服务器端，通过对所有客户端的梯度进行平均，计算出全局梯度，从而创建全局机器遗忘模型。全局Lipschitz性质确保了该模型与每个客户端上的本地机器遗忘模型在由Lipschitz常数确定的距离内紧密对齐。因此，全局机器遗忘模型可以在一定程度上保持本地机器遗忘模型的认证半径和数据移除的预算。

**强化学习**

Shaik等人提出了FRAMU[^43]，这是一个基于注意力的利用联邦强化学习的机器遗忘框架。FRAMU可以在单模态和多模态场景下执行过时、无关和私密数据的遗忘。它采用联邦强化学习架构，利用本地代理进行实时数据收集和模型更新，利用中央服务器和FedAvg算法进行聚合和全局模型更新，并使用注意力层动态权衡学习和遗忘中每个数据点的相关性。注意力层充当专门的逼近器，通过为数据点分配注意力分数来增强各个代理的学习能力，指示其在本地学习中的相关性，并通过与环境和反馈的交互不断完善模型。如果这些分数低于特定预设的阈值，数据将被视为过时或无关，因此从模型中删除。

**其他方法**

FATS算法[^70]也可以用于样本遗忘，方法是从包含特定样本之前的迭代开始重新训练。

联邦学习也可以采用替代数据表示，例如知识图谱（KG），即用三元组的形式表示的现实世界实体及其关系的结构化知识库。知识图谱也被采用来推进表示学习技术，即KG嵌入将实体及其关系组合成统一的语义空间。FL可以与KG嵌入集成，从而产生一种新的范式，即联邦知识图谱嵌入学习。在这方面，朱等人提出了FedLU[^77]，这是一个用于异构KG嵌入的联邦学习框架。FedLU使用一种遗忘方法来从本地嵌入中擦除特定的知识。一方面，客户端的本地知识通过基于硬性和软性混淆的干扰阶段来擦除，这反映了遗忘集中三元组分数与其负样本（与本地KG的交集为空的三元组集）之间的距离。另一方面，为了限制干扰造成的模型性能下降，使用不包含遗忘集的本地客户端KG通过知识蒸馏来恢复。

潘等人[^69]介绍的框架在当客户端希望删除其部分数据时也适用。如果要删除的数据不是中心点，则无需采取任何操作。但如果它是一个中心点，客户端需要消除该中心点，并使用特定分布从其本地数据中抽样选择一个新的中心点。然后，更新后的中心点向量发送到服务器，服务器重新对新向量集执行聚类，生成一组新的服务器中心点。

## 参考文献

[^27]: X. Gao, X. Ma, J. Wang, Y. Sun, B. Li, S. Ji, P. Cheng, and J. Chen, “Verifi: Towards verifiable federated unlearning,” arXiv preprint arXiv:2205.12709, 2022. 
[^41]: Y. Liu, Z. Ma, X. Liu, and J. Ma, “Learn to forget: Userlevel memorization elimination in federated learning,” arXiv preprint arXiv:2003.10933, vol. 1, 2020. 
[^43]: T. Shaik, X. Tao, L. Li, H. Xie, T. Cai, X. Zhu, and Q. Li, “FRAMU: Attention-based Machine Unlearning using Federated Reinforcement Learning,” arXiv preprint arXiv:2309.10283, 2023. 
[^50]: A. Halimi, S. Kadhe, A. Rawat, and N. Baracaldo, “Federated unlearning: How to efficiently erase a client in fl?” arXiv preprint arXiv:2207.05521, 2022. 
[^51]: N. Su and B. Li, “Asynchronous federated unlearning,” in IEEE INFOCOM 2023 - IEEE Conference on Computer Communications, 2023, pp. 1–10. 
[^52]: Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, “The right to be forgotten in federated learning: An efficient realization with rapid retraining,” in IEEE INFOCOM 2022-IEEE Conference on Computer Communications. IEEE, 2022, pp. 1749–1758. 
[^53]: J. Wang, S. Guo, X. Xie, and H. Qi, “Federated Unlearning via ClassDiscriminative Pruning,” in Proceedings of the ACM Web Conference 2022, ser. WWW ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 622–632. 
[^54]: G. Liu, X. Ma, Y. Yang, C. Wang, and J. Liu, “Federaser: Enabling efficient client-level data removal from federated learning models,” in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS), 2021, pp. 1–10. 
[^55]: J. Gong, O. Simeone, and J. Kang, “Bayesian variational federated learning and unlearning in decentralized networks,” in 2021 IEEE 22nd International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2021, pp. 216–220. 
[^56]: Y. Zhao, P. Wang, H. Qi, J. Huang, Z. Wei, and Q. Zhang, “Federated unlearning with momentum degradation,” IEEE Internet of Things Journal, pp. 1–1, 2023. 
[^57]: W. Yuan, H. Yin, F. Wu, S. Zhang, T. He, and H. Wang, “Federated unlearning for on-device recommendation,” in Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, 2023, pp. 393–401. 
[^58]: X. Cao, J. Jia, Z. Zhang, and N. Z. Gong, “Fedrecover: Recovering from poisoning attacks in federated learning using historical informa-tion,” in 2023 IEEE Symposium on Security and Privacy (SP), 2023, pp. 1366–1383. 
[^59]: X. Guo, P. Wang, S. Qiu, W. Song, Q. Zhang, X. Wei, and D. Zhou, “FAST: Adopting Federated Unlearning to Eliminating Malicious Terminals at Server Side,” IEEE Transactions on Network Science and Engineering, pp. 1–14, 2023. 
[^60]: C. Wu, S. Zhu, and P. Mitra, “Federated unlearning with knowledge distillation,” arXiv preprint arXiv:2201.09441, 2022. 
[^61]: G. Ye, Q. V. H. Nguyen, and H. Yin, “Heterogeneous Decentralized Machine Unlearning with Seed Model Distillation,” arXiv preprint arXiv:2308.13269, 2023. 
[^62]: G. Li, L. Shen, Y. Sun, Y. Hu, H. Hu, and D. Tao, “Subspace based federated unlearning,” arXiv preprint arXiv:2302.12448, 2023. 
[^63]: M. Alam, H. Lamri, and M. Maniatakos, “Get rid of your trail: Remotely erasing backdoors in federated learning,” arXiv preprint arXiv:2304.10638, 2023. 
[^64]: J. Gong, J. Kang, O. Simeone, and R. Kassab, “Forget-svgd: Particlebased bayesian federated unlearning,” in 2022 IEEE Data Science and Learning Workshop (DSLW), 2022, pp. 1–6. 
[^65]: J. Gong, O. Simeone, and J. Kang, “Compressed particle-based federated bayesian learning and unlearning,” IEEE Communications Letters, vol. 27, no. 2, pp. 556–560, 2023. 
[^66]: W. Wang, Z. Tian, C. Zhang, A. Liu, and S. Yu, “BFU: Bayesian Federated Unlearning with Parameter Self-Sharing,” in Proceedings of the 2023 ACM Asia Conference on Computer and Communications Security, ser. ASIA CCS ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 567–578. 
[^67]: L. Zhang, T. Zhu, H. Zhang, P. Xiong, and W. Zhou, “FedRecovery: Differentially Private Machine Unlearning for Federated Learning Frameworks,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 4732–4746, 2023. 
[^68]: Y. Liu, Z. Ma, Y. Yang, X. Liu, J. Ma, and K. Ren, “RevFRF: Enabling Cross-Domain Random Forest Training With Revocable Federated Learning,” IEEE Transactions on Dependable and Secure Computing, vol. 19, no. 6, pp. 3671–3685, 2022. 
[^69]: C. Pan, J. Sima, S. Prakash, V. Rana, and O. Milenkovic, “Machine unlearning of federated clusters,” arXiv preprint arXiv:2210.16424, 2022. 
[^70]: Y. Tao, C.-L. Wang, M. Pan, D. Yu, X. Cheng, and D. Wang, “Communication efficient and provable federated unlearning,” arXiv preprint arXiv:2401.11018, 2024. 
[^71]: A. Dhasade, Y. Ding, S. Guo, A.-m. Kermarrec, M. De Vos, and L. Wu, “QuickDrop: Efficient Federated Unlearning by Integrated Dataset Distillation,” arXiv preprint arXiv:2311.15603, 2023. 
[^72]: P. Wang, Z. Yan, M. S. Obaidat, Z. Yuan, L. Yang, J. Zhang, Z. Wei, and Q. Zhang, “Edge Caching with Federated Unlearning for Lowlatency V2X Communications,” IEEE Communications Magazine, pp. 1–7, 2023. 
[^73]: R. Jin, M. Chen, Q. Zhang, and X. Li, “Forgettable Federated Linear Learning with Certified Data Removal,” arXiv preprint arXiv:2306.02216, 2023. 
[^74]: H. Xia, S. Xu, J. Pei, R. Zhang, Z. Yu, W. Zou, L. Wang, and C. Liu, “FedME2: Memory Evaluation & Erase Promoting Federated Unlearning in DTMN,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 11, pp. 3573–3588, 2023. 
[^75]: Z. Xiong, W. Li, Y. Li, and Z. Cai, “Exact-Fun: An Exact and Efficient Federated Unlearning Approach.” 
[^76]: T. Che, Y. Zhou, Z. Zhang, L. Lyu, J. Liu, D. Yan, D. Dou, and J. Huan, “Fast federated machine unlearning with nonlinear functional theory,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 4241–4268. 
[^77]: X. Zhu, G. Li, and W. Hu, “Heterogeneous Federated Knowledge Graph Embedding Learning and Unlearning,” in Proceedings of the ACM Web Conference 2023, ser. WWW ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 2444–2454. 
[^78]: J. Nocedal, “Updating quasi-newton matrices with limited storage,” Mathematics of Computation, vol. 35, no. 151, pp. 773–782, 1980. 
[^84]: J. Geng, Z. Chen, Y. Wang, H. Woisetschlaeger, S. Schimmler, R. Mayer, Z. Zhao, and C. Rong, “A Survey on Dataset Distillation: Approaches, Applications and Future Directions,” arXiv preprint arXiv:2305.01975, 2023. 
[^85]: Z. Zhang, Y. Zhou, X. Zhao, T. Che, and L. Lyu, “Prompt certified machine unlearning with randomized gradient smoothing and quantization,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 13 433–13 455.