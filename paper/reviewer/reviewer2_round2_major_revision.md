# Reviewer 2 - Round 2

Recommendation: Prepare A Major Revision

Comments:

The authors' revision effort is commendable, but the result has placed the paper in a self-contradictory position: the STID comparison demonstrates that naive retraining of a lightweight model achieves comparable performance at similar cost, thereby weakening the necessity of the proposed continual learning framework. The problem setting has value, but the foundation of the core contribution is currently not solid.

Frankly speaking: based on the authors' attitude toward revision, a major revision could be justified; based on the rigor of the contribution and the empirical evidence, the current manuscript is closer to a rejection standard.

Additional Questions:

Summary of Evaluation: Fair

Organization: 3

Clarity: 3

Length: 3

References: 2

Correctness: 2

Significance: 2

Originality: 2

Attachments: 2

If Survey Coverage: 2

Contribution: 2

Please make very detailed technical and editorial comments and suggestions in your comments. If it is necessary to provide mathematical corrections, please email them to us as a pdf file. If you must get other information back to us that cannot be sent via email, please mail it to us. Your comments are an invaluable aid to the author to help in improving the overall technical quality, utility, and readability of the material. Such comments are not just useful, they are necessary to maintain the quality of the articles that are published in the SMC Transactions. Particular attention should be given to details that guide possible revisions, or that clearly explain reasons for rejection.

What are the contributions of the paper?:

The paper proposes CoMemNet, a continual learning framework for traffic prediction under evolving sensor networks. It combines an Online/Target dual-branch architecture, a Wasserstein-based drift sampler, and a temporal memory buffer (TMRB-N). The authors also construct two new datasets and benchmark against STID, EAC, and other baselines.

What are the additional ways in which the paper could be improved?:

The authors have responded diligently, adding substantial experiments, clarifying the protocol, and correcting the Wasserstein formulation. However, the revision exposes a more fundamental issue: the newly added STID comparison undermines, rather than supports, the paper's central claim. Details follow.

The paper's core selling point is that continual learning is more efficient than naive retraining. Yet the chosen baseline, STID, is an extremely lightweight pure-MLP model (the original paper reports 5.24 seconds per epoch on PEMS04). In Table VIII of the response, current-period retraining takes 190.89s across 7 periods, versus CoMemNet's 155.37s - a mere 35-second difference (~18%). Meanwhile, Table IX shows that retrained STID achieves comparable accuracy to CoMemNet (MAE difference < 0.8; on PEMSD3(S), STID even achieves better RMSE: 22.651 vs. 23.207).

This creates an awkward situation: a four-year-old pure-MLP model with no continual learning mechanism, trained from scratch each period at nearly identical computational cost, delivers essentially equivalent performance. Where, then, is the justification for CoMemNet's dual-branch design, EMA updates, Wasserstein sampling, and memory buffer? By choosing a baseline whose retraining is already extremely fast, the authors have inadvertently demonstrated that continual learning may be unnecessary in this setting.

Furthermore, the 3256s all-history retraining cost in Table VIII is driven by the heavy STGNN backbone used for that particular baseline. If all-history retraining also adopted STID, the total cost would drop dramatically, further eroding any efficiency advantage of CoMemNet.

In my first-round comment (R2.2), I requested comparison with a "strong recent static retrained model." STID was published at CIKM 2022, four years ago. The spatial-temporal forecasting literature has advanced considerably since then, with numerous stronger static models now available, including PatchTST (ICLR 2023), DLinear (AAAI 2023), iTransformer (ICLR 2024), and TimeMixer (ICLR 2024).

PEMSD3(S), PEMSD4(L), and PEMSD8(M) are all self-constructed datasets. The STID results reported on these datasets cannot be cross-validated against the original STID paper. What criteria were used for dataset construction? Results on at least one standard benchmark (e.g., PEMS04 or PEMS08) under the same protocol should be provided for community verification.

If you are suggesting additional references they must be entered in the text box provided. All suggestions must include full bibliographic information plus a DOI.


---

# 审稿人 2 - 第二轮中文翻译

**建议：大修（Prepare A Major Revision）**

## 总体意见

作者的修订工作值得肯定，但修订后的结果使论文陷入了一个自相矛盾的处境：与 STID 的比较表明，对一个轻量模型进行朴素的重新训练，便可以以相近的成本获得可比的性能，从而削弱了所提出持续学习框架的必要性。该问题设定本身具有价值，但核心贡献的基础目前并不牢固。

坦率地说：基于作者对修订工作的认真态度，给予大修是可以成立的；但基于该贡献的严谨性和现有实验证据，当前稿件更接近应当拒稿的标准。

## 附加评价

- 总体评价：一般（Fair）
- 组织结构：3
- 清晰度：3
- 篇幅：3
- 参考文献：2
- 正确性：2
- 重要性：2
- 原创性：2
- 附件：2
- 若为综述论文的覆盖范围：2
- 贡献：2

请在评审意见中给出非常详细的技术性和编辑性意见与建议。如有必要提供数学上的修正，请将其以 PDF 文件通过电子邮件发送给我们。如有无法通过电子邮件发送的其他材料，请邮寄给我们。此类意见对于作者提高文章的总体技术质量、实用性和可读性具有不可替代的价值，也是维持 SMC Transactions 论文发表质量所必需的。应特别关注能够指导可能修订的细节，或清楚说明拒稿理由的细节。

## 论文的贡献是什么？

本文提出 CoMemNet：一个面向演化传感器网络交通预测的持续学习框架。该框架结合了 Online/Target 双分支架构、基于 Wasserstein 距离的漂移采样器，以及时间记忆缓冲区（TMRB-N）。作者还构建了两个新数据集，并与 STID、EAC 和其他基线进行了比较。

## 论文还可以如何改进？

作者已经认真回应，补充了大量实验，澄清了实验协议，并修正了 Wasserstein 公式。然而，修订引出了一个更根本的问题：新增的 STID 比较并非支持，而是削弱了论文的核心主张。具体如下。

论文的核心卖点是，持续学习比朴素的重新训练更高效。然而，所选择的基线 STID 是一个极其轻量的纯 MLP 模型（原论文报告其在 PEMS04 上每个 epoch 仅需 5.24 秒）。在回复信的表 VIII 中，当前周期重新训练在 7 个周期上总共需要 190.89 秒，而 CoMemNet 需要 155.37 秒，两者仅相差约 35 秒（约 18%）。与此同时，表 IX 表明重新训练的 STID 达到了与 CoMemNet 可比的精度（MAE 差异小于 0.8；在 PEMSD3(S) 上，STID 的 RMSE 甚至优于 CoMemNet：22.651 对 23.207）。

这造成了一个尴尬的局面：一个四年前的纯 MLP 模型，没有任何持续学习机制，在几乎相同的计算成本下实现了本质上等价的性能。那么，CoMemNet 的双分支设计、EMA 更新、Wasserstein 采样和记忆缓冲区的合理性在哪里？作者选择了一个重新训练本身已经非常快速的基线，反而无意中表明在这一问题设定下持续学习可能并非必要。

此外，表 VIII 中 3256 秒的全历史重新训练成本，是由该特定基线所使用的重型 STGNN backbone 驱动的。如果全历史重新训练也采用 STID，其总成本将显著下降，从而进一步削弱持续学习的效率优势。

在我的第一轮意见（R2.2）中，我要求与“强的、近期的静态重新训练模型”比较。STID 发表在 CIKM 2022，距今已有四年。时空预测文献在这期间已有显著进展，现在有许多更强的静态模型，包括 PatchTST（ICLR 2023）、DLinear（AAAI 2023）、iTransformer（ICLR 2024）和 TimeMixer（ICLR 2024）。

PEMSD3(S)、PEMSD4(L) 和 PEMSD8(M) 都是作者自行构建的数据集。本文在这些数据集上报告的 STID 结果无法与 STID 原论文进行交叉验证。数据集构建使用了什么标准？应当至少在一个标准基准数据集（例如 PEMS04 或 PEMS08）上，以相同协议报告结果，以便社区验证。

如果建议补充参考文献，必须将其填入指定文本框。所有建议均须包含完整的书目信息和 DOI。
