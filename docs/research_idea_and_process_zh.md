# 面向语言模型的边界稳定语音 Token：研究思路与实验过程

> 项目基线：LLM-Codec / AUV 语音 codec  
> 当前核心方法：Quantization-Boundary Consistency（QBC）  
> 文档状态：截至 2026-09-20 的研究记录  
> 结论边界：当前结果足以证明机制有效，但还不足以支持大规模、多语言泛化结论

## 1. 一句话概括

现有方法通常用连续 latent 的距离衡量语音 token 是否稳定，但真正决定 VQ token 的是 latent 位于哪个 codebook Voronoi cell。我们的核心观点是：

> **离散 token 的不稳定首先是量化决策边界问题，而不只是连续表征距离问题。**

因此，本项目不再只要求两个语音 view 的 latent 靠近，而是直接在部署时真实使用的 VQ 空间中扩大 signed decision margin，使同一语音片段在上下文变化后仍落在相同的量化 cell 中；随后再观察这种稳定性是否能降低语言模型的建模难度。

整个研究闭环是：

```text
现象：同一波形片段在 full / slice 上得到不同 token
  ↓
诊断：token flip 由 nuisance displacement D 和局部边界半径 R 共同决定
  ↓
机制：Gamma = D / R 比单独的 latent displacement 更能解释 flip
  ↓
干预：QBC 直接扩大真实 VQ 决策边界的 signed margin
  ↓
结果：flip 稳定下降，但存在小幅且可重复的重建代价
  ↓
下游：seed-23 上 token LM 的 PPL 明显下降，但仍需多种子和跨语言验证
```

## 2. 为什么要研究这个问题

### 2.1 VQ token 在 SpeechLM 中的作用

语音 tokenizer 把连续波形压缩成离散 token：

```text
waveform → codec encoder → continuous latent → vector quantizer → token IDs
```

这些 token 同时承担两类任务：

1. 保存足够的声学信息，使 decoder 能重建语音；
2. 形成适合语言模型学习的离散序列，使 LM 能预测、生成和理解语音。

传统 neural codec 主要优化重建质量和码率，但“适合重建”不等于“适合语言建模”。如果相同的语音内容仅仅因为切片位置、上下文或轻微无关扰动就产生不同 token，语言模型会看到多个近似等价但 ID 不同的答案，从而增加条件熵和预测难度。

### 2.2 最初方案为什么不够新

最初的想法是对 full utterance、slice 和轻微扰动后的语音施加 consistency loss，使它们得到更稳定的 token。但相关工作已经覆盖了相当一部分叙事：

- ACL 2025 DRI 工作已研究 full-vs-slice、phase perturbation 和 latent consistency；
- StableToken 已指出量化边界附近的轻微扰动会改变 token；
- LLM-Codec 已显式优化 token 的 LM predictability；
- ReLMCodec 已从 pre-quantization phoneme structure 角度设计更可预测的 token。

因此，不能把“发现 token 不稳定”或“稳定 token 有利于 LM”作为主要创新。

### 2.3 最终收敛出的研究空缺

已有 latent consistency 通常优化：

$$
\lVert z^{\mathrm{full}}_t-z^{\mathrm{slice}}_t\rVert_2^2.
$$

但实际 token 由下式决定：

$$
k^*(u)=\arg\min_k \lVert u-q_k\rVert_2^2.
$$

两个 latent 即使非常接近，只要跨过 Voronoi 边界，token 就会改变；反过来，两个 latent 即使相距较远，只要仍在同一个 cell 中，token 就不变。因此：

$$
\text{latent consistency}\neq\text{discrete decision consistency}.
$$

本项目真正要验证的是：

> **真实 VQ 决策几何是否提供了 latent displacement 之外的信息，以及直接优化该几何能否稳定 token 并改善 LM predictability。**

## 3. 核心数学定义

### 3.1 使用真实量化空间

AUV 并不是直接在 encoder 原始 hidden 上量化，而是经过：

```text
encoder hidden → in_proj → L2 normalization → nearest-code lookup
```

因此所有诊断和 QBC 训练都必须在真实量化空间中进行：

$$
u_t=\frac{\operatorname{in\_proj}(z_t)}
{\lVert\operatorname{in\_proj}(z_t)\rVert_2},
\qquad
q_k=\frac{e_k}{\lVert e_k\rVert_2}.
$$

定义平方欧氏距离：

$$
d_k(u)=\lVert u-q_k\rVert_2^2.
$$

### 3.2 Signed boundary margin

设 full view 的 teacher token 为 $k$，相对于竞争 codeword $j$ 的 signed boundary margin 为：

$$
s_j(u;k)=
\frac{d_j(u)-d_k(u)}{2\lVert q_j-q_k\rVert_2}.
$$

- $s_j>0$：仍在 teacher cell 一侧；
- $s_j=0$：位于 $k/j$ 决策边界；
- $s_j<0$：已经越过边界。

teacher cell 的最近边界半径为：

$$
R(u;k)=\min_{j\ne k}s_j(u;k).
$$

诊断阶段扫描全部 20,480 个 deployed codeword，避免把“距离最近的 codeword”错误地当成“边界最近的 competitor”。

### 3.3 位移、证书与边界压力

对于同一波形片段的 full 和 slice view，定义真实量化空间中的位移：

$$
D_t=\lVert u_t^{\mathrm{full}}-u_t^{\mathrm{slice}}\rVert_2.
$$

如果位移小于 full point 到最近边界的合法 chord radius，则 token 一定不会改变：

$$
D_t<R_t\Longrightarrow
c_t^{\mathrm{full}}=c_t^{\mathrm{slice}}.
$$

这给出了局部 VQ stability certificate。进一步定义：

$$
\Gamma_t=\frac{D_t}{R_t}.
$$

它衡量 nuisance displacement 相对于当前可用鲁棒半径的大小：

- $\Gamma<1$：理论上保证不 flip；
- $\Gamma\approx1$：接近危险边界；
- $\Gamma\gg1$：位移远大于局部安全半径，flip 风险高。

单位球面上还可区分 ambient Euclidean boundary score、angular radius 和 chord radius。当前实现使用与实际 normalized VQ 一致的精确球面/chord 几何进行 certificate 判断，而不是在 encoder 原始 hidden 上近似。

## 4. 三个核心研究问题

本项目最终被压缩为三个可检验问题：

### Q1：边界几何是否提供额外解释力？

> 给定相同的 latent displacement，离 VQ boundary 更近的 frame 是否更容易发生 token flip？

### Q2：这些 flip 是否真的增加 LM uncertainty？

> 在完全相同的 LM prefix 下，用 slice token 替换 full token 是否产生额外 NLL？这种代价是否集中在高 $D/R$ 区域？

### Q3：直接干预真实决策边界是否有效？

> 扩大真实 VQ 空间中的 signed margin，是否比只约束 latent 距离更直接地降低 flip，并且不会通过 codebook collapse 伪造更低 PPL？

## 5. 方法：Quantization-Boundary Consistency

### 5.1 训练 view

每个训练样本构造两个严格对齐的 view：

- teacher：包含完整上下文的 full utterance；
- student：从同一波形中截取的 4 秒 slice。

所有 crop start 都满足：

$$
\text{crop start}=320n,
$$

从而与 AUV 的 320-sample hop 对齐，排除 frame-grid 错位造成的伪 DRI。

### 5.2 QBC loss

full view 产生 stop-gradient teacher code $k_t^c$。对 slice view 计算它相对于 teacher cell 的最危险 signed margin：

$$
R_t^{\mathrm{view}}
=\min_{j\ne k_t^c}s_j(u_t^{\mathrm{view}};k_t^c).
$$

QBC 使用 hinge 目标：

$$
L_{\mathrm{QBC}}
=\frac{1}{T}\sum_t
w_t[m-R_t^{\mathrm{view}}]_+.
$$

其中 $w_t$ 由 teacher confidence 决定，低置信度 teacher 不应被强行当作唯一正确答案。

当前实现的重要约束是：

1. QBC 在真实 `in_proj → L2 normalize → VQ` 空间计算；
2. 训练时扫描完整 codebook；
3. 只优化 crop 内部 150 帧，减少边界 receptive-field 影响；
4. QBC 路径对 codebook 使用 stop-gradient；
5. 原始 VQ/codebook loss 仍正常更新 codebook；
6. decoder 在主实验中冻结，避免把重建改善或恶化混入 encoder stability 结论。

总损失为：

$$
L=L_{\mathrm{codec}}
+\lambda_{\mathrm{mel}}L_{\mathrm{mel}}
+\lambda_{\mathrm{VQ}}L_{\mathrm{VQ}}
+\lambda_{\mathrm{QBC}}L_{\mathrm{QBC}}.
$$

主实验采用：

```text
lambda_mel = 5
lambda_vq = 1
lambda_qbc = 0  （matched control）
lambda_qbc = 10 （QBC）
```

### 5.3 为什么对 QBC 路径冻结 codebook

如果 QBC 同时更新 encoder 和 codebook，模型可能通过移动 codeword 或整体改变 cell 边界来降低 loss，而不是让相同内容的 representation 更稳定；极端情况下还可能造成 codebook collapse。因此主实验只让 QBC 更新 encoder representation，codebook 仍由原本的 VQ 目标学习。

### 5.4 Quantizer-native LM bridge 的状态

研究中还提出了让 LM gradient 直接沿真实 quantizer geometry 回传的方案：

$$
p(k\mid u,E_{\mathrm{VQ}})
=\operatorname{softmax}\left(-\frac{\lVert u-q_k\rVert^2}{\tau}\right),
$$

$$
y_{\mathrm{ST}}
=p+\operatorname{sg}[\operatorname{onehot}(k^*)-p].
$$

forward 使用真实 hard VQ token，backward 使用真实 codebook geometry 的 soft gradient。这与 learned proxy bridge 的区别是：LM gradient 与部署时决定 token 的边界完全一致。

但是，公开的 `llm-codec.pt` 只包含导出的 AUV codec state dict，没有论文训练阶段的 learned Gumbel bridge 参数。因此：

> **Quantizer-native bridge 目前仍是下一阶段方案，不属于已经完成的实验证据。**

## 6. 实验过程

### 阶段 A：先诊断，不直接训练

第一步不是增加 loss，而是验证问题是否真的由 decision geometry 解释。

在 LibriSpeech `test-clean` 和 `test-other` 上构造 full-vs-slice 对，执行：

1. hop-aligned 4 秒 crop；
2. forward hook 提取真实 8-D `in_proj` 输出；
3. L2 normalization；
4. 扫描完整 20,480 codebook；
5. 验证官方 token ID 与独立重算 ID 完全一致；
6. 计算 $D$、$R$、$\Gamma=D/R$ 和 token flip；
7. 验证 $D<R$ 的 frame 是否保持零 certificate violation；
8. 比较 `D only` 与 `D + R` 对 flip 的 group-CV AUROC。

初始诊断使用每个 split 32 个 speaker、每个 speaker 一个 utterance，共 6,400 paired frames，其中 4,800 帧属于 crop interior。

### 阶段 B：隔离 LM prefix confound

如果直接比较 full token sequence 和 slice token sequence 的 NLL，两者 prefix 可能早已不同，无法判断当前 NLL 变化来自当前 token 还是历史 token。

因此定义 common-context 指标：

$$
\Delta\mathrm{NLL}_t=
-\log p(c_t^{\mathrm{slice}}\mid c_{<t}^{\mathrm{full}})
+\log p(c_t^{\mathrm{full}}\mid c_{<t}^{\mathrm{full}}).
$$

两项使用完全相同的 full/clean prefix，只替换当前位置的目标 token。

公开 LM adapter 与当前 codec token 版本不匹配，因此没有强行使用旧 adapter；项目重新从当前 `llm-codec.pt` 提取 `train-clean-100` token，并训练版本匹配的小型 causal LM。

### 阶段 C：从强干预验证机制

先使用较强 QBC 验证 loss 是否真的能改变目标几何。2,000-step、`lambda_qbc=5` 的实验使：

- test-clean flip 相对 matched control 降低 6.11 pp；
- test-other flip 相对 matched control 降低 4.67 pp；
- signed margin、median $\Gamma$ 和 certified coverage 同步改善。

但其重建代价明显，因此该设置只用于证明“直接干预边界确实能控制 token stability”，不是最终配方。

### 阶段 D：寻找稳定性与重建质量的 Pareto 点

随后完成 `lambda_qbc=1/2/5` sweep，并提高重建权重。最终选择：

```text
max_steps = 1000 optimizer updates
grad_accum_steps = 8
lambda_mel = 5
lambda_vq = 1
lambda_qbc = 10
```

每个 optimizer update 使用 8 个独立 crop，因此每个 arm 实际看到 8,000 个有效 4 秒 crop。这里的 `max_steps=1000` 指 optimizer updates，而不是 micro-examples。

### 阶段 E：严格 matched control 与多种子复现

为了只测 QBC 的增量效果，每个 seed 的 control 和 QBC 保持以下条件完全一致：

- 初始化 checkpoint；
- 随机种子；
- crop 顺序与 crop RNG；
- optimizer、学习率和 update 数；
- reconstruction/VQ loss；
- 有效样本和跳过样本顺序。

唯一差异是 `lambda_qbc=0` 或 `10`。

训练种子为 23、41、71。主评估对每个 split 使用：

- 128 个 utterance；
- 32 个 speaker；
- 每个 speaker 最多 4 个 utterance；
- 25,600 个 aligned frames；
- 19,200 个 interior frames。

统计推断采用 paired speaker-cluster bootstrap，而不是把同一 speaker/utterance 中高度相关的 frame 当成独立样本。

### 阶段 F：重建、codebook 和 LM 检查

为了排除“token 更稳定只是因为信息塌缩”，同时检查：

- log-Mel reconstruction；
- MR-STFT reconstruction；
- codebook utilization；
- unigram entropy 和 effective vocabulary；
- fixed-LM full/slice NLL；
- 从头训练同架构小 LM 的 CE/PPL；
- context modeling gain。

## 7. 已完成的主要结果

### 7.1 几何诊断支持 Q1

四个 codec/split 诊断中，`D + R` 对 token flip 的 group-CV AUROC 均高于 `D only`：

| Split | Codec | D only AUROC | D + R AUROC |
| --- | --- | ---: | ---: |
| test-clean | AUV | 0.7849 | 0.8570 |
| test-clean | LLM-Codec | 0.7974 | 0.8498 |
| test-other | AUV | 0.7914 | 0.8541 |
| test-other | LLM-Codec | 0.7787 | 0.8510 |

所有 certified frames 都是零 flip，未发现 certificate violation。AUV test-clean 中，$\Gamma<1$ 时 flip rate 为 0，而 $\Gamma\ge8$ 时达到 77.5%。

另一个重要发现是：LLM-Codec 虽然减小了 full-vs-slice latent displacement，但同时也减小了局部 boundary radius，因此没有显著降低 token flip。这说明只优化 latent displacement 不能保证离散决策更稳定。

### 7.2 Common-context 结果支持 Q2 的主要部分

在版本匹配的固定 LM 下，flip frames 的平均 common-context penalty 为：

$$
+0.221\ \text{nats/token}
=+0.319\ \text{bits/token},
$$

utterance-cluster bootstrap 95% CI 为 `[+0.114, +0.339]` nats。最明显的 LM 代价集中在 $\Gamma\ge8$ 区域。

但不能把这称为“因果链路”：它仍属于控制了 prefix 的 mechanistic/observational evidence。真正的干预证据来自后续 QBC 训练。

### 7.3 多种子 QBC 稳定性结果支持 Q3

| Seed | test-clean flip 变化 | Speaker 95% CI | test-other flip 变化 | Speaker 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 23 | -2.25 pp | [-3.08, -1.40] | -2.52 pp | [-3.70, -1.41] |
| 41 | -2.77 pp | [-3.71, -1.88] | -1.91 pp | [-2.98, -0.82] |
| 71 | -1.75 pp | [-2.53, -0.98] | -1.57 pp | [-2.58, -0.62] |
| Mean ± seed SD | **-2.26 ± 0.51 pp** | 3/3 改善 | **-2.00 ± 0.48 pp** | 3/3 改善 |

六个 seed×split 的 speaker-bootstrap CI 均不跨零。interior-frame 平均改善为：

- test-clean：`-2.17 ± 0.64 pp`；
- test-other：`-1.64 ± 0.34 pp`。

这说明 QBC 对 full-vs-slice token stability 的改善具有多种子可重复性。

### 7.4 改善不是免费的

相对于各自 matched control，QBC 的重建代价为：

| Split | log-Mel 相对变化 | MR-STFT 相对变化 |
| --- | ---: | ---: |
| test-clean | +1.11 ± 0.03% | +0.57 ± 0.11% |
| test-other | +1.01 ± 0.13% | +0.69 ± 0.11% |

因此准确表述应是：

> QBC 获得了小幅但稳定、可重复的 token stability 改善，同时付出约 0.5%–1.1% 的谱重建代价。

不能把它描述为无代价的全面提升。

### 7.5 LM predictability 结果目前只有 seed 23

使用 seed-23 batch-8 QBC checkpoint，在完整 `train-clean-100` 上重新提取 token，并从头训练相同的 10.1M 参数 causal LM：

| Metric | Released codec | QBC | 变化 |
| --- | ---: | ---: | ---: |
| Used codes | 20,480 | 20,480 | 无 collapse |
| Unigram entropy | 9.47498 | 9.46484 | -0.01014 nats |
| LM validation CE | 8.24376 | **8.02795** | -0.21581 nats |
| LM validation PPL | 3,803.8 | **3,065.5** | **-19.41%** |
| Context modeling gain | 1.24185 | **1.45096** | +0.20911 nats |

PPL 改善远大于 unigram entropy 的变化，并且全部 20,480 个 code 仍被使用，因此不像是 codebook collapse 导致的伪改善。

但该完整 intrinsic-LM 实验只完成了 seed 23：

> **它是很强的后续信号，但目前不能称为多种子 LM 因果证据。**

## 8. 过程中得到的负结果和经验

### 8.1 不能只看训练时 agreement

训练时 margin 或 agreement 上升不等于测试集稳定性改善。最终判断必须来自固定 paired test cohort 和 cluster-level CI。

### 8.2 不能只和 released checkpoint 比

普通 continued fine-tuning 本身也会改变 codec。只有同 seed、同 crop、同 optimizer 的 matched control 才能隔离 QBC 的作用。

### 8.3 frame 不是独立样本

同一 utterance、同一 speaker 的 frame 高度相关。把 25,600 帧直接当独立样本会严重夸大显著性，因此 primary CI 必须 resample speakers。

### 8.4 decoder adaptation 不是当前主路线

允许 decoder 一起适配虽然能改善 stability，但造成更大的 reconstruction penalty。它被保留为负向 ablation，不作为主配方。

### 8.5 固定 LM 不等于 intrinsic predictability

固定 LM NLL 回答的是“新 token 是否兼容这个既有 LM”；重新训练相同 LM 回答的是“这套 token 是否本质上更容易建模”。两类指标必须分开解释。

### 8.6 剩余 flip 的单次代价不一定降低

QBC 显著减少 flip 数量，也降低 fixed LM 的总体 full/slice NLL；但每个剩余 flip 的 common-context substitution penalty 没有显著下降。QBC 当前主要解决“减少 flip”，而不是保证“剩余 flip 都更无害”。

### 8.7 公开 bridge 无法做 proxy mismatch 诊断

公开 checkpoint 没有 learned Gumbel bridge 权重，因此目前无法可靠重建：

$$
\mathbf 1[\arg\max\ell_t\ne c_t^{\mathrm{VQ}}].
$$

这项实验需要获得训练期完整 checkpoint，或重新训练官方 LLM-Codec bridge。

## 9. 数据规模与证据边界

### 9.1 当前实际使用规模

- 数据源：LibriSpeech `train-clean-100`，总计约 100.591 小时、28,539 个 utterance；
- 每个 control/QBC arm 的实际训练 exposure：8,000 个有效 4 秒 crop，约 8.89 小时的 crop exposure，不代表覆盖了完整 100 小时且不保证唯一；
- 训练种子：23、41、71；
- 每个测试 split：128 utterances、32 speakers、每 speaker 最多 4 条；
- 每个测试 split：25,600 aligned frames、19,200 interior frames。

### 9.2 当前数据足以支持什么

当前结果足以支持：

1. 部署 VQ 的 boundary radius 提供 latent displacement 之外的解释力；
2. $D<R$ 是有效的局部稳定性 certificate；
3. 直接优化 signed boundary margin 能在多种子、两个 LibriSpeech split 上稳定降低 flip；
4. 该稳定性提升伴随小幅且可复现的重建代价；
5. seed-23 上存在很有潜力的 intrinsic LM predictability 改善。

### 9.3 当前数据不能支持什么

当前结果尚不能支持：

- 对 LibriSpeech 960h 的训练规模泛化；
- 对中文、噪声、说话风格或跨域数据的泛化；
- 多种子 LM PPL 的稳定改善；
- WER/CER、speaker similarity、prosody 等完整下游结论；
- quantizer-native bridge 优于 proxy bridge；
- production-level 或最终论文级的广泛结论。

## 10. 下一步实验优先级

### P0：扩大评估，而不是立即增加更多 loss

1. 在完整 `test-clean` / `test-other` 上评估，并对每个 utterance 使用多个 crop；
2. 保持 speaker-cluster bootstrap；
3. 同时报告 flip、interior flip、signed margin、$\Gamma$、certificate coverage 和 reconstruction；
4. 复现 seed 41、71 的完整 intrinsic-LM 训练，确认 PPL 结论是否跨 seed 成立。

### P1：扩大训练规模

1. 在 LibriSpeech 960h 上训练 1–2 个种子；
2. control/QBC 继续保持 matched sample order；
3. 根据计算量决定完整 codebook scan 或先 `Top-L by distance → Top-M by signed margin`；
4. 若使用近似，先证明最近边界检索 `Recall@M > 99.5%`。

### P2：跨语言独立验证

使用 WenetSpeech-M 做中文验证，重点回答：

- QBC 是否只在 LibriSpeech 英文朗读语音上有效；
- boundary geometry 与 flip 的关系是否跨语言成立；
- 中文 token LM、CER 和重建质量是否保持同样 trade-off。

### P3：完整 codec-LM co-design

在获得或重训完整 LLM-Codec bridge 后，完成：

1. proxy bridge 与真实 VQ argmin 的 disagreement；
2. disagreement 与 low-$R$ frame 的关系；
3. proxy FTP、native FTP、QBC 和 native FTP + QBC 的完整 ablation；
4. 验证 LM gradient 沿真实 quantizer geometry 回传是否优于 learned proxy bridge。

建议的最终 ablation：

| 方法 | Latent MSE | QBC | Proxy bridge | Native bridge | FTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| AUV |  |  |  |  |  |
| ACL'25-style | ✓ |  |  |  |  |
| AUV + QBC |  | ✓ |  |  |  |
| LLM-Codec |  |  | ✓ |  | ✓ |
| LLM-Codec + QBC |  | ✓ | ✓ |  | ✓ |
| Native-FTP |  |  |  | ✓ | ✓ |
| Native-FTP + QBC |  | ✓ |  | ✓ | ✓ |

## 11. 论文叙事建议

### 11.1 推荐主线

不要把论文写成“提出两个 consistency loss”，而应写成：

> Existing consistency methods regularize latent displacement, while discrete token instability is fundamentally governed by the deployed quantizer's local decision geometry. We characterize this geometry with a certified robustness radius, directly enlarge the signed Voronoi margin, and study how the resulting token stability affects language-model predictability.

中文表述：

> 现有方法主要约束连续表征位移，但离散 token 是否改变由部署量化器的局部决策几何决定。我们用可验证的局部鲁棒半径刻画这种几何，直接扩大 signed Voronoi margin，并研究由此获得的 token 稳定性如何影响语言模型可预测性。

### 11.2 当前可成立的贡献

1. **几何刻画**：用部署 VQ 的局部 robustness radius 和 $\Gamma=D/R$ 定量解释 DRI，并证明边界信息在 latent displacement 之外仍有预测价值；
2. **直接干预**：提出 QBC，在真实 normalized VQ 空间直接扩大 teacher-cell signed margin，而不是要求两个 latent 完全重合；
3. **严格验证**：通过 hop-aligned crops、matched controls、多训练种子和 speaker-cluster bootstrap，验证 stability/reconstruction trade-off；
4. **LM 后果**：用 common-context NLL 隔离 prefix confound，并用版本匹配、从头训练的 LM 检查 intrinsic predictability。

### 11.3 目前不能声称的贡献

- 首次发现 token instability 与 quantization boundary 有关；
- 首次发现稳定 token 有利于 SpeechLM；
- 已证明 quantizer-native bridge 优于 LLM-Codec proxy bridge；
- 已实现无损重建的稳定性提升；
- 已证明多语言、大规模泛化。

### 11.4 暂定题目

在 native bridge 尚未完成前，更稳妥的题目是：

> **Boundary-Stable Speech Tokens via Decision-Aware Vector Quantization**

如果后续 native bridge 实验成立，可升级为：

> **Boundary-Stable Neural Speech Codecs via Quantizer-Native Language-Model Training**

## 12. 给导师看的简短版本

我们研究的是 neural codec token 在相同语音片段、不同上下文下发生变化的问题。已有方法通常让 full 和 slice 的连续 latent 靠近，但 VQ token 是否改变真正取决于 latent 有没有跨过 codebook 的 Voronoi 决策边界。我们因此在 AUV/LLM-Codec 的真实量化空间中定义局部边界半径 $R$、上下文位移 $D$ 和边界压力 $\Gamma=D/R$。实验表明，边界半径能在 latent 位移之外显著提升 token flip 的预测能力，且 $D<R$ 的样本没有发生 flip。在此基础上，我们提出 QBC，直接扩大 slice 表征相对于 full teacher cell 的 signed margin。三个训练种子上，QBC 在 LibriSpeech test-clean/test-other 分别稳定降低约 2.26/2.00 个百分点的 flip rate，但带来约 0.5%–1.1% 的谱重建代价。seed-23 的从头 LM 实验中，PPL 从 3,803.8 降至 3,065.5，且 20,480 个 code 全部保持活跃。目前结果足以证明机制有效，但 LM 结论仍需多种子复现，并需要 LibriSpeech 960h 和 WenetSpeech-M 验证大规模及跨语言泛化。

## 13. 代码与结果索引

| 内容 | 路径 |
| --- | --- |
| 几何定义与 certificate | `experiments/vq_geometry.py` |
| QBC loss | `experiments/qbc_loss.py` |
| QBC 训练 | `experiments/train_qbc_codec.py` |
| full-vs-slice 诊断 | `experiments/qbc_diagnostic.py` |
| paired stability 对比 | `experiments/compare_diagnostics.py` |
| 重建评估 | `experiments/eval_codec_reconstruction.py` |
| 多种子聚合 | `experiments/aggregate_qbc_multiseed.py` |
| token 提取 | `experiments/extract_codec_tokens.py` |
| 小型 token LM | `experiments/train_small_token_lm.py` |
| common-context LM | `experiments/common_context_small_lm.py` |
| 仓库安全版结果摘要 | `docs/results.md` |
| 完整本地实验报告 | `artifacts/qbc_diagnostic_v2/README.md` |

## 14. 参考工作

- [Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](https://aclanthology.org/2025.acl-long.1498/)
- [Augmentation Invariant Discrete Representation for Generative Spoken Language Modeling](https://aclanthology.org/2023.iwslt-1.46/)
- [LLM-Codec: Neural Audio Codec Meets Language Model Objectives](https://arxiv.org/abs/2604.17852)
- [StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5186171eebc16ab214e4058ec9fcf63a-Abstract-Conference.html)
- [ReLMCodec: Designing Predictable Speech Tokens from Pre-Quantization Phoneme Structure](https://arxiv.org/abs/2608.08286)

