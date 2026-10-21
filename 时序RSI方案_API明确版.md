下面这版可以直接作为代码框架的设计说明。**保留两个逻辑角色：AdapterAgent 负责执行校正，MetaAgent 负责根据反馈修改 Adapter 和自身流程。数值计算在本地进行；MetaAgent 通过语言模型 API 进行审查、提案并生成工具调用，本地 editor 执行具体修改。**

整个系统有两条反馈链：

$$
\boxed{\text{预测}\rightarrow\text{真实结果}\rightarrow\text{校正反馈}}
$$

$$
\boxed{\text{修改}\rightarrow\text{后续表现}\rightarrow\text{修改反馈}}
$$

第一条帮助系统适应数据，第二条让 MetaAgent 改进自己的修改方法。

下面明确区分：哪些已经有严格数学依据，哪些是我们需要通过实验验证的机制设计。

---

**1．任务定义与信息约束**

设当前时间为 $t$，输入过去 $L$ 个时间点、$D$ 个变量：

$$
X_t\in\mathbb R^{L\times D}.
$$

预测未来 $H$ 个时间点：

$$
Y_t=(y_{t+1},\ldots,y_{t+H}) \in\mathbb R^{H\times D}.
$$

冻结的时序基础模型输出：

$$
B_t=F(X_t).
$$

Adapter 输出残差校正：

$$
\Delta_t=A_{\theta,\psi}(X_t,B_t,\mathcal B_t).
$$

其中：

- $\theta$：当前校正器的数值参数；
- $\psi$：校正器的结构、特征、窗口、拟合与遗忘规则；
- $\mathcal B_t$：当前已经获得真实结果的历史记录。

**区分两种更新很重要：**

- 本地正常拟合，更新 $\theta$，不需要调用 LLM；
- MetaAgent 修订方法，修改 $\psi$ 或程序代码，需要调用 LLM。

最终预测为：

$$
\boxed{ \hat Y_{t,h,c} = B_{t,h,c}+g_{t,c}\Delta_{t,h,c} }
$$

第一版采用**每个变量一个 $g_{t,c}$，在预测长度 $H$ 上共享**，避免一开始就估计过多门控参数。

基础模型只通过 `predict(X)` 暴露预测结果。实验中可以本地加载，但 Adapter 和 MetaAgent 不读取其梯度、隐藏状态或权重。

**时间约束：**

一条在时间 $i$ 发出的完整预测，到 $i+H$ 才能获得全部真实结果。因此：

$$
\boxed{ \mathcal I_t=\{i:i+H\le t\}. }
$$

第一版统一使用完整揭晓的预测记录，不做部分 horizon 的复杂更新。

---

**2．AdapterAgent：一个可执行、可修改的本地校正程序**

它至少提供三个接口：

```python
predict(context, base_forecast) -> correction
update(matured_records) -> None
describe() -> configuration_and_state
```

这里的 Agent 是可被检查、修改和评价的任务程序，**不是要求每次预测都调用 LLM**。

MetaAgent 可以修改：

| 可修改对象 | 示例                 |
| ----- | ------------------ |
| 输入特征  | 最近残差、时间位置、基础预测趋势   |
| 历史范围  | 最近多少条已完成记录         |
| 遗忘方式  | 均匀窗口、指数衰减          |
| 拟合规则  | 正则化、更新频率           |
| 校正器形式 | 常数偏差、线性残差模型、其他轻量模型 |
| 程序实现  | 特征处理、拟合与预测代码       |

**建议初始版本足够简单。**

例如，初始 Adapter 使用最近 $W_A$ 条已完成记录的平均残差：

$$
E_{i,h,c}=Y_{i,h,c}-B_{i,h,c},
$$

$$
\mu_{t,c} = \frac{1}{|\mathcal W_t|H} \sum_{i\in\mathcal W_t}\sum_{h=1}^{H}E_{i,h,c},
$$

$$
\boxed{ \Delta_{t,h,c}=\mu_{t,c}. }
$$

其中 $\mathcal W_t\subseteq\mathcal I_t$ 是最近的历史窗口。

这个版本只能修正常数偏差，能力有限，但实现清楚，适合作为初始程序和简单基线。之后 MetaAgent 可以把它改成线性模型、分预测步长校正等。

无已完成记录时，令：

$$
\Delta_t=0.
$$

不需要为了启动系统，再引入一个额外模型。

---

**3．门控 $g$：使用已经确定的指数加权岭回归**

这一部分保持固定，第一版不让 MetaAgent 修改其公式，方便清楚地评价 RSI 的作用。

为避免变量量纲不同，先用初始历史区间确定尺度 $s_c>0$，例如该变量的标准差，并设置一个数值下限。

定义：

$$
\tilde E_{i,h,c}=\frac{E_{i,h,c}}{s_c}, \qquad \tilde\Delta_{i,h,c}=\frac{\Delta_{i,h,c}}{s_c}.
$$

尺度只用过去数据确定，不能用完整测试集计算。

对每个变量 $c$，门控目标为：

$$
\boxed{ J_{t,c}(g) = \sum_{i\in\mathcal I_t} w_{t,i} \left[ \frac1H\sum_{h=1}^{H} (\tilde E_{i,h,c}-g\tilde\Delta_{i,h,c})^2 \right] +\lambda_g g^2, \quad 0\le g\le1. }
$$

这里：

- $w_{t,i}$：近期记录的权重更大；
- 平方误差：衡量校正后的预测表现；
- $\lambda_g g^2$：在证据不足时，倾向于少校正。

使用指数权重：

$$
w_{t,i} = \frac{2^{-a_{t,i}/h_g}} {\sum_{j\in\mathcal I_t}2^{-a_{t,j}/h_g}},
$$

其中 $h_g$ 是门控记忆的半衰期。

严格求解得到：

$$
\boxed{ g_{t,c} = \operatorname{clip}_{[0,1]} \left( \frac{ \sum_iw_{t,i}\frac1H\sum_h \tilde E_{i,h,c}\tilde\Delta_{i,h,c} }{ \sum_iw_{t,i}\frac1H\sum_h \tilde\Delta_{i,h,c}^{2}+\lambda_g } \right). }
$$

代码里不需要每次遍历历史。令：

$$
\rho=2^{-1/h_g},
$$

每得到一条新的已完成预测记录，计算：

$$
u_{n,c} = \frac1H\sum_h\tilde E_{n,h,c}\tilde\Delta_{n,h,c},
$$

$$
v_{n,c} = \frac1H\sum_h\tilde\Delta_{n,h,c}^{2}.
$$

更新三个统计量：

$$
S_{n,c}=\rho S_{n-1,c}+u_{n,c},
$$

$$
Q_{n,c}=\rho Q_{n-1,c}+v_{n,c},
$$

$$
Z_n=\rho Z_{n-1}+1.
$$

得到下一次预测使用的门控：

$$
\boxed{ g_{\mathrm{next},c} = \operatorname{clip}_{[0,1]} \left( \frac{S_{n,c}}{Q_{n,c}+\lambda_gZ_n} \right). }
$$

初始化时 $S=Q=Z=0$，直接返回 $g=0$。

这里先按**等间隔、按序到达的完整预测记录**实现，半衰期单位是记录数。以后需要不规则时间间隔时，再改为按实际时间衰减。

必须保存当时实际产生的 $\Delta_i$。**不能在看见真实值、重新训练 Adapter 之后，再补算历史校正作为门控证据。**

这一模块保证的是给定加权历史目标下的最优解，不保证未来每次都改善。

---

**4．MetaAgent 的输入：一份预测报告，加一份修改记录**

MetaAgent 不直接接收无限增长的全部原始数据。由本地代码生成简洁报告；必要时，它可以通过工具查看具体历史片段。

第一版报告包含：

| 内容                   | 用途             |
| -------------------- | -------------- |
| 基础预测与最终预测的 MSE       | 当前适配是否有效       |
| 校正后平均有符号误差           | 是否持续高估或低估      |
| 最近几个区间的表现            | 错误是否持续、是否变化    |
| 各变量、各预测步长的主要问题       | 错误集中在哪里        |
| 当前 $g$ 与原始校正幅度摘要 | 校正是被缩小了，还是充分采用 |
| 当前 Adapter 配置和相关代码   | 能改什么           |
| 最近修改及其实际结果           | 避免重复无效修改       |

例如：

$$
\bar r_{k,c} = \frac{1}{|\mathcal B_k|H} \sum_{i\in\mathcal B_k}\sum_h \left(Y_{i,h,c}-\hat Y_{i,h,c}\right).
$$

这是第 $k$ 个反馈区间中，变量 $c$ 的平均有符号误差。

这些都是直接统计量，**不训练额外的诊断模型，也不根据某个单一统计量自动强制触发修改。**

修改记录建议采用结构化格式：

```python
RevisionRecord(
    revision_id,
    parent_adapter_version,
    parent_meta_version,
    observed_problem,
    diagnosis,
    proposed_change,
    expected_effect,
    actual_effect,
    accepted,
)
```

其中 `actual_effect` 由评价程序填写，不能让 MetaAgent 自己编写评价分数。

---

**5．什么时候修改：让 MetaAgent 选择行动，但要求给出可检查的修改依据**

每获得一批新的完整反馈，进行一次审查：

$$
\boxed{ a_k=M_{\phi_k}(R_k,\mathcal H_k), }
$$

其中：

- $R_k$：本次预测报告；
- $\mathcal H_k$：修改历史；
- $\phi_k$：MetaAgent 当前的诊断与提案流程。

行动集合：

$$
\boxed{ a_k\in \{\mathrm{WAIT},\text{REVISE\_ADAPTER}, \text{REVISE\_META}\}. }
$$

建议输出统一格式：

```
{
  "action": "REVISE_ADAPTER",
  "evidence": ["相关区间或修改记录的编号"],
  "diagnosis": "当前问题的具体解释",
  "change": "准备修改哪个部分",
  "expected_effect": "后续应该观察到什么变化"
}
```

不需要它输出一个未经校准的“置信度 0.93”。

三种行动的含义：

| 行动               | 初始决策原则                     |
| ---------------- | -------------------------- |
| `WAIT`           | 当前没有明确的新修改依据，或需要更多反馈       |
| `REVISE_ADAPTER` | 有具体的预测问题，并能提出针对该问题的候选修改    |
| `REVISE_META`    | 过去修改的结果暴露了诊断、提案或经验利用流程中的缺陷 |

**这些原则是 MetaAgent 的初始工作方式，不是永久写死的全部逻辑。** 其具体诊断顺序和提案方式可以被修订。

但是，数据访问时序、损失定义和评价程序保持固定。

审查按批进行，实际修改不必按固定周期进行：

$$
\boxed{\text{固定审查机会}\neq\text{固定修改时间}.}
$$

正在评价候选时，第一版暂停新的结构修改，继续进行正常预测、参数拟合与门控更新。

---

**6．Adapter 修改如何评价与接受**

MetaAgent 生成编辑工具调用，本地 editor 将其中的配置变更或代码补丁应用到独立副本，得到候选 $A'$。旧版本 $A$ 继续提供正式预测，候选在本地同时产生预测。

两者共享同一份基础模型输出 $B_t$，不重复运行基础模型。

先进行必要检查：

- 能否加载；
- 输出形状是否为 $H\times D$；
- 是否产生 NaN 或无穷值；
- 是否遵守历史数据访问接口。

这些检查只能证明程序可运行，不能证明预测有效。

**新版本使用自己的门控统计。**

为了不把旧版本的表现错误地归给新版本，实质性修订后的候选从自己的门控状态开始。建议设置一个短的候选预热阶段：

1. 候选先生成预测并积累真实反馈；
2. 更新候选自己的门控；
3. 预热结束后，进入预先规定长度的正式比较区间。

正式比较只使用**候选创建之后产生的预测及其后续真实结果**。

定义固定的标准化损失：

$$
\ell_i(A) = \frac1{HD} \sum_{h,c} \left( \frac{Y_{i,h,c}-\hat Y^{A}_{i,h,c}}{s_c} \right)^2.
$$

这里的 $\hat Y^A_i$ 是当时实际保存的最终预测，包括当时的 $g$。

在同一评价集合 $\mathcal V_j$ 上：

$$
\boxed{ D_j = \frac1{|\mathcal V_j|} \sum_{i\in\mathcal V_j} \left[\ell_i(A)-\ell_i(A')\right]. }
$$

第一版采用简单规则：

$$
\boxed{ D_j>0\Rightarrow\text{接受候选}; \qquad D_j\le0\Rightarrow\text{保留旧版本}. }
$$

这是经验比较规则，**不包装成显著性检验**。区间长度事先确定，不能每天查看结果、恰好获胜就提前结束。

评价期间：

- 校正器结构和拟合规则固定；
- 数值参数可以按照各版本预先确定的规则继续在线更新；
- 门控可以继续因果更新；
- 两个版本使用相同的可用真实数据。

接受候选时，直接保留其已经运行得到的参数和门控状态，不重新初始化。

---

**7．MetaAgent 自我修订：改的是产生修改的方法**

MetaAgent 可以修改：

- 诊断流程；
- 决策提示词；
- 修改提案的生成步骤；
- 失败案例的检索与使用；
- 分析代码和报告组织方式。

例如，把：

> 看到总体 MSE 增大，就优先缩短窗口。

改成：

> 提议缩短窗口前，先检查错误方向是否持续，并查看过去同类修改是否增加了波动。

形式上：

$$
\boxed{ \phi'\leftarrow \operatorname{SelfRevise}_{\phi} (\phi,R_k,\mathcal H_k). }
$$

当前 MetaAgent 通过 API 生成针对自身程序或指令的编辑工具调用，本地 editor 创建新版本；当前调用结束后，新版本才能进入试用。

**第一版建议为自我修改增加一次小规模的新旧对照。** 你已经表示不必过度吝啬 API，这笔调用比直接相信自我反思更有价值。

具体做法：

1. 固定同一个起始 Adapter 和同一份反馈。
2. 旧 MetaAgent、新 MetaAgent 各有一次提案机会。
3. 两者都可以提出一个 Adapter 候选，或选择等待。
4. 用相同的后续数据评价候选。
5. 比较哪一个 MetaAgent 带来了更好的可接受改进。

设当前 Adapter 的评价损失为 $L_0$，旧、新 MetaAgent 提出的候选损失分别为 $L_{\mathrm{old}}$、$L_{\mathrm{new}}$。如果选择等待，就令候选损失为 $L_0$。

定义：

$$
U_{\mathrm{old}} = L_0-\min(L_0,L_{\mathrm{old}}),
$$

$$
U_{\mathrm{new}} = L_0-\min(L_0,L_{\mathrm{new}}).
$$

这表示：在仍然可以保留当前 Adapter 的前提下，该 MetaAgent 的一次提案带来了多少改善。

接受规则：

$$
\boxed{ U_{\mathrm{new}}>U_{\mathrm{old}} \Rightarrow \text{接受新的 MetaAgent}. }
$$

平局保留旧版本。Adapter 则选择本轮候选中实际表现最好的版本。

这种方式有三个优点：

- 新旧 MetaAgent 从同一个起点出发；
- 不需要引入新的预测器；
- 自我修改必须体现在实际提案效果上。

**它仍然只是一次提案能力的局部检验。** 不能据此证明长期自我改进，也不能完整评价“更会等待”这样的长期决策能力。长期收益必须通过完整流式实验和固定 MetaAgent 的对照验证。

这也是为什么我们不把一次自我修订成功写成“已证明 RSI”。

---

**8．完整运行顺序**

特别注意：**先处理当前已经到达的真实反馈，再发出新的预测。**

```python
for current_time in stream:
    # 1. 取出到当前时刻已经完全揭晓的预测
    matured = pending.pop_ready(current_time)

    # 2. 用当时保存的预测计算误差：本地执行
    evaluator.record(matured)

    # 3. 更新各版本自己的门控统计：本地执行
    update_versioned_gates(matured)

    # 4. 将真实结果加入训练缓存
    buffer.append(matured)

    # 5. 各运行版本按自己的固定规则更新数值参数：本地执行
    update_running_adapters(buffer, matured)

    # 6. 本地评价器决定接受或拒绝，并写入修改记录
    finalize_completed_trials()

    # 7. 满足审查批量且没有正在进行的修订试验时，调用 MetaAgent
    if review_due() and not trial_in_progress():
        report = build_report()

        # LLM API：产生行动、修改依据及可执行的工具调用
        decision = meta_agent.review(report, revision_history)

        # 本地 editor 执行修改、检查并登记试验；失败才按预算请求修复
        # REVISE_META 还需按第7节调用新旧MetaAgent各产生一次提案
        handle_decision_with_tools(decision)

    # 8. 时序基础模型在本地运行一次，模拟黑盒API
    base_forecast = foundation.predict(context)

    # 9. AdapterAgent及临时候选在本地生成校正和最终预测
    forecasts = predict_all_running_versions(context, base_forecast)

    # 10. 保存原始校正、当时门控和最终预测，等待未来真实值
    pending.add(forecasts)
```

如果 API 调用或代码修改失败，本轮继续使用现有版本，记录失败即可，不能阻断预测主循环。

真实部署可以异步执行 MetaAgent。离线实验即使按顺序等待，也必须限制其只读取当时已经可用的数据。

---

**9．代码目录与关键数据对象**

建议先按下面划分：

```
project/
  foundation/
    wrapper.py

  adapters/
    base.py
    mean_residual.py

  gating/
    discounted_ridge.py

  meta/
    agent.py
    prompts.py
    tools.py

  streaming/
    runner.py
    buffer.py
    pending.py

  evaluation/
    metrics.py
    adapter_trial.py
    meta_trial.py

  revisions/
    manager.py
    records.py

  configs/
    experiment.yaml
```

其中：

- `adapters/`、部分 `meta/` 内容允许产生候选修改；
- `streaming/` 的时间访问规则和 `evaluation/` 的评价标准固定；
- 候选代码放在独立版本中，不直接覆盖当前正式版本。

每条预测至少记录：

```python
ForecastRecord(
    forecast_id,
    origin_time,
    target_times,
    adapter_version,
    meta_version,
    base_forecast,
    raw_correction,
    gate,
    final_forecast,
)
```

真实结果到齐后再附加：

```python
target_values
loss_base
loss_final
```

**版本标识不可省略。** 切换 Adapter 后，旧版本尚未到达的反馈仍可能陆续出现，不能把它们的旧校正结果用于新版本门控。

完整数值记录留在本地，MetaAgent 默认只看到摘要与必要片段。

---

**10．API 使用方式：MetaAgent 决策并调用工具，本地 editor 执行修改**

**本节只明确执行分工，不改变 AdapterAgent + MetaAgent 的双 Agent 架构，也不改变前面的门控、触发和评价机制。** 两个 Agent 可以共用同一个语言模型后端；不要求每次预测分别调用两次 API。

**10.1 哪些环节调用 API**

| 环节 | 执行者 | 是否需要 LLM API |
| --- | --- | --- |
| 基础时序预测 $B_t=F(X_t)$ | 本地冻结的 foundation model，模拟黑盒接口 | 否 |
| AdapterAgent 产生校正 $\Delta_t$ | AdapterAgent 执行当前本地程序 | 否 |
| AdapterAgent 正常拟合、更新数值参数 $\theta$ | 本地训练工具 | 否 |
| 计算门控 $g$、误差和报告摘要 | 本地数值程序 | 否 |
| 审查反馈，选择 WAIT / REVISE_ADAPTER / REVISE_META | MetaAgent | 是 |
| 确定具体修改并生成编辑工具调用 | MetaAgent | 是；可以与审查合并在一次响应中 |
| 查看文件、替换内容、写入配置或代码 | 本地 editor | 否；若需模型读完结果再决定，后续推理另计一次 API |
| 运行检查、拟合候选、安排后续评价 | 本地工具执行器 | 否 |
| 根据后续真实值接受或拒绝候选 | 固定评价程序 | 否 |
| 自我修订后，新旧 MetaAgent 各产生一次对照提案 | 新旧 MetaAgent 分别调用同一个 LLM 后端 | 是；按第7节执行，不能当作零API开销 |

AdapterAgent 的角色保持为执行校正并按当前规则学习的任务 Agent。MetaAgent 的角色保持为修改 AdapterAgent 以及自身改进流程的元 Agent。editor 只是工具，不增加第三个 Agent。

**10.2 一次审查给 API 什么内容**

输入由第4节报告与修改历史组成，默认提供：

- 当前预测表现和最近几个区间的变化；
- 当前 AdapterAgent 的配置、接口及必要代码片段；
- 近期相关修改的“判断—操作—预期—实际结果”；
- 当前 MetaAgent 的可编辑操作指令；
- 可用工具说明及本次调用预算。

完整时序数组、全部训练样本和全部代码保留在本地。需要细查时，MetaAgent 可以调用读取或统计工具，再据结果继续推理。不能为了省调用而删掉决定修改所必需的信息。

**10.3 API 输出的是可执行的工具请求，而不只是一句修改建议**

在第5节决策格式上增加 `tool_calls`。例如，下面表示在候选副本中把历史窗口改为64；数字仅用于演示：

```json
{
  "action": "REVISE_ADAPTER",
  "evidence": ["report_018", "revision_006"],
  "diagnosis": "已有反馈提示当前历史窗口可能响应过慢",
  "change": "将候选Adapter的历史窗口从128改为64",
  "expected_effect": "在后续评价区间降低持续偏差与MSE",
  "tool_calls": [
    {
      "tool_name": "editor",
      "tool_input": {
        "command": "str_replace",
        "path": "adapter/config.json",
        "old_str": "\"history_window\": 128",
        "new_str": "\"history_window\": 64"
      }
    }
  ]
}
```

这里的路径由执行器解析到本次候选副本中。editor 校验原文是否匹配后执行替换；正式版本不被直接覆盖。WAIT 可以返回空的 `tool_calls`。

此格式是我们项目的请求接口。HyperAgents 源码采用相同的“模型产生工具参数、本地执行”分工，但它的默认循环会在工具返回后继续询问模型；我们不必照搬每一步都回到LLM的交互方式。

**editor 的能力边界要明确：** 它可以确定性地替换文本、更新配置、插入代码，却不会凭一句“改成更好的模型”自主生成新算法。常规参数修改只需简短参数；真正新增功能时，MetaAgent 仍要通过API生成必要的局部代码，放进 `new_str` 等参数。这些代码内容仍计入API输出，不把它们伪装成本地免费生成。

可以为常见配置修改提供 `set_config` 等便利操作，但保留通用局部编辑能力，不能把 RSI 限制为只能从固定参数表中选择。

**10.4 本地自动执行，只有需要新判断时才继续调用 API**

收到 `REVISE_ADAPTER` 后，执行器自动：

1. 创建当前 AdapterAgent 的独立候选副本。
2. 执行 editor 请求。
3. 检查加载、输出形状和数值是否正常。
4. 用当前已揭晓的历史拟合或初始化候选。
5. 按第6节登记候选预热与未来评价，不在当前时刻伪造未来结果。
6. 保存修改内容与预期效果，结束本次审查。

成功完成这些确定步骤后，不必再请求模型逐一批准训练、确认保存或宣布成功。若编辑失败、报错或者缺少必要信息，则把简短错误和相关代码发回MetaAgent，由它修复或放弃。本轮仍受调用预算约束。

收到 `REVISE_META` 后，editor 在独立副本中修改 MetaAgent 的诊断代码、操作指令、提案流程或记忆使用方式。随后按第7节启动新旧MetaAgent的提案对照：这一步会产生额外API调用；后续训练、预热、预测与评分仍在本地完成。

**MetaAgent 修改自己，指修改自己的可执行流程与持久化操作指令，不是修改API背后语言模型的权重。** 新流程必须在之后实际参与决策，不能仅写入一段不会被使用的总结。

**10.5 调用次数、预算与记录**

先使用同一个语言模型后端完成审查、编辑提案和自我修订，不额外安排一个“写代码Agent”。新旧MetaAgent对照使用相同的后端和相同的调用上限，差异来自被修订的MetaAgent流程。

- WAIT：通常一次审查调用。
- 简单配置或小补丁：上下文充分时，一次审查响应即可包含工具请求。
- 需要查代码或修复的修改：按需继续调用，允许有用的多轮交互。
- MetaAgent自我修订：除了编辑自身，还要计算新旧版本各自提案的调用量。

每个审查／修订会话先设最多4次API调用，包含该会话的首次决策、必要读取后的继续推理和修复。MetaAgent对照中的旧、新提案会话分别使用同样的上限，不能偷偷只给其中一方更多调试机会。上限是工程起点，开发实验中可以统一调整。

调用次数可按互不重叠的类别记录：

$$
N_{\mathrm{API}}
=N_{\mathrm{regular\ review}}
+N_{\mathrm{followup}}
+N_{\mathrm{meta\ trial\ proposal}}.
$$

其中第一项是常规审查的首次调用；第二项是所有会话的后续推理与修复；第三项是自我修订对照中旧、新MetaAgent的首次提案调用。错误重试也按实际情况计入日志。

同时记录输入token、输出token、供应商返回的其他计费token、耗时和费用。工具执行次数单独记录，不能把一次本地编辑等同于一次API请求。

**调用上限只是执行约束，不放入预测损失函数。** 论文分别报告预测效果与API消耗。

保留原方案用于跑通框架的起始参数：

| 配置 | 起始设置 |
| --- | --- |
| 每次审查新增的完整预测记录 | 32 条 |
| 初始平均残差 Adapter 窗口 | 128 条 |
| 历史训练缓存上限 | 2,048 条 |
| 门控半衰期 $h_g$ | 64 条完整预测记录 |
| 门控正则 $\lambda_g$ | 标准化后先试 $0.01$ |
| 候选门控预热 | 32 条完整预测记录 |
| 正式候选评价 | 后续 64 条预测记录 |
| 常规 Adapter 候选数 | 同时最多 1 个 |
| MetaAgent 对照 | 旧、新各提案 1 次；需要的编辑与修复受相同预算约束 |
| 每个审查／修订会话API调用上限 | 先设 4 次，非必须用满 |

这里的“32条记录”不是“32个目标数值”：每条记录包含一次完整的 $H\times D$ 预测。预测区间可能重叠，不能直接把记录当作独立样本计算普通置信区间。

这些参数是运行起点，不是理论最优值。窗口和半衰期在开发集上检查敏感性，不用最终测试结果反复挑选。

源码分工核对依据（HyperAgents提交 `59a68f6`）：

- [MetaAgent入口](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/meta_agent.py)
- [LLM与工具交互循环](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/agent/llm_withtools.py)
- [本地editor实现](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/agent/tools/edit.py)

---

**11．第一轮实验需要回答的三个问题**

先跑下面四组就足以检查核心设计：

| 实验组          | 门控 | Adapter 修订 | MetaAgent 自我修订 |
| ------------ | -- | ---------- | -------------- |
| 固定 Adapter   | 有  | 无          | 无              |
| 固定周期修订       | 有  | 定期尝试       | 无              |
| 固定 MetaAgent | 有  | 根据反馈选择     | 无              |
| 完整方法         | 有  | 根据反馈选择     | 有              |

所有组使用相同的基础模型、数据顺序、反馈延迟和门控实现。需要 LLM 的组尽量匹配审查机会与调用上限。

重点检查：

1. **反馈驱动的修改，是否比定期修改更有效？**
2. **MetaAgent 自我修订，是否比固定 MetaAgent 更有效？**
3. **在噪声增加但没有明显新规律时，系统是否会反复进行无效修改？**

除预测误差外，保存修改接受率、连续失败次数、版本变化及其对应的实际收益。选几个具体案例展示“诊断流程改了什么，后来行为怎样改变”，比只有最终 MSE 更能说明 RSI 是否真正起作用。

**实现时优先完成“时间顺序正确的预测记录、门控更新、候选评价”这三个底层模块，再接入 MetaAgent。** 这样 Agent 提出的修改无论多复杂，都必须经过同一套可复现的真实反馈流程；而我们已经推导好的 $g$ 可以从第一天就独立运行。