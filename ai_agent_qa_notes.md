# 关于 AI Agent 的对话整理

> 整理自 2026-04-19 对话  
> 起点：「我还不知道 agent 准确来说是什么？作什么用？怎么用？上哪儿得到 agent？怎么自己搭建或制作 agent？但我听说这个名词铺天盖地，到处都在说这个名词。引起我好奇。」

---

## Q1：Agent 准确来说是什么？

**Agent（智能体）= 能自主决策、调用工具、完成多步骤任务的 AI 程序。**

和普通 AI（比如 ChatGPT 问答）的区别：

| | 普通 AI | Agent |
|---|---|---|
| 工作方式 | 你问一句，它答一句 | 你给个目标，它自己拆步骤去完成 |
| 能用工具吗 | ❌ | ✅ 搜索、写代码、发邮件… |
| 需要人介入 | 每一步都要你参与 | 只需要在目标层面介入 |
| 典型用时 | 秒级 | 分钟到小时 |

**类比：**  
普通 AI 像一个很聪明的人，但只会"回答问题"，不会"主动做事"。  
Agent 像一个有手有脚还会查资料的助理，你说"帮我做 X"，它会自己想办法搞定。

---

## Q2：Agent 作什么用？（有哪些实际场景）

### 场景 1：研究助手
```
你：帮我调研「GPU 显存优化」技术
Agent 自动：
  → 搜索最新论文
  → 读取摘要
  → 对比各方法
  → 输出总结报告
```

### 场景 2：代码助手（如 GitHub Copilot Agent）
```
你：修复这个 bug
Agent 自动：
  → 读代码文件
  → 定位问题
  → 修改代码
  → 运行测试
  → 验证通过
```

### 场景 3：个人助理
```
你：帮我安排明天日程
Agent 自动：
  → 读你的邮件
  → 分析优先级
  → 写入日历
  → 发送提醒
```

### 场景 4：自动化业务流程
- 监控竞品价格 → 自动调价
- 客服自动回复 → 复杂问题升级人工
- 爬取数据 → 清洗 → 生成报表

---

## Q3：Agent 内部是怎么工作的？

```
┌─────────────────────────────────────┐
│              Agent 核心             │
│                                     │
│  🧠 LLM（大脑）                     │
│     ↓ 决策：下一步做什么？           │
│  🔧 工具调用（Tools）               │
│     - 搜索网页                      │
│     - 读写文件                      │
│     - 执行代码                      │
│     - 调用 API                      │
│     ↓                               │
│  📋 记忆（Memory）                  │
│     - 短期：当前对话上下文           │
│     - 长期：数据库 / 向量检索        │
│     ↓                               │
│  🔄 循环：执行 → 观察结果 → 再决策  │
└─────────────────────────────────────┘
```

核心模式叫 **ReAct**（Reasoning + Acting）：
1. 思考（Thought）：现在应该做什么？
2. 行动（Action）：调用工具
3. 观察（Observation）：看结果
4. 重复，直到目标达成

---

## Q4：上哪儿得到 Agent？（直接用，不用自己搭）

### 现成产品（拿来就用）

| 产品 | 是什么 | 地址 |
|---|---|---|
| **ChatGPT** | OpenAI 的 Agent，可用工具、联网、跑代码 | chat.openai.com |
| **GitHub Copilot** | 代码 Agent，就是你现在用的这个 | VS Code 内置 |
| **Perplexity AI** | 搜索 + 推理 Agent | perplexity.ai |
| **Notion AI** | 文档写作 Agent | notion.so |
| **Cursor** | 代码 Agent（比 Copilot 更激进） | cursor.sh |
| **Claude** | Anthropic 的 Agent | claude.ai |

### 无代码平台（自己配置，不用写代码）

| 平台 | 特点 |
|---|---|
| **Dify** | 开源，可视化拖拽，支持自部署 |
| **n8n** | 自动化工作流，连接各种 APP |
| **Coze（扣子）** | 字节出品，中文友好，免费额度大 |
| **FastGPT** | 开源，适合搭知识库 Agent |

---

## Q5：怎么自己搭建 Agent？

### 方式一：无代码（最快，今天就能试）

**用 Dify 或 Coze：**
1. 注册账号
2. 新建 Agent 应用
3. 选大模型（GPT-4o / Claude 等）
4. 添加工具（搜索、代码执行、自定义 API）
5. 写系统提示词（告诉 Agent 它是谁、做什么）
6. 测试 → 发布

### 方式二：写代码（Python）

**主流框架：**

| 框架 | 特点 | 难度 |
|---|---|---|
| **LangChain** | 最成熟，生态最大 | ⭐⭐⭐ |
| **LangGraph** | 图结构工作流，更灵活 | ⭐⭐⭐⭐ |
| **CrewAI** | 多 Agent 协作，像团队分工 | ⭐⭐⭐ |
| **AutoGen** | 微软出品，多 Agent 对话 | ⭐⭐⭐ |

**最简单的 LangChain Agent 代码示例：**

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

# 工具：网页搜索
search = DuckDuckGoSearchRun()
tools = [Tool(name="Search", func=search.run, description="搜索网页")]

# 初始化 Agent
llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# 运行
result = agent.run("帮我找最新的 CUDA 12 新特性")
print(result)
```

### 方式三：直接调用 API（最底层）

用 OpenAI 的 Function Calling / Tool Use API，自己管理循环逻辑。  
适合完全定制化需求，但需要自己写更多代码。

---

## Q6：为什么现在这个词铺天盖地？

2023 年底到 2024 年，LLM 能力突破了一个关键门槛：**工具调用变得可靠了**。

之前：LLM 调用工具经常出错、乱来  
现在：GPT-4o、Claude 3.5 等模型工具调用准确率很高

所以：
- 大量公司开始把 Agent 用于实际业务
- 开发者发现原来 AI 能"做事"不只是"说话"
- VC 大量投钱，媒体铺天盖地报道

**一个判断：**
> Agent 现在类似 2007 年的"移动互联网"——概念刚爆发，真正的杀手级应用还在涌现中。现在学，时机正好。

---

## 和 GPU / CUDA 学习的关联

Agent 框架是 Python 应用层，但底层跑在 GPU 上：

```
你的 Agent 代码（Python）
    ↓
调用大模型推理（OpenAI API 或本地模型）
    ↓
推理引擎（vLLM、TensorRT-LLM）
    ↓
CUDA Kernel（你正在学的这层！）
    ↓
GPU 硬件
```

学了 CUDA 之后，你能理解推理引擎是怎么用 GPU 加速 LLM 推理的——这是 AI 系统栈中最接近硬件的那层。

---

## 推荐下一步

| 行动 | 时间 | 目标 |
|---|---|---|
| 注册 Coze（扣子）试玩 | 30 分钟 | 感受 Agent 能做什么 |
| 看吴恩达「AI Agents」免费课 | 2-3 小时 | 理解原理 | 
| 跑一个 LangChain 示例 | 1 小时 | 写第一个 Agent |
| 结合 GPU 学习：读 vLLM 架构 | 未来 | 理解推理加速 |

> 吴恩达课程地址：[learn.deeplearning.ai](https://learn.deeplearning.ai)（免费，英文）

---

*整理自 2026-04-19 对话*
