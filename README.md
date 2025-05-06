# oi-wiki-agent

一个基于 RAG（Retrieval-Augmented Generation）的问答应用，结合 **OceanBase 向量数据库** 与 **Streamlit 前端界面**，为算法竞赛知识库 [OI Wiki](https://github.com/OI-wiki/OI-wiki) 提供智能问答支持。

## ✨ 项目亮点

- 🚀 使用 OceanBase 作为向量检索库，快速定位相关知识片段
- 🧠 集成 LLM 意图分析、查询重写、板块判断等模块
- 💡 实现 RAG 架构，有效结合检索与生成的优势
- 💻 采用 Streamlit 搭建前端界面，交互友好、易于部署

## 🚀 快速开始

### ⬇️ 拉取仓库和安装环境

```bash
git clone https://github.com/Mai-icy/oi-wiki-agent.git
cd oi-wiki-agent
pip install -r requirements.txt
```

### ⚙️ 项目配置

创建编写.env文件

```
API_KEY=你的阿里云百炼平台API_KEY
LLM_MODEL="qwen-turbo-latest"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

HF_ENDPOINT=https://hf-mirror.com
BGE_MODEL_PATH=BAAI/bge-m3

OLLAMA_URL=
OLLAMA_TOKEN=
TABLE_NAME=

OPENAI_EMBEDDING_API_KEY=你的阿里云百炼平台API_KEY
OPENAI_EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
OPENAI_EMBEDDING_MODEL=text-embedding-v3

UI_LANG="zh"

# 你的Oceanbase数据库连接信息
DB_HOST=
DB_PORT=
DB_USER=
DB_NAME=
DB_PASSWORD=

```

### 📘 创建知识库

```bash
git clone https://github.com/OI-wiki/OI-wiki.git doc/
python oi_wiki_loader.py
```

### 🚀 开始

```
streamlit run app.py
```