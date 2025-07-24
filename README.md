\# 🛡️ IoT Security Intelligent Assistant



基于RAG+Ollama技术的IoT安全智能分析系统，提供专业的漏洞分析、威胁检测和安全评估服务。



\## ✨ 主要特性



\- 🤖 \*\*本地大模型\*\*: 集成Ollama支持，保护数据隐私

\- 📚 \*\*RAG知识库\*\*: 结合学术研究和CVE漏洞数据库

\- 🔍 \*\*智能检索\*\*: 混合语义检索和关键词匹配

\- 💬 \*\*Web界面\*\*: 友好的Streamlit聊天界面

\- 🎯 \*\*专业分析\*\*: CVE分析、威胁检测、安全评估

\- 🚀 \*\*一键启动\*\*: 自动环境检查和依赖安装



\## 🏗️ 系统架构



```

IoT Security Assistant

├── RAG Agent (rag\_agent.py)           # 核心智能体

├── Ollama Client (ollama\_client.py)   # 本地大模型客户端

├── RAG System (rag\_preprocessing.py)  # 知识检索系统

├── Prompt Templates (prompt\_templates.py) # 提示工程

├── Web Interface (streamlit\_app.py)   # Web聊天界面

└── Launcher (launch\_web.py)           # 一键启动器

```



\## 🚀 快速开始



\### 1. 环境要求



\- Python 3.8+

\- 8GB+ RAM推荐

\- Ollama服务



\### 2. 安装步骤



```bash

\# 克隆项目

git clone https://github.com/hhh678654/iot-security-assistant.git

cd iot-security-assistant



\# 安装依赖

pip install -r requirements.txt



\# 安装Ollama (如果未安装)

\# Windows: 下载 https://ollama.ai/download

\# macOS: brew install ollama

\# Linux: curl -fsSL https://ollama.ai/install.sh | sh



\# 启动Ollama服务

ollama serve



\# 下载推荐模型

ollama pull llama3.1:7b

```



\### 3. 启动系统



```bash

\# 一键启动Web界面

python launch\_web.py



\# 或直接启动Streamlit

streamlit run streamlit\_app.py

```



访问 http://localhost:8501 开始使用！



\## 📖 使用示例



\### CVE漏洞分析

```

Q: dglogik inc dglux服务器存在哪些漏洞？

A: 提供详细的漏洞分析、影响评估和缓解措施

```



\### 威胁检测

```

Q: IoT设备异常网络行为分析

A: 基于MITRE ATT\&CK框架进行威胁分析

```



\### 安全评估

```

Q: 评估智能家居安全风险

A: 多维度安全评估和改进建议

```



\## 🛠️ 配置选项



\### Ollama模型配置

```python

config = {

&nbsp;   "model\_name": "llama3.1:7b",  # 可选: qwen2, mistral等

&nbsp;   "temperature": 0.7,           # 创意度 (0.0-1.0)

&nbsp;   "max\_tokens": 2000,          # 最大回复长度

&nbsp;   "timeout": 60                # 超时时间

}

```



\### RAG检索配置

```python

rag\_config = {

&nbsp;   "use\_rag": True,             # 启用知识库

&nbsp;   "top\_k": 5,                  # 检索文档数

&nbsp;   "hybrid\_alpha": 0.7          # 混合检索权重

}

```



\## 📁 项目结构



```

iot-security-assistant/

├── README.md                    # 项目说明

├── requirements.txt             # Python依赖

├── launch\_web.py               # 启动器

├── streamlit\_app.py            # Web界面

├── rag\_agent.py                # 核心Agent

├── ollama\_client.py            # Ollama客户端

├── rag\_preprocessing.py        # RAG预处理

├── prompt\_templates.py         # 提示模板

├── data/                       # 数据目录

│   ├── academic\_data.csv       # 学术数据

│   └── cve\_data.csv           # CVE数据

├── rag\_indices/               # RAG索引文件

│   ├── semantic\_index.faiss   # 语义索引

│   └── indexed\_documents.json # 索引文档

└── cache/                     # 缓存目录

```



\## 🎯 核心功能



\### 1. CVE漏洞分析

\- 详细漏洞信息解析

\- CVSS评分分析

\- 影响范围评估

\- 缓解措施建议



\### 2. 威胁检测分析

\- 基于MITRE ATT\&CK框架

\- 攻击路径分析

\- IoT特定风险评估

\- 检测策略建议



\### 3. 安全评估

\- 多维度安全评估

\- 设备/网络/数据安全

\- 合规性检查

\- 优先级改进建议



\### 4. 防护建议

\- 分层防护策略

\- 技术实施方案

\- 成本效益分析

\- 实施路线图



\## 🔧 高级功能



\### 提示工程系统

\- Few-shot学习示例

\- Chain-of-Thought推理

\- 动态提示优化

\- 任务特定模板



\### 混合检索RAG

\- 语义向量检索

\- TF-IDF关键词匹配

\- 智能结果融合

\- 上下文感知排序



\### 性能优化

\- 响应缓存机制

\- 智能批处理

\- 内存使用优化

\- 并发请求支持



\## 📊 性能指标



| 指标 | 数值 |

|------|------|

| 平均响应时间 | < 5秒 |

| 知识库文档 | 10,000+ |

| 支持语言 | 中文/英文 |

| 并发用户 | 10+ |

| 内存使用 | < 4GB |



\## 🤝 贡献指南



1\. Fork项目

2\. 创建特性分支: `git checkout -b feature/amazing-feature`

3\. 提交更改: `git commit -m 'Add amazing feature'`

4\. 推送分支: `git push origin feature/amazing-feature`

5\. 创建Pull Request



\## 📝 开发计划



\- \[ ] 添加更多IoT设备类型支持

\- \[ ] 集成更多威胁情报源

\- \[ ] 支持多语言界面

\- \[ ] 移动端适配

\- \[ ] API接口开发

\- \[ ] 企业版功能



\## ⚠️ 注意事项



1\. \*\*模型要求\*\*: 推荐使用7B+参数模型获得最佳效果

2\. \*\*硬件配置\*\*: 建议8GB+内存，支持GPU加速

3\. \*\*网络环境\*\*: 首次运行需下载模型文件（数GB）

4\. \*\*数据隐私\*\*: 所有处理均在本地完成，不上传外部服务



\## 🐛 故障排除



\### Ollama服务问题

```bash

\# 检查服务状态

ollama list



\# 重启服务

ollama serve



\# 重新下载模型

ollama pull llama3.1:7b

```



\### RAG索引问题

```bash

\# 重建索引

python rag\_preprocessing.py



\# 检查索引文件

ls rag\_indices/

```



\### 依赖安装问题

```bash

\# 升级pip

pip install --upgrade pip



\# 清理缓存重装

pip cache purge

pip install -r requirements.txt --force-reinstall

```



\## 📄 许可证



本项目采用 MIT 许可证 - 查看 \[LICENSE](LICENSE) 文件了解详情。



\## 🙏 致谢



\- \[Ollama](https://ollama.ai/) - 本地大模型运行时

\- \[Streamlit](https://streamlit.io/) - Web应用框架

\- \[Sentence Transformers](https://www.sbert.net/) - 语义检索

\- \[FAISS](https://faiss.ai/) - 向量数据库

\- \[MITRE ATT\&CK](https://attack.mitre.org/) - 威胁分析框架



---



<div align="center">

&nbsp; <sub>Built with ❤️ by IoT Security Community</sub>

</div>

