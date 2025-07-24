# rag_agent.py - 任务分类，中枢Agent
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

# 导入Ollama客户端
from ollama_client import OllamaClient, OllamaResponse

# 导入已有的RAG系统
try:
    from rag_preprocessing import HybridRAGSystem
    from prompt_templates import PromptTemplateManager
except ImportError:
    print("⚠️ 未找到RAG或Prompt模块，将使用简化模式")
    HybridRAGSystem = None
    PromptTemplateManager = None


class TaskType(Enum):
    """任务类型枚举"""
    CVE_LOOKUP = "cve_lookup"
    THREAT_DETECTION = "threat_detection"
    SECURITY_ASSESSMENT = "security_assessment"
    PREVENTION_ADVICE = "prevention_advice"
    RESEARCH_SUMMARY = "research_summary"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    GENERAL_QUERY = "general_query"


@dataclass
class SimpleAgentResponse:
    """Agent响应"""
    content: str
    confidence: float
    sources: List[Dict]
    task_type: str
    processing_time: float
    model_used: str
    metadata: Dict


class SimpleRAGAgent:
    """RAG+Ollama Agent"""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()

        # 初始化Ollama客户端
        self.ollama_client = OllamaClient(
            model_name=self.config.get("model_name", "llama3.1:7b"),
            base_url=self.config.get("base_url", "http://localhost:11434"),
            timeout=self.config.get("timeout", 60)
        )

        # 初始化RAG系统（如果可用）
        self.rag_system = None
        self.documents = None
        self._initialize_rag_system()

        # 性能统计
        self.stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "rag_searches": 0,
            "average_confidence": 0.0
        }

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "model_name": "llama3.1:7b",
            "base_url": "http://localhost:11434",
            "timeout": 60,
            "max_tokens": 2000,
            "temperature": 0.7,
            "use_rag": True,
            "rag_top_k": 5,
            "base_dir": "."
        }

    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger("SimpleRAGAgent")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_rag_system(self):
        """初始化RAG系统"""
        if not self.config.get("use_rag", True):
            self.logger.info("RAG系统已禁用")
            return

        try:
            if HybridRAGSystem is None:
                self.logger.warning("RAG系统不可用，将在简化模式下运行")
                return

            # 初始化RAG系统
            rag_config = {
                'embedding_model': 'all-MiniLM-L6-v2',
                'top_k': self.config.get('rag_top_k', 5),
                'hybrid_alpha': 0.7
            }

            self.rag_system = HybridRAGSystem(rag_config)

            # 加载预建索引
            self._load_rag_indices()

            if self.documents:
                self.logger.info(f"RAG系统初始化成功，加载了 {len(self.documents)} 个文档")
            else:
                self.logger.warning("RAG索引文件不存在，将在无知识库模式下运行")

        except Exception as e:
            self.logger.error(f"RAG系统初始化失败: {e}")
            self.rag_system = None
            self.documents = None

    def _load_rag_indices(self):
        """加载RAG索引"""
        try:
            import faiss
            import pickle
            from pathlib import Path

            base_dir = Path(self.config.get('base_dir', '.'))
            index_dir = base_dir / "rag_indices"

            # 检查必要文件是否存在
            required_files = [
                "indexed_documents.json",
                "semantic_index.faiss",
                "tfidf_matrix.pkl",
                "tfidf_vectorizer.pkl"
            ]

            missing_files = []
            for file in required_files:
                if not (index_dir / file).exists():
                    missing_files.append(file)

            if missing_files:
                self.logger.warning(f"RAG索引文件缺失: {missing_files}")
                return

            # 加载文档
            with open(index_dir / "indexed_documents.json", 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

            # 加载语义索引
            self.semantic_index = faiss.read_index(str(index_dir / "semantic_index.faiss"))

            # 加载TF-IDF相关数据
            with open(index_dir / "tfidf_matrix.pkl", 'rb') as f:
                self.tfidf_matrix = pickle.load(f)

            with open(index_dir / "tfidf_vectorizer.pkl", 'rb') as f:
                self.rag_system.tfidf_vectorizer = pickle.load(f)

        except Exception as e:
            self.logger.error(f"RAG索引加载失败: {e}")
            self.documents = None

    def classify_task(self, user_input: str) -> TaskType:
        """简单的任务分类"""
        user_input_lower = user_input.lower()

        if any(keyword in user_input_lower for keyword in ['cve-', 'vulnerability', '漏洞']):
            return TaskType.CVE_LOOKUP
        elif any(keyword in user_input_lower for keyword in ['attack', 'threat', '威胁', '攻击']):
            return TaskType.THREAT_DETECTION
        elif any(keyword in user_input_lower for keyword in ['assessment', 'risk', '风险', '评估']):
            return TaskType.SECURITY_ASSESSMENT
        elif any(keyword in user_input_lower for keyword in ['prevention', 'protect', '防护', '缓解']):
            return TaskType.PREVENTION_ADVICE
        elif any(keyword in user_input_lower for keyword in ['research', 'summary', '研究', '总结']):
            return TaskType.RESEARCH_SUMMARY
        elif any(keyword in user_input_lower for keyword in ['analyze', 'analysis', '分析']):
            return TaskType.VULNERABILITY_ANALYSIS
        else:
            return TaskType.GENERAL_QUERY

    def search_knowledge(self, query: str, task_type: TaskType) -> List[Dict]:
        """搜索相关知识"""
        if not self.rag_system or not self.documents:
            return []

        try:
            # 执行混合检索
            results = self.rag_system.hybrid_search(
                query=query,
                semantic_index=self.semantic_index,
                tfidf_matrix=self.tfidf_matrix,
                documents=self.documents,
                top_k=self.config.get('rag_top_k', 5),
                alpha=0.7
            )

            self.stats["rag_searches"] += 1
            self.logger.info(f"检索到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            self.logger.error(f"知识检索失败: {e}")
            return []

    def create_prompt(self, user_input: str, task_type: TaskType, retrieved_docs: List[Dict]) -> str:
        """创建简化的提示"""

        # 基础提示模板
        base_prompt = f"""你是一名专业的IoT安全专家。请基于提供的信息回答用户的问题。

用户问题：{user_input}

任务类型：{task_type.value}
"""

        # 添加检索到的上下文
        if retrieved_docs:
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:], 1):  # 使用的文档数量
                source_type = "学术研究" if doc['source_type'] == 'academic' else "CVE漏洞数据"

                context_part = f"参考资料 {i} ({source_type}):\n"

                # 添加元数据
                if doc['source_type'] == 'cve':
                    metadata = doc.get('metadata', {})
                    context_part += f"CVE ID: {metadata.get('cve_id', 'N/A')}\n"
                    context_part += f"严重程度: {metadata.get('severity', 'N/A')}\n"
                elif doc['source_type'] == 'academic':
                    metadata = doc.get('metadata', {})
                    context_part += f"研究领域: {metadata.get('label', 'N/A')}\n"

                # 添加文档内容
                text = doc.get('text', '')
                preview = text[:400] + "..." if len(text) > 400 else text
                context_part += f"内容: {preview}\n"

                context_parts.append(context_part)

            context = "\n".join(context_parts)
            base_prompt += f"\n参考信息：\n{context}\n"
        else:
            base_prompt += "\n参考信息：无相关技术文档。\n"

        # 根据任务类型添加特定指导
        task_instructions = {
            TaskType.CVE_LOOKUP: """
请按照以下结构分析CVE漏洞：
1. 漏洞概述和基本信息
2. 技术细节和影响分析
3. 风险评估和严重程度
4. 缓解措施和建议""",

            TaskType.THREAT_DETECTION: """
请按照以下结构分析威胁：
1. 威胁识别和分类
2. 攻击向量和手段
3. 影响范围评估
4. 检测和应对策略""",

            TaskType.SECURITY_ASSESSMENT: """
请按照以下结构进行安全评估：
1. 当前安全状况分析
2. 潜在风险识别
3. 安全控制评估
4. 改进建议和优先级""",

            TaskType.PREVENTION_ADVICE: """
请按照以下结构提供防护建议：
1. 防护目标明确
2. 分层防护策略
3. 具体技术措施
4. 实施步骤和注意事项""",

            TaskType.RESEARCH_SUMMARY: """
请按照以下结构总结研究：
1. 研究背景和目标
2. 主要发现和创新点
3. 技术趋势分析
4. 实际应用价值""",

            TaskType.VULNERABILITY_ANALYSIS: """
请按照以下结构分析漏洞：
1. 漏洞原理和成因
2. 利用方式和条件
3. 影响评估
4. 修复和防护方案""",

            TaskType.GENERAL_QUERY: """
请提供专业、准确、实用的回答，包括：
1. 直接回答核心问题
2. 相关技术解释
3. 实践建议
4. 进一步学习资源"""
        }

        instruction = task_instructions.get(task_type, task_instructions[TaskType.GENERAL_QUERY])
        base_prompt += f"\n回答要求：{instruction}\n"

        base_prompt += """
请确保回答：
- 基于提供的参考信息
- 专业准确，易于理解
- 具有实际操作价值
- 适合IoT安全环境特点

现在请开始回答："""

        return base_prompt

    def generate_response(self, prompt: str, task_type: TaskType) -> OllamaResponse:
        """使用Ollama生成响应"""
        try:
            return self.ollama_client.generate(
                prompt=prompt,
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.7)
            )
        except Exception as e:
            self.logger.error(f"Ollama响应生成失败: {e}")
            raise

    def calculate_confidence(self, response: OllamaResponse, retrieved_docs: List[Dict],
                             task_type: TaskType) -> float:
        """计算响应置信度"""
        factors = []

        # 1. 基于检索文档质量 (40%)
        if retrieved_docs:
            avg_score = sum(doc.get('hybrid_score', 0) for doc in retrieved_docs) / len(retrieved_docs)
            factors.append(avg_score * 0.4)
        else:
            factors.append(0.2)  # 无检索文档时的基础分数

        # 2. 基于响应长度和结构 (30%)
        content_length = len(response.content)
        if content_length > 800:
            length_score = 0.9
        elif content_length > 400:
            length_score = 0.8
        elif content_length > 200:
            length_score = 0.7
        else:
            length_score = 0.5
        factors.append(length_score * 0.3)

        # 3. 基于内容质量指标 (20%)
        content = response.content.lower()
        quality_indicators = [
            ("建议" in content or "recommendation" in content),
            ("分析" in content or "analysis" in content),
            ("安全" in content or "security" in content),
            ("iot" in content or "物联网" in content),
            len(re.findall(r'[.。]', content)) > 5  # 句子数量
        ]
        quality_score = sum(quality_indicators) / len(quality_indicators)
        factors.append(quality_score * 0.2)

        # 4. 基于任务匹配度 (10%)
        task_keywords = {
            TaskType.CVE_LOOKUP: ["cve", "漏洞", "vulnerability", "cvss"],
            TaskType.THREAT_DETECTION: ["威胁", "攻击", "threat", "attack"],
            TaskType.SECURITY_ASSESSMENT: ["评估", "风险", "assessment", "risk"],
            TaskType.PREVENTION_ADVICE: ["防护", "建议", "prevention", "protect"],
            TaskType.RESEARCH_SUMMARY: ["研究", "总结", "research", "study"],
            TaskType.VULNERABILITY_ANALYSIS: ["分析", "原理", "analysis", "mechanism"]
        }

        keywords = task_keywords.get(task_type, [])
        if keywords:
            matches = sum(1 for keyword in keywords if keyword in content)
            task_match_score = min(matches / len(keywords), 1.0)
        else:
            task_match_score = 0.5
        factors.append(task_match_score * 0.1)

        confidence = sum(factors)
        return min(confidence, 0.95)  # 最大置信度限制

    def extract_recommendations(self, content: str) -> List[str]:
        """提取建议列表"""
        recommendations = []

        # 查找建议相关的模式
        patterns = [
            r'建议[:：]\s*(.+?)(?=\n\n|\n[0-9]|\n[一二三四五六七八九十]|$)',
            r'推荐[:：]\s*(.+?)(?=\n\n|\n[0-9]|\n[一二三四五六七八九十]|$)',
            r'措施[:：]\s*(.+?)(?=\n\n|\n[0-9]|\n[一二三四五六七八九十]|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # 分割成单独的建议项
                items = re.split(r'\n[0-9]\.|\n[一二三四五六七八九十]\.|\n[-*•]\s*', match.strip())
                for item in items:
                    clean_item = item.strip().replace('\n', ' ')
                    if clean_item and len(clean_item) > 10:
                        recommendations.append(clean_item)

        # 如果没找到明确的建议，查找编号列表
        if not recommendations:
            numbered_items = re.findall(r'^[0-9]+\.(.+)', content, re.MULTILINE)
            recommendations.extend([item.strip() for item in numbered_items if len(item.strip()) > 15])

        return recommendations[:8]  # 限制数量

    def prepare_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """准备源文档信息"""
        sources = []

        for doc in retrieved_docs[:10]:  # 最多10个源
            source = {
                "id": doc.get('id'),
                "type": doc.get('source_type'),
                "score": round(doc.get('hybrid_score', 0), 3),
                "preview": doc.get('text', '')[:120] + "..."
            }

            if doc.get('source_type') == 'cve':
                metadata = doc.get('metadata', {})
                source.update({
                    "cve_id": metadata.get('cve_id'),
                    "severity": metadata.get('severity'),
                    "cvss_score": metadata.get('base_score')
                })
            elif doc.get('source_type') == 'academic':
                metadata = doc.get('metadata', {})
                source.update({
                    "research_area": metadata.get('label'),
                    "year": metadata.get('year')
                })

            sources.append(source)

        return sources

    def process_query(self, user_input: str) -> SimpleAgentResponse:
        """处理用户查询的主流程"""
        start_time = time.time()

        try:
            self.stats["total_queries"] += 1
            self.logger.info(f"处理查询: {user_input}")

            # 1. 任务分类
            task_type = self.classify_task(user_input)
            self.logger.info(f"任务类型: {task_type.value}")

            # 2. 知识检索
            retrieved_docs = self.search_knowledge(user_input, task_type)

            # 3. 创建提示
            prompt = self.create_prompt(user_input, task_type, retrieved_docs)

            # 4. 生成响应
            ollama_response = self.generate_response(prompt, task_type)

            # 5. 计算置信度
            confidence = self.calculate_confidence(ollama_response, retrieved_docs, task_type)

            # 6. 提取建议
            recommendations = self.extract_recommendations(ollama_response.content)

            # 7. 准备源信息
            sources = self.prepare_sources(retrieved_docs)

            # 8. 创建最终响应
            processing_time = time.time() - start_time

            response = SimpleAgentResponse(
                content=ollama_response.content,
                confidence=confidence,
                sources=sources,
                task_type=task_type.value,
                processing_time=processing_time,
                model_used=ollama_response.model,
                metadata={
                    "rag_docs_found": len(retrieved_docs),
                    "recommendations_count": len(recommendations),
                    "tokens_used": ollama_response.tokens_used,
                    "llm_response_time": ollama_response.response_time,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # 9. 更新统计
            self.stats["successful_responses"] += 1

            # 更新平均置信度
            total_confidence = self.stats.get("total_confidence", 0) + confidence
            self.stats["total_confidence"] = total_confidence
            self.stats["average_confidence"] = total_confidence / self.stats["successful_responses"]

            self.logger.info(f"查询处理完成，耗时: {processing_time:.2f}s, 置信度: {confidence:.1%}")
            return response

        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")

            # 返回错误响应
            return SimpleAgentResponse(
                content=f"抱歉，处理您的查询时发生了错误：{str(e)}",
                confidence=0.0,
                sources=[],
                task_type="error",
                processing_time=time.time() - start_time,
                model_used="error",
                metadata={"error": str(e)}
            )

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        # 检查Ollama连接
        ollama_status = self.ollama_client.test_connection()

        return {
            "ollama": {
                "status": ollama_status["status"],
                "model": self.ollama_client.model_name,
                "available": self.ollama_client.is_available()
            },
            "rag": {
                "enabled": self.rag_system is not None,
                "documents_loaded": len(self.documents) if self.documents else 0,
                "searches_performed": self.stats["rag_searches"]
            },
            "performance": self.stats,
            "ollama_stats": self.ollama_client.get_stats()
        }

    def interactive_mode(self):
        """交互模式"""
        print("🤖 IoT安全智能助手 (Ollama版)")
        print("=" * 50)
        print("输入 'quit' 退出，'help' 查看帮助，'status' 查看状态")

        # 检查系统状态
        status = self.get_system_status()
        if not status["ollama"]["available"]:
            print("❌ Ollama服务不可用，请先启动Ollama服务")
            print("启动命令: ollama serve")
            return

        print(f"✅ 系统就绪 - 模型: {status['ollama']['model']}")

        if status["rag"]["enabled"]:
            print(f"📚 知识库: {status['rag']['documents_loaded']} 个文档")
        else:
            print("⚠️ 知识库未加载，将在基础模式下运行")

        while True:
            try:
                user_input = input("\n🔍 请输入您的问题: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用IoT安全智能助手！")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower() == 'status':
                    self._show_status()
                    continue

                if not user_input:
                    continue

                print("\n🤔 正在分析和生成回答...")

                # 处理查询
                response = self.process_query(user_input)

                # 显示回答
                self._display_response(response)

            except KeyboardInterrupt:
                print("\n👋 感谢使用IoT安全智能助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

    def _show_help(self):
        """显示帮助"""
        help_text = """
🆘 IoT安全智能助手帮助
===================

🎯 主要功能:
• CVE漏洞分析: "分析CVE-2023-0001的影响"
• 威胁检测: "IoT设备异常网络行为分析"  
• 安全评估: "评估智能家居安全风险"
• 防护建议: "如何防护IoT设备DDoS攻击"
• 研究总结: "IoT加密技术最新进展"

💡 使用技巧:
• 使用具体的技术术语
• 描述详细的场景信息
• 支持中英文查询
• 可以要求step-by-step分析

🔧 命令:
• help: 显示帮助
• status: 查看系统状态  
• quit: 退出系统

📊 示例查询:
• "什么是IoT安全？"
• "智能摄像头有哪些安全漏洞？"
• "如何保护工业IoT网络？"
        """
        print(help_text)

    def _show_status(self):
        """显示系统状态"""
        status = self.get_system_status()

        print("\n📊 系统状态")
        print("=" * 30)

        # Ollama状态
        ollama = status["ollama"]
        print(f"🤖 Ollama:")
        print(f"  状态: {'✅ 可用' if ollama['available'] else '❌ 不可用'}")
        print(f"  模型: {ollama['model']}")

        # RAG状态
        rag = status["rag"]
        print(f"📚 知识库:")
        print(f"  状态: {'✅ 已加载' if rag['enabled'] else '❌ 未加载'}")
        print(f"  文档数: {rag['documents_loaded']}")
        print(f"  检索次数: {rag['searches_performed']}")

        # 性能统计
        perf = status["performance"]
        print(f"📈 性能统计:")
        print(f"  总查询数: {perf['total_queries']}")
        print(f"  成功响应: {perf['successful_responses']}")
        print(f"  平均置信度: {perf['average_confidence']:.1%}")

        # Ollama统计
        ollama_stats = status["ollama_stats"]
        print(f"🔧 Ollama统计:")
        for key, value in ollama_stats.items():
            print(f"  {key}: {value}")

    def _display_response(self, response: SimpleAgentResponse):
        """显示响应"""
        print("\n" + "=" * 80)
        print("🤖 IoT安全专家回答")
        print("=" * 80)

        # 显示元信息
        print(f"🎯 任务类型: {response.task_type}")
        print(
            f"📊 置信度: {'█' * int(response.confidence * 10)}{'░' * (10 - int(response.confidence * 10))} {response.confidence:.1%}")
        print(f"🤖 使用模型: {response.model_used}")
        print(f"⏱️ 处理时间: {response.processing_time:.2f}s")
        print(f"📄 Token使用: {response.metadata.get('tokens_used', 'N/A')}")

        # 显示主要回答
        print(f"\n📝 详细分析:")
        print("-" * 60)
        print(response.content)

        # 显示参考源
        if response.sources:
            print(f"\n📚 参考来源 ({len(response.sources)} 个):")
            print("-" * 40)
            for i, source in enumerate(response.sources[:3], 1):
                source_type = "📚 学术研究" if source['type'] == 'academic' else "🚨 CVE漏洞"
                print(f"{i}. {source_type} [相关度: {source['score']:.3f}]")

                if source['type'] == 'cve' and source.get('cve_id'):
                    print(f"   🆔 CVE ID: {source['cve_id']}")
                elif source['type'] == 'academic' and source.get('research_area'):
                    print(f"   📖 研究领域: {source['research_area']}")

                print(f"   📄 内容: {source['preview']}")


# 使用示例和测试
def demo_simple_agent():
    """演示简化Agent"""
    print("🚀 简化RAG+Ollama Agent演示")
    print("=" * 50)

    # 创建Agent
    config = {
        "model_name": "llama3.1:7b",  # 可以改为其他模型
        "max_tokens": 1500,
        "temperature": 0.7,
        "use_rag": True,
        "rag_top_k": 3
    }

    try:
        agent = SimpleRAGAgent(config)

        # 检查系统状态
        status = agent.get_system_status()
        print(f"系统状态:")
        print(f"  Ollama: {'✅' if status['ollama']['available'] else '❌'}")
        print(f"  模型: {status['ollama']['model']}")
        print(f"  知识库: {'✅' if status['rag']['enabled'] else '❌'}")

        if not status['ollama']['available']:
            print("\n❌ Ollama不可用，请先启动服务:")
            print("  ollama serve")
            print("  ollama pull llama3.1:7b")
            return

        # 测试查询
        test_queries = [
            "什么是IoT安全？",
            "智能家居设备有哪些常见漏洞？",
            "如何防护IoT设备免受DDoS攻击？"
        ]

        print(f"\n💬 测试查询:")
        for query in test_queries:
            print(f"\n📝 查询: {query}")
            print("-" * 50)

            try:
                response = agent.process_query(query)

                print(f"✅ 任务类型: {response.task_type}")
                print(f"📊 置信度: {response.confidence:.1%}")
                print(f"⏱️ 耗时: {response.processing_time:.2f}s")
                print(f"📚 使用源: {len(response.sources)} 个")

                # 显示回答预览
                preview = response.content[:200] + "..." if len(response.content) > 200 else response.content
                print(f"📄 回答预览: {preview}")

            except Exception as e:
                print(f"❌ 查询失败: {e}")

        # 启动交互模式
        print(f"\n🎮 启动交互模式...")
        agent.interactive_mode()

    except Exception as e:
        print(f"❌ 演示失败: {e}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行演示
    demo_simple_agent()