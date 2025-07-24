# prompt_templates.py - IoT安全领域提示模板库
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime
from abc import ABC, abstractmethod


class PromptType(Enum):
    """提示类型枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class TaskComplexity(Enum):
    """任务复杂度枚举"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class PromptTemplate:
    """提示模板数据结构"""
    name: str
    description: str
    template: str
    variables: List[str]
    task_type: str
    complexity: TaskComplexity
    examples: List[Dict] = None
    constraints: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.constraints is None:
            self.constraints = []


class IoTSecurityPromptLibrary:
    """IoT安全领域提示模板库"""

    def __init__(self):
        self.templates = {}
        self.examples_db = {}
        self._initialize_templates()
        self._initialize_examples()

    def _initialize_templates(self):
        """初始化提示模板"""

        # CVE分析模板
        self.templates["cve_analysis"] = PromptTemplate(
            name="CVE漏洞分析",
            description="专门用于分析CVE漏洞的详细信息",
            template="""你是一名专业的IoT安全专家，请基于以下信息对CVE漏洞进行专业分析：

## 分析目标
CVE ID: {cve_id}
漏洞描述: {description}
CVSS评分: {cvss_score}
严重程度: {severity}
影响设备: {affected_devices}
漏洞类型: {vulnerability_types}

## 分析要求
请按照以下结构进行分析：

### 1. 漏洞概述
- 简要描述漏洞的本质和触发条件
- 解释技术细节和根本原因

### 2. 影响分析
- 评估对IoT设备的具体影响
- 分析攻击者可能的利用方式
- 估算影响范围和严重程度

### 3. 技术分析
- 详细解释漏洞的技术机制
- 分析攻击向量和利用路径
- 评估攻击复杂度和可检测性

### 4. 风险评估
- 基于CVSS评分分析风险等级
- 考虑在IoT环境中的特殊风险
- 评估业务影响和数据安全风险

### 5. 缓解措施
- 提供立即可执行的临时措施
- 建议长期的安全改进方案
- 给出检测和监控建议

## 输出格式
请使用结构化的Markdown格式，确保内容专业、准确、actionable。

## 参考信息
{retrieved_context}
""",
            variables=["cve_id", "description", "cvss_score", "severity", "affected_devices", "vulnerability_types",
                       "retrieved_context"],
            task_type="cve_lookup",
            complexity=TaskComplexity.EXPERT,
            constraints=[
                "必须基于技术事实进行分析",
                "避免过度技术术语，保持专业但易懂",
                "提供具体可行的建议",
                "结合IoT环境特点进行分析"
            ]
        )

        # 威胁检测模板
        self.templates["threat_detection"] = PromptTemplate(
            name="IoT威胁检测分析",
            description="用于检测和分析IoT环境中的安全威胁",
            template="""作为IoT安全威胁分析专家，请对以下威胁进行全面分析：

## 威胁信息
威胁描述: {threat_description}
目标环境: {target_environment}
观察到的行为: {observed_behavior}

## 分析框架

### 1. 威胁识别
- 威胁类型分类
- 攻击手段识别
- 威胁源分析

### 2. 攻击路径分析
运用MITRE ATT&CK框架分析：
- Initial Access (初始访问)
- Execution (执行)
- Persistence (持久化)
- Lateral Movement (横向移动)
- Impact (影响)

### 3. IoT特定风险评估
考虑IoT环境的特殊性：
- 设备资源限制
- 网络通信特点
- 物理访问风险
- 大规模部署影响

### 4. 检测策略
- 技术检测指标
- 行为异常模式
- 网络流量特征
- 设备状态监控

### 5. 响应建议
- 即时响应措施
- 隔离和遏制策略
- 恢复和修复计划
- 预防再次发生的措施

## 推理过程
请详细说明你的分析思路和判断依据。

## 参考资料
{retrieved_context}
""",
            variables=["threat_description", "target_environment", "observed_behavior", "retrieved_context"],
            task_type="threat_detection",
            complexity=TaskComplexity.COMPLEX
        )

        # 安全评估模板
        self.templates["security_assessment"] = PromptTemplate(
            name="IoT安全评估",
            description="对IoT系统进行全面的安全评估",
            template="""作为IoT安全评估专家，请对以下系统进行专业的安全评估：

## 评估对象
系统类型: {system_type}
部署环境: {deployment_environment}
设备规模: {device_scale}
业务关键性: {business_criticality}

## 评估维度

### 1. 设备安全
- 硬件安全特性
- 固件安全性
- 身份认证机制
- 访问控制实现

### 2. 通信安全
- 数据传输加密
- 协议安全性
- 网络架构安全
- 密钥管理

### 3. 数据安全
- 数据分类和标记
- 存储安全
- 隐私保护
- 数据生命周期管理

### 4. 运营安全
- 安全监控能力
- 事件响应机制
- 补丁管理流程
- 供应链安全

### 5. 合规性评估
- 相关法规要求
- 行业标准符合性
- 安全框架对标

## 评估方法
请使用以下方法进行评估：
1. 基于知识库的最佳实践对比
2. 威胁建模和风险分析
3. 控制措施有效性评估
4. 残余风险评估

## 输出要求
- 风险等级评定（高/中/低）
- 具体风险清单
- 优先级改进建议
- 实施路线图

## 知识库信息
{retrieved_context}
""",
            variables=["system_type", "deployment_environment", "device_scale", "business_criticality",
                       "retrieved_context"],
            task_type="security_assessment",
            complexity=TaskComplexity.COMPLEX
        )

        # 防护建议模板
        self.templates["prevention_advice"] = PromptTemplate(
            name="IoT安全防护建议",
            description="为IoT环境提供专业的安全防护建议",
            template="""作为IoT安全防护专家，请为以下情况提供专业的防护建议：

## 防护需求
安全问题: {security_issue}
防护目标: {protection_target}
环境约束: {environment_constraints}
资源限制: {resource_limitations}

## 防护策略框架

### 1. 分层防护策略
**设备层防护:**
- 硬件安全模块(HSM)
- 安全启动机制
- 固件完整性保护
- 设备身份认证

**网络层防护:**
- 网络分段和隔离
- 流量监控和过滤
- 入侵检测系统
- VPN和加密通道

**应用层防护:**
- 应用安全开发
- API安全保护
- 数据加密存储
- 访问权限控制

**管理层防护:**
- 安全策略制定
- 人员安全培训
- 供应链安全管理
- 应急响应计划

### 2. 技术实施建议
请提供具体的技术实施方案：
- 推荐的安全产品和工具
- 配置参数和部署方案
- 集成和兼容性考虑
- 性能影响评估

### 3. 成本效益分析
- 实施成本估算
- 风险降低效果
- ROI分析
- 分阶段实施建议

### 4. 实施路线图
- 短期措施（1-3个月）
- 中期规划（3-12个月）
- 长期战略（1-3年）
- 关键里程碑

## 最佳实践参考
{retrieved_context}
""",
            variables=["security_issue", "protection_target", "environment_constraints", "resource_limitations",
                       "retrieved_context"],
            task_type="prevention_advice",
            complexity=TaskComplexity.MODERATE
        )

        # 研究总结模板
        self.templates["research_summary"] = PromptTemplate(
            name="IoT安全研究总结",
            description="总结IoT安全领域的研究成果和趋势",
            template="""作为IoT安全研究专家，请对以下研究主题进行专业总结：

## 研究主题
研究领域: {research_topic}
时间范围: {time_range}
关注重点: {focus_areas}

## 总结框架

### 1. 研究概览
- 研究背景和动机
- 主要研究问题
- 研究方法和途径

### 2. 核心发现
**技术创新:**
- 新技术和方法
- 算法和模型改进
- 工具和框架发展

**实证研究:**
- 实验结果和数据
- 性能评估和比较
- 案例研究分析

**理论贡献:**
- 理论模型和框架
- 概念定义和分类
- 原理机制解释

### 3. 技术趋势分析
- 新兴技术方向
- 成熟技术应用
- 技术融合趋势
- 标准化进展

### 4. 挑战和机遇
**当前挑战:**
- 技术局限性
- 实施障碍
- 成本和复杂性问题

**未来机遇:**
- 技术发展空间
- 市场需求驱动
- 政策支持方向

### 5. 实践应用
- 工业应用案例
- 最佳实践总结
- 实施经验教训
- 应用前景展望

### 6. 研究方向建议
- 亟需解决的问题
- 有前景的研究方向
- 跨学科合作机会
- 产学研结合建议

## 学术资料参考
{retrieved_context}
""",
            variables=["research_topic", "time_range", "focus_areas", "retrieved_context"],
            task_type="research_summary",
            complexity=TaskComplexity.EXPERT
        )

        # 一般查询模板
        self.templates["general_query"] = PromptTemplate(
            name="IoT安全一般查询",
            description="处理IoT安全领域的一般性查询",
            template="""作为IoT安全专家，请回答以下问题：

## 用户问题
{user_question}

## 回答要求
请提供专业、准确、实用的回答，包括：

### 1. 直接回答
- 针对问题的核心回答
- 关键概念和术语解释
- 重要注意事项

### 2. 技术细节
- 相关技术原理
- 实施方法和步骤
- 工具和资源推荐

### 3. 实践建议
- 最佳实践指导
- 常见问题避免
- 进一步学习资源

### 4. 案例举例
- 相关应用案例
- 成功经验分享
- 失败教训总结

## 回答风格
- 保持专业性和准确性
- 使用清晰易懂的语言
- 提供actionable的建议
- 结合最新技术发展

## 参考信息
{retrieved_context}
""",
            variables=["user_question", "retrieved_context"],
            task_type="general_query",
            complexity=TaskComplexity.SIMPLE
        )

    def _initialize_examples(self):
        """初始化Few-shot示例库"""

        # CVE分析示例
        self.examples_db["cve_analysis"] = [
            {
                "input": {
                    "cve_id": "CVE-2023-0001",
                    "description": "Buffer overflow vulnerability in IoT device firmware",
                    "cvss_score": "7.5",
                    "severity": "High"
                },
                "output": """## CVE-2023-0001 漏洞分析

### 1. 漏洞概述
这是一个影响IoT设备固件的缓冲区溢出漏洞，攻击者可以通过构造恶意数据包触发溢出，可能导致远程代码执行。

### 2. 影响分析
- **直接影响**: 设备完全失控，可被远程操作
- **间接影响**: 可能成为僵尸网络节点，影响网络安全
- **业务影响**: 服务中断，数据泄露风险

### 3. 技术分析
缓冲区溢出发生在网络数据包处理模块，由于缺乏边界检查导致栈溢出，攻击者可以覆盖返回地址实现代码执行。

### 4. 风险评估
CVSS 7.5分属于高风险漏洞，在IoT环境中风险更高因为：
- 设备通常缺乏安全更新机制
- 大规模部署难以快速修复
- 可能影响关键基础设施

### 5. 缓解措施
**立即措施**:
- 隔离受影响设备
- 实施网络访问控制
- 监控异常网络流量

**长期措施**:
- 部署固件更新
- 实施输入验证
- 启用栈保护机制"""
            }
        ]

        # 威胁检测示例
        self.examples_db["threat_detection"] = [
            {
                "input": {
                    "threat_description": "IoT设备异常网络连接",
                    "observed_behavior": "大量对外连接，数据传输异常"
                },
                "output": """## IoT设备异常连接威胁分析

### 1. 威胁识别
- **威胁类型**: 可能的僵尸网络感染或数据窃取
- **攻击手段**: 恶意软件感染或远程控制
- **威胁等级**: 高

### 2. 攻击路径分析
**Initial Access**: 利用设备漏洞或弱认证
**Execution**: 植入恶意代码
**Command and Control**: 建立C2通信
**Exfiltration**: 数据外传或参与攻击

### 3. 检测策略
- 监控网络流量基线偏差
- 检测未授权外部连接
- 分析数据传输模式异常
- 监控设备行为变化

### 4. 响应建议
**立即隔离**: 断开网络连接
**深度分析**: 检查设备固件完整性
**清理修复**: 重置设备或更新固件
**预防措施**: 加强网络监控和访问控制"""
            }
        ]

    def get_template(self, task_type: str) -> Optional[PromptTemplate]:
        """获取指定任务类型的模板"""
        return self.templates.get(task_type)

    def get_examples(self, task_type: str, num_examples: int = 2) -> List[Dict]:
        """获取指定任务类型的示例"""
        examples = self.examples_db.get(task_type, [])
        return examples[:num_examples]

    def list_templates(self) -> Dict[str, str]:
        """列出所有可用模板"""
        return {name: template.description for name, template in self.templates.items()}


class ChainOfThoughtPromptBuilder:
    """思维链提示构建器"""

    def __init__(self):
        self.reasoning_patterns = {
            "analysis": [
                "首先，让我分析问题的核心要素",
                "接下来，我需要考虑相关的技术因素",
                "然后，我将评估潜在的风险和影响",
                "最后，我会提供具体的建议和解决方案"
            ],
            "investigation": [
                "让我逐步调查这个安全问题",
                "第一步：收集和分析相关信息",
                "第二步：识别可能的攻击向量",
                "第三步：评估威胁的严重程度",
                "第四步：制定相应的防护措施"
            ],
            "problem_solving": [
                "让我系统性地解决这个问题",
                "问题分解：将复杂问题分解为子问题",
                "方案评估：分析各种可能的解决方案",
                "最优选择：选择最适合的解决方案",
                "实施规划：制定具体的实施计划"
            ]
        }

    def build_cot_prompt(self, base_prompt: str, reasoning_type: str = "analysis") -> str:
        """构建思维链提示"""
        cot_instruction = f"""
在回答之前，请按照以下思考步骤：

{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_patterns.get(reasoning_type, self.reasoning_patterns["analysis"])))}

现在，请按照上述思考步骤来处理以下问题：

{base_prompt}

请在回答中明确展示你的推理过程。
"""
        return cot_instruction

    def build_step_by_step_prompt(self, task_description: str, steps: List[str]) -> str:
        """构建分步骤提示"""
        step_prompt = f"""
请按照以下步骤完成任务：{task_description}

具体步骤：
{chr(10).join(f"步骤 {i + 1}: {step}" for i, step in enumerate(steps))}

请按顺序执行每个步骤，并在回答中说明每个步骤的执行结果。
"""
        return step_prompt


class FewShotPromptBuilder:
    """Few-shot学习提示构建器"""

    def __init__(self, prompt_library: IoTSecurityPromptLibrary):
        self.prompt_library = prompt_library

    def build_few_shot_prompt(self, task_type: str, user_input: str,
                              num_examples: int = 2) -> str:
        """构建few-shot提示"""

        # 获取示例
        examples = self.prompt_library.get_examples(task_type, num_examples)

        if not examples:
            return user_input

        # 构建few-shot提示
        few_shot_prompt = "以下是一些示例，展示了如何处理类似的IoT安全问题：\n\n"

        for i, example in enumerate(examples, 1):
            few_shot_prompt += f"## 示例 {i}\n"
            few_shot_prompt += f"**输入**: {example['input']}\n"
            few_shot_prompt += f"**输出**: {example['output']}\n\n"

        few_shot_prompt += f"现在，请按照上述示例的格式和质量标准，处理以下问题：\n\n{user_input}"

        return few_shot_prompt


class PromptOptimizer:
    """提示优化器"""

    def __init__(self):
        self.optimization_history = []
        self.performance_metrics = {}

    def optimize_prompt_length(self, prompt: str, max_length: int = 2000) -> str:
        """优化提示长度"""
        if len(prompt) <= max_length:
            return prompt

        # 保留重要部分，压缩次要内容
        lines = prompt.split('\n')
        essential_lines = []
        optional_lines = []

        for line in lines:
            if any(keyword in line.lower() for keyword in ['##', '###', '要求', '分析', '建议']):
                essential_lines.append(line)
            else:
                optional_lines.append(line)

        # 先加入重要内容
        optimized_prompt = '\n'.join(essential_lines)

        # 逐步添加可选内容
        for line in optional_lines:
            if len(optimized_prompt) + len(line) + 1 <= max_length:
                optimized_prompt += '\n' + line
            else:
                break

        return optimized_prompt

    def add_contextual_constraints(self, prompt: str, constraints: List[str]) -> str:
        """添加上下文约束"""
        if not constraints:
            return prompt

        constraints_text = "\n## 重要约束条件\n"
        constraints_text += "\n".join(f"- {constraint}" for constraint in constraints)

        return prompt + "\n" + constraints_text

    def enhance_clarity(self, prompt: str) -> str:
        """增强提示清晰度"""
        # 添加结构化标记
        enhanced_prompt = prompt

        # 确保有明确的任务说明
        if "## 任务目标" not in enhanced_prompt:
            task_section = "\n## 任务目标\n请提供专业、准确、actionable的IoT安全分析和建议。\n"
            enhanced_prompt = task_section + enhanced_prompt

        # 确保有输出格式要求
        if "## 输出格式" not in enhanced_prompt:
            format_section = "\n## 输出格式\n请使用结构化的Markdown格式，包含清晰的标题和要点。\n"
            enhanced_prompt += format_section

        return enhanced_prompt


class PromptTemplateManager:
    """提示模板管理器"""

    def __init__(self):
        self.prompt_library = IoTSecurityPromptLibrary()
        self.cot_builder = ChainOfThoughtPromptBuilder()
        self.few_shot_builder = FewShotPromptBuilder(self.prompt_library)
        self.optimizer = PromptOptimizer()

    def generate_prompt(self, task_type: str, user_input: str,
                        context: Dict = None, use_cot: bool = True,
                        use_few_shot: bool = True, num_examples: int = 1) -> str:
        """生成完整的提示"""

        # 获取基础模板
        template = self.prompt_library.get_template(task_type)
        if not template:
            template = self.prompt_library.get_template("general_query")

        # 准备变量
        variables = {}
        if context:
            variables.update(context)

        # 填充模板变量
        try:
            base_prompt = template.template.format(**variables)
        except KeyError as e:
            # 如果变量缺失，使用默认值
            for var in template.variables:
                if var not in variables:
                    variables[var] = f"[{var}]"
            base_prompt = template.template.format(**variables)

        # 应用Few-shot学习
        if use_few_shot and num_examples > 0:
            base_prompt = self.few_shot_builder.build_few_shot_prompt(
                task_type, base_prompt, num_examples
            )

        # 应用思维链
        if use_cot:
            reasoning_type = "analysis" if task_type in ["cve_lookup", "vulnerability_analysis"] else "investigation"
            base_prompt = self.cot_builder.build_cot_prompt(base_prompt, reasoning_type)

        # 优化提示
        optimized_prompt = self.optimizer.enhance_clarity(base_prompt)

        # 添加约束条件
        if template.constraints:
            optimized_prompt = self.optimizer.add_contextual_constraints(
                optimized_prompt, template.constraints
            )

        # 长度优化
        final_prompt = self.optimizer.optimize_prompt_length(optimized_prompt)

        return final_prompt

    def evaluate_prompt_effectiveness(self, prompt: str, response: str,
                                      expected_quality: float) -> Dict:
        """评估提示效果"""
        metrics = {
            "prompt_length": len(prompt),
            "response_length": len(response),
            "structure_score": self._evaluate_structure(response),
            "clarity_score": self._evaluate_clarity(response),
            "completeness_score": self._evaluate_completeness(response)
        }

        metrics["overall_score"] = (
                metrics["structure_score"] * 0.3 +
                metrics["clarity_score"] * 0.3 +
                metrics["completeness_score"] * 0.4
        )

        return metrics

    def _evaluate_structure(self, response: str) -> float:
        """评估响应结构性"""
        # 检查标题数量
        headers = len(re.findall(r'^#{1,6}\s', response, re.MULTILINE))
        # 检查列表使用
        lists = len(re.findall(r'^\s*[-*+]\s', response, re.MULTILINE))

        structure_score = min((headers * 0.1 + lists * 0.05), 1.0)
        return structure_score

    def _evaluate_clarity(self, response: str) -> float:
        """评估响应清晰度"""
        # 简单的清晰度评估：句子长度、段落结构等
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # 理想句子长度约15-25词
        clarity_score = max(0, 1 - abs(avg_sentence_length - 20) / 20)
        return clarity_score

    def _evaluate_completeness(self, response: str) -> float:
        """评估响应完整性"""
        # 检查关键部分是否存在
        key_elements = [
            "分析", "建议", "措施", "风险", "影响"
        ]

        found_elements = sum(1 for element in key_elements if element in response)
        completeness_score = found_elements / len(key_elements)

        return completeness_score

    def get_template_stats(self) -> Dict:
        """获取模板统计信息"""
        stats = {
            "total_templates": len(self.prompt_library.templates),
            "templates_by_complexity": {},
            "templates_by_task": {}
        }

        for template in self.prompt_library.templates.values():
            # 按复杂度统计
            complexity = template.complexity.value
            stats["templates_by_complexity"][complexity] = \
                stats["templates_by_complexity"].get(complexity, 0) + 1

            # 按任务类型统计
            task_type = template.task_type
            stats["templates_by_task"][task_type] = \
                stats["templates_by_task"].get(task_type, 0) + 1

        return stats


# 使用示例和测试
def demo_prompt_engineering():
    """演示提示工程功能"""
    print("🎨 IoT安全Prompt Engineering系统演示")
    print("=" * 50)

    # 初始化管理器
    manager = PromptTemplateManager()

    # 演示不同类型的提示生成
    test_cases = [
        {
            "task_type": "cve_lookup",
            "user_input": "分析CVE-2023-0001的安全影响",
            "context": {
                "cve_id": "CVE-2023-0001",
                "description": "Buffer overflow in IoT firmware",
                "cvss_score": "7.5",
                "severity": "High",
                "affected_devices": "IoT routers",
                "vulnerability_types": "buffer_overflow",
                "retrieved_context": "相关技术文档和分析报告..."
            }
        },
        {
            "task_type": "threat_detection",
            "user_input": "检测IoT设备的异常网络行为",
            "context": {
                "threat_description": "设备频繁外联",
                "target_environment": "智能家居",
                "observed_behavior": "大量DNS查询和数据传输",
                "retrieved_context": "网络监控数据和威胁情报..."
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 测试案例 {i}: {test_case['task_type']}")
        print("-" * 40)

        # 生成基础提示
        basic_prompt = manager.generate_prompt(
            task_type=test_case["task_type"],
            user_input=test_case["user_input"],
            context=test_case["context"],
            use_cot=False,
            use_few_shot=False
        )

        # 生成增强提示（CoT + Few-shot）
        enhanced_prompt = manager.generate_prompt(
            task_type=test_case["task_type"],
            user_input=test_case["user_input"],
            context=test_case["context"],
            use_cot=True,
            use_few_shot=True,
            num_examples=1
        )

        print(f"📝 基础提示长度: {len(basic_prompt)} 字符")
        print(f"✨ 增强提示长度: {len(enhanced_prompt)} 字符")
        print(f"📈 增强幅度: {((len(enhanced_prompt) - len(basic_prompt)) / len(basic_prompt) * 100):.1f}%")

        # 显示提示预览
        print(f"\n📄 增强提示预览:")
        preview = enhanced_prompt[:300] + "..." if len(enhanced_prompt) > 300 else enhanced_prompt
        print(preview)

    # 显示模板统计
    stats = manager.get_template_stats()
    print(f"\n📊 模板库统计:")
    print(f"总模板数: {stats['total_templates']}")
    print(f"复杂度分布: {stats['templates_by_complexity']}")
    print(f"任务类型分布: {stats['templates_by_task']}")


# prompt_evaluator.py - 提示评估系统
class PromptEvaluator:
    """提示效果评估器"""

    def __init__(self):
        self.evaluation_criteria = {
            "clarity": {
                "weight": 0.25,
                "description": "提示的清晰度和易理解性"
            },
            "specificity": {
                "weight": 0.25,
                "description": "提示的具体性和针对性"
            },
            "completeness": {
                "weight": 0.25,
                "description": "提示的完整性和全面性"
            },
            "effectiveness": {
                "weight": 0.25,
                "description": "提示的效果和实用性"
            }
        }

    def evaluate_prompt_quality(self, prompt: str, task_type: str) -> Dict:
        """评估提示质量"""
        scores = {}

        # 清晰度评估
        scores["clarity"] = self._evaluate_clarity(prompt)

        # 具体性评估
        scores["specificity"] = self._evaluate_specificity(prompt, task_type)

        # 完整性评估
        scores["completeness"] = self._evaluate_completeness(prompt)

        # 效果性评估
        scores["effectiveness"] = self._evaluate_effectiveness(prompt)

        # 计算总分
        total_score = sum(
            scores[criterion] * self.evaluation_criteria[criterion]["weight"]
            for criterion in scores
        )

        return {
            "total_score": total_score,
            "detailed_scores": scores,
            "recommendations": self._generate_improvement_recommendations(scores)
        }

    def _evaluate_clarity(self, prompt: str) -> float:
        """评估清晰度"""
        clarity_indicators = [
            ("结构化标题", len(re.findall(r'^#{1,6}\s', prompt, re.MULTILINE)) > 0),
            ("明确指令", any(word in prompt for word in ["请", "需要", "要求", "分析"])),
            ("分步骤", "步骤" in prompt or "第一" in prompt),
            ("示例说明", "示例" in prompt or "例如" in prompt)
        ]

        clarity_score = sum(indicator[1] for indicator in clarity_indicators) / len(clarity_indicators)
        return clarity_score

    def _evaluate_specificity(self, prompt: str, task_type: str) -> float:
        """评估具体性"""
        task_specific_keywords = {
            "cve_lookup": ["CVE", "漏洞", "CVSS", "影响", "缓解"],
            "threat_detection": ["威胁", "攻击", "检测", "行为", "异常"],
            "security_assessment": ["评估", "风险", "安全", "控制", "合规"],
            "prevention_advice": ["防护", "建议", "措施", "策略", "实施"]
        }

        keywords = task_specific_keywords.get(task_type, [])
        if not keywords:
            return 0.5  # 默认分数

        found_keywords = sum(1 for keyword in keywords if keyword in prompt)
        specificity_score = found_keywords / len(keywords)

        return min(specificity_score, 1.0)

    def _evaluate_completeness(self, prompt: str) -> float:
        """评估完整性"""
        essential_components = [
            ("任务描述", any(word in prompt for word in ["任务", "目标", "要求"])),
            ("输入信息", "{" in prompt and "}" in prompt),
            ("输出格式", any(word in prompt for word in ["格式", "结构", "输出"])),
            ("约束条件", any(word in prompt for word in ["约束", "限制", "注意"])),
            ("参考信息", any(word in prompt for word in ["参考", "基于", "信息"]))
        ]

        completeness_score = sum(component[1] for component in essential_components) / len(essential_components)
        return completeness_score

    def _evaluate_effectiveness(self, prompt: str) -> float:
        """评估效果性"""
        effectiveness_factors = [
            ("actionable指令", any(word in prompt for word in ["具体", "详细", "actionable"])),
            ("专业术语", any(word in prompt for word in ["专业", "技术", "安全"])),
            ("实践导向", any(word in prompt for word in ["实施", "应用", "实践"])),
            ("质量要求", any(word in prompt for word in ["准确", "可靠", "有效"]))
        ]

        effectiveness_score = sum(factor[1] for factor in effectiveness_factors) / len(effectiveness_factors)
        return effectiveness_score

    def _generate_improvement_recommendations(self, scores: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if scores["clarity"] < 0.7:
            recommendations.append("增加结构化标题和分步骤说明")

        if scores["specificity"] < 0.7:
            recommendations.append("添加更多任务特定的关键词和要求")

        if scores["completeness"] < 0.7:
            recommendations.append("补充缺失的组件（输出格式、约束条件等）")

        if scores["effectiveness"] < 0.7:
            recommendations.append("增强实践导向和actionable指令")

        return recommendations


# prompt_optimizer_advanced.py - 高级提示优化器
class AdvancedPromptOptimizer:
    """高级提示优化器"""

    def __init__(self):
        self.optimization_strategies = {
            "length_optimization": self._optimize_length,
            "structure_enhancement": self._enhance_structure,
            "clarity_improvement": self._improve_clarity,
            "context_integration": self._integrate_context
        }

    def optimize_prompt(self, prompt: str, optimization_goals: List[str] = None) -> str:
        """优化提示"""
        if optimization_goals is None:
            optimization_goals = list(self.optimization_strategies.keys())

        optimized_prompt = prompt

        for goal in optimization_goals:
            if goal in self.optimization_strategies:
                optimized_prompt = self.optimization_strategies[goal](optimized_prompt)

        return optimized_prompt

    def _optimize_length(self, prompt: str) -> str:
        """优化长度"""
        # 移除多余的空行
        lines = [line for line in prompt.split('\n') if line.strip()]

        # 合并相似的段落
        optimized_lines = []
        current_section = []

        for line in lines:
            if line.startswith('#') and current_section:
                # 新的章节开始，处理当前章节
                if current_section:
                    optimized_lines.extend(self._compress_section(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # 处理最后一个章节
        if current_section:
            optimized_lines.extend(self._compress_section(current_section))

        return '\n'.join(optimized_lines)

    def _compress_section(self, section_lines: List[str]) -> List[str]:
        """压缩章节内容"""
        if len(section_lines) <= 3:
            return section_lines

        # 保留标题和关键内容
        compressed = [section_lines[0]]  # 标题

        # 合并内容行
        content_lines = section_lines[1:]
        if content_lines:
            # 保留要点，合并描述
            key_points = [line for line in content_lines if line.strip().startswith('-')]
            descriptions = [line for line in content_lines if not line.strip().startswith('-') and line.strip()]

            if descriptions:
                compressed.append(' '.join(descriptions))
            compressed.extend(key_points)

        return compressed

    def _enhance_structure(self, prompt: str) -> str:
        """增强结构"""
        lines = prompt.split('\n')
        enhanced_lines = []

        current_level = 0
        for line in lines:
            stripped = line.strip()

            if stripped.startswith('#'):
                # 标题行
                level = len(stripped.split()[0])
                if level > current_level + 1:
                    # 添加中间级别标题
                    for i in range(current_level + 1, level):
                        enhanced_lines.append('#' * i + ' 分析要点')
                current_level = level
                enhanced_lines.append(line)
            elif stripped.startswith('-') or stripped.startswith('*'):
                # 列表项
                enhanced_lines.append(line)
            elif stripped:
                # 普通内容，确保适当的层级
                enhanced_lines.append(line)
            else:
                # 空行
                enhanced_lines.append(line)

        return '\n'.join(enhanced_lines)

    def _improve_clarity(self, prompt: str) -> str:
        """改进清晰度"""
        # 添加明确的操作指令
        clarity_markers = [
            ("请分析", "请详细分析"),
            ("需要", "需要重点关注"),
            ("应该", "建议采用"),
            ("可以", "可以考虑")
        ]

        improved_prompt = prompt
        for original, replacement in clarity_markers:
            improved_prompt = improved_prompt.replace(original, replacement)

        # 添加结果期望说明
        if "输出要求" not in improved_prompt:
            output_requirement = "\n## 输出要求\n请提供结构化、专业、actionable的分析结果。\n"
            improved_prompt += output_requirement

        return improved_prompt

    def _integrate_context(self, prompt: str) -> str:
        """整合上下文"""
        # 确保上下文信息被有效利用
        if "{retrieved_context}" in prompt:
            context_instruction = """
## 上下文信息使用指导
请充分利用提供的参考信息：
- 引用相关的技术细节
- 结合具体的案例数据
- 对比不同的观点和方法
- 验证分析结论的准确性
"""
            prompt = prompt.replace(
                "{retrieved_context}",
                "{retrieved_context}" + context_instruction
            )

        return prompt


# dynamic_prompt_generator.py - 动态提示生成器
class DynamicPromptGenerator:
    """动态提示生成器"""

    def __init__(self, template_manager: PromptTemplateManager):
        self.template_manager = template_manager
        self.user_feedback_history = []
        self.performance_history = []

    def generate_adaptive_prompt(self, task_type: str, user_input: str,
                                 user_context: Dict = None,
                                 performance_history: List[Dict] = None) -> str:
        """生成自适应提示"""

        # 分析用户历史偏好
        user_preferences = self._analyze_user_preferences(user_context)

        # 基于性能历史调整策略
        optimization_strategy = self._determine_optimization_strategy(performance_history)

        # 生成基础提示
        base_prompt = self.template_manager.generate_prompt(
            task_type=task_type,
            user_input=user_input,
            context=user_context,
            use_cot=optimization_strategy.get("use_cot", True),
            use_few_shot=optimization_strategy.get("use_few_shot", True),
            num_examples=optimization_strategy.get("num_examples", 1)
        )

        # 应用用户偏好
        adapted_prompt = self._apply_user_preferences(base_prompt, user_preferences)

        return adapted_prompt

    def _analyze_user_preferences(self, user_context: Dict = None) -> Dict:
        """分析用户偏好"""
        preferences = {
            "detail_level": "moderate",  # simple, moderate, detailed
            "technical_depth": "moderate",  # basic, moderate, advanced
            "output_format": "structured",  # concise, structured, comprehensive
            "language_style": "professional"  # casual, professional, academic
        }

        if user_context:
            # 基于用户角色调整偏好
            user_role = user_context.get("user_role", "")
            if "researcher" in user_role.lower():
                preferences["technical_depth"] = "advanced"
                preferences["output_format"] = "comprehensive"
            elif "manager" in user_role.lower():
                preferences["detail_level"] = "simple"
                preferences["output_format"] = "concise"

        return preferences

    def _determine_optimization_strategy(self, performance_history: List[Dict] = None) -> Dict:
        """确定优化策略"""
        strategy = {
            "use_cot": True,
            "use_few_shot": True,
            "num_examples": 1
        }

        if performance_history:
            # 基于历史性能调整策略
            avg_confidence = sum(h.get("confidence", 0) for h in performance_history) / len(performance_history)

            if avg_confidence < 0.6:
                # 置信度低，增加few-shot示例
                strategy["num_examples"] = 2
                strategy["use_cot"] = True
            elif avg_confidence > 0.85:
                # 置信度高，可以简化
                strategy["num_examples"] = 0
                strategy["use_cot"] = False

        return strategy

    def _apply_user_preferences(self, prompt: str, preferences: Dict) -> str:
        """应用用户偏好"""
        adapted_prompt = prompt

        # 调整详细程度
        if preferences.get("detail_level") == "simple":
            adapted_prompt = self._simplify_prompt(adapted_prompt)
        elif preferences.get("detail_level") == "detailed":
            adapted_prompt = self._enhance_detail(adapted_prompt)

        # 调整技术深度
        if preferences.get("technical_depth") == "basic":
            adapted_prompt = self._reduce_technical_complexity(adapted_prompt)
        elif preferences.get("technical_depth") == "advanced":
            adapted_prompt = self._increase_technical_depth(adapted_prompt)

        return adapted_prompt

    def _simplify_prompt(self, prompt: str) -> str:
        """简化提示"""
        # 移除过于详细的说明
        lines = prompt.split('\n')
        simplified_lines = []

        skip_next = False
        for line in lines:
            if skip_next:
                skip_next = False
                continue

            if "详细" in line or "具体" in line:
                # 简化详细要求
                simplified_line = line.replace("详细", "").replace("具体", "")
                if simplified_line.strip():
                    simplified_lines.append(simplified_line)
            else:
                simplified_lines.append(line)

        return '\n'.join(simplified_lines)

    def _enhance_detail(self, prompt: str) -> str:
        """增强细节"""
        # 添加更多详细要求
        enhanced_prompt = prompt

        if "## 分析要求" in enhanced_prompt:
            detail_enhancement = """
### 深度分析要求
- 提供详细的技术实现细节
- 包含具体的配置参数和示例
- 分析多种可能的场景和变化
- 提供完整的参考资料链接
"""
            enhanced_prompt = enhanced_prompt.replace(
                "## 分析要求",
                "## 分析要求" + detail_enhancement
            )

        return enhanced_prompt

    def _reduce_technical_complexity(self, prompt: str) -> str:
        """降低技术复杂度"""
        # 替换复杂术语
        simplifications = {
            "CVSS评分": "安全评级",
            "攻击向量": "攻击方式",
            "漏洞利用": "安全问题利用",
            "缓解措施": "解决方案"
        }

        simplified_prompt = prompt
        for complex_term, simple_term in simplifications.items():
            simplified_prompt = simplified_prompt.replace(complex_term, simple_term)

        return simplified_prompt

    def _increase_technical_depth(self, prompt: str) -> str:
        """增加技术深度"""
        # 添加高级技术要求
        technical_enhancement = """
### 高级技术分析
- 提供底层技术原理解释
- 包含相关协议和标准分析
- 分析与其他安全机制的交互
- 提供量化的风险评估模型
"""

        if "## 分析要求" in prompt:
            enhanced_prompt = prompt.replace(
                "## 分析要求",
                "## 分析要求" + technical_enhancement
            )
        else:
            enhanced_prompt = prompt + technical_enhancement

        return enhanced_prompt

    def collect_feedback(self, prompt: str, response: str,
                         user_satisfaction: float, improvement_suggestions: List[str]):
        """收集用户反馈"""
        feedback = {
            "prompt": prompt,
            "response": response,
            "satisfaction": user_satisfaction,
            "suggestions": improvement_suggestions,
            "timestamp": datetime.now().isoformat()
        }

        self.user_feedback_history.append(feedback)

        # 基于反馈调整模板
        self._update_templates_based_on_feedback(feedback)

    def _update_templates_based_on_feedback(self, feedback: Dict):
        """基于反馈更新模板"""
        if feedback["satisfaction"] < 0.6:
            # 低满意度，需要改进
            for suggestion in feedback["suggestions"]:
                if "更简洁" in suggestion:
                    # 用户希望更简洁的输出
                    pass
                elif "更详细" in suggestion:
                    # 用户希望更详细的分析
                    pass
                # 可以根据具体建议调整模板


if __name__ == "__main__":
    demo_prompt_engineering()