# ollama_client.py - 本地大模型
import json
import logging
import time
import hashlib
import pickle
import os
import subprocess
import platform
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class OllamaResponse:
    """Ollama响应数据结构"""
    content: str
    model: str
    tokens_used: int
    response_time: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OllamaCache:
    """本地缓存系统"""

    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_age = timedelta(hours=max_age_hours)
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """生成缓存键"""
        content = f"{prompt}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[OllamaResponse]:
        """获取缓存响应"""
        cache_key = self._get_cache_key(prompt, model)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # 检查缓存是否过期
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < self.max_age:
                    return cached_data['response']
                else:
                    # 缓存过期，删除文件
                    os.remove(cache_file)
            except Exception as e:
                # 缓存文件损坏，删除
                try:
                    os.remove(cache_file)
                except:
                    pass

        return None

    def set(self, prompt: str, model: str, response: OllamaResponse):
        """设置缓存"""
        cache_key = self._get_cache_key(prompt, model)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        cached_data = {
            'timestamp': datetime.now().isoformat(),
            'response': response
        }

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            logging.error(f"缓存保存失败: {e}")


class OllamaClient:
    """优化的Ollama客户端"""

    def __init__(self, model_name: str = "llama3",  # 默认使用您的llama3
                 base_url: str = "http://localhost:11434",
                 timeout: int = 60):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.cache = OllamaCache()
        self.logger = logging.getLogger(__name__)

        # 统计信息
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "successful_requests": 0,
            "failed_requests": 0
        }

    def is_ollama_installed(self) -> bool:
        """检查Ollama是否安装（支持Windows）"""
        try:
            # Windows下可能需要不同的检查方式
            ollama_commands = ["ollama", "ollama.exe"]

            for cmd in ollama_commands:
                try:
                    result = subprocess.run(
                        [cmd, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        shell=(platform.system() == "Windows")
                    )
                    if result.returncode == 0:
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue

            return False

        except Exception as e:
            self.logger.debug(f"检查Ollama安装失败: {e}")
            return False

    def is_available(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama服务不可用: {e}")
            return False

    def list_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    model_name = model.get("name", "")
                    # 处理模型名称格式化（如 llama3:latest -> llama3）
                    if ":" in model_name:
                        base_name = model_name.split(":")[0]
                        models.extend([model_name, base_name])
                    else:
                        models.append(model_name)
                return list(set(models))  # 去重
            return []
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {e}")
            return []

    def pull_model(self, model_name: str = None) -> bool:
        """拉取模型"""
        model = model_name or self.model_name
        try:
            import requests

            print(f"🔄 正在拉取模型: {model}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                stream=True,
                timeout=600  # 10分钟超时
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                status = data["status"]
                                if "completed" in data and "total" in data:
                                    progress = data["completed"] / data["total"] * 100
                                    print(f"📥 {status}: {progress:.1f}%")
                                else:
                                    print(f"📥 {status}")
                        except json.JSONDecodeError:
                            pass
                print(f"✅ 模型拉取完成: {model}")
                return True
            else:
                print(f"❌ 模型拉取失败: {response.text}")
                return False

        except Exception as e:
            print(f"❌ 模型拉取错误: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 2000,
                 temperature: float = 0.7, stream: bool = False) -> OllamaResponse:
        """生成响应"""
        self.stats["requests"] += 1

        # 1. 检查缓存
        cached_response = self.cache.get(prompt, self.model_name)
        if cached_response:
            self.stats["cache_hits"] += 1
            self.logger.info("使用缓存响应")
            return cached_response

        # 2. 调用Ollama API
        try:
            import requests

            start_time = time.time()

            # 准备请求数据
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["Human:", "Assistant:", "User:", "AI:", "\n\nHuman:", "\n\nUser:"]
                }
            }

            # 发送请求
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            response_time = time.time() - start_time

            # 创建响应对象
            content = result.get("response", "").strip()

            # 清理响应内容
            content = self._clean_response(content)

            ollama_response = OllamaResponse(
                content=content,
                model=self.model_name,
                tokens_used=result.get("eval_count", 0),
                response_time=response_time,
                metadata={
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "eval_duration": result.get("eval_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "done": result.get("done", False)
                }
            )

            # 更新统计
            self.stats["total_tokens"] += ollama_response.tokens_used
            self.stats["total_time"] += response_time
            self.stats["successful_requests"] += 1

            # 缓存响应
            self.cache.set(prompt, self.model_name, ollama_response)

            self.logger.info(f"Ollama响应完成: 耗时{response_time:.2f}s, "
                             f"tokens={ollama_response.tokens_used}")

            return ollama_response

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Ollama生成失败: {e}")

            # 返回错误响应
            return OllamaResponse(
                content=f"抱歉，生成响应时出现错误: {str(e)}",
                model=self.model_name,
                tokens_used=0,
                response_time=0.0,
                metadata={"error": str(e)}
            )

    def _clean_response(self, content: str) -> str:
        """清理响应内容"""
        # 移除可能的停止词残留
        stop_words = ["Human:", "Assistant:", "User:", "AI:"]
        for stop_word in stop_words:
            if content.endswith(stop_word):
                content = content[:-len(stop_word)].strip()

        # 移除多余的换行
        content = content.replace("\n\n\n", "\n\n")

        return content.strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> OllamaResponse:
        """聊天模式"""
        # 将消息列表转换为单个prompt
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")  # 提示AI开始回答
        full_prompt = "\n\n".join(prompt_parts)

        return self.generate(full_prompt, **kwargs)

    def get_stats(self) -> Dict:
        """获取使用统计"""
        total_requests = self.stats["requests"]
        cache_hit_rate = self.stats["cache_hits"] / max(total_requests, 1)
        success_rate = self.stats["successful_requests"] / max(total_requests, 1)

        actual_requests = total_requests - self.stats["cache_hits"]
        avg_response_time = self.stats["total_time"] / max(actual_requests, 1)

        return {
            "total_requests": total_requests,
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "success_rate": f"{success_rate:.1%}",
            "total_tokens": self.stats["total_tokens"],
            "average_response_time": f"{avg_response_time:.2f}s",
            "current_model": self.model_name,
            "service_available": self.is_available(),
            "failed_requests": self.stats["failed_requests"]
        }

    def test_connection(self) -> Dict:
        """测试连接和基本功能"""
        print("🔍 测试Ollama连接...")

        # 1. 检查Ollama安装
        if not self.is_ollama_installed():
            return {
                "status": "not_installed",
                "message": "Ollama未检测到",
                "suggestions": [
                    "请确认Ollama已正确安装",
                    "Windows: 检查环境变量PATH",
                    "重启命令提示符后重试"
                ]
            }

        # 2. 检查服务可用性
        if not self.is_available():
            return {
                "status": "service_unavailable",
                "message": "Ollama服务不可用",
                "suggestions": [
                    "确认Ollama服务正在运行: ollama serve",
                    "检查端口是否正确: " + self.base_url,
                    "检查防火墙设置"
                ]
            }

        # 3. 检查模型是否存在
        available_models = self.list_models()
        print(f"可用模型: {available_models}")

        if self.model_name not in available_models:
            # 尝试常见的模型名称变体
            model_variants = [
                self.model_name,
                f"{self.model_name}:latest",
                self.model_name.split(":")[0] if ":" in self.model_name else f"{self.model_name}:latest"
            ]

            found_model = None
            for variant in model_variants:
                if variant in available_models:
                    found_model = variant
                    break

            if found_model:
                self.model_name = found_model
                print(f"✅ 找到模型变体: {found_model}")
            else:
                return {
                    "status": "model_missing",
                    "message": f"模型 {self.model_name} 不存在",
                    "available_models": available_models,
                    "suggestions": [
                        f"使用可用模型: {available_models[0] if available_models else '无'}",
                        f"或拉取所需模型: ollama pull {self.model_name}"
                    ]
                }

        # 4. 测试生成功能
        try:
            test_prompt = "Hello! Please respond with 'Connection test successful.'"
            print(f"测试提示: {test_prompt}")

            response = self.generate(test_prompt, max_tokens=30, temperature=0.1)

            if "error" in response.metadata:
                return {
                    "status": "generation_failed",
                    "message": "生成测试失败",
                    "error": response.metadata["error"]
                }

            return {
                "status": "success",
                "message": "连接测试成功",
                "model": self.model_name,
                "response_time": f"{response.response_time:.2f}s",
                "test_response": response.content[:100],
                "tokens_used": response.tokens_used
            }

        except Exception as e:
            return {
                "status": "test_failed",
                "message": "连接测试失败",
                "error": str(e)
            }


# 优化的设置函数
def setup_ollama_auto():
    """自动设置Ollama（针对现有环境优化）"""
    print("🚀 Ollama自动设置助手")
    print("=" * 40)

    # 创建客户端
    client = OllamaClient()

    # 1. 检查Ollama安装
    if not client.is_ollama_installed():
        print("❌ Ollama未正确检测到")
        print("请确认:")
        print("  1. Ollama已安装")
        print("  2. 环境变量PATH已配置")
        print("  3. 重启了命令提示符")
        return False, None

    print("✅ Ollama已安装")

    # 2. 检查服务状态
    if not client.is_available():
        print("❌ Ollama服务未运行")
        print("请在另一个命令提示符窗口运行: ollama serve")
        print("然后保持该窗口打开")
        return False, None

    print("✅ Ollama服务运行中")

    # 3. 检查可用模型
    available_models = client.list_models()
    print(f"📦 当前已安装 {len(available_models)} 个模型")

    if available_models:
        print("可用模型:")
        for model in available_models:
            print(f"  - {model}")

        # 自动选择最佳模型
        preferred_models = ["llama3", "llama3:latest", "qwen2:7b", "qwen2:1.5b"]
        selected_model = None

        for preferred in preferred_models:
            if preferred in available_models:
                selected_model = preferred
                break

        if selected_model:
            client.model_name = selected_model
            print(f"✅ 自动选择模型: {selected_model}")
        else:
            # 使用第一个可用模型
            client.model_name = available_models[0]
            print(f"✅ 使用模型: {available_models[0]}")
    else:
        print("⚠️ 未检测到任何模型")
        print("这可能是模型名称格式问题，尝试继续...")

    # 4. 测试连接
    test_result = client.test_connection()

    if test_result["status"] == "success":
        print(f"\n🎉 设置完成！")
        print(f"当前模型: {client.model_name}")
        print(f"测试响应时间: {test_result['response_time']}")
        print(f"测试回答: {test_result.get('test_response', 'N/A')}")
        return True, client
    else:
        print(f"\n❌ 设置失败: {test_result['message']}")
        if "suggestions" in test_result:
            print("建议:")
            for suggestion in test_result["suggestions"]:
                print(f"  💡 {suggestion}")
        return False, None


def demo_ollama_client():
    """演示Ollama客户端功能"""
    print("🤖 Ollama客户端演示 (llama3优化版)")
    print("=" * 50)

    # 自动设置
    success, client = setup_ollama_auto()
    if not success:
        print("\n❌ 自动设置失败，请检查上述建议后重试")
        return

    # IoT安全相关测试查询
    test_queries = [
        {
            "query": "What is IoT security?",
            "category": "基础概念"
        },
        {
            "query": "List the top 5 IoT security threats.",
            "category": "威胁识别"
        },
        {
            "query": "How to secure smart home devices?",
            "category": "防护建议"
        },
        {
            "query": "Explain buffer overflow attacks on IoT devices.",
            "category": "技术分析"
        }
    ]

    print(f"\n💬 IoT安全测试查询:")
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        category = test_case["category"]

        print(f"\n{i}. 【{category}】{query}")
        print("-" * 50)

        try:
            start_time = time.time()
            response = client.generate(query, max_tokens=200, temperature=0.7)
            total_time = time.time() - start_time

            print(f"✅ 生成成功")
            print(f"📊 模型: {response.model}")
            print(f"⏱️ 总耗时: {total_time:.2f}s")
            print(f"🔧 生成耗时: {response.response_time:.2f}s")
            print(f"📄 Tokens: {response.tokens_used}")
            print(f"📝 回答:")
            print(f"{response.content}")

        except Exception as e:
            print(f"❌ 生成失败: {e}")

    # 显示统计信息
    print(f"\n📊 使用统计:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 缓存测试
    print(f"\n🔄 缓存功能测试:")
    test_query = "What is IoT?"

    print("第一次查询（无缓存）:")
    start_time = time.time()
    response1 = client.generate(test_query, max_tokens=50)
    time1 = time.time() - start_time
    print(f"  耗时: {time1:.2f}s")

    print("第二次查询（使用缓存）:")
    start_time = time.time()
    response2 = client.generate(test_query, max_tokens=50)
    time2 = time.time() - start_time
    print(f"  耗时: {time2:.2f}s")
    print(f"  缓存加速: {(time1 / time2):.1f}x" if time2 > 0 else "  缓存命中")

    print(f"\n🎉 演示完成！客户端工作正常。")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行演示
    demo_ollama_client()