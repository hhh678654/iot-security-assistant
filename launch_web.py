#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_web.py - 一键启动Streamlit网页界面
自动检查环境并启动Web服务
"""

import sys
import subprocess
import os
import time
from pathlib import Path


def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")

    required_packages = [
        "streamlit",
        "sentence-transformers",
        "faiss-cpu",
        "scikit-learn",
        "pandas",
        "numpy"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")

    if missing_packages:
        print(f"\n📦 需要安装以下包:")
        print(f"pip install {' '.join(missing_packages)}")

        install = input("\n是否自动安装? (y/n): ").lower().strip()
        if install == 'y':
            print("🔧 正在安装...")
            try:
                subprocess.check_call([
                                          sys.executable, "-m", "pip", "install"] + missing_packages
                                      )
                print("✅ 安装完成!")
            except subprocess.CalledProcessError:
                print("❌ 安装失败，请手动安装")
                return False
        else:
            return False

    return True


def check_project_files():
    """检查项目文件"""
    print("\n🔍 检查项目文件...")

    required_files = [
        "rag_agent.py",
        "ollama_client.py",
        "streamlit_app.py"
    ]

    missing_files = []

    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            missing_files.append(file)
            print(f"  ❌ {file}")

    if missing_files:
        print(f"\n❌ 缺失文件: {missing_files}")
        return False

    return True


def check_ollama_service():
    """检查Ollama服务"""
    print("\n🔍 检查Ollama服务...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("  ✅ Ollama服务运行中")

            # 检查模型
            data = response.json()
            models = [model.get("name", "") for model in data.get("models", [])]
            if models:
                print(f"  📦 可用模型: {models}")
                return True
            else:
                print("  ⚠️ 未找到模型，请安装: ollama pull llama3")
                return False
        else:
            print("  ❌ Ollama服务响应异常")
            return False

    except requests.exceptions.ConnectionError:
        print("  ❌ 无法连接Ollama服务")
        print("  💡 请在另一个命令提示符运行: ollama serve")
        return False
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False


def launch_streamlit():
    """启动Streamlit应用"""
    print("\n🚀 启动Web界面...")

    try:
        # 构建启动命令
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ]

        print("📱 正在启动浏览器...")
        print("🌐 访问地址: http://localhost:8501")
        print("⏹️ 按 Ctrl+C 停止服务")

        # 启动服务
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")


def main():
    """主函数"""
    print("🛡️ IoT安全智能助手 - Web界面启动器")
    print("=" * 50)

    # 检查依赖
    # if not check_dependencies():
    #     print("\n❌ 依赖检查失败")
    #     return False

    # 检查项目文件
    if not check_project_files():
        print("\n❌ 项目文件检查失败")
        return False

    # 检查Ollama服务
    ollama_ok = check_ollama_service()
    if not ollama_ok:
        print("\n⚠️ Ollama服务未就绪")
        continue_anyway = input("是否继续启动Web界面? (y/n): ").lower().strip()
        if continue_anyway != 'y':
            return False

    # 启动Web界面
    print("\n✅ 环境检查完成")
    time.sleep(1)
    launch_streamlit()

    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 启动被用户中断")
    except Exception as e:
        print(f"\n❌ 启动器异常: {e}")
        import traceback

        traceback.print_exc()