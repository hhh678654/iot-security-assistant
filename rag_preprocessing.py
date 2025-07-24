# IoT安全智能分析系统 - 数据预处理与RAG构建

import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib

# 文本处理相关
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 数据处理
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# 配置和工具
import logging
import pickle
import os
from pathlib import Path


class IoTSecurityDataProcessor:
    """IoT安全数据预处理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_nlp_tools()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_nlp_tools(self):
        """初始化NLP工具"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

            # 加载spaCy模型（用于更好的文本处理）
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy模型未找到，使用基础文本处理")
                self.nlp = None

        except Exception as e:
            self.logger.error(f"NLP工具初始化失败: {e}")

    def load_academic_data(self, filepath: str) -> pd.DataFrame:
        """加载学术数据集"""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"成功加载学术数据: {len(df)} 条记录")

            # 数据清洗
            df = df.dropna(subset=['text', 'label'])
            df['text'] = df['text'].astype(str)
            df['label'] = df['label'].astype(str)
            df['year'] = pd.to_numeric(df['year'], errors='coerce')

            self.logger.info(f"清洗后学术数据: {len(df)} 条记录")
            return df

        except Exception as e:
            self.logger.error(f"加载学术数据失败: {e}")
            raise

    def load_cve_data(self, filepath: str) -> pd.DataFrame:
        """加载CVE数据集"""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"成功加载CVE数据: {len(df)} 条记录")

            # 数据清洗和类型转换
            df = df.dropna(subset=['description', 'text'])
            df['published'] = pd.to_datetime(df['published'], errors='coerce')
            df['last_modified'] = pd.to_datetime(df['last_modified'], errors='coerce')
            df['base_score'] = pd.to_numeric(df['base_score'], errors='coerce')

            self.logger.info(f"清洗后CVE数据: {len(df)} 条记录")
            return df

        except Exception as e:
            self.logger.error(f"加载CVE数据失败: {e}")
            raise

    def text_chunking(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
        """智能文本分块"""
        if not text or len(text.strip()) == 0:
            return []

        chunks = []

        # 优先按句子分割
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = sent_tokenize(text)

        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # 如果当前块加上新句子超过限制，保存当前块
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'length': current_length,
                    'type': 'semantic'
                })

                # 重叠处理：保留最后一些内容
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(overlap_text) + sentence_length + 1
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length + (1 if current_chunk else 0)

        # 添加最后一个块
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'length': current_length,
                'type': 'semantic'
            })

        return chunks

    def process_academic_data(self, df: pd.DataFrame) -> List[Dict]:
        """处理学术数据为RAG格式"""
        processed_docs = []

        for idx, row in df.iterrows():
            # 文本分块
            chunks = self.text_chunking(row['text'])

            for chunk_idx, chunk in enumerate(chunks):
                doc = {
                    'id': f"academic_{idx}_{chunk_idx}",
                    'source_type': 'academic',
                    'source_id': idx,
                    'text': chunk['text'],
                    'metadata': {
                        'label': row['label'],
                        'year': row['year'],
                        'doi': row.get('doi', ''),
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'chunk_type': chunk['type']
                    }
                }
                processed_docs.append(doc)

        self.logger.info(f"学术数据处理完成: {len(processed_docs)} 个文档块")
        return processed_docs

    def process_cve_data(self, df: pd.DataFrame) -> List[Dict]:
        """处理CVE数据为RAG格式"""
        processed_docs = []

        for idx, row in df.iterrows():
            # CVE描述通常较短，不需要分块，但可以组合多个字段
            combined_text = f"{row['description']}"
            if pd.notna(row.get('text')) and row['text'] != row['description']:
                combined_text += f" Additional context: {row['text']}"

            doc = {
                'id': f"cve_{row['id']}",
                'source_type': 'cve',
                'source_id': row['id'],
                'text': combined_text,
                'metadata': {
                    'cve_id': row['id'],
                    'severity': row['severity'],
                    'base_score': row['base_score'],
                    'base_severity': row['base_severity'],
                    'published': row['published'].isoformat() if pd.notna(row['published']) else None,
                    'device_types': row.get('device_types', ''),
                    'vulnerability_types': row.get('vulnerability_types', ''),
                    'vector_string': row.get('vector_string', ''),
                    'keywords_matched': row.get('keywords_matched', '')
                }
            }
            processed_docs.append(doc)

        self.logger.info(f"CVE数据处理完成: {len(processed_docs)} 个文档")
        return processed_docs

    def generate_qa_pairs(self, docs: List[Dict]) -> List[Dict]:
        """生成问答对用于后续微调"""
        qa_pairs = []

        for doc in docs:
            if doc['source_type'] == 'academic':
                # 学术文档的问答对生成
                questions = [
                    f"What is {doc['metadata']['label']}?",
                    f"Explain the concept of {doc['metadata']['label']}",
                    f"What are the key findings about {doc['metadata']['label']}?"
                ]

                for question in questions:
                    qa_pairs.append({
                        'question': question,
                        'answer': doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'],
                        'source_id': doc['id'],
                        'metadata': doc['metadata']
                    })

            elif doc['source_type'] == 'cve':
                # CVE文档的问答对生成
                cve_id = doc['metadata']['cve_id']
                questions = [
                    f"What is {cve_id}?",
                    f"Describe the vulnerability {cve_id}",
                    f"What are the security implications of {cve_id}?",
                    f"What devices are affected by {cve_id}?"
                ]

                for question in questions:
                    qa_pairs.append({
                        'question': question,
                        'answer': doc['text'],
                        'source_id': doc['id'],
                        'metadata': doc['metadata']
                    })

        self.logger.info(f"生成问答对: {len(qa_pairs)} 对")
        return qa_pairs

    def save_processed_data(self, docs: List[Dict], qa_pairs: List[Dict], output_dir: str):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存文档
        with open(os.path.join(output_dir, 'processed_documents.json'), 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        # 保存问答对
        with open(os.path.join(output_dir, 'qa_pairs.json'), 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        # 保存统计信息
        stats = {
            'total_documents': len(docs),
            'academic_docs': len([d for d in docs if d['source_type'] == 'academic']),
            'cve_docs': len([d for d in docs if d['source_type'] == 'cve']),
            'total_qa_pairs': len(qa_pairs),
            'processing_time': datetime.now().isoformat()
        }

        with open(os.path.join(output_dir, 'processing_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"数据保存完成: {output_dir}")


class HybridRAGSystem:
    """混合检索RAG系统"""

    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.load_models()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_models(self):
        """加载模型"""
        try:
            # 加载语义检索模型
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            self.sentence_model = SentenceTransformer(model_name)
            self.logger.info(f"加载语义模型: {model_name}")

            # 初始化TF-IDF向量器（用于BM25风格的检索）
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

    def build_vector_index(self, documents: List[Dict]) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
        """构建向量索引"""
        self.logger.info("开始构建向量索引...")

        # 提取文本
        texts = [doc['text'] for doc in documents]

        # 生成语义向量
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)

        # 构建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 内积索引

        # 归一化向量（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))

        self.logger.info(f"向量索引构建完成: {len(documents)} 个文档, 维度 {dimension}")
        return embeddings, index

    def build_tfidf_index(self, documents: List[Dict]) -> np.ndarray:
        """构建TF-IDF索引"""
        self.logger.info("开始构建TF-IDF索引...")

        texts = [doc['text'] for doc in documents]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        self.logger.info(f"TF-IDF索引构建完成: {tfidf_matrix.shape}")
        return tfidf_matrix

    def semantic_search(self, query: str, index: faiss.IndexFlatIP,
                        documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """语义检索"""
        # 查询向量化
        query_embedding = self.sentence_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # 检索
        scores, indices = index.search(query_embedding.astype('float32'), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(documents):
                result = documents[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                result['search_type'] = 'semantic'
                results.append(result)

        return results

    def keyword_search(self, query: str, tfidf_matrix: np.ndarray,
                       documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """关键词检索"""
        # 查询向量化
        query_tfidf = self.tfidf_vectorizer.transform([query])

        # 计算相似度
        similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # 获取最相关的文档
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # 只返回有相关性的结果
                result = documents[idx].copy()
                result['score'] = float(similarities[idx])
                result['rank'] = i + 1
                result['search_type'] = 'keyword'
                results.append(result)

        return results

    def hybrid_search(self, query: str, semantic_index: faiss.IndexFlatIP,
                      tfidf_matrix: np.ndarray, documents: List[Dict],
                      top_k: int = 10, alpha: float = 0.7) -> List[Dict]:
        """混合检索"""
        # 获取两种检索结果
        semantic_results = self.semantic_search(query, semantic_index, documents, top_k * 2)
        keyword_results = self.keyword_search(query, tfidf_matrix, documents, top_k * 2)

        # 合并结果并重新评分
        combined_results = {}

        # 处理语义检索结果
        for result in semantic_results:
            doc_id = result['id']
            combined_results[doc_id] = result.copy()
            combined_results[doc_id]['semantic_score'] = result['score']
            combined_results[doc_id]['keyword_score'] = 0

        # 处理关键词检索结果
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['score']
            else:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['semantic_score'] = 0
                combined_results[doc_id]['keyword_score'] = result['score']

        # 计算混合分数
        for doc_id in combined_results:
            semantic_score = combined_results[doc_id]['semantic_score']
            keyword_score = combined_results[doc_id]['keyword_score']
            combined_results[doc_id]['hybrid_score'] = (
                    alpha * semantic_score + (1 - alpha) * keyword_score
            )
            combined_results[doc_id]['search_type'] = 'hybrid'

        # 排序并返回前k个结果
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        # 重新分配排名
        for i, result in enumerate(sorted_results[:top_k]):
            result['rank'] = i + 1

        return sorted_results[:top_k]

    def save_indices(self, semantic_index: faiss.IndexFlatIP,
                     tfidf_matrix: np.ndarray, embeddings: np.ndarray,
                     documents: List[Dict], output_dir: str):
        """保存索引和相关数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(semantic_index, os.path.join(output_dir, 'semantic_index.faiss'))

        # 保存embeddings
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)

        # 保存TF-IDF相关数据
        with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        with open(os.path.join(output_dir, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(tfidf_matrix, f)

        # 保存文档数据
        with open(os.path.join(output_dir, 'indexed_documents.json'), 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        self.logger.info(f"索引保存完成: {output_dir}")


def main():
    """主函数：完整的数据预处理和RAG构建流程"""

    # 配置
    config = {
        'chunk_size': 800,
        'overlap': 100,
        'embedding_model': 'all-MiniLM-L6-v2',
        'top_k': 10,
        'hybrid_alpha': 0.7
    }

    # 文件路径（需要根据实际情况调整）
    academic_file = 'cleaned_academic_data.csv'
    cve_file = 'cleaned_iot_cve_data.csv'
    output_dir = 'processed_data'
    index_dir = 'rag_indices'

    print("=== IoT安全智能分析系统 - 数据预处理与RAG构建 ===")

    # 第一步：数据预处理
    print("\n1. 开始数据预处理...")
    processor = IoTSecurityDataProcessor(config)

    # 加载数据
    academic_df = processor.load_academic_data(academic_file)
    cve_df = processor.load_cve_data(cve_file)

    # 处理数据
    academic_docs = processor.process_academic_data(academic_df)
    cve_docs = processor.process_cve_data(cve_df)

    # 合并文档
    all_documents = academic_docs + cve_docs

    # 生成问答对
    qa_pairs = processor.generate_qa_pairs(all_documents)

    # 保存处理后的数据
    processor.save_processed_data(all_documents, qa_pairs, output_dir)

    # 第二步：构建RAG系统
    print("\n2. 开始构建RAG系统...")
    rag_system = HybridRAGSystem(config)

    # 构建索引
    embeddings, semantic_index = rag_system.build_vector_index(all_documents)
    tfidf_matrix = rag_system.build_tfidf_index(all_documents)

    # 保存索引
    rag_system.save_indices(semantic_index, tfidf_matrix, embeddings, all_documents, index_dir)

    # 第三步：测试检索功能
    print("\n3. 测试检索功能...")
    test_queries = [
        "IoT device security vulnerabilities",
        "smart home privacy protection",
        "DDoS attack on IoT networks",
        "encryption methods for IoT devices"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = rag_system.hybrid_search(
            query, semantic_index, tfidf_matrix, all_documents, top_k=3
        )

        for i, result in enumerate(results, 1):
            print(f"  {i}. [分数: {result['hybrid_score']:.3f}] "
                  f"{result['source_type']} - {result['text'][:100]}...")

    print(f"\n=== 处理完成 ===")
    print(f"总文档数: {len(all_documents)}")
    print(f"学术文档: {len(academic_docs)}")
    print(f"CVE文档: {len(cve_docs)}")
    print(f"问答对: {len(qa_pairs)}")
    print(f"输出目录: {output_dir}")
    print(f"索引目录: {index_dir}")


if __name__ == "__main__":
    main()