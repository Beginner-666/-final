import json
import os
import sys
import csv
import boto3
import time
import re
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
from api_request_schema import api_request_list, get_model_ids

# 下载nltk资源
nltk.download('punkt')

# 加载文本文件
def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"加载文件出错：{str(e)}")
        return ""

# 文本分割
def split_text(text, chunk_size=2500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 计算文本向量
def compute_embeddings(chunks, model):
    return model.encode(chunks)

# 构建FAISS索引
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离
    index.add(embeddings)
    return index

# 检索相关段落
def retrieve_relevant_chunks(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# 人物关系处理类（仅用于回答生成）
class CharacterRelations:
    def __init__(self, triples_file):
        self.relations = self.load_relations(triples_file)
        self.relation_graph = self.build_relation_graph()

    def load_relations(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                return [{
                    'head': row['head'].strip(),
                    'tail': row['tail'].strip(),
                    'relation': row['relation'].strip(),
                    'label': row['label'].strip()
                } for row in reader if all(field in row for field in ['head', 'tail', 'relation', 'label'])]
        except Exception as e:
            print(f"加载人物关系出错：{str(e)}")
            return []

    def build_relation_graph(self):
        graph = {}
        for rel in self.relations:
            graph.setdefault(rel['head'], []).append(rel['tail'])
            graph.setdefault(rel['tail'], []).append(rel['head'])
        return graph

    def get_relations(self, name):
        return self.relation_graph.get(name, [])

# Bedrock交互部分
model_id = os.getenv('MODEL_ID', 'meta.llama3-70b-instruct-v1')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if model_id not in get_model_ids():
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

api_request = api_request_list[model_id]
config = {
    'log_level': 'none',  # One of: info, debug, none
    'region': aws_region,
    'bedrock': {
        'api_request': api_request
    }
}

bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])

def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)

class BedrockModelsWrapper:
    @staticmethod
    def define_body(text):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                body['prompt'] = f"""
                    <|begin_of_text|>
                    <|start_header_id|>user<|end_header_id|>
                    {text}, please output in Chinese.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
            else:
                body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                body['messages'] = [
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            else:
                body['prompt'] = f'\n\nHuman: {text}\n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = text
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        else:
            raise Exception('Unknown model provider.')

        return body

    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]

        chunk_obj = ''
        text = ''
        if model_provider == 'amazon':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
        elif model_provider == 'meta':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['generation']
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                if chunk_obj['type'] == 'message_delta':
                    print(f"\nStop reason: {chunk_obj['delta']['stop_reason']}")
                    print(f"Stop sequence: {chunk_obj['delta']['stop_sequence']}")
                    print(f"Output tokens: {chunk_obj['usage']['output_tokens']}")

                if chunk_obj['type'] == 'content_block_delta':
                    if chunk_obj['delta']['type'] == 'text_delta':
                        text = chunk_obj['delta']['text']
            else:
                # Claude2.x
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['completion']
        elif model_provider == 'cohere':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = ' '.join([c["text"] for c in chunk_obj['generations']])
        elif model_provider == 'mistral':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputs'][0]['text']
        else:
            raise NotImplementedError('Unknown model provider.')

        printer(f'[DEBUG] {chunk_obj}', 'debug')
        return text

def to_text_generator(bedrock_stream):
    prefix = ''

    if bedrock_stream:
        for event in bedrock_stream:
            chunk = BedrockModelsWrapper.get_stream_chunk(event)
            if chunk:
                text = BedrockModelsWrapper.get_stream_text(chunk)

                if '.' in text:
                    a = text.split('.')[:-1]
                    to_print = ''.join([prefix, '.'.join(a), '. '])
                    prefix = text.split('.')[-1]
                    print(to_print, flush=True, end='')
                    yield to_print
                else:
                    prefix = ''.join([prefix, text])

        if prefix != '':
            print(prefix, flush=True, end='')
            yield f'{prefix}.'

        print('\n')

class BedrockWrapper:
    def __init__(self):
        self.speaking = False

    def is_speaking(self):
        return self.speaking

    def invoke_bedrock(self, text):
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        body = BedrockModelsWrapper.define_body(text)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            printer('[DEBUG] Capturing Bedrocks response/bedrock_stream', 'debug')
            bedrock_stream = response.get('body')
            printer(f"[DEBUG] Bedrock_stream: {bedrock_stream}", 'debug')

            text_gen = to_text_generator(bedrock_stream)
            printer('[DEBUG] Created bedrock stream to text generator', 'debug')

            for text in text_gen:
                pass

        except Exception as e:
            print(e)
            time.sleep(2)
            self.speaking = False

        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')

def main():
    # 初始化
    text = load_text('hongloumeng.txt')
    chunks = split_text(text)
    
    # 加载人物关系数据（仅用于回答生成）
    relations = CharacterRelations('triples.csv')
    
    # 加载模型与构建索引
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = compute_embeddings(chunks, model)
    index = build_faiss_index(embeddings)

    bedrock_wrapper = BedrockWrapper()

    while True:
        query = input("\n红楼梦问答（输入q退出）: ").strip()
        if not query or query.lower() == 'q':
            break

        # 仅向量检索
        relevant_chunks = retrieve_relevant_chunks(query, model, index, chunks)
        
        # 展示结果
        print("\n相关段落：")
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            print(f"【段落{i}】")
            print(chunk[:500])  # 展示段落前500字
            print("-" * 50)

        # 提取查询中的人物（仅用于回答生成）
        query_persons = re.findall(r'[贾史王薛林]+\w+', query)
        person_relations = {}
        for person in query_persons:
            person_relations[person] = relations.get_relations(person)

        # 构建上下文并提问
        context = "\n".join(relevant_chunks[:3])
        if context:
            prompt = f"""你是一名红学专家，请根据《红楼梦》文本证据和红学研究成果回答问题，要求：
1. 证据引用（必须）
   - 直接引用原文段落（标注章回）
   - 关联核心情节发展
2. 逻辑推导（必须）
   - 提取问题中的核心人物
   - 基于文本证据的因果关系分析
   - 结合人物关系图谱（triples.csv）
3. 错误纠正（可选）
   - 问题中的事实性错误用[×]标注
   - 补充正确信息及依据
4. 名词阐释（可选）
   - 解释特殊术语（如"金玉良缘"）
   - 说明文化背景
5. 结论总结（必须）
6. 严禁推测文本未描述的情节
            
问题：{query}
文本线索：{'...'.join([c[:300] for c in relevant_chunks[:5]])}
人物关系：{json.dumps(person_relations, ensure_ascii=False)}"""
            bedrock_wrapper.invoke_bedrock(prompt)
        else:
            print("未找到相关内容")

if __name__ == "__main__":
    main()