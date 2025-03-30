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
from concurrent.futures import ThreadPoolExecutor
from api_request_schema import api_request_list, get_model_ids
import faiss
import nltk
nltk.download('punkt_tab')



# 加载文本文件
def load_text(file_path):
    """加载指定路径的文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"加载文件出错：{str(e)}")
        return ""

# 加载词典
def load_dictionary(file_path):
    """加载词典文件，每行一个词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"加载词典出错：{str(e)}")
        return []

# 改进的文本分割（结合语义边界）
def split_text(text, dictionary, chunk_size=2500, overlap=300):
    """智能文本分割算法，结合叙事边界"""
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        boundary_keywords = ['说道', '心想', '却说', '且说', '次日', '话说', '正是','荣枯转瞬', '春去秋来', '月满则亏', '寒塘鹤影',
    '更漏声残', '元宵灯宴', '中秋赏月', '海棠诗社',
    '金秋菊宴', '白雪红梅', '芒种饯春', '花朝月夕',
    
    # 地点与场景
    '大观园内', '荣禧堂前', '潇湘竹影', '蘅芜清芬',
    '太虚幻境', '铁槛寺外', '梨香院中', '沁芳亭畔',
    '藕榭听雨', '凹晶联诗', '栊翠品茶', '芦雪联句',
    
    # 家族与关系
    '金玉良缘', '木石前盟', '四大家族', '贾史王薛',
    '主仆尊卑', '嫡庶之争', '姊妹情深', '父子隔阂',
    '护官符悬', '抄检大观园', '凤姐弄权', '宝玉摔玉',
    
    # 诗词与隐喻
    '葬花吟罢', '题帕三绝', '芙蓉诔成', '好了歌启',
    '风月宝鉴', '绛珠还泪', '通灵蒙尘', '判词谶语',
    '满纸荒唐言', '机关算尽太聪明', '一朝春尽红颜老',
    
    # 情感与冲突
    '泪尽而逝', '冷香丸苦', '心比天高', '情深不寿',
    '含酸讥诮', '痴儿呆语', '绣春囊祸', '抄家惊变',
    '机关算尽', '世态炎凉', '大厦倾颓', '白茫茫真干净',
    
    # 象征与意象
    '金陵十二钗', '太虚幻境册', '绛珠仙草', '神瑛侍者',
    '落红成阵', '金锁沉埋', '脂浓粉香', '青埂峰下',
    '警幻仙子', '风月孽债', '镜花水月', '飞鸟各投林']

        for sent in sentences:
            # 检查语义边界词
            if any(re.search(rf'\b{re.escape(kw)}\b', sent) for kw in boundary_keywords):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = current_chunk[-overlap//20:]  # 按句子重叠
                    current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sent)
            current_length += len(sent)

            # 达到块大小时分割
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap//20:]  # 按句子保留重叠
                current_length = sum(len(s) for s in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks if chunks else [text[:2500]]
    except Exception as e:
        print(f"文本分割出错：{str(e)}")
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 加载事件标签和情节关键词
def load_labels_and_keywords():
    try:
        with open('event_labels.json', 'r', encoding='utf-8') as f:
            event_labels = json.load(f)
        with open('plot_keywords.json', 'r', encoding='utf-8') as f:
            plot_keywords = json.load(f)
        return event_labels, plot_keywords
    except Exception as e:
        print(f"加载标签出错：{str(e)}")
        return {}, {}

# 计算归一化Embedding
def compute_embeddings(chunks):
    """计算并归一化文本嵌入"""
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(chunks, show_progress_bar=True)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    except Exception as e:
        print(f"Embedding计算失败：{str(e)}")
        return np.array([])

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
            print(f"加载关系出错：{str(e)}")
            return []

    def build_relation_graph(self):
        graph = {}
        for rel in self.relations:
            graph.setdefault(rel['head'], []).append((rel['tail'], rel['label']))
            graph.setdefault(rel['tail'], []).append((rel['head'], rel['label']))
        return graph

    def find_related_characters(self, name, depth=5):
        visited = set()
        queue = [(name, 0)]
        while queue:
            current, current_depth = queue.pop(0)
            if current_depth > depth or current not in self.relation_graph:
                continue
            for neighbor, _ in self.relation_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
        return list(visited)

def get_related_terms(query, plot_keywords):
    term_info = plot_keywords.get(query, {})
    aliases = term_info.get('aliases', [])
    category = term_info.get('category', '其他')
    return list(set([query] + aliases + 
        [k for k, v in plot_keywords.items() if v.get('category') == category]))

# 改进的语义增强检索
def semantic_enhanced_retrieval(query, chunks, plot_keywords, relations, model, index):
    try:
        # 增强关键词匹配
        keyword_weights = {'人物':4.0, '事件':3.5, '地点':3.0, '物品':2.5, '其他':2.0}
        all_keywords = get_related_terms(query, plot_keywords)
        keyword_scores = {}
        
        for i, chunk in enumerate(chunks):
            score = 0
            chunk_lower = chunk.lower()
            position_weight = 1 + (i/len(chunks))  # 位置权重
            for keyword in all_keywords:
                kw_lower = keyword.lower()
                freq = len(re.findall(rf'\b{re.escape(kw_lower)}\b', chunk_lower))
                category = plot_keywords.get(keyword, {}).get('category', '其他')
                score += freq * keyword_weights[category] * position_weight
            keyword_scores[i] = score

        # 人物关系扩展
        related_people = relations.find_related_characters(query)
        relation_scores = {}
        if query in plot_keywords and plot_keywords[query].get('category') == '人物':
                related_people = relations.find_related_characters(query)
                relation_scores[i] = sum(1 for p in related_people if re.search(rf'\b{re.escape(p)}\b', chunk))
        else:
            relation_scores[i] = 0

        # 语义相似度计算
        query_embedding = model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        similarities, indices = index.search(query_embedding, len(chunks))
        semantic_scores = similarities[0]

        # 动态权重综合评分
        combined_scores = []
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1e-5
        max_relation = max(relation_scores.values()) if relation_scores else 1e-5
        
        for i in range(len(chunks)):
            # 安全除法（分母+极小值）
            keyword_score = keyword_scores.get(i, 0) / (max_keyword + 1e-5)
            relation_score = relation_scores.get(i, 0) / (max_relation + 1e-5)
            
            # 权重动态调整
            if semantic_scores[i] > 0.7:
                total = 0.1 * keyword_score + 0.1 * relation_score + 0.8 * semantic_scores[i]
            elif semantic_scores[i] > 0.5:
                total = 0.2 * keyword_score + 0.1 * relation_score + 0.7 * semantic_scores[i]
            else:
                total = 0.3 * keyword_score + 0.1 * relation_score + 0.6 * semantic_scores[i]
            
            
            combined_scores.append((i, total))

        sorted_indices = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        return [chunks[i] for i, _ in sorted_indices[:5]]
    except Exception as e:
        print(f"检索失败：{str(e)}")
        return []

# 高亮显示（优化颜色代码）
def highlight_keywords(text, query, plot_keywords):
    keywords = get_related_terms(query, plot_keywords)
    color_codes = {'人物': '\033[1;31m', '地点': '\033[1;32m', '事件': '\033[1;33m'}
    reset = '\033[0m'
    
    highlighted = text
    for kw in sorted(keywords, key=len, reverse=True):
        category = plot_keywords.get(kw, {}).get('category', '其他')
        color = color_codes.get(category, '\033[1;34m')
        highlighted = re.sub(
            rf'\b({re.escape(kw)})\b',  # 修正括号闭合
            f'{color}\\1{reset}',
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted


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
    relations = CharacterRelations('triples.csv')
    dictionary = load_dictionary('hongloumeng_dict.txt')
    text = load_text('hongloumeng.txt')
    chunks = split_text(text, dictionary, chunk_size=1500, overlap=200)
    event_labels, plot_keywords = load_labels_and_keywords()
    
    if not chunks:
        print("错误：文本分割失败")
        return
    
    # 计算Embedding
    chunk_embeddings = compute_embeddings(chunks)
    if chunk_embeddings.size == 0:
        print("错误：Embedding计算失败")
        return
    
    # 构建FAISS索引
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积
    index.add(chunk_embeddings)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    bedrock_wrapper = BedrockWrapper()

    # 交互循环
    while True:
        try:
            query = input("\n请输入查询内容（输入q退出）: ").strip()
            if query.lower() == 'q':
                break
            if not query:
                continue

            # 语义检索
            relevant_chunks = semantic_enhanced_retrieval(
                query, chunks, plot_keywords, relations, model, index
            )
            
            # 显示结果
            print("\n相关段落：")
            for i, chunk in enumerate(relevant_chunks[:3], 1):
                print(f"【段落{i}】")
                print(highlight_keywords(chunk, query, plot_keywords))
                print("-"*60)

            # 构建上下文
            context = "\n".join([
                f"上下文{i+1}: {chunk[:800]}..." 
                for i, chunk in enumerate(relevant_chunks[:3])
            ])
            prompt = f"""基于以下上下文和人物关系回答问题：
{context}

相关人物关系（前3条）：
{json.dumps(relations.relations[:3], ensure_ascii=False, indent=2)}

问题：{query}
请用中文给出详细解答："""

            # 调用Bedrock
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(bedrock_wrapper.invoke_bedrock, prompt)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理异常：{str(e)}")

if __name__ == "__main__":
    main()