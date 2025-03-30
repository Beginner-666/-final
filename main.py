import json
import os
import sys
import csv
import time
import boto3
import pytesseract
import easyocr
import asyncio
import pyaudio
import sounddevice
import re
import cv2
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image, ImageEnhance, ImageFilter
from fine_tunning_data import ft_data
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from api_request_schema import api_request_list, get_model_ids

#pytessract初始化配置
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # 使用安装tesseract的地址

# Bedrock模型配置
claude = "anthropic.claude-3-sonnet-20240229-v1:0"
llama = 'meta.llama3-70b-instruct-v1'
model_id = os.getenv('MODEL_ID', llama)
aws_region = os.getenv('AWS_REGION', 'us-east-1')
api_request = api_request_list[model_id]
if model_id not in get_model_ids():
    print(f'Error: Model ID {model_id} is not valid. Choose from: {get_model_ids()}')
    sys.exit(0)
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=aws_region)
#辅助函数
def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)

# 统一配置
config = {
    'log_level': 'info',# One of: info, debug, none
    'region': aws_region,
    'bedrock': {'api_request': api_request},

    'polly': {
        'Engine': 'neural',
        'LanguageCode': 'cmn-CN',
        'VoiceId': 'Zhiyu',
        'OutputFormat': 'pcm',
    },
    'voice_output': False,

    'ocr': {
        'preprocess_level': 2,
        'use_easyocr': True,
        'math_detection': True
    }   
}

#OCR增强模块

#图像预处理
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图片文件")

        #去噪
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        #对比度增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge((l,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        #二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        #形态学处理
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(cleaned)
    except Exception as e:
        print(f"图片预处理失败: {e}")
        return Image.open(image_path)  # 
    
#混合tessract和easyocr识别图片中的文字
def hybrid_ocr(image_path):
    """混合OCR引擎识别"""
    try:
        #预处理
        img = preprocess_image(image_path)
        
        # Tesseract（中英文+公式优化）
        custom_config = r'--oem 3 --psm 6 -l chi_sim+eng+equ'
        tesseract_text = pytesseract.image_to_string(img, config=custom_config)

        # EasyOCR
        if config['ocr']['use_easyocr']:
            try:
                reader = easyocr.Reader(['ch_sim','en'])
                easyocr_result = reader.readtext(np.array(img))
                easyocr_text = ' '.join([text for (_, text, _) in easyocr_result])
                # 选择更长的结果
                text = tesseract_text if len(tesseract_text) > len(easyocr_text) else easyocr_text
            except ImportError:
                text = tesseract_text
        else:
            text = tesseract_text

        # 数学公式标记
        if config['ocr']['math_detection']:
            text = detect_math_expressions(text)
        return text.strip()
    except Exception as e:
        print(f"OCR处理异常: {e}")
        return ""
#识别并标记数学公式
def detect_math_expressions(text):
    patterns = [
        r'\$.*?\$',             # LaTeX公式
        r'\\\(.*?\\\)',         # 行内公式
        r'\\\[.*?\\\]',         # 行间公式
        r'\b(sin|cos|tan|log|lim|sum|integral)\b',
        r'\d+[\+\-\*/^]\d+',    # 基本运算
        r'[α-ω]',               # 希腊字母
        r'[∫∮⋂⋃]'             # 数学符号
    ]
    for pattern in patterns:
        text = re.sub(pattern, lambda m: f'[MATH]{m.group()}[/MATH]', text)
    return text

#语音模块

#使用Transcribe将语音转换为文字
transcribe_streaming = TranscribeStreamingClient(region=config['region'])
#使用Polly进行语音播报
polly = boto3.client('polly', region_name=config['region'])
#使用Pyaudio播放音频
p = pyaudio.PyAudio()  # Audio interface

#语音输入
class UserInputManager:
    shutdown_executor = False
    executor = None

    @staticmethod
    def set_executor(executor):
        UserInputManager.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputManager.shutdown_executor = True

    @staticmethod
    def is_executor_set():
        return UserInputManager.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputManager.shutdown_executor

class SpeechEventHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_stream):
        super().__init__(transcript_stream)
        self.transcript = []
        self.final_transcript = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if results:
            for result in results:
                if not result.is_partial:
                    transcript = result.alternatives[0].transcript
                    self.final_transcript = transcript
                    print(transcript, end=' ', flush=True)

class SpeechInput:
    def __init__(self):
        self.silence_counter = 0
        self.stop_event = asyncio.Event()  # 终止事件触发器
        self.SILENCE_THRESHOLD = 0.02      # 静音阈值（需根据麦克风调整）
        self.MAX_SILENCE_FRAMES = 15       # 持续静音帧数阈值

    #音频回调函数，使用能量计算检测是否静音
    def _audio_callback(self, indata, frame_count, time_info, status):
        """"""
        volume_norm = np.linalg.norm(indata) * 10  # 计算音频能量
        if volume_norm < self.SILENCE_THRESHOLD:
            self.silence_counter += 1
        else:
            self.silence_counter = 0
        
        # 触发静音终止
        if self.silence_counter > self.MAX_SILENCE_FRAMES:
            self.stop_event.set()
    #语音输入主函数，通过静音或回车键终止该次输入
    async def get_voice_input(self):
        # 重置终止事件
        self.stop_event.clear()  
    
        # 启动语音流任务
        stream = await transcribe_streaming.start_stream_transcription(
            language_code="zh-CN",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        async def write_chunks():
            try:
                async for chunk, status in self._mic_generator():
                    if self.stop_event.is_set():
                        break
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
            finally:
                await stream.input_stream.end_stream()
        
        # 启动回车键监听任务
        async def wait_for_enter():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input, "\n按回车键结束录音...")  # 阻塞式监听
            self.stop_event.set()  # 触发终止

        handler = SpeechEventHandler(stream.output_stream)
        await asyncio.gather(
            write_chunks(),
            handler.handle_events(),
            wait_for_enter(),  # 回车监听
            return_exceptions=True
        )
        return handler.final_transcript
    #麦克风生成器函数
    async def _mic_generator(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()
        def callback(indata, frames, time_info, status):  # 明确接收status参数
            self._audio_callback(indata, frames, time_info, status)  # 传递所有参数
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))
        with sounddevice.RawInputStream(
            channels=1, 
            samplerate=16000, 
            callback=callback, 
            blocksize=2048 * 2, 
            dtype="int16"
        ):
            while not self.stop_event.is_set():  # 检查终止事件
                yield await input_queue.get()

#使用Polly和Pyaudio进行音频播放
class AudioPlayer:
    def __init__(self):
        self.stream = p.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=16000, 
            output=True,
            frames_per_buffer=4096
            )
        self.chunk = 1024
        
    def play(self, audio_data):
        self.stream.write(audio_data)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

    def play_text(self, text):
        response = polly.synthesize_speech(
            Text=text,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )
        try:
            while True:
                data = response['AudioStream'].read(self.chunk)
                if not data:
                    break
                if UserInputManager.is_shutdown_scheduled():
                    raise KeyboardInterrupt
                self.play(data)
        except KeyboardInterrupt:
            print("\n语音输出已中断")
        finally:
            self.stop()
#RAG模块
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


#Bedrock大语言模型交互模块

class BedrockModelsWrapper:
    #构造API请求体
    @staticmethod
    def define_body(text,language = 'Chinese',history = None):
        
        language_request = f"please output in English" if language=='English' else f"请用中文回答"
        if history is None:
            history = {
                'previous_dialogue':None,
                'antepenult_dialogue':None,
                'penult_dialogue':None,
                'last_dialogue':None,
            }
        
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']
        
        
        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                #添加历史对话信息
                history_message = f"""
                    {f'Last dialogue :{history['last_dialogue']}\n' if history['last_dialogue'] else None}
                    {f'Penult dialogue : {history['penult_dialogue']}\n' if history['penult_dialogue'] else None}
                    {f'Antepenult dialogue : {history['antepenult_dialogue']}\n' if history['antepenult_dialogue'] else None}
                    {f'Remote dialogue : {history['previous_dialogue']}\n' if history['previous_dialogue'] else None}
                    """
                body['prompt'] = f"""
                    <|history_record|>
                    {history_message}
                    <|end_history_record|>
                    <|begin_of_text|>
                    <|start_header_id|>user<|end_header_id|>
                    {text}, {language_request}.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
                #and take history record into consideration
            else: 
                body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                # 读取所需要的微调数据
                mesg = ft_data['anthropic']['claude-3']['messages']
                system_mesg = ft_data['anthropic']['claude-3']['system']
                
                # 检查最后一个角色是否是 'assistant'
                if mesg[-1]['role'] == 'assistant':
                    #添加历史对话信息
                    if history['last_dialogue']:
                        #mesg.append({'role':"user",'content':f"{history_message}\nCurrent dialogue:{text}"})
                        if history['previous']:
                            mesg.append({'role':"user",'content':f"Here is our dialogue history,please bear this in  you mind.{history['previous']}"})
                            mesg.append({"role": "assistant","content":"OK,I will keep it in mind"})
                        if history['antepenult_dialogue']:
                            mesg.append(history['antepenult_dialogue'][0])
                            mesg.append(history['antepenult_dialogue'][1])
                        if history['penult_dialogue']:
                            mesg.append(history['penult_dialogue'][0])
                            mesg.append(history['penult_dialogue'][1])
                        if history['last_dialogue']:
                            mesg.append(history['last_dialogue'][0])
                            mesg.append(history['last_dialogue'][1])
                    #加入当前对话信息
                    mesg.append({'role':"user",'content':text})
                else:
                    #添加历史对话信息
                    if history['last_dialogue']:
                        mesg.pop()
                        if history['previous_dialogue']:
                            mesg.append({'role':"user",'content':f"Here is our dialogue history,please bear this in  you mind.{history['previous_dialogue']}"})
                            mesg.append({"role": "assistant","content":"OK,I will keep it in mind"})
                        if history['antepenult_dialogue']:
                            mesg.append(history['antepenult_dialogue'][0])
                            mesg.append(history['antepenult_dialogue'][1])
                        if history['penult_dialogue']:
                            mesg.append(history['penult_dialogue'][0])
                            mesg.append(history['penult_dialogue'][1])
                        if history['last_dialogue']:
                            mesg.append(history['last_dialogue'][0])
                            mesg.append(history['last_dialogue'][1])
                        #加入当前对话信息
                        mesg.append({'role':"user",'content':text})
                    else:
                        mesg[-1]['content'] += text

                #mesg.insert(0,{'role':"user",'content':f"{system_mesg}.{language_request}"})
                body['messages'] = mesg
                # body['system'] = f"{system_mesg}"
        elif model_provider == 'cohere':
            body['prompt'] = text
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        else:
            raise Exception('Unknown model provider.')
        return body
    #获取大语言模型返回流
    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')
    #从返回流中获取文本形式回应
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
            chunk_obj = json.loads(chunk.get('bytes').decode())
            event_type = chunk_obj.get('type')
            if event_type == 'content_block_delta':
                return chunk_obj['delta'].get('text', '')
            elif event_type == 'content_block_start':
                return chunk_obj['content_block'].get('text', '')
            elif event_type == 'message_stop':
                printer('[DEBUG] Stream completed', 'debug')
                return ''
            else:
                return ''
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
    
class BedrockWrapper:
    def __init__(self):
        self.speaking = False
        self.voice_output_enabled = config['voice_output']
        self.audio_player = None
        self._initialize_audio_player()  # 初始化时立即尝试创建音频播放器
    #初始化音频播放器
    def _initialize_audio_player(self):
        if self.voice_output_enabled:
            try:
                self.audio_player = AudioPlayer()
                # 验证音频流是否活跃（确保设备可用）
                test_stream = self.audio_player.stream
                if not test_stream.is_active():
                    raise RuntimeError("音频设备未正确初始化")
                print("[DEBUG] 音频播放器初始化成功")
            except Exception as e:
                print(f"初始化音频播放器失败: {e}")
                self.voice_output_enabled = False
                config['voice_output'] = False
                self.audio_player = None
    #语音播放
    def play_text(self, text):

        if not config['voice_output'] or config['log_level'] == 'none':
            return
        if not text:
            return
        
        # 二次初始化音频播放器
        if self.audio_player is None and config['voice_output']:
            print("[WARN] 音频播放器未初始化，尝试重新初始化...")
            self._initialize_audio_player()
            if self.audio_player is None:
                print("[ERROR] 无法初始化音频播放器，跳过语音输出")
                self.voice_output_enabled = False
                config['voice_output'] = False
                return
            
        if self.audio_player:
            try:
                self.audio_player.play_text(text)
            except Exception as e:
                print(f"语音播放失败: {e}")
                # 禁用语音输出以避免重复错误
                self.voice_output_enabled = False
                config['voice_output'] = False
                self._cleanup_audio_player()
    #清理音频资源
    def _cleanup_audio_player(self):
        if self.audio_player:
            try:
                self.audio_player.stop()
            except:
                pass
            self.audio_player = None
    #调用大语言模型获取回应文本
    def invoke_bedrock(self, text, language='Chinese', history=None):
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        #特殊标记数学公式
        if '[MATH]' in text:
            text = f"这是一段包含数学公式的文本，请特别注意[MATH]标签内的内容：{text}",
        #构建请求体
        body = BedrockModelsWrapper.define_body(text, language, history)
        
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            #传递请求
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )
            #获取返回文本
            bedrock_stream = response.get('body')
            full_text = ""
            if bedrock_stream:
                for event in bedrock_stream:
                    chunk = BedrockModelsWrapper.get_stream_chunk(event)
                    if chunk:
                        text = BedrockModelsWrapper.get_stream_text(chunk)
                        full_text += text

            #语音输出
            if config['log_level'] != 'none' and config['voice_output'] and self.audio_player:
                try:
                    print("\n[DEBUG] 正在发送语音输出...")
                    self.audio_player.play_text(full_text)
                except Exception as e:
                    print(f"语音输出异常: {e}")
                    self.voice_output_enabled = False
                    config['voice_output'] = False

            printer('\n[DEBUG] Bedrock generation completed', 'debug')
            return full_text
        
        except Exception as e:
            print(f"Error invoking Bedrock: {e}")
        finally:
            self.speaking = False
#判断输入文本是否为文件地址
def is_file_path(s):
    if '/' in s or '\\' in s:
        return True
    common_extensions = ['.txt', '.csv', '.json', '.xml']
    if any(s.endswith(ext) for ext in common_extensions):
        return True
    if os.path.isabs(s) or os.path.exists(s):
        return True
    return False
#判断文件地址是否为图片
def is_image_file(file_path):
    """检查是否为支持的图片格式"""
    return any(file_path.lower().endswith(ext) for ext in ['.png','.jpg','.jpeg','.bmp','.tiff'])
#询问用户是否启用语音输出，返回布尔值
def ask_voice_output():
    while True:
        choice = input("是否启用语音输出？(y/n)回车默认否: ").strip().lower()
        if choice == '':  # 用户直接回车
            return False
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False
        print("⚠️ 请输入 y(yes) 或 n(no)！")  # 输入非法时提示
#询问用户是否启用RAG，返回布尔值
def ask_RAG_request():
    while True:
        choice = input("是否启用RAG？(y/n)回车默认否: ").strip().lower()
        if choice == '':  # 用户直接回车
            return False
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False
        print("⚠️ 请输入 y(yes) 或 n(no)！")  # 输入非法时提示
#询问用户是否调用微调数据，返回布尔值
def ask_fine_tunning():
    while True:
        choice = input("是否调用微调数据？(y/n)回车默认否: ").strip().lower()
        if choice == '':  # 用户直接回车
            return False
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False
        print("⚠️ 请输入 y(yes) 或 n(no)！")  # 输入非法时提示
#RAG使用的文本交互
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
#获取用户输入
def get_input():
    while True:
        user_input = input('请输入文件地址/问题/或输入"语音"开始录音: ').strip('"\'')
        
        # 语音输入处理
        if user_input.lower() in ["语音", "voice", "speech"]:
            print("\n请开始说话...（静音一段时间自动结束 或 按回车键手动结束）")
            try:
                # 延迟初始化语音输入模块
                if not hasattr(get_input, "speech_input"):
                    get_input.speech_input = SpeechInput()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                voice_text = loop.run_until_complete(SpeechInput().get_voice_input())
                print(f"\n识别结果: {voice_text}")
                return voice_text
            except Exception as e:
                print(f"语音识别错误: {e}")
                continue
        
        # 图片文件处理
        if is_file_path(user_input):
            print("\n检测到文件输入")
            if is_image_file(user_input):
                print("\n检测到图片输入。")
                print("\n正在识别图片内容...")
                # 延迟初始化OCR模块
                if not hasattr(get_input, "ocr_initialized"):
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                    get_input.ocr_initialized = True
                ocr_result = hybrid_ocr(user_input)
                print("\nOCR识别结果：")
                print(ocr_result)

                if '[MATH]' in ocr_result:
                    print("\n※ 检测到数学公式已用[MATH]标签标记")
            
                follow_up = input("\n请输入针对此内容的问题（直接回车使用原始文本）：")
                return f"{ocr_result} {follow_up}" if follow_up else ocr_result
            else:
                print("不支持的文件类型，请重新输入。")
        # 普通文本输入
        else:
            print("检测到文本输入。")
        return user_input
#主程序
#输入图片地址使用OCR/输入“语音”使用语音输入/输入文本问题
#输入exit退出
def process():
    #记忆历史信息
    history = {
        'previous_dialogue': None,
        'antepenult_dialogue': None,
        'penult_dialogue': None,
        'last_dialogue': None,
    }
    flag = True    #用于记忆
    #语音输入设置
    config['voice_output'] = ask_voice_output()
    print(f"语音输出: {"开启" if config['voice_output'] else "关闭"}")
    #Bedrock初始化
    bedrock_wrapper = BedrockWrapper()
    UserInputManager.set_executor(ThreadPoolExecutor(max_workers=1))
    RAG_request = ask_RAG_request()
    if RAG_request:
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
    else:
        while True:
            user_input = get_input()
            if user_input.lower() == 'exit':
                break

            printer(f'\n[INFO] User input: {user_input}', 'info')
             #语音检测
            lang = detect(user_input[-10:])  # 检测后10个字符
            language = f""
            if lang == 'en':
                language = f"English"
            elif lang =='zh' or lang =='zh-cn':
                language = f"Chinese"
            else :
                language = f"Chinese"
            #获取回应
            response = bedrock_wrapper.invoke_bedrock(user_input, language, history)
            print(response)

            #更新记忆
            sum_text = (
                f"请用中文整合下面两段对话或对话摘要，"
                f"记住用户的个人信息，"
                f"区分用户和助手的话，"
                f"记住对话的关键信息，"
                f"不要添加额外的信息："
            )
            if history['antepenult_dialogue']:
                if flag:
                    history['previous_dialogue'] = history['antepenult_dialogue']
                    flag = False
                else:
                    previous_dialogue = f"{sum_text}\n对话记录：{history['antepenult_dialogue']}\n更早的对话记录：{history['previous_dialogue']}\n"
                    history['previous_dialogue'] = bedrock_wrapper.invoke_bedrock(previous_dialogue)
            history['antepenult_dialogue'] = history['penult_dialogue']
            '''if history['last_dialogue']:
                penult_dialogue = f"{sum_text1}\n{history['last_dialogue']}\n"
                history['penult_dialogue'] = bedrock_wrapper.invoke_bedrock(penult_dialogue)'''
            history['penult_dialogue'] = history['last_dialogue']
            history['last_dialogue'] =(
            {
                "role": f"user",
                "content": f"{user_input}"
            },
            {
                "role": f"assistant",
                "content": f"{response}"
            }
        )
            #通过打印检查记忆
            '''
            if history['last_dialogue']: 
                print('last\n')
                print(history['last_dialogue'])
            if history['penult_dialogue']: 
                print('penult\n')
                print(history['penult_dialogue'])
            if history['antepenult_dialogue']: 
                print('antepenult\n')
                print(history['antepenult_dialogue'])
            if history['previous_dialogue']: 
                print('previous\n')
                print(history['previous_dialogue'])
            '''
# ================== 启动信息 ==================
info_text = f'''
*************************************************************
[系统配置]
- 当前模型: {config['bedrock']['api_request']['modelId']}
- AWS区域: {config['region']}
- OCR模式: {"Tesseract+EasyOCR" if config['ocr']['use_easyocr'] else "Tesseract"}
- 数学公式检测: {"开启" if config['ocr']['math_detection'] else "关闭"}
- 语音输出: {"开启" if config['voice_output'] else "关闭"}
- 日志级别: {config['log_level']}
[使用说明]
1. 直接输入问题进行文本对话。文本中包含\或//可能被误识为文件
2. 输入图片路径识别图中文字(图片路径请勿带有空格或特殊字符).
3. 输入"语音"或"voice""speech"进行语音输入
4. 输入 exit 退出程序
*************************************************************
'''

print(info_text)

if __name__ == "__main__":
    try:
        process()
    finally:
        # 清理音频模块
        if p:  
            p.terminate()
        # 清理OpenCV窗口    
        cv2.destroyAllWindows()  

