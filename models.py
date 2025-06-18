# models.py
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_llm(model_name="EEVE-Korean-10.8B:latest"):
    """로컬 LLM 모델 초기화"""
    try:
        llm = ChatOllama(model=model_name)
        return llm
    except Exception as e:
        raise Exception(f"LLM 모델 초기화 중 오류 발생: {str(e)}")

def initialize_embeddings(model_name="jhgan/ko-sbert-nli"):
    """한국어 임베딩 모델 초기화"""
    try:
        # 모델 설정 파라미터
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    except Exception as e:
        raise Exception(f"임베딩 모델 초기화 실패: {str(e)}")