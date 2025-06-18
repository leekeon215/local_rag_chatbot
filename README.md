# Local RAG Chatbot
LangChain과 Ollama를 활용한 로컬 지식 기반 챗봇 프로젝트입니다. 다양한 문서 파일을 업로드하여, 해당 지식에 기반한 대화형 질의응답이 가능합니다.

## 주요 특징
**로컬 LLM 및 임베딩**: Ollama 기반 LLM과 HuggingFace 임베딩을 사용하여 완전한 로컬 환경에서 동작.

**다양한 문서 지원**: PDF, CSV, TXT, HTML, Markdown 등 다양한 문서 포맷 지원.

**하이브리드 검색**: Dense(Qdrant)와 Sparse(BM25) 검색을 결합한 Ensemble Retriever 적용.

**Streamlit UI**: 웹 기반 인터페이스로 손쉬운 사용.

## 설치 방법
### 레포지토리 클론
```
git clone https://github.com/leekeon215/local_rag_chatbot.git
cd local_rag_chatbot
```
### 필수 패키지 설치
```
pip install -r requirements.txt
```
### Ollama 설치 및 모델 다운로드
Ollama 공식 사이트에서 설치 후, 원하는 LLM 모델 다운로드.

## 실행 방법
```
streamlit run app.py
```
웹 브라우저에서 http://localhost:8501로 접속
