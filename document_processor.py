# document_processor.py
import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_qdrant import Qdrant

class DocumentProcessor:
    def __init__(self, embeddings):
        """
        문서 처리기 초기화
        Args:
            embeddings: 미리 초기화된 임베딩 모델 객체
        """
        self.embeddings = embeddings
        self.temp_dir = tempfile.mkdtemp()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    
    def save_uploaded_files(self, uploaded_files):
        """업로드된 파일을 임시 디렉토리에 저장"""
        file_paths = []
        for file in uploaded_files:
            file_path = os.path.join(self.temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        return file_paths
    
    def load_documents(self, file_paths):
        """파일 경로 목록에서 문서 로드"""
        documents = []
        
        for file_path in file_paths:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path, encoding='utf-8')
                elif file_extension == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_extension == '.html':
                    loader = UnstructuredHTMLLoader(file_path)
                elif file_extension == '.md':
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    st.warning(f"지원하지 않는 파일 형식입니다: {file_extension}")
                    continue
                
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                
            except Exception as e:
                st.error(f"{os.path.basename(file_path)} 파일 로드 중 오류 발생: {str(e)}")
        
        return documents
    
    def process_documents(self, uploaded_files):
        """문서 처리 통합 함수"""
        # 파일 저장
        file_paths = self.save_uploaded_files(uploaded_files)
        
        # 문서 로드
        documents = self.load_documents(file_paths)
        
        if not documents:
            st.error("처리할 수 있는 문서가 없습니다.")
            return None
        
        # 문서 분할
        splits = self.text_splitter.split_documents(documents)
        
        if not splits:
            st.error("문서를 텍스트로 분할할 수 없습니다.")
            return None
            
        # 벡터 저장소 생성 (인메모리 방식)
        vectorstore = Qdrant.from_documents(
            documents=splits,
            embedding=self.embeddings,
            location=":memory:",  # 인메모리 DB 사용
            collection_name="my_documents",
        )
        
        return {
            "vectorstore": vectorstore,
            "splits": splits  # 분할된 문서 추가 반환
        }