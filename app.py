# app.py
import streamlit as st
import os
import tempfile
from models import initialize_llm, initialize_embeddings
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from utils import display_source_documents

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="로컬 RAG 챗봇",
        page_icon="🤖",
        layout="wide"
    )
    
    # 앱 제목
    st.title("🤖 로컬 RAG 챗봇")
    st.subheader("LangChain과 Ollama를 사용한 로컬 지식 기반 챗봇")
    
    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
        
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = None
        
    if "llm" not in st.session_state:
        st.session_state.llm = None

    # 사이드바 설정
    with st.sidebar:
        st.header("📚 문서 및 설정")
        
        # 모델 선택
        llm_model = st.selectbox(
            "LLM 모델 선택",
            ["EEVE-Korean-10.8B:latest"], # 사용 가능한 다른 Ollama 모델 추가 가능
            index=0
        )
        
        embed_model = st.selectbox(
            "임베딩 모델 선택",
            ["jhgan/ko-sbert-nli"], # 사용 가능한 다른 HuggingFace 임베딩 모델 추가 가능
            index=0
        )
        
        # 모델 초기화 버튼
        if st.button("모델 초기화"):
            with st.spinner("모델 초기화 중..."):
                try:
                    # LLM 및 임베딩 모델 초기화
                    st.session_state.llm = initialize_llm(llm_model)
                    embeddings = initialize_embeddings(embed_model)
                    
                    # DocumentProcessor 인스턴스 생성 (임베딩 모델 전달)
                    st.session_state.doc_processor = DocumentProcessor(embeddings=embeddings)
                    
                    st.success("모델 초기화 완료!")
                except Exception as e:
                    st.error(f"모델 초기화 중 오류 발생: {str(e)}")
        
        # 구분선
        st.markdown("---")
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "참조할 문서를 업로드하세요",
            type=["pdf", "csv", "txt", "html", "md"],
            accept_multiple_files=True
        )
        
        # 문서 처리 버튼
        if uploaded_files and st.button("문서 처리"):
            if not st.session_state.doc_processor:
                st.error("먼저 모델을 초기화해주세요.")
            else:
                with st.spinner("문서 처리 중..."):
                    try:
                        # 벡터 저장소와 분할 문서 함께 반환
                        processing_result = st.session_state.doc_processor.process_documents(uploaded_files)
                        
                        if processing_result:
                            # RAG 시스템 초기화 시 분할 문서 전달
                            st.session_state.rag_system = RAGSystem(
                                llm=st.session_state.llm,
                                vectorstore=processing_result["vectorstore"],
                                splits=processing_result["splits"]
                            )
                            st.success("문서 처리 완료! 이제 질문할 수 있습니다.")
                        else:
                            st.error("문서 처리에 실패했습니다. 업로드된 파일을 확인해주세요.")
                            
                    except Exception as e:
                        st.error(f"문서 처리 중 오류 발생: {str(e)}")
        
        # 대화 초기화 버튼
        if st.button("대화 초기화"):
            if st.session_state.rag_system:
                st.session_state.rag_system.reset_memory()
            st.session_state.chat_history = []
            st.success("대화가 초기화되었습니다.")
    
    # 메인 화면 - 채팅 인터페이스
    chat_container = st.container()

    with chat_container:
        # 채팅 기록 표시
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if user_query := st.chat_input("질문을 입력하세요..."):
            # 입력 검증
            if not st.session_state.rag_system:
                st.error("먼저 모델을 초기화하고 문서를 처리해주세요.")
            else:
                # 사용자 메시지 표시 및 저장
                st.chat_message("user").markdown(user_query)
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # AI 응답 생성
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    
                    with st.spinner("답변 생성 중..."):
                        try:
                            # 쿼리 처리
                            result = st.session_state.rag_system.process_query(user_query)
                            response = result["answer"]
                            
                            # 응답 표시
                            response_container.markdown(response)
                            
                            # 출처 문서 정보 표시
                            with st.expander("참조 문서"):
                                display_source_documents(result["source_documents"])
                        
                        except Exception as e:
                            st.error(f"답변 생성 중 오류 발생: {str(e)}")
                            response = "죄송합니다, 답변을 생성하는 중에 오류가 발생했습니다."
                            response_container.markdown(response)
                
                # AI 응답 저장
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()