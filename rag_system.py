# rag_system.py
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class RAGSystem:
    def __init__(self, llm, vectorstore, splits):
        self.llm = llm
        self.vectorstore = vectorstore
        self.splits = splits
        self.ensemble_retriever = self._create_hybrid_retriever()
        self.memory = self._create_memory()
        self.chain = self._create_chain()

    def _create_hybrid_retriever(self):
        # Qdrant (Dense) Retriever
        qdrant_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5}
        )

        # BM25 (Sparse) Retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents=self.splits,
            k=3,
            # preprocess_func는 문자열을 토큰 리스트로 변환해야 합니다.
            preprocess_func=lambda text: text.lower().strip().split()
        )

        # Ensemble Retriever (두 검색 결과를 결합)
        return EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.4, 0.6]
        )

    def _create_memory(self):
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer' # 체인의 출력 키와 일치
        )

    def _create_chain(self):
        # 복잡한 refine 체인 대신, 안정적이고 표준적인 ConversationalRetrievalChain을 사용
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.ensemble_retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True, # 디버깅을 위해 True로 유지, 배포 시 False로 변경 가능
        )

    def process_query(self, query):
        try:
            # .invoke()는 최신 LangChain에서 권장되는 호출 방식입니다.
            result = self.chain.invoke({"question": query})
            return result
        except Exception as e:
            raise Exception(f"쿼리 처리 중 오류 발생: {str(e)}")

    def reset_memory(self):
        """대화 기록을 초기화하고 체인을 재생성합니다."""
        self.memory = self._create_memory() # 일관된 방식으로 메모리 생성
        self.chain = self._create_chain()