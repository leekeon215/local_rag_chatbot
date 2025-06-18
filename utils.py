# utils.py
import streamlit as st
from langchain_core.documents import Document

def display_source_documents(docs: list[Document], max_chars=300):
    """출처 문서 정보 표시"""
    if not docs:
        st.info("참조된 문서가 없습니다.")
        return

    for i, doc in enumerate(docs):
        st.markdown(f"**📖 문서 {i+1}**")
        
        content_preview = doc.page_content[:max_chars]
        if len(doc.page_content) > max_chars:
            content_preview += "..."
        st.markdown(f"**내용:** {content_preview}")
        
        if hasattr(doc, 'metadata') and doc.metadata:
            st.markdown("**메타데이터:**")
            # 메타데이터를 좀 더 보기 좋게 표시
            meta_str = ", ".join([f"`{key}`: {value}" for key, value in doc.metadata.items()])
            st.markdown(meta_str)
        
        st.markdown("---")