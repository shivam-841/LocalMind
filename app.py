import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="LocalMind RAG", layout="wide")
st.title("🧠 LocalMind - Private Document AI")

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Sending to API for processing..."):
            # Send the file via HTTP POST
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/ingest", files=files)
            
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error("Ingestion failed.")

st.header("2. Ask Questions")
question = st.text_input("What would you like to know about your documents?")

if st.button("Submit") and question:
    with st.spinner("Asking API..."):
        response = requests.post(f"{API_URL}/ask", json={"question": question})
        
        if response.status_code == 200:
            data = response.json()
            
            st.markdown("### Answer")
            st.info(data["answer"])
            
            # Diagnostic UI: Let us see exactly what FAISS retrieved
            with st.expander("🔍 View Retrieved Context (Diagnostics)"):
                if not data.get("context"):
                    st.write("No context retrieved.")
                else:
                    for i, ctx in enumerate(data["context"]):
                        st.markdown(f"**Rank {i+1} | Source:** {ctx['source']} | **L2 Distance:** `{ctx['distance']:.4f}`")
                        # Use a quote block to display the exact chunk text
                        st.markdown(f"> {ctx['text']}")
        else:
            st.error("Failed to get an answer.")