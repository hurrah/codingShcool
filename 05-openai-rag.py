import streamlit as st
from openai import OpenAI
import time
import io
import logging
from tavily import TavilyClient  # Tavily 클라이언트 라이브러리 추가
import requests  # 인터넷 검색을 위한 라이브러리 추가
import json  # 세션 저장을 위한 라이브러리 추가

# 로깅 설정
logging.basicConfig(level=logging.ERROR)

# Streamlit 앱 설정
st.set_page_config(page_title="OpenAI RAG 시스템", layout="wide")

# 세션 상태 초기화
if 'client' not in st.session_state:
    st.session_state.client = None
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'thread' not in st.session_state:
    st.session_state.thread = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'tavily_client' not in st.session_state:
    st.session_state.tavily_client = None
if 'saved_sessions' not in st.session_state:
    st.session_state.saved_sessions = {}

# OpenAI 및 Tavily 클라이언트 초기화
def initialize_clients(openai_api_key, tavily_api_key):
    st.session_state.client = OpenAI(api_key=openai_api_key)
    st.session_state.tavily_client = TavilyClient(api_key=tavily_api_key)

# OpenAI 관련 함수
def create_vector_store(client):
    try:
        return client.beta.vector_stores.create(name="PDF 문서")
    except Exception as e:
        logging.error(f"벡터 저장소 생성 오류: {e}")
        return None

def upload_and_process_files(client, files, vector_store_id):
    try:
        file_streams = [io.BytesIO(file.read()) for file in files]
        file_names = [file.name for file in files]
        return client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[(name, stream) for name, stream in zip(file_names, file_streams)]
        )
    except Exception as e:
        logging.error(f"파일 업로드 및 처리 오류: {e}")
        return None

def create_assistant(client):
    try:
        return client.beta.assistants.create(
            name="PDF 파일 검색 어시스턴트",
            instructions="업로드된 PDF 파일을 검색하고 관련 정보를 제공하는 어시스턴트입니다.",
            tools=[{"type": "file_search"}],
            model="gpt-4o"
        )
    except Exception as e:
        logging.error(f"어시스턴트 생성 오류: {e}")
        return None

def update_assistant(client, assistant_id, vector_store_id):
    try:
        return client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
    except Exception as e:
        logging.error(f"어시스턴트 업데이트 오류: {e}")
        return None

def create_thread(client):
    try:
        return client.beta.threads.create()
    except Exception as e:
        logging.error(f"스레드 생성 오류: {e}")
        return None

def add_message_and_get_response(client, thread_id, assistant_id, content):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            time.sleep(1)
        
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        return messages.data[0].content[0].text.value
    except Exception as e:
        logging.error(f"메시지 처리 오류: {e}")
        return "요청 처리 중 오류가 발생했습니다."

def stream_response(client, thread_id, assistant_id, content):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            time.sleep(1)
        
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        for message in messages.data[0].content:
            yield message.text.value
    except Exception as e:
        logging.error(f"메시지 처리 오류: {e}")
        yield "요청 처리 중 오류가 발생했습니다."

def internet_search(query):
    try:
        response = requests.get(f"https://api.example.com/search?q={query}")
        if response.status_code == 200:
            return response.json()["results"]
        else:
            return "인터넷 검색 중 오류가 발생했습니다."
    except Exception as e:
        logging.error(f"인터넷 검색 오류: {e}")
        return "인터넷 검색 중 오류가 발생했습니다."

# 사용자 입력 처리 함수
def process_user_input():
    if st.session_state.user_input and st.session_state.client and st.session_state.assistant and st.session_state.thread:
        user_input = st.session_state.user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.user_input = ""  # 입력 필드 초기화
        
        # Tavily Search를 사용하여 문서 검색
        tavily_results = st.session_state.tavily_client.search(user_input)
        if tavily_results:
            response = tavily_results
        else:
            # 문서에 없는 경우 인터넷 검색
            response = internet_search(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# 세션 저장 함수
def save_session():
    if st.session_state.messages:
        session_summary = " ".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
        session_name = session_summary[:50]  # 세션 이름을 요약된 내용의 첫 50자로 설정
        st.session_state.saved_sessions[session_name] = st.session_state.messages.copy()
        st.session_state.messages = []  # 현재 세션 초기화
        st.success(f"세션 '{session_name}'이(가) 저장되었습니다.")

# 세션 복원 함수
def load_session(session_name):
    if session_name in st.session_state.saved_sessions:
        st.session_state.messages = st.session_state.saved_sessions[session_name].copy()
        st.success(f"세션 '{session_name}'이(가) 복원되었습니다.")

# 세션 초기화 함수
def clear_session():
    st.session_state.messages = []
    st.success("세션이 초기화되었습니다.")

# 메인 UI
st.title("OpenAI RAG 시스템")

# 사이드바: API 키 입력 및 파일 업로드
with st.sidebar:
    st.title("설정")
    openai_api_key = st.text_input("OpenAI API 키", type="password")
    tavily_api_key = st.text_input("Tavily API 키", type="password")
    if openai_api_key and tavily_api_key:
        initialize_clients(openai_api_key, tavily_api_key)
    
    uploaded_files = st.file_uploader("PDF 파일 업로드", type="pdf", accept_multiple_files=True)

    st.title("저장된 세션")
    for session_name in st.session_state.saved_sessions.keys():
        if st.button(session_name):
            load_session(session_name)

    # 세션 저장 버튼 추가
    if st.button("세션 저장"):
        save_session()

# 오른쪽 상단에 세션 초기화 버튼 추가
st.button("세션 초기화", on_click=clear_session)

# 메인 로직
if st.session_state.client and uploaded_files:
    if st.session_state.vector_store is None:
        st.session_state.vector_store = create_vector_store(st.session_state.client)

    if st.session_state.vector_store:
        file_batch = upload_and_process_files(st.session_state.client, uploaded_files, st.session_state.vector_store.id)

        if st.session_state.assistant is None:
            st.session_state.assistant = create_assistant(st.session_state.client)

        if st.session_state.assistant:
            st.session_state.assistant = update_assistant(st.session_state.client, st.session_state.assistant.id, st.session_state.vector_store.id)

        if st.session_state.thread is None:
            st.session_state.thread = create_thread(st.session_state.client)

# 대화 히스토리 표시
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력 처리
prompt = st.chat_input("질문해주세요")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in stream_response(st.session_state.client, st.session_state.thread.id, st.session_state.assistant.id, prompt):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 스크롤을 최하단으로 이동
st.empty()