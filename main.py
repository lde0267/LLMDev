import streamlit as st
from utils import load_and_split_pdf, create_vector_db, load_vector_db, create_retrieval_qa
from prompts import create_chat_prompt, get_response

# 웹 인터페이스
st.title("Personal ChatBot")

with st.sidebar:

    #사이드바 타이틀
    st.sidebar.title("파일 업로드")

    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    # 스타일 선택
    st.title("Style")
    left, middle, right = st.columns(3)

    # 세션 상태 초기화 (세션 상태에 'style' 키가 없다면 기본값 설정)
    if 'style' not in st.session_state:
        st.session_state.style = "neutral"

    # 버튼을 눌러 선택된 스타일을 저장
    humor_button = left.button("😂", use_container_width=True, key="humor")
    sparta_button = middle.button("🤠", use_container_width=True, key="sparta")
    cool_button = right.button("😎", use_container_width=True, key="hood")

    if humor_button:
        st.session_state.style = "humor"
    elif sparta_button:
        st.session_state.style = "sparta"
    elif cool_button:
        st.session_state.style = "hood"
    
    style = st.session_state.style

    # 스타일 표시
    st.write(f"선택한 스타일은: {style}")

    st.divider()

    # 요약하는 칸
    st.title("Summary")
    txt, = st.text_area(
        "Text to analyze",
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        "was the epoch of incredulity, it was the season of Light, it was the "
        "season of Darkness, it was the spring of hope, it was the winter of "
        "despair, (...)",
    ),
    
    st.divider()

    #과목 선택 칸
    st.title("Class")
    option = st.selectbox(
    "Choose Class",
    ("통계학", "물리학", "정치학"),
    index=None,
    placeholder="Select contact method...",
    )

    st.write("You selected:", option)

    st.divider()

if uploaded_file:
    # PDF 파일 저장
    file_path = f"./uploaded_files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("파일 업로드 완료!")

    # # 데이터베이스 생성
    # if st.sidebar.button("데이터베이스 생성"):
    #     texts = load_and_split_pdf(file_path)
    #     create_vector_db(texts)
    #     st.success("데이터베이스 생성 완료!")

# 데이터베이스 불러오기
db = load_vector_db()
retriever = db.as_retriever()

# Initialize session state with an initial system message if it's not already set
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot only for you. How can I help you?"}
    ]

# Display all previous messages from the conversation history
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])


# 질의응답 수행
prompt = st.chat_input(placeholder="Ask me anything!")

if prompt:
    # Add the user's message to the history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    qa_chain = create_retrieval_qa(retriever)
    q_results = qa_chain(prompt)

    full_prmpt = create_chat_prompt(prompt, q_results, style)
    results = get_response(full_prmpt)

    st.session_state["messages"].append({"role": "assistant", "content": results})
    st.chat_message("assistant").write(results)
