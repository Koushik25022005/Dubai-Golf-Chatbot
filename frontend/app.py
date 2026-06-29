import streamlit as st
import uuid
import sys
import os
import zipfile
import urllib.request
from openai import OpenAI
# Add parent dir to path to import knowledge base if needed
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from frontend.database import save_message, get_messages, save_feedback, get_all_sessions, delete_session
from dotenv import load_dotenv


# ---- LOADING ENVIRONEMNT VARIABLES ------
load_dotenv('/Users/koushik/ts_chat/.env', override=True)

for key, value in st.secrets.items():
    os.environ.setdefault(key, str(value))
# --- CLIENT AUTHENTICATION --- #
openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key= os.environ.get('TS_CHAT_OPENROUTER_API_KEY')   
        )

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state.get("password") == "admin123":
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop()  # Do not continue if password incorrect

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- MAIN APP ---
st.title("⛳ Dubai Golf Chatbot")

# --- SIDEBAR: SETTINGS & CHAT HISTORY ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
            "Model Selection",
            ["openai/gpt-oss-20b:free", "openai/gpt-oss-120b:free"]
        )
    response_style = st.selectbox("Response Style", ["Brief", "Detailed"])
    
    st.divider()
    
    if st.button("➕ New Chat Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
        
    st.subheader("Chat History")
    all_sessions = get_all_sessions()
    for session in all_sessions:
        # Highlight the current session
        sid = session["session_id"]
        is_current = sid == st.session_state.session_id

        if st.session_state.get("confirm_delete") == sid:
            # Confirmation step shown inline for this specific session
            st.warning(f"Delete '{session['title']}'? This can't be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, delete", key=f"confirm_yes_{sid}", use_container_width=True):
                    delete_session(sid)
                    del st.session_state["confirm_delete"]
                    # If we just deleted the session we were viewing, start a fresh one
                    if sid == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.pop("latest_prompt", None)
                        st.session_state.pop("latest_response", None)
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", key=f"confirm_no_{sid}", use_container_width=True):
                    del st.session_state["confirm_delete"]
                    st.rerun()
        else:
            col1, col2 = st.columns([5, 1])
            button_type = "primary" if is_current else "secondary"
            with col1:
                if st.button(f"💬 {session['title']}", key=session["session_id"], use_container_width=True, type=button_type):
                    st.session_state.session_id = session["session_id"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{sid}", use_container_width=True):
                    st.session_state["confirm_delete"] = sid
                    st.rerun()

# Load messages from DB for the current session
db_messages = get_messages(st.session_state.session_id)
messages = [{"role": m.role, "content": m.content} for m in db_messages]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

from knowledge_base.vector_db_onnx_bm25 import HybridSearchKnowledgeBase

CHROMA_DIR = "knowledge_base/chroma_db"
DB_URL = "https://github.com/Koushik25022005/Dubai-Golf-Chatbot/releases/download/kb-v1/chroma_db.zip"
 
if not os.path.exists(CHROMA_DIR):
    with st.spinner("Setting up knowledge base, this may take a moment..."):
        zip_path = "chroma_db.zip"
        urllib.request.urlretrieve(DB_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("knowledge_base")
        os.remove(zip_path)

@st.cache_resource
def get_knowledge_base():
    try:
        return HybridSearchKnowledgeBase()
    except Exception as e:
        st.error(f"Could not load Knowledge Base: {e}")
        return None

kb = get_knowledge_base()

# --- CHAT INPUT ---
if prompt := st.chat_input("How may I help you ?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message
    save_message(st.session_state.session_id, "user", prompt)
    
    # Simulate assistant response using Knowledge Base
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        context = ""
        if kb:
            try:
                # Retrieve from Vector DB
                results = kb.search(prompt, top_k=2)
                if results:
                    context = "\n\n".join([f"> {res}" for res in results])
                else:
                    context = "> No relevant information found in the knowledge base."
            except Exception as e:
                context = f"> Error retrieving data: {e}"
        # Build prompt structure
        system_prompt = ("You are a Dubai Golf chatbot. Answer the user question""using ONLY the context provided. If the answer is not in the context, ""politely say that you can contact the customer services team")

        if response_style == "Brief":
            system_prompt += "Keep your answer extremely brief (50-60 words). Include the relevant information that the user has asked for in the query."
        else:
            system_prompt += "Provide a detailed and comprehensive explanation."

        user_content = f"Context Information:\n{context}\n\nUser Questioni:\n{prompt}"

        # Call the OpenRouter API
        try:
            response = openrouter_client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                        ]
                )
            response_text = response.choices[0].message.content
        except Exception as e:
            response_text = f"❌ Error communicating with OpenRouter: {e}"

        # Display the result
        message_placeholder.markdown(response_text)

        save_message(st.session_state.session_id, "assistant", response_text)

        
        # Format the response based on the selected style
        if response_style == "Brief":
            # Just take the first bit of the context for a brief answer
            brief_context = context[:200] + "..." if len(context) > 200 else context
            response_text = f"**[{model_choice} - Brief]**\n\n{brief_context}"
        else:
            response_text = f"**[{model_choice} - Detailed]**\n\nBased on the retrieved context:\n\n{context}"
            
        message_placeholder.markdown(response_text)
        
    save_message(st.session_state.session_id, "assistant", response_text)
    
    # Store latest prompt and response in session to show feedback UI
    st.session_state["latest_prompt"] = prompt
    st.session_state["latest_response"] = response_text
    st.rerun()

# --- USER FEEDBACK ---
# Display feedback UI for the latest response
if "latest_response" in st.session_state:
    st.write("---")
    st.write("How was the response?")
    
    # Using columns for 5 stars
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i-1]:
            if st.button(f"{i} ⭐", key=f"star_{i}"):
                save_feedback(
                    st.session_state.session_id,
                    st.session_state["latest_prompt"],
                    st.session_state["latest_response"],
                    i
                )
                st.success("Thank you for your feedback!")
                # Remove feedback UI after submission
                del st.session_state["latest_response"]
                del st.session_state["latest_prompt"]
                st.rerun()
