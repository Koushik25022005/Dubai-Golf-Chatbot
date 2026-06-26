import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from frontend.admin_auth import authenticate_admin
from knowledge_base.manage_data import (
    export_to_sqlite,
    import_from_sqlite,
    import_from_pdf,
    import_from_csv,
    rebuild_knowledge_base
)

st.set_page_config(page_title="Admin Panel", page_icon="⚙️")

st.title("⚙️ Admin Dashboard")

# Initialize session state for admin auth
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

# Login Form
if not st.session_state["admin_authenticated"]:
    st.subheader("Admin Login")
    with st.form("admin_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if authenticate_admin(username, password):
                st.session_state["admin_authenticated"] = True
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
else:
    # Admin Dashboard
    st.write("You have access to manage the knowledge base.")

    if st.button("Logout"):
        st.session_state["admin_authenticated"] = False
        st.rerun()

    st.divider()

    st.subheader("1. Download Knowledge Base")
    st.write("Download the current knowledge base as an SQLite file to edit offline.")

    export_path = os.path.join(os.path.dirname(__file__), "knowledge_base_editable.db")

    try:
        if st.button("Prepare Download"):
            with st.spinner("Generating SQLite file..."):
                export_to_sqlite(export_path)
            st.success("Ready for download!")

        if os.path.exists(export_path):
            with open(export_path, "rb") as file:
                st.download_button(
                    label="📥 Download SQLite DB",
                    data=file,
                    file_name="knowledge_base_editable.db",
                    mime="application/octet-stream"
                )
    except Exception as e:
        st.error(f"Error preparing download: {e}")

    st.divider()

    st.subheader("2. Upload Knowledge Base")
    st.write("Upload your modified SQLite file(s), PDFs, or CSVs to update the chatbot's knowledge. **Warning: New records will be appended. The vector index will be rebuilt automatically.**")

    uploaded_files = st.file_uploader("Upload files", type=['db', 'sqlite', 'pdf', 'csv'], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Apply Changes & Rebuild Index"):
            with st.spinner("Processing files and rebuilding Vector Index (this might take a minute)..."):
                try:
                    for uploaded_file in uploaded_files:
                        ext = os.path.splitext(uploaded_file.name)[1].lower()
                        original_name = os.path.basename(uploaded_file.name)
                        temp_path = os.path.join(os.path.dirname(__file__), f"uploaded_kb_{original_name}")

                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        try:
                            # rebuild_index=False on every branch -- we only
                            # want to rebuild once, after the whole batch is
                            # processed, not after every individual file.
                            if ext in (".db", ".sqlite"):
                                import_from_sqlite(temp_path, rebuild_index=False)
                            elif ext == ".pdf":
                                import_from_pdf(temp_path, rebuild_index=False)
                            elif ext == ".csv":
                                import_from_csv(temp_path, rebuild_index=False)
                            else:
                                raise ValueError(f"Unsupported file type: {ext}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                    # After all files are processed, rebuild the vector index once
                    rebuild_knowledge_base()
                    st.success("Knowledge Base successfully updated and re-indexed with all uploaded files!")
                except Exception as e:
                    st.error(f"Error updating knowledge base: {e}")
