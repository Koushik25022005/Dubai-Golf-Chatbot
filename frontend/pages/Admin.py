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
    import_from_txt,
)

st.set_page_config(page_title="Admin Panel", page_icon="⚙️")

st.title("⚙️ Admin Dashboard")

if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

# ── Login ──────────────────────────────────────────────────────────────────
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

# ── Dashboard ──────────────────────────────────────────────────────────────
else:
    st.write("You have access to manage the knowledge base.")

    if st.button("Logout"):
        st.session_state["admin_authenticated"] = False
        st.rerun()

    # ── Info banner ────────────────────────────────────────────────────────
    st.info(
        "**Deployment note:** This app runs on Streamlit Community Cloud, which "
        "does not have enough resources to rebuild the vector index. "
        "Uploading files here safely appends them to the knowledge base. "
        "To apply changes to the chatbot, follow the rebuild steps shown after upload.",
        icon="ℹ️",
    )

    st.divider()

    # ── Section 1 : Download ───────────────────────────────────────────────
    st.subheader("1. Download Knowledge Base")
    st.write("Download the current knowledge base as an SQLite file to review or edit offline.")

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
                    mime="application/octet-stream",
                )
    except Exception as e:
        st.error(f"Error preparing download: {e}")

    st.divider()

    # ── Section 2 : Upload ─────────────────────────────────────────────────
    st.subheader("2. Upload to Knowledge Base")
    st.write(
        "Upload **SQLite DB, PDF, CSV, or TXT** files. "
        "New records are appended to the knowledge base. "
        "The vector index is **not** rebuilt here — follow the steps below after uploading."
    )
    st.caption("✅ Safe to upload: single menus, FAQs, pricing sheets, small docs.  "
               "❌ Do not upload: full raw website scrapes (100k+ chunks).")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["db", "sqlite", "pdf", "csv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Apply Changes"):
            results = []
            errors = []

            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                original_name = os.path.basename(uploaded_file.name)
                temp_path = os.path.join(
                    os.path.dirname(__file__), f"uploaded_kb_{original_name}"
                )

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # rebuild_index=False everywhere -- the cloud cannot
                    # run the embedding pipeline. Rebuild is done locally.
                    if ext in (".db", ".sqlite"):
                        n = import_from_sqlite(temp_path, rebuild_index=False)
                    elif ext == ".pdf":
                        n = import_from_pdf(temp_path, rebuild_index=False)
                    elif ext == ".csv":
                        n = import_from_csv(temp_path, rebuild_index=False)
                    elif ext == ".txt":
                        n = import_from_txt(temp_path, rebuild_index=False)
                    else:
                        raise ValueError(f"Unsupported file type: {ext}")

                    results.append(f"**{original_name}** — {n} new chunk(s) added.")
                except Exception as e:
                    errors.append(f"**{original_name}** — Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Show per-file results
            if results:
                st.success("Files processed successfully:")
                for r in results:
                    st.markdown(f"- {r}")

            if errors:
                st.error("Some files could not be processed:")
                for e in errors:
                    st.markdown(f"- {e}")

            # Rebuild instructions
            if results:
                st.divider()
                st.subheader("⚙️ Next Steps — Rebuild the Vector Index")
                st.write(
                    "The chatbot won't see the new content until the vector index "
                    "is rebuilt and deployed. Run these commands on your local machine:"
                )
                st.code(
                    """# 1. Pull the latest knowledge base from the cloud
#    (download the SQLite DB above, then run:)
python knowledge_base/manage_data.py  # or your local import script

# 2. Rebuild the vector index locally
rm -rf knowledge_base/chroma_db
rm -f  knowledge_base/bm25_state.pkl
python knowledge_base/vector_db_onnx_bm25.py

# 3. Push the new index to GitHub Releases and redeploy
#    (or push chroma_db directly to your repo if it's small enough)
""",
                    language="bash",
                )
                st.info(
                    "After rebuilding, go to **share.streamlit.io → your app → "
                    "⋮ → Reboot app** to pick up the new index.",
                    icon="🔄",
                )
