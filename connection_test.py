import streamlit as st
import psycopg2
from supabase import create_client

st.title("Connection Test")

try:
    psycopg2.connect(st.secrets["DATABASE_URL"]).close()
    st.success("✅ PostgreSQL: CONNECTED")
except Exception as e:
    st.error(f"❌ PostgreSQL: {e}")

try:
    create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    )
    st.success("✅ Supabase: CONNECTED")
except Exception as e:
    st.error(f"❌ Supabase: {e}")