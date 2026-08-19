import streamlit as st

st.title("Secrets Test")

st.write("DATABASE_URL:")
st.code(st.secrets.get("DATABASE_URL", "NOT FOUND").split("@")[-1])

st.write("SUPABASE_DATABASE_URL:")
st.code(
    st.secrets.get(
        "SUPABASE_DATABASE_URL",
        "NOT FOUND"
    ).split("@")[-1]
)