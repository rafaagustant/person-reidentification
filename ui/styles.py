import streamlit as st


def apply_styles():
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.6rem; max-width: 1380px;}
        .metric-card {border: 1px solid rgba(120,120,120,.22); border-radius: 8px; padding: 14px 16px; background: rgba(127,127,127,.06);}
        .small-muted {color: #777; font-size: .92rem;}
        div.stButton > button {border-radius: 8px; height: 2.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
