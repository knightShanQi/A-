import streamlit as st

from a_share_predictor.database_source import load_env_file

st.set_page_config(
    page_title="A股智能研判工作台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_env_file()

from a_share_predictor.dashboard import main


if __name__ == "__main__":
    main()
