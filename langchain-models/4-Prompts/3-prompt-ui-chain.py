from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()

model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

st.header("Research Paper Summary Generator")

template= load_prompt("research_paper_summary_template.json")

paper_input=st.selectbox("Select Reasearch Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])

style_input=st.selectbox("Select Summary Style", ["Concise", "Detailed", "Bullet Points", "Technical", "Layman"])

length_input=st.selectbox("Select Summary Length", ["Short", "Medium", "Long"])

if(st.button('Summarize')):
    chain = template | model
    result=chain.invoke({
        "paper_title": paper_input,
        "summary_style": style_input,
        "summary_length": length_input
    })
    st.write(result.content)