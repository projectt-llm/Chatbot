# =============================================================================
# COPYRIGHT NOTICE
# -----------------------------------------------------------------------------
# This source code is the intellectual property of Aditya Pandey.
# Any unauthorized reproduction, distribution, or modification of this code 
# is strictly prohibited.
# If you wish to use or modify this code for your project, please ensure 
# to give full credit to Aditya Pandey.
#
# PROJECT DESCRIPTION
# -----------------------------------------------------------------------------
# This code is for a chatbot crafted with powerful prompts, designed to 
# utilize the Gemini API. It is tailored to assist cybersecurity researchers.
#
# Author: Aditya Pandey
# =============================================================================

#Import Frameworks
import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from google.generativeai import GenerativeModel
from langchain.chains import SequentialChain

# Function to load OpenAI model and get respones

def get_gemini_response(input,image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input!="":
       response = model.generate_content([input,image])
    else:
       response = model.generate_content(image)
    return response.text

#streamlit framework
st.title(" CyberNuKe ðŸ¤– ")
input_text=st.text_input("Search your Security Topic")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
submit=st.button("Tell me about the image")
if submit:

    response=get_gemini_response(input,image)
    st.subheader("The Response is")
    st.write(response)


# Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['Topic'],
    template="Tell me everything about and explain in so informative descriptive way about {Topic}"
)

# Function to configure and save the API key in local storage
def save_api_key(api_key):
    # Save the API key in local storage
    st.session_state.gemini_api_key = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Check if the API key is already saved in session_state
if 'gemini_api_key' in st.session_state:
    saved_api_key = st.session_state.gemini_api_key
    os.environ["GOOGLE_API_KEY"] = saved_api_key
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Check if the API key is valid (e.g., perform a test request)
    is_valid_api = True  # Replace this with your actual validation logic

    if is_valid_api:
        st.success("APIs are loaded. You can now trigger your queries.")
    else:
        st.error("Key got infected by zombies. Please check your API key.")

else:
    # If not saved, allow the user to enter the API key
    api_key = st.text_input("Enter Your Gemini Pro API Key:", type="password")
    if api_key:
        save_api_key(api_key)
        st.success("API key saved. Hold on to load APIs.")

# Select the model
model = genai.GenerativeModel('gemini-pro')

# Memory

Topic_memory = ConversationBufferMemory(input_key='Topic', memory_key='chat_history')
Policy_memory = ConversationBufferMemory(input_key='security policies', memory_key='chat_history')
Practice_memory = ConversationBufferMemory(input_key='Practice', memory_key='description_history')

## GEMINI LLMS
llm = ChatGoogleGenerativeAI(model="gemini-pro")
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='security policies',memory=Topic_memory)

# Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=['Policy'],
    template="write best {security policies} and perfect code snippet for implementing secure coding to this {Topic} "
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='Practice',memory=Policy_memory)
# Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=['Practice'],
    template="Implement  5 major best Cybersecurity {Practice} for this {Topic} to mitigate this . also give me a major cyberattack which is done by this {Topic} in well informative way . "
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=Practice_memory)
parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['Topic'],output_variables=['security policies','Practice','description'],verbose=True)


if input_text:
    st.text(parent_chain({'Topic':input_text}))

    with st.expander('Your Topic'):
        st.info(Topic_memory.buffer)

    with st.expander('Major Practices'):
        st.info(Practice_memory.buffer)
st.markdown("---")
st.text("                           Created with â¤ï¸  ")
