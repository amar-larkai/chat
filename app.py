import os
import gradio as gr
import random
import time
import pickle
import streamlit as st

from utils import  get_model

chain = get_model()

def load_embeddings_from_pickle(file_path):
    with open(file_path, "rb") as file:
        embeddings = pickle.load(file)
    return embeddings


embeddings = load_embeddings_from_pickle("embeddings.pkl")

# with gr.Blocks(title="Larkai chatbot ") as demo:
#     gr.Markdown("ECG Assistant")
#     chatbot = gr.Chatbot(title="Query")
#     msg = gr.Textbox()

#     clear = gr.Button("Clear")
    
#     embeddings = load_embeddings_from_pickle("embeddings.pkl")
#     def respond(message, chat_history):
#         docs = embeddings.similarity_search(message)
#         response = chain.run(input_documents=docs, question=message)
#         chat_history.append((message, response))
#         time.sleep(1)
#         return "",chat_history

#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
#     clear.click(lambda: None, None, chatbot, queue=False)

# demo.launch(share=True)
def app():
    # Set the app title
    # st.set_page_config(page_title="Katonic Support Bot")

    # Add a title and description
    st.title("Larkai Ecg chatbot")
    st.write("Ask any question regarding ecg the bot  will give you an answer:")

    # Add a text input for the user's question
    question = st.text_input("Your question", "")

    # Add a button to submit the question
    if st.button("Ask"):
        # Generate a random answer from the list
        docs = embeddings.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)

        # Display the answer
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {answer}")
        
# Run the app
if __name__ == '__main__':
    app()
