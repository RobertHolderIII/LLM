import os
from groq import Groq
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import gradio as gr

client = Groq(api_key = os.getenv('GROQ_API_KEY'))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index('ds9-documents')
docsearch = PineconeVectorStore(index=index, 
                                embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

def transcript_chat_completion(user_question, history):
    
    relevent_docs = docsearch.similarity_search(user_question)
    delimiter =  '\n\n------------------------------------------------------\n\n'
    num_docs = 12
    relevant_transcripts = delimiter.join([doc.page_content for doc in relevent_docs[:num_docs]])
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''Use this transcript or transcripts to answer any user questions, citing specific quotes:

                {transcript}
                '''.format(transcript=relevant_transcripts)
            },
            {
                "role": "user",
                "content": user_question,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

gr.ChatInterface(
    fn=transcript_chat_completion, 
    type="messages"
).launch()


