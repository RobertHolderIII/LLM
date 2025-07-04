import requests
from serpapi import GoogleSearch
import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
import os
import gradio as gr
import concurrent.futures
from pypdf import PdfReader

# env file is for local testing.
# for huggingface, upload API keys as secrets
load_dotenv('../../.env')

client = Groq(api_key = os.getenv('GROQ_API_KEY'))
os.environ["TOKENIZERS_PARALLELISM"] = "false" #prevents a warning message from langchain
chroma_client = None
COLLECTION_NAME = 'webpages'

def summarize(client, transcript):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text."
            },
            {
                "role": "user",
                "content": f"Please summarize the following articles:\n\n{transcript}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

def web_search(query="US tariff news, analysis, and predictions", language='en', max_search_res=10):
    if query == 'test':
        return [f'test-link-{i}' for i in range(max_search_res)]
        
    gl_dict = {
        'zh-cn': 'cn', # Language: Simplified Chinese
        'en': 'us'
    }
    params = {
        'q': query,
        'engine': 'google',
        'api_key': os.getenv('SERPAPI_API_KEY'),
        #'tbs': f'cdr:1,cd_min:{cd_min.strftime("%m/%d/%Y")},cd_max:{cd_max.strftime("%m/%d/%Y")}',
        'hl': language,                 
        'gl': gl_dict[language],
        'num': max_search_res
    }
        
    search = GoogleSearch(params)
    results = search.get_dict()
    res_keys = list(results.keys())
    print(f'received search results with keys {res_keys}')
    if 'error' in res_keys:
        error_msg = results["error"]
        print(error_msg)
        search_res = error_msg
    else:
        search_res = [result.get('link') for result in results["organic_results"]]

    return search_res


def _get_text_inner(url):
    print(f'downloading {url}...')
    response = requests.get(url, timeout=10)  # optional internal timeout
    html = response.text
    return BeautifulSoup(html, features="html.parser").get_text().strip()


def get_text(url, timeout_sec=3):
    if url.startswith('test-link-'):
        return 'url' + '-text-text-text'
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_get_text_inner, url)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f'\t**** download timed out for {url}')
            return '{skip}{timeout}'
        except Exception as e:
            print(f'\t**** could not download {url}: {e}')
            return '{skip}{could not download}'


def gen_status(status, running_status):
    yield status, f'{running_status}\n{status}'

## REFACTOR!
def create_vector_store_from_upload(files, running_status):
    
    status =  f'I see {len(files)} files, e.g. {files[0].name}'
    running_status += f'\n{status}'
    yield status, running_status

    documents = []
    ids = []
    metadatas = []

    
    
    for i,file in enumerate(files):
        status = f'processing {i+1}/{len(files)}: {file}...'
        running_status += f'\n{status}'
        yield status, running_status

        _, ext = os.path.splitext(file.name)
        #print(f'found {ext=}')
        
        contents = ''
        if ext == '.pdf':
            reader = PdfReader(file.name)
            for page in reader.pages:
                contents += '\n' + page.extract_text()

        elif ext == '.txt':
            contents = file.read().decode('utf-8')
            
        else:
            status = f'\tunsupported extention {ext}'
            running_status += f'\n{status}'
            yield status, running_status

        #print(f'found {contents=}')
            
        if contents:
            documents.append(contents)
            ids.append(str(i))
            metadatas.append({'source-url': file.name})
            
    yield from vector_store_helper(documents, ids, metadatas, running_status)

        
def create_vector_store(search_terms, query_lang, max_search_res, running_status):
    status = f'searching for documents using terms: {search_terms}'
    running_status += '\n' + status
    yield status, running_status
    
    urls = web_search(search_terms, query_lang, int(max_search_res))

    # check for web search error, such as bad API key
    if isinstance(urls, str):
        # handle error
        status = urls
        running_status += '\n' + status
        yield status, running_status
        return
    
    documents = []
    ids = []
    metadatas = []
    for i,url in enumerate(urls):
        status = f'downloading {i+1}/{len(urls)}: {url}...'
        txt = get_text(url)
        if '{skip}' in txt:
            status += txt
        else:
            documents.append(txt)
            ids.append(str(i))
            metadatas.append({'source-url': url})
        running_status += '\n' + status
        yield status, running_status

    yield from vector_store_helper(documents, ids, metadatas, running_status)

def vector_store_helper(documents, ids, metadatas, running_status):
        
    # prevents client getting stale
    # https://github.com/langchain-ai/langchain/issues/26884
    chromadb.api.client.SharedSystemClient.clear_system_cache() 

    global chroma_client
    chroma_client = chromadb.Client()

    # get rid of old collection
    try:
        chroma_client.get_collection(COLLECTION_NAME)
    except:
        # Collection does not exist
        pass
    else:
        chroma_client.delete_collection(COLLECTION_NAME)
        
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
        
    # TODO need to handle the case where search_res has a URL that we did not use
    # `add` uses Chroma's default sentence embedding model
    status = f'Indexing documents...'
    running_status += '\n' + status
    yield status, running_status

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    status = f'Document store recreated with {collection.count()} documents'
    running_status += '\n' + status
    yield status, running_status

def retrieve_documents(prompt, num_docs):
    collection = chroma_client.get_collection(COLLECTION_NAME)
    retrieval = collection.query(
        query_texts=[prompt],
        n_results=int(num_docs),
        include=["documents", "metadatas"]
    )
    return retrieval

def generate_summaries(prompt, num_docs):
    retrieval = retrieve_documents(prompt, num_docs)
    relevant_docs = retrieval['documents'][0]

    summaries = []
    for i, doc in enumerate(relevant_docs):
        yield f'summarizing {i+1}/{len(relevant_docs)} documents','In progress...'
        if prompt == 'test':
            summary = f'summary of {doc}'
        else:
            summary = summarize(client,doc[:5000])
        summaries.append(summary)

    
    # format summaries
    out = ''
    metadatas = retrieval['metadatas'][0]
    urls = [data['source-url'] for data in metadatas]
    for idx, (url, summary) in enumerate(zip(urls, summaries)):
        out += (f'{idx+1}: {url}\n---------------------\n{summary}\n\n')

    yield f'summarized {len(relevant_docs)} documents', out


def comprehensive_summary(filter_prompt, sum_prompt, num_docs, running_status):

    status_msg = f'choosing relevant documents based on prompt: {filter_prompt}...'
    running_status += '\n' + status_msg
    yield status_msg, 'In progress...', running_status
    
    retrieval = retrieve_documents(filter_prompt, num_docs)
    relevant_docs = retrieval['documents'][0]

    # limit is 12000 tokens ~ 48000 characters
    char_per_doc = int(12000 / len(relevant_docs) * 0.7)
    
    relevant_docs = [doc[:char_per_doc] for doc in relevant_docs]

    transcript = '\n------\n'.join(relevant_docs)

    status_msg = f'generating summaries based on prompt: {sum_prompt}...'
    running_status += '\n' + status_msg
    yield status_msg, 'In progress...', running_status
    
    if filter_prompt == 'test' or sum_prompt == 'test':
        summary = 'test-comprehensive-summary'
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes text."
                },
                {
                    "role": "user",
                    "content": f"{sum_prompt}:\n\n{transcript}",
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        summary = chat_completion.choices[0].message.content
    status_msg = f'generated summary of {len(relevant_docs)} documents based on prompt: "{sum_prompt}"'
    
    yield status_msg, summary, f'{running_status}\n{status_msg}'

    
def disable_btn(b):
    return gr.update(interactive=False)
def enable_btn(b):
    return gr.update(interactive=True)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## Tarrif Intel Tool

        Tool will
        - perform a web search for documents
        - pick the most relevant ones
        - generate summaries
        """
        )


    with gr.Tab('Analysis'):
        #
        # search and vector store
        #
        with gr.Row('Search'):
            search_terms = gr.Dropdown(label='search query',
                                       show_label=True,
                                       allow_custom_value=True,
                                       choices=["US tariff news, analysis, and predictions",
                                                "关税新闻、分析与预测",
                                                "world reaction to US tariff policy"
                                                ]
                                       )
            search_language = gr.Dropdown(label='query language',
                                       show_label=True,
                                       #allow_custom_value=True,
                                       choices=["en",
                                                "zh-cn",
                                                ]
                                       )
            num_search_res = gr.Dropdown(label='maximum search results',
                                         show_label=True,
                                         allow_custom_value=True,
                                         choices=[10, 25, 50, 100]
                                         )

        with gr.Row('Execute'):
            update_btn = gr.Button('Search for documents')
            gr.Markdown('Or upload file archive:')
            upload_btn = gr.File(file_count='multiple', height=120)
        status = gr.Textbox(show_label=False)
            
            
        #
        # summarization
        #
        with gr.Row():
            num_relevant_docs = gr.Dropdown(label='extract relevant documents',
                                            show_label=True,
                                            allow_custom_value=True,
                                            choices=[5, 10, 25]
                                            )
            prompt = gr.Dropdown(label='filter documents',
                                 show_label=True,
                                 allow_custom_value=True,
                                 choices=["what are the most important bits of news to know about US tariff policy and analysis of future actions?"
                                          ]
                                 )
            sum_prompt = gr.Dropdown(label='summarization',
                                 show_label=True,
                                 allow_custom_value=True,
                                 choices=["Please provide an overall summary of the points from the following articles in English",
                                          "Please give the top 5 consensus points from the following articles in English.  Please also provide any outlying points of interest and their source"
                                          ]
                                 )
        sum_btn = gr.Button('Generate summaries', interactive=False)
                
                
        sum_status = gr.Textbox(show_label=False)
        out = gr.TextArea(label='Summarizations', show_label=True)


    #
    # Tab for logging information
    #
    with gr.Tab('Log'):
        all_status = gr.TextArea(lines=25, show_label=False)
        

    #
    # event listeners 
    #
    update_btn.click(
        fn=disable_btn,
        inputs=update_btn,
        outputs=sum_btn
    ).then(
        fn=create_vector_store,
        inputs=[search_terms, search_language, num_search_res, all_status],
        outputs=[status, all_status]
    ).then(
        fn=enable_btn,
        inputs=update_btn,
        outputs=sum_btn
    )

    upload_btn.upload(
        fn=disable_btn,
        inputs=update_btn,
        outputs=sum_btn
    ).then(
        fn=create_vector_store_from_upload,
        inputs=[upload_btn, all_status],
        outputs=[status, all_status]
    ).then(
        fn=enable_btn,
        inputs=update_btn,
        outputs=sum_btn
    )

                                     
    sum_btn.click(
        #fn=generate_summaries,
        fn=comprehensive_summary,
        inputs=[prompt, sum_prompt, num_relevant_docs, all_status],
        outputs=[sum_status, out, all_status]
    )
    
    gr.Markdown(
        """
        [2025-05-17] initial deployment<br>
        """
        )
    
if __name__ == "__main__":
    print('v2025-05-17--2102')
    print('use `gradio app.py` to run in reload mode')
    demo.launch()
