import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import wikipedia
import arxiv
import time
from youtubesearchpython import VideosSearch
from groq import Groq


with st.spinner(" Loading..."):
     time.sleep(1)
load_dotenv()

st.set_page_config(page_title="AI agents", page_icon="🤖")
st.markdown("""
    <h1 style='text-align:center;
               background: linear-gradient(to right, #FF8C00, #00C853, #00BFA6, #FFD600);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: bold;
               font-size: 50px;'>
        🤖 ReAct AI Agent
    </h1>
""", unsafe_allow_html=True)

st.sidebar.header('Setting')

api_key= st.sidebar.text_input("GROQ Api Key \n (Optional)",type="password") or os.getenv("api_key","")

model_name=st.sidebar.selectbox("Model",["qwen/qwen3-32b","llama-3.1-8b-instant","openai/gpt-oss-20b"],index=0)
max_steps=st.sidebar.slider("Max Resoning steps",1,6,3)


st.success(""" ***FEATURES:*** \n This agent helps you find answers using four tools: **Arxiv** (for research papers), **Wikipedia** (for general info), **Web Search** (for up-to-date content), and **YouTube Search** (for searching video explanations). It will gather info from these Sources and give you **links** so you can check them out yourself.  """)

def tool_web_search(query,k=4):
    with DDGS () as ddg:
        results=ddg.text(query,region="us-en",max_results=k)
    lines=[]
    for r in results:
        title,link,body=r.get("title",""),r.get("href",""),r.get("body","")
    lines.append(f"-{title}-{link}\n {body}\n Source:{link}")
    return"\n".join(lines)if lines else "no Result Found"

def tool_wikipedia(query,sentence=2):
    try:
         wikipedia.set_lang("en")
         pages=wikipedia.search(query,results=1)
         if not pages:
             return"no wikipedia page found."
         summary=wikipedia.summary(pages[0],sentences=sentence)
         link = f"https://en.wikipedia.org/wiki/{pages[0].replace(' ', '_')}"
         return f"Wikipedia: {pages[0]}\n{(summary)} \n Source:{link}" 
    except Exception as e:
        return f"Wikipedia error: {e}"
    
def tool_arxiv(query):
    try:
        search=arxiv.Search(query=query,max_results=1, sort_by=arxiv.SortCriterion.Relevance)
        results=list(search.results())
        if not results:
            return"No Arxiv paper found."
        paper=results[0]
        snippet=(paper.summary or "").replace("\n","")[:400]
        return f"arxiv:{paper.title}\n Link:{paper.entry_id}\n{snippet}....\n Source:{paper.entry_id}"
    except Exception as e:
         return f"arxiv error: {e}"
    
def tool_youtube_search(query, k=4):
    search = VideosSearch(query, limit=k)
    results = search.result().get('result', [])
    lines=[]
    for r in results:
        title,link,desc_snippet=r.get("title",""),r.get("link",""),r.get( "descriptionSnippet", [])
        # Join all snippet texts into a single string
        desc = " ".join([d.get("text", "") for d in desc_snippet])
    lines.append(f"-{title}-{link}\n {desc} \n Source:{link}")
    return"\n".join(lines)if lines else "no Result Found"

system_prompt=""" you are a helpful research assistantwith access of 
1) websearch 2) Wikipedia 3)Arxiv 4) Youtube Search
- DO NOT call any external API or Groq tool directly.
- Only indicate your actions using this exact text format:
    Thought: <what you will do next>
    Action: <choose one of Websearch, Wikipedia, Arxiv,Youtube search>
    Action Input: <search phrase>
- Wait for the observation to be provided by the system before continuing.
- Repeat this loop until you can answer the user question.
- When ready to give the final answer, write exactly:
    Final Answer: <your short, clear answer in English>
    Source:<add source link of given information>

 """
 
action_re=re.compile(r"^Action:\s*(Websearch,WIkipedia,Arxiv,Youtube Search)",re.I)
input_re=re.compile(r"^Action Input:\s*(.*)",re.I)


def mini_agent(client,model,question,max_iter=3):
    """Run a small resoning loop manually(no langchain)."""
    transcript=[f"User Question:{question}"]
    observation=None
    for step in range(1,max_iter+1):
        convo = system_prompt +"\n"+"\n".join(transcript)
    if observation:
        convo +=f"\nObservation:{observation}"

    resp=client.chat.completions.create(model=model,messages=[
    {"role":"system","content":system_prompt},
    {"role":'user',"content": convo}
    ],
    temperature=0.2,
    max_tokens=9100)

    text=resp.choices[0].message.content or""
    with st.expander(f"Step{step}",expanded=False):
        st.write(text)

    if"Final Answer:"in text:
        return text.split("Final Answer:",1)[1].strip()
    action,action_input=None,None 
    for line in text.splitlines():
        if action_re.match(line):
            action=action_re.match(line).group(1).title()
            if input_re.match(line):
                action_input=action_re.match(line).group(1).strip()
                if not action or not action_input:
                    return"could not understand next step.try again."
                
                
    if action=="Websearch":
        observation=tool_web_search(action_input)
    elif action=="wikipedia":
        observation=tool_wikipedia(action_input)
    elif action=="Arxiv":
        observation=tool_arxiv(action_input)
    elif action=="tool_youtube_search":
        observation=tool_youtube_search(action_input)
    else:
        observation=f"Unkown Tools:{action}"

    transcript.append(f"Thought: i will use {action}.")
    transcript.append(f"Action:{action}")
    transcript.append(f"Action Input:{action_input}")
    transcript.append(f"Obsevation:{observation}")

    summary=client.chat.completions.create(model=model,messages=[{"role":"system","content":"summarize briefly in english."},{"role":"user","content":"\n".join(transcript)},],
                                           temperature=0.2,
                                           max_tokens=342,)
    return summary.choices[0].message.content

query=st.chat_input("Ask me a Question and and I will Answer it." )
if query:
    st.chat_message("user").write(query)

if not api_key:
    st.error("Please enter valid API KEY in sidebar or .env")
else:
    client=Groq(api_key=api_key)
    with st.spinner("Thinking...."):
        answer=mini_agent(client,model=model_name,question=query,max_iter=max_steps)
        st.chat_message("assistant").write(answer)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
       bottom: 10px;
        center: 10px;
        font-size: 12px;
        color: gray;
        z-index: 99999;
    }
    </style>
    <div class="footer">Created by Joan Abid</div>
    """,
    unsafe_allow_html=True
)









