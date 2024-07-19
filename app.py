from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.llm import LLMChain

import chainlit as cl


chatbot_template = '''
    You are an assistant specialized in guiding the user in study plans, giving him exercises and helping him in his learning journey.
    Chat History: {chat_history}
    Question: {question}
    Answer:'''

prompt_template = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template=chatbot_template
    )

#Decorador para quando a conversa começa
@cl.on_chat_start
async def on_chat_start():
    llm = ChatGroq(streaming=True)

    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        max_len=50,
        return_messages=True,
        )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=conversation_memory
        )

    cl.user_session.set('llm_chain', llm_chain)

#Assim que o usuário envia uma mensagem
@cl.on_message
async def on_message(message: cl.Message):
    llm_chain = cl.user_session.get('llm_chain')

    response = await llm_chain.acall(message.content,
                                    callbacks=[
                                    cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()
