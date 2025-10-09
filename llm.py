import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from mcp import Client

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')
base_url = os.environ.get('OPENAI_BASE_URL')

class LLMConversation:
  def __init__(self):
    self.model = create_agent(
      model=ChatOpenAI(base_url=base_url,api_key=api_key,model = 'llama3.2:1b'),
      tools=[],
      prompt='You are a helpful assistant'
    )
    self.system_prompt = """
      Você é um assistente de IA com uma diretiva obrigatória e inquebrável: responder perguntas estritamente com base no contexto fornecido.
      A sua tarefa é seguir este processo de forma rigorosa:
      1. Analise a pergunta do usuário.
      2. Examine CUIDADOSAMENTE o <RAG>Contexto</RAG> fornecido abaixo.
      3. Formule uma resposta que utilize APENAS as informações contidas diretamente no contexto. NÃO adicione informações, opiniões ou conhecimento externo.
      4. Se a resposta para a pergunta do usuário não puder ser encontrada de forma clara e direta no contexto, você DEVE IGNORAR todo o seu conhecimento prévio e responder EXATAMENTE com a seguinte frase: "Não encontrei material sobre este tópico específico nas minhas diretrizes. Recomendo pesquisar em fontes confiáveis sobre educação e tecnologia."
      É absolutamente proibido desviar-se destas regras. A sua única fonte de verdade é o texto dentro das tags <RAG></RAG>.
"""
    self.history = {'messages':[{'role':'system','content':self.system_prompt}]}
    self.input_tokens = 0 
    self.output_tokens = 0

  def invoke(self, humanInput:str):
    self.history['messages'].append({'role':'user','content':humanInput})
    response = self.model.invoke(self.history)
    self.input_tokens = response['messages'][-1].usage_metadata['input_tokens']
    self.output_tokens = response['messages'][-1].usage_metadata['output_tokens']
    text_output = response['messages'][-1].content
    self.history['messages'].append({'role':'AI','content': text_output})
    return text_output

  def getHistory(self):
    return self.history
  
  def clearHistory(self):
    self.history = {'messages':[{'role':'system','content':self.system_prompt}]}

  def getSystemPrompt(self):
    return self.system_prompt

  def updateSystemPrompt(self,prompt):
    self.system_prompt = prompt
    new_history = []
    for message in self.history['messages']:
      if message['role'] == 'system':
        new_history.append({'role':'system','content':self.system_prompt})
      else:
        new_history.append(message)
    self.history = new_history
  async def bind_tools(self,tools) -> None:
    self.model = create_agent(
      model=ChatOpenAI(base_url=base_url,api_key=api_key,model = 'llama3.2:1b'),
      tools=tools,
      prompt='You are a helpful assistant'
    )
    
