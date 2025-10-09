import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from retrieval import Retriever

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')
base_url = os.environ.get('OPENAI_BASE_URL')

retriever = Retriever()

class LlmConversation:
  """A class that menages both the usage, 
  atachables and parameters of the LLM"""
  def __init__(self):
    """Initialize the LlmConversation instance with
      default model, tools, and system prompt."""
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
    self.collection = None

  def invoke(self, human_input:str):
    """Invoke the LLM with the given human input, update 
    history and token counts, and return the response.

    Args:
        humanInput (str): The input message from the human user.

    Returns:
        str: The text output from the LLM.
    """
    #todo: fazer método melhor de escolher qual método usar
    self.history['messages'].append({'role':'user','content':human_input})
    rag = retriever.sentence_window_retrieval(query=human_input,
                                              collection_name=self.collection)
    self.history['messages'].append({'role':'RAG','query':human_input,'result': rag})
    response = self.model.invoke(self.history)
    self.input_tokens = response['messages'][-1].usage_metadata['input_tokens']
    self.output_tokens = response['messages'][-1].usage_metadata['output_tokens']
    text_output = response['messages'][-1].content
    self.history['messages'].append({'role':'AI','content': text_output})
    return text_output

  def get_history(self):
    """Return the conversation history.

    Returns:
        dict: The history dictionary containing messages.
    """
    return self.history

  def clear_history(self):
    """Clear the conversation history, resetting it to the
        initial state with the system prompt."""
    self.history = {'messages':[{'role':'system','content':self.system_prompt}]}

  def get_system_prompt(self):
    """Return the current system prompt.

    Returns:
        str: The system prompt string.
    """
    return self.system_prompt

  def update_system_prompt(self,prompt):
    """Update the system prompt and modify the history accordingly.

    Args:
        prompt (str): The new system prompt to set.
    """
    self.system_prompt = prompt
    new_history = []
    for message in self.history['messages']:
      if message['role'] == 'system':
        new_history.append({'role':'system','content':self.system_prompt})
      else:
        new_history.append(message)
    self.history = new_history

  async def bind_tools(self,tools) -> None:
    """Bind new tools to the model agent.

    Args:
        tools: The tools to bind to the agent.
    """
    self.model = create_agent(
      model=ChatOpenAI(base_url=base_url,api_key=api_key,model = 'llama3.2:1b'),
      tools=tools,
      prompt='You are a helpful assistant'
    )
