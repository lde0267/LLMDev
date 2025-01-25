from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
os.environ["OPENAI_API_KEY"] = "2sg8kxsseRytW3HOGXaGe1ESnMlAz9qGW1vpZ6EpkmQbCP2FfHdJJQQJ99BAACfhMk5XJ3w3AAAAACOGmcz2"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://youngwook-ai.openai.azure.com"


def create_chat_prompt(question, q_results, style):
    style_template = {
        "humor": """You are a funny guys. Explain {question} in a humorous way. Start with 아재개그, "
            "then give a clear explanation in simple terms. Use silly examples or comparisons "
            "to make the concept stick, and end with a pun or a funny remark.""",

        "sparta": """You are 스파르타 teacher. Explain {question} step by step in a straightforward and logical manner."
            "명령하는 어조로 써줘. Then, provide a detailed explanation with logical reasoning and practical examples."
            "Approach the explanation as if you are training a student for mastery, delivering commands with clarity and precision.""",

        "hood":  """You are a 후드 teacher. Explain {question} 쌈뽕. Use modern slang or "
            "뜨@겁나,  스@근하게,  맛@탱도리,  쌈@뽕과 같은 단어를 사용해서 설명해줘줘"
            "provide a detailed explanation with logical reasoning and practical examples.""",

        "neutral":  """You are a kind teacher. Explain {question} clearly and patiently, ensuring the student "
            "feels supported and encouraged throughout the explanation."""
    }
    q_results_text = q_results.get('result', '')
    system_message = SystemMessagePromptTemplate.from_template(style_template[style] + q_results_text)
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    return chat_prompt.format_prompt(question=question).to_messages()

def get_response(chat_prompt):
    """Generate a complete response from the AzureChatOpenAI model."""
    chatgpt = AzureChatOpenAI(
        deployment_name="dev-gpt-4o-mini",
        streaming=False  # Disable streaming for a full response
    )
    
    # Use the `call` method to get the complete response at once
    response = chatgpt(chat_prompt)
    
    # Extract the content from the response
    return response.content
