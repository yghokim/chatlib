from os import path, getcwd

from dotenv import find_dotenv, set_key

# Create env files if not exist ========================================================

env_file = find_dotenv()
if not path.exists(env_file):
    env_file = open(path.join(getcwd(), '.env'), 'w', encoding='utf-8')
    env_file.close()
    env_file = find_dotenv()

openai_api_key = input("Enter OpenAI API Key: ").strip()

if openai_api_key is not None and len(openai_api_key) > 0:
    set_key(env_file, "OPENAI_API_KEY", openai_api_key)
else:
    print("Skip setting up OpenAI API")

azure_llama2_host = input("Enter Azure Llama2 target address: ").strip()

if azure_llama2_host is not None and len(azure_llama2_host) > 0:
    set_key(env_file, "AZURE_LLAMA2_HOST", azure_llama2_host)
    azure_llama2_key = input("Enter Azure Llama2 key: ").strip()
    if azure_llama2_key is not None and len(azure_llama2_key) > 0:
        set_key(env_file, "AZURE_LLAMA2_KEY", azure_llama2_key)
else:
    print("Skip setting up Azure Llama2.")

print("Wrote .env file.")
