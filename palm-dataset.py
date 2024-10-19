import os
import csv
import dotenv
import google.generativeai as palm

# load environment variables
dotenv.load_dotenv()
# load API key from environment variables
api_key = os.environ.get('GEMINI')

palm.configure(api_key=api_key)
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
print(model)


