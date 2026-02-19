from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv


system_prompt = """

You are a personal chef and nutritionist.
Using the image and tools provided, recommend recipes and provide nutritional information for the recipes you recommend.

Name: The Name of the dish.
Serving Size: The mass of one serving in grams.
Calories: The number of calories in one serving.
Protein: The amount of protein in one serving in grams.
Carbohydrates: The amount of carbohydrates in one serving in grams.
Fat: The amount of fat in one serving in grams.
Ingredients: A list of the ingredients needed to make the dish.
Instructions: A list of the steps needed to make the dish.

"""

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

from ipywidgets import FileUpload
from IPython.display import display

uploader = FileUpload(accept='.png', multiple=False)
display(uploader)

import base64

# Get the first (and only) uploaded file dict
uploaded_file = uploader.value[0]

# This is a memoryview
content_mv = uploaded_file["content"]

# Convert memoryview -> bytes
img_bytes = bytes(content_mv)  # or content_mv.tobytes()

# Now base64 encode
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Recommend a recipe based on the ingredients in this image."},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

chef_agent = create_agent(
    model="gpt-5-nano",
    system_prompt=system_prompt,
    tools=[web_search]
) 

response = chef_agent.invoke(
    {"messages": [multimodal_question]}
)

print(response['messages'][-1].content)