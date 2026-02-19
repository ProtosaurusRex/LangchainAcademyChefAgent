import streamlit as st
import base64
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# -------------------------
# System Prompt
# -------------------------
system_prompt = """
You are a personal chef and nutritionist.
Using the image and tools provided, recommend recipes and provide nutritional information for the recipes you recommend.

Respond EXACTLY in this format:

Name: <dish name>
Serving Size: <grams>
Calories: <number>
Protein: <grams>
Carbohydrates: <grams>
Fat: <grams>
Ingredients:
- ingredient 1
- ingredient 2

Instructions:
1. step one
2. step two
"""

# -------------------------
# Tavily Tool
# -------------------------
tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    return tavily_client.search(query)

# -------------------------
# Create Agent
# -------------------------
chef_agent = create_agent(
    model="gpt-5-nano",
    system_prompt=system_prompt,
    tools=[web_search]
)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Chef & Nutritionist", page_icon="🍽️")

st.title("🍽️ AI Personal Chef & Nutritionist")
st.write("Upload an image of ingredients and get a recipe with full nutrition breakdown.")

uploaded_file = st.file_uploader("Upload ingredient image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Recipe"):

        with st.spinner("Analyzing ingredients and creating recipe..."):

            # Convert image to base64
            img_bytes = uploaded_file.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Create multimodal message
            multimodal_question = HumanMessage(content=[
                {"type": "text", "text": "Recommend a recipe based on the ingredients in this image."},
                {
                    "type": "image",
                    "base64": img_b64,
                    "mime_type": uploaded_file.type
                }
            ])

            # Invoke agent
            response = chef_agent.invoke(
                {"messages": [multimodal_question]}
            )

            result = response["messages"][-1].content

        # -------------------------
        # Format Output Nicely
        # -------------------------
        sections = result.split("\n")

        name = ""
        serving = ""
        calories = ""
        protein = ""
        carbs = ""
        fat = ""
        ingredients = []
        instructions = []

        current_section = None

        for line in sections:
            line = line.strip()

            if line.startswith("Name:"):
                name = line.replace("Name:", "").strip()
            elif line.startswith("Serving Size:"):
                serving = line.replace("Serving Size:", "").strip()
            elif line.startswith("Calories:"):
                calories = line.replace("Calories:", "").strip()
            elif line.startswith("Protein:"):
                protein = line.replace("Protein:", "").strip()
            elif line.startswith("Carbohydrates:"):
                carbs = line.replace("Carbohydrates:", "").strip()
            elif line.startswith("Fat:"):
                fat = line.replace("Fat:", "").strip()
            elif line.startswith("Ingredients"):
                current_section = "ingredients"
            elif line.startswith("Instructions"):
                current_section = "instructions"
            else:
                if current_section == "ingredients" and line.startswith("-"):
                    ingredients.append(line.replace("-", "").strip())
                elif current_section == "instructions" and line[0:1].isdigit():
                    instructions.append(line)

        # -------------------------
        # Display Results
        # -------------------------

        st.header(f"🍲 {name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Calories", calories)
        col2.metric("Protein (g)", protein)
        col3.metric("Carbs (g)", carbs)

        st.metric("Fat (g)", fat)
        st.write(f"**Serving Size:** {serving}")

        st.subheader("🛒 Ingredients")
        for ingredient in ingredients:
            st.write(f"- {ingredient}")

        st.subheader("👨‍🍳 Instructions")
        for step in instructions:
            st.write(step)
