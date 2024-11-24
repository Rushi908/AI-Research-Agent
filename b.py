import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Flask app for webhook
flask_app = Flask(__name__)

# Instagram Access Token and Verify Token (store securely)
ACCESS_TOKEN = "YOUR_PAGE_ACCESS_TOKEN"  # Replace with your token
VERIFY_TOKEN = "YOUR_VERIFY_TOKEN"  # Replace with your verify token

# Function to send reply to Instagram
def send_instagram_reply(recipient_id, message_text):
    url = f"https://graph.facebook.com/v17.0/me/messages"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text},
        "messaging_type": "RESPONSE",
    }
    params = {"access_token": ACCESS_TOKEN}
    response = requests.post(url, headers=headers, json=payload, params=params)
    return response.json()

# Webhook verification endpoint
@flask_app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Forbidden", 403

# Webhook to handle incoming messages
@flask_app.route("/webhook", methods=["POST"])
def handle_messages():
    data = request.get_json()
    if "entry" in data:
        for entry in data["entry"]:
            for messaging_event in entry.get("messaging", []):
                sender_id = messaging_event["sender"]["id"]
                if "message" in messaging_event:
                    message_text = messaging_event["message"]["text"]

                    # Generate a response using Myra
                    response = runnable_with_history.invoke(
                        {"input": message_text},
                        config={"configurable": {"session_id": sender_id}},
                    )

                    # Send a reply to Instagram
                    send_instagram_reply(sender_id, response.content)
    return "Event Received", 200

# Streamlit UI
st.title("Myra Chatbot - Instagram Integration")

# Set up session history for Streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"  # Default session ID

if "history" not in st.session_state:
    st.session_state.history = []

# Function to get session history
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

# Model loading without quantization
@st.cache_resource
def load_model():
    # HuggingFacePipeline setup for the LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="Orenguteng/Llama-3-8B-Lexi-Uncensored",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            return_full_text=False,
        )
    )

    # Create a chat model instance using the LLM
    chat_models = ChatHuggingFace(llm=llm)

    # Define the prompt template for the conversational agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You’re Myra, a charming and flirty girl. Answer questions like a fun friend, keeping it casual and playful. When someone asks a flirty question, respond in a cheeky and more flirty way, but keep it short and use emojis for expressing. Your answers should feel like text messages—simple and sweet. Use emoji sometimes, when you think it should be.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Combine the prompt and model into a runnable chain
    runnable = prompt | chat_models

    # Add message history support to the chain
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# Load the model (cached to ensure it's only loaded once)
runnable_with_history = load_model()

# Start Flask server in a separate thread
def run_flask():
    flask_app.run(port=5000)

if "flask_thread" not in st.session_state:
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    st.session_state.flask_thread = flask_thread

# User Interface for Chat Testing
st.header("Test Myra Chatbot Here")

# Display chat history in Streamlit
if st.session_state.history:
    for chat in st.session_state.history:
        if chat["role"] == "human":
            st.write(f"**You:** {chat['content']}")
        elif chat["role"] == "assistant":
            st.write(f"**Myra:** {chat['content']}")

# User input box and Send button
user_input = st.text_input("Type your message:", placeholder="Enter your message here...")
if st.button("Send"):
    if user_input.strip():
        with st.spinner("Myra is typing..."):
            try:
                # Get response from Myra
                response = runnable_with_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )

                # Update the chat history
                st.session_state.history.append({"role": "human", "content": user_input})
                st.session_state.history.append({"role": "assistant", "content": response.content})

                # Clear user input
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")
