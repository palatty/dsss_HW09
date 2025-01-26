from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace 'YOUR_API_TOKEN' with your bot's API token
API_TOKEN = '7939705782:AAGGTz7_QPn1fp2-okphgvP5SCi5swXo97o'

# Load TinyLlama model and tokenizer
print("Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")

# Define a function to generate a response using the LLM
def generate_response(prompt):
    # Add a system prompt to guide the model
    system_prompt = "You are a helpful AI assistant. Provide detailed and informative responses to user queries."
    full_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"
    
    # Tokenize the input and generate a response
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    # Generate response with parameters optimized for longer outputs
    outputs = model.generate(
        **inputs,
        max_length=300,          # Increase max_length for longer replies
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the AI's response (remove the system prompt and user input)
    response = response.split("AI:")[-1].strip()
    return response

# Define a function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am your AI assistant. Send me a message!')

# Define a function to handle incoming messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(f"Received message: {user_message}")  # Log the received message

    # Generate a response using the LLM
    response = generate_response(user_message)
    print(f"Generated response: {response}")  # Log the generated response

    # Send the response back to the user
    await update.message.reply_text(response)

# Set up the bot
if __name__ == '__main__':
    print("Starting bot...")
    application = ApplicationBuilder().token(API_TOKEN).build()

    # Add handlers for commands and messages
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Bot is running...")
    application.run_polling()
