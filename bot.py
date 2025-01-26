


from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackContext
from telegram.ext.filters import TEXT
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the LLM (replace "gpt2" with your chosen model)
MODEL_NAME = "gpt2"  # Use "tiny-llama" if applicable
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Telegram Bot Token (from BotFather)
TELEGRAM_TOKEN = "7939705782:AAGGTz7_QPn1fp2-okphgvP5SCi5swXo97o"

# Async start command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! I am your AI Assistant. How can I help you?")

# Async message handler
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text

    # Prepare input for the LLM
    inputs = tokenizer.encode(user_input, return_tensors="pt")

    # Generate response from the LLM
    outputs = model.generate(
        inputs,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Send the response back to the user
    await update.message.reply_text(response)

# Main function to run the bot
def main():
    # Create the Telegram bot application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers for start and messages
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(TEXT, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
