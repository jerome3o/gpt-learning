import os
import discord
from run_benchmarks import load_model
from discord import Intents

TOKEN = os.getenv("DISCORD_TOKEN")

# MODEL = "databricks/dolly-v2-12b"
# MODEL = "StabilityAI/stablelm-tuned-alpha-7b"
MODEL = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"

# Create bot instance
client = discord.Client(intents=Intents.all())

model = load_model(MODEL)


# Register event listener for new messages
@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")


@client.event
async def on_message(message):
    print("message received")
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Respond to messages in a channel named "ai-agentc"
    if message.channel.name == "agentc-chat" and message.content:
        async with message.channel.typing():
            model_resp = model(message.content, 1)[0]
            resp = f"```\nMODEL: {MODEL}\n```\n{model_resp}"

            print(resp)
            await message.channel.send(resp[:2000])


# Run the bot
client.run(TOKEN)
