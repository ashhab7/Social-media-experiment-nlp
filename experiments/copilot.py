import asyncio
from sydney import SydneyClient

async def main():
    async with SydneyClient() as sydney:
        while True:
            prompt = input("You: ")
            if prompt == "!reset":
                await sydney.reset_conversation()
            else:
                response = await sydney.ask(prompt, search = False, citations = True)
                print(f"Copilot: {response}")

asyncio.run(main())
