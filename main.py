import os
from dotenv import load_dotenv

def main():
    load_dotenv()  # loads .env from current/project directory
    api_key = os.getenv("ANTHROPIC_API_KEY")
    print("Key loaded:", bool(api_key))

if __name__ == "__main__":
    main()