import pandas as pd
from agent import ECommerceAgent
from data_loader import DataLoader

PRODUCT_CSV = "Product_Statistics.csv"
FAQ_CSV = "FAQ.csv"

def main():
    print("📦 Welcome to the E-Commerce Agent CLI!")
    print("Type 'exit' to quit.\n")

    data_loader = DataLoader(PRODUCT_CSV, FAQ_CSV)
    products_df, faq_df = data_loader.load_data()

    agent = ECommerceAgent(
        products_df=products_df,
        faq_df=faq_df,
        model_name="gpt-3.5-turbo-0125",
        temperature=0.2,
        api_key="your-openai-api-key"
    )

    while True:
        user_query = input("🧑 You: ")
        if user_query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = agent.process_query(user_query)
        print(f"🤖 Agent: {response}\n")

if __name__ == "__main__":
    main()
