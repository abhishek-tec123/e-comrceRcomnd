#!/usr/bin/env python3
"""
Smart Recommendation System Demo - Clean Version
Handles exact matches, keyword-based search, and hybrid recommendations (user + query).
"""

import pandas as pd
from recommendation_functions import RecommendationEngine


class SmartRecommendationDemo:
    def __init__(self, data_path, subset_rows=5000):
        self.data_path = data_path
        self.subset_rows = subset_rows
        self.engine = RecommendationEngine()
        self.df = self._load_data()

    def _load_data(self):
        print("\nLoading data...")
        try:
            df = pd.read_csv(self.data_path)
            df = self.engine.preprocess_data(df, subset_rows=self.subset_rows)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            exit(1)

    def _display_recommendations(self, recs, title):
        if not recs.empty:
            print(f"\n{title}")
            print("-" * 60)
            for i, (idx, row) in enumerate(recs.iterrows(), 1):
                print(f"  {i}. {row['Name']}")
                print(f"     Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")
                if 'score' in row:
                    print(f"     Score: {row['score']:.3f}")
                print()
        else:
            print(f"{title} → No results found.")

    def _exact_match_exists(self, query):
        return any(self.df['Name'].str.lower() == query.lower())

    def run(self):
        print("=" * 80)
        print("SMART RECOMMENDATION SYSTEM DEMO")
        print("=" * 80)
        print("Search with full product names, keywords, or partial matches.")
        print("=" * 80)

        while True:
            print("\n" + "-" * 80)
            query = input("Enter your search query (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for testing the smart recommendation system!")
                break

            if not query:
                print("⚠️  Please enter a valid search query.")
                continue

            print(f"\n🔍 Searching for: '{query}'")

            try:
                if self._exact_match_exists(query):
                    recs = self.engine.smart_recommendations(self.df, query, top_n=5)
                    self._display_recommendations(recs, "📊 Smart Recommendations (Exact Match Found)")
                else:
                    recs = self.engine.keyword_based_search(self.df, query, top_n=5)
                    if not recs.empty:
                        self._display_recommendations(recs, "🔎 Keyword-based Search")
                    else:
                        user_id = self.df.index[1000]  # Changeable fallback user
                        recs = self.engine.smart_hybrid_recommendations(self.df, user_id, query, top_n=5)
                        self._display_recommendations(recs, "🤝 Smart Hybrid Recommendations (Fallback)")
            except Exception as e:
                print(f"❌ Error during recommendation: {e}")


if __name__ == "__main__":
    DATA_PATH = "/Users/abhishek/Desktop/e-commerceRecomnndSystm/src/e-commrce/wallmart-5k-data.csv"
    demo = SmartRecommendationDemo(DATA_PATH)
    demo.run()
