# #!/usr/bin/env python3
# """
# Interactive demo for the smart recommendation system.
# This script allows you to test different types of queries and see how the system responds.
# """

# import pandas as pd
# from recommendation_functions import RecommendationEngine

# def main():
#     print("=" * 80)
#     print("SMART RECOMMENDATION SYSTEM DEMO")
#     print("=" * 80)
#     print("This demo shows how the system handles both exact product names and keywords.")
#     print("You can now search for products using:")
#     print("• Full product names (e.g., 'GOJO Fast Towels - 225 Count Bucket')")
#     print("• Keywords (e.g., 'towels', 'cream', 'mascara')")
#     print("• Partial matches (e.g., 'toothbrush', 'gel')")
#     print("=" * 80)
    
#     # Instantiate the recommendation engine
#     engine = RecommendationEngine()

#     # Load data
#     try:
#         print("\nLoading data...")
#         data_path = "/Users/abhishek/Desktop/e-commerceRecomnndSystm/src/e-commrce/wallmart-5k-data.csv"
#         df = pd.read_csv(data_path)
#         df = engine.preprocess_data(df, subset_rows=5000)  # Use subset for faster demo
#         print(f"✅ Data loaded successfully! Shape: {df.shape}")
#     except Exception as e:
#         print(f"❌ Error loading data: {e}")
#         print("Make sure you're in the correct directory with the data file.")
#         return
    
#     # Interactive demo
#     while True:
#         print("\n" + "-" * 80)
#         query = input("Enter your search query (or 'quit' to exit): ").strip()
        
#         if query.lower() in ['quit', 'exit', 'q']:
#             print("Thanks for testing the smart recommendation system!")
#             break
        
#         if not query:
#             print("Please enter a search query.")
#             continue
        
#         print(f"\n🔍 Searching for: '{query}'")
#         print("-" * 60)
        
#         # Test smart recommendations
#         try:
#             print("📊 Smart Recommendations:")
#             smart_recs = engine.smart_recommendations(df, query, top_n=5)
            
#             if not smart_recs.empty:
#                 for i, (idx, row) in enumerate(smart_recs.iterrows(), 1):
#                     # print(f"  {i}. {row['Name']}")
#                     # print(f"     Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")
#                     # if 'score' in row:
#                     #     print(f"     Score: {row['score']:.3f}")
#                     # print()
#                     pass
#             else:
#                 print("  No smart recommendations found.")
                
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
        
#         # Test keyword-based search
#         try:
#             print("🔎 Keyword-based Search:")
#             keyword_recs = engine.keyword_based_search(df, query, top_n=5)
            
#             if not keyword_recs.empty:
#                 for i, (idx, row) in enumerate(keyword_recs.iterrows(), 1):
#                     print(f"  {i}. {row['Name']}")
#                     print(f"     Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")
#                     if 'score' in row:
#                         print(f"     Score: {row['score']:.3f}")
#                     print()
#             else:
#                 print("  No keyword-based results found.")
                
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
        
#         # Test smart hybrid recommendations
#         try:
#             print("🤝 Smart Hybrid Recommendations (user + query):")
#             user_id = df.index[1000]  # You can change this to any valid user
#             smart_hybrid_recs = engine.smart_hybrid_recommendations(df, user_id, query, top_n=5)
            
#             if not smart_hybrid_recs.empty:
#                 for i, (idx, row) in enumerate(smart_hybrid_recs.iterrows(), 1):
#                     print(f"  {i}. {row['Name']}")
#                     print(f"     Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")
#                     if 'score' in row:
#                         print(f"     Score: {row['score']:.3f}")
#                     print()
#             else:
#                 print("  No smart hybrid recommendations found.")
                
#         except Exception as e:
#             print(f"  ❌ Error: {e}")

# if __name__ == "__main__":
#     main() 





from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from recommendation_functions import RecommendationEngine

# Initialize FastAPI app
app = FastAPI(title="Smart Recommendation API")

# Path to your CSV file
DATA_PATH = "/Users/abhishek/Desktop/e-commerceRecomnndSystm/src/e-commrce/wallmart-5k-data.csv"

# Load and preprocess data
engine = RecommendationEngine()
df = pd.read_csv(DATA_PATH)
df = engine.preprocess_data(df, subset_rows=5000)

# -------------------------------
# Pydantic models
# -------------------------------

class RecommendationRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5
    user_id: Optional[int] = None


class RecommendationResponse(BaseModel):
    name: str
    brand: str
    rating: float
    review_count: int
    score: Optional[float] = None
    image_url: Optional[str] = None

# -------------------------------
# POST Endpoint: /recommend
# -------------------------------

@app.post("/recommend", response_model=List[RecommendationResponse])
def recommend(request: RecommendationRequest):
    query_lower = request.query.lower()
    exact_match = df['Name'].str.lower().eq(query_lower).any()

    try:
        if exact_match:
            recs = engine.smart_recommendations(df, request.query, top_n=request.top_n)
        elif request.user_id is not None and request.user_id in df.index:
            recs = engine.smart_hybrid_recommendations(df, request.user_id, request.query, top_n=request.top_n)
        else:
            recs = engine.keyword_based_search(df, request.query, top_n=request.top_n)
    except Exception as e:
        return [{
            "name": "Error",
            "brand": str(e),
            "rating": 0,
            "review_count": 0,
            "score": None,
            "image_url": None
        }]
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    def safe_int(val, default=0):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default
        
    results = []
    for _, row in recs.iterrows():
        results.append({
            "name": row['Name'],
            "brand": row.get('Brand', ''),
            "rating": safe_float(row.get('Rating'), 0.0),
            "review_count": safe_int(row.get('ReviewCount'), 0),
            "score": float(row['score']) if 'score' in row else None,
            "image_url": row.get('ImageURL', None)
        })

    return results
