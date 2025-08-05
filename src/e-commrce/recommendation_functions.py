# Imports
import os
import time
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import random
import re

class RecommendationEngine:
    # --- Constants ---
    PREPROCESSED_DATA_PATH = 'preprocessed_data.pkl'
    TFIDF_CACHE_PATH = 'tfidf_cache.joblib'
    COSINE_CACHE_PATH = 'cosine_cache.joblib'
    INDIAN_CITIES = [
        'Delhi', 'noida', 'Bengaluru', 'Hyderabad', 'indore'
    ]

    def __init__(self, spacy_model="en_core_web_sm"):
        print("[LOG] Loading spaCy model '{}'...".format(spacy_model))
        self.nlp = spacy.load(spacy_model)
        print("[LOG] spaCy model loaded.")

    @staticmethod
    def cache_load_or_compute(cache_path, compute_fn, force_recompute=False, *args, **kwargs):
        if not force_recompute and os.path.exists(cache_path):
            print(f"[LOG] Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                result = joblib.load(f)
            return result
        result = compute_fn(*args, **kwargs)
        with open(cache_path, 'wb') as f:
            joblib.dump(result, f)
        print(f"[LOG] Saved to cache: {cache_path}")
        return result

    def batch_clean_and_extract_tags(self, texts):
        docs = self.nlp.pipe((str(text).lower() for text in texts), batch_size=32, n_process=2)
        return [', '.join([token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]) for doc in docs]

    def extract_keywords_from_text(self, text):
        if not text or text == '':
            return ''
        doc = self.nlp(str(text).lower())
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                token.text.isalnum() and 
                len(token.text) > 2 and 
                token.text not in STOP_WORDS):
                keywords.append(token.text)
        for chunk in doc.noun_chunks:
            phrase = ' '.join([token.text for token in chunk 
                              if token.text.isalnum() and 
                              token.text not in STOP_WORDS and 
                              len(token.text) > 2])
            if len(phrase) > 3:
                keywords.append(phrase)
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'PERSON', 'GPE']:
                entity_text = ' '.join([token.text for token in ent 
                                       if token.text.isalnum() and 
                                       token.text not in STOP_WORDS])
                if len(entity_text) > 2:
                    keywords.append(entity_text)
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        return ', '.join(unique_keywords)

    def assign_random_locations(self, df, location_col='Location'):
        if location_col not in df.columns:
            df[location_col] = [random.choice(self.INDIAN_CITIES) for _ in range(len(df))]
        return df

    def assign_random_ages(self, df, age_col='Age', min_age=18, max_age=55):
        if age_col not in df.columns:
            df[age_col] = [random.randint(min_age, max_age) for _ in range(len(df))]
        return df

    def preprocess_data(self, df, cache_path=None, force_reprocess=False, subset_rows=None):
        if cache_path is None:
            cache_path = self.PREPROCESSED_DATA_PATH
        start_time = time.time()
        if subset_rows is not None:
            print(f"[LOG] Using only the first {subset_rows} rows for development/testing.")
            df = df.head(subset_rows).copy()
        if not force_reprocess and os.path.exists(cache_path):
            print(f"[LOG] Loading preprocessed data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                loaded_df = pickle.load(f)
            print(f"[LOG] Preprocessing loaded from cache in {time.time() - start_time:.2f} seconds.")
            return loaded_df
        print("[LOG] Starting data preprocessing...")
        df['Product Rating'] = df['Product Rating'].fillna(0)
        df['Product Reviews Count'] = df['Product Reviews Count'].fillna(0)
        df['Product Category'] = df['Product Category'].fillna('')
        df['Product Brand'] = df['Product Brand'].fillna('')
        df['Product Description'] = df['Product Description'].fillna('')
        column_name_mapping = {
            'Product Rating': 'Rating',
            'Product Reviews Count': 'ReviewCount',
            'Product Category': 'Category',
            'Product Brand': 'Brand',
            'Product Name': 'Name',
            'Product Image Url': 'ImageURL',
            'Product Description': 'Description',
            'Product Tags': 'Tags'
        }
        df.rename(columns=column_name_mapping, inplace=True)
        if 'Id' in df.columns:
            df.set_index('Uniq Id', inplace=True)
        else:
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'UserIndex'
        df = self.assign_random_locations(df, location_col='Location')
        df = self.assign_random_ages(df, age_col='Age')
        columns_to_extract_tags_from = ['Category', 'Brand', 'Description']
        for column in columns_to_extract_tags_from:
            print(f"[LOG] Batch cleaning and extracting tags for column: {column}")
            df[column] = self.batch_clean_and_extract_tags(df[column])
        print("[LOG] Extracting keywords from product names and descriptions...")
        df['Keywords'] = df['Name'].apply(self.extract_keywords_from_text)
        df['Description_Keywords'] = df['Description'].apply(self.extract_keywords_from_text)
        columns_to_keep = [
            'Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating',
            'Category', 'Tags', 'Description', 'Location', 'Age', 'Keywords', 'Description_Keywords'
        ]
        df = df[[col for col in columns_to_keep if col in df.columns]]
        print("[LOG] Data preprocessing complete. Saving to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"[LOG] Preprocessed data saved to {cache_path}")
        print(f"[LOG] Preprocessing finished in {time.time() - start_time:.2f} seconds.")
        return df

    @staticmethod
    def compute_tfidf(train_data):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'])
        return tfidf_vectorizer, tfidf_matrix

    @staticmethod
    def compute_cosine_similarity_matrix(tfidf_matrix):
        return cosine_similarity(tfidf_matrix, tfidf_matrix)

    def compute_and_cache_tfidf(self, train_data, cache_path=None, force_recompute=False):
        if cache_path is None:
            cache_path = self.TFIDF_CACHE_PATH
        def compute():
            return self.compute_tfidf(train_data)
        return self.cache_load_or_compute(cache_path, compute, force_recompute)

    def compute_and_cache_cosine(self, tfidf_matrix, cache_path=None, force_recompute=False):
        if cache_path is None:
            cache_path = self.COSINE_CACHE_PATH
        def compute():
            return self.compute_cosine_similarity_matrix(tfidf_matrix)
        return self.cache_load_or_compute(cache_path, compute, force_recompute)

    def keyword_based_search(self, train_data, query, top_n=10, location=None, age=None):
        if not query or query.strip() == '':
            return pd.DataFrame()
        query = query.lower().strip()
        query_keywords = self.extract_keywords_from_text(query)
        results = []
        for idx, row in train_data.iterrows():
            score = 0
            name_lower = str(row['Name']).lower()
            category_lower = str(row['Category']).lower()
            brand_lower = str(row['Brand']).lower()
            description_lower = str(row['Description']).lower()
            keywords_lower = str(row['Keywords']).lower()
            desc_keywords_lower = str(row['Description_Keywords']).lower()
            if query in name_lower:
                score += 100
            if any(keyword in name_lower for keyword in query_keywords.split(', ')):
                score += 50
            if query in category_lower or any(keyword in category_lower for keyword in query_keywords.split(', ')):
                score += 30
            if query in brand_lower or any(keyword in brand_lower for keyword in query_keywords.split(', ')):
                score += 25
            if query in description_lower or any(keyword in description_lower for keyword in query_keywords.split(', ')):
                score += 20
            if query in keywords_lower or any(keyword in keywords_lower for keyword in query_keywords.split(', ')):
                score += 15
            if query in desc_keywords_lower or any(keyword in desc_keywords_lower for keyword in query_keywords.split(', ')):
                score += 10
            query_words = query.split()
            for word in query_words:
                if len(word) > 2:
                    if word in name_lower:
                        score += 5
                    if word in category_lower:
                        score += 3
                    if word in description_lower:
                        score += 2
            if score > 0:
                results.append({
                    'index': idx,
                    'score': score,
                    'Name': row['Name'],
                    'ReviewCount': row['ReviewCount'],
                    'Brand': row['Brand'],
                    'ImageURL': row['ImageURL'],
                    'Rating': row['Rating'],
                    'Location': row['Location'],
                    'Age': row['Age']
                })
        results.sort(key=lambda x: x['score'], reverse=True)
        if results:
            result_df = pd.DataFrame(results[:top_n*2])
            result_df = result_df[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'Location', 'Age', 'score']]
            filtered = result_df
            if location:
                filtered = filtered[filtered['Location'] == location]
            if age is not None:
                filtered = filtered[filtered['Age'].between(age-2, age+2)]
            if len(filtered) >= top_n:
                return filtered.head(top_n)
            else:
                others = result_df.drop(filtered.index) if len(filtered) > 0 else result_df
                return pd.concat([filtered, others]).head(top_n)
        return pd.DataFrame()

    def smart_recommendations(self, train_data, query, top_n=10, location=None, age=None):
        if query in train_data['Name'].values:
            print(f"Found exact match for '{query}', using content-based recommendations...")
            return self.content_based_recommendations(train_data, query, top_n, location=location, age=age)
        print(f"No exact match found for '{query}', using keyword-based search...")
        return self.keyword_based_search(train_data, query, top_n, location=location, age=age)

    def content_based_recommendations(self, train_data, item_name, top_n=10, tfidf_cache=None, cosine_cache=None, location=None, age=None):
        if item_name not in train_data['Name'].values:
            print(f"Item '{item_name}' not found in the training data.")
            return pd.DataFrame()
        if tfidf_cache is None:
            tfidf_cache = self.TFIDF_CACHE_PATH
        if cosine_cache is None:
            cosine_cache = self.COSINE_CACHE_PATH
        tfidf_vectorizer, tfidf_matrix = self.compute_and_cache_tfidf(train_data, tfidf_cache)
        cosine_similarities_content = self.compute_and_cache_cosine(tfidf_matrix, cosine_cache)
        item_index = train_data[train_data['Name'] == item_name].index[0]
        similar_items = list(enumerate(cosine_similarities_content[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        top_similar_items = similar_items[1:]
        recommended_item_indices = [x[0] for x in top_similar_items]
        similarity_scores = [x[1] for x in top_similar_items]
        recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'Location', 'Age']].copy()
        recommended_items_details['score'] = similarity_scores
        filtered = recommended_items_details
        if location:
            filtered = filtered[filtered['Location'] == location]
        if age is not None:
            filtered = filtered[filtered['Age'].between(age-2, age+2)]
        if len(filtered) >= top_n:
            return filtered.head(top_n)
        others = recommended_items_details.drop(filtered.index)
        return pd.concat([filtered, others]).head(top_n)

    def collaborative_filtering_recommendations(self, train_data, target_user_id, top_n=10, use_sparse=True, location=None, age=None):
        # Ensure 'Rating' is numeric
        train_data = train_data.copy()
        train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce').fillna(0)
        user_item_matrix = train_data.pivot_table(index=train_data.index, columns='Name', values='Rating', aggfunc='mean').fillna(0)
        if use_sparse:
            user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
            user_similarity = cosine_similarity(user_item_matrix_sparse)
        else:
            user_similarity = cosine_similarity(user_item_matrix)
        if target_user_id not in user_item_matrix.index:
            print(f"User ID {target_user_id} not found in the data.")
            return pd.DataFrame()
        target_user_index = user_item_matrix.index.get_loc(target_user_id)
        user_similarities = user_similarity[target_user_index]
        similar_users_indices = user_similarities.argsort()[::-1][1:]
        recommended_items = []
        for user_index in similar_users_indices:
            rated_by_similar_user = user_item_matrix.iloc[user_index]
            not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
            recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])
        recommended_items_details = train_data[train_data['Name'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'Location', 'Age']]
        filtered = recommended_items_details
        if location:
            filtered = filtered[filtered['Location'] == location]
        if age is not None:
            filtered = filtered[filtered['Age'].between(age-2, age+2)]
        if len(filtered) >= top_n:
            return filtered.head(top_n)
        others = recommended_items_details.drop(filtered.index)
        return pd.concat([filtered, others]).head(top_n)

    def hybrid_recommendations(self, train_data, target_user_id, item_name, top_n=10, location=None, age=None):
        content_based_rec = self.content_based_recommendations(train_data, item_name, top_n, location=location, age=age)
        collaborative_filtering_rec = self.collaborative_filtering_recommendations(train_data, target_user_id, top_n, location=location, age=age)
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
        filtered = hybrid_rec
        if location:
            filtered = filtered[filtered['Location'] == location]
        if age is not None:
            filtered = filtered[filtered['Age'].between(age-2, age+2)]
        if len(filtered) >= top_n:
            return filtered.head(top_n)
        others = hybrid_rec.drop(filtered.index)
        return pd.concat([filtered, others]).head(top_n)

    def smart_hybrid_recommendations(self, train_data, target_user_id, query, top_n=10, location=None, age=None):
        smart_recs = self.smart_recommendations(train_data, query, top_n, location=location, age=age)
        if not smart_recs.empty:
            collaborative_filtering_rec = self.collaborative_filtering_recommendations(train_data, target_user_id, top_n, location=location, age=age)
            hybrid_rec = pd.concat([smart_recs, collaborative_filtering_rec]).drop_duplicates()
            filtered = hybrid_rec
            if location:
                filtered = filtered[filtered['Location'] == location]
            if age is not None:
                filtered = filtered[filtered['Age'].between(age-2, age+2)]
            if len(filtered) >= top_n:
                return filtered.head(top_n)
            others = hybrid_rec.drop(filtered.index)
            return pd.concat([filtered, others]).head(top_n)
        return self.collaborative_filtering_recommendations(train_data, target_user_id, top_n, location=location, age=age) 
