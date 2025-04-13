import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("top_summaries_per_cluster.csv")

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Session Summary Finder")
st.write("Enter keywords related to a topic, and get the most relevant session summaries.")

# Input box for keywords
keywords = st.text_input("Enter keywords (e.g., regression, sampling, clustering)", "")

if st.button("Find Relevant Session"):
    if keywords.strip() == "":
        st.warning("Please enter at least one keyword.")
    else:
        # Encode user query
        query_embedding = model.encode([keywords])

        # Encode summaries from CSV (only once for top summaries)
        df['embedding'] = df['cleaned_summary'].apply(lambda x: model.encode(x))

        # Compute cosine similarity
        df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], query_embedding)[0][0])

        # Find the session (cluster) with highest average similarity
        best_cluster = df.groupby('Cluster')['similarity'].mean().idxmax()
        top_matches = df[df['Cluster'] == best_cluster].sort_values(by='cluster_rank').head(3)

        st.success(f"Top 3 summaries from the most relevant session (Cluster {best_cluster}):")

        # Show summaries in a text area
        summary_text = "\n\n".join(f"• {row['cleaned_summary']}" for _, row in top_matches.iterrows())
        st.text_area("Top Summaries", value=summary_text, height=300)
