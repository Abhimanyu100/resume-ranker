import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_desc, resumes_dict):
    names = list(resumes_dict.keys())
    texts = list(resumes_dict.values())
    
    all_docs = [job_desc] + texts
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    results = []
    for i, name in enumerate(names):
        results.append({
            "Candidate": name,
            "Match (%)": round(scores[0][i] * 100, 2)
        })
        
    df = pd.DataFrame(results).sort_values(by="Match (%)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df

# --- Web App UI ---
st.set_page_config(page_title="Resume Ranker", page_icon="🎯")

st.title("🎯 AI Resume Ranker")
st.write("Paste your Job Description and Candidate Resumes below to see who matches best.")

# Input fields
jd = st.text_area("📋 Paste Job Description here:", height=150)

st.subheader("📄 Paste Candidate Resumes")
col1, col2, col3 = st.columns(3)

with col1:
    res1 = st.text_area("Candidate 1 Resume:", height=200)
with col2:
    res2 = st.text_area("Candidate 2 Resume:", height=200)
with col3:
    res3 = st.text_area("Candidate 3 Resume:", height=200)

# Submit Button
if st.button("Rank Candidates", type="primary"):
    if not jd.strip():
        st.error("Please provide a Job Description.")
    else:
        # Only include resumes that actually have text typed in them
        resumes = {}
        if res1.strip(): resumes["Candidate 1"] = res1
        if res2.strip(): resumes["Candidate 2"] = res2
        if res3.strip(): resumes["Candidate 3"] = res3
        
        if not resumes:
            st.error("Please paste at least one resume.")
        else:
            with st.spinner("Analyzing text..."):
                final_df = rank_resumes(jd, resumes)
                st.success("Analysis Complete!")
                
                # Display beautiful table
                st.dataframe(
                    final_df.style.background_gradient(cmap='Greens', subset=['Match (%)']),
                    use_container_width=True,
                    hide_index=True
                )