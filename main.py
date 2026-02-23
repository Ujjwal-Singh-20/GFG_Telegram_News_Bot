from email.utils import parsedate_to_datetime
import os
import numpy as np
import feedparser
from dotenv import load_dotenv
from datetime import datetime, timezone
from supabase import create_client
import requests
from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai
import google.genai as genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# client = OpenAI(api_key=OPENAI_API_KEY)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.api_key = GOOGLE_API_KEY
else:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# model = genai.GenerativeModel("gemini-2.5-flash-lite")#gemini-1.5-flash")
client = genai.Client()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_NAME}/pipeline/feature-extraction"

def generate_hf_embeddings(texts):
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True}
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.text}")
        
    embeddings = response.json()
    return np.array(embeddings)

RSS_FEEDS = {
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Wired AI": "https://www.wired.com/feed/tag/ai/latest/rss",
    "CNET": "https://rss.app/feeds/HcnmXBH6p38I4hqK.xml",
    "Techradar": "https://www.techradar.com/feeds.xml",
    # "ArsTechnica": "https://arstechnica.com/gadgets/feed/",      # giving 4-5 days old news
    "Reuters": "https://news.google.com/rss/search?q=site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen",
}

SIMILARITY_THRESHOLD = 0.85


def fetch_articles():
    articles = []
    today = datetime.now(timezone.utc).date()

    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)

        for entry in feed.entries[:10]:
            published_date = None

            if entry.get("published"):
                try:
                    # Try ISO 8601 format first
                    published_date = datetime.fromisoformat(
                        entry["published"].replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    try:
                        # Try RFC 2822 format (handles GMT correctly)
                        published_date = parsedate_to_datetime(
                            entry["published"]
                        ).date()
                    except Exception as e:
                        print(
                            f"Could not parse date for article: {entry.get('title', '')}",
                            e,
                        )

            # Only include articles published today
            if published_date == today:
                articles.append({
                    "source": source,
                    "title": entry.get("title", ""),
                    "content": entry.get("summary", ""),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", None),
                })

    return articles


def deduplicate_articles(articles):
    texts = [
        a["title"] + "\n\n" + a["content"][:500]
        for a in articles
    ]

    embeddings = generate_hf_embeddings(texts)

    unique_indices = []
    used = set()

    for i in range(len(embeddings)):
        if i in used:
            continue

        unique_indices.append(i)

        for j in range(i + 1, len(embeddings)):
            if j in used:
                continue

            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            if sim > SIMILARITY_THRESHOLD:
                used.add(j)

    unique_articles = [articles[i] for i in unique_indices]
    unique_embeddings = [embeddings[i] for i in unique_indices]

    return unique_articles, unique_embeddings




def insert_articles_and_embeddings(articles, embeddings):
    for article, embedding in zip(articles, embeddings):

        # 1️⃣ Insert article first
        response = supabase.table("articles_master").upsert(
            {
                "url": article["url"],
                "title": article["title"],
                "content": article["content"],
                "source": article["source"],
                "published_at": article["published_at"]
            },
            on_conflict="url"
        ).execute()

        article_id = response.data[0]["id"]

        # 2️⃣ Insert embedding
        supabase.table("article_embeddings").upsert(
            {
                "article_id": article_id,
                "embedding": embedding.tolist()
            },
            on_conflict="article_id"
        ).execute()

        print("Inserted:", article["title"])



# LLM SELECT BEST 10 + SUMMARIZE
def select_and_summarize(articles):
    block = "\n\n---\n\n".join([
        f"{i+1}. {a['title']}\n{a['content'][:500]}"
        for i, a in enumerate(articles)
    ])

    prompt = f"""
    You are a tech news editor.

    From the articles provided below, select the 10 most significant and high-impact stories specifically about technology and AI.

    Strictly exclude:
    - Any non-tech articles.
    - Stock market updates, financial reports, or investment news.

    For each selected article, write a concise 2-3 line summary. Return ONLY a JSON list in the following format:

    [
    {{
        "index": number,
        "summary": "2-3 line summary"
    }}
    ]

    Articles:
    {block}
    """

    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.3
    # )
    # response = model.generate_content(prompt)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", # or gemini-3-flash-preview, etc.
        contents=prompt
    )
    print(response.text)#choices[0].message.content)

    return response.text#choices[0].message.content



def store_selected(articles, selected_json):
    import json
    import re
    from datetime import datetime

    # print("--- RAW LLM RESPONSE START ---")
    # print(selected_json)
    # print("--- RAW LLM RESPONSE END ---")

    # Extract content inside markdown code blocks
    match = re.search(r"```(?:json)?\s*(.*?)```", selected_json, re.DOTALL)
    if match:
        clean_json = match.group(1).strip()
    else:
        # If no markdown blocks, try to find the first [ and last ]
        match_fallback = re.search(r"(\[.*\])", selected_json, re.DOTALL)
        if match_fallback:
            clean_json = match_fallback.group(1).strip()
        else:
            clean_json = selected_json.strip()

    try:
        selected = json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON: {e}")
        print("Cleaned JSON was:", clean_json)
        return

    today = datetime.now().date().isoformat()

    # Clear all existing entries before insertion
    print("Clearing all existing entries in daily_digest...")
    supabase.table("daily_digest").delete().neq("id", 0).execute() # Delete all rows where id != 0 (hack to delete all rows as GTE/NEQ works)

    for item in selected:
        idx = item["index"] - 1
        summary = item["summary"]
        rank = item.get("index") # Using the 1-based index as rank
        article = articles[idx]

        # 1️⃣ Update summary in articles_master
        supabase.table("articles_master").update(
            {"summary": summary}
        ).eq("url", article["url"]).execute()

        # 2️⃣ Get article_id (since we don't have it here)
        response = supabase.table("articles_master").select("id").eq("url", article["url"]).execute()
        if response.data:
            article_id = response.data[0]["id"]

            # 3️⃣ Insert into daily_digest
            supabase.table("daily_digest").upsert(
                {
                    "article_id": article_id,
                    "digest_date": today,
                    "rank": rank
                },
                on_conflict="article_id, digest_date"
            ).execute()

            print(f"Stored in Digest (Rank {rank}):", article["title"])




#MAIN
def run_pipeline():
    print("Fetching news...")
    articles = fetch_articles()

    print("Deduplicating...")
    unique_articles, unique_embeddings = deduplicate_articles(articles)

    print(f"Unique articles: {len(unique_articles)}")

    if len(unique_articles) == 0:
        return

    print("Inserting articles + embeddings...")
    insert_articles_and_embeddings(unique_articles, unique_embeddings)

    print("Selecting best 10 via LLM...")
    selected_json = select_and_summarize(unique_articles)

    print("Updating selected articles in DB...")
    store_selected(unique_articles, selected_json)


    print("Preparing Telegram digest...")
    from telegram_opr import send_telegram_message
    from datetime import datetime
    import json
    import re

    total_sent = 0
    # We need to re-parse or rely on store_selected to return what it found.
    # For simplicity, let's extract the JSON again or pass it.
    match = re.search(r"```(?:json)?\s*(.*?)```", selected_json, re.DOTALL)
    clean_json = match.group(1).strip() if match else selected_json.strip()
    try:
        selected_data = json.loads(clean_json)
        total_sent = len(selected_data)
        
        digest_msg = f"<b>Daily Tech Hub — {datetime.now().strftime('%d %b %Y')}</b>\n"
        digest_msg += "━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, item in enumerate(selected_data):
            idx = item["index"] - 1
            summary = item["summary"]
            article = unique_articles[idx]
            
            digest_msg += f"<b>{article['title']}</b>\n"
            digest_msg += f"<i>{article['source']}</i>\n"
            digest_msg += f"<blockquote>{summary}</blockquote>\n"
            digest_msg += f"🔗 <a href='{article['url']}'>Read Full Story</a>\n\n"
        
        digest_msg += "━━━━━━━━━━━━━━━━━━\n"
        digest_msg += "<i>Stay informed.</i>"
        
        send_telegram_message(digest_msg)
    except Exception as e:
        print(f"Error preparing Telegram message: {e}")

    # Store analytics
    print("Storing analytics...")
    try:
        supabase.table("news_stats").insert({
            "total_fetched": len(articles),
            "total_unique": len(unique_articles),
            "total_sent": total_sent,
            "date": datetime.now().date().isoformat()
        }).execute()
        print("Analytics stored successfully.")
    except Exception as e:
        print(f"Error storing analytics: {e}")

    print("Done.")


if __name__ == "__main__":
    run_pipeline()
