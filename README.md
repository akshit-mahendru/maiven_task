# Readme

I have tried to make it a bit more organised than just a single script. Each step has its own file with proper comments. Head to `/main.py` to see more how everything connects there or install dependencies in shell using `pip install -r requirements.txt` and then run `python main.py`. There are few other command line options mention in the `/main.py` However, those were mostly useful for debugging and demonstration. 

## How It Works

1. **Data Cleaning**: Standardizes fields likes dates and other useful nlp transformations, handles HTML content.
2. **SQL Integration**: Basic connection and loading functions and fetching policies from SQL db.
3. **Personalization**: Computes relevance scores based on recency, and sector match.
4. **Fallback Recommendations**: This was added at the end and some of it code should be in `/src/db.py`. as there were not many direct matches for few jurisdictions like `australia`. I fetched similar sector policies with some `relevance` score.

Output and Logs are stored as well. I have kept raw and processed file too. I added Healthcare as a sector for Carbon Management Strategy Policy (Germany) To test the Fallback Recommendations when no jurisdiction match

## Scaling

Scaling this policy recommender system have various dimensions that we could design but, I want to focus on the following

### Caching: 

Policy recommendations have high re-usability but they should remain up to date. Storing Recommendations in Cache form and able to invalidate them . **This would require us to always send the request of a given user to the same recommendation server alternatively can use shared distributed cache** 

### Sharding: 

Shard the policy database along geography boundaries could decrease the query time drastically (PostgreSQL with partitioning for sharding). **Even the Data ingestion pipeline for policy updates should leverage CDC (Change Data Capture)** 

### Modelling:

In modelling we can break it down in two stages (can have more but for simplicity) Candiate Generation and Ranking.

First, **Candidate (policy) generator** serves as a high-recall retriever, aiming to surface as many true positives as possible while reducing the overall search space. **The aim is to reduce the amount of vectors we had to search from to a managable amount.** 

Using Approx. Nearest Neighbours via hierarchical navigable small worlds (HNSW) (HNSW is my preference but would also look into Nearest Neighbours if the amount of data is less than assumed) and then they can be ranked in order of relevance using **Ranker**. Large models like the embedding model for policies and Ranker can be trained offline on a regular schedule. For real-time inference — especially to handle updates from external sessions — a quantized, lightweight version of the model can be used to serve predictions with lower latency.

**(Note: There are other dimensions we can optimize like Distillation, Leveraging GPUs and CPUs effectively, Monitoring to identify bottle necks)**  

