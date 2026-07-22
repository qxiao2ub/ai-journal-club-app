# James AI Journal Club App

A Streamlit prototype for an AI-based journal club founded by James. The app helps high-school students discover frontier AI videos, subscribe to tutorials, discuss AI topics, and experience a simple AI brain that personalizes recommendations.

## What is included

- `app.py` - Streamlit website app.
- `data/` - Sample CSV data for videos, sessions, member profiles, interactions, and discussion threads.
- `notebooks/AI_Journal_Club_App_Colab.ipynb` - Colab-friendly notebook that demonstrates the core AI concepts.
- `requirements.txt` - Python packages needed to run locally or on Streamlit Community Cloud.
- `.streamlit/config.toml` - Basic Streamlit theme configuration.

## Main features

1. **Frontier AI video library**
   - AI agents, deep learning, reinforcement learning, RAG, multimodal AI, AI safety, generative AI, and deployment.
   - Each item has a high-school level summary and a resource link.

2. **Session subscriptions**
   - Students can subscribe to tutorials and workshops.
   - Demo subscriptions are stored in Streamlit session state.

3. **Discussion channels**
   - Students can post demo discussions in topic-based channels.
   - In a production app, replace session-state posts with a database.

4. **AI brain**
   - Supervised machine learning recommender predicts student interest from survey profiles and previous likes.
   - Deep learning concept layer explains advanced topics using high-school analogies.
   - Reinforcement learning feedback loop updates topic weights from likes and dislikes.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository, for example `ai-journal-club-app`.
2. Upload all files from this folder.
3. Go to Streamlit Community Cloud.
4. Select the GitHub repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Use the notebook in Google Colab

1. Open `notebooks/AI_Journal_Club_App_Colab.ipynb` in Colab.
2. Run the setup cell.
3. Run each section to see how the supervised ML, neural-network concept layer, and reinforcement-learning feedback loop work.

## Important production notes

This is a portfolio prototype. Before using it with real students, add:

- Parent or school-approved privacy policy.
- Strong moderation for discussions.
- Secure login.
- Database storage.
- Data deletion controls.
- Human review of learning content.
- Clear labels for AI-generated explanations.

## Suggested future upgrades

- Replace sample video links with the club's curated video database.
- Add an admin upload page for new videos and tutorials.
- Add RAG over club notes and approved articles.
- Add learning badges and progress dashboards.
- Add teacher moderation and safety reports.
