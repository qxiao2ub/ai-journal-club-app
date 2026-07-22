from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="James AI Journal Club",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> Dict[str, pd.DataFrame]:
    data = {
        "videos": pd.read_csv(DATA_DIR / "videos.csv"),
        "sessions": pd.read_csv(DATA_DIR / "sessions.csv"),
        "users": pd.read_csv(DATA_DIR / "user_profiles.csv"),
        "interactions": pd.read_csv(DATA_DIR / "interactions.csv"),
        "threads": pd.read_csv(DATA_DIR / "discussions.csv"),
    }
    data["sessions"]["date"] = pd.to_datetime(data["sessions"]["date"])
    return data


def video_text(row: pd.Series) -> str:
    fields = [
        row.get("title", ""),
        row.get("topic", ""),
        row.get("level", ""),
        row.get("summary", ""),
        row.get("tags", ""),
        row.get("frontier_theme", ""),
    ]
    return " ".join(str(x) for x in fields if pd.notna(x))


def profile_text(row: pd.Series) -> str:
    fields = [
        row.get("preferred_topics", ""),
        row.get("desired_level", ""),
        f"grade {row.get('grade', '')}",
        f"math comfort {row.get('math_comfort', '')}",
        f"coding comfort {row.get('coding_comfort', '')}",
        f"time budget {row.get('time_budget_minutes', '')}",
    ]
    return " ".join(str(x) for x in fields if pd.notna(x))


def pair_text_from_merged(row: pd.Series) -> str:
    return (
        f"Member profile: {row['preferred_topics']} {row['desired_level']} "
        f"grade {row['grade']} math {row['math_comfort']} coding {row['coding_comfort']}. "
        f"Content: {row['title']} {row['topic']} {row['level']} {row['summary']} {row['tags']}"
    )


def pair_text(user_row: pd.Series, video_row: pd.Series) -> str:
    return f"Member profile: {profile_text(user_row)}. Content: {video_text(video_row)}"


@st.cache_resource
def train_supervised_model(users: pd.DataFrame, videos: pd.DataFrame, interactions: pd.DataFrame):
    merged = interactions.merge(users, on="member_id", how="left").merge(videos, on="content_id", how="left")
    merged = merged.dropna(subset=["liked"])
    if merged["liked"].nunique() < 2:
        return None
    x_train = merged.apply(pair_text_from_merged, axis=1)
    y_train = merged["liked"].astype(int)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    return model


def ensure_session_state() -> None:
    st.session_state.setdefault("subscriptions", [])
    st.session_state.setdefault("new_posts", [])
    st.session_state.setdefault("rl_weights", {})


def get_user_weights(user_id: str) -> Dict[str, float]:
    all_weights = st.session_state.setdefault("rl_weights", {})
    all_weights.setdefault(user_id, {})
    return all_weights[user_id]


def update_feedback(user_id: str, topic: str, reward: float) -> None:
    weights = get_user_weights(user_id)
    old = weights.get(topic, 0.0)
    weights[topic] = round(old + 0.20 * reward, 3)


def topic_match_score(preferred_topics: str, topic: str) -> float:
    preferred = [p.strip().lower() for p in str(preferred_topics).split(";")]
    return 1.0 if str(topic).strip().lower() in preferred else 0.0


def recommend_videos(
    selected_user_id: str,
    users: pd.DataFrame,
    videos: pd.DataFrame,
    interactions: pd.DataFrame,
    hide_watched: bool = True,
) -> pd.DataFrame:
    user_row = users.loc[users["member_id"] == selected_user_id].iloc[0]
    model = train_supervised_model(users, videos, interactions)

    items = videos.copy()
    if hide_watched:
        watched = set(
            interactions.loc[
                (interactions["member_id"] == selected_user_id) & (interactions["watched"] == 1),
                "content_id",
            ]
        )
        items = items.loc[~items["content_id"].isin(watched)].copy()

    if items.empty:
        return items

    pair_texts = [pair_text(user_row, row) for _, row in items.iterrows()]
    if model is None:
        supervised_prob = np.full(len(items), 0.50)
    else:
        supervised_prob = model.predict_proba(pair_texts)[:, 1]

    docs = [profile_text(user_row)] + [video_text(row) for _, row in items.iterrows()]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf[0], tfidf[1:]).ravel()

    preferred = str(user_row["preferred_topics"])
    topic_scores = np.array([topic_match_score(preferred, topic) for topic in items["topic"]])
    freshness = 1.0 / (1.0 + (items["freshness_days"].astype(float).to_numpy() / 30.0))
    weights = get_user_weights(selected_user_id)
    rl_scores = np.array([np.tanh(weights.get(topic, 0.0)) for topic in items["topic"]])

    items["supervised_like_probability"] = supervised_prob
    items["content_similarity"] = similarity
    items["topic_match"] = topic_scores
    items["freshness_score"] = freshness
    items["feedback_learning_score"] = rl_scores
    items["recommendation_score"] = (
        0.45 * supervised_prob
        + 0.25 * similarity
        + 0.15 * topic_scores
        + 0.10 * freshness
        + 0.05 * rl_scores
    )
    return items.sort_values("recommendation_score", ascending=False)


EXPLAINER = {
    "AI Agents": {
        "big_idea": "An AI agent is a system that can set a goal, choose steps, use tools, and check progress.",
        "analogy": "Think of it like a careful lab partner: it makes a plan, uses a calculator or search tool, and then checks whether the answer makes sense.",
        "why": "Agents matter because they move AI from only answering questions toward helping complete multi-step projects.",
        "activity": "Give an agent a school-safe task, then ask it to show its plan before it takes action.",
    },
    "Deep Learning": {
        "big_idea": "Deep learning uses layers of artificial neurons to turn raw data into useful patterns.",
        "analogy": "A neural network is like a team of students passing notes: each layer adds one more clue until the final layer makes a decision.",
        "why": "It powers many frontier systems, including language models, image generators, and speech tools.",
        "activity": "Draw a three-layer network that turns study habits into a quiz-score prediction.",
    },
    "Reinforcement Learning": {
        "big_idea": "Reinforcement learning teaches an AI through rewards for good actions and penalties for bad actions.",
        "analogy": "It is like learning a video game: try a move, see the score, and improve the next move.",
        "why": "It is useful for games, robots, recommendations, and systems that improve from feedback.",
        "activity": "Design a reward rule for a robot that should reach a goal without bumping into walls.",
    },
    "Retrieval-Augmented Generation": {
        "big_idea": "RAG lets an AI look up relevant documents before it answers.",
        "analogy": "Instead of guessing from memory, the AI gets a library card and checks notes first.",
        "why": "It can reduce hallucinations and connect answers to trusted club resources.",
        "activity": "Give the AI three short paragraphs and ask it to answer only from those paragraphs.",
    },
    "AI Safety": {
        "big_idea": "AI safety asks whether a system is reliable, fair, private, and aligned with human goals.",
        "analogy": "It is like testing a bridge before people cross it: power is not enough; it must be safe.",
        "why": "Students need to understand both what AI can do and what can go wrong.",
        "activity": "For any demo, write one benefit, one risk, and one test that could catch a problem.",
    },
    "Multimodal AI": {
        "big_idea": "Multimodal AI combines different kinds of information, such as text, images, audio, and video.",
        "analogy": "It is like a student using eyes, ears, and reading notes together to understand a lesson.",
        "why": "It helps AI interact with the real world more naturally.",
        "activity": "Compare a caption written by a human with a caption generated for the same image.",
    },
    "Generative AI": {
        "big_idea": "Generative AI creates new text, images, audio, code, or video from patterns it learned.",
        "analogy": "It is like remixing many examples into a new draft, not copying one exact source.",
        "why": "It changes how people brainstorm, design, explain, and prototype ideas.",
        "activity": "Ask for three different explanations of the same idea and compare which is clearest.",
    },
    "Deployment": {
        "big_idea": "Deployment is the work of turning an AI idea into a tool people can actually use.",
        "analogy": "A science fair prototype becomes useful when it has instructions, a clean interface, and safety checks.",
        "why": "Portfolio projects need to be runnable, documented, and responsible.",
        "activity": "Turn one notebook function into a Streamlit button and test it with a friend.",
    },
}


def explain_topic(topic: str, learner_context: str = "") -> str:
    card = EXPLAINER.get(topic, None)
    if card is None:
        card = {
            "big_idea": f"{topic} is an AI topic that can be understood by asking what data it uses, what pattern it learns, and how people check its output.",
            "analogy": "Imagine a new school club activity: first learn the rules, then try examples, then improve with feedback.",
            "why": "The key is to connect the advanced idea to a simple input, a process, and an output.",
            "activity": "Write one example input and one example output for this topic.",
        }
    extra = ""
    if learner_context.strip():
        extra = f"\n\nPersonal connection: Based on your note, connect this topic to: {learner_context.strip()}"
    return (
        f"Big idea: {card['big_idea']}\n\n"
        f"High-school analogy: {card['analogy']}\n\n"
        f"Why it matters: {card['why']}\n\n"
        f"Try it in journal club: {card['activity']}"
        f"{extra}"
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:12px; padding:16px; margin-bottom:12px;">
        <h4 style="margin-top:0;">{title}</h4>
        <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    ensure_session_state()
    data = load_data()
    videos = data["videos"]
    sessions = data["sessions"]
    users = data["users"]
    interactions = data["interactions"]
    threads = data["threads"]

    st.sidebar.title("James AI Journal Club")
    page = st.sidebar.radio(
        "Explore",
        ["Home", "Video Recommender", "Sessions", "Discussion Channels", "AI Brain Lab", "Portfolio Notes"],
    )
    st.sidebar.caption("Prototype app for a high-school AI journal club.")

    if page == "Home":
        st.title("James AI Journal Club App")
        st.write(
            "A prototype learning community where students discover frontier AI videos, "
            "subscribe to tutorials, discuss ideas, and see how an AI brain can personalize learning."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Videos", len(videos))
        c2.metric("Tutorial sessions", len(sessions))
        c3.metric("Discussion posts", len(threads) + len(st.session_state["new_posts"]))
        c4.metric("Sample members", len(users))

        st.subheader("App concept")
        st.markdown(
            """
            1. Students watch short, up-to-date AI learning resources written at a high-school level.
            2. Members subscribe to tutorials and sessions led by James and guest mentors.
            3. Channels let students post questions, reactions, and project ideas.
            4. The AI brain recommends content, explains difficult topics, and improves from feedback.
            """
        )

        st.subheader("AI brain architecture")
        st.code(
            """
Open resources + club videos + survey data
        |
        v
Supervised machine learning recommender
        |
        v
High-school explanation engine + neural-network concept layer
        |
        v
Reinforcement learning feedback loop from likes/dislikes/subscriptions
            """.strip(),
            language="text",
        )

    elif page == "Video Recommender":
        st.title("Personalized Frontier AI Videos")
        user_label = st.selectbox(
            "Choose a sample member profile",
            users.apply(lambda r: f"{r['member_id']} - {r['name']} (grade {r['grade']})", axis=1),
        )
        user_id = user_label.split(" - ")[0]
        hide_watched = st.checkbox("Hide videos this member already watched", value=True)
        recs = recommend_videos(user_id, users, videos, interactions, hide_watched=hide_watched)
        selected_user = users.loc[users["member_id"] == user_id].iloc[0]
        st.info(
            f"Preferred topics: {selected_user['preferred_topics']} | Desired level: {selected_user['desired_level']} | "
            f"Time budget: {selected_user['time_budget_minutes']} minutes"
        )

        if recs.empty:
            st.warning("No recommendations left after filtering. Uncheck the filter to see all content.")
        else:
            st.subheader("Top recommendations")
            for _, row in recs.head(6).iterrows():
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### {row['title']}")
                    st.write(row["summary"])
                    st.caption(
                        f"Topic: {row['topic']} | Level: {row['level']} | Estimated: {row['estimated_minutes']} minutes | "
                        f"Score: {row['recommendation_score']:.2f}"
                    )
                    st.link_button("Open video resource", row["source_url"])
                with right:
                    st.write("Teach the app")
                    if st.button("Like", key=f"like_{user_id}_{row['content_id']}"):
                        update_feedback(user_id, row["topic"], 1.0)
                        st.success("Feedback saved. This topic is now weighted higher for this member.")
                    if st.button("Not for me", key=f"dislike_{user_id}_{row['content_id']}"):
                        update_feedback(user_id, row["topic"], -1.0)
                        st.warning("Feedback saved. This topic is now weighted lower for this member.")
                st.divider()

            st.subheader("Why the AI recommended these")
            st.dataframe(
                recs[
                    [
                        "title",
                        "topic",
                        "supervised_like_probability",
                        "content_similarity",
                        "topic_match",
                        "freshness_score",
                        "feedback_learning_score",
                        "recommendation_score",
                    ]
                ].head(10),
                use_container_width=True,
                hide_index=True,
            )

    elif page == "Sessions":
        st.title("Subscribe to AI Sessions and Tutorials")
        topics = ["All"] + sorted(sessions["topic"].unique().tolist())
        topic_filter = st.selectbox("Filter by topic", topics)
        shown = sessions if topic_filter == "All" else sessions.loc[sessions["topic"] == topic_filter]
        for _, row in shown.sort_values("date").iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['title']}")
                st.write(row["description"])
                st.caption(
                    f"{row['date'].date()} | Topic: {row['topic']} | Difficulty: {row['difficulty']} | Teacher: {row['teacher']} | Capacity: {row['capacity']}"
                )
                if st.button("Subscribe", key=f"sub_{row['session_id']}"):
                    if row["session_id"] not in st.session_state["subscriptions"]:
                        st.session_state["subscriptions"].append(row["session_id"])
                    st.success("Subscribed in this demo session.")
        subscribed = sessions.loc[sessions["session_id"].isin(st.session_state["subscriptions"])]
        st.subheader("My demo subscriptions")
        if subscribed.empty:
            st.write("No sessions subscribed yet.")
        else:
            st.dataframe(subscribed[["date", "title", "topic", "difficulty"]], hide_index=True, use_container_width=True)

    elif page == "Discussion Channels":
        st.title("Discussion Channels")
        all_threads = threads.copy()
        if st.session_state["new_posts"]:
            all_threads = pd.concat([all_threads, pd.DataFrame(st.session_state["new_posts"])], ignore_index=True)
        channel = st.selectbox("Channel", ["All"] + sorted(all_threads["channel"].unique().tolist()))
        shown = all_threads if channel == "All" else all_threads.loc[all_threads["channel"] == channel]
        for _, row in shown.sort_values("timestamp", ascending=False).iterrows():
            with st.container(border=True):
                st.markdown(f"**#{row['channel']} - {row['topic']}**")
                st.write(row["post"])
                st.caption(f"By {row['author']} | {row['timestamp']} | Upvotes: {row['upvotes']}")

        st.subheader("Add a demo post")
        with st.form("new_post_form", clear_on_submit=True):
            author = st.text_input("Name", value="New Member")
            channel_name = st.text_input("Channel", value="ai-brain-ideas")
            topic_name = st.text_input("Topic", value="AI Agents")
            post = st.text_area("Post", value="I wonder how we can test whether a recommendation is actually helpful.")
            submitted = st.form_submit_button("Post to demo board")
        if submitted and post.strip():
            st.session_state["new_posts"].append(
                {
                    "thread_id": f"NEW{len(st.session_state['new_posts']) + 1:03d}",
                    "channel": channel_name.strip() or "general",
                    "author": author.strip() or "New Member",
                    "topic": topic_name.strip() or "AI",
                    "post": post.strip(),
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "upvotes": 0,
                }
            )
            st.success("Post added for this demo session.")

    elif page == "AI Brain Lab":
        st.title("AI Brain Lab")
        st.write("This page turns advanced AI ideas into high-school level explanations and shows the app architecture.")
        col1, col2 = st.columns([1, 1])
        with col1:
            topic = st.selectbox("Choose a topic", sorted(EXPLAINER.keys()))
            learner_context = st.text_area(
                "Optional learner context",
                placeholder="Example: I like biology, robotics, or science fair projects.",
            )
            if st.button("Explain it simply"):
                st.markdown("#### High-school explanation")
                st.write(explain_topic(topic, learner_context))
        with col2:
            st.subheader("Three AI layers")
            st.markdown(
                """
                **1. Supervised ML:** learns from surveys, watch history, and likes to predict useful videos.  
                **2. Deep neural network concept layer:** represents advanced AI topics as patterns that can be simplified with analogies.  
                **3. Reinforcement learning:** updates topic weights when students like, skip, or subscribe.
                """
            )
            st.subheader("Current feedback weights")
            user_for_weights = st.selectbox("Member", users["member_id"].tolist(), key="weights_user")
            weights = get_user_weights(user_for_weights)
            if weights:
                st.bar_chart(pd.DataFrame({"topic": list(weights.keys()), "weight": list(weights.values())}).set_index("topic"))
            else:
                st.write("No feedback yet. Use the Like or Not for me buttons in the recommender.")

        st.subheader("Responsible AI checklist")
        st.checkbox("Show why each recommendation was made.", value=True)
        st.checkbox("Let students control their interests and delete feedback.", value=True)
        st.checkbox("Use human review before publishing official learning content.", value=True)
        st.checkbox("Avoid collecting sensitive personal data from minors.", value=True)

    elif page == "Portfolio Notes":
        st.title("GitHub and Streamlit Portfolio Notes")
        st.markdown(
            """
            Use this prototype as a portfolio-ready starting point.

            **Suggested repository name:** `ai-journal-club-app`

            **Local run command:**
            ```bash
            pip install -r requirements.txt
            streamlit run app.py
            ```

            **GitHub upload steps:**
            1. Create a new GitHub repository.
            2. Upload `app.py`, `requirements.txt`, `README.md`, `.streamlit/`, `data/`, and `notebooks/`.
            3. Commit the files.
            4. Deploy on Streamlit Community Cloud by selecting the repo and setting `app.py` as the entry file.

            **Next upgrades:**
            - Replace sample CSVs with real club videos and session links.
            - Add account login and a database such as SQLite, Firebase, or Supabase.
            - Add moderation tools for discussion posts.
            - Add a real LLM or retrieval system only after adding privacy and safety controls.
            """
        )


if __name__ == "__main__":
    main()
