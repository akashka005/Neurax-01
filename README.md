# Backend Overview

This document describes the `/backend` folder of Nexus and explains how each code file works. Every important file is listed with sample code and a short explanation of its behavior.

## Purpose

The backend provides:
- a FastAPI-based API for chat, feedback, and health checks
- an emotion classification pipeline
- action selection using reinforcement learning
- response generation with human-friendly templates
- training scripts for dataset preparation, emotion model training, and clustering

## Folder structure

- `prepare_dataset.py`
- `train_emotion.py`
- `train_cluster.py`
- `requirements.txt`
- `app/`
  - `main.py`
  - `api/`
    - `schemas.py`
    - `routes/chat.py`
    - `routes/feedback.py`
    - `routes/health.py`
  - `models/`
    - `emotion_model.py`
    - `rl_agent.py`
  - `pipeline/`
    - `nlp_pipeline.py`
    - `feature_builder.py`
    - `clustering.py`
    - `decision_engine.py`
    - `response_builder.py`
  - `services/`
    - `chat_service.py`
    - `feedback_service.py`
    - `user_service.py`
- `saved_models/` — model artifacts saved after training

---

## Top-level scripts and training helpers

### `prepare_dataset.py`

This script reads the raw data CSV, extracts messages and emotion labels, balances classes, and saves a cleaned dataset.

```python
import pandas as pd
import ast

df = pd.read_csv("data/raw/train.csv")

texts = []
labels = []
for _, row in df.iterrows():
    try:
        dialog = ast.literal_eval(row["dialog"])
        emotion = row["emotion"]
        emotion = emotion.replace("[", "").replace("]", "")
        emotion = [int(x) for x in emotion.split()]

        for t, e in zip(dialog, emotion):
            texts.append(t)
            labels.append(e)

    except:
        continue
new_df = pd.DataFrame({
    "text": texts,
    "label": labels
})
emotion_map = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "sad",
    6: "surprise"
}

new_df["label"] = new_df["label"].map(emotion_map)
neutral_df = new_df[new_df["label"] == "neutral"]
other_df = new_df[new_df["label"] != "neutral"]
neutral_sample = neutral_df.sample(frac=0.2, random_state=42)
balanced_df = pd.concat([neutral_sample, other_df])
balanced_df = balanced_df.dropna()
balanced_df = balanced_df[balanced_df["text"].str.len() > 3]
TARGET = 300

dfs = []
for label in balanced_df["label"].unique():
    subset = balanced_df[balanced_df["label"] == label]
    if len(subset) > TARGET:
        subset = subset.sample(n=TARGET, random_state=42)
    else:
        subset = subset.sample(n=TARGET, replace=True, random_state=42)
    dfs.append(subset)

final_df = pd.concat(dfs).sample(frac=1, random_state=42)
final_df.to_csv("data/processed/emotion_dataset.csv", index=False)

print("FINAL BALANCED DATASET CREATED")
print(final_df["label"].value_counts())
```

What it does:
- reads raw dialog data
- converts string-encoded lists into Python lists
- maps numeric emotion ids to strings
- down-samples `neutral` and up-samples smaller classes to balance labels
- saves a cleaned emotion dataset for training

### `train_emotion.py`

This file trains the text classifier used to infer emotion from user messages.

```python
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("data/processed/emotion_dataset.csv")
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("\nMODEL PERFORMANCE:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "saved_models/emotion_model.pkl")
joblib.dump(vectorizer, "saved_models/vectorizer.pkl")
```

What it does:
- trains `LogisticRegression` using TF-IDF features
- evaluates performance on a held-out split
- saves both the model and vectorizer for production use

### `train_cluster.py`

This script trains a simple KMeans cluster model used to categorize conversation style.

```python
import pandas as pd
import joblib
from sklearn.cluster import KMeans


df = pd.read_csv("data/processed/emotion_dataset.csv")
emotion_map = {
    "neutral": 0,
    "happy": 1,
    "anger": 2,
    "fear": 3,
    "sad": 4,
    "disgust": 5,
    "surprise": 6
}

df["emotion_num"] = df["label"].map(emotion_map)
df["length"] = df["text"].apply(lambda x: len(str(x).split()))
df["intensity"] = df["text"].apply(lambda x: 1 if "!" in str(x) else 0)
X = df[["emotion_num", "length", "intensity"]]
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
joblib.dump(kmeans, "saved_models/kmeans.pkl")
```

What it does:
- converts emotion to numeric form
- creates basic text features: length and intensity
- trains a KMeans model to group similar messages
- saves the cluster model

### `requirements.txt`

```text
fastapi
uvicorn
transformers
torch
```

This file declares backend dependencies used by the API and model code.

---

## App package files

### `app/main.py`

This is the FastAPI app entrypoint.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, feedback, health

app = FastAPI(
    title="AI Mental Health Companion",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat")
app.include_router(feedback.router, prefix="/feedback")
app.include_router(health.router, prefix="/health")

@app.get("/")
def root():
    return {"message": "AI Mental Health Companion API is running"}
```

What it does:
- builds the API app
- allows CORS from any client
- mounts the chat, feedback, and health routers

### `app/api/schemas.py`

Defines the request and response object shapes.

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    emotion: str
    confidence: float
    intent: str
    action: str

class FeedbackRequest(BaseModel):
    user_id: str
    reward: float
```

This ensures incoming requests are validated.

### `app/api/routes/chat.py`

```python
from fastapi import APIRouter
from app.api.schemas import ChatRequest
from app.services.chat_service import handle_chat

router = APIRouter()

@router.post("/")
def chat(req: ChatRequest):
    return handle_chat(req.user_id, req.message)
```

It accepts chat POST requests and forwards them to the main chat service.

### `app/api/routes/feedback.py`

```python
from fastapi import APIRouter
from app.api.schemas import FeedbackRequest
from app.services.feedback_service import handle_feedback

router = APIRouter()

@router.post("/")
def feedback(req: FeedbackRequest):
    return handle_feedback(req.user_id, req.reward)
```

It accepts feedback POST requests and forwards them to the RL feedback service.

### `app/api/routes/health.py`

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def health():
    return {"status": "ok"}
```

A lightweight endpoint for liveness checks.

---

## Models

### `app/models/emotion_model.py`

This module loads the saved classifier and vectorizer, then predicts emotion.

```python
import joblib

model = joblib.load("saved_models/emotion_model.pkl")
vectorizer = joblib.load("saved_models/vectorizer.pkl")

def predict_emotion(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = float(max(probs))

    return prediction, confidence
```

What it does:
- loads persisted artifacts from training
- transforms a text string into feature vectors
- returns the predicted emotion and its confidence score

### `app/models/rl_agent.py`

This file implements a simple Q-learning decision policy.

```python
import random
import json
import os

ACTIONS = ["empathy", "motivation", "advice", "distraction", "clarify"]
Q_TABLE_PATH = "saved_models/rl/q_table.json"
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.3
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

def load_q_table():
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_q_table(q_table):
    os.makedirs(os.path.dirname(Q_TABLE_PATH), exist_ok=True)
    with open(Q_TABLE_PATH, "w") as f:
        json.dump(q_table, f)

q_table = load_q_table()

def get_state(emotion, cluster, intent):
    return f"{emotion}_{cluster}_{intent}"


def choose_action(emotion, cluster, intent):
    global EPSILON
    state = get_state(emotion, cluster, intent)

    if state not in q_table:
        q_table[state] = {a: 0.0 for a in ACTIONS}

    if random.random() < EPSILON:
        action = random.choice(ACTIONS)
    else:
        action = max(q_table[state], key=q_table[state].get)

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    return action


def update_q(emotion, cluster, intent, action, reward):
    state = get_state(emotion, cluster, intent)
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in ACTIONS}
    current_q = q_table[state][action]
    max_future_q = max(q_table[state].values())
    new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
    q_table[state][action] = new_q
    save_q_table(q_table)
```

What it does:
- uses a Q-table keyed by `emotion_cluster_intent`
- chooses a random action sometimes for exploration
- otherwise chooses the highest-value action
- updates values when feedback arrives

---

## Pipeline

### `app/pipeline/nlp_pipeline.py`

This module converts text into an emotion label and confidence.

```python
from app.models.emotion_model import predict_emotion


def analyze_text(text):
    emotion, raw_confidence = predict_emotion(text)
    intent = "support"
    try:
        raw_confidence = float(raw_confidence)
    except Exception:
        raw_confidence = 0.0
    raw_confidence = max(0.0, min(1.0, raw_confidence))
    if emotion != "neutral":
        confidence = max(raw_confidence, 0.55)
    else:
        confidence = raw_confidence
    return emotion, confidence, intent
```

What it does:
- calls the emotion model
- clamps confidence to `[0.0, 1.0]`
- ensures non-neutral emotions still return a minimum confidence
- currently always returns `support` intent

### `app/pipeline/feature_builder.py`

```python
def build_features(user_id, emotion):
    mapping = {
        "sad": 0, "happy": 1, "angry": 2,
        "fear": 3, "love": 4, "surprise": 5
    }
    return [mapping.get(emotion, 0)]
```

What it does:
- converts emotion into a single numeric feature for clustering
- falls back to `0` when emotion is missing

### `app/pipeline/clustering.py`

```python
import joblib
import numpy as np

try:
    kmeans = joblib.load("saved_models/kmeans.pkl")
except Exception:
    kmeans = None


def assign_cluster(features):
    if kmeans is None:
        return fallback_cluster(features)
    try:
        features_array = np.array(features).reshape(1, -1)
        cluster = int(kmeans.predict(features_array)[0])
        return cluster
    except Exception:
        return fallback_cluster(features)


def fallback_cluster(features):
    try:
        return int(features[0]) % 3
    except Exception:
        return 0
```

What it does:
- loads the KMeans model if available
- predicts cluster from feature vector
- falls back to a simple deterministic cluster when prediction fails

### `app/pipeline/decision_engine.py`

This module decides which response action should be used.

```python
from app.models.rl_agent import choose_action

SAFE_ACTIONS = ["empathy", "motivation", "advice", "distraction", "clarify"]

def normalize_confidence(conf):
    try:
        conf = float(conf)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, conf))


def fallback_action(emotion):
    mapping = {
        "sad": "empathy",
        "fear": "empathy",
        "anger": "empathy",
        "disgust": "clarify",
        "happy": "motivation",
        "surprise": "distraction",
        "neutral": "clarify"
    }
    return mapping.get(emotion, "empathy")


def safe_choose_action(emotion, cluster, intent):
    try:
        action = choose_action(emotion, cluster, intent)
    except Exception:
        return None
    if action not in SAFE_ACTIONS:
        return None
    return action


def decide_action(emotion, cluster, confidence, intent):
    confidence = normalize_confidence(confidence)
    if confidence < 0.25:
        return "clarify"
    if confidence < 0.5:
        return fallback_action(emotion)
    if confidence < 0.75:
        action = safe_choose_action(emotion, cluster, intent)
        return action if action else fallback_action(emotion)
    action = safe_choose_action(emotion, cluster, intent)
    return action if action else fallback_action(emotion)
```

What it does:
- chooses `clarify` for very low confidence
- uses fallback emotion-based actions for moderate confidence
- uses RL-selected actions for high confidence
- ensures unsafe actions are rejected

### `app/pipeline/response_builder.py`

This module builds the final text returned to users.

```python
import random
from collections import defaultdict

LAST_RESPONSES = defaultdict(list)
MAX_HISTORY = 5

RESPONSES = {
    "empathy": {
        "sad": [
            "That sounds really exhausting. Feeling lost like that can be heavy.",
            "I’m really sorry you’re feeling this way.",
            "It seems like things have been weighing on you a lot.",
            "That must be really tough to deal with."
        ],
        "fear": [
            "That sounds overwhelming.",
            "It’s okay to feel this way.",
            "I can imagine that feels really intense."
        ],
        "anger": [
            "That sounds frustrating.",
            "I hear you. That must be really irritating."
        ],
        "default": [
            "I’m here with you.",
            "You don’t have to go through this alone.",
            "I’m listening."
        ]
    },
    "motivation": {
        "sad": [
            "Even if it feels hard right now, things can get better step by step.",
            "You've handled difficult moments before — you can get through this too."
        ],
        "default": [
            "Keep going — you're stronger than you think.",
            "This moment doesn’t define you. You’ve got this."
        ]
    },
    "advice": {
        "default": [
            "Try taking a small break and resetting your mind.",
            "Focus on one small thing at a time — it helps reduce overwhelm.",
            "A short walk or deep breaths might help clear your thoughts."
        ]
    },
    "distraction": {
        "default": [
            "Maybe listen to music or watch something light for a bit.",
            "How about doing something you enjoy for a few minutes?",
            "A small distraction might help shift your mood a bit."
        ]
    },
    "clarify": {
        "default": [
            "I might be misunderstanding — could you tell me a bit more?",
            "I'm not completely sure I got that.",
            "Can you share a bit more about how you're feeling?"
        ]
    }
}

OPENERS = ["", "Hey — ", "It's okay. ", "Take your time. "]
CLOSERS = ["", " I'm here for you.", " You’re not alone.", " We can figure this out together."]
FOLLOWUPS = [
    "Want to share more about what’s been going on?",
    "Do you want to talk about what’s making you feel this way?",
    "I’m here to listen if you want to open up."
]


def generate_response(emotion, action, user_id="default", confidence=1.0):
    emotion = emotion if emotion else "default"
    action_block = RESPONSES.get(action, {})
    emotion_responses = action_block.get(emotion) or action_block.get("default")
    history = LAST_RESPONSES[user_id]
    available = [r for r in emotion_responses if r not in history]

    if not available:
        available = emotion_responses

    base = random.choice(available)
    history.append(base)
    if len(history) > MAX_HISTORY:
        history.pop(0)

    if confidence < 0.4:
        opener = random.choice(["", "I might be wrong, but ", "It seems like "])
    else:
        opener = random.choice(OPENERS)

    closer = random.choice(CLOSERS)
    followup = ""
    if action in ["empathy", "clarify"] and random.random() < 0.5:
        followup = " " + random.choice(FOLLOWUPS)

    response = f"{opener}{base}{closer}{followup}".strip()
    return response
```

What it does:
- selects a template based on action and emotion
- avoids repeating responses for the same user
- adds optional opener, closer, and follow-up phrases
- returns the final string sent to the frontend

---

## Services

### `app/services/chat_service.py`

This is the main orchestrator for chat requests.

```python
from app.pipeline.nlp_pipeline import analyze_text
from app.pipeline.feature_builder import build_features
from app.pipeline.clustering import assign_cluster
from app.pipeline.decision_engine import decide_action
from app.pipeline.response_builder import generate_response
from app.models.rl_agent import get_state
from app.services.user_service import set_last_interaction


def handle_chat(user_id, message):
    if not message or not isinstance(message, str):
        return {
            "response": "I'm here to listen. Could you tell me a bit more?",
            "emotion": "neutral",
            "confidence": 0.0,
            "intent": "support",
            "action": "empathy"
        }

    message = message.strip()
    try:
        emotion, confidence, intent = analyze_text(message)
    except Exception:
        emotion, confidence, intent = "neutral", 0.0, "support"

    try:
        features = build_features(user_id, emotion, message)
    except Exception:
        features = [0]

    try:
        cluster = assign_cluster(features)
    except Exception:
        cluster = 0

    try:
        action = decide_action(emotion, cluster, confidence)
    except Exception:
        action = "empathy"

    try:
        state = get_state(emotion, cluster)
        set_last_interaction(user_id, state, action)
    except Exception:
        pass

    try:
        response = generate_response(emotion, action)
    except Exception:
        response = "I'm here with you. Do you want to talk more about it?"

    return {
        "response": response,
        "emotion": emotion,
        "confidence": round(float(confidence), 3),
        "intent": intent,
        "action": action,
        "cluster": cluster
    }
```

What it does:
- validates the incoming message
- analyzes the message for emotion and confidence
- converts emotion to cluster features
- decides the response action
- stores the user interaction for feedback learning
- returns the final response payload

### `app/services/feedback_service.py`

This service updates the RL model after user feedback.

```python
from app.models.rl_agent import update_q
from app.services.user_service import get_last_interaction, clear_user_state


def handle_feedback(user_id: str, reward: float):
    last = get_last_interaction(user_id)
    if not last:
        return {
            "status": "no_interaction",
            "message": "No previous interaction found to learn from"
        }

    emotion = last.get("emotion")
    cluster = last.get("cluster")
    intent = last.get("intent")
    action = last.get("action")
    if emotion is None or intent is None or action is None or cluster is None:
        return {
            "status": "invalid_state",
            "message": "Incomplete interaction data for learning"
        }

    try:
        reward = float(reward)
    except Exception:
        reward = 0.0
    reward = max(-1.0, min(1.0, reward))

    try:
        update_q(emotion, int(cluster), intent, action, reward)
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to update learning",
            "error": str(e)
        }

    clear_user_state(user_id)
    return {
        "status": "success",
        "message": "Learning updated successfully",
        "data": {
            "emotion": emotion,
            "cluster": int(cluster),
            "intent": intent,
            "action": action,
            "reward": reward
        }
    }
```

What it does:
- retrieves the last saved state for the user
- validates that state includes action, emotion, cluster, and intent
- clamps reward between -1 and 1
- updates the Q-table via `update_q`
- clears the state so learning is not repeated accidentally

### `app/services/user_service.py`

This file keeps per-user state in memory.

```python
from typing import Dict, Optional

USER_STATE: Dict[str, dict] = {}

def set_last_interaction(user_id: str, data: dict) -> None:
    if not user_id or not isinstance(user_id, str):
        return
    if not isinstance(data, dict):
        return

    emotion = data.get("emotion")
    cluster = data.get("cluster")
    intent = data.get("intent")
    action = data.get("action")
    if emotion is None or intent is None or action is None or cluster is None:
        return

    try:
        cluster = int(cluster)
    except Exception:
        cluster = 0

    USER_STATE[user_id] = {
        "emotion": str(emotion),
        "cluster": cluster,
        "intent": str(intent),
        "action": str(action)
    }


def get_last_interaction(user_id: str) -> Optional[dict]:
    return USER_STATE.get(user_id)


def clear_user_state(user_id: str) -> None:
    if user_id in USER_STATE:
        del USER_STATE[user_id]


def clear_all_states() -> None:
    USER_STATE.clear()
```

What it does:
- stores the last decision state for each user
- allows the feedback service to retrieve and clear that state
- is intentionally simple and in-memory only

---

## How the chat request flows

1. Frontend sends POST `/chat/` with `user_id` and `message`.
2. `chat_service.handle_chat()` validates the input.
3. `nlp_pipeline.analyze_text()` predicts emotion and confidence.
4. `feature_builder.build_features()` converts emotion to numeric features.
5. `clustering.assign_cluster()` chooses a conversation cluster.
6. `decision_engine.decide_action()` selects the response action.
7. `response_builder.generate_response()` builds the returned message.
8. The backend returns:
   - `response`
   - `emotion`
   - `confidence`
   - `intent`
   - `action`
   - `cluster`

## How feedback works

1. Frontend sends POST `/feedback/` with `user_id` and a numeric reward.
2. `feedback_service.handle_feedback()` fetches the last saved interaction.
3. It validates emotion, cluster, intent, and action.
4. Reward is normalized to `[-1.0, 1.0]`.
5. `rl_agent.update_q()` updates the Q-table.
6. The `user_service` clears the stored state for that user.

---

## Notes and caveats

- The user state store is in-memory only; it resets on server restart.
- The RL Q-table is saved to `saved_models/rl/q_table.json` and grows over time.
- The pipeline currently uses only one text feature for clustering.
- The system assumes the emotion model and vectorizer are trained and available.

## Running the backend

1. Install Python dependencies.
2. Generate training data and models:
   - `python prepare_dataset.py`
   - `python train_emotion.py`
   - `python train_cluster.py`
3. Run the API server:
   - `uvicorn app.main:app --reload`

---

## Summary

This backend provides a text-based emotional chatbot API with training helpers, model loading, a reinforcement learning action policy, and response generation. Each module is designed to keep analysis, decision-making, and text generation separate.
