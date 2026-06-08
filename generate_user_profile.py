#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User History Based Item Prediction - Simple LLM Approach

Inputs:
- user2items_str.json: userID -> [item_str, ...]
- text_pickle: {item_str: short_text_description, ...}

Outputs:
- user_prediction.json

Behavior:
- Use up to 10 most recent items per user as history
- Send only item text to LLM
- LLM generates a recommendation sentence for the next likely interaction
"""

import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import requests


# ============================
# OpenAI API utilities
# ============================

def ensure_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Please export OPENAI_API_KEY.")
    return key


def call_openai_chat_gpt5(
    messages: List[Dict[str, Any]],
    model: str,
    retry: int = 4,
    timeout: int = 60,
) -> str:
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = ensure_openai_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    last_err = None
    for attempt in range(retry):
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_err = str(e)
        # backoff
        time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"OpenAI call failed. Last error: {last_err}")


# ============================
# Prompt template
# ============================

SYSTEM_USER_PREDICTION = """You are a shopping recommendation assistant that predicts what items a user is likely to interact with next.

You will be given a list of items that the user has previously interacted with or purchased (with item title).
Please reason about the user's purchase history. Users may have multiple interests or consistent preferences.

Goal: Based on this user's history and preferences shown through these items, write a natural recommendation sentence about what the user is likely to interact with or purchase next.

Constraints:
- The sentence should be specific and realistic (TYPE/brand-level is okay).
- You can include a specific brand.
"""


def get_item_text(item_id: str, item_texts: Dict[str, str]) -> str:
    return item_texts.get(item_id, f"(no description, item_id={item_id})")


def build_user_prediction_content(
    user_id: str,
    item_ids: List[str],
    item_texts: Dict[str, str],
) -> str:
    lines = [
        f"User ID: {user_id}",
        "",
        "[User History] (oldest \u2192 newest)",
        "",
    ]
    for idx, iid in enumerate(item_ids, 1):
        text = get_item_text(iid, item_texts)
        lines.append(f"{idx}. TEXT: {text}")
        lines.append("")
    lines.append(
        "Task: Based on this user's history, write a recommendation sentence describing what the user is most likely to interact with or purchase next."
    )
    return "\n".join(lines)


# ============================
# User prediction run
# ============================

def run_user_prediction(
    user2items_path: str,
    text_pickle_path: str,
    out_json_path: str,
    model: str,
    user2id_path: Optional[str] = None,
    max_items_per_user: int = 10,
    max_users: Optional[int] = None,
) -> None:
    with open(user2items_path, "r", encoding="utf-8") as f:
        user2items_str: Dict[str, List[str]] = json.load(f)

    with open(text_pickle_path, "rb") as f:
        item_texts: Dict[str, str] = pickle.load(f)

    user2id_map = None
    if user2id_path and os.path.exists(user2id_path):
        with open(user2id_path, "r", encoding="utf-8") as f:
            raw_user2id = json.load(f)
        user2id_map = {str(k): int(v) for k, v in raw_user2id.items()}

    if user2id_map is not None:
        ordered_users = [uid for uid, _ in sorted(user2id_map.items(), key=lambda x: x[1])]
        user_ids = [uid for uid in ordered_users if uid in user2items_str]
        print(f"[USER PREDICTION] user order: data user2id ({user2id_path})")
    else:
        user_ids = list(user2items_str.keys())
        print("[USER PREDICTION] user order: inter.json key order")

    if max_users is not None:
        user_ids = user_ids[:max_users]

    results: Dict[str, Any] = {}

    print(f"[USER PREDICTION] users to process: {len(user_ids)}")

    for idx, uid in enumerate(user_ids, 1):
        items = user2items_str[uid]
        if len(items) < 2:
            continue

        history_items = items[:-1]
        recent_items = history_items[-max_items_per_user:]

        # Build prompt
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_USER_PREDICTION}
        ]
        user_prompt = build_user_prediction_content(
            user_id=uid,
            item_ids=recent_items,
            item_texts=item_texts,
        )
        messages.append({"role": "user", "content": user_prompt})

        try:
            raw = call_openai_chat_gpt5(messages=messages, model=model)
        except Exception as e:
            results[uid] = {
                "source_items": recent_items,
                "error": str(e),
            }
            print(f"[USER PREDICTION] Error on user {uid}: {e}")
            continue

        recommendation_sentence = (raw or "").strip().replace("\n", " ")
        recommendation_sentence = " ".join(recommendation_sentence.split())

        # Normalize response to a plain sentence when JSON is returned
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("recommendation_sentence"), str):
                    recommendation_sentence = parsed["recommendation_sentence"].strip()
                elif isinstance(parsed.get("predicted_items"), list) and parsed["predicted_items"]:
                    first_item = str(parsed["predicted_items"][0]).strip()
                    recommendation_sentence = f"The user is likely to interact with or purchase {first_item} next."
        except Exception:
            pass

        results[uid] = {
            "source_items": recent_items,
            "raw_response": raw,
            "recommendation_sentence": recommendation_sentence,
        }

        if idx % 10 == 0:
            print(f"[USER PREDICTION] processed {idx}/{len(user_ids)} users")

        if idx % 100 == 0:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[USER PREDICTION] saved to {out_json_path}")


# ============================
# main
# ============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="User History Based Item Prediction - Simple LLM Approach"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sports",
        help="Dataset name (for default paths)",
    )
    dataset = parser.parse_known_args()[0].dataset

    parser.add_argument(
        "--user2items",
        type=str,
        default=f"data/{dataset}/inter.json",
        help="Path to user2items_str.json (userID -> [item_str,...])",
    )
    parser.add_argument(
        "--text_pickle",
        type=str,
        default=f"data/{dataset}/title.pickle",
        help="Pickle path of item_str -> short text description",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"data/{dataset}/generated_intent.json",
        help="Output JSON path for user prediction results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI chat model name (e.g., gpt-4o-mini, gpt-4o)",
    )
    parser.add_argument(
        "--user2id_path",
        type=str,
        default=f"data/{dataset}/user2id.json",
        help="Path to user2id.json for user ordering",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=1000,
        help="Max number of users to process (None for all)",
    )
    parser.add_argument(
        "--max_items_per_user",
        type=int,
        default=10,
        help="Maximum number of recent items per user",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("===== USER PREDICTION START =====")
    run_user_prediction(
        user2items_path=args.user2items,
        text_pickle_path=args.text_pickle,
        out_json_path=args.output,
        model=args.model,
        user2id_path=args.user2id_path,
        max_items_per_user=args.max_items_per_user,
        max_users=args.max_users,
    )
    print("===== USER PREDICTION COMPLETE =====")


if __name__ == "__main__":
    main()
