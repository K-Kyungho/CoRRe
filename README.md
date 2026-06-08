# CoRRe
Official github repository for **Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals**. (CIKM 2026 Short Paper Track Under Review)


## GitHub Quick Start Guide 🚀

This repo follows a 3-step pipeline:

- 🧹 `refinement.py`: Propagate GPT item features with co-purchase graph (i.e., direction refinement) and magnitude refinement.
- 🧠 `generate_user_profile.py`: Generate user preference sentences from history.
- 🔎 `item_retrieval.py`: Retrieve candidates from propagated item embeddings and run Hit@K / NDCG evaluation.

All scripts now use dataset paths under `data/{dataset}` by default. (`sports`, `toys`, `beauty`)

---

## ✅ Common setup

- Set your OpenAI API key

```bash
export OPENAI_API_KEY=your-openai-api-key
```

- Expected input layout

```text
data/{dataset}/
├── inter.json: User-item interaction data.
├── item2id.json: Mapping from item IDs to internal indices.
├── user2id.json: Mapping from user IDs to internal indices.
├── title.pickle: Item titles.
├── gpt_feat.npy: LLM-based embeddings of item titles.
├── generated_intent.json: LLM-generated user intents/profiles.
└── refined_itemfeat.npy: Refined item embeddings.

```

`{dataset}` should be one of: `sports`, `toys`, `beauty`

Due to file size limitations, gpt_feat.npy and refined_itemfeat.npy are provided separately. Please download them from the following Dropbox link and place them in the appropriate feature directory:
https://www.dropbox.com/scl/fo/rtloyxm3qoawuclb2utgf/AOrrzXzDQN7hYT154RXY8jU?rlkey=5jprf6flw3g9x7hxcqj1vkzl2&st=uy53dm1v&dl=0

## 1) Run `refinement.py`

### Command

```bash
[Sports] python refinement.py --dataset sports --fusion_alpha 0.4 --degree_alpha 0.1
[Toys] python refinement.py --dataset toys --fusion_alpha 0.8 --degree_alpha 0.05
[Beauty] python refinement.py --dataset beauty --fusion_alpha 0.2 --degree_alpha 0.05
```

### Output

- `data/{dataset}/refined_itemfeat.npy` (You can use our uploaded version.)


---

## 2) Run `generate_user_profile.py`

### Command

```bash
[Sports] python generate_user_profile.py --dataset sports
[Toys] python generate_user_profile.py --dataset toys
[Beauty] python generate_user_profile.py --dataset beauty
```

### Output

- `data/{dataset}/generated_intent.json` (You can use our uploaded version.)

---

## 3) Run `item_retrieval.py`

### Command

```bash
[Sports] python item_retrieval.py --dataset sports --topk 10 20 30 50
[Toys] python item_retrieval.py --dataset toys --topk 10 20 30 50
[Beauty] python item_retrieval.py --dataset beauty --topk 10 20 30 50
```


---

## 🛠 Useful options

- `refinement.py`
  - `--dataset` : `sports` (default), `toys`, `beauty`
  - `--num_layers` : number of propagation layers (default: `1`)
  - `--fusion_alpha` : combined ratio between original and propagated embeddings
  - `--apply_degree_multiply` / `--no-apply_degree_multiply` : multiply each final item embedding by `d^degree_alpha` (default: enabled)
  - `--degree_alpha` : exponent for degree scaling (default: `0.0`)

- `generate_user_profile.py`
  - `--dataset` : `sports` (default), `toys`, `beauty`
  - `--max_users` : number of users to process (`None` for all), default is 1,000 in our paper.
  - `--max_items_per_user` : number of most recent items from history (default: `10`)
  - `--model` : chat model name

- `item_retrieval.py`
  - `--dataset` : `sports` (default), `toys`, `beauty`
  - `--topk` : list of Top-K values
  - `--model` : embedding model
  - `--degree_alpha` : popularity re-ranking factor
  - `--mask_train_items` / `--no_mask_train_items` : whether to exclude train items from retrieval candidates
