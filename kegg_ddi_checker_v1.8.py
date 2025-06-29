# -*- coding: utf-8 -*-
"""
KEGG Drug–Drug Interaction Checker (Streamlit App)
-------------------------------------------------
日本語（ブランド名・一般名いずれも可）の医薬品名を入力すると、KEGG REST `/ddi` API を使って
相互作用を検索し、結果を表形式で表示する簡易チェッカーです。

* **入力をクリア** ボタンでワンクリックでテキストボックスをリセットできます（`st.session_state` の安全な更新）。

### 実行方法
```bash
streamlit run kegg_ddi_checker_v1.8.py
```
依存:
```bash
pip install streamlit pandas requests tqdm
```
同じフォルダに `kegg_drug_list_merged_ja_v1.1.csv` を配置してください。

# 20250628: "CI,P"を"禁忌・併注"に変換 v1.6
# 20250628: 同一入力薬（kegg_idが同じ）の検出と表示 v1.7

"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from difflib import get_close_matches

import pandas as pd
import requests
import streamlit as st

# =============================================================================
# 定数
# =============================================================================
CSV_PATH = Path(__file__).with_name("kegg_drug_list_merged_ja_v1.1.csv")
API_BASE = "https://rest.kegg.jp/ddi/"
INTERACTION_JP = {"CI": "禁忌", "P": "併用注意", "CI,P": "禁忌・併注", "P,CI": "禁忌・併注"}
TIMEOUT = 20

# =============================================================================
# 正規化ユーティリティ
# =============================================================================
JP_CHAR = re.compile("[一-龠ぁ-ゔァ-ヴー]")
REMOVE_CHARS = re.compile(r"[\s・･\-‐–—／/®™\(\)（）\[\]【】,、，.．｡･:：'\"`]+")


def _clean_alias(name: str) -> str:
    """CSV の raw 文字列を前処理（(JAN) など除去）"""
    return name.replace("(JAN)", "").replace("(JP)", "").strip()


def _normalize(key: str) -> str:
    """マッチング用キー：不要文字を削除して小文字化"""
    return REMOVE_CHARS.sub("", key).lower()

# =============================================================================
# CSV → マッピング
# =============================================================================

def _find_col(df: pd.DataFrame, cand: set[str]):
    for col in df.columns:
        if col.strip().lower() in cand:
            return col
    return None


@st.cache_data(show_spinner=False)
def load_maps(csv_path: Path = CSV_PATH):
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    col_id = _find_col(df, {"kegg_id", "kegg"})
    col_jp = _find_col(df, {"drug_name_jp", "drug_jp", "drug_name_j"})
    col_jp0 = _find_col(df, {"drug_name_0_jp", "drug_name0_jp", "drug_brand", "drug_name0"})
    col_en = _find_col(df, {"drug_name", "drug_en", "name"})

    if not col_id or not col_jp:
        st.error("CSV に kegg_id / drug_name_jp 列が見つかりませんでした。")
        st.stop()

    jp2id: dict[str, str] = {}
    id2jp: dict[str, str] = {}
    id2jp0: dict[str, str] = {}

    for _, row in df.iterrows():
        kid = row[col_id].strip()
        if not kid:
            continue

        jp_raw = _clean_alias(row[col_jp])
        jp0_raw = _clean_alias(row[col_jp0]) if col_jp0 else ""
        en_raw = _clean_alias(row[col_en]) if col_en else ""

        aliases = [a.strip() for a in re.split(r"[;/]", jp_raw) if a.strip()]
        aliases0 = [a.strip() for a in re.split(r"[;/]", jp0_raw) if a.strip()]
        aliases.extend(aliases0)
        if en_raw:
            aliases.append(en_raw)

        if not aliases:
            continue

        for al in set(aliases):
            jp2id[_normalize(al)] = kid

        rep = next((a for a in aliases if JP_CHAR.search(a)), None) or aliases[0]
        id2jp[kid] = rep
        id2jp0[kid] = jp0_raw or rep

    return jp2id, id2jp, id2jp0

# =============================================================================
# KEGG REST /ddi
# =============================================================================

def _call_ddi(endpoint: str):
    url = API_BASE + endpoint
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text.split("\n")
    except Exception as e:
        st.error(f"KEGG API Error: {e}")
        return None


def _strip_prefix(kid: str) -> str:
    return kid.replace("dr:", "").replace("cpd:", "").strip()


def fetch_ddi(ids: list[str]):
    if len(ids) < 2:
        return [], []

    recs = _call_ddi("+".join(ids))
    if recs is None:
        recs = []
        for a, b in combinations(ids, 2):
            r = _call_ddi(f"{a}+{b}")
            if r:
                recs.extend(r)

    ddi: list[tuple[str, str, str, str]] = []
    ids_in_results: set[str] = set()
    for line in recs:
        sp = line.split("\t")
        if len(sp) >= 4:
            a_id = _strip_prefix(sp[0])
            b_id = _strip_prefix(sp[1])
            ids_in_results.update([a_id, b_id])
            ddi.append((a_id, b_id, sp[2], sp[3]))

    ids_not_in_results = [i for i in ids if i not in ids_in_results]
    return ddi, ids_not_in_results

# =============================================================================
# 入力名 → KEGG ID
# =============================================================================

def resolve_input(names: list[str], jp2id: dict[str, str]):
    ids: list[str] = []
    unknown: list[str] = []
    id_display: dict[str, str] = {}

    for raw in names:
        key = _normalize(raw)
        hit = jp2id.get(key)
        if not hit:
            partial = [v for k, v in jp2id.items() if key and key in k]
            hit = partial[0] if partial else None
        if not hit:
            close = get_close_matches(key, jp2id.keys(), n=1, cutoff=0.75)
            hit = jp2id[close[0]] if close else None
        if hit:
            ids.append(hit)
            id_display[hit] = raw
        else:
            unknown.append(raw)

    return ids, unknown, id_display

# =============================================================================
# Streamlit UI
# =============================================================================

st.title("KEGG 医薬品相互作用チェック β版")

jp2id_map, id2jp_map, id2jp0_map = load_maps()

# ---- コールバック：入力をクリア ------------------------------------------------

def clear_input():
    st.session_state["drug_input"] = ""

# ---- 入力ウィジェット --------------------------------------------------------
user_text = st.text_input(
    "医薬品名をスペース区切りで入力 (例: クラビット カンデサルタン ピロカルピン)",
    key="drug_input",
)

st.button("入力をクリア", type="secondary", on_click=clear_input)

# 入力なしなら終了
if not user_text.strip():
    st.stop()

# --------------------------
# 入力処理
# --------------------------
raw_list = user_text.strip().split()
kegg_ids, unknowns, disp_map = resolve_input(raw_list, jp2id_map)

# --- 同じ KEGG ID の薬剤が複数入力されていた場合を検出 ---
id_counts = {}
for kid in kegg_ids:
    id_counts[kid] = id_counts.get(kid, 0) + 1
same_id_list = [id2jp_map.get(k, k) for k, v in id_counts.items() if v > 1]

if same_id_list:
    st.info("同一Kegg_idの薬剤が入力されました。商品名: " + ", ".join(same_id_list))

if unknowns:
    st.warning("データベースにない薬剤: " + ", ".join(unknowns))

if len(kegg_ids) < 2:
    st.info("2 つ以上入力してください。")
    st.stop()

with st.spinner("相互作用を検索中..."):
    ddi, no_ddi_ids = fetch_ddi(kegg_ids)

# --------------------------
# 相互作用なし → ここで終了
# --------------------------
if not ddi:
    st.success("相互作用は検出されませんでした。")
    st.stop()

# 入力薬のうち相互作用行に現れなかったものを参考表示
if no_ddi_ids:
    st.info(
        "入力間で相互作用がみつからなかった薬剤: "
        + ", ".join(id2jp_map.get(x, x) for x in no_ddi_ids)
    )

# --------------------------
# 結果テーブル
# --------------------------
out_rows = []
for a, b, code, mech in ddi:
    name_a = disp_map.get(a, id2jp_map.get(a, a))
    name_b = disp_map.get(b, id2jp_map.get(b, b))
    code_disp = INTERACTION_JP.get(code, code)
    out_rows.append(
        {
            "薬剤A": name_a,
            "薬剤B": name_b,
            "相互作用": code_disp,
            "機序": mech,
        }
    )

st.subheader("⚠️ 相互作用リスト ⚠️")
st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

# =============================================================================
# =============================================================================
