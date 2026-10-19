"""
Competitor Price Tracker (GrabFood)
-----------------------------------
Upload checkout screenshots -> Gemini reads them -> fills the price sheet columns
-> formulas calculated -> saved to a CSV database -> weekly summary pivot.

Run:  streamlit run app.py
"""
import json
import time
import io
import math
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

DB_PATH = "competitor_prices.csv"
# Tried in order; if one is overloaded (503) or rate-limited (429), the next is used.
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"]

COLUMNS = [
    "Week", "Platform", "Ownership", "Category", "Brand", "Menu Name",
    "Menu Net Price", "Cart Voucher %", "Cart Voucher Min Purchase",
    "Cart Voucher Max Disc", "Qty for Voucher", "Platform Fee", "Packaging Fee",
    "Voucher Basket Total", "Price/qty after Voucher",
    "Notes",
]
FORMULA_COLS = ["Qty for Voucher", "Voucher Basket Total", "Price/qty after Voucher"]

# ----------------------------------------------------------------------------
# 1. Extraction with Gemini (vision + JSON schema = reliable structured output)
# ----------------------------------------------------------------------------
class Checkout(BaseModel):
    brand: str = Field(description="Restaurant / brand name, e.g. 'KFC'")
    store_name: str = Field(description="Full outlet name as shown")
    menu_name: str = Field(description="Menu item name exactly as shown")
    menu_unit_price: float = Field(description="Price of ONE unit in IDR, before voucher")
    quantity_in_cart: int
    subtotal: float
    voucher_name: str
    voucher_percent: float = Field(description="e.g. 60 for 60%; 0 if none/unknown")
    voucher_min_purchase: float = Field(description="Min. order / Min. belanja; 0 if unknown")
    voucher_max_discount: float = Field(description="Max discount / Maks. diskon / s.d.; 0 if unknown")
    voucher_discount_applied: float = Field(description="Discount actually deducted")
    platform_fee: float = Field(description="'Platform fee' / 'Biaya platform'")
    packaging_fee: float = Field(description="'Restaurant & partner fees' / 'Biaya restoran & mitra' / packaging fee / 'Biaya kemasan'")
    delivery_fee: float = Field(description="'Delivery fee' / 'Ongkos kirim'")
    total_shown: float
    uncertain_fields: str = Field(description="Fields unreadable or guessed; empty if none")


PROMPT = """These are screenshot(s) from the GrabFood Indonesia app for ONE menu item
(checkout page, and possibly the voucher/promo detail page). Extract the values.
- All money is IDR; return plain numbers (e.g. 'Rp34.000' -> 34000).
- The app may be in English or Indonesian. Label mapping:
  * platform_fee  <- "Platform fee" / "Biaya platform"
  * packaging_fee <- "Restaurant & partner fees" / "Biaya restoran & mitra" /
                     "Packaging fee" / "Biaya kemasan"
  * delivery_fee  <- "Delivery fee" / "Ongkos kirim"
  * voucher       <- Promo / Discount / Diskon / Voucher; "Min. order"/"Min. belanja",
                     "Max"/"Maks."/"s.d." = max discount
- If a fee line has an info icon or a breakdown, use the total shown on that line.
- If the cart has quantity > 1, menu_unit_price is the price of ONE unit.
- voucher_percent: e.g. 60 for 60%.
- Use 0 (or "") for any field that is not visible. Never invent numbers;
  list anything unclear in uncertain_fields."""


@st.cache_data(show_spinner=False)
def extract(images: tuple, api_key: str) -> dict:
    """images: tuple of (bytes, media_type). Cached so reruns don't re-call the API."""
    client = genai.Client(api_key=api_key)
    contents = [types.Part.from_bytes(data=b, mime_type=mt) for b, mt in images]
    contents.append(PROMPT)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Checkout,
        temperature=0,
    )
    last_err = None
    for model in MODELS:
        for attempt in range(3):            # retry each model up to 3x
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents, config=config)
                return json.loads(resp.text)
            except Exception as e:
                last_err = e
                msg = str(e)
                if not any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500")):
                    raise                    # real error (bad key, bad image) -> stop
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"All Gemini models busy, try again in a minute. Last error: {last_err}")


# ----------------------------------------------------------------------------
# 2. Formulas (same logic as your sheet)
# ----------------------------------------------------------------------------
def num(x):
    try:
        v = float(x)
        return 0.0 if math.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0


def apply_formulas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, r in df.iterrows():
        price = num(r["Menu Net Price"])
        pct = num(r["Cart Voucher %"])
        min_po = num(r["Cart Voucher Min Purchase"])
        max_disc = num(r["Cart Voucher Max Disc"])
        if price <= 0:
            continue
        # Qty needed to reach the voucher's minimum purchase
        qty = max(1, math.ceil(min_po / price)) if min_po else 1
        subtotal = price * qty
        disc = subtotal * pct / 100
        if max_disc:
            disc = min(disc, max_disc)
        basket = subtotal - disc + num(r["Platform Fee"]) + num(r["Packaging Fee"])
        df.at[i, "Qty for Voucher"] = qty
        df.at[i, "Voucher Basket Total"] = round(basket)
        df.at[i, "Price/qty after Voucher"] = round(basket / qty)
    return df


# ----------------------------------------------------------------------------
# 3. Storage
# ----------------------------------------------------------------------------
def load_db() -> pd.DataFrame:
    if os.path.exists(DB_PATH):
        return pd.read_csv(DB_PATH, parse_dates=["Week"])
    return pd.DataFrame(columns=COLUMNS)


def save_db(df: pd.DataFrame):
    df.to_csv(DB_PATH, index=False)


def summary_pivot(db: pd.DataFrame) -> pd.DataFrame:
    if db.empty:
        return pd.DataFrame()
    d = db.copy()
    d["Week"] = pd.to_datetime(d["Week"]).dt.date
    return d.pivot_table(index=["Category", "Brand", "Menu Name"], columns="Week",
                         values="Price/qty after Voucher", aggfunc="last")


def color_changes(row):
    """Red = price went up vs previous week, green = went down."""
    styles, prev = [], None
    for v in row:
        s = ""
        if prev is not None and pd.notna(v) and pd.notna(prev):
            if v > prev:
                s = "background-color:#f4cccc"
            elif v < prev:
                s = "background-color:#b7e1cd"
        styles.append(s)
        prev = v if pd.notna(v) else prev
    return styles


# ----------------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Competitor Price Tracker", layout="wide")
st.title("🍗 Competitor Price Tracker")

with st.sidebar:
    api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password")
    today = date.today()
    week = st.date_input("Week", today - timedelta(days=today.weekday()))
    platform = st.selectbox("Platform", ["GrabFood", "GoFood", "ShopeeFood"])
    db = load_db()
    categories = sorted(db["Category"].dropna().unique().tolist()) or ["1 Pcs Chicken Rice Set"]

tab_in, tab_db, tab_sum = st.tabs(["📤 Upload", "📋 Database", "📊 Summary"])

with tab_in:
    files = st.file_uploader("Checkout screenshots (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"],
                             accept_multiple_files=True)
    one_item = st.checkbox("All uploads belong to ONE menu item (e.g. checkout + voucher detail page)")

    if files and st.button("🔍 Extract", type="primary", disabled=not api_key):
        groups = [files] if one_item else [[f] for f in files]
        rows, prog = [], st.progress(0.0)
        for n, group in enumerate(groups):
            imgs = tuple((f.getvalue(), f.type or "image/png") for f in group)
            try:
                x = extract(imgs, api_key)
            except Exception as e:
                st.error(f"{group[0].name}: {e}")
                continue
            note = x.get("uncertain_fields") or ""
            rows.append({
                "Week": pd.Timestamp(week), "Platform": platform, "Ownership": "Competitor",
                "Category": categories[0], "Brand": x.get("brand", ""),
                "Menu Name": x.get("menu_name", ""),
                "Menu Net Price": x.get("menu_unit_price", 0),
                "Cart Voucher %": x.get("voucher_percent", 0),
                "Cart Voucher Min Purchase": x.get("voucher_min_purchase", 0),
                "Cart Voucher Max Disc": x.get("voucher_max_discount", 0),
                "Platform Fee": x.get("platform_fee", 0),
                "Packaging Fee": x.get("packaging_fee", 0),
                "Notes": f"⚠ check: {note}" if note else "",
                "_source": ", ".join(f.name for f in group),
            })
            prog.progress((n + 1) / len(groups))
        st.session_state["draft"] = pd.DataFrame(rows)

    if "draft" in st.session_state and not st.session_state["draft"].empty:
        st.subheader("Review & fill manual columns")
        st.caption("Yellow-flag anything in Notes. Formula columns recalculate below.")
        draft = st.session_state["draft"]
        edited = st.data_editor(
            draft,
            num_rows="dynamic", use_container_width=True, key="editor",
            column_config={
                "Week": st.column_config.DateColumn(format="D/M/YYYY"),
                "Ownership": st.column_config.SelectboxColumn(options=["Us", "Competitor"]),
                "Category": st.column_config.SelectboxColumn(options=categories + ["(new — type in Notes)"]),
                "_source": st.column_config.TextColumn("Screenshot", disabled=True),
            },
        )
        result = apply_formulas(edited.reindex(columns=COLUMNS + ["_source"]))
        st.subheader("Result")
        st.dataframe(result, use_container_width=True)

        if st.button("💾 Save to database"):
            db = pd.concat([load_db(), result[COLUMNS]], ignore_index=True)
            save_db(db)
            del st.session_state["draft"]
            st.success(f"Saved {len(result)} row(s).")
            st.rerun()

with tab_db:
    db = load_db()
    edited_db = st.data_editor(db, num_rows="dynamic", use_container_width=True, key="dbedit")
    if st.button("Save changes & recalculate"):
        save_db(apply_formulas(edited_db))
        st.rerun()

with tab_sum:
    pv = summary_pivot(load_db())
    if pv.empty:
        st.info("No data yet.")
    else:
        st.caption("Final Price (After Voucher)/Qty — red = up vs last week, green = down")
        st.dataframe(pv.style.apply(color_changes, axis=1).format("{:,.0f}", na_rep=""),
                     use_container_width=True)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            load_db().to_excel(xw, sheet_name="Raw", index=False)
            pv.to_excel(xw, sheet_name="Summary")
        st.download_button("⬇ Download Excel", buf.getvalue(), "competitor_prices.xlsx")
