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
# Category dropdown options (edit this list to add/remove categories)
CATEGORIES = [
    "1 Pcs Chicken Rice Set",
    "1 Pcs Chicken Rice Set (Penyet)",
    "1 Pcs Chicken Rice Set (Geprek)",
    "1 Pcs Chicken Complete Set",
    "2 Pcs Chicken Rice Set",
    "2 Pcs Chicken Complete Set",
    "5 Pcs Wings Ala Carte",
]

# Store names that belong to Hangry (case-insensitive "contains" match)
HANGRY_BRANDS = [
    "Moon Chicken", "Hot Side Story", "Uno Chicken", "Mak Dura", "Ayam Koplo",
    "Bu Tumbar", "Bawang Putri", "TerSereh", "Soto Jeniper", "Barendang",
    "Bishi Bento", "San Gyu", "Salted Lab",
]


def ownership(*names) -> str:
    text = " ".join(str(n or "") for n in names).lower()
    return "Hangry" if any(b.lower() in text for b in HANGRY_BRANDS) else "Competitor"


FORMULA_COLS = ["Qty for Voucher", "Voucher Basket Total", "Price/qty after Voucher"]

# ----------------------------------------------------------------------------
# 1. Extraction with Gemini (vision + JSON schema = reliable structured output)
# ----------------------------------------------------------------------------
class Checkout(BaseModel):
    brand: str = Field(description="Restaurant / brand name, e.g. 'KFC'")
    store_name: str = Field(description="Full outlet name as shown")
    menu_name: str = Field(description="Menu item name exactly as shown")
    menu_unit_price: float = Field(description="Price of ONE unit in IDR, before voucher")
    quantity_in_cart: int = Field(description="Quantity of the menu item in the cart: the number in the circle/badge next to the item (e.g. 3). 1 if not shown")
    subtotal: float = Field(description="'Subtotal (Incl. Tax)' / 'Subtotal (termasuk pajak)' amount")
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
- subtotal <- the "Subtotal (Incl. Tax)" / "Subtotal (termasuk pajak)" line. This is the key value.
- quantity_in_cart <- the number inside the circle next to the item (quantity stepper).
- If the cart has quantity > 1, menu_unit_price is the price of ONE unit.
- voucher_percent: e.g. 60 for 60%.
- Use 0 (or "") for any field that is not visible. Never invent numbers;
  list anything unclear in uncertain_fields."""


def net_price(x: dict) -> float:
    """Menu Net Price = Subtotal (Incl. Tax) per unit; falls back to the item price."""
    sub = num(x.get("subtotal"))
    qty = max(1, int(num(x.get("quantity_in_cart")) or 1))
    return round(sub / qty) if sub > 0 else num(x.get("menu_unit_price"))


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
        # Qty = cart qty from screenshot, raised if the voucher's min purchase needs more
        cart_qty = max(1, int(num(r.get("Qty for Voucher")) or 1))
        need_qty = math.ceil(min_po / price) if min_po else 1
        qty = max(cart_qty, need_qty)
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
# Data is stored in a Google Sheet through a small Apps Script web app (no Cloud console).
# Secrets needed:
#   SHEET_WEBAPP_URL = "https://script.google.com/macros/s/.../exec"
#   SHEET_TOKEN      = "same secret word as TOKEN in the Apps Script"
# Without them the app falls back to a local CSV, which Streamlit Cloud WIPES on restart.
NUM_COLS = ["Menu Net Price", "Cart Voucher %", "Cart Voucher Min Purchase",
            "Cart Voucher Max Disc", "Qty for Voucher", "Platform Fee", "Packaging Fee",
            "Voucher Basket Total", "Price/qty after Voucher"]


def use_sheets() -> bool:
    try:
        return "SHEET_WEBAPP_URL" in st.secrets and "SHEET_TOKEN" in st.secrets
    except Exception:
        return False


def sheet_call(action: str, rows=None):
    import requests
    url, token = st.secrets["SHEET_WEBAPP_URL"], st.secrets["SHEET_TOKEN"]
    if action == "read":
        r = requests.get(url, params={"token": token}, timeout=60)
    else:
        r = requests.post(url, data=json.dumps({"token": token, "action": action, "rows": rows,
                                          "header": COLUMNS}),
                          headers={"Content-Type": "text/plain"}, timeout=60)
    r.raise_for_status()
    try:
        res = r.json()
    except ValueError:
        hint = "Google returned a sign-in page" if "accounts.google" in r.url or "<html" in r.text[:200].lower() else "Unexpected reply"
        raise RuntimeError(f"{hint}. The Apps Script web app must be deployed with access "
                           "'Anyone' (work accounts may block this; use a personal Gmail sheet).")
    if "error" in res:
        raise RuntimeError(f"Google Sheet: {res['error']}")
    return res


def to_rows(df: pd.DataFrame) -> list:
    out = clean(df)
    out["Week"] = out["Week"].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(out.notna(), "")
    return out.values.tolist()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=COLUMNS)
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce", dayfirst=False)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_db() -> pd.DataFrame:
    if use_sheets():
        values = sheet_call("read").get("rows", [])
        data = values[1:] if values and values[0] == COLUMNS else values
        return clean(pd.DataFrame(data, columns=COLUMNS) if data else pd.DataFrame(columns=COLUMNS))
    if os.path.exists(DB_PATH):
        return clean(pd.read_csv(DB_PATH))
    return clean(pd.DataFrame(columns=COLUMNS))


def safe_load_db() -> pd.DataFrame:
    """Load without crashing the app; show the real reason on screen."""
    try:
        df = load_db()
        st.session_state["db_ok"] = True
        return df
    except Exception as e:
        st.session_state["db_ok"] = False
        st.error(f"Could not read the Google Sheet: {e}\n\n"
                 "Check: (1) Apps Script deployment access = 'Anyone', "
                 "(2) SHEET_TOKEN matches TOKEN in the script, "
                 "(3) SHEET_WEBAPP_URL ends with /exec.")
        return clean(pd.DataFrame(columns=COLUMNS))


def save_db(df: pd.DataFrame):
    """Replace everything (used after editing the Database tab)."""
    if use_sheets():
        sheet_call("replace", [COLUMNS] + to_rows(df))
    else:
        clean(df).to_csv(DB_PATH, index=False)


def append_db(df: pd.DataFrame):
    """Add new rows only (used when saving new screenshots)."""
    if use_sheets():
        sheet_call("append", to_rows(df))
    else:
        save_db(pd.concat([load_db(), clean(df)], ignore_index=True))


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
if not use_sheets():
    st.warning("⚠ Not connected to Google Sheets. Data is saved temporarily and will be "
               "lost when the app restarts. Download the Excel after each session.")

with st.sidebar:
    api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password")
    today = date.today()
    week = st.date_input("Week", today - timedelta(days=today.weekday()))
    platform = st.selectbox("Platform", ["GrabFood", "GoFood", "ShopeeFood"])
    db = safe_load_db()
    # dropdown list + any older categories already saved in the database
    categories = CATEGORIES + [c for c in db["Category"].dropna().unique() if c not in CATEGORIES]
    batch_category = st.selectbox(
        "Category for this batch", categories,
        help="Applied to new screenshots. You can still change each row in the table.")

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
                "Week": pd.Timestamp(week), "Platform": platform,
                "Ownership": ownership(x.get("store_name"), x.get("brand")),
                "Category": batch_category, "Brand": x.get("brand", ""),
                "Menu Name": x.get("menu_name", ""),
                "Menu Net Price": net_price(x),
                "Qty for Voucher": max(1, int(num(x.get("quantity_in_cart")) or 1)),
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
                "Ownership": st.column_config.SelectboxColumn(options=["Hangry", "Competitor"]),
                "Category": st.column_config.SelectboxColumn("Category", options=categories, required=True),
                "_source": st.column_config.TextColumn("Screenshot", disabled=True),
            },
        )
        result = apply_formulas(edited.reindex(columns=COLUMNS + ["_source"]))
        st.subheader("Result")
        st.dataframe(result, use_container_width=True)

        if st.button("💾 Save to database"):
            try:
                append_db(result[COLUMNS])
            except Exception as e:
                st.error(f"Could not save to Google Sheet: {e}")
                st.stop()
            del st.session_state["draft"]
            st.success(f"Saved {len(result)} row(s).")
            st.rerun()

with tab_db:
    with st.expander("Import old data (Excel/CSV downloaded from this app or your sheet)"):
        up = st.file_uploader("File", type=["xlsx", "csv"], key="import")
        if up is not None and st.button("Add to database"):
            new = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up, sheet_name=0)
            append_db(clean(new))
            st.success(f"Imported {len(new)} row(s).")
            st.rerun()
    db = safe_load_db()
    edited_db = st.data_editor(db, num_rows="dynamic", use_container_width=True, key="dbedit")
    if st.button("Save changes & recalculate", disabled=not st.session_state.get("db_ok", False)):
        save_db(apply_formulas(edited_db))
        st.rerun()

with tab_sum:
    pv = summary_pivot(safe_load_db())
    if pv.empty:
        st.info("No data yet.")
    else:
        st.caption("Final Price (After Voucher)/Qty — red = up vs last week, green = down")
        st.dataframe(pv.style.apply(color_changes, axis=1).format("{:,.0f}", na_rep=""),
                     use_container_width=True)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            safe_load_db().to_excel(xw, sheet_name="Raw", index=False)
            pv.to_excel(xw, sheet_name="Summary")
        st.download_button("⬇ Download Excel", buf.getvalue(), "competitor_prices.xlsx")
