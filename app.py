import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import date, datetime
import calendar

# =========================
# Budget App (Full Version)
# =========================
# Features:
# - SQLite persistence (transactions, budgets, recurring rules)
# - Add income/expense (forms) with "Saved"
# - Month filter (All time + per-month)
# - KPI cards (Income/Expenses/Net)
# - Transactions table with Edit + Delete (true CRUD)
# - Search + filters (type/category/date range)
# - Category budgets per month + remaining
# - Recurring transactions (monthly/weekly/biweekly) with "Generate for month"
# - Import CSV (mapped) + Export CSV
# - Data hygiene (date normalization, safe types)
# - Reset buttons and confirmations

APP_TITLE = "Budget App (Full)"
DB_PATH = Path("budget_app.db")

st.set_page_config(page_title=APP_TITLE, layout="wide")

# -------------------------
# DB Utilities
# -------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dt TEXT NOT NULL,              -- ISO datetime/date string
        type TEXT NOT NULL,            -- 'Income'/'Expense'
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        note TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT NOT NULL,           -- 'YYYY-MM'
        category TEXT NOT NULL,
        budget REAL NOT NULL,
        UNIQUE(month, category)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS recurring_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,            -- 'Income'/'Expense'
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        note TEXT,
        freq TEXT NOT NULL,            -- 'Monthly'/'Weekly'/'Biweekly'
        day_of_month INTEGER,          -- for Monthly: 1-31
        weekday INTEGER,               -- for Weekly/Biweekly: 0=Mon..6=Sun
        start_dt TEXT NOT NULL,        -- ISO date
        active INTEGER NOT NULL DEFAULT 1
    );
    """)

    conn.commit()
    conn.close()

def iso_date(d: date) -> str:
    return pd.to_datetime(d).date().isoformat()

def iso_dt_str_to_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")

def month_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")

def run_query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def exec_sql(sql: str, params: tuple = ()) -> None:
    conn = get_conn()
    conn.execute(sql, params)
    conn.commit()
    conn.close()

def exec_many(sql: str, rows: list[tuple]) -> None:
    conn = get_conn()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()

# -------------------------
# Data Access
# -------------------------
def fetch_transactions() -> pd.DataFrame:
    df = run_query_df("SELECT id, dt, type, category, amount, note FROM transactions")
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["type"] = df["type"].astype(str)
    df["category"] = df["category"].astype(str)
    df["note"] = df["note"].astype("string")
    df = df.dropna(subset=["dt"])
    return df

def add_transaction(dt_val: date, txn_type: str, category: str, amount: float, note: str) -> None:
    exec_sql(
        "INSERT INTO transactions (dt, type, category, amount, note) VALUES (?, ?, ?, ?, ?)",
        (iso_date(dt_val), txn_type, category.strip(), float(amount), (note or "").strip() or None),
    )

def update_transaction(txn_id: int, dt_val: date, txn_type: str, category: str, amount: float, note: str) -> None:
    exec_sql(
        "UPDATE transactions SET dt=?, type=?, category=?, amount=?, note=? WHERE id=?",
        (iso_date(dt_val), txn_type, category.strip(), float(amount), (note or "").strip() or None, int(txn_id)),
    )

def delete_transaction(txn_id: int) -> None:
    exec_sql("DELETE FROM transactions WHERE id=?", (int(txn_id),))

def fetch_budgets(month: str) -> pd.DataFrame:
    df = run_query_df("SELECT month, category, budget FROM budgets WHERE month=?", (month,))
    if df.empty:
        return df
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0.0)
    return df

def upsert_budgets(month: str, budgets: pd.DataFrame) -> None:
    # budgets columns: category, budget
    rows = []
    for _, r in budgets.iterrows():
        cat = str(r["category"]).strip()
        bud = float(pd.to_numeric(r["budget"], errors="coerce") or 0.0)
        if cat:
            rows.append((month, cat, bud))
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM budgets WHERE month=?", (month,))
    cur.executemany("INSERT INTO budgets (month, category, budget) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()

def fetch_rules(active_only: bool = True) -> pd.DataFrame:
    if active_only:
        df = run_query_df("""SELECT * FROM recurring_rules WHERE active=1""")
    else:
        df = run_query_df("""SELECT * FROM recurring_rules""")
    if df.empty:
        return df
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["day_of_month"] = pd.to_numeric(df["day_of_month"], errors="coerce").astype("Int64")
    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").astype("Int64")
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df["active"] = df["active"].astype(int)
    return df

def add_rule(rule: dict) -> None:
    exec_sql(
        """
        INSERT INTO recurring_rules (type, category, amount, note, freq, day_of_month, weekday, start_dt, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rule["type"],
            rule["category"].strip(),
            float(rule["amount"]),
            (rule.get("note") or "").strip() or None,
            rule["freq"],
            rule.get("day_of_month"),
            rule.get("weekday"),
            iso_date(rule["start_dt"]),
            1 if rule.get("active", True) else 0,
        ),
    )

def set_rule_active(rule_id: int, active: bool) -> None:
    exec_sql("UPDATE recurring_rules SET active=? WHERE id=?", (1 if active else 0, int(rule_id)))

def delete_rule(rule_id: int) -> None:
    exec_sql("DELETE FROM recurring_rules WHERE id=?", (int(rule_id),))

# -------------------------
# Recurring Generation
# -------------------------
def month_start_end(month_str: str) -> tuple[date, date]:
    y, m = map(int, month_str.split("-"))
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, 1), date(y, m, last_day)

def generate_dates_for_rule(rule_row: pd.Series, month_str: str) -> list[date]:
    start, end = month_start_end(month_str)
    rule_start = pd.to_datetime(rule_row["start_dt"]).date()

    if end < rule_start:
        return []

    out: list[date] = []

    freq = rule_row["freq"]
    if freq == "Monthly":
        dom = int(rule_row["day_of_month"]) if pd.notna(rule_row["day_of_month"]) else 1
        dom = max(1, min(31, dom))
        # clamp to month length
        last_day = calendar.monthrange(start.year, start.month)[1]
        dom = min(dom, last_day)
        d = date(start.year, start.month, dom)
        if d >= start and d <= end and d >= rule_start:
            out.append(d)

    elif freq in ("Weekly", "Biweekly"):
        weekday = int(rule_row["weekday"]) if pd.notna(rule_row["weekday"]) else 0
        weekday = max(0, min(6, weekday))

        # find first occurrence of weekday on/after max(start, rule_start)
        anchor = max(start, rule_start)
        days_ahead = (weekday - anchor.weekday()) % 7
        first = anchor + pd.Timedelta(days=days_ahead)
        step = 7 if freq == "Weekly" else 14

        d = first
        while d <= pd.Timestamp(end):
            out.append(pd.to_datetime(d).date())
            d = pd.Timestamp(d) + pd.Timedelta(days=step)

        # de-dupe just in case
        out = sorted(set(out))

    return out

def generate_recurring_for_month(month_str: str, dry_run: bool = False) -> pd.DataFrame:
    """
    Create transactions for active rules in the given month.
    Avoid duplicates by checking (dt, type, category, amount, note) existing rows.
    Returns a DataFrame of the transactions that would be/were added.
    """
    rules = fetch_rules(active_only=True)
    if rules.empty:
        return pd.DataFrame(columns=["dt", "type", "category", "amount", "note", "rule_id"])

    existing = fetch_transactions()
    existing_key = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            key = (
                pd.to_datetime(r["dt"]).date().isoformat(),
                r["type"],
                r["category"],
                float(r["amount"]),
                (str(r["note"]) if pd.notna(r["note"]) else ""),
            )
            existing_key.add(key)

    rows_to_add = []
    for _, rule in rules.iterrows():
        dates = generate_dates_for_rule(rule, month_str)
        for d in dates:
            key = (
                d.isoformat(),
                rule["type"],
                str(rule["category"]),
                float(rule["amount"]),
                (str(rule["note"]) if pd.notna(rule["note"]) else ""),
            )
            if key in existing_key:
                continue
            rows_to_add.append({
                "dt": d,
                "type": rule["type"],
                "category": str(rule["category"]),
                "amount": float(rule["amount"]),
                "note": (str(rule["note"]) if pd.notna(rule["note"]) else ""),
                "rule_id": int(rule["id"]),
            })

    add_df = pd.DataFrame(rows_to_add)
    if add_df.empty:
        return add_df

    if not dry_run:
        exec_many(
            "INSERT INTO transactions (dt, type, category, amount, note) VALUES (?, ?, ?, ?, ?)",
            [
                (r["dt"].isoformat(), r["type"], r["category"], float(r["amount"]), (r["note"] or "").strip() or None)
                for _, r in add_df.iterrows()
            ],
        )
    return add_df

# -------------------------
# CSV Import/Export
# -------------------------
def export_csv(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out["dt"] = pd.to_datetime(out["dt"]).dt.date.astype(str)
    csv = out[["dt", "type", "category", "amount", "note"]].to_csv(index=False)
    return csv.encode("utf-8")

def import_csv_to_db(csv_bytes: bytes, mapping: dict) -> tuple[int, int]:
    """
    mapping: {'date': col, 'type': col, 'category': col, 'amount': col, 'note': col or None}
    Returns: (inserted_count, skipped_count)
    """
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    required = ["date", "type", "category", "amount"]
    for k in required:
        if k not in mapping or mapping[k] not in df.columns:
            raise ValueError(f"Missing mapping for {k}")

    tmp = pd.DataFrame()
    tmp["dt"] = pd.to_datetime(df[mapping["date"]], errors="coerce")
    tmp["type"] = df[mapping["type"]].astype(str).str.strip()
    tmp["category"] = df[mapping["category"]].astype(str).str.strip()
    tmp["amount"] = pd.to_numeric(df[mapping["amount"]], errors="coerce")
    tmp["note"] = ""
    if mapping.get("note") and mapping["note"] in df.columns:
        tmp["note"] = df[mapping["note"]].astype(str)

    tmp = tmp.dropna(subset=["dt", "amount"])
    tmp = tmp[tmp["type"].isin(["Income", "Expense"])]
    tmp = tmp[tmp["category"].astype(str).str.len() > 0]

    if tmp.empty:
        return (0, 0)

    existing = fetch_transactions()
    existing_key = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            key = (
                pd.to_datetime(r["dt"]).date().isoformat(),
                r["type"],
                r["category"],
                float(r["amount"]),
                (str(r["note"]) if pd.notna(r["note"]) else ""),
            )
            existing_key.add(key)

    inserted = 0
    skipped = 0
    rows = []
    for _, r in tmp.iterrows():
        key = (
            pd.to_datetime(r["dt"]).date().isoformat(),
            r["type"],
            r["category"],
            float(r["amount"]),
            (str(r["note"]) if pd.notna(r["note"]) else ""),
        )
        if key in existing_key:
            skipped += 1
            continue
        rows.append((
            pd.to_datetime(r["dt"]).date().isoformat(),
            r["type"],
            r["category"],
            float(r["amount"]),
            (str(r["note"]) if pd.notna(r["note"]) else "").strip() or None
        ))
        inserted += 1

    if rows:
        exec_many("INSERT INTO transactions (dt, type, category, amount, note) VALUES (?, ?, ?, ?, ?)", rows)

    return (inserted, skipped)

# -------------------------
# App State
# -------------------------
init_db()

if "flash" not in st.session_state:
    st.session_state.flash = ""

def set_flash(msg: str) -> None:
    st.session_state.flash = msg

def show_flash():
    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = ""

# -------------------------
# Load Data
# -------------------------
tx = fetch_transactions()
if not tx.empty:
    tx["month"] = tx["dt"].dt.to_period("M").astype(str)

# -------------------------
# Top Controls
# -------------------------
st.title(APP_TITLE)
show_flash()

left, mid, right = st.columns([1.4, 1.2, 1.2])

with left:
    if tx.empty:
        month_options = ["All time"]
    else:
        month_options = ["All time"] + sorted(tx["month"].unique())[::-1]
    selected_month = st.selectbox("View month", month_options)

with mid:
    q_search = st.text_input("Search (category/note)", value="")

with right:
    filter_type = st.selectbox("Type filter", ["All", "Income", "Expense"])

# -------------------------
# Derive View DF
# -------------------------
view = tx.copy()
if not view.empty:
    if selected_month != "All time":
        view = view[view["month"] == selected_month]
    if filter_type != "All":
        view = view[view["type"] == filter_type]
    if q_search.strip():
        s = q_search.strip().lower()
        view = view[
            view["category"].astype(str).str.lower().str.contains(s, na=False)
            | view["note"].astype(str).str.lower().str.contains(s, na=False)
        ]

# -------------------------
# KPIs
# -------------------------
def kpi(df: pd.DataFrame) -> tuple[float, float, float]:
    if df is None or df.empty:
        return (0.0, 0.0, 0.0)
    inc = df[df["type"] == "Income"]["amount"].sum()
    exp = df[df["type"] == "Expense"]["amount"].sum()
    return (float(inc), float(exp), float(inc - exp))

inc, exp, net = kpi(view)

c1, c2, c3 = st.columns(3)
c1.metric("Income", f"${inc:,.2f}")
c2.metric("Expenses", f"${exp:,.2f}")
c3.metric("Net", f"${net:,.2f}")

st.divider()

# =========================
# Sidebar: Add + Recurring
# =========================
with st.sidebar:
    st.header("Add Income")

    with st.form("income_form", clear_on_submit=True):
        i_cat = st.text_input("Income Source", placeholder="Paycheck, Bonus, etc.")
        i_amt = st.number_input("Income Amount", min_value=0.0, step=0.01)
        i_dt = st.date_input("Income Date", value=date.today())
        i_note = st.text_input("Note (optional)", key="i_note")
        i_submit = st.form_submit_button("Save Income")
        if i_submit:
            if i_cat.strip() and i_amt > 0:
                add_transaction(i_dt, "Income", i_cat, i_amt, i_note)
                set_flash("Saved")
                st.rerun()
            else:
                st.error("Income source and amount required")

    st.divider()

    st.header("Add Expense")

    with st.form("expense_form", clear_on_submit=True):
        e_cat = st.text_input("Expense Category", placeholder="Rent, Groceries, etc.")
        e_amt = st.number_input("Expense Amount", min_value=0.0, step=0.01, key="e_amt")
        e_dt = st.date_input("Expense Date", value=date.today(), key="e_dt")
        e_note = st.text_input("Note (optional)", key="e_note")
        e_submit = st.form_submit_button("Save Expense")
        if e_submit:
            if e_cat.strip() and e_amt > 0:
                add_transaction(e_dt, "Expense", e_cat, e_amt, e_note)
                set_flash("Saved")
                st.rerun()
            else:
                st.error("Expense category and amount required")

    st.divider()

    st.header("Recurring Rules")

    with st.expander("Add recurring rule", expanded=False):
        r_type = st.selectbox("Type", ["Expense", "Income"], key="r_type")
        r_cat = st.text_input("Category", key="r_cat")
        r_amt = st.number_input("Amount", min_value=0.0, step=0.01, key="r_amt")
        r_note = st.text_input("Note (optional)", key="r_note")

        r_freq = st.selectbox("Frequency", ["Monthly", "Weekly", "Biweekly"], key="r_freq")

        dom = None
        wd = None
        if r_freq == "Monthly":
            dom = st.number_input("Day of month (1-31)", min_value=1, max_value=31, value=1, step=1)
        else:
            wd_label = st.selectbox("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(wd_label)

        r_start = st.date_input("Start date", value=date.today(), key="r_start")

        if st.button("Save rule"):
            if r_cat.strip() and r_amt > 0:
                add_rule({
                    "type": r_type,
                    "category": r_cat,
                    "amount": r_amt,
                    "note": r_note,
                    "freq": r_freq,
                    "day_of_month": int(dom) if dom is not None else None,
                    "weekday": int(wd) if wd is not None else None,
                    "start_dt": r_start,
                    "active": True,
                })
                set_flash("Saved")
                st.rerun()
            else:
                st.error("Category and amount required")

    rules = fetch_rules(active_only=False)
    if rules.empty:
        st.caption("No recurring rules yet.")
    else:
        for _, r in rules.sort_values("active", ascending=False).iterrows():
            label = f"#{int(r['id'])} | {r['type']} | {r['category']} | ${float(r['amount']):,.2f} | {r['freq']}"
            with st.expander(label, expanded=False):
                st.write(f"Start: {pd.to_datetime(r['start_dt']).date().isoformat()}")
                if r["freq"] == "Monthly" and pd.notna(r["day_of_month"]):
                    st.write(f"Day of month: {int(r['day_of_month'])}")
                if r["freq"] in ("Weekly", "Biweekly") and pd.notna(r["weekday"]):
                    st.write(f"Weekday: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(r['weekday'])]}")
                st.write(f"Note: {r['note'] if pd.notna(r['note']) else ''}")

                colA, colB, colC = st.columns(3)
                with colA:
                    if r["active"] == 1:
                        if st.button("Deactivate", key=f"deact_{int(r['id'])}"):
                            set_rule_active(int(r["id"]), False)
                            set_flash("Saved")
                            st.rerun()
                    else:
                        if st.button("Activate", key=f"act_{int(r['id'])}"):
                            set_rule_active(int(r["id"]), True)
                            set_flash("Saved")
                            st.rerun()
                with colB:
                    if selected_month != "All time":
                        if st.button("Generate for selected month", key=f"gen_{int(r['id'])}"):
                            # generate all active rules for month; keeps UX simple
                            added = generate_recurring_for_month(selected_month, dry_run=False)
                            set_flash(f"Saved ({len(added)} added)")
                            st.rerun()
                    else:
                        st.caption("Pick a month to generate.")
                with colC:
                    if st.button("Delete rule", key=f"delrule_{int(r['id'])}"):
                        delete_rule(int(r["id"]))
                        set_flash("Saved")
                        st.rerun()

# =========================
# Main: Budgets
# =========================
st.subheader("Budgets")
if selected_month == "All time":
    st.info("Select a specific month to set and track budgets.")
else:
    month_tx = tx.copy()
    if not month_tx.empty:
        month_tx = month_tx[month_tx["month"] == selected_month]

    month_exp = month_tx[month_tx["type"] == "Expense"] if not month_tx.empty else pd.DataFrame()
    spent = (
        month_exp.groupby("category")["amount"].sum().reset_index().rename(columns={"amount": "spent"})
        if not month_exp.empty else pd.DataFrame(columns=["category", "spent"])
    )

    b = fetch_budgets(selected_month)

    merged = spent.merge(b, on="category", how="left")
    merged["budget"] = pd.to_numeric(merged.get("budget", 0), errors="coerce").fillna(0.0)
    merged["remaining"] = merged["budget"] - merged["spent"]

    # Ensure categories with budget but no spend also show up
    if not b.empty:
        only_b = b[~b["category"].isin(spent["category"] if not spent.empty else [])].copy()
        if not only_b.empty:
            only_b["spent"] = 0.0
            only_b["remaining"] = only_b["budget"] - only_b["spent"]
            merged = pd.concat([merged, only_b[["category", "spent", "budget", "remaining"]]], ignore_index=True)

    merged = merged.sort_values("spent", ascending=False)

    st.caption("Edit Budget values and click Save Budgets.")
    edited = st.data_editor(
        merged[["category", "spent", "budget", "remaining"]],
        disabled=["spent", "remaining"],
        use_container_width=True,
        key=f"bud_editor_{selected_month}",
    )
    colS, colAdd = st.columns([1, 2])
    with colS:
        if st.button("Save Budgets"):
            to_save = edited[["category", "budget"]].copy()
            to_save["budget"] = pd.to_numeric(to_save["budget"], errors="coerce").fillna(0.0)
            upsert_budgets(selected_month, to_save)
            set_flash("Saved")
            st.rerun()
    with colAdd:
        with st.expander("Add a new budget category", expanded=False):
            new_cat = st.text_input("New category name", key="new_budget_cat")
            new_bud = st.number_input("Budget amount", min_value=0.0, step=0.01, key="new_budget_amt")
            if st.button("Add budget category"):
                if new_cat.strip():
                    current = fetch_budgets(selected_month)
                    row = pd.DataFrame([{"month": selected_month, "category": new_cat.strip(), "budget": float(new_bud)}])
                    # upsert behavior
                    conn = get_conn()
                    conn.execute(
                        "INSERT INTO budgets (month, category, budget) VALUES (?, ?, ?) "
                        "ON CONFLICT(month, category) DO UPDATE SET budget=excluded.budget",
                        (selected_month, new_cat.strip(), float(new_bud))
                    )
                    conn.commit()
                    conn.close()
                    set_flash("Saved")
                    st.rerun()
                else:
                    st.error("Category required")

st.divider()

# =========================
# Main: Transactions (CRUD)
# =========================
st.subheader("Transactions (Edit/Delete)")

if view.empty:
    st.info("No transactions in this view.")
else:
    display = view.copy()
    display = display.sort_values("dt", ascending=False)
    display["dt"] = display["dt"].dt.date.astype(str)
    display = display[["id", "dt", "type", "category", "amount", "note"]]

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Select a transaction ID below to edit or delete.")
    all_ids = display["id"].tolist()
    selected_id = st.selectbox("Transaction ID", all_ids)

    txn_row = run_query_df(
        "SELECT id, dt, type, category, amount, note FROM transactions WHERE id=?",
        (int(selected_id),)
    )
    if not txn_row.empty:
        r = txn_row.iloc[0]
        edit_dt = pd.to_datetime(r["dt"]).date()
        edit_type = r["type"]
        edit_cat = r["category"]
        edit_amt = float(r["amount"])
        edit_note = "" if pd.isna(r["note"]) else str(r["note"])

        with st.expander("Edit selected transaction", expanded=True):
            with st.form("edit_form"):
                e1, e2 = st.columns(2)
                with e1:
                    new_dt = st.date_input("Date", value=edit_dt)
                    new_type = st.selectbox("Type", ["Income", "Expense"], index=0 if edit_type == "Income" else 1)
                with e2:
                    new_cat = st.text_input("Category", value=edit_cat)
                    new_amt = st.number_input("Amount", min_value=0.0, step=0.01, value=edit_amt)
                new_note = st.text_input("Note", value=edit_note)

                save_edit = st.form_submit_button("Save changes")

                if save_edit:
                    if new_cat.strip() and new_amt > 0:
                        update_transaction(int(selected_id), new_dt, new_type, new_cat, new_amt, new_note)
                        set_flash("Saved")
                        st.rerun()
                    else:
                        st.error("Category and amount required")

        with st.expander("Delete selected transaction", expanded=False):
            st.warning("This is permanent.")
            confirm = st.text_input("Type DELETE to confirm", value="", key="del_confirm")
            if st.button("Delete transaction"):
                if confirm.strip().upper() == "DELETE":
                    delete_transaction(int(selected_id))
                    set_flash("Saved")
                    st.rerun()
                else:
                    st.error("You must type DELETE to confirm.")

st.divider()

# =========================
# Main: Insights
# =========================
st.subheader("Insights")

if view.empty:
    st.info("Add some transactions to see insights.")
else:
    exp_view = view[view["type"] == "Expense"].copy()
    if exp_view.empty:
        st.info("No expenses in this view.")
    else:
        by_cat = exp_view.groupby("category")["amount"].sum().sort_values(ascending=False)
        st.caption("Spending by Category")
        st.bar_chart(by_cat)

        top_n = min(5, len(by_cat))
        st.caption(f"Top {top_n} spending categories")
        top_df = by_cat.head(top_n).reset_index()
        top_df.columns = ["category", "spent"]
        st.dataframe(top_df, use_container_width=True, hide_index=True)

st.divider()

# =========================
# Import / Export
# =========================
st.subheader("Import / Export")

cA, cB = st.columns(2)

with cA:
    st.caption("Export current view to CSV")
    if not view.empty:
        csv_bytes = export_csv(view.rename(columns={"dt": "dt"}))
        st.download_button(
            "Download CSV (current view)",
            data=csv_bytes,
            file_name="budget_export.csv",
            mime="text/csv"
        )
    else:
        st.caption("Nothing to export yet.")

with cB:
    st.caption("Import CSV (map columns)")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        st.dataframe(df_in.head(20), use_container_width=True)

        cols = list(df_in.columns)
        m_date = st.selectbox("Date column", cols, index=0)
        m_type = st.selectbox("Type column (must be Income/Expense)", cols, index=1 if len(cols) > 1 else 0)
        m_cat = st.selectbox("Category column", cols, index=2 if len(cols) > 2 else 0)
        m_amt = st.selectbox("Amount column", cols, index=3 if len(cols) > 3 else 0)
        m_note = st.selectbox("Note column (optional)", ["(none)"] + cols)

        if st.button("Import now"):
            try:
                inserted, skipped = import_csv_to_db(
                    up.getvalue(),
                    {
                        "date": m_date,
                        "type": m_type,
                        "category": m_cat,
                        "amount": m_amt,
                        "note": None if m_note == "(none)" else m_note,
                    }
                )
                set_flash(f"Saved ({inserted} imported, {skipped} skipped)")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

st.divider()

# =========================
# Admin / Reset
# =========================
st.subheader("Admin")

a1, a2, a3 = st.columns(3)

with a1:
    if st.button("Generate recurring for selected month"):
        if selected_month == "All time":
            st.error("Pick a month first.")
        else:
            added = generate_recurring_for_month(selected_month, dry_run=False)
            set_flash(f"Saved ({len(added)} added)")
            st.rerun()

with a2:
    if st.button("Vacuum DB (cleanup)"):
        conn = get_conn()
        conn.execute("VACUUM;")
        conn.commit()
        conn.close()
        set_flash("Saved")
        st.rerun()

with a3:
    with st.expander("Nuke everything (danger)", expanded=False):
        st.warning("Deletes ALL transactions, budgets, and recurring rules.")
        confirm_all = st.text_input("Type NUKE to confirm", value="", key="nuke_confirm")
        if st.button("NUKE ALL DATA"):
            if confirm_all.strip().upper() == "NUKE":
                conn = get_conn()
                conn.execute("DELETE FROM transactions;")
                conn.execute("DELETE FROM budgets;")
                conn.execute("DELETE FROM recurring_rules;")
                conn.commit()
                conn.close()
                set_flash("Saved")
                st.rerun()
            else:
                st.error("Type NUKE to confirm.")
