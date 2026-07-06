import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
DEFAULT_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
API_BASE = "https://api.football-data.org/v4"
CACHE_TTL = 60 * 60

st.set_page_config(page_title="Football Probabilities Pro", layout="wide")
st.title("Football Probabilities Pro")

COMPETITIONS_FALLBACK = {
    "Premier League": "PL",
    "Champions League": "CL",
    "Ligue 1": "FL1",
    "Ligue 2": "FL2",
    "Bundesliga": "BL1",
    "2. Bundesliga": "BL2",
    "Serie A": "SA",
    "Serie B": "SB",
    "La Liga": "PD",
    "Eredivisie": "DED",
    "Primeira Liga": "PPL",
    "Championship": "ELC",
    "Euro": "EC",
    "World Cup": "WC",
}

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def implied_prob(odds):
    odds = safe_float(odds)
    if np.isnan(odds) or odds <= 0:
        return np.nan
    return 1.0 / odds

def pct(x):
    return round(float(x) * 100, 2) if x is not None and not np.isnan(x) else np.nan

def score_label(i, j):
    return f"{i}-{j}"

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"home_team", "away_team", "home_goals", "away_goals"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: home_team, away_team, home_goals, away_goals")
    if "date" not in df.columns:
        df["date"] = pd.NaT
    if "competition" not in df.columns:
        df["competition"] = "unknown"
    if "country" not in df.columns:
        df["country"] = "unknown"
    return df

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_competitions(token):
    url = f"{API_BASE}/competitions"
    headers = {"X-Auth-Token": token}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("competitions", [])

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_matches(token, competition_id, season):
    url = f"{API_BASE}/competitions/{competition_id}/matches"
    headers = {"X-Auth-Token": token}
    params = {"season": season, "status": "FINISHED"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("matches", [])
    rows = []
    for m in data:
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home", None)
        ag = ft.get("away", None)
        if hg is None or ag is None:
            continue
        comp = m.get("competition", {})
        area = comp.get("area", {})
        rows.append({
            "date": m.get("utcDate", None),
            "competition": comp.get("name", competition_id),
            "competition_code": comp.get("code", competition_id),
            "country": area.get("name", "unknown"),
            "home_team": m.get("homeTeam", {}).get("name", ""),
            "away_team": m.get("awayTeam", {}).get("name", ""),
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(rows)

def league_averages(df):
    return {
        "home_avg": df["home_goals"].mean() if len(df) else 1.35,
        "away_avg": df["away_goals"].mean() if len(df) else 1.10
    }

def team_stats(df, team):
    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]
    return {
        "home_scored": home["home_goals"].mean() if len(home) else np.nan,
        "home_conceded": home["away_goals"].mean() if len(home) else np.nan,
        "away_scored": away["away_goals"].mean() if len(away) else np.nan,
        "away_conceded": away["home_goals"].mean() if len(away) else np.nan,
        "home_matches": len(home),
        "away_matches": len(away),
    }

def estimate_lambdas(df, home_team, away_team, blend=0.65):
    la = league_averages(df)
    hs = team_stats(df, home_team)
    as_ = team_stats(df, away_team)
    home_attack = hs["home_scored"] if not np.isnan(hs["home_scored"]) else la["home_avg"]
    home_def = hs["home_conceded"] if not np.isnan(hs["home_conceded"]) else la["away_avg"]
    away_attack = as_["away_scored"] if not np.isnan(as_["away_scored"]) else la["away_avg"]
    away_def = as_["away_conceded"] if not np.isnan(as_["away_conceded"]) else la["home_avg"]
    lam_home = blend * (home_attack * away_def / max(la["away_avg"], 0.01)) + (1 - blend) * la["home_avg"]
    lam_away = blend * (away_attack * home_def / max(la["home_avg"], 0.01)) + (1 - blend) * la["away_avg"]
    return max(lam_home, 0.05), max(lam_away, 0.05)

def score_matrix(lh, la, max_goals=8):
    hp = poisson.pmf(np.arange(max_goals + 1), lh)
    ap = poisson.pmf(np.arange(max_goals + 1), la)
    mat = np.outer(hp, ap)
    return pd.DataFrame(mat, index=range(max_goals + 1), columns=range(max_goals + 1))

def probs_1x2(mat):
    arr = mat.values
    home = np.tril(arr, -1).sum()
    draw = np.trace(arr)
    away = np.triu(arr, 1).sum()
    return {"Home": home, "Draw": draw, "Away": away}

def probs_ou(mat, line=2.5):
    over = 0.0
    under = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i + j > line:
                over += mat.iat[i, j]
            else:
                under += mat.iat[i, j]
    return {"Over": over, "Under": under}

def probs_btts(mat):
    yes = mat.iloc[1:, 1:].values.sum()
    return {"Yes": yes, "No": 1 - yes}

def top_scores(mat, n=10):
    rows = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rows.append({"score": score_label(i, j), "home_goals": i, "away_goals": j, "probability": mat.iat[i, j]})
    return pd.DataFrame(rows).sort_values("probability", ascending=False).head(n)

def compare_odds(model_probs, market_odds):
    rows = []
    for k, mp in model_probs.items():
        mo = safe_float(market_odds.get(k))
        ip = implied_prob(mo)
        fair = (1.0 / mp) if mp and mp > 0 else np.nan
        edge = (mp - ip) if (not np.isnan(mp) and not np.isnan(ip)) else np.nan
        rows.append({
            "market": k,
            "model_prob": mp,
            "fair_odds": fair,
            "book_odds": mo,
            "implied_prob": ip,
            "edge": edge
        })
    return pd.DataFrame(rows)

def prepare_df(df):
    df = df.copy()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    return df

tabs = st.tabs(["Données", "Calculateur", "Analyse équipe", "Cotes", "Export", "Aide"])
tab_data, tab_calc, tab_team, tab_odds, tab_export, tab_help = tabs

with tab_data:
    st.subheader("Chargement des données")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### CSV local")
        uploaded = st.file_uploader("Importer un CSV", type=["csv"])
        if uploaded:
            try:
                df_csv = load_csv(uploaded)
                st.session_state["df"] = df_csv
                st.success(f"{len(df_csv)} matchs chargés.")
            except Exception as e:
                st.error(str(e))

    with c2:
        st.markdown("### API football-data.org")
        token = st.text_input("Token API", value=DEFAULT_TOKEN, type="password")
        if token:
            if st.button("Charger la liste des compétitions"):
                try:
                    comps = fetch_competitions(token)
                    comp_df = pd.DataFrame([{
                        "code": c.get("code", ""),
                        "name": c.get("name", ""),
                        "country": c.get("area", {}).get("name", ""),
                        "type": c.get("type", ""),
                        "plan": c.get("plan", "")
                    } for c in comps])
                    st.session_state["comp_df"] = comp_df
                    st.success(f"{len(comp_df)} compétitions récupérées.")
                except Exception as e:
                    st.error(f"Erreur compétitions: {e}")

        if "comp_df" in st.session_state:
            st.dataframe(st.session_state["comp_df"], use_container_width=True)

        comp_choice = st.selectbox("Compétition", list(COMPETITIONS_FALLBACK.keys()))
        season_choice = st.number_input("Saison", min_value=2000, max_value=2030, value=2025, step=1)
        if st.button("Charger les matchs via API"):
            if not token:
                st.error("Ajoute un token.")
            else:
                try:
                    df_api = fetch_matches(token, COMPETITIONS_FALLBACK[comp_choice], int(season_choice))
                    if df_api.empty:
                        st.warning("Aucun match trouvé.")
                    else:
                        st.session_state["df"] = df_api
                        st.success(f"{len(df_api)} matchs chargés.")
                except Exception as e:
                    st.error(f"Erreur API: {e}")

    if "df" in st.session_state:
        st.write("Aperçu des données")
        st.dataframe(st.session_state["df"].head(200), use_container_width=True)
    else:
        st.info("Charge un CSV ou utilise l'API.")

with tab_calc:
    st.subheader("Calcul des probabilités")
    if "df" not in st.session_state:
        st.info("Charge d'abord des données.")
    else:
        df = prepare_df(st.session_state["df"])
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
        c1, c2, c3 = st.columns([1, 1, 1])
        home_team = c1.selectbox("Équipe domicile", teams)
        away_team = c2.selectbox("Équipe extérieur", [t for t in teams if t != home_team])
        max_goals = c3.slider("Buts max", 5, 12, 8)
        blend = st.slider("Poids stats équipe", 0.0, 1.0, 0.65, 0.05)
        ou_line = st.selectbox("Ligne Over/Under", [0.5, 1.5, 2.5, 3.5, 4.5], index=2)

        if st.button("Calculer"):
            lh, la = estimate_lambdas(df, home_team, away_team, blend=blend)
            mat = score_matrix(lh, la, max_goals=max_goals)
            p12 = probs_1x2(mat)
            pou = probs_ou(mat, line=ou_line)
            pbtts = probs_btts(mat)
            scores = top_scores(mat, n=10)

            st.session_state["last_calc"] = {
                "home_team": home_team,
                "away_team": away_team,
                "lambda_home": lh,
                "lambda_away": la,
                "mat": mat,
                "p12": p12,
                "pou": pou,
                "pbtts": pbtts,
                "scores": scores
            }

            st.write(pd.DataFrame([{
                "lambda_home": round(lh, 3),
                "lambda_away": round(la, 3),
                "home_win_%": pct(p12["Home"]),
                "draw_%": pct(p12["Draw"]),
                "away_win_%": pct(p12["Away"]),
                "over_%": pct(pou["Over"]),
                "under_%": pct(pou["Under"]),
                "btts_yes_%": pct(pbtts["Yes"]),
                "btts_no_%": pct(pbtts["No"]),
            }]))

            st.subheader("Scores probables")
            disp = scores.copy()
            disp["probability"] = (disp["probability"] * 100).round(2)
            st.dataframe(disp, use_container_width=True)

            st.subheader("Heatmap")
            long_df = pd.DataFrame([
                {"home_goals": i, "away_goals": j, "prob": mat.loc[i, j]}
                for i in mat.index for j in mat.columns
            ])
            fig = px.density_heatmap(long_df, x="away_goals", y="home_goals", z="prob", color_continuous_scale="Blues")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

with tab_team:
    st.subheader("Analyse équipe")
    if "df" not in st.session_state:
        st.info("Charge d'abord des données.")
    else:
        df = prepare_df(st.session_state["df"])
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
        team = st.selectbox("Équipe", teams)
        s = team_stats(df, team)
        st.write(pd.DataFrame([{
            "home_matches": s["home_matches"],
            "away_matches": s["away_matches"],
            "home_scored": round(s["home_scored"], 3) if not np.isnan(s["home_scored"]) else np.nan,
            "home_conceded": round(s["home_conceded"], 3) if not np.isnan(s["home_conceded"]) else np.nan,
            "away_scored": round(s["away_scored"], 3) if not np.isnan(s["away_scored"]) else np.nan,
            "away_conceded": round(s["away_conceded"], 3) if not np.isnan(s["away_conceded"]) else np.nan,
        }]))

        hg = df[df["home_team"] == team]["home_goals"].value_counts().sort_index()
        ag = df[df["away_team"] == team]["away_goals"].value_counts().sort_index()
        chart_df = pd.DataFrame({"home": hg, "away": ag}).fillna(0).reset_index().rename(columns={"index": "goals"})
        if not chart_df.empty:
            fig = px.bar(chart_df, x="goals", y=["home", "away"], barmode="group")
            st.plotly_chart(fig, use_container_width=True)

with tab_odds:
    st.subheader("Cotes")
    if "last_calc" not in st.session_state:
        st.info("Fais un calcul d'abord.")
    else:
        lc = st.session_state["last_calc"]
        c1, c2 = st.columns(2)
        with c1:
            home_odds = st.number_input("Cote Home", min_value=1.01, value=2.10, step=0.01)
            draw_odds = st.number_input("Cote Draw", min_value=1.01, value=3.30, step=0.01)
            away_odds = st.number_input("Cote Away", min_value=1.01, value=3.60, step=0.01)
            over_odds = st.number_input("Cote Over", min_value=1.01, value=1.90, step=0.01)
            under_odds = st.number_input("Cote Under", min_value=1.01, value=1.90, step=0.01)

        with c2:
            odds_file = st.file_uploader("Importer cotes CSV optionnel", type=["csv"])
            if odds_file:
                try:
                    odf = pd.read_csv(odds_file)
                    st.session_state["odf"] = odf
                    st.dataframe(odf, use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        if st.button("Comparer au modèle"):
            comp_1x2 = compare_odds(lc["p12"], {"Home": home_odds, "Draw": draw_odds, "Away": away_odds})
            comp_ou = compare_odds(lc["pou"], {"Over": over_odds, "Under": under_odds})
            st.write("1X2")
            st.dataframe(comp_1x2.assign(
                model_pct=lambda x: (x["model_prob"] * 100).round(2),
                implied_pct=lambda x: (x["implied_prob"] * 100).round(2),
                edge_pct=lambda x: (x["edge"] * 100).round(2),
                fair_odds=lambda x: x["fair_odds"].round(2),
            ), use_container_width=True)
            st.write("Over/Under")
            st.dataframe(comp_ou.assign(
                model_pct=lambda x: (x["model_prob"] * 100).round(2),
                implied_pct=lambda x: (x["implied_prob"] * 100).round(2),
                edge_pct=lambda x: (x["edge"] * 100).round(2),
                fair_odds=lambda x: x["fair_odds"].round(2),
            ), use_container_width=True)

with tab_export:
    st.subheader("Export")
    if "last_calc" not in st.session_state:
        st.info("Fais un calcul d'abord.")
    else:
        lc = st.session_state["last_calc"]
        summary = pd.DataFrame([{
            "home_team": lc["home_team"],
            "away_team": lc["away_team"],
            "lambda_home": lc["lambda_home"],
            "lambda_away": lc["lambda_away"],
            "home_win_prob": lc["p12"]["Home"],
            "draw_prob": lc["p12"]["Draw"],
            "away_win_prob": lc["p12"]["Away"],
            "over_prob": lc["pou"]["Over"],
            "under_prob": lc["pou"]["Under"],
            "btts_yes_prob": lc["pbtts"]["Yes"],
            "btts_no_prob": lc["pbtts"]["No"],
        }])
        top_scores_df = lc["scores"].copy()
        out = pd.concat([summary, top_scores_df], ignore_index=True, sort=False)
        st.download_button("Télécharger CSV", data=to_csv_bytes(out), file_name="football_probabilities_export.csv", mime="text/csv")

with tab_help:
    st.subheader("Aide")
    st.markdown("""
- Installe les dépendances avec `pip install -r requirements.txt`.
- Lance l’app en local avec `streamlit run app.py`.
- Sur Streamlit Cloud, mets ton token API dans les Secrets.
- football-data.org fournit la liste des compétitions via `/v4/competitions` et les matchs via `/v4/competitions/{id}/matches` [web:30][web:88].
- `st.file_uploader` sert à importer ton CSV dans l’interface [web:38].
- Le déploiement se fait depuis un repo GitHub relié à Streamlit Community Cloud [web:87][web:88].
""")
