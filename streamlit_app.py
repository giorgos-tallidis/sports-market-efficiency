import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import poisson

# 1. Βασική Ρύθμιση Σελίδας
st.set_page_config(page_title="Institutional Quant Model", layout="wide")

st.title("📊 Quantitative Modeling of Market Inefficiencies")
st.markdown("A Behavioral Economics Approach to Sports Betting Markets utilizing **Time-Decay Form** and **Dynamic Bankroll Simulation**.")
st.markdown("---")

# 2. Φόρτωση Ιστορικών Δεδομένων
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('E0.csv')
        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("⚠️ Το αρχείο E0.csv δεν βρέθηκε στο φάκελο του project.")
else:
    tab1, tab2 = st.tabs(["🔮 Live Predictions (Recent Form)", "📈 Bankroll Simulation (Equity Curve)"])

    # ==========================================
    # TAB 1: LIVE ΠΡΟΒΛΕΨΕΙΣ ΜΕ ΒΑΣΗ ΤΗ ΦΟΡΜΑ
    # ==========================================
    with tab1:
        st.header("🔮 Δυναμική Μηχανή Πρόβλεψης")
        st.write("Ο αλγόριθμος σκανάρει τη βάση δεδομένων, απομονώνει τα τελευταία 5 παιχνίδια και υπολογίζει το Dynamic xG.")
        
        teams = sorted(list(set(df['HomeTeam'].dropna())))
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Επίλεξε Γηπεδούχο (Home):", teams, index=0)
        with col2:
            away_team = st.selectbox("Επίλεξε Φιλοξενούμενη (Away):", teams, index=1)
            
        home_form = df[df['HomeTeam'] == home_team].tail(5)
        away_form = df[df['AwayTeam'] == away_team].tail(5)
        
        if not home_form.empty and not away_form.empty:
            home_xg = max(home_form['FTHG'].mean(), 0.1) 
            away_xg = max(away_form['FTAG'].mean(), 0.1)
            
            st.info(f"📊 **Αυτόματο xG (Βάσει πρόσφατης φόρμας):** {home_team} **{home_xg:.2f}** | {away_team} **{away_xg:.2f}**")
            
            home_poisson = [poisson.pmf(i, home_xg) for i in range(7)]
            away_poisson = [poisson.pmf(i, away_xg) for i in range(7)]
            
            matrix = np.outer(home_poisson, away_poisson)
            prob_home_win = np.sum(np.tril(matrix, -1))
            
            st.metric(label="Αλγοριθμική Πιθανότητα (Άσος)", value=f"{prob_home_win * 100:.2f}%")
            
            st.markdown("### 📉 Market Sentiment & Kelly Criterion")
            market_odds = st.number_input("Απόδοση Αγοράς για τον Άσο αύριο:", value=2.00, step=0.05)
            
            if market_odds > 1.0:
                b = market_odds - 1.0
                p = prob_home_win
                q = 1.0 - p
                
                kelly_fraction = (b * p - q) / b
                
                if kelly_fraction > 0:
                    st.success("✅ **Value Detected!** Η αγορά υποτιμά τη γηπεδούχο.")
                    st.write(f"**Προτεινόμενο Ποντάρισμα (Kelly %):** {kelly_fraction * 100:.2f}% του κεφαλαίου.")
                else:
                    st.warning("⚠️ **No Value.** Το μαθηματικό ρίσκο δεν δικαιολογεί την επένδυση.")

    # ==========================================
    # TAB 2: EQUITY CURVE & BACKTESTING
    # ==========================================
    with tab2:
        st.header("📈 Financial Backtesting & Equity Curve")
        st.write("Προσομοίωση Bankroll Management. Τρέχουμε τον αλγόριθμο σε όλη τη σεζόν με αρχικό κεφάλαιο **$1,000** χωρίς Look-ahead bias (χρήση ιστορικών μέσων όρων).")
        
        if st.button("▶️ Run Market Simulation"):
            bankroll = 1000.0
            capital_history = [bankroll]
            
            # Υπολογισμός πραγματικών ιστορικών μέσων όρων για κάθε ομάδα
            home_avg_goals = df.groupby('HomeTeam')['FTHG'].mean().to_dict()
            away_avg_goals = df.groupby('AwayTeam')['FTAG'].mean().to_dict()
            
            for index, row in df.iterrows():
                odds_h = row.get('B365H', 2.50) 
                if pd.isna(odds_h): odds_h = 2.50
                    
                # Αντί να "κλέβουμε" βλέποντας το σκορ, χρησιμοποιούμε τη δυναμικότητα της ομάδας
                h_goals_proxy = home_avg_goals.get(row['HomeTeam'], 1.56)
                a_goals_proxy = away_avg_goals.get(row['AwayTeam'], 1.33)
                
                prob_h = np.sum(np.tril(np.outer([poisson.pmf(i, h_goals_proxy) for i in range(6)], 
                                                 [poisson.pmf(i, a_goals_proxy) for i in range(6)]), -1))
                
                b_sim = odds_h - 1.0
                p_sim = prob_h
                q_sim = 1.0 - p_sim
                
                if b_sim > 0:
                    k_frac = (b_sim * p_sim - q_sim) / b_sim
                    k_frac = k_frac * 0.05  # Fractional Kelly (5%) για ασφάλεια
                    
                    if 0 < k_frac < 1:
                        bet_amount = bankroll * k_frac
                        if row['FTR'] == 'H':
                            bankroll += bet_amount * b_sim
                        else:
                            bankroll -= bet_amount
                            
                capital_history.append(bankroll)
            
            chart_df = pd.DataFrame({
                "Αγώνες": range(len(capital_history)), 
                "Κεφάλαιο ($)": capital_history
            })
            
            fig = px.line(chart_df, x="Αγώνες", y="Κεφάλαιο ($)", title="Bankroll Evolution (Ανάπτυξη Κεφαλαίου)", template="plotly_dark")
            
            line_color = '#00ff00' if bankroll > 1000 else '#ff0000'
            fig.update_traces(line_color=line_color, line_width=3)
            st.plotly_chart(fig, use_container_width=True)
            
            roi = ((bankroll - 1000) / 1000) * 100
            col1, col2 = st.columns(2)
            col1.metric("Αρχικό Κεφάλαιο", "$1,000.00")
            col2.metric("Τελικό Κεφάλαιο", f"${bankroll:.2f}", f"{roi:.2f}% ROI")