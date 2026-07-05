import streamlit as st
import numpy as np
from scipy.stats import poisson

# Ρύθμιση της σελίδας
st.set_page_config(page_title="Quant Modeling", layout="wide")

# Τίτλος και Εισαγωγή
st.title("📊 Quantitative Modeling of Market Inefficiencies")
st.subheader("A Behavioral Economics Approach to Sports Betting Markets")
st.write("Αυτό το μοντέλο συγκρίνει τις μαθηματικές πιθανότητες (Poisson) με τις αποδόσεις της αγοράς για να εντοπίσει Emotional Overreactions και αναποτελεσματικότητα στην αγορά.")

st.markdown("---")

# Δημιουργία δύο στηλών για το Interface
col1, col2 = st.columns(2)

with col1:
    st.header("1. Αλγοριθμικές Πιθανότητες (Poisson Model)")
    st.write("Βάλε τα Expected Goals (xG) των ομάδων όπως τα υπολόγισε το μοντέλο σου:")
    
    # Inputs για xG
    home_xg = st.number_input("Home Team xG", value=1.50, step=0.10, format="%.2f")
    away_xg = st.number_input("Away Team xG", value=1.20, step=0.10, format="%.2f")
    
    # Υπολογισμός Πιθανοτήτων Poisson
    max_goals = 6
    home_poisson = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_poisson = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    matrix = np.outer(home_poisson, away_poisson)
    prob_home_win = np.sum(np.tril(matrix, -1))
    prob_draw = np.sum(np.diag(matrix))
    prob_away_win = np.sum(np.triu(matrix, 1))
    
    # Εμφάνιση Αποτελεσμάτων Μοντέλου
    st.metric(label="Home Win Probability", value=f"{prob_home_win * 100:.2f}%")
    st.write(f"Draw Probability: **{prob_draw * 100:.2f}%**")
    st.write(f"Away Win Probability: **{prob_away_win * 100:.2f}%**")

with col2:
    st.header("2. Αποδόσεις Αγοράς (Market Sentiment)")
    st.write("Βάλε την απόδοση (Odds) που δίνει η αγορά για τον Άσο:")
    
    # Input για τις αποδόσεις
    market_odds = st.number_input("Market Odds (Home Win)", value=2.00, step=0.05, format="%.2f")
    
    # Υπολογισμός Implied Probability της αγοράς
    market_prob = (1 / market_odds) * 100 if market_odds > 0 else 0.0
    st.info(f"Πιθανότητα που πιστεύει η Αγορά: {market_prob:.2f}%")
    
    st.markdown("---")
    
    # 📈 ΝΕΟ ΚΟΜΜΑΤΙ: RISK MANAGEMENT (KELLY CRITERION)
    st.header("📈 Διαχείριση Ρίσκου (Kelly Criterion)")
    st.write("Οικονομική βελτιστοποίηση κεφαλαίου με βάση το μαθηματικό σου πλεονέκτημά:")
    
    if market_odds > 1.0:
        # b = καθαρή απόδοση (odds - 1)
        b = market_odds - 1.0
        p = prob_home_win
        q = 1.0 - p
        
        # Τύπος Kelly
        kelly_fraction = (b * p - q) / b
        
        if kelly_fraction > 0:
            st.success(f"✅ **Value Detected!** Ο αλγόριθμος εντόπισε αναποτελεσματικότητα στην αγορά.")
            st.metric(label="Προτεινόμενο Ποσοστό Επένδυσης (Kelly %)", value=f"{kelly_fraction * 100:.2f}%")
            st.caption("Αυτό το ποσοστό μεγιστοποιεί τον μακροπρόθεσμο ρυθμό ανάπτυξης του κεφαλαίου σου (Bankroll) μειώνοντας το ρίσκο της διακύμανσης (Variance).")
        else:
            st.warning("⚠️ **No Value Detected.** Η απόδοση της αγοράς είναι χαμηλότερη από τη μαθηματική πιθανότητα. Μην επενδύσεις.")
    else:
        st.error("Οι αποδόσεις της αγοράς πρέπει να είναι μεγαλύτερες από 1.00.")
