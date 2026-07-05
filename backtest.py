import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

def predict_match(home_team, away_team, profile_df, avg_h, avg_a, max_goals=6):
    home_row = profile_df[profile_df['Team'] == home_team]
    away_row = profile_df[profile_df['Team'] == away_team]

    h_attack = float(home_row['HomeAttack'].values[0]) if not home_row.empty else 1.0
    h_defense = float(home_row['HomeDefense'].values[0]) if not home_row.empty else 1.0
    a_attack = float(away_row['AwayAttack'].values[0]) if not away_row.empty else 1.0
    a_defense = float(away_row['AwayDefense'].values[0]) if not away_row.empty else 1.0

    lam = h_attack * a_defense * avg_h
    mu = a_attack * h_defense * avg_a

    home_poisson = [poisson.pmf(i, lam) for i in range(max_goals + 1)]
    away_poisson = [poisson.pmf(i, mu) for i in range(max_goals + 1)]

    matrix = np.outer(home_poisson, away_poisson)
    prob_home_win = np.sum(np.tril(matrix, -1))
    prob_away_win = np.sum(np.triu(matrix, 1))
    prob_draw = np.sum(np.diag(matrix))

    probabilities = {'H': prob_home_win, 'D': prob_draw, 'A': prob_away_win}
    predicted_outcome = max(probabilities, key=probabilities.get)

    return predicted_outcome, probabilities, lam, mu

def main():
    base_dir = Path(__file__).resolve().parent
    ιστορικά_αρχεία = [
        base_dir / 'E0 (1).csv',
        base_dir / 'E0 (2).csv',
        base_dir / 'E0 (3).csv',
        base_dir / 'E0 (4).csv',
        base_dir / 'E0 (5).csv',
    ]
    αρχείο_backtest = base_dir / 'E0.csv'

    print('🔄 Φάση 1: Φόρτωση και Ένωση Ιστορικών Δεδομένων για Εκπαίδευση...')

    όλες_οι_ιστορικές_σεζόν = []
    for όνομα_αρχείου in ιστορικά_αρχεία:
        if όνομα_αρχείου.exists():
            df = pd.read_csv(όνομα_αρχείου, on_bad_lines='skip')
            στήλες = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            df = df[[col for col in στήλες if col in df.columns]].dropna()
            όλες_οι_ιστορικές_σεζόν.append(df)
            print(f'✔️ Φορτώθηκε: {όνομα_αρχείου.name} ({len(df)} αγώνες)')
        else:
            print(f'⚠️ Δεν βρέθηκε: {όνομα_αρχείου.name}')

    if not όλες_οι_ιστορικές_σεζόν:
        print('❌ Σφάλμα: Δεν βρέθηκαν τα ιστορικά αρχεία!')
        return

    train_df = pd.concat(όλες_οι_ιστορικές_σεζόν, ignore_index=True)
    print(f'✔️ Φορτώθηκαν {len(train_df)} ιστορικοί αγώνες.')

    avg_home_goals = train_df['FTHG'].mean()
    avg_away_goals = train_df['FTAG'].mean()

    home_stats = train_df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).reset_index()
    home_stats.columns = ['Team', 'HomeGoalsScored', 'HomeGoalsConceded']
    home_stats['HomeAttack'] = home_stats['HomeGoalsScored'] / avg_home_goals
    home_stats['HomeDefense'] = home_stats['HomeGoalsConceded'] / avg_away_goals

    away_stats = train_df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).reset_index()
    away_stats.columns = ['Team', 'AwayGoalsScored', 'AwayGoalsConceded']
    away_stats['AwayAttack'] = away_stats['AwayGoalsScored'] / avg_away_goals
    away_stats['AwayDefense'] = away_stats['AwayGoalsConceded'] / avg_home_goals

    team_profile = pd.merge(
        home_stats[['Team', 'HomeAttack', 'HomeDefense']],
        away_stats[['Team', 'AwayAttack', 'AwayDefense']],
        on='Team', how='outer'
    ).fillna(1.0)

    print('✔️ Υπολογίστηκε η Δύναμη Επίθεσης και Άμυνας για όλες τις ομάδες.')
    print('-' * 60)

    print('🔄 Φάση 2: Έναρξη Backtesting & Quantitative Capital Simulation στο αρχείο E0.csv...')
    if not αρχείο_backtest.exists():
        print(f'❌ Σφάλμα: Το αρχείο {αρχείο_backtest.name} δεν βρέθηκε για το Backtest!')
        return

    test_df = pd.read_csv(αρχείο_backtest, on_bad_lines='skip')
    
    # Φιλτράρουμε να υπάρχουν και οι αποδόσεις της Bet365 (B365H) για σωστό οικονομικό backtest
    χρειάζομαι_στήλες = ['HomeTeam', 'AwayTeam', 'FTR', 'B365H']
    test_df = test_df[[col for col in χρειάζομαι_στήλες if col in test_df.columns]].dropna()

    # Οικονομικές Παράμετροι (Quant Strategy)
    αρχικό_κεφάλαιο = 1000.0
    κεφάλαιο = αρχικό_κεφάλαιο
    συνολικά_στοιχήματα = 0
    κερδισμένα_στοιχήματα = 0
    σωστές_προβλέψεις = 0
    συνολικά_ματς = len(test_df)

    for _, row in test_df.iterrows():
        πρόβλεψη, πιθανότητες, _, _ = predict_match(
            row['HomeTeam'], row['AwayTeam'], team_profile, avg_home_goals, avg_away_goals
        )
        
        πραγματικό_αποτέλεσμα = row['FTR']
        if πρόβλεψη == πραγματικό_αποτέλεσμα:
            σωστές_προβλέψεις += 1

        # --- ΥΠΟΛΟΓΙΣΜΟΣ KELLY CRITERION ΓΙΑ ΤΟΝ ΑΣΟ (HOME WIN) ---
        odds_h = float(row['B365H'])
        prob_h = πιθανότητες['H']
        
        b = odds_h - 1.0  # Καθαρή απόδοση
        p = prob_h
        q = 1.0 - p
        
        if b > 0:
            kelly_f = (b * p - q) / b
            # Χρησιμοποιούμε Fractional Kelly (10% του Kelly) για ρεαλιστική διαχείριση κινδύνου (Risk Management)
            fractional_kelly = kelly_f * 0.10 
            
            # Αν εντοπιστεί μαθηματικό Value (edge) και το ποντάρισμα είναι θετικό
            if fractional_kelly > 0:
                συνολικά_στοιχήματα += 1
                ποσό_στοιχήματος = κεφάλαιο * fractional_kelly
                
                if πραγματικό_αποτέλεσμα == 'H':
                    κεφάλαιο += ποσό_στοιχήματος * b
                    κερδισμένα_στοιχήματα += 1
                else:
                    κεφάλαιο -= ποσό_στοιχήματος

    ποσοστό_επιτυχίας = (σωστές_προβλέψεις / συνολικά_ματς) * 100 if συνολικά_ματς else 0.0
    win_rate_bets = (κερδισμένα_στοιχήματα / συνολικά_στοιχήματα) * 100 if συνολικά_στοιχήματα else 0.0
    roi = ((κεφάλαιο - αρχικό_κεφάλαιο) / αρχικό_κεφάλαιο) * 100

    print('-' * 60)
    print('🏆 ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ QUANT BACKTESTING:')
    print(f'📅 Συνολικά Παιχνίδια στη Σεζόν: {συνολικά_ματς}')
    print(f'🎯 Σωστές Προβλέψεις Αλγορίθμου: {σωστές_προβλέψεις} (Ακρίβεια: {ποσοστό_επιτυχίας:.2f}%)')
    print('-' * 60)
    print('💰 ΑΠΟΤΕΛΕΣΜΑΤΑ ΔΙΑΧΕΙΡΙΣΗΣ ΚΕΦΑΛΑΙΟΥ (KELLY CRITERION):')
    print(f'💵 Αρχικό Κεφάλαιο: ${αρχικό_κεφάλαιο:.2f}')
    print(f'📊 Παιχνίδια που εντοπίστηκε Value: {συνολικά_στοιχήματα}')
    print(f'✅ Κερδισμένα Στοιχήματα: {κερδισμένα_στοιχήματα} (Win Rate Στοιχημάτων: {win_rate_bets:.2f}%)')
    print(f'🏁 Τελικό Κεφάλαιο: ${κεφάλαιο:.2f}')
    print(f'📈 Απόδοση Επένδυσης (Return on Investment - ROI): {roi:.2f}%')
    print('-' * 60)

if __name__ == '__main__':
    main()