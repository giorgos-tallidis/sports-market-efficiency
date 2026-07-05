import math
import os
from pathlib import Path

import pandas as pd

try:
    from scipy.stats import poisson
    poisson_available = True
except ImportError:
    poisson_available = False


def poisson_pmf(k, lam):
    if poisson_available:
        return poisson.pmf(k, lam)
    return math.exp(-lam) * lam**k / math.factorial(k)


def poisson_match_probabilities(
    home_team,
    away_team,
    attack_strength_home,
    defense_strength_home,
    attack_strength_away,
    defense_strength_away,
    league_avg_home_goals,
    league_avg_away_goals,
    max_goals=6
):
    lambda_home = league_avg_home_goals * attack_strength_home * defense_strength_away
    lambda_away = league_avg_away_goals * attack_strength_away * defense_strength_home

    home_goal_probs = [poisson_pmf(i, lambda_home) for i in range(max_goals + 1)]
    away_goal_probs = [poisson_pmf(i, lambda_away) for i in range(max_goals + 1)]

    score_matrix = pd.DataFrame(
        index=range(max_goals + 1),
        columns=range(max_goals + 1),
        dtype=float
    )

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            score_matrix.loc[home_goals, away_goals] = (
                home_goal_probs[home_goals] * away_goal_probs[away_goals]
            )

    home_win_prob = score_matrix.where(score_matrix.index.values[:, None] > score_matrix.columns.values).sum().sum()
    draw_prob = score_matrix.where(score_matrix.index.values[:, None] == score_matrix.columns.values).sum().sum()
    away_win_prob = score_matrix.where(score_matrix.index.values[:, None] < score_matrix.columns.values).sum().sum()

    outcome_probs = {
        'Home Win': home_win_prob,
        'Draw': draw_prob,
        'Away Win': away_win_prob,
        'Modeled Total Probability': score_matrix.to_numpy().sum()
    }

    expected_goals = {
        'home_team': home_team,
        'away_team': away_team,
        'lambda_home': lambda_home,
        'lambda_away': lambda_away
    }

    return score_matrix, outcome_probs, expected_goals


base_dir = Path(__file__).resolve().parent
csv_files = [
    base_dir / 'E0.csv',
    base_dir / 'E0 (1).csv',
    base_dir / 'E0 (2).csv',
    base_dir / 'E0 (3).csv',
    base_dir / 'E0 (4).csv',
    base_dir / 'E0 (5).csv',
]

όλες_οι_σεζόν = []

print('🔄 Ξεκινάει η φόρτωση και ο καθαρισμός των δεδομένων...')
print('-' * 50)

for όνομα_αρχείου in csv_files:
    if όνομα_αρχείου.exists():
        df = pd.read_csv(όνομα_αρχείου, on_bad_lines='skip')

        στήλες_που_θέλουμε = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        df = df[[col for col in στήλες_που_θέλουμε if col in df.columns]]

        if {'HomeTeam', 'AwayTeam'} <= set(df.columns):
            df = df.dropna(subset=['HomeTeam', 'AwayTeam'])

        όλες_οι_σεζόν.append(df)
        print(f"✔️ Το αρχείο {όνομα_αρχείου.name} φορτώθηκε! Αγώνες σεζόν: {len(df)}")
    else:
        print(f"❌ Προσοχή: Το αρχείο {όνομα_αρχείου.name} δεν βρέθηκε στον φάκελο!")

if όλες_οι_σεζόν:
    συνολικά_δεδομένα = pd.concat(όλες_οι_σεζόν, ignore_index=True)
    print('-' * 50)
    print('📊 Η ένωση ολοκληρώθηκε επιτυχώς!')
    print(f"⚽ Συνολικοί αγώνες για ανάλυση: {len(συνολικά_δεδομένα)}")

    if 'FTHG' in συνολικά_δεδομένα.columns and 'FTAG' in συνολικά_δεδομένα.columns:
        συνολικά_γκολ_γήπεδο = συνολικά_δεδομένα['FTHG'].sum()
        συνολικά_γκολ_εκτός = συνολικά_δεδομένα['FTAG'].sum()
        γενικό_σύνολο_γκολ = συνολικά_γκολ_γήπεδο + συνολικά_γκολ_εκτός
        μέσος_όρος_γκολ = γενικό_σύνολο_γκολ / len(συνολικά_δεδομένα)
        avg_home_goals_league = συνολικά_δεδομένα['FTHG'].mean()
        avg_away_goals_league = συνολικά_δεδομένα['FTAG'].mean()
    else:
        συνολικά_γκολ_γήπεδο = 0
        συνολικά_γκολ_εκτός = 0
        γενικό_σύνολο_γκολ = 0
        μέσος_όρος_γκολ = 0
        avg_home_goals_league = 0
        avg_away_goals_league = 0

    if 'FTR' in συνολικά_δεδομένα.columns:
        αποτελέσματα = συνολικά_δεδομένα['FTR'].value_counts()
    else:
        αποτελέσματα = pd.Series(dtype='int64')

    print('\n🔥 ΠΡΩΤΑ ΣΤΑΤΙΣΤΙΚΑ ΣΤΟΙΧΕΙΑ ΤΟΥ ΑΛΓΟΡΙΘΜΟΥ:')
    print(f"🏠 Συνολικά γκολ γηπεδούχων: {int(συνολικά_γκολ_γήπεδο)}")
    print(f"🚀 Συνολικά γκολ φιλοξενούμενων: {int(συνολικά_γκολ_εκτός)}")
    print(f"📈 Μέσος όρος γκολ ανά αγώνα: {μέσος_όρος_γκολ:.2f}")
    print(f"📊 Μέσος όρος γκολ Εντός Έδρας: {avg_home_goals_league:.2f}")
    print(f"📊 Μέσος όρος γκολ Εκτός Έδρας: {avg_away_goals_league:.2f}")
    print(f"🏆 Νίκες Έδρας (H): {αποτελέσματα.get('H', 0)} | Ισοπαλίες (D): {αποτελέσματα.get('D', 0)} | Διπλά (A): {αποτελέσματα.get('A', 0)}")

    if {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'} <= set(συνολικά_δεδομένα.columns):
        home_stats = συνολικά_δεδομένα.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).reset_index()
        home_stats.columns = ['Team', 'HomeGoalsScored', 'HomeGoalsConceded']
        home_stats['HomeAttackStrength'] = home_stats['HomeGoalsScored'] / avg_home_goals_league
        home_stats['HomeDefenseStrength'] = home_stats['HomeGoalsConceded'] / avg_away_goals_league

        away_stats = συνολικά_δεδομένα.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).reset_index()
        away_stats.columns = ['Team', 'AwayGoalsScored', 'AwayGoalsConceded']
        away_stats['AwayAttackStrength'] = away_stats['AwayGoalsScored'] / avg_away_goals_league
        away_stats['AwayDefenseStrength'] = away_stats['AwayGoalsConceded'] / avg_home_goals_league

        top_home_attackers = home_stats.sort_values(by='HomeAttackStrength', ascending=False).head(5)
        top_away_attackers = away_stats.sort_values(by='AwayAttackStrength', ascending=False).head(5)

        print('\n🔥 TOP 5 Ομάδες - Δύναμη Επίθεσης (Εντός Έδρας):')
        for _, row in top_home_attackers.iterrows():
            print(f"⚽ {row['Team']}: {row['HomeAttackStrength']:.2f} (Γκολ: {row['HomeGoalsScored']:.2f})")

        print('\n🔥 TOP 5 Ομάδες - Δύναμη Επίθεσης (Εκτός Έδρας):')
        for _, row in top_away_attackers.iterrows():
            print(f"⚽ {row['Team']}: {row['AwayAttackStrength']:.2f} (Γκολ: {row['AwayGoalsScored']:.2f})")

        sample_home = top_home_attackers.iloc[0]
        sample_away = top_away_attackers.iloc[0]

        score_matrix, outcome_probs, expected_goals = poisson_match_probabilities(
            sample_home['Team'],
            sample_away['Team'],
            sample_home['HomeAttackStrength'],
            sample_home['HomeDefenseStrength'],
            sample_away['AwayAttackStrength'],
            sample_away['AwayDefenseStrength'],
            avg_home_goals_league,
            avg_away_goals_league,
            max_goals=6
        )

        print(f"\n🎯 Παράδειγμα πρόβλεψης για: {sample_home['Team']} vs {sample_away['Team']}")
        print(f"   Αναμενόμενα γκολ: {expected_goals['lambda_home']:.2f} - {expected_goals['lambda_away']:.2f}")
        print(f"   Πιθανότητες 1X2: Άσος {outcome_probs['Home Win']:.2%}, Χ {outcome_probs['Draw']:.2%}, Διπλό {outcome_probs['Away Win']:.2%}")

        best_scores = score_matrix.stack().sort_values(ascending=False).head(5)
        print('\n   Κορυφαία προβλεπόμενα σκορ:')
        for (home_goals, away_goals), prob in best_scores.items():
            print(f"   {home_goals}-{away_goals}: {prob:.2%}")
    else:
        print('\n⚠️ Δεν υπάρχουν όλα τα απαιτούμενα πεδία για τον υπολογισμό των δυνάμεων ομάδων.')

    print('\n👀 Μια γρήγορη ματιά στα δεδομένα σου (Πρώτες 5 γραμμές):')
    print(συνολικά_δεδομένα.head())
else:
    print('⚠️ Δεν βρέθηκε κανένα αρχείο. Σιγουρέψου ότι τα αρχεία CSV είναι στον ίδιο φάκελο με το model.py!')
