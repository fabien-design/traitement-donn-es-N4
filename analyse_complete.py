import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IMAGE_FOLDER = './images/'


df_initial = pd.read_csv('./result.csv', sep=';')
print(df_initial.dtypes)

df = pd.read_csv('./result.csv', sep=';')

print("\n" + "="*80)
print("Aperçu des premières lignes:")
print(df.head(5))
print(df.info())

print(f"- Valeurs manquantes par colonne:")
print(df.isnull().sum())
print(f"\n- Doublons détectés: {df.duplicated().sum()}") # normal l'Id est unique

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)
print(f"Lignes après suppression: {len(df)}")

# Optimiser les types
df['Id'] = df['Id'].astype('int16')
df['gaming_interest_score'] = df['gaming_interest_score'].astype('float32')
df['insta_design_interest_score'] = df['insta_design_interest_score'].astype('float32')
df['football_interest_score'] = df['football_interest_score'].astype('float32')
df['recommended_product'] = df['recommended_product'].astype('string')
# Conversion correcte de campaign_success en booléen (gère les espaces)
df['campaign_success'] = df['campaign_success'].astype(str).str.strip().map({'True': True, 'False': False})
df['age'] = df['age'].astype('int8')
df['canal_recommande'] = df['canal_recommande'].astype('string')

print(f"\nMémoire utilisée APRES optimisation:")
print(df.info())

df['recommended_product'] = df['recommended_product'].str.strip().str.lower()
df['canal_recommande'] = df['canal_recommande'].str.strip().str.lower()

# Corrections des erreurs
print("\n" + "="*80)
print("Corrections des erreurs typographiques:")
nb_fornite = (df['recommended_product'] == 'fornite').sum()
df['recommended_product'] = df['recommended_product'].replace('fornite', 'fortnite')
print(f"- 'fornite' → 'fortnite': {nb_fornite} ligne(s) corrigée(s)")

nb_test = (df['recommended_product'] == 'test').sum()
df = df[df['recommended_product'] != 'test'].copy()

nb_non_defini = (df['canal_recommande'] == 'non_defini').sum()
df = df[df['canal_recommande'] != 'non_defini'].copy()
print(f"- Lignes avec canal 'non_defini' supprimées: {nb_non_defini} ligne(s)")
print(f"\nLignes après corrections: {len(df)}")

# Méthode 1: Détection classique (scores doivent être entre 0 et 100)
print("\nMéthode 1: Détection avec [0, 100]")
gaming_invalids = df[(df['gaming_interest_score'] < 0) | (df['gaming_interest_score'] > 100)]
insta_invalids = df[(df['insta_design_interest_score'] < 0) | (df['insta_design_interest_score'] > 100)]
football_invalids = df[(df['football_interest_score'] < 0) | (df['football_interest_score'] > 100)]

print(f"- gaming_interest_score hors [0,100]: {len(gaming_invalids)}")
if len(gaming_invalids) > 0:
    print("\nLignes avec gaming_interest_score anormal:")
    print(gaming_invalids[['Id', 'gaming_interest_score', 'insta_design_interest_score', 'football_interest_score', 'age', 'recommended_product']].to_string(index=False))

print(f"\n- insta_design_interest_score hors [0,100]: {len(insta_invalids)}")
if len(insta_invalids) > 0:
    print("\nLignes avec insta_design_interest_score anormal:")
    print(insta_invalids[['Id', 'gaming_interest_score', 'insta_design_interest_score', 'football_interest_score', 'age', 'recommended_product']].to_string(index=False))

print(f"\n- football_interest_score hors [0,100]: {len(football_invalids)}")
if len(football_invalids) > 0:
    print("\nLignes avec football_interest_score anormal:")
    print(football_invalids[['Id', 'gaming_interest_score', 'insta_design_interest_score', 'football_interest_score', 'age', 'recommended_product']].to_string(index=False))

# Méthode 2: Z-score (valeurs > 3 écarts-types = anomalies)
print("\n" + "="*80)
print("Méthode 2: Z-score (seuil = 3)")
score_columns = ['gaming_interest_score', 'insta_design_interest_score', 'football_interest_score']
z_score_anomalies = {}

for col in score_columns:
    # Calcul manuel du z-score (méthode du cours)
    moyenne = df[col].mean()
    ecart_type = df[col].std()
    z_scores = np.abs((df[col] - moyenne) / ecart_type)
    anomalies = z_scores > 3
    z_score_anomalies[col] = anomalies
    print(f"- {col}: {anomalies.sum()} anomalies détectées")

    if anomalies.sum() > 0:
        print(f"\nLignes avec {col} anormal (Z-score > 3):")
        anomalies_df = df[anomalies].copy()
        anomalies_df['z_score'] = z_scores[anomalies]
        print(anomalies_df[['Id', 'gaming_interest_score', 'insta_design_interest_score', 'football_interest_score', 'age', 'z_score']].to_string(index=False))
        print()

# Méthode 3: IQR (Interquartile Range)
print("\n" + "="*80)
print("Méthode 3: IQR (Interquartile Range)")
iqr_anomalies = {}

for col in score_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = (df[col] < lower_bound) | (df[col] > upper_bound)
    iqr_anomalies[col] = anomalies
    print(f"- {col}: {anomalies.sum()} anomalies (bornes: [{lower_bound:.2f}, {upper_bound:.2f}])")

    if anomalies.sum() > 0:
        print(f"\nLignes avec {col} anormal (IQR):")
        anomalies_df = df[anomalies].copy()
        anomalies_df['Q1'] = Q1
        anomalies_df['Q3'] = Q3
        anomalies_df['borne_inf'] = lower_bound
        anomalies_df['borne_sup'] = upper_bound
        print(anomalies_df[['Id', 'gaming_interest_score', 'insta_design_interest_score', 'football_interest_score', 'age', 'borne_inf', 'borne_sup']].to_string(index=False))
        print()

# Visualisations des anomalies détectées
gaming_anomalies = (df['gaming_interest_score'] < 0) | (df['gaming_interest_score'] > 100)
insta_anomalies = (df['insta_design_interest_score'] < 0) | (df['insta_design_interest_score'] > 100)
football_anomalies = (df['football_interest_score'] < 0) | (df['football_interest_score'] > 100)

# Graphique pour Méthode 1: Détection [0,100]
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Méthode 1: Détection d\'anomalies [0, 100]', fontsize=16, fontweight='bold')

scores_viz = [
    ('gaming_interest_score', 'Gaming Interest Score', gaming_anomalies),
    ('insta_design_interest_score', 'Instagram Design Interest Score', insta_anomalies),
    ('football_interest_score', 'Football Interest Score', football_anomalies)
]

for idx, (col_name, title, anomaly_mask) in enumerate(scores_viz):
    ax = axes[idx]
    x = df['age']
    y = df[col_name]
    mean_val = y.mean()

    ax.scatter(x[~anomaly_mask], y[~anomaly_mask], c='green', alpha=0.6, s=30, label='Points valides')
    if anomaly_mask.sum() > 0:
        ax.scatter(x[anomaly_mask], y[anomaly_mask], c='red', alpha=0.8, s=50, marker='x', label=f'Anomalies ({anomaly_mask.sum()})')

    ax.axhline(y=0, color='blue', linestyle='-', linewidth=2, alpha=0.7, label='Min (0)')
    ax.axhline(y=mean_val, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Moyenne ({mean_val:.2f})')
    ax.axhline(y=100, color='blue', linestyle='-', linewidth=2, alpha=0.7, label='Max (100)')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Score')
    ax.set_ylim(min(y.min() - 10, -10), max(y.max() + 10, 110))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/methode1_detection_0_100.png', dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé: {IMAGE_FOLDER}/methode1_detection_0_100.png")
plt.close()

# Graphique pour Méthode 2: Z-score
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Méthode 2: Détection d\'anomalies par Z-score (seuil = 3)', fontsize=16, fontweight='bold')

for idx, col in enumerate(score_columns):
    ax = axes[idx]
    x = df['age']
    y = df[col]
    mean_val = y.mean()
    std_val = y.std()
    anomaly_mask = z_score_anomalies[col]

    ax.scatter(x[~anomaly_mask], y[~anomaly_mask], c='green', alpha=0.6, s=30, label='Points valides')
    if anomaly_mask.sum() > 0:
        ax.scatter(x[anomaly_mask], y[anomaly_mask], c='red', alpha=0.8, s=50, marker='x', label=f'Anomalies ({anomaly_mask.sum()})')

    ax.axhline(y=mean_val, color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'Moyenne ({mean_val:.2f})')
    ax.axhline(y=mean_val + 3*std_val, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'+3σ ({mean_val + 3*std_val:.2f})')
    ax.axhline(y=mean_val - 3*std_val, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'-3σ ({mean_val - 3*std_val:.2f})')

    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/methode2_zscore.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: {IMAGE_FOLDER}/methode2_zscore.png")
plt.close()

# Graphique pour Méthode 3: IQR
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Méthode 3: Détection d\'anomalies par IQR (Interquartile Range)', fontsize=16, fontweight='bold')

for idx, col in enumerate(score_columns):
    ax = axes[idx]
    x = df['age']
    y = df[col]
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomaly_mask = iqr_anomalies[col]

    ax.scatter(x[~anomaly_mask], y[~anomaly_mask], c='green', alpha=0.6, s=30, label='Points valides')
    if anomaly_mask.sum() > 0:
        ax.scatter(x[anomaly_mask], y[anomaly_mask], c='red', alpha=0.8, s=50, marker='x', label=f'Anomalies ({anomaly_mask.sum()})')

    ax.axhline(y=Q1, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q1 ({Q1:.2f})')
    ax.axhline(y=Q3, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q3 ({Q3:.2f})')
    ax.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Borne inf ({lower_bound:.2f})')
    ax.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Borne sup ({upper_bound:.2f})')

    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/methode3_iqr.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: {IMAGE_FOLDER}/methode3_iqr.png")
plt.close()

# Supprimer les scores invalides
df_clean = df[
    (df['gaming_interest_score'] >= 0) & (df['gaming_interest_score'] <= 100) &
    (df['insta_design_interest_score'] >= 0) & (df['insta_design_interest_score'] <= 100) &
    (df['football_interest_score'] >= 0) & (df['football_interest_score'] <= 100)
].copy()

print(f"\nLignes après nettoyage des anomalies: {len(df_clean)}")
print(f"Anomalies supprimées: {len(df) - len(df_clean)}")
print(f"\nMémoire utilisée APRES nettoyage:")
print(df_clean.info())


# =============================================================================

# KPI 1: Taux de réussite global
taux_reussite_global = (df_clean['campaign_success'].sum() / len(df_clean)) * 100
print(f"\nTaux de réussite global: {taux_reussite_global:.2f}%")

# KPI 2: Taux de réussite par produit
print("\nTaux de réussite par produit:")
reussite_par_produit = df_clean.groupby('recommended_product', observed=True)['campaign_success'].agg(['sum', 'count'])
reussite_par_produit['taux'] = (reussite_par_produit['sum'] / reussite_par_produit['count']) * 100
print(reussite_par_produit.sort_values('taux', ascending=False))

# KPI 3: Taux de réussite par support (canal)
print("\nTaux de réussite par support:")
reussite_par_support = df_clean.groupby('canal_recommande', observed=True)['campaign_success'].agg(['sum', 'count'])
reussite_par_support['taux'] = (reussite_par_support['sum'] / reussite_par_support['count']) * 100
print(reussite_par_support.sort_values('taux', ascending=False))

# KPI 4: Taux de réussite par tranche d'âge
print("\nTaux de réussite par tranche d'âge:")
df_clean['tranche_age'] = pd.cut(df_clean['age'], bins=[0, 12, 18, 25, 35, 50, 100],
                                  labels=['0-12', '13-18', '19-25', '26-35', '36-50', '51+'])
reussite_par_age = df_clean.groupby('tranche_age', observed=True)['campaign_success'].agg(['sum', 'count'])
reussite_par_age['taux'] = (reussite_par_age['sum'] / reussite_par_age['count']) * 100
print(reussite_par_age)

# KPI 5: Statistiques par segment d'intérêt
print("\nStatistiques descriptives des scores:")
print(df_clean[score_columns].describe())

# Créer des segments d'intérêt basés sur le score le plus élevé
def determine_segment(row):
    scores = {
        'gaming': row['gaming_interest_score'],
        'design': row['insta_design_interest_score'],
        'football': row['football_interest_score']
    }
    return max(scores, key=scores.get)

df_clean['segment_principal'] = df_clean.apply(determine_segment, axis=1)

print("\nTaux de réussite par segment d'intérêt principal:")
reussite_par_segment = df_clean.groupby('segment_principal', observed=True)['campaign_success'].agg(['sum', 'count'])
reussite_par_segment['taux'] = (reussite_par_segment['sum'] / reussite_par_segment['count']) * 100
print(reussite_par_segment.sort_values('taux', ascending=False))


# Visu 2: KPI - Taux de réussite par produit
fig, ax = plt.subplots(figsize=(12, 6))
reussite_par_produit_sorted = reussite_par_produit.sort_values('taux', ascending=True)
bars = ax.barh(reussite_par_produit_sorted.index, reussite_par_produit_sorted['taux'], color='steelblue')
ax.set_xlabel('Taux de réussite (%)', fontsize=12)
ax.set_title('Taux de réussite par produit recommandé', fontsize=14, fontweight='bold')
ax.axvline(x=taux_reussite_global, color='red', linestyle='--', linewidth=2, label=f'Moyenne globale ({taux_reussite_global:.2f}%)')
ax.legend()
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
            ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/kpi_produit.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: {IMAGE_FOLDER}/kpi_produit.png")

# Visualisation 3: KPI - Taux de réussite par support
fig, ax = plt.subplots(figsize=(10, 6))
reussite_par_support_sorted = reussite_par_support.sort_values('taux', ascending=True)
bars = ax.barh(reussite_par_support_sorted.index, reussite_par_support_sorted['taux'], color='coral')
ax.set_xlabel('Taux de réussite (%)', fontsize=12)
ax.set_title('Taux de réussite par support/canal', fontsize=14, fontweight='bold')
ax.axvline(x=taux_reussite_global, color='red', linestyle='--', linewidth=2, label=f'Moyenne globale ({taux_reussite_global:.2f}%)')
ax.legend()
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
            ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/kpi_support.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: {IMAGE_FOLDER}/kpi_support.png")

# Visualisation 4: KPI - Taux de réussite par tranche d'âge
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(reussite_par_age.index.astype(str), reussite_par_age['taux'], color='mediumseagreen')
ax.set_xlabel('Tranche d\'âge', fontsize=12)
ax.set_ylabel('Taux de réussite (%)', fontsize=12)
ax.set_title('Taux de réussite par tranche d\'âge', fontsize=14, fontweight='bold')
ax.axhline(y=taux_reussite_global, color='red', linestyle='--', linewidth=2, label=f'Moyenne globale ({taux_reussite_global:.2f}%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/kpi_age.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: {IMAGE_FOLDER}/kpi_age.png")

# Visualisation 5: KPI - Taux de réussite par segment
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(reussite_par_segment.index, reussite_par_segment['taux'], color='mediumpurple')
ax.set_xlabel('Segment d\'intérêt principal', fontsize=12)
ax.set_ylabel('Taux de réussite (%)', fontsize=12)
ax.set_title('Taux de réussite par segment d\'intérêt', fontsize=14, fontweight='bold')
ax.axhline(y=taux_reussite_global, color='red', linestyle='--', linewidth=2, label=f'Moyenne globale ({taux_reussite_global:.2f}%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/kpi_segment.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: {IMAGE_FOLDER}/kpi_segment.png")

# =============================================================================

# Convertir campaign_success en numérique pour la corrélation
df_clean['campaign_success_num'] = df_clean['campaign_success'].astype(int)

correlation_cols = ['gaming_interest_score', 'insta_design_interest_score',
                     'football_interest_score', 'age', 'campaign_success_num']
corr_matrix = df_clean[correlation_cols].corr()

print("\nMatrice de corrélation:")
print(corr_matrix)

# Visualisation 6: Matrice de corrélation avec matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10)

# Ajouter une barre de couleur
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Corrélation', rotation=270, labelpad=15)
ax.set_title('Matrice de corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMAGE_FOLDER}/correlation.png', dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé: {IMAGE_FOLDER}/correlation.png")

# Identifier les corrélations les plus fortes avec campaign_success
print("\nCorrélations avec le succès de la campagne:")
success_corr = corr_matrix['campaign_success_num'].sort_values(ascending=False)
print(success_corr)


# Pour le storytelling

print("\nMeilleure combinaison Produit + Canal:")
best_combos = df_clean.groupby(['recommended_product', 'canal_recommande'], observed=True)['campaign_success'].agg(['sum', 'count'])
best_combos['taux'] = (best_combos['sum'] / best_combos['count']) * 100
best_combos = best_combos[best_combos['count'] >= 7]
print(best_combos.sort_values('taux', ascending=False).head(10))
    
# Analyser par âge et segment
print("\nAnalyse par âge et segment principal:")
age_segment_analysis = df_clean.groupby(['tranche_age', 'segment_principal'], observed=True).agg({
    'campaign_success': ['sum', 'count'],
    'gaming_interest_score': 'mean',
    'insta_design_interest_score': 'mean',
    'football_interest_score': 'mean'
})
age_segment_analysis['taux_reussite'] = (age_segment_analysis[('campaign_success', 'sum')] /
                                          age_segment_analysis[('campaign_success', 'count')] * 100)
print(age_segment_analysis.sort_values('taux_reussite', ascending=False).head(15))

# Identifier le groupe le plus vulnérable

vulnerabilite_par_groupe = df_clean.groupby(['tranche_age', 'segment_principal'], observed=True).agg({
    'campaign_success': ['sum', 'count']
})
vulnerabilite_par_groupe['taux'] = (vulnerabilite_par_groupe[('campaign_success', 'sum')] /
                                     vulnerabilite_par_groupe[('campaign_success', 'count')] * 100)
vulnerabilite_par_groupe = vulnerabilite_par_groupe[vulnerabilite_par_groupe[('campaign_success', 'count')] >= 20]
groupe_plus_vulnerable = vulnerabilite_par_groupe.sort_values('taux', ascending=False).head(1)

print("\nGroupe le plus vulnérable:")
print(groupe_plus_vulnerable)

if len(groupe_plus_vulnerable) > 0:
    tranche_vuln = groupe_plus_vulnerable.index[0][0]
    segment_vuln = groupe_plus_vulnerable.index[0][1]

    print(f"\n→ Tranche d'âge: {tranche_vuln}")
    print(f"→ Segment d'intérêt: {segment_vuln}")

    groupe_data = df_clean[(df_clean['tranche_age'] == tranche_vuln) &
                           (df_clean['segment_principal'] == segment_vuln)]

    print(f"\nNombre de personnes dans ce groupe: {len(groupe_data)}")
    print(f"Taux de réussite: {(groupe_data['campaign_success'].sum() / len(groupe_data) * 100):.2f}%")

    print("\nProduits les plus efficaces pour ce groupe:")
    produits_efficaces = groupe_data.groupby('recommended_product', observed=True)['campaign_success'].agg(['sum', 'count'])
    produits_efficaces['taux'] = (produits_efficaces['sum'] / produits_efficaces['count']) * 100
    print(produits_efficaces[produits_efficaces['count'] >= 3].sort_values('taux', ascending=False))

    print("\nCanaux les plus efficaces pour ce groupe:")
    canaux_efficaces = groupe_data.groupby('canal_recommande', observed=True)['campaign_success'].agg(['sum', 'count'])
    canaux_efficaces['taux'] = (canaux_efficaces['sum'] / canaux_efficaces['count']) * 100
    print(canaux_efficaces[canaux_efficaces['count'] >= 3].sort_values('taux', ascending=False))


df_clean.to_csv('result_clean.csv', sep=';', index=False)
print("Saved here : result_clean.csv")
