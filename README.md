# Clustering unter Rauscheinfluss: exakte Versuchsdokumentation

Dieses Repository untersucht, wie empfindlich verschiedene Clustering-Verfahren auf gezielt eingebrachtes Rauschen reagieren und wie stark eine einfache, kontrollierte Vorverarbeitung die Ergebnisse verbessert.

Die zentrale Implementierung liegt im Notebook [clustering-evaluation.ipynb](/Users/max/workspace/clust-methods-noise-eval/notebook/clustering-evaluation.ipynb). Diese README beschreibt den tatsächlich implementierten Ablauf so, dass die Entstehung der Ergebnisse vollständig nachvollzogen werden kann.

## Ziel des Experiments

Untersucht werden drei Fragen:

- Wie verändern sich Clustering-Qualität und Clusterstabilität unter verschiedenen Rauschformen?
- Wie unterscheiden sich klassische und robuste Verfahren unter identischen Bedingungen?
- Wie stark verbessert sich das Ergebnis, wenn bekannte Störanteile vor dem Clustering gezielt entfernt werden?

Verglichen werden vier Verfahren:

- `k-Means`
- `Hierarchisches agglomeratives Clustering`
- `DBSCAN`
- `RTKM`

## Implementierungsort

Wichtige Dateien:

- Hauptnotebook: [notebook/clustering-evaluation.ipynb](/Users/max/workspace/clust-methods-noise-eval/notebook/clustering-evaluation.ipynb)
- RTKM-Implementierung: [libs/RTKM.py](/Users/max/workspace/clust-methods-noise-eval/libs/RTKM.py)
- Laplacian-Score-Funktion: [libs/laplacian_score.py](/Users/max/workspace/clust-methods-noise-eval/libs/laplacian_score.py)
- Älteres Referenznotebook für Hilfsfunktionen: [notebook/eval-clust-methods.ipynb](/Users/max/workspace/clust-methods-noise-eval/notebook/eval-clust-methods.ipynb)

## Überblick über die Versuchsmatrix

Es gibt insgesamt 19 Versuchsbedingungen:

- 1 Basisszenario ohne Rauschen
- 3 Ausreißer-Szenarien ohne Vorverarbeitung
- 3 Ausreißer-Szenarien mit Vorverarbeitung
- 3 Zwischencluster-Ausreißer-Szenarien ohne Vorverarbeitung
- 3 Zwischencluster-Ausreißer-Szenarien mit Vorverarbeitung
- 3 Szenarien mit irrelevanten Merkmalen ohne Vorverarbeitung
- 3 Szenarien mit irrelevanten Merkmalen mit Vorverarbeitung

Für jede Bedingung werden 20 Wiederholungen durchgeführt.

In jeder Wiederholung werden 4 Verfahren ausgeführt.

Damit entstehen:

- `19 × 20 × 4 = 1520` Einzelresultate in der Rohresultattabelle
- `19 × 4 = 76` aggregierte Stabilitätseinträge

Für jede Versuchsbedingung und jedes Verfahren wird die Stabilität aus allen paarweisen Vergleichen der 20 Wiederholungen berechnet:

- Anzahl paarweiser Vergleiche: `20 choose 2 = 190`

## Exakte Grundkonfiguration

Die globale Konfiguration ist im Notebook in `CONFIG` definiert.

### Basisdatensatz

Der Basisdatensatz wird genau einmal mit `make_blobs` erzeugt:

- `n_samples = 1000`
- `centers = 3`
- `n_features = 10`
- `cluster_std = 1.0`
- `random_state = 42`

Zusätzlich werden gespeichert:

- `BASE_OBSERVATION_IDS = np.arange(1000)`
- originale Ground-Truth-Labels `y_base`
- Namen der Basismerkmale `feature_00` bis `feature_09`

Die Ground-Truth-Labels werden nur zur internen Kontrolle und für Visualisierung verwendet, nicht als Trainingssignal für die Clustering-Verfahren.

### Wiederholungs-Seeds

Verwendet werden exakt diese 20 Seeds:

`[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]`

Zusätzlich existieren methodenspezifische Seed-Offests:

- `kmeans: +11`
- `hierarchisch: +23`
- `dbscan: +37`
- `rtkm: +53`

Der effektiv übergebene Seed eines Verfahrens in einer Wiederholung ist also:

`method_seed = repeat_seed + METHOD_SEED_OFFSETS[verfahren]`

Hinweis:

- `k-Means` nutzt den Seed direkt über `random_state`
- `RTKM` nutzt den Seed über `np.random.seed(seed)` vor der Initialisierung
- `DBSCAN` und hierarchisches Clustering sind modellseitig deterministisch; zwischen Wiederholungen variiert dort primär die Datensatzrealisierung

## Standardisierte Ablaufreihenfolge pro Lauf

Jeder einzelne Lauf folgt exakt dieser Reihenfolge:

1. Eine Datensatzvariante wird erzeugt.
2. Falls vorgesehen, wird gezielte Vorverarbeitung angewendet.
3. Der resultierende Datensatz wird standardisiert.
4. Das Clustering-Verfahren wird auf den standardisierten Daten ausgeführt.
5. Qualitätsmetriken werden auf den Clustering-Labels berechnet.
6. Die Labels werden für die Stabilitätsanalyse auf die ursprünglichen 1000 Basisbeobachtungen ausgerichtet.

Die Standardisierung erfolgt immer mit:

- `StandardScaler()`

Wichtig:

- Die Verfahren arbeiten immer auf den vollständigen Merkmalen der jeweiligen Datensatzvariante.
- PCA wird nur für 2D-Visualisierung verwendet.

## Rauschformen und ihre Implementierung

### 1. Ohne Rauschen

Im Basisszenario wird direkt auf dem ursprünglichen `make_blobs`-Datensatz gearbeitet:

- 1000 Beobachtungen
- 10 Merkmale
- keine künstlichen Beobachtungen
- keine künstlichen Merkmale

### 2. Ausreißer

Die Funktion `add_outliers(...)` wurde aus dem älteren Notebook übernommen und in die neue Variant-Struktur eingebettet.

#### Parameter

- `noise_level_percent ∈ {10, 20, 30}`
- `min_distance_factor = 2.5`
- `seed = repeat_seed`

#### Erzeugungslogik

Für eine gegebene Rauschstufe:

- Anzahl neuer Ausreißer:
  - 10 % → 100 zusätzliche Beobachtungen
  - 20 % → 200 zusätzliche Beobachtungen
  - 30 % → 300 zusätzliche Beobachtungen

Die Punkte werden so erzeugt:

1. Der Schwerpunkt des Basisdatensatzes wird berechnet.
2. Die Distanzen aller Basisbeobachtungen zu diesem Schwerpunkt werden berechnet.
3. Der typische Radius wird als 95%-Perzentil dieser Distanzen definiert.
4. Neue Kandidaten werden aus einer Normalverteilung um den Schwerpunkt gezogen:
   - Mittelwert: `center`
   - Skala: `X.std(axis=0) * 3`
5. Nur Kandidaten mit
   - `Distanz > min_distance_factor * typical_radius`
   werden akzeptiert.
6. Die akzeptierten Punkte werden an den Datensatz angehängt.

Zusätzlich wird gespeichert:

- `observation_ids` für Basis- und künstliche Beobachtungen
- `is_base_observation`, um ursprüngliche und künstlich erzeugte Punkte auseinanderzuhalten

### 3. Zwischencluster-Ausreißer

Die Funktion `add_intercluster_outliers(...)` ergänzt künstliche Beobachtungen gezielt zwischen den empirischen Clusterzentren des Basisdatensatzes.

#### Parameter

- `noise_level_percent ∈ {10, 20, 30}`
- `seed = repeat_seed`
- `lambda_min = 0.20`
- `jitter_scale = 0.35`
- `exclusion_radius_factor = 1.0`

#### Erzeugungslogik

Für eine gegebene Rauschstufe:

- Anzahl neuer Zwischencluster-Ausreißer:
  - 10 % → 100 zusätzliche Beobachtungen
  - 20 % → 200 zusätzliche Beobachtungen
  - 30 % → 300 zusätzliche Beobachtungen

Die Punkte werden so erzeugt:

1. Für jedes Ground-Truth-Cluster wird das empirische Zentrum berechnet.
2. Aus allen Clusterzentren werden paarweise Verbindungen gebildet.
3. Die künstlichen Punkte werden möglichst gleichmäßig auf diese Clusterpaare verteilt.
4. Für jedes gewählte Paar wird ein Punkt auf der Verbindungsstrecke erzeugt:
   - `base_point = lambda * centroid_a + (1 - lambda) * centroid_b`
   - mit `lambda ∈ [0.20, 0.80]`
5. Zusätzlich wird ein zufälliger Jitter mit
   - `scale = X.std(axis=0) * 0.35`
   addiert.
6. Ein Kandidat wird nur akzeptiert, wenn er außerhalb der 95%-Clusterkerne aller Cluster liegt.
7. Die akzeptierten Punkte werden an den Datensatz angehängt.

Zusätzlich wird gespeichert:

- `observation_ids` für Basis- und künstliche Beobachtungen
- `is_base_observation`, um ursprüngliche und künstlich erzeugte Punkte auseinanderzuhalten
- technische Generator-Metadaten wie Sampling-Versuche, Akzeptanzrate und die genutzten Clusterpaare

### 4. Irrelevante Merkmale

Die Funktion `add_irrelevant_features(...)` wurde aus dem älteren Notebook übernommen und um Metadaten erweitert.

#### Parameter

- `noise_level_percent ∈ {10, 20, 30}`
- `seed = repeat_seed`

#### Anzahl zusätzlicher Merkmale

Die Anzahl irrelevanter Merkmale ist:

- `noise_level_percent / 10`

Also:

- 10 % → 1 zusätzliches irrelevantes Merkmal
- 20 % → 2 zusätzliche irrelevante Merkmale
- 30 % → 3 zusätzliche irrelevante Merkmale

#### Erzeugungslogik

Für die Zusatzmerkmale werden Zufallsvariablen erzeugt mit:

- Mittelwert: mittlerer Mittelwert der Basismerkmale
- Standardabweichung: mittlere Standardabweichung der Basismerkmale

Konkret:

- `mean = np.mean(X, axis=0).mean()`
- `std = np.std(X, axis=0).mean()`
- `irrelevant = rng.normal(loc=mean, scale=std, size=(n_samples, n_irrelevant))`

Die künstlichen Merkmale werden rechts an die ursprüngliche Merkmalsmatrix angehängt.

Zusätzlich wird gespeichert:

- `feature_names`
- `is_artificial_feature`

## Vorverarbeitung

Die Vorverarbeitung ist absichtlich kontrolliert und kennt die Art der Störung, aber nicht die späteren Clusterlabels.

### A. Vorverarbeitung bei Ausreißern

Implementiert über:

- `remove_outliers_iqr(...)`
- `preprocess_outlier_variant(...)`

#### Parameter

- `multiplier = 1.5`
- `min_features_outside = 2`

#### Logik

Für jedes Merkmal werden berechnet:

- `Q1`
- `Q3`
- `IQR = Q3 - Q1`
- untere Grenze: `Q1 - 1.5 * IQR`
- obere Grenze: `Q3 + 1.5 * IQR`

Ein Datenpunkt wird entfernt, wenn er in mindestens 2 Merkmalen außerhalb dieser Grenzen liegt.

Wichtig:

- Diese Vorverarbeitung entfernt nicht “die bekannten künstlichen Ausreißer per ID”.
- Stattdessen wird eine regelbasierte Ausreißerentfernung auf den erzeugten Daten angewendet.
- Dadurch kann es auch vorkommen, dass einzelne Basisbeobachtungen entfernt werden.

### B. Vorverarbeitung bei Zwischencluster-Ausreißern

Implementiert über:

- `compute_soft_cluster_assignments_from_centers(...)`
- `compute_assignment_margins(...)`
- `calibrate_intercluster_margin_threshold(...)`
- `preprocess_intercluster_outlier_variant(...)`

#### Kalibrierung

Vor dem eigentlichen Experiment wird auf dem unverrauschten Basisdatensatz einmalig ein konservativer Margin-Schwellenwert kalibriert.

Verwendet werden:

- `k = 3`
- `kmeans_init = "k-means++"`
- `kmeans_n_init = 20`
- `calibration_random_state = 42`
- `quantile = 0.01`
- `epsilon = 1e-12`

#### Logik

1. Die aktuelle Datensatzvariante wird innerhalb der Vorverarbeitung temporär standardisiert.
2. Auf dieser standardisierten Hilfskopie werden mit `k-Means` vorläufige Clusterzentren bestimmt.
3. Für jeden Punkt werden inverse quadrierte Distanzen zu den Zentren berechnet und zu weichen Zugehörigkeiten normiert.
4. Die Margin ist die Differenz zwischen der größten und zweitgrößten weichen Zugehörigkeit.
5. Punkte mit `margin < threshold` werden entfernt.

Wichtig:

- Die Filterung wird auf Basis standardisierter Hilfsdaten entschieden, aber auf die ursprüngliche Variant-Struktur angewendet.
- Dadurch bleibt die Hauptpipeline unverändert: erst Vorverarbeitung, danach die reguläre Standardisierung für das Clustering.
- Die Methode ist bewusst konservativ kalibriert, damit primär uneindeutige Zwischencluster-Punkte und nur wenige reguläre Randpunkte entfernt werden.

### C. Vorverarbeitung bei irrelevanten Merkmalen

Implementiert über:

- `select_features_by_laplacian_score(...)`
- `laplacian_score(...)` aus [libs/laplacian_score.py](/Users/max/workspace/clust-methods-noise-eval/libs/laplacian_score.py)

#### Parameter

- `n_keep = CONFIG["n_features"] = 10`
- `k_nearest = 5`

#### Logik

1. Für alle Merkmale wird ein Laplacian Score berechnet.
2. Die 10 Merkmale mit dem kleinsten Score werden behalten.
3. Alle übrigen Merkmale werden verworfen.

Das bedeutet:

- Bei Vorverarbeitung irrelevanter Merkmale wird die Merkmalsanzahl wieder auf 10 reduziert.
- Welche 10 Merkmale übrig bleiben, wird datengetrieben über den Laplacian Score entschieden.
- Während des Experimentlaufs wird pro Wiederholung ausgegeben, welche Merkmale ausgewaehlt wurden.
- Jedes ausgewaehlte Merkmal wird dabei als `standard` oder `kuenstlich_hinzugefuegt` gekennzeichnet.

## DBSCAN-Kalibrierung

DBSCAN verwendet nicht denselben `eps` für alle Datensatzvarianten, sondern einen `eps` pro Dimensionalität.

### Verwendete Dimensionalitäten

Kalibriert wird für:

- 10 Merkmale
- 11 Merkmale
- 12 Merkmale
- 13 Merkmale

Das entspricht:

- Basisdatensatz
- Basisdatensatz + 1 irrelevantes Merkmal
- Basisdatensatz + 2 irrelevante Merkmale
- Basisdatensatz + 3 irrelevante Merkmale

### Referenzdaten für die Kalibrierung

Für jede Ziel-Dimensionalität wird eine saubere Referenzvariante gebaut:

- Bei 10 Merkmalen: der Basisdatensatz
- Bei 11 bis 13 Merkmalen: Basisdatensatz plus entsprechende Anzahl irrelevanter Merkmale

Für diese Referenzvarianten wird standardisiert und anschließend `eps` bestimmt.

### `min_samples`

Für DBSCAN gilt:

- `min_samples = 2 * d`

mit `d = aktuelle Dimensionalität`

Also:

- 10 Merkmale → `min_samples = 20`
- 11 Merkmale → `min_samples = 22`
- 12 Merkmale → `min_samples = 24`
- 13 Merkmale → `min_samples = 26`

### `eps`-Bestimmung

`eps` wird numerisch aus der k-distance-Kurve geschätzt:

1. Für jeden Punkt wird die Distanz zum `min_samples`-ten Nachbarn bestimmt.
2. Diese Distanzen werden sortiert.
3. Daraus werden numerisch erste und zweite Ableitung berechnet:

```python
first_derivative = np.gradient(k_distances)
second_derivative = np.gradient(first_derivative)
knee_index = np.argmax(second_derivative)
eps = k_distances[knee_index]
```

Dieser `eps` bleibt anschließend innerhalb derselben Dimensionalität für alle Wiederholungen und Rauschstufen konstant.

Die Kalibrierung wird exportiert nach:

- `results/clustering_evaluation/tables/dbscan_kalibrierung.csv`

## Exakte Modellparameter

### 1. k-Means

- `n_clusters = 3`
- `init = "k-means++"`
- `n_init = 20`
- `random_state = method_seed`

### 2. Hierarchisches Clustering

- `n_clusters = 3`
- `linkage = "ward"`

Für Dendrogramme wird bei Beispielplots zusätzlich:

- `compute_distances = True`

verwendet.

### 3. DBSCAN

- `eps = DBSCAN_PARAMS_BY_DIM[aktuelle_dimensionalitaet]["eps"]`
- `min_samples = DBSCAN_PARAMS_BY_DIM[aktuelle_dimensionalitaet]["min_samples"]`

### 4. RTKM

Implementiert über [libs/RTKM.py](/Users/max/workspace/clust-methods-noise-eval/libs/RTKM.py).

Im Notebook wird aufgerufen:

- `k = 3`
- `num_members = 1`
- `percent_outliers = alpha`

`alpha` ist:

- Basisszenario ohne Rauschen: `0.0`
- irrelevante Merkmale: `0.0`
- Ausreißer 10 %: `0.091`
- Ausreißer 20 %: `0.167`
- Ausreißer 30 %: `0.231`
- Zwischencluster-Ausreißer 10 %: `0.091`
- Zwischencluster-Ausreißer 20 %: `0.167`
- Zwischencluster-Ausreißer 30 %: `0.231`

Zusätzlich:

- vor RTKM wird `np.random.seed(method_seed)` gesetzt

Wichtig:

- RTKM liefert intern Ausreißer als eigenes Label `k`
- dieses Label wird im Notebook anschließend auf `-1` umcodiert

## Bewertungsmaße

### Qualitätsmaße

Es wird eine Qualitätsmetrik verwendet:

- `SIL+` (`silhouette_plus`)

Die Funktion `compute_quality_metrics(...)` berechnet dieses Maß auf den standardisierten Daten.

#### Umgang mit Noise

Vor der Berechnung werden alle Beobachtungen mit Label `-1` ausgeschlossen:

- bei DBSCAN: Noise-Punkte
- bei RTKM: auf `-1` umcodierte Ausreißer

Wenn danach weniger als 2 Cluster vorhanden sind oder zu wenige gültige Beobachtungen übrig bleiben, wird:

- `NaN` für die jeweilige Metrik gespeichert

### Zusätzliche DBSCAN-Information

Pro Lauf werden zusätzlich gespeichert:

- Anzahl erkannter Cluster
- Anzahl der als Noise markierten Punkte
- Anteil der als Noise markierten Punkte

Diese Informationen stehen in der Rohresultattabelle.

## Stabilitätsmaß

Die Stabilität wird über den `Adjusted Rand Index (ARI)` gemessen.

### Grundidee

Für jede Versuchsbedingung und jedes Verfahren liegen 20 Clusterlösungen aus 20 Wiederholungen vor.

Diese 20 Labelvektoren werden paarweise verglichen.

### Wichtig für die Szenarien mit künstlichen Beobachtungen

Die Stabilitätsanalyse erfolgt immer auf den ursprünglichen 1000 Basisbeobachtungen.

Dazu wird nach jedem Lauf:

- der Labelvektor auf die Basisbeobachtungen zurückprojiziert
- über `observation_ids` auf die ursprünglichen `BASE_OBSERVATION_IDS` ausgerichtet
- für fehlende Basisbeobachtungen, die durch Vorverarbeitung entfernt wurden, `-1` gesetzt

Dadurch ist die Stabilitätsberechnung auch dann konsistent, wenn Beobachtungen durch IQR- oder Margin-Vorverarbeitung entfernt werden.

### Aggregation der Stabilität

Für jede Bedingung und jedes Verfahren werden berechnet:

- `mittlerer_paarweiser_ari`
- `std_paarweiser_ari`
- `anzahl_vergleiche = 190`

## Szenarien im Detail

Die vollständige Bedingungsmatrix wird durch `build_condition_grid()` erzeugt.

### 1. Benchmark ohne Rauschen

- `scenario = "ohne_rauschen"`
- `rauschform = "none"`
- `rauschstufe_prozent = 0`
- `vorverarbeitung = False`

### 2. Ausreißer ohne Vorverarbeitung

Für `rauschstufe_prozent ∈ {10, 20, 30}`:

- `scenario = "mit_rauschen"`
- `rauschform = "ausreisser"`
- `vorverarbeitung = False`

### 3. Ausreißer mit Vorverarbeitung

Für `rauschstufe_prozent ∈ {10, 20, 30}`:

- `scenario = "mit_rauschen_und_vorverarbeitung"`
- `rauschform = "ausreisser"`
- `vorverarbeitung = True`

### 4. Zwischencluster-Ausreißer ohne Vorverarbeitung

Für `rauschstufe_prozent ∈ {10, 20, 30}`:

- `scenario = "mit_rauschen"`
- `rauschform = "zwischencluster_ausreisser"`
- `vorverarbeitung = False`

### 5. Zwischencluster-Ausreißer mit Vorverarbeitung

Für `rauschstufe_prozent ∈ {10, 20, 30}`:

- `scenario = "mit_rauschen_und_vorverarbeitung"`
- `rauschform = "zwischencluster_ausreisser"`
- `vorverarbeitung = True`

### 6. Irrelevante Merkmale ohne Vorverarbeitung

Für `rauschstufe_prozent ∈ {10, 20, 30}`:

- `scenario = "mit_rauschen"`
- `rauschform = "irrelevante_merkmale"`
- `vorverarbeitung = False`

### 7. Irrelevante Merkmale mit Vorverarbeitung

Für `rauschstufe_prozent ∈ {10, 20, 30}`:

- `scenario = "mit_rauschen_und_vorverarbeitung"`
- `rauschform = "irrelevante_merkmale"`
- `vorverarbeitung = True`

## Ergebnisdateien und ihre Bedeutung

Alle Ergebnisse werden unter `results/clustering_evaluation/` gespeichert.

### Tabellen

Im Ordner `results/clustering_evaluation/tables/` werden erzeugt:

- `rohergebnisse.csv`
- `stabilitaet.csv`
- `aggregierte_ergebnisse.csv`
- `benchmark_ohne_rauschen.csv`
- `aggregierte_ergebnisse_ausreisser_ohne_vorverarbeitung.csv`
- `aggregierte_ergebnisse_ausreisser_mit_vorverarbeitung.csv`
- `aggregierte_ergebnisse_zwischencluster_ausreisser_ohne_vorverarbeitung.csv`
- `aggregierte_ergebnisse_zwischencluster_ausreisser_mit_vorverarbeitung.csv`
- `aggregierte_ergebnisse_irrelevante_merkmale_ohne_vorverarbeitung.csv`
- `aggregierte_ergebnisse_irrelevante_merkmale_mit_vorverarbeitung.csv`
- `intercluster_margin_threshold.csv`
- `intercluster_margin_preprocessing.csv`
- `laplacian_merkmalsauswahl_irrelevante_merkmale_mit_vorverarbeitung.csv`
- `basis_beobachtungen.csv`
- `basis_merkmale.csv`
- `dbscan_kalibrierung.csv`

### Metadaten

Zusätzlich:

- `results/clustering_evaluation/run_metadata.json`

Diese Datei enthält:

- die `CONFIG`
- die `repeat_seeds`
- die DBSCAN-Kalibrierungsparameter
- Paketversionen

### Grafiken

Alle Grafiken werden als `SVG` in `results/clustering_evaluation/figures/` gespeichert.

Dazu gehören unter anderem:

- PCA-Darstellungen der Datensatzvarianten
- Clustervergleichsplots
- DBSCAN-k-distance-Kalibrierung
- gruppierte Dendrogramme für hierarchisches Clustering
- Verlaufsplots für `SIL+` und Stabilität

## Wie die Aggregation zustande kommt

### Rohresultate

In `raw_results` steht:

- eine Zeile pro
  - Versuchsbedingung
  - Wiederholung
  - Verfahren

Wichtige Spalten:

- `scenario`
- `rauschform`
- `rauschstufe_prozent`
- `vorverarbeitung`
- `wiederholung`
- `wiederholungs_seed`
- `verfahren`
- `aktuelle_anzahl_merkmale`
- `aktuelle_anzahl_beobachtungen`
- `kuenstliche_beobachtungen`
- `kuenstliche_merkmale`
- `gefundene_cluster`
- `noise_punkte_dbscan`
- `noise_anteil_dbscan`
- `silhouette_plus`
- `laplacian_auswahl_verwendet`
- `laplacian_ausgewaehlte_merkmale_mit_typ`
- `laplacian_ausgewaehlte_standardmerkmale`
- `laplacian_ausgewaehlte_kuenstliche_merkmale`
- `laplacian_anzahl_standardmerkmale_ausgewaehlt`
- `laplacian_anzahl_kuenstliche_merkmale_ausgewaehlt`
- `dbscan_eps`
- `dbscan_min_samples`
- `rtkm_alpha`

### Qualitätsaggregation

Die Qualitätsmetriken werden über alle 20 Wiederholungen gemittelt:

- `silhouette_plus_mean`
- `silhouette_plus_std`

### Zusatztabelle fuer Laplacian-Merkmalsauswahl

Die Datei `laplacian_merkmalsauswahl_irrelevante_merkmale_mit_vorverarbeitung.csv` enthaelt fuer das Szenario `irrelevante_merkmale` mit Vorverarbeitung pro Wiederholung:

- die durch den Laplacian Score ausgewaehlten Merkmale inklusive Typkennzeichnung
- die ausgewaehlten `standard`-Merkmale
- die ausgewaehlten `kuenstlich_hinzugefuegt`-Merkmale
- die Anzahl der ausgewaehlten Merkmale je Typ

### Zusatztabellen fuer Margin-Vorverarbeitung

Die Datei `intercluster_margin_threshold.csv` enthaelt die einmalige Kalibrierung des Margin-Schwellenwerts auf dem sauberen Basisdatensatz inklusive:

- Quantil
- Schwellenwert
- `k-Means`-Konfiguration
- numerischer Stabilisierung
- Kenngrößen der sauberen Margin-Verteilung

Die Datei `intercluster_margin_preprocessing.csv` enthaelt fuer das Szenario `zwischencluster_ausreisser` mit Vorverarbeitung pro Wiederholung:

- den verwendeten Margin-Schwellenwert
- die Anzahl entfernter Beobachtungen insgesamt
- die Anzahl entfernter Basisbeobachtungen
- die Anzahl entfernter künstlicher Zwischencluster-Ausreißer
- die verbleibende Beobachtungszahl nach Vorverarbeitung

### Stabilitätsaggregation

Aus `stability_results` kommen:

- `mittlerer_paarweiser_ari`
- `std_paarweiser_ari`
- `anzahl_vergleiche`

Diese werden per Merge mit den Qualitätsaggregaten zusammengeführt.

## Was man beim erneuten Ausführen beachten sollte

Für reproduzierbare Ergebnisse:

1. Das Notebook vollständig von oben nach unten ausführen.
2. Keine Zellen in anderer Reihenfolge ausführen.
3. Falls bestehende Notebook-Ausgaben sichtbar sind:
   die relevanten Zellen erneut laufen lassen, damit Tabellen und Plots auf dem aktuellen Code basieren.

## Kurzfassung der wichtigsten Zahlen

- Basisdatensatz: `1000 × 10`
- Verfahren: `4`
- Rauschstufen: `10 %, 20 %, 30 %`
- Wiederholungen pro Bedingung: `20`
- Versuchsbedingungen: `19`
- Rohresultate: `1520`
- Aggregierte Stabilitätseinträge: `76`
- Stabilitätsvergleiche pro Bedingung und Verfahren: `190`
- Exportformat der Abbildungen: `SVG`

## Fazit zur Nachvollziehbarkeit

Die Ergebnisse entstehen deterministisch aus:

- einem festen Basisdatensatz
- einer festen Seed-Liste
- explizit definierten Rauschgeneratoren
- explizit definierten Vorverarbeitungsschritten
- fest codierten Modellparametern
- einer klaren Aggregationslogik

Wer das Notebook vollständig neu ausführt, erhält denselben Ergebnisaufbau, dieselben Exportdateien und dieselbe Logik der Auswertung.
