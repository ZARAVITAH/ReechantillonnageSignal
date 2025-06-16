ANALYSEUR DE RE-ECHANTILLONNAGE DE SIGNAUX VIBRATOIRES
App : https://reechantillonnage-angelico-zaravita-ammbb4tgroksf8b97cy2zz.streamlit.app/ (secours : https://reechantillonnage-by-angelico-zaravita-signal-rswr9pjrsbdqpxhp.streamlit.app/ )
Application web scientifique pour l'analyse et le ré-échantillonnage de signaux vibratoires basée sur des méthodes de traitement du signal éprouvées.
 Objectif
Résoudre le problème d'insuffisance de données vibratoires enregistrées en appliquant des techniques de ré-échantillonnage scientifiquement validées pour augmenter la résolution temporelle des signaux.
 Méthodes Implémentées
 Méthode                                	Cas d'usage	Référence scientifique
Interpolation Spline Cubique	Données modérées, bruit faible à moyen	Unser (1999)
FFT Resampling	Signaux périodiques ou réguliers	Oppenheim & Schafer
SWT + Interpolation	Signaux transitoires/non-stationnaires	Mallat (2008)
Gaussian Process Regression	Très peu de points, modélisation fine	Rasmussen & Williams (2006)
Fonctionnalités
•	📊 Génération de signaux synthétiques : Signaux vibratoires réalistes avec composantes fréquentielles industrielles
•	📁 Import de données CSV : Support format personnalisé (séparateur ;, temps en ms)
•	🔄 Ré-échantillonnage intelligent : 4 méthodes scientifiques avec paramètres ajustables
•	📈 Visualisation comparative : Superposition des méthodes avec signal original
•	📋 Métriques d'évaluation : MSE, MAE, RMSE, Corrélation, SNR
•	🌊 Analyse spectrale : Comparaison FFT pour validation
•	💾 Export des résultats : Téléchargement CSV des signaux traités
•	💡 Recommandations automatiques : Sélection optimale basée sur les performances


Format des Données
Fichier CSV attendu :
•	Séparateur : ; (point-virgule)
•	Première ligne : En-têtes (ignorée)
•	Colonne 1 : Temps en millisecondes
•	Colonne 2 : Amplitude du signal
Exemple :
Temps(ms);Amplitude
0;0.125
10;0.234
20;-0.156
...

Utilisation
1.	Sélection des données :
o	Signal synthétique (paramètres configurables)
o	Import fichier CSV
2.	Configuration :
o	Nombre de points de ré-échantillonnage
o	Sélection des méthodes à appliquer
3.	Analyse :
o	Visualisation comparative
o	Métriques de performance
o	Analyse spectrale
4.	Export :
o	Téléchargement des résultats CSV
o	Recommandations d'usage
 
 
 Base Scientifique
Références
•	Unser, M. (1999) - "Splines: A perfect fit for signal and image processing" - IEEE Signal Processing Magazine
•	Oppenheim, A. & Schafer, R. - "Discrete-Time Signal Processing" - Prentice Hall
•	Mallat, S. (2008) - "A Wavelet Tour of Signal Processing" - Academic Press
•	Rasmussen, C. & Williams, C. (2006) - "Gaussian Processes for Machine Learning" - MIT Press
 Technologies Utilisées
•	Frontend : Streamlit
•	Traitement du Signal : SciPy, PyWavelets
•	Machine Learning : Scikit-learn
•	Visualisation : Matplotlib, Seaborn
•	Données : Pandas, NumPy

Métriques d'Évaluation
Métrique	Description	Usage
MSE	Mean Squared Error	Erreur quadratique moyenne
MAE	Mean Absolute Error	Erreur absolue moyenne
RMSE	Root Mean Squared Error	Racine de l'erreur quadratique
Corrélation	Coefficient de Pearson	Fidélité du signal
SNR	Signal-to-Noise Ratio	Rapport signal/bruit (dB)


Cas d'Usage Typiques
Maintenance Prédictive
•	Analyse de vibrations de machines tournantes
•	Détection de défauts de roulements
•	Surveillance d'équipements industriels

Recherche & Développement
•	Augmentation de résolution de données expérimentales
•	Validation de modèles numériques
•	Préparation de datasets pour IA


Contrôle Qualité
•	Analyse de signaux réels
•	Validation de capteurs vibratoires
•	Optimisation de paramètres d'acquisition
Auteurs
•	A. ANGELICO et ZARAVITA 
Remerciements
•	Communauté scientifique pour les méthodes de référence
•	Équipe Streamlit pour l'excellent framework
•	Contributeurs des bibliothèques open-source utilisées
Contact
•	Email: Angelico@vibra-service.com   zaravitamds18@gmail.com 


