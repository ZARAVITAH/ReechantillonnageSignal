import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.fft import fft, ifft, fftfreq
import pywt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from datetime import datetime
from io import BytesIO, StringIO
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Ré-échantillonnage Vibratoire",
    page_icon="📊",
    layout="wide"
)

class VibrationResampler:
    """
    Classe pour le ré-échantillonnage de signaux vibratoires
    Méthodes basées sur la littérature scientifique
    """
    
    def __init__(self):
        self.methods = {
            'spline': 'Interpolation Spline Cubique',
            'fft': 'FFT Resampling',
            'swt': 'SWT + Interpolation',
            'gpr': 'Gaussian Process Regression'
        }
    
    def generate_synthetic_signal(self, duration=2, fs_original=100, noise_level=0.1):
        """
        Génère un signal vibratoire synthétique complexe
        Basé sur les caractéristiques typiques des signaux vibratoires industriels
        """
        t = np.linspace(0, duration, int(fs_original * duration))
        
        # Composantes fréquentielles typiques d'un signal vibratoire
        # Fréquence fondamentale (rotation)
        f1 = 25  # Hz
        # Harmoniques
        f2 = 2 * f1
        f3 = 3 * f1
        # Fréquence de défaut (roulement par exemple)
        f_defaut = 73  # Hz
        
        # Signal composite
        signal_clean = (
            1.0 * np.sin(2 * np.pi * f1 * t) +
            0.5 * np.sin(2 * np.pi * f2 * t + np.pi/4) +
            0.3 * np.sin(2 * np.pi * f3 * t + np.pi/3) +
            0.2 * np.sin(2 * np.pi * f_defaut * t) +
            0.1 * np.sin(2 * np.pi * 150 * t)  # Haute fréquence
        )
        
        # Modulation d'amplitude (effet non-stationnaire)
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)
        signal_clean *= modulation
        
        # Ajout de bruit gaussien
        noise = noise_level * np.random.normal(0, 1, len(t))
        signal_noisy = signal_clean + noise
        
        return t, signal_noisy
    
    def spline_interpolation(self, t_orig, y_orig, n_points):
        """
        Interpolation spline cubique
        Référence: Unser, M. (1999). Splines: A perfect fit for signal and image processing
        """
        # Création des nouveaux points temporels
        t_new = np.linspace(t_orig[0], t_orig[-1], n_points)
        
        # Interpolation spline cubique
        cs = interpolate.CubicSpline(t_orig, y_orig, bc_type='natural')
        y_new = cs(t_new)
        
        return t_new, y_new
    
    def fft_resampling(self, t_orig, y_orig, n_points):
        """
        Ré-échantillonnage par FFT (méthode de zero-padding spectral)
        Référence: Oppenheim & Schafer - Discrete-Time Signal Processing
        """
        N_orig = len(y_orig)
        
        # Calcul de la FFT
        Y_fft = fft(y_orig)
        
        # Zero-padding ou troncature dans le domaine fréquentiel
        if n_points > N_orig:
            # Upsampling: zero-padding
            Y_new = np.zeros(n_points, dtype=complex)
            mid = N_orig // 2
            Y_new[:mid] = Y_fft[:mid]
            Y_new[-mid:] = Y_fft[-mid:]
            # Correction d'amplitude
            Y_new *= n_points / N_orig
        else:
            # Downsampling: troncature
            mid_new = n_points // 2
            mid_orig = N_orig // 2
            Y_new = np.zeros(n_points, dtype=complex)
            Y_new[:mid_new] = Y_fft[:mid_new]
            Y_new[-mid_new:] = Y_fft[-mid_new:]
        
        # IFFT pour retourner au domaine temporel
        y_new = np.real(ifft(Y_new))
        t_new = np.linspace(t_orig[0], t_orig[-1], n_points)
        
        return t_new, y_new
    
    def swt_interpolation(self, t_orig, y_orig, n_points, wavelet='db4'):
        """
        Transformée en ondelettes stationnaire + interpolation
        Référence: Mallat, S. (2008). A Wavelet Tour of Signal Processing
        """
        # Nombre de niveaux de décomposition
        max_levels = pywt.swt_max_level(len(y_orig))
        levels = min(4, max_levels)  # Limiter à 4 niveaux
        
        # SWT décomposition
        coeffs = pywt.swt(y_orig, wavelet, levels)
        
        # Interpolation de chaque niveau de coefficients
        t_new = np.linspace(t_orig[0], t_orig[-1], n_points)
        coeffs_interp = []
        
        for cA, cD in coeffs:
            # Interpolation des coefficients d'approximation et de détail
            f_cA = interpolate.interp1d(t_orig, cA, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')
            f_cD = interpolate.interp1d(t_orig, cD, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')
            
            cA_new = f_cA(t_new)
            cD_new = f_cD(t_new)
            coeffs_interp.append((cA_new, cD_new))
        
        # Reconstruction du signal
        y_new = pywt.iswt(coeffs_interp, wavelet)
        
        return t_new, y_new
    
    def gpr_interpolation(self, t_orig, y_orig, n_points):
        """
        Régression par processus gaussien
        Référence: Rasmussen & Williams (2006). Gaussian Processes for Machine Learning
        """
        # Reshape pour sklearn
        X_orig = t_orig.reshape(-1, 1)
        
        # Définition du kernel (RBF + Matern pour capturer différentes caractéristiques)
        kernel = RBF(length_scale=0.1) + Matern(length_scale=0.1, nu=1.5)
        
        # Création du modèle GPR
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Régularisation pour la stabilité numérique
            n_restarts_optimizer=3
        )
        
        # Entraînement
        gpr.fit(X_orig, y_orig)
        
        # Prédiction
        t_new = np.linspace(t_orig[0], t_orig[-1], n_points)
        X_new = t_new.reshape(-1, 1)
        y_new, std = gpr.predict(X_new, return_std=True)
        
        return t_new, y_new, std
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calcul des métriques d'évaluation
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Corrélation de Pearson
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(y_true**2)
        noise_power = np.mean((y_true - y_pred)**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'SNR (dB)': snr
        }

def main():
    st.title("🔬 Analyseur de Ré-échantillonnage de signaux vibratoires")
    st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <p style="font-size: 18px;"># **Application Web scientifique pour l'analyse et le ré-échantillonnage de signaux vibratoires**</p>
</div>

<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
    <h3>📌 OBJECTIF PRINCIPAL</h3>
    <p>Résoudre le <strong>problème d'insuffisance de données vibratoires enregistrées</strong> en appliquant des techniques de ré-échantillonnage scientifiquement validées pour augmenter la résolution temporelle des signaux.</p>
</div>

<div style="margin-bottom: 30px;">
    <h2>✨ Fonctionnalités</h2>
    <ul style="list-style-type: none; padding-left: 0;">
        <li style="margin-bottom: 10px;">• 📊 <strong>Génération de signaux synthétiques</strong>: Signaux vibratoires réalistes avec composantes fréquentielles industrielles</li>
        <li style="margin-bottom: 10px;">• 📁 <strong>Import de données CSV</strong>: Support format personnalisé (séparateur ;, temps en ms)</li>
        <li style="margin-bottom: 10px;">• 🔄 <strong>Ré-échantillonnage intelligent</strong>: 4 méthodes scientifiques avec paramètres ajustables</li>
        <li style="margin-bottom: 10px;">• 📈 <strong>Visualisation comparative</strong>: Superposition des méthodes avec signal original</li>
        <li style="margin-bottom: 10px;">• 📋 <strong>Métriques d'évaluation</strong>: MSE, MAE, RMSE, Corrélation, SNR</li>
        <li style="margin-bottom: 10px;">• 🌊 <strong>Analyse spectrale</strong>: Comparaison FFT pour validation</li>
        <li style="margin-bottom: 10px;">• 💾 <strong>Export des résultats</strong>: Téléchargement CSV des signaux traités</li>
        <li style="margin-bottom: 10px;">• 🔗 <strong>Compatibilité</strong>: Les signaux exportés sont directement utilisables dans l'application <a href="https://blsd-analyse-bwgu6rqt9v52qetbazwkdv.streamlit.app/" target="_blank" style="color: #1e88e5; font-weight: bold;">Analyse Vibratoire BLSD</a></li>
        <li style="margin-bottom: 10px;">• 💡 <strong>Recommandations automatiques</strong>: Sélection optimale basée sur les performances</li>
    </ul>
</div>

<div style="background-color: #e6f7ff; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
    <h2>📚 Méthodes Scientifiques Implémentées</h2>
    <ul>
        <li style="margin-bottom: 12px;"><strong>Spline Cubique</strong>: Unser (1999) - Optimal pour données modérées avec bruit faible</li>
        <li style="margin-bottom: 12px;"><strong>FFT Resampling</strong>: Oppenheim & Schafer - Idéal pour signaux périodiques</li>
        <li style="margin-bottom: 12px;"><strong>SWT + Interpolation</strong>: Mallat (2008) - Signaux transitoires/non-stationnaires</li>
        <li style="margin-bottom: 12px;"><strong>Gaussian Process Regression</strong>: Rasmussen & Williams (2006) - Modélisation fine avec peu de points</li>
    </ul>
</div>

<div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
    <p>Ce projet a été réalisé par <strong>A. Angelico</strong> et <strong>ZARAVITA</strong> dans le cadre de l'analyse vibratoire avancée</p>
    <p style="font-size: 14px; color: #666;">© 2024 - Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)
    
    # Initialisation de l'analyseur
    resampler = VibrationResampler()
    
    # Sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres de Configuration")
    
    # Choix du type de données
    data_source = st.sidebar.radio(
        "Source des données :",
        ["Signal synthétique", "Données uploadées"]
    )
    
    if data_source == "Signal synthétique":
        st.sidebar.subheader("Paramètres du signal synthétique")
        duration = st.sidebar.slider("Durée (s)", 1, 10, 2)
        fs_original = st.sidebar.slider("Fréquence d'échantillonnage originale (Hz)", 50, 500, 100)
        noise_level = st.sidebar.slider("Niveau de bruit", 0.0, 0.5, 0.1)
        
        # Génération du signal
        t_orig, y_orig = resampler.generate_synthetic_signal(duration, fs_original, noise_level)
        
    else:
        # Upload du fichier CSV
        uploaded_file = st.file_uploader("Importez votre fichier CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                # Lecture du fichier CSV
                data = pd.read_csv(uploaded_file, sep=";", skiprows=1)
                # Conversion des colonnes en numpy arrays
                time = data.iloc[:, 0].values / 1000  # Conversion en secondes
                amplitude = data.iloc[:, 1].values
                
                # Assignation aux variables utilisées par le reste du code
                t_orig = time
                y_orig = amplitude
                
                # Aperçu du dataset
                if st.checkbox("Afficher les 5 premières lignes du dataset"):
                    st.write(data.head())
                
                st.success("✅ Données chargées avec succès")
                st.info(f"📊 {len(t_orig)} points chargés | Durée: {t_orig[-1]:.2f}s")
                
            except Exception as e:
                st.error(f"❌ Erreur de lecture: {e}")
                return
        else:
            st.info("📁 Veuillez télécharger un fichier CSV")
            return
    
    # Paramètres de ré-échantillonnage
    st.sidebar.subheader("Paramètres de ré-échantillonnage")
    #n_points = st.sidebar.number_input(
    #    "Nombre de points souhaités",
    #    min_value=256,
    #    max_value=32768,                 # au lieu de 5000 fixe ou ça
    #    value=len(t_orig) * 2,                     # au lieu de len(t_orig) * 2, min(len(t_orig) * 2, max_points)
    #    step=10
    #)
    # Définir les options discrètes en puissances de 2
    min_power = 8  # 2^8 = 256
    max_power = 15  # 2^15 = 32768
    options = [2**i for i in range(min_power, max_power + 1)]

    # Trouver la valeur par défaut la plus proche parmi les options
    default_value = len(t_orig) * 2
    closest_value = min(options, key=lambda x: abs(x - default_value))
    
    n_points = st.sidebar.selectbox(
       "Nombre de points souhaités",
        options=options,
        index=options.index(closest_value)
     )
    
    # Sélection des méthodes
    selected_methods = st.sidebar.multiselect(
        "Méthodes à appliquer :",
        list(resampler.methods.keys()),
        default=['spline', 'fft'],
        format_func=lambda x: resampler.methods[x]
    )
    
    if not selected_methods:
        st.warning("⚠️ Veuillez sélectionner au moins une méthode")
        return
    
    # Affichage du signal original
    st.header("📈 Signal Original")
    fig_orig, ax_orig = plt.subplots(figsize=(12, 4))
    ax_orig.plot(t_orig, y_orig, 'b-', linewidth=1, alpha=0.8, label='Signal original')
    ax_orig.set_xlabel('Temps (s)')
    ax_orig.set_ylabel('Amplitude')
    ax_orig.set_title(f'Signal Original ({len(t_orig)} points)')
    ax_orig.grid(True, alpha=0.3)
    ax_orig.legend()
    st.pyplot(fig_orig)
    
    # Informations sur le signal
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Points originaux", len(t_orig))
    with col2:
        st.metric("Durée", f"{t_orig[-1]:.2f} s")
    with col3:
        st.metric("Fréq. d'échantillonnage", f"{len(t_orig)/(t_orig[-1]-t_orig[0]):.1f} Hz")
    with col4:
        st.metric("Amplitude max", f"{np.max(np.abs(y_orig)):.3f}")
    
    # Application des méthodes de ré-échantillonnage
    if st.button("🚀 Lancer l'analyse", type="primary"):
        results = {}
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, method in enumerate(selected_methods):
            status_text.text(f"Application de la méthode: {resampler.methods[method]}")
            
            try:
                if method == 'spline':
                    t_new, y_new = resampler.spline_interpolation(t_orig, y_orig, n_points)
                    results[method] = {'t': t_new, 'y': y_new}
                    
                elif method == 'fft':
                    t_new, y_new = resampler.fft_resampling(t_orig, y_orig, n_points)
                    results[method] = {'t': t_new, 'y': y_new}
                    
                elif method == 'swt':
                    t_new, y_new = resampler.swt_interpolation(t_orig, y_orig, n_points)
                    results[method] = {'t': t_new, 'y': y_new}
                    
                elif method == 'gpr':
                    t_new, y_new, std = resampler.gpr_interpolation(t_orig, y_orig, n_points)
                    results[method] = {'t': t_new, 'y': y_new, 'std': std}
                
            except Exception as e:
                st.error(f"❌ Erreur avec la méthode {method}: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(selected_methods))
        
        status_text.text("✅ Analyse terminée!")
        
        # Visualisation des résultats
        st.header("📊 Résultats du Ré-échantillonnage")
        
        # Graphique de comparaison
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Signal original
        ax.plot(t_orig, y_orig, 'ko-', markersize=3, linewidth=1, 
                alpha=0.7, label='Signal original')
        
        # Signaux interpolés
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, (method, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            ax.plot(data['t'], data['y'], color=color, linewidth=2, 
                   alpha=0.8, label=f'{resampler.methods[method]} ({n_points} pts)')
            
            # Ajout de l'incertitude pour GPR
            if method == 'gpr' and 'std' in data:
                ax.fill_between(data['t'], 
                               data['y'] - 2*data['std'], 
                               data['y'] + 2*data['std'], 
                               alpha=0.2, color=color, 
                               label=f'Incertitude ±2σ')
        
        ax.set_xlabel('Temps (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Comparaison des Méthodes de Ré-échantillonnage', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Calcul des métriques de performance
        st.header("📋 Métriques de Performance")
        
        # Création d'un signal de référence haute résolution pour évaluation
        t_ref, y_ref = resampler.spline_interpolation(t_orig, y_orig, n_points)
        
        metrics_data = []
        for method, data in results.items():
            # Interpolation pour avoir la même base temporelle
            f_interp = interpolate.interp1d(data['t'], data['y'], kind='linear', 
                                           bounds_error=False, fill_value='extrapolate')
            y_eval = f_interp(t_ref)
            
            metrics = resampler.calculate_metrics(y_ref, y_eval)
            metrics['Méthode'] = resampler.methods[method]
            metrics_data.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Affichage des métriques
        st.dataframe(
            metrics_df.set_index('Méthode').round(4),
            use_container_width=True
        )
        
        # Graphique des métriques
        fig_metrics, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics_to_plot = ['MSE', 'MAE', 'Correlation', 'SNR (dB)']
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            values = [m[metric] for m in metrics_data]
            methods = [m['Méthode'] for m in metrics_data]
            
            bars = ax.bar(methods, values, color=colors[:len(methods)])
            ax.set_title(f'{metric}')
            ax.set_ylabel(metric)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Ajout des valeurs sur les barres
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig_metrics)
        
        # Analyse spectrale
        st.header("🌊 Analyse Spectrale")
        
        fig_fft, ax_fft = plt.subplots(figsize=(12, 6))
        
        # FFT du signal original
        freqs_orig = fftfreq(len(t_orig), t_orig[1] - t_orig[0])[:len(t_orig)//2]
        fft_orig = np.abs(fft(y_orig))[:len(t_orig)//2]
        ax_fft.plot(freqs_orig, fft_orig, 'k-', linewidth=2, alpha=0.7, label='Original')
        
        # FFT des signaux interpolés
        for i, (method, data) in enumerate(results.items()):
            dt = data['t'][1] - data['t'][0]
            freqs = fftfreq(len(data['t']), dt)[:len(data['t'])//2]
            fft_signal = np.abs(fft(data['y']))[:len(data['t'])//2]
            
            color = colors[i % len(colors)]
            ax_fft.plot(freqs, fft_signal, color=color, linewidth=1.5, 
                       alpha=0.8, label=resampler.methods[method])
        
        ax_fft.set_xlabel('Fréquence (Hz)')
        ax_fft.set_ylabel('Amplitude')
        ax_fft.set_title('Comparaison Spectrale des Méthodes')
        ax_fft.set_xlim(0, min(50, np.max(freqs_orig)))
        ax_fft.grid(True, alpha=0.3)
        ax_fft.legend()
        st.pyplot(fig_fft)
        
        # Téléchargement des résultats
        st.header("💾 Téléchargement des Résultats")
        
        method_to_download = st.selectbox(
            "Choisir la méthode à télécharger :",
            list(results.keys()),
            format_func=lambda x: resampler.methods[x]
        )
        
        if method_to_download in results:
            download_data = pd.DataFrame({
                'time[ms]': results[method_to_download]['t']*1000,       #------------------------------------------
                'amplitude[g]': results[method_to_download]['y']         #---------------------------------
            })
            # Créer l'en-tête personnalisé-----------------------------------------------------------------------------------------
            today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = (
                  f"//Reéchantillonnage par A. ANGELICO et ZARAVITA\t"
                  f"date:{today}\t"
                  f"Nombre des points originales: {len(t_orig)}\t"
                  f"Nombres des points finales: {n_points}\n"
            )
            csv_buffer = StringIO()
            csv_buffer.write(header)
            download_data.to_csv(csv_buffer, index=False, sep=';')
            csv_content = csv_buffer.getvalue()
            #--------------------------------------------AJOUT
            
            #csv_content += "time[ms]\tamplitude[g]\t\n"
            # Ajouter les données avec point-virgule comme séparateur
            #for _, row in download_data.iterrows():
            #    csv_content += f"{row['time[ms]']:.7f}\t{row['amplitude[g]']:.7f}\t\n"

             #csv_content += download_data.to_csv(
             #   sep='\t', 
             #   index=False, 
             #  header=True,  # Garder les noms de colonnes
             #   float_format="%.7f",  # Format des nombres flottants
             #   lineterminator='\n'  # Terminaison de ligne standard
             #)
    
            # Créer le buffer de téléchargement
            #csv_buffer = BytesIO()
            #csv_buffer.write(csv_content.encode('utf-8'))
            #csv_buffer.seek(0)
            
            #-------------------------VRAI
            #csv_buffer = BytesIO()
            #download_data.to_csv(csv_buffer, index=False)
            #csv_buffer.seek(0)
            
            st.download_button(
                label=f"📥 Télécharger {resampler.methods[method_to_download]}",
                data=csv_content,                       #csv_buffer.getvalue(),
                file_name=f"signal_resampled_{method_to_download}_{n_points}pts_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
                mime="text/csv"
            )
        
        # Recommandations
        st.header("💡 Recommandations")
        
        best_method_corr = max(metrics_data, key=lambda x: x['Correlation'])
        best_method_snr = max(metrics_data, key=lambda x: x['SNR (dB)'])
        
        st.success(f"""
        **Analyse des résultats :**
        
        • **Meilleure corrélation** : {best_method_corr['Méthode']} (r = {best_method_corr['Correlation']:.3f})
        • **Meilleur SNR** : {best_method_snr['Méthode']} ({best_method_snr['SNR (dB)']:.1f} dB)
        
        **Conseils d'utilisation :**
        - Pour signaux périodiques réguliers → FFT Resampling
        - Pour signaux avec transitoires → SWT + Interpolation  
        - Pour très peu de points → Gaussian Process Regression
        - Pour usage général → Interpolation Spline
        """)

if __name__ == "__main__":
    main()
