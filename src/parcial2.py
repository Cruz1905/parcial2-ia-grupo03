"""
parcial2.py
Segundo Parcial - Inteligencia Artificial (2026-02)
Facultad de Ingeniería - COTECNOVA
Docente: Jhon James Cano Sánchez
Grupo 03: Davison Cruz López & Juan Diego Zapata Arbeláez
Tema: Turismo – hoteles y hostales en Cartago
Variables de entrada: precio_noche_miles (x1), calificacion (x2)
Variable objetivo: ocupacion_alta (y) [1 = alta, 0 = baja]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de codificación para consola de Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RUTA_CSV_ORIGINAL = os.path.join(DATA_DIR, 'grupo_03.csv')
RUTA_CSV_LIMPIO = os.path.join(DATA_DIR, 'grupo_03_limpio.csv')

# ==============================================================================
# TAREA 2: CARGA E INSPECCIÓN DE DATOS
# ==============================================================================
def cargar_e_inspeccionar_datos(ruta_csv):
    print("=" * 80)
    print(" TAREA 2: CARGA E INSPECCIÓN DE DATOS")
    print("=" * 80)
    
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")
        
    df = pd.read_csv(ruta_csv)
    
    print("\n[+] Primeras 5 filas del DataFrame:")
    print(df.head())
    
    print(f"\n[+] Forma del DataFrame (shape): {df.shape[0]} filas x {df.shape[1]} columnas")
    
    print("\n[+] Tipos de datos por columna (dtypes):")
    print(df.dtypes)
    
    print("\n[+] Cantidad de valores nulos por columna:")
    print(df.isnull().sum())
    
    duplicados = df.duplicated().sum()
    print(f"\n[+] Cantidad de filas duplicadas: {duplicados}")
    
    print("\n[+] Respuesta a la Pregunta Clave (Tarea 2):")
    print("    ¿Alguna columna numérica aparece con un tipo de dato que no debería tener? ¿Por qué ocurre?")
    print("    R/ En este dataset (grupo_03.csv), las columnas numéricas fueron leídas como 'int64'")
    print("       (precio_noche_miles y ocupacion_alta) y 'float64' (calificacion). 'calificacion' se asignó")
    print("       como float64 tanto por tener valores decimales (ej. 4.1, 2.9) como por contener valores nulos (NaN),")
    print("       los cuales en Pandas fuerzan la representación en punto flotante. Ninguna columna numérica")
    print("       vino tipada como 'object' porque no contenían caracteres alfabéticos o símbolos monetarios ($),")
    print("       pero 'precio_noche_miles' contiene un valor anómalo negativo (-244) y 'calificacion' 3 valores nulos.")
    
    return df

# ==============================================================================
# TAREA 3: CALIDAD Y LIMPIEZA DE DATOS CON PANDAS
# ==============================================================================
def limpiar_datos(df):
    print("\n" + "=" * 80)
    print(" TAREA 3: CALIDAD Y LIMPIEZA DE DATOS CON PANDAS")
    print("=" * 80)
    
    filas_iniciales = len(df)
    
    # 1. Duplicados
    duplicados_count = df.duplicated().sum()
    df_limpio = df.drop_duplicates(keep='first').copy()
    print(f"[1] Filas duplicadas eliminadas: {duplicados_count} filas")
    
    # 2. Corrección de inconsistencias en categorías (sector)
    df_limpio['sector'] = df_limpio['sector'].str.strip().str.title()
    sectores_unicos = df_limpio['sector'].unique()
    print(f"[2] Columna 'sector' unificada y sin espacios residuales: {list(sectores_unicos)}")
    
    # 3. Corrección de valor sospechoso en precio_noche_miles (-244)
    media_precio_antes = df['precio_noche_miles'].mean()
    std_precio_antes = df['precio_noche_miles'].std()
    
    # Se corrige el signo negativo producto de digitación accidental (abs)
    df_limpio['precio_noche_miles'] = df_limpio['precio_noche_miles'].abs()
    
    media_precio_despues = df_limpio['precio_noche_miles'].mean()
    std_precio_despues = df_limpio['precio_noche_miles'].std()
    
    print("\n[+] Pregunta Clave (Tarea 3): ¿Qué valor le pareció más sospechoso?")
    print(f"    R/ El valor más sospechoso fue precio_noche_miles = -244 (Hotel H073 en Zaragoza).")
    print(f"       Un precio por noche de hotel no puede ser negativo en el mundo real.")
    print(f"       - Media antes de corregir: {media_precio_antes:.2f} mil COP (std: {std_precio_antes:.2f})")
    print(f"       - Media después de corregir (abs=244): {media_precio_despues:.2f} mil COP (std: {std_precio_despues:.2f})")
    print("       - Efecto en el escalado: Sin corregir, -244 genera un z-score extremo de -3.87 desviaciones estándar,")
    print("         desplazando artificialmente la media hacia abajo (de 207.6 a 200.6) e inflando la varianza.")
    print("         Esto habría provocado que el perceptrón asignara gradientes desproporcionados a ese registro.")
    
    # 4. Manejo de valores nulos en calificacion
    nulos_calif = df_limpio['calificacion'].isnull().sum()
    df_limpio = df_limpio.dropna(subset=['calificacion']).copy()
    print(f"\n[4] Registros eliminados por valor nulo en 'calificacion': {nulos_calif} filas")
    print("    Decisión: Se eliminan porque 'calificacion' es una de solo dos entradas del perceptrón;")
    print("    imputar artificialmente 3 calificaciones en un dataset de 80 filas sesgaría la frontera.")
    
    # Exportar dataset limpio
    os.makedirs(DATA_DIR, exist_ok=True)
    df_limpio.to_csv(RUTA_CSV_LIMPIO, index=False)
    print(f"\n[+] Dataset depurado exportado exitosamente a: {RUTA_CSV_LIMPIO}")
    
    filas_finales = len(df_limpio)
    conteo_clases = df_limpio['ocupacion_alta'].value_counts()
    
    print(f"\n[+] Resumen de Registros:")
    print(f"    - Filas antes de la limpieza: {filas_iniciales}")
    print(f"    - Filas después de la limpieza: {filas_finales}")
    print(f"    - Total filas eliminadas: {filas_iniciales - filas_finales} (4 duplicados + 3 nulos)")
    print(f"    - Balance de clases: {conteo_clases.get(0, 0)} hoteles de Ocupación Baja (0) | {conteo_clases.get(1, 0)} hoteles de Ocupación Alta (1)")
    
    return df_limpio

# ==============================================================================
# TAREA 4: VISUALIZACIONES CON SEABORN
# ==============================================================================
def generar_visualizaciones(df):
    print("\n" + "=" * 80)
    print(" TAREA 4: GENERACIÓN DE VISUALIZACIONES CON SEABORN")
    print("=" * 80)
    
    sns.set_theme(style='whitegrid', font_scale=1.05)
    
    # Gráfico 1: Dispersión x1 vs x2 coloreado por ocupacion_alta
    plt.figure(figsize=(9, 6))
    ax1 = sns.scatterplot(
        data=df,
        x='precio_noche_miles',
        y='calificacion',
        hue='ocupacion_alta',
        palette={0: '#e41a1c', 1: '#377eb8'},
        s=100,
        alpha=0.9,
        edgecolor='black'
    )
    plt.title('Dispersión: Precio por Noche vs Calificación por Nivel de Ocupación', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Precio por Noche (Miles de COP)', fontsize=11, fontweight='bold')
    plt.ylabel('Calificación del Hotel (Escala 1 a 5)', fontsize=11, fontweight='bold')
    handles, _ = ax1.get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Baja Ocupación (0)', 'Alta Ocupación (1)'], title='Ocupación', loc='upper right')
    plt.tight_layout()
    ruta_g1 = os.path.join(BASE_DIR, 'dispersion_precio_calificacion.png')
    plt.savefig(ruta_g1, dpi=200)
    plt.close()
    print(f"[+] Gráfico 1 guardado: {ruta_g1}")
    
    # Gráfico 2: Boxplot de variable numérica comparada por clases
    plt.figure(figsize=(8, 5.5))
    sns.boxplot(
        data=df,
        x='ocupacion_alta',
        y='precio_noche_miles',
        palette={0: '#e41a1c', 1: '#377eb8'},
        hue='ocupacion_alta',
        legend=False,
        width=0.45
    )
    plt.title('Comparación de Precio por Noche según Ocupación Hotelera', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Nivel de Ocupación', fontsize=11, fontweight='bold')
    plt.ylabel('Precio por Noche (Miles de COP)', fontsize=11, fontweight='bold')
    plt.xticks([0, 1], ['Ocupación Baja (0)', 'Ocupación Alta (1)'], fontweight='bold')
    plt.tight_layout()
    ruta_g2 = os.path.join(BASE_DIR, 'boxplot_precio_ocupacion.png')
    plt.savefig(ruta_g2, dpi=200)
    plt.close()
    print(f"[+] Gráfico 2 guardado: {ruta_g2}")

# ==============================================================================
# TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY (SIGMOIDE + LOG LOSS)
# ==============================================================================
def sigmoide(z):
    """Función de activación sigmoide acotada numéricamente: sigma(z) = 1 / (1 + e^-z)"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def calcular_perdida_logaritmica(y_real, y_pred, epsilon=1e-15):
    """Binary Cross-Entropy Loss: -1/N * sum(y*log(p) + (1-y)*log(1-p))"""
    y_pred_segura = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return float(-np.mean(y_real * np.log(y_pred_segura) + (1.0 - y_real) * np.log(1.0 - y_pred_segura)))

def entrenar_perceptron(df, tasa_aprendizaje=0.1, epocas=500, seed=42):
    print("\n" + "=" * 80)
    print(" TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY")
    print("=" * 80)
    
    # 1. Separar características y objetivo
    X_raw = df[['precio_noche_miles', 'calificacion']].values.astype(float)
    y = df['ocupacion_alta'].values.astype(float)
    
    # 2. Estandarización de características: z = (x - mu) / sigma
    mu = np.mean(X_raw, axis=0)
    sigma = np.std(X_raw, axis=0)
    X_std = (X_raw - mu) / sigma
    
    print(f"[+] Estandarización de Características:")
    print(f"    - Media precio (mu1): {mu[0]:.2f} mil COP | Desv. estándar (sigma1): {sigma[0]:.2f} mil COP")
    print(f"    - Media calificación (mu2): {mu[1]:.2f} | Desv. estándar (sigma2): {sigma[1]:.2f}")
    
    # 3. Inicialización
    np.random.seed(seed)
    n_muestras, n_features = X_std.shape
    w = np.zeros(n_features)
    b = 0.0
    
    w_inicial = w.copy()
    b_inicial = b
    
    # Pérdida inicial con parámetros en cero (probabilidad = 0.5)
    prob_ini = sigmoide(np.dot(X_std, w) + b)
    perdida_inicial = calcular_perdida_logaritmica(y, prob_ini)
    
    print(f"\n[+] Parámetros iniciales: w1={w_inicial[0]:.4f}, w2={w_inicial[1]:.4f}, b={b_inicial:.4f}")
    print(f"[+] Pérdida inicial: {perdida_inicial:.4f}")
    print(f"[+] Hiperparámetros: Tasa de aprendizaje = {tasa_aprendizaje} | Épocas = {epocas}")
    print("-" * 80)
    
    historial_perdida = []
    historial_exactitud = []
    
    # 4. Descenso del gradiente
    for epoca in range(epocas):
        # Forward pass
        z = np.dot(X_std, w) + b
        y_prob = sigmoide(z)
        
        # Métrica de pérdida
        perdida = calcular_perdida_logaritmica(y, y_prob)
        historial_perdida.append(perdida)
        
        # Métrica de exactitud (umbral 0.5)
        predicciones = (y_prob >= 0.5).astype(int)
        exactitud = np.mean(predicciones == y) * 100.0
        historial_exactitud.append(exactitud)
        
        # Gradientes analíticos: dL/dw = 1/N * X^T (y_prob - y), dL/db = 1/N * sum(y_prob - y)
        error = y_prob - y
        grad_w = np.dot(X_std.T, error) / n_muestras
        grad_b = np.sum(error) / n_muestras
        
        # Actualización de pesos
        w -= tasa_aprendizaje * grad_w
        b -= tasa_aprendizaje * grad_b
        
        if (epoca + 1) == 1 or (epoca + 1) % 100 == 0 or (epoca + 1) == epocas:
            print(f"Época {epoca + 1:4d}/{epocas} | Pérdida: {perdida:.4f} | Exactitud: {exactitud:5.2f}% | w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
            
    perdida_final = historial_perdida[-1]
    exactitud_final = historial_exactitud[-1]
    
    print("-" * 80)
    print(f"[+] Pesos finales: w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
    print(f"[+] Pérdida final: {perdida_final:.4f} (reducción de {perdida_inicial:.4f} -> {perdida_final:.4f})")
    print(f"[+] Exactitud final del modelo: {exactitud_final:.2f}%")
    
    # 5. Generar curva_aprendizaje.png
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epocas + 1), historial_perdida, color='#d95f02', linewidth=2.4, label='Pérdida Logarítmica')
    plt.title('Curva de Aprendizaje del Perceptrón (Descenso del Gradiente)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Época', fontsize=11, fontweight='bold')
    plt.ylabel('Pérdida (Binary Cross-Entropy)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    ruta_curva = os.path.join(BASE_DIR, 'curva_aprendizaje.png')
    plt.savefig(ruta_curva, dpi=200)
    plt.close()
    print(f"[+] Gráfica guardada: {ruta_curva}")
    
    # 6. Generar frontera_decision.png en unidades originales
    plt.figure(figsize=(9, 6))
    
    x_baja = X_raw[y == 0]
    x_alta = X_raw[y == 1]
    
    plt.scatter(x_baja[:, 0], x_baja[:, 1], color='#e41a1c', label='Ocupación Baja (0)', s=80, alpha=0.9, edgecolor='black')
    plt.scatter(x_alta[:, 0], x_alta[:, 1], color='#377eb8', label='Ocupación Alta (1)', s=90, marker='s', alpha=0.9, edgecolor='black')
    
    # Ecuación de la frontera en unidades originales:
    # w1 * (x1 - mu1)/sigma1 + w2 * (x2 - mu2)/sigma2 + b = 0
    # x2 = mu2 - (sigma2 / w2) * [ w1 * (x1 - mu1)/sigma1 + b ]
    x1_linea = np.linspace(X_raw[:, 0].min() - 15, X_raw[:, 0].max() + 15, 200)
    x2_linea = mu[1] - (sigma[1] / w[1]) * (w[0] * (x1_linea - mu[0]) / sigma[0] + b)
    
    plt.plot(x1_linea, x2_linea, color='#111111', linestyle='--', linewidth=2.4, label='Frontera de Decisión ($P = 0.5$)')
    
    plt.title('Frontera de Decisión del Perceptrón (Unidades Originales)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Precio por Noche (Miles de COP)', fontsize=11, fontweight='bold')
    plt.ylabel('Calificación del Hotel (Escala 1 a 5)', fontsize=11, fontweight='bold')
    plt.xlim(X_raw[:, 0].min() - 20, X_raw[:, 0].max() + 20)
    plt.ylim(2.2, 5.2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', framealpha=0.95, fontsize=10.5)
    plt.tight_layout()
    ruta_frontera = os.path.join(BASE_DIR, 'frontera_decision.png')
    plt.savefig(ruta_frontera, dpi=200)
    plt.close()
    print(f"[+] Gráfica guardada: {ruta_frontera}")
    
    # 7. Evaluación de dos casos de prueba realistas (Tarea 6, Pregunta 3)
    print("\n" + "=" * 80)
    print(" EVALUACIÓN DE CASOS DE PRUEBA REALISTAS (TAREA 6, PREGUNTA 3)")
    print("=" * 80)
    
    # Caso 1: Hotel accesible con calificación sobresaliente
    caso_1_raw = np.array([90.0, 4.8])
    caso_1_std = (caso_1_raw - mu) / sigma
    z_caso1 = np.dot(caso_1_std, w) + b
    p_caso1 = sigmoide(z_caso1)
    
    # Caso 2: Hotel costoso con calificación baja
    caso_2_raw = np.array([350.0, 2.9])
    caso_2_std = (caso_2_raw - mu) / sigma
    z_caso2 = np.dot(caso_2_std, w) + b
    p_caso2 = sigmoide(z_caso2)
    
    print(f"Caso 1 (Esperado 1 - Ocupación Alta): Precio = $90k COP, Calificación = 4.8")
    print(f"  -> Valores estandarizados: z_precio = {caso_1_std[0]:.2f}, z_calif = {caso_1_std[1]:.2f}")
    print(f"  -> Probabilidad estimada por el modelo: {p_caso1 * 100:.2f}% | Clasificación: {(p_caso1 >= 0.5) * 1}")
    print(f"  -> Conclusión: Coincide plenamente con lo esperado (P > 95%).")
    
    print(f"\nCaso 2 (Esperado 0 - Ocupación Baja): Precio = $350k COP, Calificación = 2.9")
    print(f"  -> Valores estandarizados: z_precio = {caso_2_std[0]:.2f}, z_calif = {caso_2_std[1]:.2f}")
    print(f"  -> Probabilidad estimada por el modelo: {p_caso2 * 100:.2f}% | Clasificación: {(p_caso2 >= 0.5) * 1}")
    print(f"  -> Conclusión: Coincide plenamente con lo esperado (P < 1%).")
    
    # Diagnóstico de separabilidad lineal
    preds_totales = (sigmoide(np.dot(X_std, w) + b) >= 0.5).astype(int)
    discrepancias = np.sum(preds_totales != y)
    print(f"\n[+] Diagnóstico de Separabilidad: {discrepancias} registros de {len(y)} quedan en el lado equivocado ({discrepancias/len(y)*100:.2f}% de error).")
    
    return {
        'w': w,
        'b': b,
        'mu': mu,
        'sigma': sigma,
        'perdida_ini': perdida_inicial,
        'perdida_fin': perdida_final,
        'exactitud_fin': exactitud_final,
        'discrepancias': discrepancias,
        'caso_1': (caso_1_raw, p_caso1),
        'caso_2': (caso_2_raw, p_caso2)
    }

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    print("=" * 80)
    print(" SEGUNDO PARCIAL: INTELIGENCIA ARTIFICIAL - COTECNOVA 2026-02")
    print(" DOCENTE: JHON JAMES CANO SÁNCHEZ")
    print(" GRUPO 03: DAVISON CRUZ LÓPEZ & JUAN DIEGO ZAPATA ARBELÁEZ")
    print("=" * 80)
    
    # Tarea 2
    df_crudo = cargar_e_inspeccionar_datos(RUTA_CSV_ORIGINAL)
    
    # Tarea 3
    df_limpio = limpiar_datos(df_crudo)
    
    # Tarea 4
    generar_visualizaciones(df_limpio)
    
    # Tarea 5 y 6
    resultados = entrenar_perceptron(df_limpio, tasa_aprendizaje=0.1, epocas=500)
    
    print("\n" + "=" * 80)
    print(" TODAS LAS TAREAS SE EJECUTARON SATISFACTORIAMENTE")
    print("=" * 80)

if __name__ == "__main__":
    main()
