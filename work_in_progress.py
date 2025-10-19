import numpy as np
import pandas as pd

# =========================================================
# Parametri fisici globali
# =========================================================
rho = 1.2      # densità aria (kg/m^3)
c = 340        # velocità suono (m/s)
dt = 60        # intervallo temporale in secondi
scale_factor = 1e-6  # fattore di scala empirico per variazione pressione


# =========================================================
# Funzione per inizializzare i punti meteo
# =========================================================
def init_points():
    """
    Inizializza i punti A, B, C, D con dati di pressione, vento e temperatura.
    Ritorna un DataFrame Pandas.
    """
    points = pd.DataFrame({
        'Nome': ['A', 'B', 'C', 'D'],
        'x': [0, 10, 0, 10],
        'y': [0, 0, 10, 10],
        'Pressione': [1012.0, 1011.5, 1012.3, 1011.8],
        'Velocità': [8.0, 9.0, 7.0, 8.5],
        'Direzione': [90, 95, 85, 92],
        'Temperatura': [18.5, 19.0, 18.0, 19.2]
    })

    # Conversione in componenti u,v
    points['u'] = points['Velocità'] * np.sin(np.deg2rad(points['Direzione']))
    points['v'] = points['Velocità'] * np.cos(np.deg2rad(points['Direzione']))
    return points


# =========================================================
# Funzione per aggiungere un punto di perturbazione
# =========================================================
def add_perturbation_point(points, x_pert=5, y_pert=0, vel_pert=3.0, dir_pert=90):
    """
    Aggiunge un punto X con velocità ridotta e direzione impostata.
    """
    pert_point = {
        'Nome': 'X',
        'x': x_pert,
        'y': y_pert,
        'Pressione': np.nan,
        'Velocità': vel_pert,
        'Direzione': dir_pert,
        'Temperatura': np.nan,
        'u': vel_pert * np.sin(np.deg2rad(dir_pert)),
        'v': vel_pert * np.cos(np.deg2rad(dir_pert))
    }
    points = pd.concat([points, pd.DataFrame([pert_point])], ignore_index=True)
    return points


# =========================================================
# Funzione per calcolare divergenza in un punto (tipo X)
# =========================================================
def compute_divergence_fd(points_df, x0, y0, R=20):
    """
    Calcola la divergenza locale del vento attorno a (x0, y0).
    Usa una stima semplificata a differenze finite.
    """
    neighbors = points_df.copy()
    dx = neighbors['x'] - x0
    dy = neighbors['y'] - y0
    dist = np.sqrt(dx**2 + dy**2)
    mask = (dist > 0) & (dist < R)
    neighbors = neighbors[mask]
    if len(neighbors) == 0:
        return 0.0

    # Valori al punto centrale (presupposto: è X)
    u0 = points_df.loc[points_df['Nome'] == 'X', 'u'].values[0]
    v0 = points_df.loc[points_df['Nome'] == 'X', 'v'].values[0]

    du = neighbors['u'] - u0
    dv = neighbors['v'] - v0
    div = np.sum(du * (neighbors['x'] - x0) / (dist**2) +
                 dv * (neighbors['y'] - y0) / (dist**2))
    div = div / len(neighbors)
    return div


# =========================================================
# Funzione per calcolare la variazione di pressione da divergenza
# =========================================================
def compute_pressure_change(divergence):
    """
    Calcola la variazione di pressione (in hPa) a partire dalla divergenza.
    """
    deltaP_pa = -rho * c**2 * divergence * dt * scale_factor
    deltaP_hpa = deltaP_pa / 100.0
    return deltaP_hpa


# =========================================================
# Funzione per aggiornare le pressioni nei punti
# =========================================================
def update_pressures(points, x_pert, y_pert, deltaP_X_hpa):
    """
    Aggiorna la pressione nei punti in base alla perturbazione a X.
    """
    new_pressures = []
    for _, p in points.iterrows():
        if p['Nome'] == 'X':
            # media delle pressioni esistenti come base
            P0 = np.mean(points.loc[points['Nome'] != 'X', 'Pressione'])
            newP = P0 + deltaP_X_hpa
        else:
            dist = np.sqrt((p['x'] - x_pert)**2 + (p['y'] - y_pert)**2)
            influence = np.exp(-dist / 10.0)  # attenuazione
            newP = p['Pressione'] + deltaP_X_hpa * influence
        new_pressures.append(newP)

    points['Nuova_Pressione'] = new_pressures
    return points


# =========================================================
# Esempio di esecuzione (COMMENTATO)
# =========================================================
# if __name__ == "__main__":
#     pts = init_points()
#     pts = add_perturbation_point(pts, x_pert=5, y_pert=0, vel_pert=3.0)
#     div_X = compute_divergence_fd(pts, 5, 0, R=20)
#     dP = compute_pressure_change(div_X)
#     pts = update_pressures(pts, 5, 0, dP)
#     print(pts[['Nome','x','y','Pressione','Nuova_Pressione','Velocità','Direzione']].round(3))
