"""
Test main pentru plot_disability_annotations
Foloseste date sintetice simple (linii drepte) pentru a verifica orientarea axelor.

Structura datelor reflecta exact ce produce main-ul real:
- all_valid_scenarios_data[scenario_name] = {"data": scenario_data, "person": person_id}
- scenario_data are coloanele: [X, Z, Y, rot_x, rot_z, rot_y, fwd_x, fwd_z, fwd_y, timestamp]
  (axele sunt deja reordonate [0,2,1] ca in main)

Trasee de test:
  - Person_1_LINE_X  : miscarea doar pe X (stanga/dreapta) → linie pe axa X
  - Person_2_LINE_Y  : miscarea doar pe Y (inaltime cap)   → linie pe axa Y (verticala)
  - Person_3_LINE_Z  : miscarea doar pe Z (inainte/inapoi) → linie pe axa Z (adancime)
  - Person_4_DIAGONAL: miscare diagonala pe toate 3 axele  → linie diagonala

Daca graficele arata linii drepte pe axele corecte → orientarea e corecta.
"""

import sys
import os
import numpy as np

# Adauga root-ul proiectului in path
# Ajusteaza acest path la structura ta de directoare
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from visualization.plotter_3d import plot_disability_annotations

# ─────────────────────────────────────────────
# Parametri date sintetice
# ─────────────────────────────────────────────
N_POINTS = 100       # numarul de puncte pe traseu
EXTRA_COLS = 7       # coloane extra: rot(3) + fwd(3) + timestamp(1)

def make_scenario_data(x_vals, y_vals, z_vals):
    """
    Construieste un scenario_data array de forma (N, 10):
    [X, Z, Y, rot_x, rot_z, rot_y, fwd_x, fwd_z, fwd_y, timestamp]

    Nota: in main, axele sunt reordonate la [X, Z, Y] inainte de a fi
    salvate in scenario_data, deci col 0=X, col 1=Z, col 2=Y.
    Celelalte coloane sunt zero (nu conteaza pentru plot).
    """
    positions = np.column_stack([x_vals, z_vals, y_vals])   # [X, Z, Y] — ordinea din main
    extra = np.zeros((N_POINTS, EXTRA_COLS))
    return np.hstack([positions, extra])


# ─────────────────────────────────────────────
# Generare date sintetice
# ─────────────────────────────────────────────

# Traseu 1: linie dreapta pe X (de la 0 la 10), Y si Z fixe
line_x = make_scenario_data(
    x_vals=np.linspace(0, 10, N_POINTS),
    y_vals=np.ones(N_POINTS) * 1.7,   # inaltime cap constanta ~1.7m
    z_vals=np.zeros(N_POINTS)
)

# Traseu 2: linie dreapta pe Y (inaltimea capului variaza 1.5 → 2.0m), X si Z fixe
line_y = make_scenario_data(
    x_vals=np.zeros(N_POINTS),
    y_vals=np.linspace(1.5, 2.0, N_POINTS),
    z_vals=np.zeros(N_POINTS)
)

# Traseu 3: linie dreapta pe Z (inainte/inapoi 0 → 10m), X si Y fixe
line_z = make_scenario_data(
    x_vals=np.zeros(N_POINTS),
    y_vals=np.ones(N_POINTS) * 1.7,
    z_vals=np.linspace(0, 10, N_POINTS)
)

# Traseu 4: diagonala pe toate 3 axele
diagonal = make_scenario_data(
    x_vals=np.linspace(0, 5, N_POINTS),
    y_vals=np.linspace(1.5, 2.0, N_POINTS),
    z_vals=np.linspace(0, 5, N_POINTS)
)

# ─────────────────────────────────────────────
# Construire all_data (structura identica cu main-ul real)
# ─────────────────────────────────────────────
all_data = {
    "Person_1_LINE_X":   {"data": line_x,    "person": "Person_1"},
    "Person_2_LINE_Y":   {"data": line_y,    "person": "Person_2"},
    "Person_3_LINE_Z":   {"data": line_z,    "person": "Person_3"},
    "Person_4_DIAGONAL": {"data": diagonal,  "person": "Person_4"},
}

# ─────────────────────────────────────────────
# Construire disability_likelihood (un status per scenariu)
# ─────────────────────────────────────────────
disability_likelihood = {
    "Person_1_LINE_X":   {"status": "LOW",    "score": 0.1},
    "Person_2_LINE_Y":   {"status": "MEDIUM", "score": 0.5},
    "Person_3_LINE_Z":   {"status": "HIGH",   "score": 0.9},
    "Person_4_DIAGONAL": {"status": "LOW",    "score": 0.2},
}

# ─────────────────────────────────────────────
# Rulare test
# ─────────────────────────────────────────────
print("=" * 60)
print("TEST: plot_disability_annotations cu date sintetice")
print("=" * 60)
print()
print("Trasee generate:")
print("  Person_1_LINE_X   → linie dreapta pe X  (verde  / LOW)")
print("  Person_2_LINE_Y   → linie dreapta pe Y  (portoc / MEDIUM)")
print("  Person_3_LINE_Z   → linie dreapta pe Z  (rosu   / HIGH)")
print("  Person_4_DIAGONAL → diagonala 3D         (verde  / LOW)")
print()
print("Validare asteptata:")
print("  - Person_1_LINE_X  : linie paralela cu axa X, Y si Z = 0")
print("  - Person_2_LINE_Y  : linie scurta verticala pe Y (1.5 → 2.0)")
print("  - Person_3_LINE_Z  : linie paralela cu axa Z, X si Y = 0")
print("  - Person_4_DIAGONAL: linie oblica in toate 3 directiile")
print()

plot_disability_annotations(all_data, disability_likelihood)

print()
print("Test finalizat. Verifica ploturile generate in:")
print("  data/output/disability_3d/<scenario_name>/")
