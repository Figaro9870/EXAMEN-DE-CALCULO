"""
EXAMEN DE CÁLCULO II — Solución completa en Python
==================================================

Requisitos (instalar una sola vez en la terminal de VS Code):
    pip install -r requirements.txt

Ejecutar:
    python examen_calculo_ii.py            -> muestra las gráficas en ventanas
    python examen_calculo_ii.py --no-show  -> solo guarda las imágenes PNG

Las gráficas se guardan en la carpeta "graficas/" junto a este archivo.
"""

import os
import sys

import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")  # sin ventanas, solo guardar imágenes

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficas")
os.makedirs(CARPETA, exist_ok=True)

x, y = sp.symbols("x y", real=True)


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA, nombre)
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print(f"  -> Gráfica guardada en: {ruta}")


def ejes_cartesianos(ax):
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)


# ----------------------------------------------------------------------
# PREGUNTA 1: y = x^3 - 3x   y   y = x^2 - 4   en x ∈ [-3, 3]
# ----------------------------------------------------------------------
def pregunta_1():
    titulo("PREGUNTA 1. Funciones polinómicas en 2D")
    f = x**3 - 3 * x
    g = x**2 - 4

    # 2. Raíces
    raices_f = sp.solve(sp.Eq(f, 0), x)
    raices_g = sp.solve(sp.Eq(g, 0), x)
    print("2) Raíces de y = x³ - 3x :", raices_f, "≈", [round(float(r), 4) for r in raices_f])
    print("   Raíces de y = x² - 4  :", raices_g)

    # 3. Intersecciones: x³ - 3x = x² - 4  ->  x³ - x² - 3x + 4 = 0
    ecuacion = sp.expand(f - g)
    xs_int = [complex(r).real for r in sp.Poly(ecuacion, x).nroots() if abs(complex(r).imag) < 1e-12]
    puntos_int = [(xi, float(g.subs(x, xi))) for xi in xs_int]
    print(f"3) Intersección: {ecuacion} = 0")
    for xi, yi in puntos_int:
        print(f"   Punto de intersección ≈ ({xi:.4f}, {yi:.4f})")
    print("   (Las otras dos soluciones son complejas: solo hay UNA intersección real.)")

    # 4. Crecimiento / decrecimiento de la cúbica
    df = sp.diff(f, x)
    criticos = sp.solve(sp.Eq(df, 0), x)
    print(f"4) f'(x) = {df} = 0  ->  x = {criticos}")
    print("   f'(x) > 0 en [-3, -1) ∪ (1, 3]  -> CRECIENTE")
    print("   f'(x) < 0 en (-1, 1)             -> DECRECIENTE")
    for c in criticos:
        tipo = "máximo local" if sp.diff(f, x, 2).subs(x, c) < 0 else "mínimo local"
        print(f"   x = {c}: {tipo} en ({c}, {f.subs(x, c)})")

    print("5) Comparación:")
    print("   - La cúbica es impar (simétrica respecto al origen), tiene 3 raíces,")
    print("     un máximo local (-1, 2) y un mínimo local (1, -2); va de -∞ a +∞.")
    print("   - La parábola es par (simétrica respecto al eje y), tiene 2 raíces,")
    print("     un mínimo absoluto en (0, -4) y abre hacia arriba.")
    print("   - Para |x| grande la cúbica crece más rápido que la parábola.")

    # 1. Gráfica
    X = np.linspace(-3, 3, 600)
    F = sp.lambdify(x, f, "numpy")
    G = sp.lambdify(x, g, "numpy")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(X, F(X), label=r"$y = x^3 - 3x$", color="tab:blue", linewidth=2)
    ax.plot(X, G(X), label=r"$y = x^2 - 4$", color="tab:orange", linewidth=2)

    for r in raices_f:
        ax.plot(float(r), 0, "o", color="tab:blue", markersize=8)
        ax.annotate(f"({float(r):.2f}, 0)", (float(r), 0), textcoords="offset points",
                    xytext=(5, 10), color="tab:blue")
    for r in raices_g:
        ax.plot(float(r), 0, "s", color="tab:orange", markersize=8)
        ax.annotate(f"({float(r):.0f}, 0)", (float(r), 0), textcoords="offset points",
                    xytext=(-40 if r < 0 else 5, 12 if r < 0 else -15), color="tab:orange")
    for xi, yi in puntos_int:
        ax.plot(xi, yi, "*", color="red", markersize=15, label="Intersección")
        ax.annotate(f"({xi:.3f}, {yi:.3f})", (xi, yi), textcoords="offset points",
                    xytext=(10, -5), color="red")
    for c in criticos:
        ax.plot(float(c), float(f.subs(x, c)), "^", color="green", markersize=8)
    ax.plot([], [], "^", color="green", label="Máx./mín. locales de la cúbica")

    ejes_cartesianos(ax)
    ax.set_xlim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Pregunta 1: raíces e intersecciones")
    ax.legend()
    guardar(fig, "pregunta1.png")


# ----------------------------------------------------------------------
# PREGUNTA 2: y = 2 sin(x)   y   y = e^(-0.2x) cos(2x)   en x ∈ [0, 4π]
# ----------------------------------------------------------------------
def pregunta_2():
    titulo("PREGUNTA 2. Funciones trigonométricas y exponenciales en 2D")
    print("2) y = 2 sin(x):  Amplitud A = |2| = 2,   Período T = 2π/1 = 2π ≈ 6.2832")
    print("3) y = e^(-0.2x) cos(2x):")
    print("   - cos(2x) oscila con período T = 2π/2 = π ≈ 3.1416.")
    print("   - El factor e^(-0.2x) es la ENVOLVENTE: la amplitud decrece")
    print("     exponencialmente de 1 (x=0) a e^(-0.8π) ≈ %.4f (x=4π)." % np.exp(-0.8 * np.pi))
    print("   - La curva queda siempre entre -e^(-0.2x) y +e^(-0.2x).")
    print("4) Comparación: 2 sin(x) oscila con amplitud CONSTANTE (2) y período 2π;")
    print("   la segunda oscila el DOBLE de rápido (período π) y su amplitud se AMORTIGUA.")
    print("5) Aplicación: un sistema masa-resorte con fricción (o un amortiguador de auto,")
    print("   un circuito RLC) se describe con y(t) = A e^(-bt) cos(ωt): oscila pero su")
    print("   energía se disipa. 2 sin(x) modela una onda ideal sin pérdidas (MAS).")

    X = np.linspace(0, 4 * np.pi, 1000)
    Y1 = 2 * np.sin(X)
    Y2 = np.exp(-0.2 * X) * np.cos(2 * X)
    env = np.exp(-0.2 * X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, Y1, label=r"$y = 2\sin(x)$  (A = 2, T = 2$\pi$)", color="tab:blue", linewidth=2)
    ax.plot(X, Y2, label=r"$y = e^{-0.2x}\cos(2x)$  (T = $\pi$)", color="tab:red", linewidth=2)
    ax.plot(X, env, "--", color="gray", label=r"Envolvente $\pm e^{-0.2x}$")
    ax.plot(X, -env, "--", color="gray")
    ejes_cartesianos(ax)
    ax.set_xticks(np.arange(0, 4 * np.pi + 0.01, np.pi / 2))
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π", "5π/2", "3π", "7π/2", "4π"])
    ax.set_xlim(0, 4 * np.pi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Pregunta 2: oscilación pura vs. oscilación amortiguada")
    ax.legend(loc="upper right")
    guardar(fig, "pregunta2.png")


# ----------------------------------------------------------------------
# PREGUNTA 3: z = x^2 + y^2 - 4   (paraboloide)
# ----------------------------------------------------------------------
def pregunta_3():
    titulo("PREGUNTA 3. Superficie parabólica z = x² + y² - 4")
    z = x**2 + y**2 - 4
    criticos = sp.solve([sp.diff(z, x), sp.diff(z, y)], [x, y], dict=True)
    for c in criticos:
        print(f"3) Punto crítico: ∂z/∂x = 2x = 0, ∂z/∂y = 2y = 0 -> (x, y) = ({c[x]}, {c[y]})")
        print(f"   z mínimo = {z.subs(c)}  -> Punto mínimo (0, 0, -4)")
    print("   (Hessiano = [[2,0],[0,2]], definido positivo -> es mínimo absoluto.)")
    print("4) Intersección con z = 0:  x² + y² = 4  -> circunferencia de radio 2 centrada en el origen.")
    print("   Sección y = 0: z = x² - 4 -> corta en x = ±2.   Sección x = 0: z = y² - 4 -> corta en y = ±2.")
    print("5) Es un PARABOLOIDE CIRCULAR (de revolución) que abre hacia arriba, con")
    print("   vértice en (0, 0, -4). Sus cortes horizontales son circunferencias y sus")
    print("   cortes verticales por el eje z son parábolas.")

    v = np.linspace(-3, 3, 120)
    XX, YY = np.meshgrid(v, v)
    ZZ = XX**2 + YY**2 - 4

    fig = plt.figure(figsize=(15, 5))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    ax3.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.85)
    t = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(2 * np.cos(t), 2 * np.sin(t), 0, color="red", linewidth=3, label="z = 0: x²+y²=4")
    ax3.scatter(0, 0, -4, color="black", s=60, label="Mínimo (0,0,-4)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title(r"$z = x^2 + y^2 - 4$")
    ax3.legend(loc="upper left", fontsize=8)

    # Secciones: ejes horizontal "x" y vertical "z", como pide el entregable
    secciones = [("y = 0", r"$z = x^2 - 4$"), ("x = 0", r"$z = y^2 - 4$ (misma forma)")]
    for i, (corte, eq) in enumerate(secciones, start=2):
        ax = fig.add_subplot(1, 3, i)
        ax.plot(v, v**2 - 4, color="tab:blue" if i == 2 else "tab:green", linewidth=2, label=eq)
        ax.plot([-2, 2], [0, 0], "ro", label="Corte con z = 0 (±2)")
        ax.plot(0, -4, "ko", label="Mínimo z = -4")
        ejes_cartesianos(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"Sección {corte}")
        ax.legend(fontsize=8)
    fig.suptitle("Pregunta 3: paraboloide circular")
    guardar(fig, "pregunta3.png")


# ----------------------------------------------------------------------
# PREGUNTA 4: z = sin(x) cos(y)   en x, y ∈ [-2π, 2π]
# ----------------------------------------------------------------------
def pregunta_4():
    titulo("PREGUNTA 4. Superficie ondulatoria z = sin(x)cos(y)")
    print("4) Como -1 ≤ sin(x) ≤ 1 y -1 ≤ cos(y) ≤ 1:")
    print("   z máximo = 1  (p. ej. x = π/2, y = 0  ó  x = -π/2, y = π)")
    print("   z mínimo = -1 (p. ej. x = -π/2, y = 0 ó  x = π/2,  y = π)")

    v = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    XX, YY = np.meshgrid(v, v)
    ZZ = np.sin(XX) * np.cos(YY)
    print(f"   Verificación numérica en la malla: max = {ZZ.max():.4f}, min = {ZZ.min():.4f}")
    print("5) Sección y = 0:   z = sin(x)·cos(0) = sin(x) -> onda senoidal de amplitud 1.")
    print("   Sección y = π/2: z = sin(x)·cos(π/2) = 0   -> recta z = 0 (la onda desaparece).")
    print("   cos(y) actúa como 'amplitud' de la onda en x: vale 1 en y = 0 y 0 en y = π/2.")

    fig = plt.figure(figsize=(15, 5))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    ax3.plot_surface(XX, YY, ZZ, cmap="coolwarm", alpha=0.9)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title(r"$z = \sin(x)\cos(y)$")

    ticks = np.arange(-2 * np.pi, 2 * np.pi + 0.01, np.pi)
    etiquetas = ["-2π", "-π", "0", "π", "2π"]
    for i, (yc, nombre) in enumerate([(0.0, "0"), (np.pi / 2, "π/2")], start=2):
        ax = fig.add_subplot(1, 3, i)
        zc = np.sin(v) * np.cos(yc)
        zc[np.abs(zc) < 1e-12] = 0.0  # cos(π/2) ≈ 6e-17 numéricamente
        eq = r"$z = \sin(x)$" if yc == 0 else r"$z = \sin(x)\cos(\pi/2) = 0$"
        ax.plot(v, zc, color="tab:purple" if i == 2 else "tab:orange", linewidth=2, label=eq)
        ejes_cartesianos(ax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(etiquetas)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"Sección y = {nombre}")
        ax.legend()
    fig.suptitle("Pregunta 4: superficie ondulatoria")
    guardar(fig, "pregunta4.png")


# ----------------------------------------------------------------------
# PREGUNTA 5 (Comodín): z = x^2 - y^2   (silla de montar)
# ----------------------------------------------------------------------
def pregunta_5():
    titulo("PREGUNTA 5. Comodín: silla de montar z = x² - y²")
    z = x**2 - y**2
    crit = sp.solve([sp.diff(z, x), sp.diff(z, y)], [x, y], dict=True)[0]
    H = sp.hessian(z, (x, y))
    print(f"- Punto crítico: ({crit[x]}, {crit[y]}, {z.subs(crit)})")
    print(f"- Hessiano = {H.tolist()}, det = {H.det()} < 0  -> PUNTO DE SILLA (ni máx. ni mín.)")
    print("- Sección y = 0: z = x²  -> parábola hacia ARRIBA (mínimo en el origen).")
    print("- Sección x = 0: z = -y² -> parábola hacia ABAJO (máximo en el origen).")
    print("- Intersección con z = 0: x² = y² -> dos rectas y = x, y = -x.")
    print("- Forma: PARABOLOIDE HIPERBÓLICO (silla de montar); los cortes horizontales")
    print("  z = c ≠ 0 son hipérbolas.")

    v = np.linspace(-3, 3, 120)
    XX, YY = np.meshgrid(v, v)
    ZZ = XX**2 - YY**2

    fig = plt.figure(figsize=(15, 5))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    ax3.plot_surface(XX, YY, ZZ, cmap="plasma", alpha=0.85)
    ax3.plot(v, v, 0 * v, color="cyan", linewidth=3, label="z = 0: y = x")
    ax3.plot(v, -v, 0 * v, color="lime", linewidth=3, label="z = 0: y = -x")
    ax3.scatter(0, 0, 0, color="black", s=60, label="Punto de silla (0,0,0)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title(r"$z = x^2 - y^2$")
    ax3.legend(loc="upper left", fontsize=8)

    for i, (corte, zc, eq, col) in enumerate(
        [("y = 0", v**2, r"$z = x^2$", "tab:blue"), ("x = 0", -(v**2), r"$z = -y^2$", "tab:red")],
        start=2,
    ):
        ax = fig.add_subplot(1, 3, i)
        ax.plot(v, zc, color=col, linewidth=2, label=eq)
        ax.plot(0, 0, "ko", label="Punto de silla")
        ejes_cartesianos(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"Sección {corte}")
        ax.legend()
    fig.suptitle("Pregunta 5: paraboloide hiperbólico (silla de montar)")
    guardar(fig, "pregunta5.png")


if __name__ == "__main__":
    pregunta_1()
    pregunta_2()
    pregunta_3()
    pregunta_4()
    pregunta_5()
    print("\nListo. Todas las gráficas están en:", CARPETA)
    if "--no-show" not in sys.argv:
        plt.show()
