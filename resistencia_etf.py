#!/usr/bin/env python3
"""
Detector de aproximación a zonas de resistencia para un ETF (p. ej. SPY).

Informativo: no genera órdenes de compra ni de venta.

Entrada: DataFrame diario con columnas high, low, close y, opcionalmente, volume.
El índice debe ser de fechas en orden ascendente. Los datos diarios tienen rezago
de una sesión; indícalo siempre en el informe.

Lógica:
  1. ATR(14) como unidad de medida de volatilidad.
  2. Pivotes altos locales (máximos con k barras a cada lado) en una ventana reciente.
  3. Agrupación de pivotes cercanos (tolerancia en ATR) en zonas de resistencia.
  4. Se elige la zona por encima del precio más cercana.
  5. Se mide si el precio se acerca: distancia en ATR y pendiente de esa distancia.
  6. Confluencia: cada señal independiente suma un punto (máximo 4).

Uso:
    python resistencia_etf.py datos.csv
    python resistencia_etf.py            # sin argumento: datos simulados de prueba

CSV esperado: columnas high, low, close, volume (volume opcional) y, si hay,
una columna date/fecha/timestamp que se usa como índice.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Zona:
    precio_bajo: float
    precio_alto: float
    toques: int
    ultimo_toque: int  # posición de la barra dentro del DataFrame

    @property
    def centro(self) -> float:
        return (self.precio_bajo + self.precio_alto) / 2


@dataclass
class Diagnostico:
    precio_actual: float
    atr: float
    zona: Optional[Zona]
    distancia_atr: Optional[float]
    acercandose: bool
    rvol: Optional[float]
    puntaje: int
    estado: str
    confianza: str
    motivos: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Indicadores y detección de niveles
# ---------------------------------------------------------------------------

def calcular_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    cierre_prev = df["close"].shift(1)
    rango = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - cierre_prev).abs(),
            (df["low"] - cierre_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rango.rolling(n).mean()


def pivotes_altos(high: pd.Series, k: int = 3) -> list[int]:
    """Posiciones de máximos locales confirmados (k barras a cada lado).

    Los últimos k pivotes no se pueden confirmar todavía; se ignoran.
    """
    v = high.to_numpy()
    resultado = []
    for i in range(k, len(v) - k):
        ventana = v[i - k : i + k + 1]
        if v[i] == ventana.max() and (ventana == v[i]).sum() == 1:
            resultado.append(i)
    return resultado


def agrupar_en_zonas(precios: np.ndarray, indices: list[int], tolerancia: float) -> list[Zona]:
    """Agrupa pivotes cuyo precio está a menos de `tolerancia` del promedio del grupo."""
    pares = sorted(zip(precios, indices))
    grupos: list[list[tuple[float, int]]] = []
    grupo: list[tuple[float, int]] = []
    for p, i in pares:
        if grupo and p - np.mean([g[0] for g in grupo]) > tolerancia:
            grupos.append(grupo)
            grupo = []
        grupo.append((p, i))
    if grupo:
        grupos.append(grupo)

    zonas = []
    for g in grupos:
        ps = [p for p, _ in g]
        zonas.append(Zona(min(ps), max(ps), len(g), max(i for _, i in g)))
    return zonas


# ---------------------------------------------------------------------------
# Análisis principal
# ---------------------------------------------------------------------------

def analizar(
    df: pd.DataFrame,
    k: int = 3,
    ventana_pivotes: int = 120,
    tol_atr: float = 0.5,
    min_toques: int = 2,
    umbral_cerca_atr: float = 1.0,
    ventana_tendencia: int = 5,
    vol_ventana: int = 20,
) -> Diagnostico:
    faltantes = {"high", "low", "close"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    df = df.copy()
    df["atr"] = calcular_atr(df)
    precio = float(df["close"].iloc[-1])
    a = float(df["atr"].iloc[-1])
    if pd.isna(a):
        raise ValueError("Historial insuficiente para calcular el ATR (se necesitan >= 15 barras).")

    motivos: list[str] = []

    # Volumen relativo frente a la mediana de las últimas sesiones
    rvol = None
    if "volume" in df.columns and df["volume"].notna().any():
        mediana = df["volume"].rolling(vol_ventana).median().iloc[-1]
        if pd.notna(mediana) and mediana > 0:
            rvol = float(df["volume"].iloc[-1] / mediana)
    else:
        motivos.append("Volumen no disponible: la señal de volumen no se evalúa.")

    # Zonas de resistencia
    inicio = max(0, len(df) - ventana_pivotes)
    sub = df.iloc[inicio:]
    idx = pivotes_altos(sub["high"], k)
    precios = sub["high"].to_numpy()[idx] if idx else np.array([])
    indices = [i + inicio for i in idx]
    zonas = [
        z
        for z in agrupar_en_zonas(precios, indices, tol_atr * a)
        if z.toques >= min_toques
    ]

    candidatas = [z for z in zonas if z.centro > precio]
    if not candidatas:
        motivos.append("No hay zonas de resistencia con toques suficientes por encima del precio.")
        return Diagnostico(precio, a, None, None, False, rvol, 0, "sin zona", "no aplica", motivos)

    zona = min(candidatas, key=lambda z: z.precio_bajo - precio)
    distancia = (zona.precio_bajo - precio) / a

    # Pendiente de la distancia en las últimas barras (zona fija)
    ult = df.iloc[-ventana_tendencia:]
    dist_hist = ((zona.precio_bajo - ult["close"]) / ult["atr"]).dropna()
    acercandose = False
    if len(dist_hist) >= 3:
        pendiente = np.polyfit(np.arange(len(dist_hist)), dist_hist.to_numpy(), 1)[0]
        acercandose = bool(pendiente < 0)

    cerca = distancia <= umbral_cerca_atr

    # Confluencia de señales independientes
    puntaje = 0
    if cerca:
        puntaje += 1
        motivos.append(f"Precio a {distancia:.2f} ATR del borde inferior de la zona.")
    if acercandose:
        puntaje += 1
        motivos.append("La distancia a la zona se reduce en las últimas sesiones.")
    if zona.toques >= 3:
        puntaje += 1
        motivos.append(f"La zona tiene {zona.toques} toques: nivel más probado.")
    if rvol is not None and rvol >= 1.2:
        puntaje += 1
        motivos.append(f"Volumen relativo elevado (RVOL {rvol:.2f}) en la última sesión.")

    if distancia < 0:
        estado = "dentro de la zona"
    elif cerca and acercandose:
        estado = "aproximándose"
    elif cerca:
        estado = "cerca, sin acercamiento claro"
    else:
        estado = "lejos"

    if estado == "lejos":
        confianza = "baja"
    elif puntaje >= 3:
        confianza = "alta"
    elif puntaje == 2:
        confianza = "media"
    else:
        confianza = "baja"

    return Diagnostico(
        precio_actual=precio,
        atr=a,
        zona=zona,
        distancia_atr=distancia,
        acercandose=acercandose,
        rvol=rvol,
        puntaje=puntaje,
        estado=estado,
        confianza=confianza,
        motivos=motivos,
    )


# ---------------------------------------------------------------------------
# Salida y utilidades
# ---------------------------------------------------------------------------

def imprimir(d: Diagnostico) -> None:
    print("=== Detector de aproximación a resistencia ===")
    print(f"Precio de cierre: {d.precio_actual:.2f} | ATR(14): {d.atr:.2f}")
    if d.zona is None:
        print("Estado: sin zona de resistencia válida por encima del precio.")
    else:
        z = d.zona
        print(
            f"Zona: {z.precio_bajo:.2f} – {z.precio_alto:.2f} "
            f"(toques: {z.toques}, última barra: {z.ultimo_toque})"
        )
        print(f"Distancia al borde inferior: {d.distancia_atr:.2f} ATR")
        print(f"Acercándose: {'sí' if d.acercandose else 'no'}")
        print(f"RVOL: {d.rvol:.2f}" if d.rvol is not None else "RVOL: no disponible")
    print(f"Estado: {d.estado} | Confianza: {d.confianza} (puntaje {d.puntaje}/4)")
    for m in d.motivos:
        print(f" - {m}")
    print("Informativo: no es recomendación de compra ni de venta.")


def cargar_csv(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df.columns = [c.strip().lower() for c in df.columns]
    col_fecha = next((c for c in ("date", "fecha", "timestamp") if c in df.columns), None)
    if col_fecha:
        df[col_fecha] = pd.to_datetime(df[col_fecha])
        df = df.set_index(col_fecha).sort_index()
    return df


def datos_demo(n: int = 250, seed: int = 7) -> pd.DataFrame:
    """Serie simulada solo para probar el script; no son datos de mercado reales."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.008, n)
    cierre = 700 * np.exp(np.cumsum(ret))
    apertura = np.r_[cierre[0], cierre[:-1]] * (1 + rng.normal(0, 0.002, n))
    alto = np.maximum(apertura, cierre) * (1 + np.abs(rng.normal(0, 0.004, n)))
    bajo = np.minimum(apertura, cierre) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volumen = rng.lognormal(16, 0.3, n)
    return pd.DataFrame(
        {"open": apertura, "high": alto, "low": bajo, "close": cierre, "volume": volumen}
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datos = cargar_csv(sys.argv[1])
    else:
        print("(Sin CSV: usando datos simulados)")
        datos = datos_demo()
    imprimir(analizar(datos))
