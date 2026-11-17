#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 BANCO DE PRUEBAS VIRTUAL PARA AMPLIFICADORES DE AUDIO DE POTENCIA (Clase AB)
===============================================================================
 Instrumentación virtual en Python: PyQt5 + Matplotlib + NumPy + SciPy

 Módulos:
   1. Conexión ............. VISA/SCPI, Hantek 6022BE/BL USB o simulador.
   2. Comparación .......... señales de entrada/salida, ganancia, fase y THD.

 Dependencias:
     pip install pyqt5 matplotlib numpy scipy pyvisa pyvisa-py
 (pyvisa-py es opcional si ya tiene NI-VISA / Keysight IO Libraries instalado.
  Para USB con pyvisa-py instale además: pip install pyusb)
 Para Hantek 6022BE/BL, consulte README.md (requiere Python 3.10 o posterior).

 Ejecución:
     python banco_pruebas_audio.py
===============================================================================
"""

import csv
import importlib
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

# PyVISA es opcional: sin él, el modo Simulador sigue funcionando.
try:
    import pyvisa
    VISA_DISPONIBLE = True
except ImportError:  # pragma: no cover
    pyvisa = None
    VISA_DISPONIBLE = False

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)


# =============================================================================
#  UTILIDADES
# =============================================================================
def fmt_si(valor: Optional[float], unidad: str, cifras: int = 4) -> str:
    """Formatea un número con prefijo SI (mV, kHz, µW...)."""
    if valor is None or not np.isfinite(valor):
        return "—"
    if valor == 0:
        return f"0 {unidad}"
    prefijos = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""),
                (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")]
    a = abs(valor)
    for factor, prefijo in prefijos:
        if a >= factor:
            return f"{valor / factor:.{cifras}g} {prefijo}{unidad}"
    return f"{valor:.3g} {unidad}"


# =============================================================================
#  PROCESAMIENTO DE SEÑAL (DSP)
# =============================================================================
@dataclass
class Armonico:
    """Un componente espectral: n=1 es la fundamental."""
    n: int
    f: float          # Hz
    v_rms: float      # V
    dbc: float        # dB relativos a la fundamental


@dataclass
class ResultadoAnalisis:
    """Resultado completo del análisis de una forma de onda."""
    fs: float
    n_muestras: int
    duracion: float
    ciclos: float
    v_dc: float
    v_pp: float
    v_pico: float
    v_rms_total: float            # fórmula directa sqrt(mean(v²))
    v_rms_ac: float               # sin componente DC
    f1: float
    v1_rms: float
    fase_fundamental: float
    armonicos: List[Armonico] = field(default_factory=list)   # n = 2..5
    thd_pct: float = float("nan")
    thdn_pct: float = float("nan")
    freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    espectro_dbv: np.ndarray = field(default_factory=lambda: np.array([]))
    avisos: List[str] = field(default_factory=list)


class AnalizadorSenal:
    """
    Análisis de forma de onda (Vrms, fundamental, armónicos y THD).

    Método espectral:
      * Se elimina la componente DC y se aplica ventana Blackman-Harris
        (4 términos, lóbulos laterales < -92 dB): evita que el "leakage" de la
        fundamental tape los armónicos cuando el muestreo NO es coherente.
      * La amplitud de cada tono se obtiene SUMANDO la energía de los bins de su
        lóbulo principal (±4 bins). Eso hace la medida insensible al
        "scalloping" (error por caer entre dos bins).
            A = sqrt( 4·S / (N·Σw²) ),  S = Σ|X_k|² del lóbulo (rfft)
      * La frecuencia fundamental se refina por interpolación parabólica.
    """
    SEMIANCHO_LOBULO = 4   # bins a cada lado del pico (Blackman-Harris)

    @staticmethod
    def analizar(t: Sequence[float], v: Sequence[float],
                 f_esperada: Optional[float] = None,
                 n_max_armonico: int = 5) -> ResultadoAnalisis:
        t = np.asarray(t, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        if t.size != v.size or v.size < 128:
            raise ValueError("Registro demasiado corto (mínimo 128 muestras).")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
            raise ValueError("La forma de onda contiene valores no válidos.")

        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            raise ValueError("Base de tiempos inválida (dt <= 0).")
        n = v.size
        fs = 1.0 / dt
        duracion = n * dt
        avisos: List[str] = []

        # ---- Métricas en el dominio del tiempo ----------------------------
        v_dc = float(np.mean(v))
        v_rms_total = float(np.sqrt(np.mean(v ** 2)))             # fórmula directa
        v_rms_ac = float(np.sqrt(max(v_rms_total ** 2 - v_dc ** 2, 0.0)))
        x = v - v_dc
        v_pp = float(np.max(v) - np.min(v))
        v_pico = float(np.max(np.abs(x)))

        # Un techo plano puede ser clipping o una forma intencional (p. ej.,
        # cuadrada). Exigir niveles intermedios evita marcar una cuadrada ideal.
        cerca_del_pico = (np.mean(np.abs(x) >= 0.995 * v_pico)
                          if v_pico > 0 else 0.0)
        niveles_intermedios = (np.mean((np.abs(x) >= 0.10 * v_pico)
                                       & (np.abs(x) <= 0.90 * v_pico))
                               if v_pico > 0 else 0.0)
        if cerca_del_pico > 0.10 and niveles_intermedios > 0.10:
            avisos.append("Posible recorte (clipping) del amplificador o de la "
                          "escala del osciloscopio: el THD no es fiable.")

        # ---- FFT con ventana ----------------------------------------------
        w = signal.windows.blackmanharris(n, sym=False)
        X = np.fft.rfft(x * w)
        mag = np.abs(X)
        mag2 = mag ** 2
        freqs = np.fft.rfftfreq(n, d=dt)
        df = float(freqs[1])
        sw = float(np.sum(w))
        sw2 = float(np.sum(w ** 2))
        nb = mag.size
        sa = AnalizadorSenal.SEMIANCHO_LOBULO

        # Espectro en dBV (valor eficaz por bin, corregido por ganancia coherente)
        amp_rms_bin = (2.0 * mag / sw) / np.sqrt(2.0)
        espectro_dbv = 20.0 * np.log10(np.maximum(amp_rms_bin, 1e-12))

        def v_rms_tono(kc: int) -> float:
            lo, hi = max(kc - sa, 1), min(kc + sa, nb - 1)
            s = float(np.sum(mag2[lo:hi + 1]))
            return float(np.sqrt(4.0 * s / (n * sw2)) / np.sqrt(2.0))

        # ---- Detección de la fundamental ----------------------------------
        k_min, k_max = 2, nb - 2
        if f_esperada:
            lo = int(max(k_min, np.floor(0.7 * f_esperada / df)))
            hi = int(min(k_max, np.ceil(1.3 * f_esperada / df)))
            if hi <= lo:
                raise ValueError("La frecuencia esperada está fuera del rango "
                                 "espectral del registro.")
        else:
            lo, hi = k_min, k_max
        k0 = lo + int(np.argmax(mag[lo:hi + 1]))

        p = 0.0
        if 1 <= k0 <= nb - 2:
            a, b, c = np.log(np.maximum(mag[k0 - 1:k0 + 2], 1e-30))
            den = a - 2.0 * b + c
            if abs(den) > 1e-12:
                p = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        f1 = (k0 + p) * df
        v1 = v_rms_tono(k0)
        if v1 < 1e-6:
            raise ValueError("No se detecta una señal periódica (amplitud ~ 0).")
        base_sin = np.sin(2.0 * np.pi * f1 * t)
        base_cos = np.cos(2.0 * np.pi * f1 * t)
        matriz = np.column_stack((base_sin, base_cos, np.ones(n)))
        coef = np.linalg.lstsq(matriz, v, rcond=None)[0]
        fase = float(np.arctan2(coef[1], coef[0]))
        ciclos = f1 * duracion
        if ciclos < 8:
            avisos.append(
                f"Solo {ciclos:.1f} ciclos capturados: use ≥ 10 ciclos "
                "(base de tiempos más lenta) para un THD preciso.")

        # ---- Armónicos 2..n_max y THD -------------------------------------
        armonicos: List[Armonico] = []
        for nh in range(2, n_max_armonico + 1):
            kh = int(round(nh * f1 / df))
            if kh + sa >= nb:
                avisos.append(
                    f"El armónico {nh} ({fmt_si(nh * f1, 'Hz')}) supera "
                    f"Nyquist ({fmt_si(fs / 2, 'Hz')}): aumente la "
                    "frecuencia de muestreo del osciloscopio.")
                break
            lo_h, hi_h = max(kh - 2, 1), min(kh + 2, nb - 1)
            kp = lo_h + int(np.argmax(mag[lo_h:hi_h + 1]))
            vh = v_rms_tono(kp)
            armonicos.append(Armonico(
                nh, nh * f1, vh,
                20.0 * np.log10(max(vh, 1e-15) / v1)))
        thd = (float(np.sqrt(sum(a_.v_rms ** 2 for a_ in armonicos))
                     / v1 * 100.0)
               if armonicos else float("nan"))
        thdn = float(
            np.sqrt(max(v_rms_ac ** 2 - v1 ** 2, 0.0)) / v1 * 100.0)

        return ResultadoAnalisis(
            fs=fs, n_muestras=n, duracion=duracion, ciclos=ciclos, v_dc=v_dc,
            v_pp=v_pp, v_pico=v_pico, v_rms_total=v_rms_total,
            v_rms_ac=v_rms_ac, f1=f1, v1_rms=v1,
            fase_fundamental=fase, armonicos=armonicos,
            thd_pct=thd, thdn_pct=thdn, freqs=freqs,
            espectro_dbv=espectro_dbv, avisos=avisos)


# =============================================================================
#  INSTRUMENTOS (capa de abstracción)
# =============================================================================
class CapturaCancelada(Exception):
    """Interrupción solicitada para una captura de larga duración."""


class InstrumentoBase:
    """Interfaz común: el resto de la aplicación no sabe si es real o simulado."""
    def identificar(self) -> str:
        raise NotImplementedError

    def capturar(self, canal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (t [s], v [V])."""
        raise NotImplementedError

    def capturar_varios(self, canales: Sequence[int]
                        ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        return {canal: self.capturar(canal) for canal in canales}

    def cerrar(self) -> None:
        pass


class OsciloscopioVISA(InstrumentoBase):
    """Osciloscopio real controlado por SCPI vía PyVISA."""

    def __init__(self, rm, recurso: str, timeout_ms: int = 5000,
                 estilo: str = "rigol"):
        self.recurso = recurso
        self.estilo = estilo      # 'rigol' | 'keysight' (fórmula de escalado)
        self.inst = rm.open_resource(recurso, open_timeout=3000)
        self.inst.timeout = timeout_ms        # evita bloqueos indefinidos
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        self.inst.chunk_size = 1024 * 1024

    def identificar(self) -> str:
        return self.inst.query("*IDN?").strip()

    def capturar(self, canal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        i = self.inst
        i.write(f":WAV:SOUR CHAN{canal}")
        i.write(":WAV:MODE NORM")
        i.write(":WAV:FORM BYTE")
        # Preámbulo: formato,tipo,puntos,cuenta,xinc,xorg,xref,yinc,yorg,yref
        pre = i.query(":WAV:PRE?").strip().split(",")
        if len(pre) < 10:
            raise RuntimeError(f"Preámbulo SCPI inesperado: {pre!r}")
        (_, _, _, _, xinc, xorg, xref, yinc, yorg, yref) = [float(p) for p in pre[:10]]
        crudo = i.query_binary_values(":WAV:DATA?", datatype="B",
                                      container=np.array).astype(float)
        if crudo.size == 0:
            raise RuntimeError("El osciloscopio no devolvió datos.")
        if self.estilo == "keysight":
            v = (crudo - yref) * yinc + yorg
        else:                                   # Rigol / Siglent
            v = (crudo - yorg - yref) * yinc
        t = (np.arange(crudo.size) - xref) * xinc + xorg
        return t, v

    def cerrar(self) -> None:
        try:
            self.inst.close()
        except Exception:
            pass


class OsciloscopioHantek6022B(InstrumentoBase):
    """Captura simultáneamente CH1 y CH2 del Hantek 6022BE/BL por USB."""

    MUESTRAS = 10000
    OPCIONES_MUESTRAS = (10_000, 50_000, 100_000, 250_000)
    MUESTRAS_INICIALES = 2048
    CODIGOS_CONTROL = {
        "ganancia_ch1": 0xE0,
        "ganancia_ch2": 0xE1,
        "frecuencia_muestreo": 0xE2,
        "numero_canales": 0xE4,
        "acoplamiento": 0xE5,
        "frecuencia_calibracion": 0xE6,
    }

    def __init__(self, encontrado, escalas, sondas,
                 sample_rate_hz=1e6, n_muestras=10_000,
                 auto_escala=True, auto_tiempo=True):
        self.modelo = encontrado.model
        if (isinstance(n_muestras, bool) or not isinstance(n_muestras, int)
                or not 128 <= n_muestras <= 250_000):
            raise ValueError("La captura Hantek debe tener entre 128 y 250000 muestras.")
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("La frecuencia de muestreo debe ser positiva.")
        self.n_muestras = n_muestras
        self.sample_rate_hz = float(sample_rate_hz)
        self.auto_escala = bool(auto_escala)
        self.auto_tiempo = bool(auto_tiempo)
        scope_device = importlib.import_module("pyhantek.device").ScopeDevice
        self.dispositivo = scope_device(encontrado)
        try:
            self.calibracion = self._leer_calibracion()
            self._configurar(escalas, sondas, self.sample_rate_hz)
        except Exception:
            self.dispositivo.close()
            raise

    def identificar(self) -> str:
        return f"Hantek {self.modelo.name}, serie {self.dispositivo.serial}"

    def _leer_calibracion(self):
        datos = self.dispositivo.read_calibration()
        if not datos or len(datos) < 80:
            return None
        return np.frombuffer(datos[:80], dtype=np.uint8).reshape(5, 8, 2)

    def _configurar(self, escalas, sondas, sample_rate_hz):
        self.canales_cfg = []
        for canal, (escala, sonda) in enumerate(zip(escalas, sondas)):
            indice = self.modelo.gain_index_for(escala / sonda)
            paso = self.modelo.gain_steps[indice]
            self.dispositivo.control(
                self.CODIGOS_CONTROL[f"ganancia_ch{canal + 1}"],
                bytes((paso.hw_gain, indice)))
            self.canales_cfg.append((indice, sonda))

        self.indice_muestreo = next(
            (i for i, tasa in enumerate(self.modelo.sample_rates)
             if np.isclose(tasa.rate, sample_rate_hz)), None)
        if self.indice_muestreo is None:
            raise ValueError(
                f"El Hantek no admite {fmt_si(sample_rate_hz, 'S/s')}.")
        self.tasa = self.modelo.sample_rates[self.indice_muestreo]
        self.dispositivo.control(
            self.CODIGOS_CONTROL["frecuencia_muestreo"],
            bytes((self.tasa.sample_id, self.indice_muestreo)))
        self.dispositivo.control(self.CODIGOS_CONTROL["numero_canales"], b"\x02")
        self.dispositivo.control(self.CODIGOS_CONTROL["acoplamiento"], b"\x11")
        self.dispositivo.control(
            self.CODIGOS_CONTROL["frecuencia_calibracion"], b"\x01")

    def _escala_adc(self, indice, canal):
        if self.calibracion is None:
            return 128.0, 1.0
        offset = int(self.calibracion[0, indice, canal])
        fine = int(self.calibracion[3, indice, canal])
        if offset in (0, 255):
            cero = 128.0
        elif fine in (0, 255):
            cero = float(offset)
        else:
            cero = offset + (fine - 128) / 250.0
        g = int(self.calibracion[2, indice, canal])
        factor = 1.0 if g in (0, 255) else 1.0 + (g - 128) / 500.0
        return cero, factor

    def ajustar_muestras_para_frecuencia(self, frecuencia_hz: float) -> int:
        if (not self.auto_tiempo or not np.isfinite(frecuencia_hz)
                or frecuencia_hz <= 0):
            return self.n_muestras
        necesarias = int(np.ceil(10.0 * self.tasa.rate / frecuencia_hz))
        self.n_muestras = next(
            (n for n in self.OPCIONES_MUESTRAS if n >= necesarias),
            self.OPCIONES_MUESTRAS[-1])
        return self.n_muestras

    def _ajustar_escala(self, canal: int, voltios: np.ndarray) -> None:
        if not self.auto_escala:
            return
        indice, sonda = self.canales_cfg[canal - 1]
        pico = float(np.max(np.abs(voltios)))
        if pico <= 0:
            return
        vdiv_objetivo = pico / (sonda * 3.0)
        pasos = self.modelo.gain_steps
        indice_nuevo = next(
            (i for i, paso in enumerate(pasos)
             if float(paso.vdiv) >= vdiv_objetivo),
            len(pasos) - 1)
        if indice_nuevo == indice:
            return
        paso = pasos[indice_nuevo]
        self.dispositivo.control(
            self.CODIGOS_CONTROL[f"ganancia_ch{canal}"],
            bytes((paso.hw_gain, indice_nuevo)))
        self.canales_cfg[canal - 1] = (indice_nuevo, sonda)

    def capturar_varios(self, canales, cancelar=None):
        if not canales or any(canal not in (1, 2) for canal in canales):
            raise ValueError("El Hantek 6022B solo tiene CH1 y CH2.")
        if cancelar is None:
            cancelar = threading.Event()
        tasa = self.tasa
        muestras_por_canal = (
            self.n_muestras * tasa.oversampling + self.MUESTRAS_INICIALES)
        muestras_por_canal = ((muestras_por_canal + 511) // 512) * 512
        omitir = muestras_por_canal - self.n_muestras * tasa.oversampling
        total_bytes = muestras_por_canal * 2
        try:
            datos = self.dispositivo.read(
                total_bytes,
                tasa.rate * tasa.oversampling * 2,
                cancelar)
        finally:
            self.dispositivo.stop_sampling()
        if cancelar.is_set():
            raise CapturaCancelada()
        if datos is None or len(datos) != total_bytes:
            raise RuntimeError("El Hantek no entregó una captura completa.")

        intercalados = np.asarray(datos, dtype=np.uint8).reshape(-1, 2)
        tiempo = np.arange(self.n_muestras, dtype=float) / tasa.rate
        resultado = {}
        for canal in set(canales):
            indice, sonda = self.canales_cfg[canal - 1]
            muestras = intercalados[omitir:, canal - 1]
            n = (muestras.size // tasa.oversampling) * tasa.oversampling
            muestras = muestras[:n].reshape(-1, tasa.oversampling).mean(axis=1)
            cero, factor = self._escala_adc(indice, canal - 1)
            voltios = ((muestras - cero) * factor
                       * sonda / self.modelo.voltage_scale[indice])
            resultado[canal] = (tiempo.copy(), voltios)
            self._ajustar_escala(canal, voltios)
        return resultado

    def cerrar(self):
        self.dispositivo.close()


class OsciloscopioSimulado(InstrumentoBase):
    """
    Simulador de un amplificador Clase AB + osciloscopio.
      CH1 = salida: ganancia(f) · Vin con armónicos, ruido y offset DC.
      CH2 = entrada (senoide limpia, Vin).
    La ganancia sigue un pasabanda de 1er orden (HP 15 Hz, LP 60 kHz, 26 dB).
    """
    PERFIL_ARMONICOS = np.array([0.30, 0.50, 0.10, 0.20])   # H2..H5 relativos
    FASES_ARMONICOS = np.array([0.7, 1.9, 2.8, 0.3])

    def __init__(self):
        self.f_gen = 1000.0       # Hz   (frecuencia del generador)
        self.vin_rms = 0.4        # Vrms (amplitud del generador)
        self.thd_pct = 0.5        # % objetivo
        self.ruido_mv = 2.0       # mVrms
        self.offset_mv = 3.0      # mV DC
        self.ganancia_dc = 20.0   # 26 dB
        self.f_hp, self.f_lp = 15.0, 60e3
        self.n_muestras, self.ciclos = 4000, 10
        self._rng = np.random.default_rng()

    def ganancia(self, f: float) -> float:
        hp = (f / self.f_hp) / np.sqrt(1 + (f / self.f_hp) ** 2)
        lp = 1.0 / np.sqrt(1 + (f / self.f_lp) ** 2)
        return self.ganancia_dc * hp * lp

    def identificar(self) -> str:
        return "SIMULADOR,Osciloscopio-Demo,0001,v1.0"

    def capturar_varios(self, canales: Sequence[int]
                        ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        time.sleep(0.05)                                # latencia "de bus"
        dur = self.ciclos / self.f_gen
        t = np.arange(self.n_muestras) / self.n_muestras * dur
        # Error de frecuencia ±0.2 % -> muestreo NO coherente (caso real)
        f_real = self.f_gen * (1 + self._rng.uniform(-0.002, 0.002))
        fase = self._rng.uniform(0, 2 * np.pi)          # disparo sin sincronía
        th = 2 * np.pi * f_real * t + fase
        resultados = {}
        for canal in canales:
            if canal == 2:
                v = self.vin_rms * np.sqrt(2) * np.sin(th)
                v += self._rng.normal(0, 0.5e-3, t.size)
            else:
                a1 = self.vin_rms * np.sqrt(2) * self.ganancia(self.f_gen)
                prof = self.PERFIL_ARMONICOS / np.sqrt(
                    np.sum(self.PERFIL_ARMONICOS ** 2))
                rel = self.thd_pct / 100.0 * prof
                v = a1 * np.sin(th)
                for k, (r, ph) in enumerate(
                        zip(rel, self.FASES_ARMONICOS), start=2):
                    v += a1 * r * np.sin(k * th + ph)
                v += (self.offset_mv * 1e-3
                      + self._rng.normal(0, self.ruido_mv * 1e-3, t.size))
            resultados[canal] = (t, v)
        return resultados

    def capturar(self, canal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        return self.capturar_varios((canal,))[canal]


# =============================================================================
#  HILOS DE TRABAJO (la GUI nunca habla con el instrumento directamente)
# =============================================================================
class TareaAsync(QThread):
    """Ejecuta una función bloqueante (escaneo, conexión) fuera del hilo de GUI."""
    terminado = pyqtSignal(object)
    fallo = pyqtSignal(str)

    def __init__(self, funcion: Callable, parent=None):
        super().__init__(parent)
        self._funcion = funcion

    def run(self):
        try:
            self.terminado.emit(self._funcion())
        except Exception as e:                         # noqa: BLE001
            self.fallo.emit(f"{type(e).__name__}: {e}")


class HiloCaptura(QThread):
    """Captura una o varias formas de onda (una vez o en bucle continuo)."""
    datos = pyqtSignal(object)      # dict {canal: (t, v)}
    error = pyqtSignal(str)

    def __init__(self, instrumento: InstrumentoBase, canales: Sequence[int],
                 continuo: bool = False, periodo_ms: int = 500, parent=None):
        super().__init__(parent)
        self.instrumento = instrumento
        self.canales = tuple(canales)
        self.continuo = continuo
        self.periodo_ms = periodo_ms
        self._activo = True
        self._cancelar = threading.Event()

    def detener(self):
        self._activo = False
        self._cancelar.set()

    def run(self):
        while self._activo:
            t0 = time.monotonic()
            try:
                if isinstance(self.instrumento, OsciloscopioHantek6022B):
                    resultado = self.instrumento.capturar_varios(
                        self.canales, self._cancelar)
                else:
                    resultado = self.instrumento.capturar_varios(self.canales)
            except CapturaCancelada:
                return
            except Exception as e:                      # noqa: BLE001
                self.error.emit(f"{type(e).__name__}: {e}")
                return
            if not self._activo:
                break
            self.datos.emit(resultado)
            if not self.continuo:
                break
            while self._activo and (time.monotonic() - t0) * 1000 < self.periodo_ms:
                self.msleep(20)


# =============================================================================
#  WIDGETS DE GRÁFICAS
# =============================================================================
class LienzoMpl(QWidget):
    """Canvas de Matplotlib + barra de herramientas (zoom, pan, guardar)."""
    def __init__(self, figsize=(8, 6), parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavigationToolbar(self.canvas, self))
        lay.addWidget(self.canvas)

    def redibujar(self):
        self.canvas.draw_idle()


# =============================================================================
#  PANEL 1: CONEXIÓN
# =============================================================================
class PanelConexion(QGroupBox):
    """Conexión por VISA o USB Hantek, además del modo Simulador."""
    conexion_cambiada = pyqtSignal(bool)
    mensaje = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("1. Conexión", parent)
        self._rm = None
        self._instrumento: Optional[InstrumentoBase] = None
        self._sim = OsciloscopioSimulado()
        self._tarea: Optional[TareaAsync] = None
        self._hantek_encontrados = {}
        self._auto_deteccion_habilitada = True
        self._auto_deteccion_iniciada = False
        self._timer_autodeteccion = QTimer(self)
        self._timer_autodeteccion.setInterval(5000)
        self._timer_autodeteccion.timeout.connect(
            self._escanear_hantek_automaticamente)

        self.chk_demo = QCheckBox("Modo Simulador / Demo")
        self.cmb_tipo = QComboBox()
        self.cmb_tipo.addItem("VISA / SCPI", "visa")
        self.cmb_tipo.addItem("Hantek 6022BE/BL (USB)", "hantek")
        self.cmb_tipo.setCurrentIndex(1)
        self.cmb_recursos = QComboBox()
        self.cmb_recursos.setEditable(True)
        self.cmb_recursos.setMinimumWidth(320)
        self.cmb_recursos.lineEdit().setPlaceholderText(
            "Escanee o escriba, p. ej. TCPIP0::192.168.1.50::INSTR")
        self.btn_escanear = QPushButton("Escanear")
        self.btn_conectar = QPushButton("Conectar")
        self.btn_conectar.setToolTip(
            "Conecta con el osciloscopio y comprueba su identificación.")
        self.btn_desconectar = QPushButton("Desconectar")
        self.btn_desconectar.setEnabled(False)
        self.cmb_escala = QComboBox()
        self.cmb_escala.addItem("Escalado Rigol / Siglent", "rigol")
        self.cmb_escala.addItem("Escalado Keysight / Agilent", "keysight")
        self.cmb_escala.setToolTip(
            "Rigol/Siglent: V = (dato − yorigin − yref)·yinc\n"
            "Keysight/Agilent: V = (dato − yref)·yinc + yorigin")
        self.lbl_idn = QLabel("Sin conexión")
        self.lbl_idn.setStyleSheet("font-weight: bold;")

        self.grp_hantek = QGroupBox("Ajustes del Hantek")
        h = QGridLayout(self.grp_hantek)
        self.cmb_escala_ch1 = self._combo_escalas()
        self.cmb_escala_ch1.setCurrentIndex(7)
        self.cmb_escala_ch2 = self._combo_escalas()
        self.cmb_escala_ch2.setCurrentIndex(4)
        self.cmb_sonda_ch1 = self._combo_sondas()
        self.cmb_sonda_ch2 = self._combo_sondas()
        for columna, (etiqueta, control) in enumerate((
                ("CH1 salida:", self.cmb_escala_ch1),
                ("Sonda:", self.cmb_sonda_ch1),
                ("CH2 entrada:", self.cmb_escala_ch2),
                ("Sonda:", self.cmb_sonda_ch2))):
            h.addWidget(QLabel(etiqueta), 0, columna * 2)
            h.addWidget(control, 0, columna * 2 + 1)
        self.cmb_tasa_hantek = QComboBox()
        self.cmb_tasa_hantek.addItem("500 kS/s", 500_000.0)
        self.cmb_tasa_hantek.setToolTip(
            "Elija una frecuencia suficiente para la señal. El límite útil "
            "es inferior a la mitad de esta tasa.")
        self.cmb_muestras_hantek = QComboBox()
        for muestras in (10_000, 50_000, 100_000, 250_000):
            self.cmb_muestras_hantek.addItem(
                f"{muestras:,}".replace(",", " "), muestras)
        self.cmb_muestras_hantek.setCurrentIndex(3)
        self.cmb_muestras_hantek.setToolTip(
            "Más muestras amplían la duración y mejoran la resolución de "
            "frecuencia, pero tardan más en transferirse.")
        self.chk_auto_escala_hantek = QCheckBox("Autoescala vertical")
        self.chk_auto_escala_hantek.setChecked(True)
        self.chk_auto_escala_hantek.setToolTip(
            "Ajusta automáticamente la escala física de cada canal después "
            "de cada captura.")
        self.chk_auto_tiempo_hantek = QCheckBox("Auto tiempo (10 ciclos)")
        self.chk_auto_tiempo_hantek.setChecked(True)
        self.chk_auto_tiempo_hantek.setToolTip(
            "Adapta el registro entre capturas para incluir unos diez ciclos.")
        h.addWidget(QLabel("Muestreo:"), 1, 0)
        h.addWidget(self.cmb_tasa_hantek, 1, 1, 1, 2)
        h.addWidget(QLabel("Muestras:"), 1, 3)
        h.addWidget(self.cmb_muestras_hantek, 1, 4, 1, 2)
        h.addWidget(self.chk_auto_escala_hantek, 2, 0, 1, 3)
        h.addWidget(self.chk_auto_tiempo_hantek, 2, 3, 1, 3)
        self.grp_hantek.setVisible(False)

        g = QGridLayout(self)
        g.addWidget(QLabel("Osciloscopio:"), 0, 0)
        g.addWidget(self.cmb_recursos, 0, 1, 1, 3)
        g.addWidget(self.btn_escanear, 0, 4)
        g.addWidget(self.cmb_tipo, 1, 0)
        g.addWidget(self.cmb_escala, 1, 1)
        g.addWidget(self.btn_conectar, 1, 2)
        g.addWidget(self.btn_desconectar, 1, 3)
        g.addWidget(self.chk_demo, 1, 4)
        g.addWidget(QLabel("Estado:"), 2, 0)
        g.addWidget(self.lbl_idn, 2, 1, 1, 4)
        g.addWidget(self.grp_hantek, 3, 0, 1, 5)

        self.cmb_tipo.currentIndexChanged.connect(self._cambio_tipo)
        self.btn_escanear.clicked.connect(self._escanear)
        self.btn_conectar.clicked.connect(self._conectar)
        self.btn_desconectar.clicked.connect(self._desconectar)
        self.chk_demo.toggled.connect(self._cambio_demo)

        self._cambio_tipo()
        if not VISA_DISPONIBLE and not self.es_hantek():
            self.lbl_idn.setText("PyVISA no instalado (pip install pyvisa pyvisa-py). "
                                 "Use el modo Simulador.")

    @staticmethod
    def _combo_escalas():
        combo = QComboBox()
        for valor, etiqueta in (
                (0.02, "20 mV/div"), (0.05, "50 mV/div"),
                (0.1, "100 mV/div"), (0.2, "200 mV/div"),
                (0.5, "500 mV/div"), (1.0, "1 V/div"),
                (2.0, "2 V/div"), (5.0, "5 V/div")):
            combo.addItem(etiqueta, valor)
        return combo

    @staticmethod
    def _combo_sondas():
        combo = QComboBox()
        combo.addItem("×1", 1)
        combo.addItem("×10", 10)
        return combo

    # ---- API pública -------------------------------------------------------
    def es_demo(self) -> bool:
        return self.chk_demo.isChecked()

    @property
    def simulador(self) -> OsciloscopioSimulado:
        return self._sim

    def instrumento(self) -> Optional[InstrumentoBase]:
        return self._sim if self.es_demo() else self._instrumento

    def estilo_escala(self) -> str:
        return self.cmb_escala.currentData()

    def es_hantek(self) -> bool:
        return self.cmb_tipo.currentData() == "hantek"

    def conexion_perdida(self, detalle: str):
        """Llamado cuando una captura falla con instrumento real."""
        self._cerrar_instrumento()
        self.lbl_idn.setText("⚠ Conexión perdida")
        self._set_ocupado(False)
        self.conexion_cambiada.emit(False)
        self.mensaje.emit("Conexión perdida con el osciloscopio.")
        QMessageBox.warning(
            self, "Conexión perdida",
            "Se perdió la comunicación con el osciloscopio.\n"
            "Revise el cable/red y vuelva a conectar.\n\nDetalle: " + detalle)

    # ---- Lógica interna ----------------------------------------------------
    def _set_ocupado(self, ocupado: bool):
        disponible = not ocupado and not self.es_demo()
        desconectado = self._instrumento is None
        self.cmb_tipo.setEnabled(disponible and desconectado)
        self.cmb_recursos.setEnabled(disponible and desconectado)
        self.btn_escanear.setEnabled(disponible and desconectado)
        self.btn_conectar.setEnabled(disponible and desconectado)
        self.btn_desconectar.setEnabled(disponible and not desconectado)
        self.cmb_escala.setEnabled(
            disponible and desconectado and not self.es_hantek())
        self.grp_hantek.setVisible(self.es_hantek() and not self.es_demo())
        for w in (self.cmb_escala_ch1, self.cmb_escala_ch2,
                  self.cmb_sonda_ch1, self.cmb_sonda_ch2,
                  self.cmb_tasa_hantek, self.cmb_muestras_hantek,
                  self.chk_auto_escala_hantek,
                  self.chk_auto_tiempo_hantek):
            w.setEnabled(disponible and desconectado and self.es_hantek())

    def _lanzar(self, funcion, ok, fallo):
        self._set_ocupado(True)
        self._tarea = TareaAsync(funcion, self)
        self._tarea.terminado.connect(lambda r: (self._set_ocupado(False), ok(r)))
        self._tarea.fallo.connect(lambda m: (self._set_ocupado(False), fallo(m)))
        self._tarea.start()

    def _obtener_rm(self):
        if not VISA_DISPONIBLE:
            raise RuntimeError("PyVISA no está instalado "
                               "(pip install pyvisa pyvisa-py).")
        if self._rm is None:
            try:
                self._rm = pyvisa.ResourceManager()        # NI-VISA / Keysight
            except Exception:                              # noqa: BLE001
                self._rm = pyvisa.ResourceManager("@py")   # backend puro Python
        return self._rm

    def _escanear(self):
        if self.es_hantek():
            self.mensaje.emit("Buscando Hantek 6022BE/BL por USB…")
            self._lanzar(self._buscar_hantek, self._escaneo_hantek_ok,
                         lambda m: self._error("Error al buscar Hantek", m))
            return
        self.mensaje.emit("Escaneando recursos VISA…")
        self._lanzar(lambda: list(self._obtener_rm().list_resources()),
                     self._escaneo_ok,
                     lambda m: self._error("Error al escanear", m))

    def iniciar_deteccion_automatica(self):
        if importlib.util.find_spec("pyhantek") is None:
            self._auto_deteccion_habilitada = False
            self.mensaje.emit(
                "Autodetección Hantek deshabilitada: falta instalar pyhantek.")
            return
        self._auto_deteccion_iniciada = True
        if self.es_hantek() and self._auto_deteccion_habilitada:
            self.mensaje.emit("Buscando y conectando el Hantek automáticamente…")
            self._timer_autodeteccion.start()
            QTimer.singleShot(0, self._escanear_hantek_automaticamente)

    def _escanear_hantek_automaticamente(self):
        if (not self._auto_deteccion_habilitada or not self.es_hantek()
                or self.es_demo() or self._instrumento is not None
                or (self._tarea is not None and self._tarea.isRunning())):
            return
        self._lanzar(
            self._buscar_hantek,
            self._escaneo_hantek_automatico_ok,
            self._error_autodeteccion)

    def _escaneo_hantek_automatico_ok(self, encontrados):
        self._escaneo_hantek_ok(encontrados, automatico=True)
        if len(encontrados) == 1:
            self._timer_autodeteccion.stop()
            QTimer.singleShot(0, lambda: self._conectar(automatico=True))
        elif len(encontrados) > 1:
            self._timer_autodeteccion.stop()
            self.mensaje.emit(
                "Hay varios Hantek; seleccione uno y pulse Conectar.")

    def _error_autodeteccion(self, detalle: str):
        self._timer_autodeteccion.stop()
        self._auto_deteccion_habilitada = False
        self._error("Autoconexión Hantek fallida", detalle)

    @staticmethod
    def _buscar_hantek():
        try:
            find_devices = importlib.import_module(
                "pyhantek.device").find_devices
        except ImportError as e:
            raise RuntimeError(
                "Falta el soporte USB del Hantek. Siga las instrucciones de "
                "instalación para Windows en README.md.") from e
        return [d for d in find_devices()
                if d.model.name in ("DSO-6022BE", "DSO-6022BL")]

    def _escaneo_hantek_ok(self, encontrados, automatico=False):
        self._hantek_encontrados = {d.label: d for d in encontrados}
        self.cmb_recursos.clear()
        self.cmb_recursos.addItems(list(self._hantek_encontrados))
        if encontrados:
            tasa_actual = self.cmb_tasa_hantek.currentData()
            self.cmb_tasa_hantek.clear()
            for tasa in encontrados[0].model.sample_rates:
                if tasa.rate >= 100_000:
                    self.cmb_tasa_hantek.addItem(
                        fmt_si(tasa.rate, "S/s"), float(tasa.rate))
            indice = self.cmb_tasa_hantek.findData(tasa_actual)
            if indice < 0:
                indice = self.cmb_tasa_hantek.findData(500_000.0)
            self.cmb_tasa_hantek.setCurrentIndex(max(indice, 0))
            if not automatico:
                self.mensaje.emit(
                    f"{len(encontrados)} Hantek 6022BE/BL encontrado(s).")
        elif not automatico:
            self.mensaje.emit(
                "No se encontró el Hantek. Revise USB y el controlador WinUSB.")

    def _escaneo_ok(self, recursos: List[str]):
        actual = self.cmb_recursos.currentText()
        self.cmb_recursos.clear()
        self.cmb_recursos.addItems(recursos)
        if actual:
            self.cmb_recursos.setEditText(actual)
        if recursos:
            self.mensaje.emit(f"{len(recursos)} recurso(s) VISA encontrado(s).")
        else:
            self.mensaje.emit("No se encontraron recursos. Puede escribir la "
                              "dirección a mano (p. ej. TCPIP0::IP::INSTR).")

    def _conectar(self, automatico=False):
        if self.es_hantek():
            encontrado = self._hantek_encontrados.get(
                self.cmb_recursos.currentText())
            if encontrado is None:
                QMessageBox.information(
                    self, "Conexión",
                    "Pulse Escanear y seleccione el Hantek 6022BE/BL.")
                return
            escalas = (self.cmb_escala_ch1.currentData(),
                       self.cmb_escala_ch2.currentData())
            sondas = (self.cmb_sonda_ch1.currentData(),
                      self.cmb_sonda_ch2.currentData())
            sample_rate_hz = self.cmb_tasa_hantek.currentData()
            n_muestras = self.cmb_muestras_hantek.currentData()
            auto_escala = self.chk_auto_escala_hantek.isChecked()
            auto_tiempo = self.chk_auto_tiempo_hantek.isChecked()
            self._timer_autodeteccion.stop()
            self._cerrar_instrumento()
            self.mensaje.emit("Conectando al Hantek 6022BE/BL…")

            def tarea_hantek():
                upload_firmware = importlib.import_module(
                    "pyhantek.device").upload_firmware
                if encontrado.needs_firmware:
                    encontrado_actualizado = upload_firmware(encontrado)
                else:
                    encontrado_actualizado = encontrado
                osc = OsciloscopioHantek6022B(
                    encontrado_actualizado, escalas, sondas,
                    sample_rate_hz, n_muestras, auto_escala, auto_tiempo)
                return osc, osc.identificar()

            self._lanzar(tarea_hantek, self._conexion_ok,
                         (self._error_autodeteccion if automatico else
                          lambda m: self._error("No se pudo conectar", m)))
            return

        recurso = self.cmb_recursos.currentText().strip()
        if not recurso:
            QMessageBox.information(self, "Conexión", "Seleccione o escriba un recurso VISA.")
            return
        self._cerrar_instrumento()
        estilo = self.estilo_escala()
        self.mensaje.emit(f"Conectando a {recurso}…")

        def tarea():
            osc = OsciloscopioVISA(self._obtener_rm(), recurso, estilo=estilo)
            try:
                return osc, osc.identificar()
            except Exception:
                osc.cerrar()
                raise
        self._lanzar(tarea, self._conexion_ok,
                     lambda m: self._error("No se pudo conectar", m))

    def _conexion_ok(self, resultado):
        self._instrumento, idn = resultado
        self._timer_autodeteccion.stop()
        self._auto_deteccion_habilitada = False
        self.lbl_idn.setText(idn)
        self._set_ocupado(False)
        self.mensaje.emit("Conexión establecida.")
        self.conexion_cambiada.emit(True)

    def _desconectar(self):
        self._timer_autodeteccion.stop()
        self._auto_deteccion_habilitada = False
        self._cerrar_instrumento()
        self.lbl_idn.setText("Sin conexión")
        self._set_ocupado(False)
        self.mensaje.emit("Desconectado.")
        self.conexion_cambiada.emit(False)

    def _cerrar_instrumento(self):
        if self._instrumento is not None:
            self._instrumento.cerrar()
            self._instrumento = None

    def _cambio_demo(self, activo: bool):
        if activo:
            self._timer_autodeteccion.stop()
        elif (self._auto_deteccion_iniciada
              and self._auto_deteccion_habilitada and self.es_hantek()):
            self._timer_autodeteccion.start()
        if activo:
            self.lbl_idn.setText("MODO DEMO — " + self._sim.identificar())
            self.conexion_cambiada.emit(True)
        else:
            self.lbl_idn.setText(self._instrumento.identificar()
                                 if self._instrumento else "Sin conexión")
            self.conexion_cambiada.emit(self._instrumento is not None)
        self._set_ocupado(False)

    def _cambio_tipo(self):
        if self.es_hantek():
            self.cmb_recursos.setEditable(False)
            self.cmb_recursos.setToolTip(
                "Seleccione el Hantek 6022BE/BL encontrado al escanear.")
        else:
            self.cmb_recursos.setEditable(True)
            self.cmb_recursos.setToolTip(
                "Recurso VISA, por ejemplo TCPIP0::192.168.1.50::INSTR.")
        self.cmb_recursos.clear()
        self._hantek_encontrados.clear()
        self._set_ocupado(False)
        if not self.es_hantek() and not VISA_DISPONIBLE:
            self.lbl_idn.setText(
                "PyVISA no instalado (pip install pyvisa pyvisa-py). "
                "Use el modo Simulador.")
        if self.es_hantek() and self._auto_deteccion_iniciada:
            if self._auto_deteccion_habilitada:
                self._timer_autodeteccion.start()
        else:
            self._timer_autodeteccion.stop()

    def _error(self, titulo: str, detalle: str):
        self.mensaje.emit(titulo)
        QMessageBox.warning(self, titulo, detalle)

    def cerrar(self):
        self._timer_autodeteccion.stop()
        self._cerrar_instrumento()

    def actualizar_ajustes_hantek(self, instrumento):
        for canal, combo in ((1, self.cmb_escala_ch1),
                             (2, self.cmb_escala_ch2)):
            indice, sonda = instrumento.canales_cfg[canal - 1]
            vdiv = instrumento.modelo.gain_steps[indice].vdiv * sonda
            seleccionado = next(
                (i for i in range(combo.count())
                 if np.isclose(combo.itemData(i), vdiv)), -1)
            if seleccionado >= 0:
                combo.setCurrentIndex(seleccionado)
        muestras_idx = self.cmb_muestras_hantek.findData(
            instrumento.n_muestras)
        if muestras_idx >= 0:
            self.cmb_muestras_hantek.setCurrentIndex(muestras_idx)


# =============================================================================
#  PESTAÑA 2: ADQUISICIÓN Y MÉTRICAS
# =============================================================================
class PestanaMedicion(QWidget):
    MAX_PUNTOS_GRAFICA = 3000

    def __init__(self, conexion: PanelConexion, parent=None):
        super().__init__(parent)
        self.conexion = conexion
        self._hilo: Optional[HiloCaptura] = None
        self._t: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t_entrada: Optional[np.ndarray] = None
        self._v_entrada: Optional[np.ndarray] = None
        self._res: Optional[ResultadoAnalisis] = None
        self._res_entrada: Optional[ResultadoAnalisis] = None
        self._anotaciones = []
        self._xmax_prev = None
        self._construir_ui()
        self.conexion.chk_demo.toggled.connect(self.grp_sim.setVisible)
        self.conexion.conexion_cambiada.connect(self._cambiar_captura_automatica)
        self.grp_sim.setVisible(self.conexion.es_demo())

    # ---- Construcción de la interfaz --------------------------------------
    def _spin(self, minimo, maximo, valor, dec, sufijo="", paso=None):
        s = QDoubleSpinBox()
        s.setRange(minimo, maximo)
        s.setDecimals(dec)
        s.setValue(valor)
        if sufijo:
            s.setSuffix(sufijo)
        if paso:
            s.setSingleStep(paso)
        return s

    def _construir_ui(self):
        # --- Parámetros de prueba
        self.spn_rl = self._spin(0.1, 10000, 8.0, 2, " Ω", 0.5)
        presets = QHBoxLayout()
        for r in (8, 4, 2, 1.8):
            b = QPushButton(f"{r:g} Ω")
            b.clicked.connect(lambda _, val=r: self.spn_rl.setValue(val))
            presets.addWidget(b)
        self.spn_vcc = self._spin(1, 500, 30.0, 2, " V", 1)
        self.spn_idc = self._spin(0, 100, 0.0, 4, " A", 0.01)
        self.cmb_alim = QComboBox()
        self.cmb_alim.addItem("Simple: P_DC = Vcc · Idc")
        self.cmb_alim.addItem("Simétrica ±Vcc: P_DC = 2 · Vcc · Idc (Idc de un riel)")
        self.cmb_entrada = QComboBox()
        self.cmb_entrada.addItems(["CH1", "CH2", "CH3", "CH4"])
        self.cmb_entrada.setCurrentIndex(1)
        self.cmb_entrada.setToolTip("Canal conectado a la entrada del amplificador.")
        self.cmb_salida = QComboBox()
        self.cmb_salida.addItems(["CH1", "CH2", "CH3", "CH4"])
        self.cmb_salida.setToolTip("Canal conectado a la salida del amplificador.")
        self.chk_auto = QCheckBox("Auto-detectar fundamental")
        self.chk_auto.setChecked(True)
        self.spn_f1 = self._spin(1, 15e6, 1000.0, 1, " Hz", 100)
        self.spn_f1.setEnabled(False)

        grp = QGroupBox("2. Señales a comparar")
        f = QFormLayout(grp)
        f.addRow("Entrada del amplificador:", self.cmb_entrada)
        f.addRow("Salida del amplificador:", self.cmb_salida)
        f.addRow("Carga R_L:", self.spn_rl)
        f.addRow("", presets)
        f.addRow("Alimentación V_CC:", self.spn_vcc)
        f.addRow("Corriente I_DC (multímetro):", self.spn_idc)
        f.addRow("Tipo de fuente:", self.cmb_alim)
        f.addRow(self.chk_auto, self.spn_f1)

        # --- Parámetros del simulador (visible solo en demo)
        self.sim_f = self._spin(10, 100e3, 1000, 1, " Hz", 100)
        self.sim_vin = self._spin(0.001, 5, 0.4, 3, " Vrms", 0.05)
        self.sim_thd = self._spin(0, 20, 0.5, 3, " %", 0.1)
        self.sim_ruido = self._spin(0, 500, 2.0, 2, " mVrms", 0.5)
        self.grp_sim = QGroupBox("Ajustes del simulador")
        fs = QFormLayout(self.grp_sim)
        fs.addRow("Frecuencia de entrada:", self.sim_f)
        fs.addRow("Señal de entrada:", self.sim_vin)
        fs.addRow("Distorsión objetivo (THD):", self.sim_thd)
        fs.addRow("Ruido:", self.sim_ruido)

        # --- Captura
        self.btn_unica = QPushButton("Capturar ambas señales")
        self.btn_continuo = QPushButton(
            "Iniciar / detener comparación continua")
        self.btn_continuo.setCheckable(True)
        self.spn_periodo = QSpinBox()
        self.spn_periodo.setRange(100, 10000)
        self.spn_periodo.setValue(500)
        self.spn_periodo.setSuffix(" ms")
        self.btn_csv = QPushButton("Exportar comparación (CSV)")
        grp_cap = QGroupBox("3. Captura")
        fc = QFormLayout(grp_cap)
        fc.addRow(self.btn_unica)
        fc.addRow(self.btn_continuo)
        fc.addRow("Periodo (continua):", self.spn_periodo)
        fc.addRow(self.btn_csv)

        self.btn_unica.clicked.connect(lambda: self._iniciar_captura(False))
        self.btn_continuo.clicked.connect(self._alternar_continuo)
        self.btn_csv.clicked.connect(self._guardar_csv)

        izq = QVBoxLayout()
        izq.addWidget(grp)
        izq.addWidget(self.grp_sim)
        izq.addWidget(grp_cap)
        izq.addStretch(1)
        w_izq = QWidget()
        w_izq.setLayout(izq)
        w_izq.setMaximumWidth(420)

        # --- Gráficas
        self.lienzo_tiempo = LienzoMpl(figsize=(10, 6))
        self.lienzo_espectro = LienzoMpl(figsize=(10, 6))
        self.ax_t = self.lienzo_tiempo.fig.add_subplot(1, 1, 1)
        self.ax_f = self.lienzo_espectro.fig.add_subplot(1, 1, 1)
        self.ax_t_in = self.ax_t.twinx()
        self.tabs_graficas = QTabWidget()
        self.tabs_graficas.addTab(self.lienzo_tiempo, "Forma de onda")
        self.tabs_graficas.addTab(self.lienzo_espectro, "Espectro")
        self.tabs_graficas.currentChanged.connect(self._redibujar_grafica_activa)
        self.ln_t, = self.ax_t.plot(
            [], [], lw=1.1, color="tab:blue", label="Salida")
        self.ln_t_in, = self.ax_t_in.plot(
            [], [], lw=1.0, color="tab:orange", label="Entrada")
        self.ax_t.set_title("Comparación en el tiempo")
        self.ax_t.set_xlabel("Tiempo [ms]")
        self.ax_t.set_ylabel("Salida [V]", color="tab:blue")
        self.ax_t_in.set_ylabel("Entrada [V]", color="tab:orange")
        self.ax_t.grid(True, alpha=0.4)
        self.ax_t.legend(handles=[self.ln_t_in, self.ln_t], loc="upper right")
        self.ln_f_in, = self.ax_f.plot(
            [], [], lw=0.8, color="tab:orange", label="Entrada")
        self.ln_f, = self.ax_f.plot(
            [], [], lw=0.8, color="tab:green", label="Salida")
        self.ln_h, = self.ax_f.plot(
            [], [], "ro", ms=5, label="Armónicos de salida")
        self.ax_f.set_title("Espectro comparado")
        self.ax_f.set_xlabel("Frecuencia [Hz]")
        self.ax_f.set_ylabel("Nivel por bin [dBV rms]")
        self.ax_f.grid(True, alpha=0.4)
        self.ax_f.legend(loc="upper right")

        # --- Resultados
        mono = "font-family: monospace; font-weight: bold;"
        filas_resumen = [("gain", "Ganancia fundamental (salida / entrada)"),
                         ("df", "Diferencia de frecuencia fundamental"),
                         ("phase", "Desfase de salida respecto a entrada"),
                         ("delta_thd", "Cambio de THD H2–H5")]
        filas_detalle = [("vpp", "Tensión pico a pico"),
                         ("vpk", "Tensión de pico (sin DC)"),
                         ("vdc", "Offset DC"),
                         ("f1", "Frecuencia fundamental de salida"),
                         ("vrms", "Tensión eficaz total de salida"),
                         ("vac", "Tensión eficaz AC de salida"),
                         ("prms", "Potencia de salida"),
                         ("pdc", "Potencia de alimentación"),
                         ("pdis", "Potencia disipada"),
                         ("eta", "Eficiencia"),
                         ("thdn", "Distorsión más ruido de salida (THD+N)")]
        grp_res = QGroupBox("Resultados")
        tabs_res = QTabWidget()
        self.lbl = {}
        self.titulos = {}
        for nombre, filas in (("Resumen", filas_resumen),
                              ("Detalles", filas_detalle)):
            pagina = QWidget()
            form = QFormLayout(pagina)
            for clave, texto in filas:
                lb = QLabel("—")
                lb.setStyleSheet(mono)
                self.lbl[clave] = lb
                titulo = QLabel(texto + ":")
                self.titulos[clave] = titulo
                form.addRow(titulo, lb)
            tabs_res.addTab(pagina, nombre)
        res_layout = QVBoxLayout(grp_res)
        res_layout.addWidget(tabs_res)
        self.lbl["gain"].setToolTip(
            "Relación entre la amplitud RMS de las frecuencias fundamentales.")
        self.lbl["phase"].setToolTip(
            "Salida menos entrada. Para que el desfase sea fiable, capture "
            "ambos canales con el mismo disparo del osciloscopio.")
        self.lbl["delta_thd"].setToolTip(
            "Diferencia en puntos porcentuales: THD de salida menos THD de entrada.")
        self.lbl["thdn"].setToolTip(
            "Incluye el contenido no fundamental y el ruido de la señal.")
        self.lbl["vac"].setToolTip(
            "La potencia usa el valor eficaz AC: el offset DC de salida no "
            "entrega potencia de audio útil a la carga.")

        self.tabla = QTableWidget(2, 4)
        self.tabla.setHorizontalHeaderLabels(
            ["Señal", "Fundamental [Hz]", "Vrms AC", "THD H2–H5"])
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setMaximumHeight(130)
        self.lbl_aviso = QLabel("")
        self.lbl_aviso.setWordWrap(True)
        self.lbl_aviso.setStyleSheet("color: #b35900; font-weight: bold;")

        inferior = QHBoxLayout()
        inferior.addWidget(grp_res, 1)
        col_tab = QVBoxLayout()
        grp_comp = QGroupBox("Comparación de señales")
        comp_layout = QVBoxLayout(grp_comp)
        comp_layout.addWidget(self.tabla)
        col_tab.addWidget(grp_comp)
        col_tab.addWidget(self.lbl_aviso)
        inferior.addLayout(col_tab, 1)
        w_inf = QWidget()
        w_inf.setLayout(inferior)
        w_inf.setMinimumHeight(0)
        scroll_resultados = QScrollArea()
        scroll_resultados.setWidgetResizable(True)
        scroll_resultados.setMinimumHeight(140)
        scroll_resultados.setWidget(w_inf)

        der = QSplitter(Qt.Vertical)
        der.addWidget(self.tabs_graficas)
        der.addWidget(scroll_resultados)
        der.setStretchFactor(0, 3)
        der.setStretchFactor(1, 1)
        der.setSizes([600, 180])

        raiz = QHBoxLayout(self)
        raiz.addWidget(w_izq)
        raiz.addWidget(der, 1)

        # Recalcular al cambiar parámetros (sin volver a capturar)
        for s in (self.spn_rl, self.spn_vcc, self.spn_idc, self.spn_f1):
            s.valueChanged.connect(self._reanalizar)
        self.cmb_alim.currentIndexChanged.connect(self._reanalizar)
        self.chk_auto.toggled.connect(self._cambiar_auto_fundamental)
        self.conexion.cmb_tasa_hantek.currentIndexChanged.connect(
            self._limitar_frecuencia_fundamental)

    def _cambiar_auto_fundamental(self, automatico: bool):
        self.spn_f1.setEnabled(not automatico)
        self._reanalizar()

    def _limitar_frecuencia_fundamental(self, *_):
        if self.conexion.es_hantek():
            tasa = self.conexion.cmb_tasa_hantek.currentData() or 1e6
            self.spn_f1.setMaximum(max(1.0, float(tasa) / 2.0))

    # ---- Captura -----------------------------------------------------------
    def _aplicar_params_sim(self):
        sim = self.conexion.simulador
        sim.f_gen = self.sim_f.value()
        sim.vin_rms = self.sim_vin.value()
        sim.thd_pct = self.sim_thd.value()
        sim.ruido_mv = self.sim_ruido.value()

    def _alternar_continuo(self, activo: bool):
        if activo:
            if not self._iniciar_captura(True):
                self.btn_continuo.setChecked(False)
        elif self._hilo is not None:
            self._hilo.detener()

    def _cambiar_captura_automatica(self, conectado: bool):
        if conectado and not self.btn_continuo.isChecked():
            self.btn_continuo.setChecked(True)
        elif not conectado and self.btn_continuo.isChecked():
            self.btn_continuo.setChecked(False)

    def _iniciar_captura(self, continuo: bool) -> bool:
        inst = self.conexion.instrumento()
        if inst is None:
            QMessageBox.information(self, "Sin instrumento",
                                    "Conecte el osciloscopio o active el modo Simulador.")
            return False
        if self._hilo is not None and self._hilo.isRunning():
            return False
        self._canal_entrada = self.cmb_entrada.currentIndex() + 1
        self._canal_salida = self.cmb_salida.currentIndex() + 1
        if self._canal_entrada == self._canal_salida:
            QMessageBox.warning(
                self, "Canales repetidos",
                "Seleccione canales distintos para la entrada y la salida.")
            return False
        if self.conexion.es_demo():
            if (self._canal_entrada, self._canal_salida) != (2, 1):
                QMessageBox.information(
                    self, "Canales del simulador",
                    "El simulador usa CH2 como entrada y CH1 como salida.")
                return False
            self._aplicar_params_sim()
        elif isinstance(inst, OsciloscopioVISA):
            inst.estilo = self.conexion.estilo_escala()
        elif self._canal_entrada > 2 or self._canal_salida > 2:
            QMessageBox.warning(
                self, "Canales no disponibles",
                "El Hantek 6022BE/BL solo dispone de CH1 y CH2.")
            return False
        canales = (self._canal_entrada, self._canal_salida)
        self._hilo = HiloCaptura(inst, canales, continuo,
                                 self.spn_periodo.value(), self)
        self._hilo.datos.connect(self._on_datos)
        self._hilo.error.connect(self._on_error)
        self._hilo.finished.connect(self._on_hilo_terminado)
        self._poner_ocupado(True)
        self._hilo.start()
        return True

    def _poner_ocupado(self, b: bool):
        self.btn_unica.setEnabled(not b)
        self.conexion.setEnabled(not b)

    def _on_datos(self, datos: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        self._t_entrada, self._v_entrada = datos[self._canal_entrada]
        self._t, self._v = datos[self._canal_salida]
        if self.conexion.es_demo() and self._hilo and self._hilo.continuo:
            self._aplicar_params_sim()      # permite mover los controles "en vivo"
        self._reanalizar()
        instrumento = self.conexion.instrumento()
        if (isinstance(instrumento, OsciloscopioHantek6022B)
                and self._res is not None and self._res_entrada is not None):
            frecuencia_minima = min(self._res.f1, self._res_entrada.f1)
            instrumento.ajustar_muestras_para_frecuencia(frecuencia_minima)
            self.conexion.actualizar_ajustes_hantek(instrumento)

    def _on_error(self, msg: str):
        self.btn_continuo.setChecked(False)
        if self.conexion.es_demo():
            QMessageBox.warning(self, "Error en el simulador", msg)
        else:
            self.conexion.conexion_perdida(msg)

    def _on_hilo_terminado(self):
        self.btn_continuo.setChecked(False)
        self._poner_ocupado(False)

    def closeEvent(self, ev):  # pragma: no cover
        self.detener()
        super().closeEvent(ev)

    def detener(self):
        if self._hilo is not None and self._hilo.isRunning():
            self._hilo.detener()
            self._hilo.wait(6000)

    # ---- Análisis y visualización -----------------------------------------
    def _reanalizar(self, *_):
        if self._t is None or self._t_entrada is None:
            return
        f_esp = None if self.chk_auto.isChecked() else self.spn_f1.value()
        try:
            self._res = AnalizadorSenal.analizar(
                self._t, self._v, f_esperada=f_esp)
            self._res_entrada = AnalizadorSenal.analizar(
                self._t_entrada, self._v_entrada, f_esperada=f_esp)
            avisos = ([f"Entrada: {av}" for av in self._res_entrada.avisos]
                      + [f"Salida: {av}" for av in self._res.avisos])
            if (abs(self._res.f1 - self._res_entrada.f1)
                    / self._res_entrada.f1 > 0.01):
                avisos.append(
                    "Las frecuencias difieren más de 1 %; compare el desfase con cuidado.")
            self.lbl_aviso.setText("  ".join(avisos))
        except ValueError as e:
            self._res = None
            self._res_entrada = None
            self.lbl_aviso.setText(f"⚠ {e}")
        self._pintar()
        self._actualizar_metricas()

    @staticmethod
    def _buscar_disparo(t, v, resultado):
        if (resultado is None or not np.isfinite(resultado.f1)
                or resultado.f1 <= 0 or not np.isfinite(resultado.v_pico)
                or resultado.v_pico <= 0):
            return None
        nivel = resultado.v_dc + 0.05 * resultado.v_pico
        arriba = v >= nivel
        indices = np.flatnonzero(~arriba[:-1] & arriba[1:])
        if indices.size < 3:
            return None
        fracciones = ((nivel - v[indices])
                      / (v[indices + 1] - v[indices]))
        cruces = t[indices] + fracciones * (t[indices + 1] - t[indices])
        periodo = 1.0 / resultado.f1
        intervalos = np.diff(cruces)
        periodicos = np.abs(intervalos / periodo - 1.0) <= 0.25
        candidatos = np.flatnonzero(periodicos[:-1] & periodicos[1:])
        if not candidatos.size:
            return None
        return float(cruces[candidatos[0]])

    @staticmethod
    def _agregar_punto_disparo(tiempo_grafica, valores_grafica, t, v,
                               disparo):
        if (disparo is None or not tiempo_grafica.size
                or tiempo_grafica[0] <= 0 or disparo < t[0]
                or disparo > t[-1]):
            return tiempo_grafica, valores_grafica
        valor_disparo = np.interp(disparo, t, v)
        return (np.concatenate(([0.0], tiempo_grafica)),
                np.concatenate(([valor_disparo], valores_grafica)))

    @staticmethod
    def _decimar_serie(x, y, max_puntos=MAX_PUNTOS_GRAFICA):
        if y.size <= max_puntos:
            return x, y
        n_por_bloque = max(1, max_puntos // 4)
        tam_bloque = (y.size + n_por_bloque - 1) // n_por_bloque
        inicios = np.arange(0, y.size, tam_bloque)
        n_relleno = inicios.size * tam_bloque - y.size
        valores_min = np.pad(
            y, (0, n_relleno), constant_values=np.inf
        ).reshape(-1, tam_bloque)
        valores_max = np.pad(
            y, (0, n_relleno), constant_values=-np.inf
        ).reshape(-1, tam_bloque)
        finales = np.minimum(inicios + tam_bloque, y.size) - 1
        indices = np.concatenate((
            inicios,
            inicios + valores_min.argmin(axis=1),
            inicios + valores_max.argmax(axis=1),
            finales,
        ))
        indices.sort()
        return x[indices], y[indices]

    @staticmethod
    def _decimar_espectro(freqs, espectro_dbv,
                         max_puntos=MAX_PUNTOS_GRAFICA):
        if espectro_dbv.size <= max_puntos:
            return freqs, espectro_dbv
        tam_bloque = (
            espectro_dbv.size + max_puntos - 1) // max_puntos
        inicios = np.arange(0, espectro_dbv.size, tam_bloque)
        n_relleno = inicios.size * tam_bloque - espectro_dbv.size
        bloques = np.pad(
            espectro_dbv, (0, n_relleno), constant_values=-np.inf
        ).reshape(-1, tam_bloque)
        indices = inicios + bloques.argmax(axis=1)
        return freqs[indices], espectro_dbv[indices]

    def _redibujar_grafica_activa(self, *_):
        lienzo = (self.lienzo_tiempo if self.tabs_graficas.currentIndex() == 0
                  else self.lienzo_espectro)
        lienzo.redibujar()

    def _pintar(self):
        r = self._res
        r_entrada = self._res_entrada
        disparo = None
        resultado_referencia = None
        nombre_referencia = ""
        referencias = (
            (self._t_entrada, self._v_entrada, r_entrada,
             f"entrada {self.cmb_entrada.currentText()}"),
            (self._t, self._v, r,
             f"salida {self.cmb_salida.currentText()}"),
        )
        for t_ref, v_ref, analisis_ref, nombre in referencias:
            disparo = self._buscar_disparo(t_ref, v_ref, analisis_ref)
            if disparo is not None:
                resultado_referencia = analisis_ref
                nombre_referencia = nombre
                break

        inicio = (disparo if disparo is not None
                  else min(float(self._t[0]), float(self._t_entrada[0])))
        tiempo = (self._t - inicio) * 1e3
        tiempo_entrada = (self._t_entrada - inicio) * 1e3
        resultado_ventana = (
            resultado_referencia if resultado_referencia is not None
            else r if r is not None else r_entrada)
        duracion = max(
            0.0, min(float(self._t[-1]), float(self._t_entrada[-1])) - inicio)
        limite_ms = duracion * 1e3
        if resultado_ventana is not None and resultado_ventana.f1 > 0:
            limite_ms = min(limite_ms, 3.0 / resultado_ventana.f1 * 1e3)

        inicio_muestras = np.searchsorted(tiempo, 0.0, side="left")
        fin = np.searchsorted(tiempo, limite_ms, side="right")
        inicio_entrada = np.searchsorted(
            tiempo_entrada, 0.0, side="left")
        fin_entrada = np.searchsorted(
            tiempo_entrada, limite_ms, side="right")
        tiempo_salida, valores_salida = self._agregar_punto_disparo(
            tiempo[inicio_muestras:fin], self._v[inicio_muestras:fin],
            self._t, self._v, disparo)
        tiempo_entrada_plot, valores_entrada_plot = self._agregar_punto_disparo(
            tiempo_entrada[inicio_entrada:fin_entrada],
            self._v_entrada[inicio_entrada:fin_entrada],
            self._t_entrada, self._v_entrada, disparo)
        self.ln_t.set_data(*self._decimar_serie(
            tiempo_salida, valores_salida))
        self.ln_t_in.set_data(*self._decimar_serie(
            tiempo_entrada_plot, valores_entrada_plot))
        self.ax_t.relim()
        self.ax_t.autoscale_view()
        self.ax_t_in.relim()
        self.ax_t_in.autoscale_view()
        if limite_ms > 0:
            self.ax_t.set_xlim(0, limite_ms)
        self.ax_t.set_title(
            "Comparación en el tiempo — "
            + (f"sincronizada con {nombre_referencia}"
               if disparo is not None else "barrido libre"))
        for a in self._anotaciones:
            a.remove()
        self._anotaciones.clear()
        if r is not None and r_entrada is not None:
            xmax = min(r.fs / 2, max(8 * r.f1, 1.0))
            fin_entrada = np.searchsorted(
                r_entrada.freqs, xmax, side="right")
            fin = np.searchsorted(r.freqs, xmax, side="right")
            self.ln_f_in.set_data(*self._decimar_espectro(
                r_entrada.freqs[:fin_entrada],
                r_entrada.espectro_dbv[:fin_entrada]))
            self.ln_f.set_data(*self._decimar_espectro(
                r.freqs[:fin], r.espectro_dbv[:fin]))
            comps = [Armonico(1, r.f1, r.v1_rms, 0.0)] + r.armonicos
            xs = [c.f for c in comps]
            ys = [20 * np.log10(max(c.v_rms, 1e-12)) for c in comps]
            self.ln_h.set_data(xs, ys)
            for c, y in zip(comps, ys):
                self._anotaciones.append(self.ax_f.annotate(
                    f"H{c.n}", (c.f, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8))
            techo = max(ys)
            if self._xmax_prev is None or abs(xmax - self._xmax_prev) > 0.1 * xmax:
                self.ax_f.set_xlim(0, xmax)
                self._xmax_prev = xmax
            self.ax_f.set_ylim(techo - 110, techo + 10)
        else:
            self.ln_f_in.set_data([], [])
            self.ln_f.set_data([], [])
            self.ln_h.set_data([], [])
        self._redibujar_grafica_activa()

    def _actualizar_metricas(self):
        L = self.lbl
        r = self._res
        r_entrada = self._res_entrada
        if r is None or r_entrada is None:
            for lb in L.values():
                lb.setText("—")
            self.tabla.clearContents()
            return
        rl, vcc, idc = self.spn_rl.value(), self.spn_vcc.value(), self.spn_idc.value()
        ganancia = (r.v1_rms / r_entrada.v1_rms
                    if r_entrada.v1_rms > 0 else float("nan"))
        ganancia_db = (20.0 * np.log10(ganancia)
                       if np.isfinite(ganancia) and ganancia > 0
                       else float("nan"))
        tiene_picos = np.isfinite(r.f1) and np.isfinite(r_entrada.f1)
        delta_f = r.f1 - r_entrada.f1 if tiene_picos else float("nan")
        delta_f_pct = (abs(delta_f) / r_entrada.f1
                       if tiene_picos and r_entrada.f1 > 0 else float("inf"))
        desfase = ((r.fase_fundamental - r_entrada.fase_fundamental + np.pi)
                   % (2.0 * np.pi) - np.pi)
        L["gain"].setText(
            f"{ganancia:.3f}×  ({ganancia_db:+.2f} dB)"
            if np.isfinite(ganancia_db) else "—")
        L["df"].setText(f"{delta_f:+.2f} Hz" if tiene_picos else "—")
        L["phase"].setText(
            f"{np.degrees(desfase):+.1f}°" if delta_f_pct <= 0.01
            else "— (frecuencias distintas)")
        if np.isfinite(r.thd_pct) and np.isfinite(r_entrada.thd_pct):
            L["delta_thd"].setText(
                f"{r.thd_pct - r_entrada.thd_pct:+.4f} puntos porcentuales")
        else:
            L["delta_thd"].setText("—")

        p_rms = r.v_rms_ac ** 2 / rl
        factor = 2.0 if self.cmb_alim.currentIndex() == 1 else 1.0
        p_dc = factor * vcc * idc
        L["f1"].setText(fmt_si(r.f1, "Hz") if np.isfinite(r.f1) else "—")
        L["vpp"].setText(fmt_si(r.v_pp, "V"))
        L["vpk"].setText(fmt_si(r.v_pico, "V"))
        L["vdc"].setText(fmt_si(r.v_dc, "V"))
        L["vrms"].setText(fmt_si(r.v_rms_total, "V"))
        L["vac"].setText(fmt_si(r.v_rms_ac, "V"))
        L["prms"].setText(fmt_si(p_rms, "W"))
        if p_dc > 0:
            eta = p_rms / p_dc * 100.0
            L["pdc"].setText(fmt_si(p_dc, "W"))
            L["pdis"].setText(fmt_si(p_dc - p_rms, "W"))
            L["eta"].setText(f"{eta:.2f} %")
            if eta > 100 and "η > 100 %" not in self.lbl_aviso.text():
                self.lbl_aviso.setText(self.lbl_aviso.text() +
                                       "  η > 100 %: revise V_CC, I_DC, R_L o el tipo de fuente.")
        else:
            L["pdc"].setText("— (ingrese I_DC)")
            L["pdis"].setText("—")
            L["eta"].setText("— (ingrese I_DC)")
        L["thdn"].setText(f"{r.thdn_pct:.4f} %")

        self.tabla.clearContents()
        comparaciones = [
            ("Entrada", self.cmb_entrada.currentText(), r_entrada),
            ("Salida", self.cmb_salida.currentText(), r),
        ]
        for i, (nombre, canal, resultado) in enumerate(comparaciones):
            thd = (f"{resultado.thd_pct:.4f} %"
                   if np.isfinite(resultado.thd_pct) else "—")
            frecuencia = (fmt_si(resultado.f1, "Hz")
                          if np.isfinite(resultado.f1) else "—")
            vals = [f"{nombre} ({canal})", frecuencia,
                    fmt_si(resultado.v_rms_ac, "V"), thd]
            for j, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                self.tabla.setItem(i, j, it)

    def _guardar_csv(self):
        if self._t is None or self._t_entrada is None:
            QMessageBox.information(self, "CSV", "Aún no hay una captura.")
            return
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar forma de onda",
                                              "forma_de_onda.csv", "CSV (*.csv)")
        if not ruta:
            return
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["t_entrada_s", "v_entrada_V",
                            "t_salida_s", "v_salida_V"])
                w.writerows(zip_longest(
                    self._t_entrada.tolist(), self._v_entrada.tolist(),
                    self._t.tolist(), self._v.tolist(), fillvalue=""))
        except OSError as e:
            QMessageBox.warning(self, "Error al guardar", str(e))


# =============================================================================
#  VENTANA PRINCIPAL
# =============================================================================
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comparador de señales — Amplificadores de audio")
        self.resize(1400, 900)

        self.conexion = PanelConexion()
        self.medicion = PestanaMedicion(self.conexion)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(self.conexion)
        lay.addWidget(self.medicion, 1)
        self.setCentralWidget(central)

        self.conexion.mensaje.connect(self.statusBar().showMessage)
        self.statusBar().showMessage(
            "Listo. Active el modo Simulador o conecte un osciloscopio para comparar entrada y salida.")
        self.conexion.iniciar_deteccion_automatica()

    def closeEvent(self, ev):
        self.medicion.detener()
        self.conexion.cerrar()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

