#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 BANCO DE PRUEBAS VIRTUAL PARA AMPLIFICADORES DE AUDIO DE POTENCIA (Clase AB)
===============================================================================
 Instrumentación virtual en Python: PyQt5 + Matplotlib + NumPy + SciPy

 Módulos:
   1. Conexión ............. VISA/SCPI, Hantek 6022BE USB o simulador.
   2. Comparación .......... señales de entrada/salida, ganancia, fase y THD.

 Dependencias:
     pip install pyqt5 matplotlib numpy scipy pyvisa pyvisa-py
 (pyvisa-py es opcional si ya tiene NI-VISA / Keysight IO Libraries instalado.
  Para USB con pyvisa-py instale además: pip install pyusb)
 Para el Hantek 6022BE: pip install pyhantek (requiere Python 3.10 o posterior).

 Ejecución:
     python banco_pruebas_audio.py
===============================================================================
"""

import csv
import importlib
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

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
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

        # Heurística de recorte (clipping): una senoide limpia tiene ~6 % de
        # muestras a menos de 0.5 % de su pico; un "techo plano" supera el 10 %.
        if v_pico > 0 and np.mean(np.abs(x) >= 0.995 * v_pico) > 0.10:
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
            avisos.append(f"Solo {ciclos:.1f} ciclos capturados: use ≥ 10 ciclos "
                          "(base de tiempo más lenta) para un THD preciso.")

        # ---- Armónicos 2..n_max y THD -------------------------------------
        armonicos: List[Armonico] = []
        for nh in range(2, n_max_armonico + 1):
            kh = int(round(nh * f1 / df))
            if kh + sa >= nb:
                avisos.append(f"El armónico {nh} ({fmt_si(nh * f1, 'Hz')}) supera "
                              f"Nyquist ({fmt_si(fs / 2, 'Hz')}): aumente la "
                              "frecuencia de muestreo del osciloscopio.")
                break
            lo_h, hi_h = max(kh - 2, 1), min(kh + 2, nb - 1)
            kp = lo_h + int(np.argmax(mag[lo_h:hi_h + 1]))
            vh = v_rms_tono(kp)
            armonicos.append(Armonico(nh, nh * f1, vh,
                                      20.0 * np.log10(max(vh, 1e-15) / v1)))

        thd = (float(np.sqrt(sum(a_.v_rms ** 2 for a_ in armonicos)) / v1 * 100.0)
               if armonicos else float("nan"))
        thdn = float(np.sqrt(max(v_rms_ac ** 2 - v1 ** 2, 0.0)) / v1 * 100.0)

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


class OsciloscopioHantek6022BE(InstrumentoBase):
    """Captura simultáneamente CH1 y CH2 del Hantek 6022BE por USB."""

    MUESTRAS = 10000
    MUESTRAS_INICIALES = 2048
    CODIGOS_CONTROL = {
        "ganancia_ch1": 0xE0,
        "ganancia_ch2": 0xE1,
        "frecuencia_muestreo": 0xE2,
        "numero_canales": 0xE4,
        "acoplamiento": 0xE5,
        "frecuencia_calibracion": 0xE6,
    }

    def __init__(self, encontrado, escalas, sondas):
        self.modelo = encontrado.model
        scope_device = importlib.import_module("pyhantek.device").ScopeDevice
        self.dispositivo = scope_device(encontrado)
        try:
            self.calibracion = self._leer_calibracion()
            self._configurar(escalas, sondas)
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

    def _configurar(self, escalas, sondas):
        self.canales_cfg = []
        for canal, (escala, sonda) in enumerate(zip(escalas, sondas)):
            indice = self.modelo.gain_index_for(escala / sonda)
            paso = self.modelo.gain_steps[indice]
            self.dispositivo.control(
                self.CODIGOS_CONTROL[f"ganancia_ch{canal + 1}"],
                bytes((paso.hw_gain, indice)))
            self.canales_cfg.append((indice, sonda))

        self.indice_muestreo = next(
            i for i, tasa in enumerate(self.modelo.sample_rates)
            if tasa.rate == 1e6)
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

    def capturar_varios(self, canales):
        if not canales or any(canal not in (1, 2) for canal in canales):
            raise ValueError("El Hantek 6022BE solo tiene CH1 y CH2.")
        tasa = self.tasa
        muestras_por_canal = (
            self.MUESTRAS * tasa.oversampling + self.MUESTRAS_INICIALES)
        muestras_por_canal = ((muestras_por_canal + 511) // 512) * 512
        omitir = muestras_por_canal - self.MUESTRAS * tasa.oversampling
        total_bytes = muestras_por_canal * 2
        try:
            datos = self.dispositivo.read(
                total_bytes,
                tasa.rate * tasa.oversampling * 2,
                threading.Event())
        finally:
            self.dispositivo.stop_sampling()
        if datos is None or len(datos) != total_bytes:
            raise RuntimeError("El Hantek no entregó una captura completa.")

        intercalados = np.asarray(datos, dtype=np.uint8).reshape(-1, 2)
        tiempo = np.arange(self.MUESTRAS, dtype=float) / tasa.rate
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

    def detener(self):
        self._activo = False

    def run(self):
        while self._activo:
            t0 = time.monotonic()
            try:
                resultado = self.instrumento.capturar_varios(self.canales)
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

        self.chk_demo = QCheckBox("Modo Simulador / Demo")
        self.cmb_tipo = QComboBox()
        self.cmb_tipo.addItem("VISA / SCPI", "visa")
        self.cmb_tipo.addItem("Hantek 6022BE (USB)", "hantek")
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

        self.grp_hantek = QGroupBox("Ajustes del Hantek (por división)")
        h = QHBoxLayout(self.grp_hantek)
        self.cmb_escala_ch1 = self._combo_escalas()
        self.cmb_escala_ch1.setCurrentIndex(7)
        self.cmb_escala_ch2 = self._combo_escalas()
        self.cmb_escala_ch2.setCurrentIndex(4)
        self.cmb_sonda_ch1 = self._combo_sondas()
        self.cmb_sonda_ch2 = self._combo_sondas()
        for etiqueta, control in (
                ("CH1 salida:", self.cmb_escala_ch1),
                ("Sonda:", self.cmb_sonda_ch1),
                ("CH2 entrada:", self.cmb_escala_ch2),
                ("Sonda:", self.cmb_sonda_ch2)):
            h.addWidget(QLabel(etiqueta))
            h.addWidget(control)
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
        if not VISA_DISPONIBLE:
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
                  self.cmb_sonda_ch1, self.cmb_sonda_ch2):
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
            self.mensaje.emit("Buscando Hantek 6022BE por USB…")
            self._lanzar(self._buscar_hantek, self._escaneo_hantek_ok,
                         lambda m: self._error("Error al buscar Hantek", m))
            return
        self.mensaje.emit("Escaneando recursos VISA…")
        self._lanzar(lambda: list(self._obtener_rm().list_resources()),
                     self._escaneo_ok,
                     lambda m: self._error("Error al escanear", m))

    @staticmethod
    def _buscar_hantek():
        try:
            find_devices = importlib.import_module(
                "pyhantek.device").find_devices
        except ImportError as e:
            raise RuntimeError(
                "Falta el controlador Python del Hantek. Instálelo con "
                "'pip install pyhantek'.") from e
        return [d for d in find_devices()
                if d.model.name == "DSO-6022BE"]

    def _escaneo_hantek_ok(self, encontrados):
        self._hantek_encontrados = {d.label: d for d in encontrados}
        self.cmb_recursos.clear()
        self.cmb_recursos.addItems(list(self._hantek_encontrados))
        if encontrados:
            self.mensaje.emit(
                f"{len(encontrados)} Hantek 6022BE encontrado(s).")
        else:
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

    def _conectar(self):
        if self.es_hantek():
            encontrado = self._hantek_encontrados.get(
                self.cmb_recursos.currentText())
            if encontrado is None:
                QMessageBox.information(
                    self, "Conexión",
                    "Pulse Escanear y seleccione el Hantek 6022BE.")
                return
            escalas = (self.cmb_escala_ch1.currentData(),
                       self.cmb_escala_ch2.currentData())
            sondas = (self.cmb_sonda_ch1.currentData(),
                      self.cmb_sonda_ch2.currentData())
            self._cerrar_instrumento()
            self.mensaje.emit("Conectando al Hantek 6022BE…")

            def tarea_hantek():
                upload_firmware = importlib.import_module(
                    "pyhantek.device").upload_firmware
                if encontrado.needs_firmware:
                    encontrado_actualizado = upload_firmware(encontrado)
                else:
                    encontrado_actualizado = encontrado
                osc = OsciloscopioHantek6022BE(
                    encontrado_actualizado, escalas, sondas)
                return osc, osc.identificar()

            self._lanzar(tarea_hantek, self._conexion_ok,
                         lambda m: self._error("No se pudo conectar", m))
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
        self.lbl_idn.setText(idn)
        self._set_ocupado(False)
        self.mensaje.emit("Conexión establecida.")
        self.conexion_cambiada.emit(True)

    def _desconectar(self):
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
                "Seleccione el Hantek 6022BE encontrado al escanear.")
        else:
            self.cmb_recursos.setEditable(True)
            self.cmb_recursos.setToolTip(
                "Recurso VISA, por ejemplo TCPIP0::192.168.1.50::INSTR.")
        self.cmb_recursos.clear()
        self._hantek_encontrados.clear()
        self._set_ocupado(False)

    def _error(self, titulo: str, detalle: str):
        self.mensaje.emit(titulo)
        QMessageBox.warning(self, titulo, detalle)

    def cerrar(self):
        self._cerrar_instrumento()


# =============================================================================
#  PESTAÑA 2: ADQUISICIÓN Y MÉTRICAS
# =============================================================================
class PestanaMedicion(QWidget):
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
        self.spn_f1 = self._spin(1, 1e6, 1000.0, 1, " Hz", 100)
        self.spn_f1.setEnabled(False)
        self.chk_auto.toggled.connect(lambda a: self.spn_f1.setEnabled(not a))

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
        self.lienzo = LienzoMpl()
        fig = self.lienzo.fig
        self.ax_t = fig.add_subplot(2, 1, 1)
        self.ax_f = fig.add_subplot(2, 1, 2)
        self.ax_t_in = self.ax_t.twinx()
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
        self.ax_f.set_ylabel("Nivel [dBV rms]")
        self.ax_f.grid(True, alpha=0.4)
        self.ax_f.legend(loc="upper right")

        # --- Resultados
        mono = "font-family: monospace; font-weight: bold;"
        filas_resumen = [("gain", "Ganancia fundamental (salida / entrada)"),
                         ("df", "Diferencia de frecuencia"),
                         ("phase", "Desfase de salida respecto a entrada"),
                         ("delta_thd", "Cambio de distorsión (salida − entrada)")]
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
        for nombre, filas in (("Resumen", filas_resumen),
                              ("Detalles", filas_detalle)):
            pagina = QWidget()
            form = QFormLayout(pagina)
            for clave, texto in filas:
                lb = QLabel("—")
                lb.setStyleSheet(mono)
                self.lbl[clave] = lb
                form.addRow(texto + ":", lb)
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
        self.tabla.setToolTip(
            "Comparación directa de frecuencia, tensión eficaz y distorsión "
            "de entrada y salida.")
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

        der = QSplitter(Qt.Vertical)
        der.addWidget(self.lienzo)
        der.addWidget(w_inf)
        der.setStretchFactor(0, 3)
        der.setStretchFactor(1, 1)

        raiz = QHBoxLayout(self)
        raiz.addWidget(w_izq)
        raiz.addWidget(der, 1)

        # Recalcular al cambiar parámetros (sin volver a capturar)
        for s in (self.spn_rl, self.spn_vcc, self.spn_idc, self.spn_f1):
            s.valueChanged.connect(self._reanalizar)
        self.cmb_alim.currentIndexChanged.connect(self._reanalizar)
        self.chk_auto.toggled.connect(self._reanalizar)

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
                "El Hantek 6022BE solo dispone de CH1 y CH2.")
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
            self._res = AnalizadorSenal.analizar(self._t, self._v, f_esperada=f_esp)
            self._res_entrada = AnalizadorSenal.analizar(
                self._t_entrada, self._v_entrada, f_esperada=f_esp)
            avisos = ([f"Entrada: {av}" for av in self._res_entrada.avisos]
                      + [f"Salida: {av}" for av in self._res.avisos])
            if abs(self._res.f1 - self._res_entrada.f1) / self._res_entrada.f1 > 0.01:
                avisos.append(
                    "Las frecuencias difieren más de 1 %; compare el desfase con cuidado.")
            self.lbl_aviso.setText("  ".join(avisos))
        except ValueError as e:
            self._res = None
            self._res_entrada = None
            self.lbl_aviso.setText(f"⚠ {e}")
        self._pintar()
        self._actualizar_metricas()

    def _pintar(self):
        self.ln_t.set_data((self._t - self._t[0]) * 1e3, self._v)
        self.ln_t_in.set_data(
            (self._t_entrada - self._t_entrada[0]) * 1e3, self._v_entrada)
        self.ax_t.relim()
        self.ax_t.autoscale_view()
        self.ax_t_in.relim()
        self.ax_t_in.autoscale_view()
        for a in self._anotaciones:
            a.remove()
        self._anotaciones.clear()
        r = self._res
        r_entrada = self._res_entrada
        if r is not None and r_entrada is not None:
            self.ln_f_in.set_data(r_entrada.freqs, r_entrada.espectro_dbv)
            self.ln_f.set_data(r.freqs, r.espectro_dbv)
            comps = [Armonico(1, r.f1, r.v1_rms, 0.0)] + r.armonicos
            xs = [c.f for c in comps]
            ys = [20 * np.log10(max(c.v_rms, 1e-12)) for c in comps]
            self.ln_h.set_data(xs, ys)
            for c, y in zip(comps, ys):
                self._anotaciones.append(self.ax_f.annotate(
                    f"H{c.n}", (c.f, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8))
            xmax = min(r.fs / 2, max(8 * r.f1, 1.0))
            if self._xmax_prev is None or abs(xmax - self._xmax_prev) > 0.1 * xmax:
                self.ax_f.set_xlim(0, xmax)
                self._xmax_prev = xmax
            techo = ys[0]
            self.ax_f.set_ylim(techo - 110, techo + 10)
        else:
            self.ln_f_in.set_data([], [])
            self.ln_f.set_data([], [])
            self.ln_h.set_data([], [])
        self.lienzo.redibujar()

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
        ganancia = r.v1_rms / r_entrada.v1_rms
        ganancia_db = 20.0 * np.log10(ganancia)
        delta_f = r.f1 - r_entrada.f1
        delta_f_pct = abs(delta_f) / r_entrada.f1
        desfase = (r.fase_fundamental - r_entrada.fase_fundamental + np.pi) % (
            2.0 * np.pi) - np.pi
        L["gain"].setText(f"{ganancia:.3f}×  ({ganancia_db:+.2f} dB)")
        L["df"].setText(f"{delta_f:+.2f} Hz")
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
        L["f1"].setText(fmt_si(r.f1, "Hz"))
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
            vals = [f"{nombre} ({canal})", fmt_si(resultado.f1, "Hz"),
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

