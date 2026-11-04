#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 BANCO DE PRUEBAS VIRTUAL PARA AMPLIFICADORES DE AUDIO DE POTENCIA (Clase AB)
===============================================================================
 Instrumentación virtual en Python: PyQt5 + Matplotlib + NumPy + SciPy + PyVISA

 Módulos:
   1. Conexión VISA ........ escaneo de recursos, prueba con *IDN?, modo demo.
   2. Medición ............. captura CH1, V(t), espectro, Vrms, P, η, THD.
   3. Bode ................. asistente paso a paso de respuesta en frecuencia.

 Dependencias:
     pip install pyqt5 matplotlib numpy scipy pyvisa pyvisa-py
 (pyvisa-py es opcional si ya tiene NI-VISA / Keysight IO Libraries instalado.
  Para USB con pyvisa-py instale además: pip install pyusb)

 Ejecución:
     python banco_pruebas_audio.py
===============================================================================
"""

import csv
import re
import sys
import time
from dataclasses import dataclass, field
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
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
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


def parsear_frecuencias(texto: str) -> List[float]:
    """Convierte '20, 100, 1k, 10k' en [20.0, 100.0, 1000.0, 10000.0]."""
    resultado = []
    for token in re.split(r"[;,\s]+", texto.strip().lower()):
        if not token:
            continue
        token = token.replace("hz", "")
        mult = 1.0
        if token.endswith("k"):
            mult, token = 1e3, token[:-1]
        valor = float(token) * mult          # lanza ValueError si es inválido
        if valor <= 0:
            raise ValueError("Las frecuencias deben ser positivas.")
        resultado.append(valor)
    if not resultado:
        raise ValueError("Ingrese al menos una frecuencia.")
    return resultado


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
            v_rms_ac=v_rms_ac, f1=f1, v1_rms=v1, armonicos=armonicos,
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

    def capturar(self, canal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        time.sleep(0.05)                                # latencia "de bus"
        dur = self.ciclos / self.f_gen
        t = np.arange(self.n_muestras) / self.n_muestras * dur
        # Error de frecuencia ±0.2 % -> muestreo NO coherente (caso real)
        f_real = self.f_gen * (1 + self._rng.uniform(-0.002, 0.002))
        fase = self._rng.uniform(0, 2 * np.pi)          # disparo sin sincronía
        th = 2 * np.pi * f_real * t + fase
        if canal == 2:
            v = self.vin_rms * np.sqrt(2) * np.sin(th)
            v += self._rng.normal(0, 0.5e-3, t.size)
            return t, v
        a1 = self.vin_rms * np.sqrt(2) * self.ganancia(self.f_gen)
        prof = self.PERFIL_ARMONICOS / np.sqrt(np.sum(self.PERFIL_ARMONICOS ** 2))
        rel = self.thd_pct / 100.0 * prof
        v = a1 * np.sin(th)
        for k, (r, ph) in enumerate(zip(rel, self.FASES_ARMONICOS), start=2):
            v += a1 * r * np.sin(k * th + ph)
        v += self.offset_mv * 1e-3 + self._rng.normal(0, self.ruido_mv * 1e-3, t.size)
        return t, v


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
                resultado = {c: self.instrumento.capturar(c) for c in self.canales}
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
#  PANEL 1: CONEXIÓN VISA
# =============================================================================
class PanelConexion(QGroupBox):
    """Escaneo de recursos VISA, prueba *IDN? y modo Simulador."""
    conexion_cambiada = pyqtSignal(bool)
    mensaje = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("1. Conexión VISA", parent)
        self._rm = None
        self._instrumento: Optional[OsciloscopioVISA] = None
        self._sim = OsciloscopioSimulado()
        self._tarea: Optional[TareaAsync] = None

        self.chk_demo = QCheckBox("Modo Simulador / Demo")
        self.cmb_recursos = QComboBox()
        self.cmb_recursos.setEditable(True)
        self.cmb_recursos.setMinimumWidth(320)
        self.cmb_recursos.lineEdit().setPlaceholderText(
            "Escanee o escriba, p. ej. TCPIP0::192.168.1.50::INSTR")
        self.btn_escanear = QPushButton("Escanear")
        self.btn_conectar = QPushButton("Conectar y probar (*IDN?)")
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

        g = QGridLayout(self)
        g.addWidget(QLabel("Osciloscopio:"), 0, 0)
        g.addWidget(self.cmb_recursos, 0, 1, 1, 3)
        g.addWidget(self.btn_escanear, 0, 4)
        g.addWidget(self.cmb_escala, 1, 1)
        g.addWidget(self.btn_conectar, 1, 2)
        g.addWidget(self.btn_desconectar, 1, 3)
        g.addWidget(self.chk_demo, 1, 4)
        g.addWidget(QLabel("Estado / *IDN?:"), 2, 0)
        g.addWidget(self.lbl_idn, 2, 1, 1, 4)

        self.btn_escanear.clicked.connect(self._escanear)
        self.btn_conectar.clicked.connect(self._conectar)
        self.btn_desconectar.clicked.connect(self._desconectar)
        self.chk_demo.toggled.connect(self._cambio_demo)

        if not VISA_DISPONIBLE:
            self.lbl_idn.setText("PyVISA no instalado (pip install pyvisa pyvisa-py). "
                                 "Use el modo Simulador.")

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

    def conexion_perdida(self, detalle: str):
        """Llamado cuando una captura falla con instrumento real."""
        self._cerrar_instrumento()
        self.lbl_idn.setText("⚠ Conexión perdida")
        self.btn_desconectar.setEnabled(False)
        self.conexion_cambiada.emit(False)
        self.mensaje.emit("Conexión perdida con el osciloscopio.")
        QMessageBox.warning(
            self, "Conexión perdida",
            "Se perdió la comunicación con el osciloscopio.\n"
            "Revise el cable/red y vuelva a conectar.\n\nDetalle: " + detalle)

    # ---- Lógica interna ----------------------------------------------------
    def _set_ocupado(self, ocupado: bool):
        for w in (self.btn_escanear, self.btn_conectar, self.cmb_recursos,
                  self.cmb_escala):
            w.setEnabled(not ocupado and not self.es_demo())
        self.btn_desconectar.setEnabled(not ocupado and self._instrumento is not None
                                        and not self.es_demo())

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
        self.mensaje.emit("Escaneando recursos VISA…")
        self._lanzar(lambda: list(self._obtener_rm().list_resources()),
                     self._escaneo_ok,
                     lambda m: self._error("Error al escanear", m))

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
        self.btn_desconectar.setEnabled(True)
        self.mensaje.emit("Conexión establecida.")
        self.conexion_cambiada.emit(True)

    def _desconectar(self):
        self._cerrar_instrumento()
        self.lbl_idn.setText("Sin conexión")
        self.btn_desconectar.setEnabled(False)
        self.mensaje.emit("Desconectado.")
        self.conexion_cambiada.emit(False)

    def _cerrar_instrumento(self):
        if self._instrumento is not None:
            self._instrumento.cerrar()
            self._instrumento = None

    def _cambio_demo(self, activo: bool):
        for w in (self.btn_escanear, self.btn_conectar, self.cmb_recursos,
                  self.cmb_escala):
            w.setEnabled(not activo)
        if activo:
            self.btn_desconectar.setEnabled(False)
            self.lbl_idn.setText("MODO DEMO — " + self._sim.identificar())
            self.conexion_cambiada.emit(True)
        else:
            self.btn_desconectar.setEnabled(self._instrumento is not None)
            self.lbl_idn.setText(self._instrumento.identificar()
                                 if self._instrumento else "Sin conexión")
            self.conexion_cambiada.emit(self._instrumento is not None)

    def _error(self, titulo: str, detalle: str):
        self.mensaje.emit(titulo)
        QMessageBox.warning(self, titulo, detalle)

    def cerrar(self):
        self._cerrar_instrumento()


# =============================================================================
#  PESTAÑA 2: ADQUISICIÓN Y MÉTRICAS
# =============================================================================
class PestanaMedicion(QWidget):
    capturando = pyqtSignal(bool)

    def __init__(self, conexion: PanelConexion, parent=None):
        super().__init__(parent)
        self.conexion = conexion
        self._hilo: Optional[HiloCaptura] = None
        self._t: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._res: Optional[ResultadoAnalisis] = None
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
        self.cmb_canal = QComboBox()
        self.cmb_canal.addItems(["CH1", "CH2", "CH3", "CH4"])
        self.chk_auto = QCheckBox("Auto-detectar fundamental")
        self.chk_auto.setChecked(True)
        self.spn_f1 = self._spin(1, 1e6, 1000.0, 1, " Hz", 100)
        self.spn_f1.setEnabled(False)
        self.chk_auto.toggled.connect(lambda a: self.spn_f1.setEnabled(not a))

        grp = QGroupBox("2. Parámetros de la prueba")
        f = QFormLayout(grp)
        f.addRow("Carga R_L:", self.spn_rl)
        f.addRow("", presets)
        f.addRow("Alimentación V_CC:", self.spn_vcc)
        f.addRow("Corriente I_DC (multímetro):", self.spn_idc)
        f.addRow("Tipo de fuente:", self.cmb_alim)
        f.addRow("Canal de salida:", self.cmb_canal)
        f.addRow(self.chk_auto, self.spn_f1)

        # --- Parámetros del simulador (visible solo en demo)
        self.sim_f = self._spin(10, 100e3, 1000, 1, " Hz", 100)
        self.sim_vin = self._spin(0.001, 5, 0.4, 3, " Vrms", 0.05)
        self.sim_thd = self._spin(0, 20, 0.5, 3, " %", 0.1)
        self.sim_ruido = self._spin(0, 500, 2.0, 2, " mVrms", 0.5)
        self.grp_sim = QGroupBox("Simulador (generador + amplificador)")
        fs = QFormLayout(self.grp_sim)
        fs.addRow("f del generador:", self.sim_f)
        fs.addRow("Vin del generador:", self.sim_vin)
        fs.addRow("THD inyectado:", self.sim_thd)
        fs.addRow("Ruido:", self.sim_ruido)

        # --- Captura
        self.btn_unica = QPushButton("Capturar una vez")
        self.btn_continuo = QPushButton("Captura continua")
        self.btn_continuo.setCheckable(True)
        self.spn_periodo = QSpinBox()
        self.spn_periodo.setRange(100, 10000)
        self.spn_periodo.setValue(500)
        self.spn_periodo.setSuffix(" ms")
        self.btn_csv = QPushButton("Guardar forma de onda (CSV)")
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
        self.ln_t, = self.ax_t.plot([], [], lw=1.0, color="tab:blue")
        self.ax_t.set_title("Señal en el dominio del tiempo")
        self.ax_t.set_xlabel("Tiempo [ms]")
        self.ax_t.set_ylabel("V(t) [V]")
        self.ax_t.grid(True, alpha=0.4)
        self.ln_f, = self.ax_f.plot([], [], lw=0.8, color="tab:green")
        self.ln_h, = self.ax_f.plot([], [], "ro", ms=5)
        self.ax_f.set_title("Espectro (FFT, ventana Blackman-Harris)")
        self.ax_f.set_xlabel("Frecuencia [Hz]")
        self.ax_f.set_ylabel("Nivel [dBV rms]")
        self.ax_f.grid(True, alpha=0.4)

        # --- Resultados
        mono = "font-family: monospace; font-weight: bold;"
        self.lbl = {}
        filas = [("f1", "Frecuencia fundamental f₁"),
                 ("vpp", "Tensión pico a pico"),
                 ("vpk", "Tensión de pico (sin DC)"),
                 ("vdc", "Offset DC"),
                 ("vrms", "V_rms (fórmula directa)"),
                 ("vac", "V_rms (AC, sin offset)"),
                 ("prms", "Potencia RMS  V²/R_L"),
                 ("pdc", "Potencia de alimentación"),
                 ("pdis", "Potencia disipada (P_DC − P_RMS)"),
                 ("eta", "Eficiencia η"),
                 ("thd", "THD (H2…H5)"),
                 ("thdn", "THD+N")]
        grp_res = QGroupBox("Métricas")
        fr = QFormLayout(grp_res)
        for clave, texto in filas:
            lb = QLabel("—")
            lb.setStyleSheet(mono)
            self.lbl[clave] = lb
            fr.addRow(texto + ":", lb)
        self.lbl["vac"].setToolTip(
            "La potencia usa el valor eficaz AC: el offset DC de salida no "
            "entrega potencia de audio útil a la carga.")

        self.tabla = QTableWidget(5, 4)
        self.tabla.setHorizontalHeaderLabels(["Armónico", "f [Hz]", "V_rms", "dBc"])
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setMaximumHeight(210)
        self.lbl_aviso = QLabel("")
        self.lbl_aviso.setWordWrap(True)
        self.lbl_aviso.setStyleSheet("color: #b35900; font-weight: bold;")

        inferior = QHBoxLayout()
        inferior.addWidget(grp_res, 1)
        col_tab = QVBoxLayout()
        col_tab.addWidget(self.tabla)
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
        if self.conexion.es_demo():
            self._aplicar_params_sim()
        else:
            inst.estilo = self.conexion.estilo_escala()
        self._canal = self.cmb_canal.currentIndex() + 1
        self._hilo = HiloCaptura(inst, (self._canal,), continuo,
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
        self.capturando.emit(b)

    def _on_datos(self, datos: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        t, v = datos[self._canal]
        self._t, self._v = t, v
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
        if self._t is None:
            return
        f_esp = None if self.chk_auto.isChecked() else self.spn_f1.value()
        try:
            self._res = AnalizadorSenal.analizar(self._t, self._v, f_esperada=f_esp)
            self.lbl_aviso.setText("  ".join(self._res.avisos))
        except ValueError as e:
            self._res = None
            self.lbl_aviso.setText(f"⚠ {e}")
        self._pintar()
        self._actualizar_metricas()

    def _pintar(self):
        t, v = self._t, self._v
        self.ln_t.set_data((t - t[0]) * 1e3, v)
        self.ax_t.relim()
        self.ax_t.autoscale_view()
        for a in self._anotaciones:
            a.remove()
        self._anotaciones.clear()
        r = self._res
        if r is not None:
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
            self.ln_f.set_data([], [])
            self.ln_h.set_data([], [])
        self.lienzo.redibujar()

    def _actualizar_metricas(self):
        L = self.lbl
        r = self._res
        if r is None:
            for lb in L.values():
                lb.setText("—")
            self.tabla.clearContents()
            return
        rl, vcc, idc = self.spn_rl.value(), self.spn_vcc.value(), self.spn_idc.value()
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
            if eta > 100:
                self.lbl_aviso.setText(self.lbl_aviso.text() +
                                       "  η > 100 %: revise V_CC, I_DC, R_L o el tipo de fuente.")
        else:
            L["pdc"].setText("— (ingrese I_DC)")
            L["pdis"].setText("—")
            L["eta"].setText("— (ingrese I_DC)")
        if np.isfinite(r.thd_pct):
            thd_db = 20 * np.log10(max(r.thd_pct, 1e-9) / 100)
            L["thd"].setText(f"{r.thd_pct:.4f} %  ({thd_db:.1f} dB)")
        else:
            L["thd"].setText("—")
        L["thdn"].setText(f"{r.thdn_pct:.4f} %")

        comps = [Armonico(1, r.f1, r.v1_rms, 0.0)] + r.armonicos
        self.tabla.clearContents()
        for i, c in enumerate(comps):
            vals = [f"H{c.n}" + (" (fund.)" if c.n == 1 else ""),
                    f"{c.f:.1f}", fmt_si(c.v_rms, "V"),
                    "0.0" if c.n == 1 else f"{c.dbc:.1f}"]
            for j, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                self.tabla.setItem(i, j, it)

    def _guardar_csv(self):
        if self._t is None:
            QMessageBox.information(self, "CSV", "Aún no hay una captura.")
            return
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar forma de onda",
                                              "forma_de_onda.csv", "CSV (*.csv)")
        if not ruta:
            return
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["t_s", "v_V"])
                w.writerows(zip(self._t.tolist(), self._v.tolist()))
        except OSError as e:
            QMessageBox.warning(self, "Error al guardar", str(e))


# =============================================================================
#  PESTAÑA 3: BARRIDO DE FRECUENCIA (DIAGRAMA DE BODE)
# =============================================================================
@dataclass
class PuntoBode:
    f_nominal: float
    f_detectada: float
    vin: float
    vout: float
    ganancia_db: float


class PestanaBode(QWidget):
    capturando = pyqtSignal(bool)

    def __init__(self, conexion: PanelConexion, parent=None):
        super().__init__(parent)
        self.conexion = conexion
        self._frecs: List[float] = []
        self._idx = 0
        self._puntos: List[PuntoBode] = []
        self._hilo: Optional[HiloCaptura] = None
        self._activo = False
        self._construir_ui()
        self._actualizar_botones()

    def _construir_ui(self):
        self.txt_frecs = QLineEdit("20, 100, 1k, 10k, 20k")
        self.txt_frecs.setToolTip("Separe con comas. Admite sufijo k (1k = 1000 Hz).")
        self.cmb_vin = QComboBox()
        self.cmb_vin.addItems(["Vin manual (Vrms)", "Vin medido en CH2"])
        self.spn_vin = QDoubleSpinBox()
        self.spn_vin.setRange(1e-4, 100)
        self.spn_vin.setDecimals(4)
        self.spn_vin.setValue(0.4)
        self.spn_vin.setSuffix(" Vrms")
        self.cmb_canal = QComboBox()
        self.cmb_canal.addItems(["CH1", "CH2", "CH3", "CH4"])

        grp = QGroupBox("Configuración del barrido")
        f = QFormLayout(grp)
        f.addRow("Frecuencias:", self.txt_frecs)
        f.addRow("Fuente de Vin:", self.cmb_vin)
        f.addRow("Vin del generador:", self.spn_vin)
        f.addRow("Canal de salida:", self.cmb_canal)

        self.btn_iniciar = QPushButton("▶ Iniciar asistente")
        self.btn_capturar = QPushButton("📥 Capturar punto")
        self.btn_omitir = QPushButton("Omitir punto")
        self.btn_cancelar = QPushButton("Cancelar")
        self.btn_limpiar = QPushButton("Limpiar resultados")
        self.btn_csv = QPushButton("Exportar CSV")
        self.btn_iniciar.clicked.connect(self._iniciar)
        self.btn_capturar.clicked.connect(self._capturar)
        self.btn_omitir.clicked.connect(self._omitir)
        self.btn_cancelar.clicked.connect(self._cancelar)
        self.btn_limpiar.clicked.connect(self._limpiar)
        self.btn_csv.clicked.connect(self._exportar_csv)

        self.lbl_paso = QLabel("Pulse «Iniciar asistente» para comenzar.")
        self.lbl_paso.setWordWrap(True)
        self.lbl_paso.setStyleSheet(
            "background:#fff8dc; border:1px solid #d4b106; padding:8px; "
            "font-size: 13px;")
        self.lbl_paso.setMinimumHeight(110)
        self.barra = QProgressBar()
        self.barra.setValue(0)
        self.lbl_bw = QLabel("Ancho de banda (−3 dB): —")
        self.lbl_bw.setWordWrap(True)
        self.lbl_bw.setStyleSheet("font-weight: bold;")

        grp_w = QGroupBox("Asistente")
        v = QVBoxLayout(grp_w)
        v.addWidget(self.lbl_paso)
        v.addWidget(self.barra)
        v.addWidget(self.btn_iniciar)
        v.addWidget(self.btn_capturar)
        h = QHBoxLayout()
        h.addWidget(self.btn_omitir)
        h.addWidget(self.btn_cancelar)
        v.addLayout(h)

        izq = QVBoxLayout()
        izq.addWidget(grp)
        izq.addWidget(grp_w)
        izq.addWidget(self.lbl_bw)
        izq.addWidget(self.btn_limpiar)
        izq.addWidget(self.btn_csv)
        izq.addStretch(1)
        w_izq = QWidget()
        w_izq.setLayout(izq)
        w_izq.setMaximumWidth(430)

        self.lienzo = LienzoMpl()
        self.ax = self.lienzo.fig.add_subplot(1, 1, 1)
        self.tabla = QTableWidget(0, 5)
        self.tabla.setHorizontalHeaderLabels(
            ["f nominal [Hz]", "f detectada [Hz]", "Vin [Vrms]", "Vout [Vrms]",
             "Ganancia [dB]"])
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setMaximumHeight(190)

        der = QVBoxLayout()
        der.addWidget(self.lienzo, 1)
        der.addWidget(self.tabla)
        raiz = QHBoxLayout(self)
        raiz.addWidget(w_izq)
        raiz.addLayout(der, 1)
        self._dibujar()

    # ---- Estado del asistente ---------------------------------------------
    def _actualizar_botones(self):
        capt = self._hilo is not None and self._hilo.isRunning()
        self.btn_iniciar.setEnabled(not self._activo and not capt)
        self.btn_capturar.setEnabled(self._activo and not capt)
        self.btn_omitir.setEnabled(self._activo and not capt)
        self.btn_cancelar.setEnabled(self._activo and not capt)
        self.txt_frecs.setEnabled(not self._activo)

    def _iniciar(self):
        if self.conexion.instrumento() is None:
            QMessageBox.information(self, "Sin instrumento",
                                    "Conecte el osciloscopio o active el modo Simulador.")
            return
        try:
            self._frecs = parsear_frecuencias(self.txt_frecs.text())
        except ValueError as e:
            QMessageBox.warning(self, "Frecuencias inválidas", str(e))
            return
        self._idx = 0
        self._puntos.clear()
        self._activo = True
        self._refrescar_tabla()
        self._dibujar()
        self._mostrar_paso()
        self._actualizar_botones()

    def _mostrar_paso(self):
        n = len(self._frecs)
        self.barra.setMaximum(n)
        self.barra.setValue(self._idx)
        if self._idx >= n:
            self._finalizar()
            return
        f = self._frecs[self._idx]
        extra = ("\n(Demo: el simulador ajusta su generador automáticamente.)"
                 if self.conexion.es_demo() else "")
        src = ("Mantenga CH2 conectado a la entrada del amplificador."
               if self.cmb_vin.currentIndex() == 1 else
               f"Indique en «Vin» el valor eficaz que fijó ({self.spn_vin.value():g} Vrms).")
        self.lbl_paso.setText(
            f"<b>Paso {self._idx + 1} de {n}</b><br>"
            f"1) Ajuste el generador a <b>{fmt_si(f, 'Hz')}</b>, senoidal.<br>"
            f"2) Ajuste la base de tiempo para ver 10 ciclos o más.<br>"
            f"3) {src}<br>4) Pulse «Capturar punto».{extra}")

    def _finalizar(self):
        self._activo = False
        self.lbl_paso.setText("✅ Barrido completado. Puede exportar el CSV o "
                              "iniciar otro barrido.")
        self._actualizar_botones()

    def _omitir(self):
        self._idx += 1
        self._mostrar_paso()
        self._actualizar_botones()

    def _cancelar(self):
        self._activo = False
        self.lbl_paso.setText("Asistente cancelado.")
        self.barra.setValue(0)
        self._actualizar_botones()

    def _limpiar(self):
        self._puntos.clear()
        self._refrescar_tabla()
        self._dibujar()

    # ---- Captura de un punto ----------------------------------------------
    def _capturar(self):
        inst = self.conexion.instrumento()
        if inst is None:
            return
        f = self._frecs[self._idx]
        if self.conexion.es_demo():
            sim = self.conexion.simulador
            sim.f_gen = f
            sim.vin_rms = self.spn_vin.value()
        else:
            inst.estilo = self.conexion.estilo_escala()
        self._canal_out = self.cmb_canal.currentIndex() + 1
        canales = [self._canal_out]
        if self.cmb_vin.currentIndex() == 1 and self._canal_out != 2:
            canales.append(2)
        self._hilo = HiloCaptura(inst, canales, continuo=False, parent=self)
        self._hilo.datos.connect(self._on_datos)
        self._hilo.error.connect(self._on_error)
        self._hilo.finished.connect(self._on_hilo_terminado)
        self.capturando.emit(True)
        self.conexion.setEnabled(False)
        self._hilo.start()
        self._actualizar_botones()

    def _on_hilo_terminado(self):
        self.capturando.emit(False)
        self.conexion.setEnabled(True)
        self._actualizar_botones()

    def _on_error(self, msg: str):
        if self.conexion.es_demo():
            QMessageBox.warning(self, "Error en el simulador", msg)
        else:
            self._cancelar()
            self.conexion.conexion_perdida(msg)

    def _on_datos(self, datos):
        f = self._frecs[self._idx]
        try:
            t, v = datos[self._canal_out]
            r_out = AnalizadorSenal.analizar(t, v)
            if self.cmb_vin.currentIndex() == 1:
                t2, v2 = datos[2]
                vin = AnalizadorSenal.analizar(t2, v2).v1_rms
            else:
                vin = self.spn_vin.value()
        except ValueError as e:
            QMessageBox.warning(self, "No se pudo analizar la captura",
                                f"{e}\n\nAjuste el osciloscopio e intente de nuevo.")
            return
        vout = r_out.v1_rms
        if abs(r_out.f1 - f) / f > 0.10:
            resp = QMessageBox.question(
                self, "Frecuencia inesperada",
                f"Se esperaba {fmt_si(f, 'Hz')} pero se detectó "
                f"{fmt_si(r_out.f1, 'Hz')}.\n¿Aceptar el punto de todos modos?")
            if resp != QMessageBox.Yes:
                return
        g_db = 20.0 * np.log10(vout / vin)
        self._puntos.append(PuntoBode(f, r_out.f1, vin, vout, g_db))
        self._refrescar_tabla()
        self._dibujar()
        self._idx += 1
        self._mostrar_paso()

    # ---- Resultados --------------------------------------------------------
    def _refrescar_tabla(self):
        self.tabla.setRowCount(len(self._puntos))
        for i, p in enumerate(self._puntos):
            vals = [f"{p.f_nominal:g}", f"{p.f_detectada:.1f}", f"{p.vin:.4f}",
                    f"{p.vout:.4f}", f"{p.ganancia_db:+.2f}"]
            for j, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                self.tabla.setItem(i, j, it)

    @staticmethod
    def _ancho_banda(f: np.ndarray, g: np.ndarray):
        """Cruces de −3 dB respecto a la ganancia a 1 kHz (o al máximo)."""
        lf = np.log10(f)
        ref = float(np.interp(3.0, lf, g)) if f[0] <= 1000 <= f[-1] else float(g.max())
        d = g - (ref - 3.0)
        f_baja = f_alta = None
        for i in range(len(f) - 1):
            if d[i] * d[i + 1] < 0 or d[i] == 0:
                fx = 10 ** (lf[i] + (0 - d[i]) * (lf[i + 1] - lf[i]) / (d[i + 1] - d[i])
                            if d[i + 1] != d[i] else lf[i])
                if d[i + 1] > d[i] and f_baja is None:
                    f_baja = fx
                elif d[i + 1] < d[i]:
                    f_alta = fx
        return ref, f_baja, f_alta

    def _dibujar(self):
        ax = self.ax
        ax.clear()
        ax.set_xscale("log")
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel("Ganancia [dB]")
        ax.set_title("Diagrama de Bode (magnitud)")
        ax.grid(True, which="both", alpha=0.4)
        ax.axvspan(20, 20e3, color="tab:green", alpha=0.06, label="Banda de audio")
        texto = "Ancho de banda (−3 dB): —"
        if self._puntos:
            pts = sorted(self._puntos, key=lambda p: p.f_nominal)
            f = np.array([p.f_nominal for p in pts])
            g = np.array([p.ganancia_db for p in pts])
            ax.plot(f, g, "o-", color="tab:blue", label="Medido")
            for fi, gi in zip(f, g):
                ax.annotate(f"{gi:+.1f}", (fi, gi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8)
            if len(f) >= 2:
                ref, fl, fh = self._ancho_banda(f, g)
                ax.axhline(ref - 3, color="tab:red", ls="--", lw=1,
                           label=f"−3 dB ({ref - 3:.1f} dB)")
                low = fmt_si(fl, "Hz") if fl else f"< {fmt_si(f[0], 'Hz')} (fuera de rango)"
                high = fmt_si(fh, "Hz") if fh else f"> {fmt_si(f[-1], 'Hz')} (fuera de rango)"
                texto = (f"Ancho de banda (−3 dB, interpolado entre puntos):\n"
                         f"f_baja = {low}   |   f_alta = {high}")
            ax.legend(loc="lower center")
        self.lbl_bw.setText(texto)
        self.lienzo.redibujar()

    def _exportar_csv(self):
        if not self._puntos:
            QMessageBox.information(self, "CSV", "No hay puntos medidos.")
            return
        ruta, _ = QFileDialog.getSaveFileName(self, "Exportar Bode", "bode.csv",
                                              "CSV (*.csv)")
        if not ruta:
            return
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["f_nominal_Hz", "f_detectada_Hz", "Vin_Vrms",
                            "Vout_Vrms", "Ganancia_dB"])
                for p in sorted(self._puntos, key=lambda p: p.f_nominal):
                    w.writerow([p.f_nominal, p.f_detectada, p.vin, p.vout,
                                p.ganancia_db])
        except OSError as e:
            QMessageBox.warning(self, "Error al guardar", str(e))

    def detener(self):
        if self._hilo is not None and self._hilo.isRunning():
            self._hilo.detener()
            self._hilo.wait(6000)


# =============================================================================
#  VENTANA PRINCIPAL
# =============================================================================
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Banco de Pruebas — Amplificadores de Audio Clase AB")
        self.resize(1400, 900)

        self.conexion = PanelConexion()
        self.medicion = PestanaMedicion(self.conexion)
        self.bode = PestanaBode(self.conexion)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.medicion, "Medición (Vrms · P · η · THD)")
        self.tabs.addTab(self.bode, "Respuesta en frecuencia (Bode)")

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(self.conexion)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # Evita capturas simultáneas desde pestañas distintas
        self.medicion.capturando.connect(lambda b: self.tabs.setTabEnabled(1, not b))
        self.bode.capturando.connect(lambda b: self.tabs.setTabEnabled(0, not b))
        self.conexion.mensaje.connect(self.statusBar().showMessage)
        self.statusBar().showMessage(
            "Listo. Active el modo Simulador o conecte un osciloscopio.")

    def closeEvent(self, ev):
        self.medicion.detener()
        self.bode.detener()
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
