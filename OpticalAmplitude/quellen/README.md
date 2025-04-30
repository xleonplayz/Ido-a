# Datenformatdokumentation für optische Amplitudenmessungen

Dieses Dokument beschreibt das Format und die Struktur der drei Hauptdateitypen, die in optischen Amplitudenmessungen verwendet werden.

## 1. Signaldaten (.dat)

### Format
- Textdatei im ASCII-Format
- Tabulatorgetrennte Werte
- Jede Zeile repräsentiert einen Messpunkt

### Struktur
```
Zeit[s]  Amplitude[V]  Phase[rad]
0.001    0.245        1.57
0.002    0.247        1.58
...      ...          ...
```

### Enthaltene Daten
- Spalte 1: Zeitpunkte in Sekunden
- Spalte 2: Gemessene Spannungsamplitude in Volt
- Spalte 3: Phasenverschiebung in Radianten

## 2. Konfigurationsdaten (.cfg)

### Format
- Textdatei im INI-Format
- Schlüssel-Wert-Paare mit Sektionen

### Struktur
```ini
[hardware]
laser_power = 10.5
detector_gain = 3
integration_time = 0.2

[measurement]
start_wavelength = 532
stop_wavelength = 632
steps = 100
repeats = 3
```

### Enthaltene Daten
- Hardware-Einstellungen (Laserleistung, Detektoreinstellungen)
- Messparameter (Wellenlängenbereich, Schrittanzahl)
- Kalibrierungsinformationen
- Experimentelle Bedingungen

## 3. Metadaten (.meta)

### Format
- JSON-formatierte Datei
- Hierarchische Struktur

### Struktur
```json
{
  "experiment": {
    "name": "Quantenpunktfluoreszenz",
    "date": "2025-04-30T14:30:00",
    "operator": "Max Mustermann"
  },
  "sample": {
    "id": "QD-A293",
    "material": "CdSe/ZnS",
    "concentration": "2.5μM"
  },
  "conditions": {
    "temperature": 293.15,
    "humidity": 45,
    "notes": "Stabiles Signal nach 10min Warmlaufzeit"
  }
}
```

### Enthaltene Daten
- Experimentdetails (Name, Zeit, Experimentator)
- Probeninformationen
- Umgebungsbedingungen
- Notizen und Beobachtungen
- Verweise auf zugehörige Dateien

Diese drei Dateitypen bilden zusammen einen vollständigen Datensatz für optische Amplitudenmessungen und ermöglichen die Reproduzierbarkeit und Nachvollziehbarkeit der Experimente.