# Prototipo Android: verificación en el teléfono (A8)

Objetivo específico 5 de la planificación. La app toma la foto y la clasifica en el teléfono, sin conexión, con el
`.tflite` de la tesis. Luego la contrasta con el mueble declarado y advierte si es una captura inválida, contenido
ajeno u otro mueble. Está escrita en Java y su única dependencia es el runtime de TFLite (LiteRT).

| Archivo | Qué hace |
|---|---|
| [`Clasificador.java`](app/src/main/java/ec/usfq/tesis/Clasificador.java) | Repite el preprocesamiento del entrenamiento (EXIF, 320 px y luego 224 × 224) y la decisión condicionada al convenio de `tesis.metrics.conditional_pred` |
| [`MainActivity.java`](app/src/main/java/ec/usfq/tesis/MainActivity.java) | Pantalla: elegir el mueble declarado, tomar o elegir la foto y ver el veredicto; la misma pantalla atiende la llamada desde ODK |
| [`Rafaga.java`](app/src/main/java/ec/usfq/tesis/Rafaga.java) | Medición por ráfagas sobre fotos reales; la lanza [`tesis/device.py`](../tesis/device.py) por adb |

## Desde ODK Collect

La app se abre desde una pregunta de foto del formulario y le devuelve la foto:

```
type: image   appearance: ex:ec.usfq.tesis.VERIFICAR(mueble=${mueble_nombre}, convenio=${muebles_del_local})
```

- `mueble` va en la nomenclatura del formulario (`CHECK OUT | LATERAL SIMPLE | …`).
- `convenio` es opcional: los muebles del local separados por comas, a los que se restringe la decisión.
- `camara` es opcional y fija la app de cámara (`'com.jeyluta.timestampcamerafree'`).

Se prueba en una copia del formulario, no en uno que esté en uso.

## Compilar

Se necesitan JDK 17 y el SDK de Android (plataforma 35, build-tools 34):

```powershell
$env:JAVA_HOME = "$env:LOCALAPPDATA\Android\jdk17"; $env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
android\gradlew.bat -p android assembleRelease        # android\app\build\outputs\apk\release\app-release.apk
```

El APK no lleva el modelo: `tesis.device --app` instala la app y le deja el `.tflite` en el teléfono. Para llevarlo
dentro del paquete, se copia a `app/src/main/assets/modelo.tflite`.

## Medir en el teléfono

La latencia de extremo a extremo se mide con el protocolo de ráfagas de la sección 5.6, sobre 120 fotos reales de ODK
de locales de prueba. Además, se comprueba que el teléfono clasifica igual que la evaluación:

```powershell
python -m tesis.fotos_app                                                          # data/fotos_app/
python -m tesis.device runs/odk --app android\app\build\outputs\apk\release\app-release.apk --fotos data/fotos_app
python -m tesis.fotos_app runs/odk/A_mobilenetv3/model_fp16.tflite runs/odk/fotos_app_por_foto.csv    # paridad
```

En otro computador basta el paquete que arma `python -m tesis.fotos_app --paquete`, con Python y el teléfono. Trae el
APK, `device.py`, las fotos y un LEEME con los pasos.
