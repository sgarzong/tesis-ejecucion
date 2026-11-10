package ec.usfq.tesis;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Verificación en el teléfono, sin conexión (objetivo 5): toma la foto, la clasifica, la contrasta con el mueble
 * declarado y advierte si no sirve como evidencia.
 * <p>
 * Desde ODK Collect, en la pregunta de foto del mueble (XLSForm):
 * <pre>
 *   type: image   appearance: ex:ec.usfq.tesis.VERIFICAR(mueble=${mueble_nombre}, convenio=${muebles_del_local})
 * </pre>
 * mueble: lo declarado, en la nomenclatura del formulario. convenio (opcional): los muebles contratados del local,
 * separados por comas; sin él, la decisión no se restringe al convenio. camara (opcional): paquete de la app de
 * cámara, p. ej. 'com.jeyluta.timestampcamerafree'. La foto vuelve a ODK como ClipData, haya alerta o no: la
 * persona decide si la repite.
 */
public class MainActivity extends Activity {
    static final String ACCION = "ec.usfq.tesis.VERIFICAR";
    private static final int CAMARA = 1, GALERIA = 2;
    private static final ExecutorService HILO = Executors.newSingleThreadExecutor();  // el modelo, en un solo hilo

    private Clasificador modelo;
    private boolean odk;
    private int declarado = -1;
    private final Set<Integer> convenio = new HashSet<>();
    private Uri foto, pendiente;  // la última foto verificada y la que está tomando la cámara
    private Spinner mueble;
    private ImageView vista;
    private TextView veredicto;
    private Button usar;

    @Override
    protected void onCreate(Bundle estado) {
        super.onCreate(estado);
        Intent i = getIntent();
        odk = ACCION.equals(i.getAction());
        if (odk) {
            declarado = Clasificador.clase(i.getStringExtra("mueble"));
            String c = i.getStringExtra("convenio");
            if (c != null)
                for (String s : c.split(",")) {
                    int k = Clasificador.clase(s);
                    if (k >= 0) convenio.add(k);
                }
        }
        if (estado != null) {  // la cámara pudo cerrar la app para liberar memoria
            foto = estado.getParcelable("foto");
            pendiente = estado.getParcelable("pendiente");
        }
        try {
            modelo = new Clasificador(this, 4);
        } catch (IOException e) {
            throw new IllegalStateException("sin modelo.tflite en el paquete", e);
        }
        setContentView(pantalla());
        if (odk && estado == null) capturar();
    }

    private View pantalla() {
        int m = Math.round(16 * getResources().getDisplayMetrics().density);
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(m, m, m, m);
        TextView t = new TextView(this);
        t.setText(odk ? "Mueble declarado: " + (declarado < 0 ? "fuera de la taxonomía" : Clasificador.NOMBRES[declarado])
                : "Mueble declarado");
        l.addView(t);
        if (!odk) {
            mueble = new Spinner(this);
            mueble.setId(1);  // para que conserve la selección si la app se recrea
            mueble.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
                    Arrays.copyOf(Clasificador.NOMBRES, Clasificador.CAPTURA_INVALIDA)));
            l.addView(mueble);
        }
        l.addView(boton(odk ? "Repetir la foto" : "Tomar foto", v -> capturar()));
        if (!odk)
            l.addView(boton("Elegir de la galería", v -> startActivityForResult(
                    new Intent(Intent.ACTION_GET_CONTENT).setType("image/*"), GALERIA)));
        veredicto = new TextView(this);
        veredicto.setTextSize(20);
        veredicto.setPadding(0, m, 0, m);
        l.addView(veredicto);
        if (odk) {
            usar = boton("Usar esta foto", v -> devolver());
            usar.setEnabled(false);
            l.addView(usar);
        }
        vista = new ImageView(this);
        vista.setAdjustViewBounds(true);
        l.addView(vista);
        ScrollView s = new ScrollView(this);
        s.addView(l);
        return s;
    }

    private Button boton(String texto, View.OnClickListener accion) {
        Button b = new Button(this);
        b.setText(texto);
        b.setOnClickListener(accion);
        return b;
    }

    private void capturar() {
        ContentValues v = new ContentValues();
        v.put(MediaStore.Images.Media.DISPLAY_NAME, "mueble_" + System.currentTimeMillis() + ".jpg");
        v.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        v.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Verificador");
        pendiente = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v);
        Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE).putExtra(MediaStore.EXTRA_OUTPUT, pendiente);
        String camara = getIntent().getStringExtra("camara");
        try {
            startActivityForResult(camara == null ? i : new Intent(i).setPackage(camara), CAMARA);
        } catch (ActivityNotFoundException e) {  // esa app no toma fotos para otras: la cámara del sistema
            Toast.makeText(this, camara + " no responde; se abre la cámara del sistema", Toast.LENGTH_LONG).show();
            startActivityForResult(i, CAMARA);
        }
    }

    @Override
    protected void onActivityResult(int pedido, int resultado, Intent datos) {
        if (resultado != RESULT_OK) {
            if (pedido == CAMARA && pendiente != null) getContentResolver().delete(pendiente, null, null);  // vacía
            pendiente = null;
            if (odk && foto == null) {  // ODK sin ninguna foto: vuelve sin respuesta
                setResult(RESULT_CANCELED);
                finish();
            }
            return;
        }
        foto = pedido == GALERIA ? datos.getData() : pendiente;
        pendiente = null;
        if (!odk) declarado = mueble.getSelectedItemPosition();
        Uri f = foto;
        int d = declarado;
        veredicto.setText("Verificando…");
        if (usar != null) usar.setEnabled(false);
        HILO.execute(() -> {
            try {
                long t0 = SystemClock.elapsedRealtimeNanos();  // la foto ya está en disco: desde aquí, extremo a extremo
                byte[] b;
                try (InputStream in = getContentResolver().openInputStream(f)) {
                    b = Clasificador.leer(in);
                }
                Bitmap derecha = Clasificador.reducir(b);
                int c = Clasificador.decidir(modelo.clasificar(derecha), convenio);
                long ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000;
                runOnUiThread(() -> mostrar(derecha, c, d, ms));
            } catch (IOException | RuntimeException e) {
                runOnUiThread(() -> veredicto.setText("No se pudo leer la foto: " + e.getMessage()));
            }
        });
    }

    private void mostrar(Bitmap derecha, int c, int d, long ms) {
        boolean alerta = c >= Clasificador.CAPTURA_INVALIDA || (d >= 0 && c != d);
        veredicto.setText(mensaje(c, d) + "\n(" + ms + " ms)");
        veredicto.setTextColor(alerta ? 0xFFC62828 : 0xFF2E7D32);
        vista.setImageBitmap(derecha);
        if (usar != null) usar.setEnabled(true);
    }

    static String mensaje(int c, int d) {
        if (c == Clasificador.CAPTURA_INVALIDA)
            return "La foto no sirve como evidencia: oscura, movida, desenfocada o mal encuadrada. Repítela.";
        if (c == Clasificador.CONTENIDO_AJENO) return "La foto no muestra un mueble de exhibición. Repítela.";
        if (d < 0) return "La foto muestra: " + Clasificador.NOMBRES[c] + ".";
        if (c != d)
            return "La foto parece de un mueble tipo " + Clasificador.NOMBRES[c] + ", pero se declaró "
                    + Clasificador.NOMBRES[d] + ". Revisa la fila elegida o repite la foto.";
        return "Coincide con lo declarado: " + Clasificador.NOMBRES[d] + ".";
    }

    private void devolver() {  // https://docs.getodk.org/collect-external-apps/: ODK toma el archivo del ClipData
        Intent r = new Intent();
        r.setClipData(ClipData.newRawUri(null, foto));
        r.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        setResult(RESULT_OK, r);
        finish();
    }

    @Override
    protected void onSaveInstanceState(Bundle estado) {
        super.onSaveInstanceState(estado);
        estado.putParcelable("foto", foto);
        estado.putParcelable("pendiente", pendiente);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Clasificador m = modelo;
        HILO.execute(m::close);  // después de una verificación en curso
    }
}
