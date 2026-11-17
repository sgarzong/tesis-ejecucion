package ec.usfq.tesis;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.ContentValues;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.HapticFeedbackConstants;
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
import java.util.Locale;
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
 * cámara, p. ej. 'com.jeyluta.timestampcamerafree'. La foto vuelve a ODK como ClipData. Sin alerta vuelve sola,
 * para no interrumpir la visita; con alerta, la persona decide si la repite o la usa de todos modos.
 */
public class MainActivity extends Activity {
    static final String ACCION = "ec.usfq.tesis.VERIFICAR";
    private static final int CAMARA = 1, GALERIA = 2;
    private static final int VERDE = 0xFF2E7D32, NARANJA = 0xFFE65100, ROJO = 0xFFC62828, GRIS = 0xFF616161,
            AZUL = 0xFF1565C0;
    private static final ExecutorService HILO = Executors.newSingleThreadExecutor();  // el modelo, en un solo hilo

    private Clasificador modelo;
    private boolean odk;
    private int declarado = -1;
    private final Set<Integer> convenio = new HashSet<>();
    private Uri foto, pendiente;  // la última foto verificada y la que está tomando la cámara
    private Spinner mueble;
    private ImageView vista;
    private LinearLayout tarjeta;  // el veredicto: titular y detalle sobre un fondo del color del resultado
    private TextView encabezado, titular, detalle;
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
            TextView t = new TextView(this);
            t.setText("Falta el modelo: instálalo con python device.py <carpeta> --app verificador.apk");
            t.setPadding(48, 48, 48, 48);
            setContentView(t);
            return;
        }
        setContentView(pantalla());
        if (odk && estado == null) capturar();
    }

    /** De arriba abajo: lo declarado, el veredicto, la foto y, al alcance del pulgar, la acción principal. */
    private View pantalla() {
        int m = dp(16);
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(m, m, m, m);
        encabezado = new TextView(this);
        encabezado.setText(declarado());
        l.addView(encabezado);
        if (!odk) {
            mueble = new Spinner(this);
            mueble.setId(1);  // para que conserve la selección si la app se recrea
            mueble.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
                    Arrays.copyOf(Clasificador.NOMBRES, Clasificador.CAPTURA_INVALIDA)));
            l.addView(mueble);
        }
        tarjeta = new LinearLayout(this);
        tarjeta.setOrientation(LinearLayout.VERTICAL);
        tarjeta.setPadding(m, m, m, m);
        GradientDrawable fondo = new GradientDrawable();
        fondo.setCornerRadius(dp(12));
        tarjeta.setBackground(fondo);
        titular = new TextView(this);
        titular.setTextSize(24);
        titular.setTypeface(Typeface.DEFAULT_BOLD);
        detalle = new TextView(this);
        detalle.setTextSize(16);
        detalle.setTextColor(0xDD000000);
        tarjeta.addView(titular);
        tarjeta.addView(detalle);
        tarjeta.setVisibility(View.GONE);
        l.addView(tarjeta, debajo(m));
        vista = new ImageView(this);
        vista.setAdjustViewBounds(true);
        vista.setMaxHeight(getResources().getDisplayMetrics().heightPixels * 45 / 100);  // los botones quedan a la vista
        l.addView(vista, debajo(m));
        l.addView(boton(odk ? "Repetir la foto" : "Tomar foto", true, v -> capturar()), debajo(m));
        if (odk) {
            usar = boton("Usar de todos modos", false, v -> devolver());
            usar.setEnabled(false);
            l.addView(usar);
        } else {
            l.addView(boton("Elegir de la galería", false, v -> startActivityForResult(
                    new Intent(Intent.ACTION_GET_CONTENT).setType("image/*"), GALERIA)));
        }
        ScrollView s = new ScrollView(this);
        s.addView(l);
        return s;
    }

    private String declarado() {
        if (!odk) return "Mueble declarado";
        String d = getIntent().getStringExtra("mueble");  // fuera de la taxonomía (OTA, ristras…): tal como viene
        return "Declarado: " + (declarado >= 0 ? Clasificador.NOMBRES[declarado]
                : d == null ? "sin dato" : d.split("\\|")[0].trim().toLowerCase(Locale.ROOT));
    }

    private int dp(int n) {
        return Math.round(n * getResources().getDisplayMetrics().density);
    }

    private static LinearLayout.LayoutParams debajo(int margen) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        p.topMargin = margen;
        return p;
    }

    /** El principal, lleno y de color; el secundario, sin borde: en cada pantalla hay una sola acción evidente. */
    private Button boton(String texto, boolean principal, View.OnClickListener accion) {
        Button b = principal ? new Button(this) : new Button(this, null, android.R.attr.borderlessButtonStyle);
        b.setText(texto);
        b.setAllCaps(false);
        b.setTextSize(18);
        b.setMinHeight(dp(56));
        if (principal) {
            b.setBackgroundTintList(ColorStateList.valueOf(AZUL));
            b.setTextColor(Color.WHITE);
        }
        b.setOnClickListener(accion);
        return b;
    }

    private void veredicto(String titulo, String texto, int color) {
        titular.setText(titulo);
        titular.setTextColor(color);
        detalle.setText(texto);
        detalle.setVisibility(texto.isEmpty() ? View.GONE : View.VISIBLE);
        ((GradientDrawable) tarjeta.getBackground()).setColor((color & 0x00FFFFFF) | 0x1F000000);  // el mismo, tenue
        tarjeta.setVisibility(View.VISIBLE);
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
        veredicto("Verificando la foto…", "", GRIS);
        encabezado.setText(declarado());
        if (usar != null) usar.setEnabled(false);
        HILO.execute(() -> {
            try {
                long t0 = SystemClock.elapsedRealtimeNanos();  // la foto ya está en disco: desde aquí, extremo a extremo
                byte[] b;
                try (InputStream in = getContentResolver().openInputStream(f)) {
                    b = Clasificador.leer(in);
                }
                int c = Clasificador.decidir(modelo.clasificar(Clasificador.reducir(b)), convenio);
                long ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000;
                boolean alerta = c >= Clasificador.CAPTURA_INVALIDA || (d >= 0 && c != d);
                if (odk && !alerta) {  // sin alerta no se interrumpe la visita: la foto vuelve sola a ODK
                    runOnUiThread(() -> {
                        Toast.makeText(this, "✓ " + mensaje(c, d)[0], Toast.LENGTH_SHORT).show();
                        devolver();
                    });
                    return;
                }
                Bitmap previa = Clasificador.previa(b, 1024);  // fuera del tiempo medido: es solo para mostrarla
                runOnUiThread(() -> mostrar(previa, c, d, ms, alerta));
            } catch (IOException | RuntimeException e) {
                runOnUiThread(() -> veredicto("No se pudo leer la foto", String.valueOf(e.getMessage()), ROJO));
            }
        });
    }

    private void mostrar(Bitmap previa, int c, int d, long ms, boolean alerta) {
        String[] m = mensaje(c, d);
        veredicto(m[0], m[1], !alerta ? VERDE : c >= Clasificador.CAPTURA_INVALIDA ? ROJO : NARANJA);
        encabezado.setText(declarado() + " · " + ms + " ms");
        vista.setImageBitmap(previa);
        if (alerta) tarjeta.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS);
        if (usar != null) usar.setEnabled(true);
    }

    /** Titular y detalle del veredicto para la clase decidida c y la declarada d (-1: fuera de la taxonomía). */
    static String[] mensaje(int c, int d) {
        if (c == Clasificador.CAPTURA_INVALIDA)
            return new String[]{"Foto inservible", "Está oscura, movida, desenfocada o mal encuadrada. Repítela."};
        if (c == Clasificador.CONTENIDO_AJENO)
            return new String[]{"No se ve un mueble", "La foto no muestra un mueble de exhibición. Repítela."};
        if (d < 0) return new String[]{"Foto válida", "Se ve un mueble tipo " + Clasificador.NOMBRES[c] + "."};
        if (c != d)
            return new String[]{"Parece otro mueble", "Se ve un mueble tipo " + Clasificador.NOMBRES[c]
                    + ", pero declaraste " + Clasificador.NOMBRES[d] + ". Revisa la fila elegida o repite la foto."};
        return new String[]{"Coincide: " + Clasificador.NOMBRES[d], "La foto muestra el mueble declarado."};
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
        if (m != null) HILO.execute(m::close);  // después de una verificación en curso
    }
}
