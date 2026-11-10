package ec.usfq.tesis;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * El modelo de la tesis en el teléfono. La foto pasa por lo mismo que en el entrenamiento: derecha según su EXIF y
 * reducida a 320 px de lado mayor (tesis.export), luego estirada a 224 × 224 (tesis.data). La máscara de la
 * superposición y la normalización van dentro del modelo, que recibe RGB float32 en [0, 255].
 */
final class Clasificador implements AutoCloseable {
    static final String[] CLASES = {"chiclera", "lateral", "check_out", "cabecera", "exhibidor_piso",
            "captura_invalida", "contenido_ajeno"};  // tesis.config.CLASSES
    static final String[] NOMBRES = {"chiclera", "lateral", "check out", "cabecera", "exhibidor de piso",
            "captura inválida", "contenido ajeno"};
    static final int CAPTURA_INVALIDA = 5, CONTENIDO_AJENO = 6;
    private static final int LADO = 224, LADO_EXPORT = 320;

    private final Interpreter red;
    private final ByteBuffer rgba = ByteBuffer.allocateDirect(LADO * LADO * 4);
    private final ByteBuffer entrada = ByteBuffer.allocateDirect(LADO * LADO * 3 * 4).order(ByteOrder.nativeOrder());
    private final float[][] salida = new float[1][CLASES.length];

    Clasificador(Context c, int hilos) throws IOException {
        try (AssetFileDescriptor fd = c.getAssets().openFd("modelo.tflite");
             FileInputStream in = new FileInputStream(fd.getFileDescriptor())) {
            red = new Interpreter(in.getChannel().map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(),
                    fd.getDeclaredLength()), new Interpreter.Options().setNumThreads(hilos).setUseXNNPACK(true));
        }
    }

    /** Bytes de la foto → foto derecha de 320 px de lado mayor, como tesis.export.upright. */
    static Bitmap reducir(byte[] jpeg) throws IOException {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length, o);
        // Como Image.draft de PIL: la mayor escala de decodificación JPEG (1, 2, 4 u 8) que deja 640 px por lado.
        int escala = Math.min(o.outWidth, o.outHeight) / (2 * LADO_EXPORT);
        o.inJustDecodeBounds = false;
        o.inSampleSize = escala >= 8 ? 8 : escala >= 4 ? 4 : escala >= 2 ? 2 : 1;
        o.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap b = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length, o);
        if (b == null) throw new IOException("no es una imagen");
        float f = (float) LADO_EXPORT / Math.max(b.getWidth(), b.getHeight());
        int w = Math.round(b.getWidth() * f), h = Math.round(b.getHeight() * f);
        // PIL reduce con caja y termina con Lanczos; aquí, mitades bilineales (promedio de 2 × 2) y un paso final.
        while (b.getWidth() >= 2 * w && b.getHeight() >= 2 * h) b = escalar(b, b.getWidth() / 2, b.getHeight() / 2);
        b = escalar(b, w, h);
        Matrix m = orientacion(new ExifInterface(new ByteArrayInputStream(jpeg))
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL));
        if (m.isIdentity()) return b;
        Bitmap r = Bitmap.createBitmap(b, 0, 0, w, h, m, true);
        b.recycle();
        return r;
    }

    /** Probabilidades de las 7 clases para la foto que deja reducir(). */
    float[] clasificar(Bitmap derecha) {
        Bitmap b = Bitmap.createScaledBitmap(derecha, LADO, LADO, true);  // tf.image.resize: bilineal, estirada
        rgba.rewind();
        b.copyPixelsToBuffer(rgba);  // bytes crudos R, G, B, A, sin la conversión de color de getPixels
        if (b != derecha) b.recycle();
        entrada.rewind();
        for (int i = 0; i < LADO * LADO * 4; i += 4)
            for (int k = 0; k < 3; k++) entrada.putFloat(rgba.get(i + k) & 0xFF);
        entrada.rewind();
        red.run(entrada, salida);
        return salida[0].clone();
    }

    /** Como tesis.metrics.conditional_pred: la clase más probable entre los muebles del convenio y las dos de
     * rechazo. Convenio vacío (no se conoce): los cinco muebles. */
    static int decidir(float[] p, Set<Integer> convenio) {
        int mejor = -1;
        for (int i = 0; i < p.length; i++)
            if ((i >= CAPTURA_INVALIDA || convenio.isEmpty() || convenio.contains(i)) && (mejor < 0 || p[i] > p[mejor]))
                mejor = i;
        return mejor;
    }

    private static final Map<String, Integer> DECLARADO = new HashMap<>();

    static {  // tesis.odk_candidatas.CLASE_DECLARADA: la nomenclatura de los formularios de ODK, y el nombre de la clase
        poner(0, "CHICLERA", "CHICLERAS");
        poner(1, "LATERAL SIMPLE", "LATEAL SIMPLE", "LATERAL DOBLE", "LATERAL DE CABECERA", "CHECK OUT LATERAL DOBLE",
                "ARETE");
        poner(2, "CHECK OUT", "CHECK OUT SIMPLE", "CAJA");
        poner(3, "CABECERA", "CABECERA FRONTAL", "CABECERA SECA", "CABECERA FASTLINE", "MUEBLE GRIS/CABECERA",
                "BANDEJA DE CABECERA");
        poner(4, "EXHIBIDOR DE PISO", "RUMA", "RUMAS", "RUMA COMPARTIDA", "MEDIA RUMA", "FOUR WAY", "FOUR WAYS",
                "MUEBLE GRIS", "MUBLE GRIS", "MUEBLE GRIS COMPARTIDO", "BIOMBO", "BANDEJA DE BIOMBO");
        for (int i = 0; i < CAPTURA_INVALIDA; i++) poner(i, CLASES[i].toUpperCase(Locale.ROOT));
    }

    private static void poner(int clase, String... nombres) {
        for (String n : nombres) DECLARADO.put(n, clase);
    }

    /** Mueble declarado en ODK ("CHECK OUT | LATERAL SIMPLE | ...": cuenta la primera parte) → clase; -1 si no es de
     * la taxonomía. */
    static int clase(String declarado) {
        if (declarado == null) return -1;
        Integer c = DECLARADO.get(declarado.split("\\|")[0].trim().toUpperCase(Locale.ROOT));
        return c == null ? -1 : c;
    }

    static byte[] leer(InputStream in) throws IOException {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        byte[] b = new byte[1 << 16];
        for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n);
        return o.toByteArray();
    }

    private static Bitmap escalar(Bitmap b, int w, int h) {
        Bitmap r = Bitmap.createScaledBitmap(b, w, h, true);
        if (r != b) b.recycle();
        return r;
    }

    private static Matrix orientacion(int exif) {  // ImageOps.exif_transpose
        Matrix m = new Matrix();
        switch (exif) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: m.setScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_180: m.setRotate(180); break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL: m.setScale(1, -1); break;
            case ExifInterface.ORIENTATION_TRANSPOSE: m.setRotate(90); m.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_90: m.setRotate(90); break;
            case ExifInterface.ORIENTATION_TRANSVERSE: m.setRotate(-90); m.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_270: m.setRotate(-90); break;
        }
        return m;
    }

    @Override
    public void close() {
        red.close();
    }
}
