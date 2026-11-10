package ec.usfq.tesis;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.TextView;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;

/**
 * Una ráfaga del protocolo de carga sostenida (secciones 5.6 y 5.7) de extremo a extremo, para python device.py --app:
 * carga el modelo y clasifica n fotos seguidas de files/fotos, desde la número `desde`, cada una desde sus bytes en
 * disco hasta las probabilidades. Escribe files/rafaga.csv (una fila por foto y una última con la carga del modelo y
 * la memoria pico del proceso, VmHWM) y se cierra.
 */
public class Rafaga extends Activity {
    @Override
    protected void onCreate(Bundle estado) {
        super.onCreate(estado);
        TextView t = new TextView(this);
        t.setText("Midiendo…");
        setContentView(t);
        new Thread(this::medir).start();
    }

    private void medir() {
        File dir = getExternalFilesDir(null);
        Intent i = getIntent();
        int desde = i.getIntExtra("desde", 0), n = i.getIntExtra("n", 30);
        StringBuilder s = new StringBuilder("archivo,leer_ms,reducir_ms,red_ms,total_ms,clase");
        for (String c : Clasificador.CLASES) s.append(",p_").append(c);
        s.append('\n');
        try {
            File[] fotos = new File(dir, "fotos").listFiles();
            if (fotos == null) fotos = new File[0];  // device.py la lanza una vez con n = 0 para crear la carpeta
            Arrays.sort(fotos);
            long t0 = System.nanoTime();
            Clasificador m = new Clasificador(this, i.getIntExtra("hilos", 4));
            double carga = ms(t0, System.nanoTime());
            for (int k = 0; k < n; k++) {
                File f = fotos[(desde + k) % fotos.length];
                long a = System.nanoTime();
                byte[] b = Files.readAllBytes(f.toPath());
                long c = System.nanoTime();
                Bitmap d = Clasificador.reducir(b);
                long e = System.nanoTime();
                float[] p = m.clasificar(d);
                long g = System.nanoTime();
                d.recycle();
                s.append(String.format(Locale.ROOT, "%s,%.3f,%.3f,%.3f,%.3f,%s", f.getName(), ms(a, c), ms(c, e),
                        ms(e, g), ms(a, g), Clasificador.CLASES[Clasificador.decidir(p, Collections.emptySet())]));
                for (float x : p) s.append(String.format(Locale.ROOT, ",%.5f", x));
                s.append('\n');
            }
            m.close();
            s.append(String.format(Locale.ROOT, "#carga_ms=%.3f,pico_kb=%d%n", carga, picoKb()));
        } catch (Exception e) {
            s.append("#error=").append(e).append('\n');
        }
        File tmp = new File(dir, "rafaga.tmp");
        try (FileWriter w = new FileWriter(tmp)) {
            w.write(s.toString());
        } catch (IOException ignored) {
        }
        tmp.renameTo(new File(dir, "rafaga.csv"));  // de una vez: device.py espera el archivo completo
        finish();
    }

    private static double ms(long a, long b) {
        return (b - a) / 1e6;
    }

    private static long picoKb() throws IOException {
        for (String l : Files.readAllLines(Paths.get("/proc/self/status")))
            if (l.startsWith("VmHWM:")) return Long.parseLong(l.replaceAll("\\D+", ""));
        return -1;
    }
}
