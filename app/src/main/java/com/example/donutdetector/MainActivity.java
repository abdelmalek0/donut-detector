package com.example.donutdetector;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;

import android.util.Log;
import android.util.Pair;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.SortedMap;


public class MainActivity extends AppCompatActivity {
    // Image Selection Constants
    private static final int PERMISSION_REQUEST_CODE = 0;
    private static final String[] permissionsWeNeed = {Manifest.permission.CAMERA};
    private ArrayList<String> permissionsDenied = null;
    private static final int PICK_IMAGE_ID = 234; // the number doesn't matter

    // UI variables
    Button select = null;
    Button predict = null;
    ImageView imageView = null;
    Bitmap main_bitmap;

    // Prediction variables
    SmartDetector smartDetector;
    private SortedMap<String, Float> predictions = null;
    final int embeddingSize = 256;

    private boolean shouldRequestPermissions() {
        if (this.permissionsDenied == null) {
            this.checkPermissions();
        }
        return this.permissionsDenied.size() > 0;
    }

    private void requestPermissionsWeNeed() {
        String[] permissions = this.permissionsDenied.toArray(new String[this.permissionsDenied.size()]);
        ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
    }

    private void checkPermissions() {
        if (this.permissionsDenied == null) {
            this.permissionsDenied = new ArrayList<>();
        }
        for (String permission : this.permissionsWeNeed) {
            if (ActivityCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_DENIED) {
                this.permissionsDenied.add(permission);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            for (int idx = 0; idx < permissions.length; idx++) {
                if (grantResults[idx] == PackageManager.PERMISSION_GRANTED) {
                    this.permissionsDenied.remove(permissions[idx]);
                }
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Define UI elements
        select = findViewById(R.id.select);
        predict = findViewById(R.id.predict);
        imageView = findViewById(R.id.imageView);

        // Get the Prediction class
        smartDetector = SmartDetector.get(getApplicationContext(), SmartDetector.MODEL_PROCESSING_MOBILENET, SmartDetector.MODEL_PROCESSING_OTHERS);

        // load the anchors
        smartDetector.load_anchors(load_anchors());
        // load the classes and embeddings
        load_data_from_csv("embeds.csv", "classes.csv");

        // Selection an image either from the gallery or the camera
        select.setOnClickListener(view -> {
            if (shouldRequestPermissions()) {
                requestPermissionsWeNeed();
            }
            Intent chooseImageIntent = ImagePicker.getPickImageIntent(this);
            startActivityForResult(chooseImageIntent, PICK_IMAGE_ID);
        });


        predict.setOnClickListener(v ->
                {
                    // Make the prediction
                    long startTime = System.nanoTime();
                    float[][] bboxes = null;
                    Pair output;
                    Pair<String, Float>[] results = null;
                    try {
                        output = smartDetector.predict(main_bitmap);
                        bboxes = (float[][]) output.first;
                        results = (Pair<String, Float>[]) output.second;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    long stopTime = System.nanoTime();
                    BitmapDrawable drawable = (BitmapDrawable) imageView.getDrawable();

                    Bitmap bitmap = drawable.getBitmap();
                    Bitmap dst = bitmap.copy(bitmap.getConfig(), true);
                    float factor_h = dst.getHeight() / 320.0f;
                    float factor_w = dst.getWidth() / 320.0f;

                    Canvas canvas = new Canvas(dst);
                    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
                    paint.setColor(Color.BLACK);
                    paint.setStyle(Paint.Style.STROKE);

                    for (int index = 0; index < bboxes.length; index++) {
                        canvas.drawRect(bboxes[index][0] * factor_w, bboxes[index][1] * factor_h, (bboxes[index][0] + bboxes[index][2]) * factor_w, (bboxes[index][1] + bboxes[index][3]) * factor_h, paint);
                        canvas.drawText((String) results[index].first, (bboxes[index][0] * factor_w) - 5, (bboxes[index][1] * factor_h) - 5, paint);
                    }

                    imageView.setImageBitmap(dst);
                    Log.d("FULL_TIME", String.valueOf((stopTime - startTime) / 1_000_000));
                }

        );
    }

    private void load_data_from_csv(String embeddingFile, String classesFile) {
        // initialize some variables
        int classesNumber = 9;
        int EmbeddingsCount = 162;
        float[][] embeds = new float[EmbeddingsCount][embeddingSize];
        String[] classes = new String[classesNumber];
        int[] indices = new int[classesNumber];

        // Reading the embedding file
        try {

            InputStreamReader isEmbds = new InputStreamReader(getAssets()
                    .open(embeddingFile));

            BufferedReader readerEmbds = new BufferedReader(isEmbds);

            // readerEmbds.readLine();
            String line;

            int i = 0;
            while ((line = readerEmbds.readLine()) != null) {
                embeds[i] = stringToArray_float(line);
                i++;
            }
        } catch (Exception ignored) {
        }


        // Reading the classes
        try {
            InputStreamReader isClasses = new InputStreamReader(getAssets()
                    .open(classesFile));
            BufferedReader readerClasses = new BufferedReader(isClasses);
            String line = null;
            int i = 0;
            while (true) {
                try {
                    if ((line = readerClasses.readLine()) == null) break;
                } catch (IOException e) {
                    e.printStackTrace();
                }
                assert line != null;
                String[] str_line = line.split(",");
                indices[i] = Integer.parseInt(str_line[0]);
                classes[i] = str_line[1];
                i++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load the data to the Predictor
        int elmentsPerClass = 0;
        for (int a = 0; a < classes.length; a++) {
            Log.d("number of embeds", String.valueOf(indices[a] - elmentsPerClass));
            float[][] embeddingsClass = new float[indices[a] - elmentsPerClass][256];
            for (int b = 0; b < indices[a] - elmentsPerClass; b++) {
                embeddingsClass[b] = embeds[elmentsPerClass + b];
            }
            elmentsPerClass = indices[a];
            smartDetector.putClass(classes[a], embeddingsClass);
        }
    }

    int[] load_anchors() {

        int[] anchors = new int[20 * 20 * 36];

        // Reading the embedding file
        try {

            InputStreamReader isEmbds = new InputStreamReader(getAssets()
                    .open("anchors.csv"));

            BufferedReader readerEmbds = new BufferedReader(isEmbds);

            // readerEmbds.readLine();
            String line;
            int i = 0;
            line = readerEmbds.readLine();
            anchors = stringToArray(line);

        } catch (Exception ignored) {
        } finally {
            return anchors;
        }

    }


    private int[] stringToArray(String line) {
        String[] str = line.split(",");
        int size = str.length;
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = Integer.parseInt(str[i]);
        }
        return arr;
    }

    private float[] stringToArray_float(String line) {
        String[] str = line.split(",");
        int size = str.length;
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = Float.parseFloat(str[i]);
        }
        return arr;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case PICK_IMAGE_ID:
                main_bitmap = ImagePicker.getImageFromResult(this, resultCode, data);
                imageView.setImageBitmap(main_bitmap);
                break;
            default:
                super.onActivityResult(requestCode, resultCode, data);
                break;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        smartDetector = null;
    }
}