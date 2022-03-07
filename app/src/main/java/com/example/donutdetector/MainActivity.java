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
    final float inputSizeRpn = 320.0f;

    // Embeddings variables
    final int embeddingSize = 256;
    int classesNumber = 9;
    int EmbeddingsCount = 162;

    // Anchors variables
    int anchorDimension = 20;
    int numberAnchors = 9;
    int numberCoordinates = 4;


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
        smartDetector.load_anchors(load_anchors("anchors.csv"));
        // load the classes and embeddings
        load_data_from_csv("embeds.csv", "classes.csv");

        // Selection an image either from the gallery or the camera
        select.setOnClickListener(view -> {
            if (shouldRequestPermissions()) requestPermissionsWeNeed();
            Intent chooseImageIntent = ImagePicker.getPickImageIntent(this);
            startActivityForResult(chooseImageIntent, PICK_IMAGE_ID);
        });

        predict.setOnClickListener(v ->
                {

                    long startTime = System.nanoTime();
                    // Intialize the variables
                    Pair<String, Float>[] results = null;
                    float[][] bboxes = null;
                    Pair output;

                    // Make the prediction
                    try {
                        output = smartDetector.predict(main_bitmap);
                        bboxes = (float[][]) output.first;
                        results = (Pair<String, Float>[]) output.second;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Drawing the bounding boxes
                    Bitmap dst = main_bitmap.copy(main_bitmap.getConfig(), true);
                    float factor_h = dst.getHeight() / inputSizeRpn;
                    float factor_w = dst.getWidth() / inputSizeRpn;

                    Canvas canvas = new Canvas(dst);
                    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

                    paint.setColor(Color.BLACK);
                    paint.setStyle(Paint.Style.STROKE);

                    for (int index = 0; index < bboxes.length; index++) {
                        canvas.drawRect(bboxes[index][0] * factor_w, bboxes[index][1] * factor_h, (bboxes[index][0] + bboxes[index][2]) * factor_w, (bboxes[index][1] + bboxes[index][3]) * factor_h, paint);
                        canvas.drawText((String) results[index].first, (bboxes[index][0] * factor_w) - 5, (bboxes[index][1] * factor_h) - 5, paint);
                    }
                    imageView.setImageBitmap(dst);

                    long stopTime = System.nanoTime();
                    Log.d("FULL_TIME", String.valueOf((stopTime - startTime) / 1_000_000));
                }

        );
    }

    /* ***** Reading files from the assets folder ***** */

    private void load_data_from_csv(String embeddingFile, String classesFile) {
        // initialize some variables
        int elmentsPerClass = 0;
        String line = null;
        int i = 0;


        float[][] embeds = new float[EmbeddingsCount][embeddingSize];
        String[] classes = new String[classesNumber];
        int[] indices = new int[classesNumber];

        // Reading the embeddings
        try {

            InputStreamReader isEmbds = new InputStreamReader(getAssets()
                    .open(embeddingFile));

            BufferedReader readerEmbds = new BufferedReader(isEmbds);
            while ((line = readerEmbds.readLine()) != null) {
                embeds[i] = stringToArrayFloat(line);
                i++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Reading the classes
        try {
            line = null;
            i = 0;
            InputStreamReader isClasses = new InputStreamReader(getAssets()
                    .open(classesFile));
            BufferedReader readerClasses = new BufferedReader(isClasses);
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

        // Load the data to the Detector
        for (int a = 0; a < classes.length; a++) {
            float[][] embeddingsClass = new float[indices[a] - elmentsPerClass][embeddingSize];
            for (int b = 0; b < indices[a] - elmentsPerClass; b++) {
                embeddingsClass[b] = embeds[elmentsPerClass + b];
            }
            elmentsPerClass = indices[a];
            smartDetector.putClass(classes[a], embeddingsClass);
        }
    }

    private int[] load_anchors(String anchorsFile) {
        String line;
        int[] anchors = new int[anchorDimension * anchorDimension * numberAnchors * numberCoordinates];

        // Reading the anchors
        try {
            InputStreamReader isEmbds = new InputStreamReader(getAssets()
                    .open(anchorsFile));

            BufferedReader readerEmbds = new BufferedReader(isEmbds);

            line = readerEmbds.readLine();
            anchors = stringToArray(line);

        } catch (Exception ignored) {
        }
        return anchors;
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

    private float[] stringToArrayFloat(String line) {
        String[] str = line.split(",");
        int size = str.length;
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = Float.parseFloat(str[i]);
        }
        return arr;
    }

    /* ***** Permisition functions ***** */

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

    /* ***** Override some important functions ***** */
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