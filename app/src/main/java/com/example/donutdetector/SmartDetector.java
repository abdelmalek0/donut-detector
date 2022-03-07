package com.example.donutdetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;


import com.example.donutdetector.ml.EmbeddingModel;
import com.example.donutdetector.ml.Head;
import com.example.donutdetector.ml.RpnModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.model.Model.Device;
import org.tensorflow.lite.support.model.Model.Options;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * @author SmartPrints-KSA
 * <p>
 * The goal of this Class is to load deep leaning models then do the inference of a given bitmap.
 */
public class SmartDetector {

    // Singleton
    private static SmartDetector smartDetector;
    // Models
    RpnModel rpn_model = null;
    EmbeddingModel embedding_model = null;
    Head head = null;

    public final static String MODEL_PROCESSING_MOBILENET = "mobilenet";
    public final static String MODEL_PROCESSING_VGG = "vgg";
    public final static String MODEL_PROCESSING_OTHERS = "";

    // Constants
    // final int cropSize = 1000;
    final int inputSizeRpn = 320;
    final int inputSizeEMBD = 100;
    final int pixelSize = 3;

    final int embeddingSize = 256;

    float iou_threshold = 0.2f;
    float confidence_threshold = 0.5f;

    int anchor_dim = 20;

    final int BYTES_SIZE = 4;

    // Variables
    Map<String, float[][]> classes = new HashMap<>();
    String rpnModelType = "";
    String embdModelType = "";
    int[] model_anchors = null;

    // time
    private long lastBBoxesGenerationTime;
    private long lastEmbeddingGenerationTime;
    private long lastHeadComparaisonTime;

    public long getLastBBoxesGenerationTime() { return lastBBoxesGenerationTime; }

    public long getLastEmbeddingGenerationTime() { return lastEmbeddingGenerationTime; }

    public long getLastHeadComparaisonTime() { return lastHeadComparaisonTime; }

    public long getLastTotalTime() { return lastBBoxesGenerationTime + lastEmbeddingGenerationTime + lastHeadComparaisonTime; }

    /**
     * Constructor
     */
    // Static call of the singleton
    public static SmartDetector get(Context context, String rpnModelType, String embdModelType) {
        if (smartDetector == null) smartDetector = new SmartDetector(context, rpnModelType, embdModelType);
        return smartDetector;
    }

    public SmartDetector(Context context, String rpnModelType, String embdModelType) {
        this.rpnModelType = rpnModelType; this.embdModelType = embdModelType;
        Options options; CompatibilityList compatList = new CompatibilityList();
        try {
            if (compatList.isDelegateSupportedOnThisDevice()) {
                // if the device has a supported GPU, add the GPU delegate
                options = new Options.Builder().setDevice(Device.GPU).build();
                Log.d("Device uses", "GPU");
            } else {
                // if the GPU is not supported, run on 4 threads
                options = new Options.Builder().setNumThreads(4).build();
                Log.d("Device uses", "CPU");
            }
            rpn_model = RpnModel.newInstance(context, options);
            embedding_model = EmbeddingModel.newInstance(context, options);
            head = Head.newInstance(context, options);
        } catch (Exception e) {
            Log.e(this.getClass().getName(), "Loading models :" + e);
        }
    }

    public Pair predict(Bitmap bitmap) throws IOException {
        // Intialize the variables
        float[][] bboxes = null; float[][] embeddings = null; Pair<String, Float>[] res = null; int counter = 0; int j = 0;

        // Image preprocessing
        bitmap = imagePreprocessing(bitmap, inputSizeRpn, inputSizeRpn);

        // Bboxes generation
        try {
            bboxes = c_to_xy(get_bboxes(bitmap));
            for (int i = 0; i < bboxes.length; i++) if (bboxes[i] != null) counter++;
        } catch (IOException e) {
            Log.e(this.getClass().getName(), "Generating the bboxes :" + e);
        }

        // Embeddings generation
        try {
            embeddings = get_embeddings(bitmap, bboxes, counter);
        } catch (Exception e) {
            Log.e(this.getClass().getName(), "Generating the embedding :" + e);
        }

        // Similarities generation
        res = new Pair[counter]; j = 0;
        try {
            for (float[] embedding : embeddings) {
                res[j] = inference(embedding);
                j++;
            }
        } catch (IOException e) {
            Log.e(this.getClass().getName(), "Generating the similarities :" + e);
        }
        return new Pair(bboxes, res);
    }


    private float[][] get_bboxes(Bitmap bitmap) throws IOException {

        // IMG_MEAN_SUBSTRACTION = [103.939, 116.779, 123.68]
        long startTime = System.nanoTime();
        float[][] bboxes = null; float[] scores = null; int starting_index = 0; int entered = 0;

        ByteBuffer imgData = convertBitmapToByteBuffer(bitmap, false, inputSizeRpn, rpnModelType);

        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]
                {1, inputSizeRpn, inputSizeRpn, pixelSize}, DataType.FLOAT32);

        inputFeature0.loadBuffer(imgData);

        // Runs model inference and gets result.
        RpnModel.Outputs output = rpn_model.process(inputFeature0);
        float[] obj = output.getOutputFeature0AsTensorBuffer().getFloatArray();
        float[] box = output.getOutputFeature1AsTensorBuffer().getFloatArray();

        for (int i = 0; i < anchor_dim; i++)
            for (int j = 0; j < anchor_dim; j++) {
                int obj_idx = 9 * (i * anchor_dim + j);
                int box_idx = 36 * (i * anchor_dim + j);
                float[] obj_ij = Arrays.copyOfRange(obj, obj_idx, obj_idx + 9);
                float[] box_ij = Arrays.copyOfRange(box, box_idx, box_idx + 36);
                ArrayList<Integer> indices = new ArrayList<>();

                for (int z = 0; z < obj_ij.length; z++) {
                    if (obj_ij[z] > confidence_threshold) indices.add(z);
                }
                if (!indices.isEmpty()) {
                    if (entered == 0) {
                        scores = new float[indices.size()];
                        bboxes = new float[indices.size()][4];
                        entered = 1;
                    } else {
                        scores = Arrays.copyOf(scores, scores.length + indices.size());
                        bboxes = Arrays.copyOf(bboxes, bboxes.length + indices.size());
                        for (int t = starting_index; t < bboxes.length; t++) bboxes[t] = new float[4];
                    }

                    for (int z = 0; z < indices.size(); z++) {
                        scores[starting_index + z] = obj_ij[indices.get(z)];

                        bboxes[starting_index + z][0] = box_ij[indices.get(z) * 4] * model_anchors[box_idx + indices.get(z) * 4 + 2] + model_anchors[box_idx + indices.get(z) * 4];
                        bboxes[starting_index + z][1] = box_ij[indices.get(z) * 4 + 1] * model_anchors[box_idx + indices.get(z) * 4 + 3] + model_anchors[box_idx + indices.get(z) * 4 + 1];
                        bboxes[starting_index + z][2] = (float) Math.exp(box_ij[indices.get(z) * 4 + 2] + Math.log(model_anchors[box_idx + indices.get(z) * 4 + 2]));
                        bboxes[starting_index + z][3] = (float) Math.exp(box_ij[indices.get(z) * 4 + 3] + Math.log(model_anchors[box_idx + indices.get(z) * 4 + 3]));
                    }
                    starting_index += indices.size();
                }
            }
        lastBBoxesGenerationTime = ((System.nanoTime() - startTime) / 1_000_000);
        return tool_nms(bboxes, scores, iou_threshold);
    }

    private float[][] get_embeddings(Bitmap bitmap, float[][] bboxes, int counter) {
        long startTime = System.nanoTime();

        float[][] embeddings = new float[counter][embeddingSize]; int j = 0; Bitmap bt = null;

        for (int i = 0; i < bboxes.length; i++) {
            bt = Bitmap.createBitmap(bitmap, Math.round(bboxes[i][0]), Math.round(bboxes[i][1]), Math.round(bboxes[i][2]), Math.round(bboxes[i][3]));
            bt = Bitmap.createScaledBitmap(bt, inputSizeEMBD, inputSizeEMBD, false);
            ByteBuffer imgData = convertBitmapToByteBuffer(bt, false, inputSizeEMBD, embdModelType);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]
                    {1, inputSizeEMBD, inputSizeEMBD, pixelSize}, DataType.FLOAT32);
            inputFeature0.loadBuffer(imgData);

            // Runs model inference and gets result.
            embeddings[j] = embedding_model.process(inputFeature0).getOutputFeature0AsTensorBuffer().getFloatArray();
            j++;
        }
        lastEmbeddingGenerationTime = ((System.nanoTime() - startTime) / 1_000_000);
        return embeddings;
    }

    private Pair<String, Float> inference(float[] embedding) throws IOException {
        long startTime = System.nanoTime();

        TreeMap<String, Float> predictions = new TreeMap<>();

        TensorBuffer baselineInput = TensorBuffer.createFixedSize(new int[]{1, embeddingSize}, DataType.FLOAT32);
        baselineInput.loadArray(embedding);

        for (Object className : classes.keySet()) {
            float max_score = 0.0F;
            for (float[] embed : classes.get(className)) {
                TensorBuffer comparedInput = TensorBuffer.createFixedSize(new int[]{1, embeddingSize}, DataType.FLOAT32);
                comparedInput.loadArray(embed);

                // Runs model inference and gets result.
                float similarityScore = head.process(baselineInput, comparedInput).getOutputFeature0AsTensorBuffer().getFloatValue(0);

                if (similarityScore > max_score) max_score = similarityScore;
            }
            predictions.put((String) className, max_score);
        }
        SortedMap sortedPredictions = (SortedMap) valueSort(predictions);

        lastHeadComparaisonTime = ((System.nanoTime() - startTime) / 1_000_000);

        Map.Entry<String, Float> entry = (Map.Entry<String, Float>) sortedPredictions.entrySet().iterator().next();
        return new Pair(entry.getKey(), entry.getValue());
    }

    /*           UTILS           */
    public void load_anchors(int[] anchors) { model_anchors = anchors; }

    public void putClass(String className, float[][] embeds) { classes.put(className, embeds); }

    private float[][] c_to_xy(float[][] bboxes) {
        for (int i = 0; i < bboxes.length; i++) {
            if (bboxes[i] != null) {
                bboxes[i][0] = bboxes[i][0] - bboxes[i][2] / 2;
                bboxes[i][1] = bboxes[i][1] - bboxes[i][3] / 2;
            }
        }
        return bboxes;
    }


    private Bitmap imagePreprocessing(Bitmap bitmap, int cropSize, int inputSize) {
        /*if (bitmap.getWidth() > cropSize && bitmap.getHeight() > cropSize) {
            bitmap = Bitmap.createBitmap(bitmap,
                    (bitmap.getWidth() - cropSize) / 2, (bitmap.getHeight() - cropSize) / 2,
                    cropSize, cropSize);
        }*/
        return Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);
    }

    private float[][] tool_nms(float[][] bboxes, float[] scores, float threshold) {
        int y = 0; int counter = 0; double iou_result;

        int[] indexes = ArrayUtils.argsort(scores);
        int[] complete_bool = new int[indexes.length];

        Arrays.fill(complete_bool, 0);
        for (int index = indexes.length - 1; index >= 0; index--) {
            int i = indexes[index];
            if (complete_bool[i] == 0) {
                complete_bool[i] = 2; counter ++;

                BoundingBox bbox = new BoundingBox(Math.round(bboxes[i][0]), Math.round(bboxes[i][1]),
                        Math.round(bboxes[i][2]), Math.round(bboxes[i][3]));
                for (int j = 0; j < complete_bool.length; j++) {
                    if (complete_bool[j] == 1 || complete_bool[j] == 2 ) continue;
                    BoundingBox compare_bbox = new BoundingBox(Math.round(bboxes[j][0]), Math.round(bboxes[j][1]),
                            Math.round(bboxes[j][2]), Math.round(bboxes[j][3]));
                    iou_result = BoundingBox.IOU(bbox, compare_bbox);
                    if (iou_result > threshold && complete_bool[j] == 0) {
                        complete_bool[j] = 1;
                    }
                }
            }
        }

        float[][] res = new float[counter][];
        for (int i = 0; i < indexes.length; i++) {
            if(complete_bool[i] == 2){
                res[y] = bboxes[i];
                y++;
            }
        }
        return res;
    }

    private static <K, V extends Comparable<V>> Map<K, V> valueSort(final Map<K, V> map) {
        Comparator<K> valueComparator = new Comparator<K>() {
            public int compare(K k1, K k2) {
                int comp = map.get(k2).compareTo(map.get(k1));
                if (comp == 0) return 1;
                else return comp;
            }
        };
        Map<K, V> sorted = new TreeMap<K, V>(valueComparator);
        sorted.putAll(map);
        return sorted;
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, Boolean quant, int inputSize, String modelType) {
        ByteBuffer byteBuffer; int pixel = 0; int[] intValues = new int[inputSize * inputSize];

        if (quant) {
            byteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * pixelSize);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(BYTES_SIZE * inputSize * inputSize * pixelSize);
        }
        byteBuffer.order(ByteOrder.nativeOrder());

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                if (quant) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    if (modelType.equals(SmartDetector.MODEL_PROCESSING_MOBILENET)) {
                        byteBuffer.putFloat(((((val >> 16) & 0xFF)) / 127.5f) - 1.0f);
                        byteBuffer.putFloat(((((val >> 8) & 0xFF)) / 127.5f) - 1.0f);
                        byteBuffer.putFloat(((((val) & 0xFF)) / 127.5f) - 1.0f);
                    } else if (modelType.equals(SmartDetector.MODEL_PROCESSING_VGG)) {
                        byteBuffer.putFloat(((((val >> 16) & 0xFF)) / 255.0f));
                        byteBuffer.putFloat(((((val >> 8) & 0xFF)) / 255.0f));
                        byteBuffer.putFloat(((((val) & 0xFF)) / 255.0f));
                    } else {
                        byteBuffer.putFloat((((val >> 16) & 0xFF)) * 1.0f);
                        byteBuffer.putFloat((((val >> 8) & 0xFF)) * 1.0f);
                        byteBuffer.putFloat((((val) & 0xFF)) * 1.0f);
                    }
                }
            }
        }
        return byteBuffer;
    }
}