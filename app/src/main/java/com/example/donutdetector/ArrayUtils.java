package com.example.donutdetector;

import java.util.Arrays;
import java.util.Comparator;

// This class is implemented to add some functionalities to the array class
// The main function here is the argsort function
public final class ArrayUtils {

    private ArrayUtils() {
    }

    public static int[] argsort(final float[] a) {
        return argsort(a, true);
    }

    public static int[] argsort(final float[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Float.compare(a[i1], a[i2]);
            }
        });
        return asArray(indexes);
    }

    public static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }
}