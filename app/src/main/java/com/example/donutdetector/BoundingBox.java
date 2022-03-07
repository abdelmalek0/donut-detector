package com.example.donutdetector;

import java.util.List;

public class BoundingBox {
    private int x, y, w, h;

    public BoundingBox(int cX, int cY, int W, int H) {
        x = cX;
        y = cY;
        w = W;
        h = H;
    }

    public BoundingBox(List<Integer> pred) {
        this(pred.get(0), pred.get(1), pred.get(2), pred.get(3));
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getW() {
        return w;
    }

    public int getH() {
        return h;
    }

    public double getWScalar() {
        return w;
    }

    public double getHScalar() {
        return h;
    }

    public String toString() {
        return x + "," + y + "," + w + "," + h;
    }

    public static double IOU(BoundingBox a, BoundingBox b) {
        int areaA = a.getH() * a.getW(), areaB = b.getH() * b.getW();
        int wTotal = Math.max(a.getX() + a.getW(), b.getX() + b.getW()) - Math.min(a.getX(), b.getX()),
                hTotal = Math.max(a.getY() + a.getH(), b.getY() + b.getH()) - Math.min(a.getY(), b.getY()),
                wOverlap = wTotal - a.getW() - b.getW(), hOverlap = hTotal - a.getH() - b.getH(),
                areaOverlap = (wOverlap >= 0 || hOverlap >= 0) ? 0 : wOverlap * hOverlap;
        return (double) areaOverlap / (areaA + areaB - areaOverlap);
    }

}