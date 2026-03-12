package com.ishland.c2me.base.common.util;

public class MemoryUtil {

    public static int[] byte2int(byte[] data) {
        if (data == null) return null;
        int[] ints = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            ints[i] = data[i] & 0xff;
        }
        return ints;
    }

    public static int roundUp(int num, int base) {
        int temp = num % base;
        if (temp < 0)
            temp = base + temp;
        if (temp == 0)
            return num;
        return num + base - temp;
    }

    public static long roundUp(long num, long base) {
        long temp = num % base;
        if (temp < 0)
            temp = base + temp;
        if (temp == 0)
            return num;
        return num + base - temp;
    }

}
