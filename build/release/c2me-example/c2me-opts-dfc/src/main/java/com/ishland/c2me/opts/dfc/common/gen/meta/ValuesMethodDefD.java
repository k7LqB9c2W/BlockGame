package com.ishland.c2me.opts.dfc.common.gen.meta;

public record ValuesMethodDefD(boolean isConst, String generatedMethod, double constValue) {

    public ValuesMethodDefD(String generatedMethod) {
        this(false, generatedMethod, Double.NaN);
    }

    public ValuesMethodDefD(double constValue) {
        this(true, null, constValue);
    }

}
