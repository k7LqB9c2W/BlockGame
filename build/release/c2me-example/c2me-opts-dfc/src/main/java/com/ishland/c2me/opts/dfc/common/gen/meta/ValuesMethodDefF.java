package com.ishland.c2me.opts.dfc.common.gen.meta;

public record ValuesMethodDefF(boolean isConst, String generatedMethod, float constValue) {

    public ValuesMethodDefF(String generatedMethod) {
        this(false, generatedMethod, Float.NaN);
    }

    public ValuesMethodDefF(float constValue) {
        this(true, null, constValue);
    }

}
