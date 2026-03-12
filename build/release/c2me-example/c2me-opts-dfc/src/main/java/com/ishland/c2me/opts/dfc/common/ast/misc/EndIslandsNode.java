package com.ishland.c2me.opts.dfc.common.ast.misc;

import com.ishland.c2me.base.mixin.access.IDensityFunctionTypesEndIslands;
import com.ishland.c2me.base.mixin.access.ISimplexNoiseSampler;
import com.ishland.c2me.opts.dfc.common.ast.AstNode;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;

import java.util.Objects;

public class EndIslandsNode extends DelegateNode {

    private final DensityFunctionTypes.EndIslands endIslands;

    public EndIslandsNode(DensityFunctionTypes.EndIslands endIslands) {
        super(endIslands);
        this.endIslands = endIslands;
    }

    @Override
    public String doCLGen(OpenCLGen.Context context) {
        int[] permutation = ((ISimplexNoiseSampler) ((IDensityFunctionTypesEndIslands) (Object) this.endIslands).getSampler()).getPermutation();
        int offset = context.allocGlobalConstDataObject(permutation);
        return "global const uint32_t * const permutation = ptr_shift_global(ctx.const_data, " + offset + ");\n" +
                "return ((double) math_end_islands_sample_global(permutation, ctx.x / 8, ctx.z / 8) - 8.0) / 128.0;\n";
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        EndIslandsNode that = (EndIslandsNode) object;
        return Objects.equals(endIslands, that.endIslands);
    }

    @Override
    public boolean relaxedEquals(AstNode o) {
        return this.equals(o);
    }

    @Override
    public int relaxedHashCode() {
        return this.hashCode();
    }
}
