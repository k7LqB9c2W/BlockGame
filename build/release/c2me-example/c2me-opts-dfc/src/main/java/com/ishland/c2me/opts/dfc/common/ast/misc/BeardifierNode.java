package com.ishland.c2me.opts.dfc.common.ast.misc;

import com.ishland.c2me.base.mixin.access.IStructureWeightSampler;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import com.ishland.flowsched.util.Assertions;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;

public class BeardifierNode extends DelegateNode {

    public BeardifierNode(DensityFunction densityFunction) {
        super(densityFunction);
        Assertions.assertTrue(densityFunction == DensityFunctionTypes.Beardifier.INSTANCE);
    }

    @Override
    public String doCLGen(OpenCLGen.Context context) {
        int offset = context.getGlobalDynamicDataOffset(DensityFunctionTypes.Beardifier.INSTANCE);
        int tableOffset = context.allocGlobalConstDataObject(IStructureWeightSampler.getSTRUCTURE_WEIGHT_TABLE());
        return "if (!ctx.rw_data) return 0.0;\n" +
                "global const sws_index_t * restrict data = df_data_offset_global(ctx.rw_data, " + offset + ");\n" +
                "global const float * restrict structureWeightSamplerTable = ptr_shift_global(ctx.const_data, " + tableOffset + ");\n" +
                "if (!data) return 0.0;\n" +
                "return df_structureWeightSampler_sample(structureWeightSamplerTable, data, ctx.x, ctx.y, ctx.z);\n";
    }

}
