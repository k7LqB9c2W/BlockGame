package com.ishland.c2me.opts.dfc.common.ast.misc;

import com.ishland.c2me.opts.dfc.common.ast.AstNode;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import net.minecraft.util.math.noise.InterpolatedNoiseSampler;

import java.util.Objects;

public class InterpolatedNoiseSamplerNode extends DelegateNode {

    private final InterpolatedNoiseSampler sampler;

    public InterpolatedNoiseSamplerNode(InterpolatedNoiseSampler sampler) {
        super(sampler);
        this.sampler = sampler;
    }

    @Override
    public String doCLGen(OpenCLGen.Context context) {
        int offset = context.allocGlobalConstDataObject(this.sampler);
        return "global const interpolated_noise_sampler_t * restrict data = ptr_shift_global(ctx.const_data, " + offset + ");\n" +
                "return math_noise_perlin_interpolated_sample_global_noinline(data, ctx.x, ctx.y, ctx.z);\n";
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        InterpolatedNoiseSamplerNode that = (InterpolatedNoiseSamplerNode) object;
        return Objects.equals(sampler, that.sampler);
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
