package com.ishland.c2me.opts.dfc.common.ast.noise;

import com.ishland.c2me.opts.dfc.common.ast.AstNode;
import com.ishland.c2me.opts.dfc.common.ast.AstTransformer;
import com.ishland.c2me.opts.dfc.common.ast.EvalType;
import com.ishland.c2me.opts.dfc.common.gen.jvm.BytecodeGen;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.InstructionAdapter;

import java.util.Objects;

public class DFTShiftNode implements AstNode {

    private final DensityFunction.Noise offsetNoise;

    public DFTShiftNode(DensityFunction.Noise offsetNoise) {
        this.offsetNoise = Objects.requireNonNull(offsetNoise);
    }

    @Override
    public double evalSingle(int x, int y, int z, EvalType type) {
        return this.offsetNoise.sample(x * 0.25, y * 0.25, z * 0.25) * 4.0;
    }

    @Override
    public void evalMulti(double[] res, int[] x, int[] y, int[] z, EvalType type) {
        for (int i = 0; i < res.length; i++) {
            res[i] = this.offsetNoise.sample(x[i] * 0.25, y[i] * 0.25, z[i] * 0.25) * 4.0;
        }
    }

    @Override
    public AstNode[] getChildren() {
        return new AstNode[0];
    }

    @Override
    public AstNode transform(AstTransformer transformer) {
        return transformer.transform(this);
    }

    @Override
    public void doBytecodeGenSingle(BytecodeGen.Context context, InstructionAdapter m, BytecodeGen.Context.LocalVarConsumer localVarConsumer) {
        String noiseField = context.newField(DensityFunction.Noise.class, this.offsetNoise);

        m.load(0, InstructionAdapter.OBJECT_TYPE);
        m.getfield(context.className, noiseField, Type.getDescriptor(DensityFunction.Noise.class));

        m.load(1, Type.INT_TYPE);
        m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
        m.dconst(0.25);
        m.mul(Type.DOUBLE_TYPE);

        m.load(2, Type.INT_TYPE);
        m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
        m.dconst(0.25);
        m.mul(Type.DOUBLE_TYPE);

        m.load(3, Type.INT_TYPE);
        m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
        m.dconst(0.25);
        m.mul(Type.DOUBLE_TYPE);

        m.invokevirtual(
                Type.getInternalName(DensityFunction.Noise.class),
                FabricLoader.getInstance().getMappingResolver().mapMethodName("intermediary", "net.minecraft.class_6910$class_7270", "method_42356", "(DDD)D"),
                "(DDD)D",
                false
        );
        m.dconst(4.0);
        m.mul(Type.DOUBLE_TYPE);
        m.areturn(Type.DOUBLE_TYPE);
    }

    @Override
    public void doBytecodeGenMulti(BytecodeGen.Context context, InstructionAdapter m, BytecodeGen.Context.LocalVarConsumer localVarConsumer) {
        String noiseField = context.newField(DensityFunction.Noise.class, this.offsetNoise);

        context.doCountedLoop(m, localVarConsumer, idx -> {
            m.load(1, InstructionAdapter.OBJECT_TYPE);
            m.load(idx, Type.INT_TYPE);

            {
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(context.className, noiseField, Type.getDescriptor(DensityFunction.Noise.class));

                m.load(2, InstructionAdapter.OBJECT_TYPE);
                m.load(idx, Type.INT_TYPE);
                m.aload(Type.INT_TYPE);
                m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
                m.dconst(0.25);
                m.mul(Type.DOUBLE_TYPE);

                m.load(3, InstructionAdapter.OBJECT_TYPE);
                m.load(idx, Type.INT_TYPE);
                m.aload(Type.INT_TYPE);
                m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
                m.dconst(0.25);
                m.mul(Type.DOUBLE_TYPE);

                m.load(4, InstructionAdapter.OBJECT_TYPE);
                m.load(idx, Type.INT_TYPE);
                m.aload(Type.INT_TYPE);
                m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
                m.dconst(0.25);
                m.mul(Type.DOUBLE_TYPE);

                m.invokevirtual(
                        Type.getInternalName(DensityFunction.Noise.class),
                        FabricLoader.getInstance().getMappingResolver().mapMethodName("intermediary", "net.minecraft.class_6910$class_7270", "method_42356", "(DDD)D"),
                        "(DDD)D",
                        false
                );
                m.dconst(4.0);
                m.mul(Type.DOUBLE_TYPE);
            }

            m.astore(Type.DOUBLE_TYPE);
        });

        m.areturn(Type.VOID_TYPE);
    }

    @Override
    public String doCLGen(OpenCLGen.Context context) {
        if (this.offsetNoise.noise() == null) {
            return "return 0.0d;\n";
        } else {
            int offset = context.allocGlobalConstDataObject(this.offsetNoise.noise());
            return "global const double_octave_sampler_data_t * restrict data = ptr_shift_global(ctx.const_data, " + offset + ");\n" +
                    "return math_noise_perlin_double_octave_sample_global_noinline(data, " +
                    "(double) ctx.x * 0.25, (double) ctx.y * 0.25, (double) ctx.z * 0.25) * 4.0;\n";
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DFTShiftNode that = (DFTShiftNode) o;
        return Objects.equals(offsetNoise, that.offsetNoise);
    }

    @Override
    public int hashCode() {
        int result = 1;

        Object o = this.getClass();
        result = 31 * result + o.hashCode();
        result = 31 * result + offsetNoise.hashCode();

        return result;
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
