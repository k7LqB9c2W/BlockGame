package com.ishland.c2me.opts.dfc.common.ast.misc;

import com.ishland.c2me.opts.dfc.common.ast.AstNode;
import com.ishland.c2me.opts.dfc.common.ast.AstTransformer;
import com.ishland.c2me.opts.dfc.common.ast.EvalType;
import com.ishland.c2me.opts.dfc.common.gen.jvm.BytecodeGen;
import com.ishland.c2me.opts.dfc.common.gen.meta.ValuesMethodDefD;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.util.math.MathHelper;
import org.objectweb.asm.Label;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.InstructionAdapter;

import java.util.Objects;

public class FindTopSurfaceNode implements AstNode {

    private final AstNode density;
    private final AstNode upperBound;
    private final AstNode lowerBound;
    private final int cellHeight;

    public FindTopSurfaceNode(AstNode density, AstNode upperBound, AstNode lowerBound, int cellHeight) {
        this.density = Objects.requireNonNull(density);
        this.upperBound = Objects.requireNonNull(upperBound);
        this.lowerBound = Objects.requireNonNull(lowerBound);
        this.cellHeight = cellHeight;
    }

    @Override
    public double evalSingle(int x, int y, int z, EvalType type) {
        int topCellBlockY = MathHelper.floor(this.upperBound.evalSingle(x, y, z, type) / this.cellHeight) * this.cellHeight;
        int lowerBoundEval = (int) this.lowerBound.evalSingle(x, y, z, type);
        for (int y1 = topCellBlockY; y1 > lowerBoundEval; y1 -= this.cellHeight) {
            if (this.density.evalSingle(x, y1, z, EvalType.NORMAL) > 0.0) {
                return y1;
            }
        }
        return lowerBoundEval;
    }

    @Override
    public void evalMulti(double[] res, int[] x, int[] y, int[] z, EvalType type) {
        this.upperBound.evalMulti(res, x, y, z, type);
        int[] topCellBlockY = new int[res.length];
        for (int i = 0; i < res.length; i++) {
            topCellBlockY[i] = MathHelper.floor(res[i] / this.cellHeight) * this.cellHeight;
        }
        this.lowerBound.evalMulti(res, x, y, z, type);
        int[] lowerBoundEval = new int[res.length];
        for (int i = 0; i < res.length; i++) {
            lowerBoundEval[i] = (int) res[i];
        }
        for (int i = 0; i < res.length; i ++) {
            res[i] = lowerBoundEval[i]; // default to lower bound
            for (int y1 = topCellBlockY[i]; y1 > lowerBoundEval[i]; y1 -= this.cellHeight) {
                if (this.density.evalSingle(x[i], y1, z[i], type) > 0.0) {
                    res[i] = y1;
                    break;
                }
            }
        }
    }

    @Override
    public AstNode[] getChildren() {
        return new AstNode[] {this.density, this.upperBound, this.lowerBound};
    }

    @Override
    public AstNode transform(AstTransformer transformer) {
        AstNode density = this.density.transform(transformer);
        AstNode upperBound = this.upperBound.transform(transformer);
        AstNode lowerBound = this.lowerBound.transform(transformer);
        if (density == this.density && upperBound == this.upperBound && lowerBound == this.lowerBound) {
            return transformer.transform(this);
        } else {
            return transformer.transform(new FindTopSurfaceNode(density, upperBound, lowerBound, this.cellHeight));
        }
    }

    @Override
    public void doBytecodeGenSingle(BytecodeGen.Context ctx, InstructionAdapter m, BytecodeGen.Context.LocalVarConsumer localVarConsumer) {
        ValuesMethodDefD densityMethod = ctx.newSingleMethod(this.density);
        ValuesMethodDefD upperBoundMethod = ctx.newSingleMethod(this.upperBound);
        ValuesMethodDefD lowerBoundMethod = ctx.newSingleMethod(this.lowerBound);

        int topCellBlockY = localVarConsumer.createLocalVariable("topCellBlockY", Type.INT_TYPE.getDescriptor());
        int lowerBoundEval = localVarConsumer.createLocalVariable("lowerBoundEval", Type.INT_TYPE.getDescriptor());
        ctx.callDelegateSingle(m, upperBoundMethod);
        m.dconst(this.cellHeight);
        m.div(Type.DOUBLE_TYPE);
        m.invokestatic(
                Type.getInternalName(MathHelper.class),
                FabricLoader.getInstance().getMappingResolver().mapMethodName("intermediary", "net.minecraft.class_3532", "method_15357", "(D)I"),
                "(D)I",
                false
        );
        m.iconst(this.cellHeight);
        m.mul(Type.INT_TYPE);
        m.store(topCellBlockY, Type.INT_TYPE);

        ctx.callDelegateSingle(m, lowerBoundMethod);
        m.cast(Type.DOUBLE_TYPE, Type.INT_TYPE);
        m.store(lowerBoundEval, Type.INT_TYPE);

        Label loopStart = new Label();
        Label loopEnd = new Label();

        int y1 = localVarConsumer.createLocalVariable("y1", Type.INT_TYPE.getDescriptor());
        m.load(topCellBlockY, Type.INT_TYPE);
        m.store(y1, Type.INT_TYPE);

        m.visitLabel(loopStart);

        m.load(y1, Type.INT_TYPE);
        m.load(lowerBoundEval, Type.INT_TYPE);
        m.ificmple(loopEnd);
        if (densityMethod.isConst()) {
            m.dconst(densityMethod.constValue());
        } else {
            m.load(0, InstructionAdapter.OBJECT_TYPE);
            m.load(1, Type.INT_TYPE);
            m.load(y1, Type.INT_TYPE);
            m.load(3, Type.INT_TYPE);
            m.getstatic(
                    Type.getInternalName(EvalType.class),
                    "NORMAL",
                    Type.getDescriptor(EvalType.class)
            );
            m.invokevirtual(ctx.className, densityMethod.generatedMethod(), BytecodeGen.Context.SINGLE_DESC, false);
        }
        m.dconst(0.0);
        m.cmpl(Type.DOUBLE_TYPE);

        Label notSatisfied = new Label();
        m.ifle(notSatisfied);
        m.load(y1, Type.INT_TYPE);
        m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
        m.areturn(Type.DOUBLE_TYPE);
        m.visitLabel(notSatisfied);

        m.load(y1, Type.INT_TYPE);
        m.iconst(this.cellHeight);
        m.sub(Type.INT_TYPE);
        m.store(y1, Type.INT_TYPE);
        m.goTo(loopStart);

        m.visitLabel(loopEnd);

        m.load(lowerBoundEval, Type.INT_TYPE);
        m.cast(Type.INT_TYPE, Type.DOUBLE_TYPE);
        m.areturn(Type.DOUBLE_TYPE);
    }

    @Override
    public void doBytecodeGenMulti(BytecodeGen.Context ctx, InstructionAdapter m, BytecodeGen.Context.LocalVarConsumer localVarConsumer) {
        ctx.delegateAllToSingle(m, localVarConsumer, this);
        m.areturn(Type.VOID_TYPE);
    }

    @Override
    public String doCLGen(OpenCLGen.Context ctx) {
        ValuesMethodDefD densityMethod = ctx.newMethod(this.density);
        ValuesMethodDefD upperBoundMethod = ctx.newMethod(this.upperBound);
        ValuesMethodDefD lowerBoundMethod = ctx.newMethod(this.lowerBound);

        StringBuilder b = new StringBuilder();
        b.append("int32_t topCellBlockY = ((int32_t) floor(").append(ctx.callDelegate(upperBoundMethod)).append(" / ").append(this.cellHeight).append(")) * ").append(this.cellHeight).append(";\n");
        b.append("int32_t lowerBoundEval = (int32_t) ").append(ctx.callDelegate(lowerBoundMethod)).append(";\n");
        b.append("for (int32_t y1 = topCellBlockY; y1 > lowerBoundEval; y1 -= ").append(this.cellHeight).append(") {\n");

        b.append("    if (");
        if (densityMethod.isConst()) {
            b.append(OpenCLGen.literal(densityMethod.constValue()));
        } else {
            b.append(densityMethod.generatedMethod()).append("(make_sample_int32_ctx(ctx.const_data, ctx.rw_data, ctx.x, y1, ctx.z, 0))");
        }
        b.append(" > 0.0) {\n");

        b.append("        return (double) y1;\n");
        b.append("    }\n");
        b.append("}\n");
        b.append("return (double) lowerBoundEval;\n");

        return b.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        FindTopSurfaceNode that = (FindTopSurfaceNode) o;
        return cellHeight == that.cellHeight && Objects.equals(density, that.density) && Objects.equals(upperBound, that.upperBound) && Objects.equals(lowerBound, that.lowerBound);
    }

    @Override
    public int hashCode() {
        int result = 1;

        result = 31 * result + this.getClass().hashCode();
        result = 31 * result + this.density.hashCode();
        result = 31 * result + this.upperBound.hashCode();
        result = 31 * result + this.lowerBound.hashCode();
        result = 31 * result + Integer.hashCode(cellHeight);

        return result;
    }

    @Override
    public boolean relaxedEquals(AstNode o) {
        if (o == null || getClass() != o.getClass()) return false;
        FindTopSurfaceNode that = (FindTopSurfaceNode) o;
        return cellHeight == that.cellHeight && this.density.relaxedEquals(that.density) && this.upperBound.relaxedEquals(that.upperBound) && this.lowerBound.relaxedEquals(that.lowerBound);
    }

    @Override
    public int relaxedHashCode() {
        int result = 1;

        result = 31 * result + this.getClass().hashCode();
        result = 31 * result + this.density.relaxedHashCode();
        result = 31 * result + this.upperBound.relaxedHashCode();
        result = 31 * result + this.lowerBound.relaxedHashCode();
        result = 31 * result + Integer.hashCode(cellHeight);

        return result;
    }
}
