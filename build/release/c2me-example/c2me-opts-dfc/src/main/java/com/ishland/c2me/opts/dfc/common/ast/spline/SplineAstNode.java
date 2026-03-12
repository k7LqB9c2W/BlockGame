package com.ishland.c2me.opts.dfc.common.ast.spline;

import com.ishland.c2me.opts.dfc.common.ast.AstNode;
import com.ishland.c2me.opts.dfc.common.ast.AstTransformer;
import com.ishland.c2me.opts.dfc.common.ast.EvalType;
import com.ishland.c2me.opts.dfc.common.ast.McToAst;
import com.ishland.c2me.opts.dfc.common.ast.misc.ConstantNode;
import com.ishland.c2me.opts.dfc.common.gen.jvm.BytecodeGen;
import com.ishland.c2me.opts.dfc.common.gen.meta.ValuesMethodDefD;
import com.ishland.c2me.opts.dfc.common.gen.meta.ValuesMethodDefF;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import com.ishland.c2me.opts.dfc.common.vif.NoisePosVanillaInterface;
import com.ishland.flowsched.util.Assertions;
import it.unimi.dsi.fastutil.Pair;
import it.unimi.dsi.fastutil.ints.IntObjectPair;
import it.unimi.dsi.fastutil.objects.Reference2ReferenceMap;
import it.unimi.dsi.fastutil.objects.Reference2ReferenceOpenHashMap;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Spline;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import org.objectweb.asm.Label;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.AnalyzerAdapter;
import org.objectweb.asm.commons.InstructionAdapter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen.literal;

public class SplineAstNode implements AstNode {

    public static final String SPLINE_METHOD_DESC = Type.getMethodDescriptor(Type.getType(float.class), Type.getType(int.class), Type.getType(int.class), Type.getType(int.class), Type.getType(EvalType.class));
    public static final String SPLINE_METHOD_DESC_CACHE1 = Type.getMethodDescriptor(Type.getType(float.class), Type.getType(int.class), Type.getType(int.class), Type.getType(int.class), Type.getType(EvalType.class), Type.getType(float.class));
    public static final String SPLINE_OCL_METHOD_DESC = "(const sample_int32_ctx_t ctx)";
    public static final String SPLINE_OCL_METHOD_DESC_CACHE1 = "(const sample_int32_ctx_t ctx, const float cache1)";
    private final Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline;
    private final Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> children = new Reference2ReferenceOpenHashMap<>();

    public SplineAstNode(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline) {
        this.spline = spline;
        this.populateChildrenMap(this.spline);
    }

    private SplineAstNode(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline, Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> children) {
        this.spline = spline;
        this.children.putAll(children);
    }

    @Override
    public double evalSingle(int x, int y, int z, EvalType type) {
        return spline.apply(new DensityFunctionTypes.Spline.SplinePos(new NoisePosVanillaInterface(x, y, z, type)));
    }

    @Override
    public void evalMulti(double[] res, int[] x, int[] y, int[] z, EvalType type) {
        for (int i = 0; i < res.length; i++) {
            res[i] = this.evalSingle(x[i], y[i], z[i], type);
        }
    }

    @Override
    public AstNode[] getChildren() {
        return this.children.values().toArray(AstNode[]::new);
    }

    @Override
    public AstNode transform(AstTransformer transformer) {
        boolean isModified = false;
        Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> modified = new Reference2ReferenceOpenHashMap<>(this.children);
        for (Map.Entry<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> entry : modified.entrySet()) {
            AstNode node = entry.getValue();
            AstNode transformed = node.transform(transformer);
            if (node != transformed) {
                isModified = true;
                entry.setValue(transformed);
            }
        }
        if (isModified) {
            return transformer.transform(new SplineAstNode(this.spline, modified));
        } else {
            return transformer.transform(this);
        }
    }

    @Override
    public void doBytecodeGenSingle(BytecodeGen.Context context, InstructionAdapter m, BytecodeGen.Context.LocalVarConsumer localVarConsumer) {
        ValuesMethodDefF splineMethod = doBytecodeGenSpline(context, this.spline, false);
        callSplineSingle(context, m, splineMethod, -1);
        m.cast(Type.FLOAT_TYPE, Type.DOUBLE_TYPE);
        m.areturn(Type.DOUBLE_TYPE);
    }

    private ValuesMethodDefF doBytecodeGenSpline(BytecodeGen.Context context, Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline, boolean cache1) {
        {
            String cachedSplineMethod = context.getCachedSplineMethod(spline, cache1);
            if (cachedSplineMethod != null) {
                return new ValuesMethodDefF(cachedSplineMethod);
            }
        }
        if (spline instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline1) {
            return new ValuesMethodDefF(spline1.value());
        }
        String name = context.nextMethodName("Spline");
        InstructionAdapter m = new InstructionAdapter(
                new AnalyzerAdapter(
                        context.className,
                        Opcodes.ACC_PRIVATE | Opcodes.ACC_FINAL,
                        name,
                        cache1 ? SPLINE_METHOD_DESC_CACHE1 : SPLINE_METHOD_DESC,
                        context.classWriter.visitMethod(
                                Opcodes.ACC_PRIVATE | Opcodes.ACC_FINAL,
                                name,
                                cache1 ? SPLINE_METHOD_DESC_CACHE1 : SPLINE_METHOD_DESC,
                                null,
                                null
                        )
                )
        );
        List<IntObjectPair<Pair<String, String>>> extraLocals = new ArrayList<>();
        Label start = new Label();
        Label end = new Label();
        m.visitLabel(start);
        BytecodeGen.Context.LocalVarConsumer localVarConsumer = (localName, localDesc) -> {
            int ordinal = extraLocals.size() + (cache1 ? 6 : 5);
            extraLocals.add(IntObjectPair.of(ordinal, Pair.of(localName, localDesc)));
            return ordinal;
        };

        if (spline instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> impl) {
//            BytecodeGen.Context.ValuesMethodDefF[] valuesMethods = impl.values().stream()
//                    .map(spline1 -> doBytecodeGenSpline(context, spline1))
//                    .toArray(BytecodeGen.Context.ValuesMethodDefF[]::new);

            String locations = context.newField(float[].class, impl.locations());
            String derivatives = context.newField(float[].class, impl.derivatives());

            int point = cache1 ? 5 : localVarConsumer.createLocalVariable("point", Type.FLOAT_TYPE.getDescriptor());
            int rangeForLocation = localVarConsumer.createLocalVariable("rangeForLocation", Type.INT_TYPE.getDescriptor());

            int lastConst = impl.locations().length - 1;

            if (!cache1) {
                ValuesMethodDefD locationFunction = context.newSingleMethod(this.children.get(impl.locationFunction()));
                context.callDelegateSingle(m, locationFunction);
                m.cast(Type.DOUBLE_TYPE, Type.FLOAT_TYPE);
                m.store(point, Type.FLOAT_TYPE);
            }

            int valuesMethodsLength = impl.values().size();
            if (valuesMethodsLength == 1) {
                m.load(point, Type.FLOAT_TYPE);
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        locations,
                        Type.getDescriptor(float[].class)
                );
                callSplineSingle(context, m, doBytecodeGenSpline(context, impl.values().getFirst(), false), -1);
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        derivatives,
                        Type.getDescriptor(float[].class)
                );
                m.iconst(0);
                m.invokestatic(
                        Type.getInternalName(SplineSupport.class),
                        "sampleOutsideRange",
                        Type.getMethodDescriptor(Type.FLOAT_TYPE, Type.FLOAT_TYPE, Type.getType(float[].class), Type.FLOAT_TYPE, Type.getType(float[].class), Type.INT_TYPE),
                        false
                );
                m.areturn(Type.FLOAT_TYPE);
            } else {
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        locations,
                        Type.getDescriptor(float[].class)
                );
                m.load(point, Type.FLOAT_TYPE);
                m.invokestatic(
                        Type.getInternalName(SplineSupport.class),
                        "findRangeForLocation",
                        Type.getMethodDescriptor(Type.INT_TYPE, Type.getType(float[].class), Type.FLOAT_TYPE),
                        false
                );
                m.store(rangeForLocation, Type.INT_TYPE);

                Label label1 = new Label();
                Label label2 = new Label();

                m.load(rangeForLocation, Type.INT_TYPE);
                m.ifge(label1);
                // rangeForLocation < 0
                m.load(point, Type.FLOAT_TYPE);
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        locations,
                        Type.getDescriptor(float[].class)
                );
                callSplineSingle(context, m, doBytecodeGenSpline(context, impl.values().getFirst(), false), -1);
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        derivatives,
                        Type.getDescriptor(float[].class)
                );
                m.iconst(0);
                m.invokestatic(
                        Type.getInternalName(SplineSupport.class),
                        "sampleOutsideRange",
                        Type.getMethodDescriptor(Type.FLOAT_TYPE, Type.FLOAT_TYPE, Type.getType(float[].class), Type.FLOAT_TYPE, Type.getType(float[].class), Type.INT_TYPE),
                        false
                );
                m.areturn(Type.FLOAT_TYPE);

                m.visitLabel(label1);
                m.load(rangeForLocation, Type.INT_TYPE);
                m.iconst(lastConst);
                m.ificmpne(label2);
                // rangeForLocation == last
                m.load(point, Type.FLOAT_TYPE);
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        locations,
                        Type.getDescriptor(float[].class)
                );
                callSplineSingle(context, m, doBytecodeGenSpline(context, impl.values().get(lastConst), false), -1);
                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        derivatives,
                        Type.getDescriptor(float[].class)
                );
                m.iconst(lastConst);
                m.invokestatic(
                        Type.getInternalName(SplineSupport.class),
                        "sampleOutsideRange",
                        Type.getMethodDescriptor(Type.FLOAT_TYPE, Type.FLOAT_TYPE, Type.getType(float[].class), Type.FLOAT_TYPE, Type.getType(float[].class), Type.INT_TYPE),
                        false
                );
                m.areturn(Type.FLOAT_TYPE);

                m.visitLabel(label2);

                int loc0 = localVarConsumer.createLocalVariable("loc0", Type.FLOAT_TYPE.getDescriptor());
                int loc1 = localVarConsumer.createLocalVariable("loc1", Type.FLOAT_TYPE.getDescriptor());
                int locDist = localVarConsumer.createLocalVariable("locDist", Type.FLOAT_TYPE.getDescriptor());
                int k = localVarConsumer.createLocalVariable("k", Type.FLOAT_TYPE.getDescriptor());
                int n = localVarConsumer.createLocalVariable("n", Type.FLOAT_TYPE.getDescriptor());
                int o = localVarConsumer.createLocalVariable("o", Type.FLOAT_TYPE.getDescriptor());
                int onDist = localVarConsumer.createLocalVariable("onDist", Type.FLOAT_TYPE.getDescriptor());
                int p = localVarConsumer.createLocalVariable("p", Type.FLOAT_TYPE.getDescriptor());
                int q = localVarConsumer.createLocalVariable("q", Type.FLOAT_TYPE.getDescriptor());

                int cache1Local = localVarConsumer.createLocalVariable("cache1Local", Type.FLOAT_TYPE.getDescriptor());

                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        locations,
                        Type.getDescriptor(float[].class)
                );
                m.load(rangeForLocation, Type.INT_TYPE);
                m.aload(Type.FLOAT_TYPE);
                m.store(loc0, Type.FLOAT_TYPE);

                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        locations,
                        Type.getDescriptor(float[].class)
                );
                m.load(rangeForLocation, Type.INT_TYPE);
                m.iconst(1);
                m.add(Type.INT_TYPE);
                m.aload(Type.FLOAT_TYPE);
                m.store(loc1, Type.FLOAT_TYPE);

                m.load(loc1, Type.FLOAT_TYPE);
                m.load(loc0, Type.FLOAT_TYPE);
                m.sub(Type.FLOAT_TYPE);
                m.store(locDist, Type.FLOAT_TYPE);

                m.load(point, Type.FLOAT_TYPE);
                m.load(loc0, Type.FLOAT_TYPE);
                m.sub(Type.FLOAT_TYPE);
                m.load(locDist, Type.FLOAT_TYPE);
                m.div(Type.FLOAT_TYPE);
                m.store(k, Type.FLOAT_TYPE);

                Label[] jumpLabels = new Label[valuesMethodsLength - 1];
                boolean[] jumpGenerated = new boolean[valuesMethodsLength - 1];
                for (int i = 0; i < valuesMethodsLength - 1; i++) {
                    jumpLabels[i] = new Label();
                }
                Label defaultLabel = new Label();
                Label label3 = new Label();

                m.load(rangeForLocation, Type.INT_TYPE);
                m.tableswitch(
                        0,
                        valuesMethodsLength - 2,
                        defaultLabel,
                        jumpLabels
                );

                for (int i = 0; i < valuesMethodsLength - 1; i++) {
                    if (jumpGenerated[i]) continue;
                    m.visitLabel(jumpLabels[i]);
                    jumpGenerated[i] = true;
                    for (int j = i + 1; j < valuesMethodsLength - 1; j++) { // deduplication
                        if (impl.values().get(i).equals(impl.values().get(j)) && impl.values().get(i + 1).equals(impl.values().get(j + 1))) {
                            m.visitLabel(jumpLabels[j]);
                            jumpGenerated[j] = true;
                        }
                    }

                    boolean optimizePure = impl.values().get(i).equals(impl.values().get(i + 1));
                    AstNode sameLocationFunction = needOptimizeSameLocationFunction(impl.values().get(i), impl.values().get(i + 1));
                    boolean doCache1 = !optimizePure && sameLocationFunction != null;
                    if (doCache1) {
                        ValuesMethodDefD locationFunction = context.newSingleMethod(sameLocationFunction);
                        context.callDelegateSingle(m, locationFunction);
                        m.cast(Type.DOUBLE_TYPE, Type.FLOAT_TYPE);
                        m.store(cache1Local, Type.FLOAT_TYPE);
                    }
                    callSplineSingle(context, m, doBytecodeGenSpline(context, impl.values().get(i), doCache1), doCache1 ? cache1Local : -1);
                    if (optimizePure) { // splines are pure
                        m.dup();
                        m.store(n, Type.FLOAT_TYPE);
                        m.store(o, Type.FLOAT_TYPE);
                    } else {
                        m.store(n, Type.FLOAT_TYPE);
                        callSplineSingle(context, m, doBytecodeGenSpline(context, impl.values().get(i + 1), doCache1), doCache1 ? cache1Local : -1);
                        m.store(o, Type.FLOAT_TYPE);
                    }
                    m.goTo(label3);
                }

                m.visitLabel(defaultLabel);
                m.iconst(0);
                m.aconst("boom");
                m.invokestatic(
                        Type.getInternalName(Assertions.class),
                        "assertTrue",
                        Type.getMethodDescriptor(Type.VOID_TYPE, Type.BOOLEAN_TYPE, Type.getType(String.class)),
                        false
                );
                m.fconst(Float.NaN); // unreachable code
                m.areturn(Type.FLOAT_TYPE);

                m.visitLabel(label3);

                m.load(o, Type.FLOAT_TYPE);
                m.load(n, Type.FLOAT_TYPE);
                m.sub(Type.FLOAT_TYPE);
                m.store(onDist, Type.FLOAT_TYPE);

                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        derivatives,
                        Type.getDescriptor(float[].class)
                );
                m.load(rangeForLocation, Type.INT_TYPE);
                m.aload(Type.FLOAT_TYPE);
                m.load(locDist, Type.FLOAT_TYPE);
                m.mul(Type.FLOAT_TYPE);
                m.load(onDist, Type.FLOAT_TYPE);
                m.sub(Type.FLOAT_TYPE);
                m.store(p, Type.FLOAT_TYPE);

                m.load(0, InstructionAdapter.OBJECT_TYPE);
                m.getfield(
                        context.className,
                        derivatives,
                        Type.getDescriptor(float[].class)
                );
                m.load(rangeForLocation, Type.INT_TYPE);
                m.iconst(1);
                m.add(Type.INT_TYPE);
                m.aload(Type.FLOAT_TYPE);
                m.neg(Type.FLOAT_TYPE);
                m.load(locDist, Type.FLOAT_TYPE);
                m.mul(Type.FLOAT_TYPE);
                m.load(onDist, Type.FLOAT_TYPE);
                m.add(Type.FLOAT_TYPE);
                m.store(q, Type.FLOAT_TYPE);

                m.load(k, Type.FLOAT_TYPE);
                m.load(n, Type.FLOAT_TYPE);
                m.load(o, Type.FLOAT_TYPE);
                m.invokestatic(
                        Type.getInternalName(MathHelper.class),
                        FabricLoader.getInstance().getMappingResolver().mapMethodName("intermediary", "net.minecraft.class_3532", "method_16439", "(FFF)F"),
                        "(FFF)F",
                        false
                );
                m.load(k, Type.FLOAT_TYPE);
                m.fconst(1.0F);
                m.load(k, Type.FLOAT_TYPE);
                m.sub(Type.FLOAT_TYPE);
                m.mul(Type.FLOAT_TYPE);
                m.load(k, Type.FLOAT_TYPE);
                m.load(p, Type.FLOAT_TYPE);
                m.load(q, Type.FLOAT_TYPE);
                m.invokestatic(
                        Type.getInternalName(MathHelper.class),
                        FabricLoader.getInstance().getMappingResolver().mapMethodName("intermediary", "net.minecraft.class_3532", "method_16439", "(FFF)F"),
                        "(FFF)F",
                        false
                );
                m.mul(Type.FLOAT_TYPE);
                m.add(Type.FLOAT_TYPE);
                m.areturn(Type.FLOAT_TYPE);
            }

        } else if (spline instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> floatFunction) {
            m.fconst(floatFunction.value());
            m.areturn(Type.FLOAT_TYPE);
        } else {
            throw new UnsupportedOperationException(String.format("Unsupported spline implementation: %s", spline.getClass().getName()));
        }

        m.visitLabel(end);
        m.visitLocalVariable("this", context.classDesc, null, start, end, 0);
        m.visitLocalVariable("x", Type.INT_TYPE.getDescriptor(), null, start, end, 1);
        m.visitLocalVariable("y", Type.INT_TYPE.getDescriptor(), null, start, end, 2);
        m.visitLocalVariable("z", Type.INT_TYPE.getDescriptor(), null, start, end, 3);
        m.visitLocalVariable("evalType", Type.getType(EvalType.class).getDescriptor(), null, start, end, 4);
        if (cache1) {
            m.visitLocalVariable("pointCached", Type.FLOAT_TYPE.getDescriptor(), null, start, end, 5);
        }
        for (IntObjectPair<Pair<String, String>> local : extraLocals) {
            m.visitLocalVariable(local.right().left(), local.right().right(), null, start, end, local.leftInt());
        }
        m.visitMaxs(0, 0);

        context.cacheSplineMethod(spline, name, cache1);

        return new ValuesMethodDefF(name);
    }

    private AstNode needOptimizeSameLocationFunction(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper>... splines) {
        if (splines.length == 0) return null;

        AstNode a1Ast;
        if (splines[0] instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline0) {
            a1Ast = this.children.get(spline0.locationFunction());
            if (a1Ast instanceof ConstantNode) {
                return null;
            }
        } else {
            return null;
        }

        for (int i = 1, splinesLength = splines.length; i < splinesLength; i++) {
            Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b = splines[i];
            if (b instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b1) {
                AstNode b1Ast = this.children.get(b1.locationFunction());
                if (!a1Ast.equals(b1Ast)) {
                    return null;
                }
            } else {
                return null;
            }
        }

        return a1Ast;
    }

    private static void callSplineSingle(BytecodeGen.Context context, InstructionAdapter m, ValuesMethodDefF target, int cache1Local) {
        if (target.isConst()) {
            m.fconst(target.constValue());
        } else {
            boolean doCache1 = cache1Local != -1;
            m.load(0, InstructionAdapter.OBJECT_TYPE);
            m.load(1, Type.INT_TYPE);
            m.load(2, Type.INT_TYPE);
            m.load(3, Type.INT_TYPE);
            m.load(4, InstructionAdapter.OBJECT_TYPE);
            if (doCache1) {
                m.load(cache1Local, Type.FLOAT_TYPE);
            }
            m.invokevirtual(context.className, target.generatedMethod(), doCache1 ? SPLINE_METHOD_DESC_CACHE1 : SPLINE_METHOD_DESC, false);
        }
    }

    @Override
    public void doBytecodeGenMulti(BytecodeGen.Context context, InstructionAdapter m, BytecodeGen.Context.LocalVarConsumer localVarConsumer) {
        context.delegateAllToSingle(m, localVarConsumer, this);
        m.areturn(Type.VOID_TYPE);
    }

    private ValuesMethodDefF doCLGenSpline(OpenCLGen.Context context, Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline, boolean cache1) {
        {
            String cachedSplineMethod = context.getCachedSplineMethod(spline, cache1);
            if (cachedSplineMethod != null) {
                return new ValuesMethodDefF(cachedSplineMethod);
            }
        }
        if (spline instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline1) {
            return new ValuesMethodDefF(spline1.value());
        }
        String name = context.nextMethodName("Spline");
        StringBuilder predef = new StringBuilder();
        StringBuilder body = new StringBuilder();

        boolean noinline = false;

        if (spline instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> impl) {

            String locations = name + "_locations";
            String derivatives = name + "_derivatives";
            predef.append("constant const float ").append(locations).append("[").append(impl.locations().length).append("] = ").append(literal(impl.locations())).append(";\n");
            predef.append("constant const float ").append(derivatives).append("[").append(impl.derivatives().length).append("] = ").append(literal(impl.derivatives())).append(";\n");

            int lastConst = impl.locations().length - 1;

            if (cache1) {
                body.append("float point = cache1;\n");
            } else {
                ValuesMethodDefD locationFunction = context.newMethod(this.children.get(impl.locationFunction()));
                body.append("float point = (float) ").append(context.callDelegate(locationFunction)).append(";\n");
            }

            int valuesMethodsLength = impl.values().size();
            if (valuesMethodsLength == 1) {
                body
                        .append("return df_spline_sampleOutsideRange_const(point, ")
                        .append(locations).append(", ")
                        .append(callSpline(doCLGenSpline(context, impl.values().getFirst(), false), null)).append(", ")
                        .append(derivatives).append(", 0);\n");
            } else {
                body
                        .append("float cache1Local;\n");

                AstNode allSameLocationFunction = needOptimizeSameLocationFunction(impl.values().toArray(Spline[]::new));
                if (allSameLocationFunction != null) {
                    body.append("cache1Local = (float) ").append(context.callDelegate(context.newMethod(allSameLocationFunction))).append(";\n");
                } else {
                    if (impl.values().stream().filter(child -> child instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper>).count() > 5) { // some abitrary number
                        noinline = true;
                    }
                }

                body
                        .append("int32_t rangeForLocation = df_spline_findRangeForLocation_const(").append(locations).append(", ").append(impl.locations().length).append(", ").append("point);\n")
                        .append("if (rangeForLocation < 0) {\n")
                        .append("    ").append("return df_spline_sampleOutsideRange_const(point, ").append(locations).append(", ").append(callSpline(doCLGenSpline(context, impl.values().getFirst(), allSameLocationFunction != null), allSameLocationFunction != null ? "cache1Local" : null)).append(", ").append(derivatives).append(", 0);\n")
                        .append("} else if (rangeForLocation == ").append(lastConst).append(") {\n")
                        .append("    ").append("return df_spline_sampleOutsideRange_const(point, ").append(locations).append(", ").append(callSpline(doCLGenSpline(context, impl.values().get(lastConst), allSameLocationFunction != null), allSameLocationFunction != null ? "cache1Local" : null)).append(", ").append(derivatives).append(", ").append(lastConst).append(");\n")
                        .append("}\n")
                        .append("float loc0 = ").append(locations).append("[rangeForLocation];\n")
                        .append("float loc1 = ").append(locations).append("[rangeForLocation + 1];\n")
                        .append("float locDist = loc1 - loc0;\n")
                        .append("float k = (point - loc0) / locDist;\n")
                        .append("float n, o;\n");

                body
                        .append("switch (rangeForLocation) {\n");

                boolean[] jumpGenerated = new boolean[valuesMethodsLength - 1];

                for (int i = 0; i < valuesMethodsLength - 1; i++) {
                    if (jumpGenerated[i]) continue;
                    body.append("    ").append("case ").append(i).append(":\n");
                    jumpGenerated[i] = true;
                    for (int j = i + 1; j < valuesMethodsLength - 1; j++) { // deduplication
                        if (impl.values().get(i).equals(impl.values().get(j)) && impl.values().get(i + 1).equals(impl.values().get(j + 1))) {
                            body.append("    ").append("case ").append(j).append(":\n");
                            jumpGenerated[j] = true;
                        }
                    }

                    boolean optimizePure = impl.values().get(i).equals(impl.values().get(i + 1));
                    AstNode sameLocationFunction;
                    boolean doCache1;
                    if (allSameLocationFunction != null) {
                        sameLocationFunction = null; // avoid double calc
                        doCache1 = true;
                    } else {
                        sameLocationFunction = needOptimizeSameLocationFunction(impl.values().get(i), impl.values().get(i + 1));
                        doCache1 = sameLocationFunction != null;
                    }
                    if (optimizePure) { // splines are pure
                        if (doCache1) { // to encourage compiler to do loop common extraction maybe
                            if (sameLocationFunction != null) { // avoid double calc
                                body.append("    ").append("    ").append("cache1Local = (float) ").append(context.callDelegate(context.newMethod(sameLocationFunction))).append(";\n");
                            }
                            body.append("    ").append("    ").append("n = o = ").append(callSpline(doCLGenSpline(context, impl.values().get(i), true), "cache1Local")).append(";\n");
                        } else {
                            body.append("    ").append("    ").append("n = o = ").append(callSpline(doCLGenSpline(context, impl.values().get(i), false), null)).append(";\n");
                        }
                    } else {
                        if (doCache1) {
                            if (sameLocationFunction != null) { // avoid double calc
                                body.append("    ").append("    ").append("cache1Local = (float) ").append(context.callDelegate(context.newMethod(sameLocationFunction))).append(";\n");
                            }
                            body.append("    ").append("    ").append("n = ").append(callSpline(doCLGenSpline(context, impl.values().get(i), true), "cache1Local")).append(";\n");
                            body.append("    ").append("    ").append("o = ").append(callSpline(doCLGenSpline(context, impl.values().get(i + 1), true), "cache1Local")).append(";\n");
                        } else {
                            body.append("    ").append("    ").append("n = ").append(callSpline(doCLGenSpline(context, impl.values().get(i), false), null)).append(";\n");
                            body.append("    ").append("    ").append("o = ").append(callSpline(doCLGenSpline(context, impl.values().get(i + 1), false), null)).append(";\n");
                        }
                    }
                    body.append("    ").append("    ").append("break;\n");
                }

                body
                        .append("    ").append("default:\n")
                        .append("    ").append("    ").append("__builtin_trap();\n")
                        .append("    ").append("    ").append("return nan((uint64_t) 0);\n") // unreachable
                        .append("}\n")
                        .append("float onDist = o - n;\n")
                        .append("float p = ").append(derivatives).append("[rangeForLocation] * locDist - onDist;\n")
                        .append("float q = -").append(derivatives).append("[rangeForLocation + 1] * locDist + onDist;\n")
                        .append("return math_lerpf(k, n, o) + k * (1.0F - k) * math_lerpf(k, p, q);\n");
            }
        } else if (spline instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> floatFunction) {
            body.append("return ").append(literal(floatFunction.value())).append(";\n");
        } else {
            throw new UnsupportedOperationException(String.format("Unsupported spline implementation: %s", spline.getClass().getName()));
        }

        context.appendRaw(
                new StringBuilder()
                        .append(predef)
                        .append("static __attribute__((pure))").append(noinline ? " FUNC_NOINLINE" : "").append(" float ").append(name).append(cache1 ? SPLINE_OCL_METHOD_DESC_CACHE1 : SPLINE_OCL_METHOD_DESC).append(" {\n")
                        .append(body.toString().indent(4))
                        .append("}\n")
                        .toString()
        );

        context.cacheSplineMethod(spline, name, cache1);

        return new ValuesMethodDefF(name);
    }

    private static String callSpline(ValuesMethodDefF target, String cache1Arg) {
        if (target.isConst()) {
            return OpenCLGen.literal(target.constValue());
        } else if (cache1Arg != null) {
            return target.generatedMethod() + "(ctx, " + cache1Arg + ")";
        } else {
            return target.generatedMethod() + "(ctx)";
        }
    }

    @Override
    public String doCLGen(OpenCLGen.Context context) {
        ValuesMethodDefF valuesMethodDefF = doCLGenSpline(context, this.spline, false);
        return "return (double) " + callSpline(valuesMethodDefF, null) + ";\n";
    }

    private void populateChildrenMap(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a) {
        if (a instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1) {
            for (Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline : a1.values()) {
                populateChildrenMap(spline);
            }
            DensityFunctionTypes.Spline.DensityFunctionWrapper locationFunction = a1.locationFunction();
            this.children.put(locationFunction, McToAst.toAst(locationFunction.function().value()));
        }
    }

    private static boolean deepEquals(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a,
                                      Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> childrenA,
                                      Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b,
                                      Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> childrenB) {
        if (a instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1 &&
                b instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b1) {
            return a1.value() == b1.value();
        } else if (a instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1 &&
                b instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b1) {
            boolean equals1 = Arrays.equals(a1.derivatives(), b1.derivatives()) &&
                    Arrays.equals(a1.locations(), b1.locations()) &&
                    a1.values().size() == b1.values().size() &&
                    childrenA.get(a1.locationFunction()).equals(childrenB.get(b1.locationFunction()));
            if (!equals1) return false;
            int size = a1.values().size();
            for (int i = 0; i < size; i++) {
                if (!deepEquals(a1.values().get(i), childrenA, b1.values().get(i), childrenB)) {
                    return false;
                }
            }

            return true;
        } else {
            return false;
        }
    }

    private static boolean deepRelaxedEquals(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a,
                                             Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> childrenA,
                                             Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b,
                                             Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> childrenB) {
        if (a instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1 &&
                b instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b1) {
            return a1.value() == b1.value();
        } else if (a instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1 &&
                b instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> b1) {
            boolean equals1 = Arrays.equals(a1.derivatives(), b1.derivatives()) &&
                    Arrays.equals(a1.locations(), b1.locations()) &&
                    a1.values().size() == b1.values().size() &&
                    childrenA.get(a1.locationFunction()).relaxedEquals(childrenB.get(b1.locationFunction()));
            if (!equals1) return false;
            int size = a1.values().size();
            for (int i = 0; i < size; i++) {
                if (!deepRelaxedEquals(a1.values().get(i), childrenA, b1.values().get(i), childrenB)) {
                    return false;
                }
            }

            return true;
        } else {
            return false;
        }
    }

    private static int deepHashcode(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a, Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> childrenA) {
        if (a instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1) {
            return Float.hashCode(a1.value());
        } else if (a instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1) {
            int result = 1;

            result = 31 * result + Arrays.hashCode(a1.derivatives());
            result = 31 * result + Arrays.hashCode(a1.locations());
            for (Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline : a1.values()) {
                result = 31 * result + deepHashcode(spline, childrenA);
            }
            result = 31 * result + childrenA.get(a1.locationFunction()).hashCode();

            return result;
        } else {
            return a.hashCode();
        }
    }

    private static int deepRelaxedHashcode(Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a, Reference2ReferenceMap<DensityFunctionTypes.Spline.DensityFunctionWrapper, AstNode> childrenA) {
        if (a instanceof Spline.FixedFloatFunction<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1) {
            return Float.hashCode(a1.value());
        } else if (a instanceof Spline.Implementation<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> a1) {
            int result = 1;

            result = 31 * result + Arrays.hashCode(a1.derivatives());
            result = 31 * result + Arrays.hashCode(a1.locations());
            for (Spline<DensityFunctionTypes.Spline.SplinePos, DensityFunctionTypes.Spline.DensityFunctionWrapper> spline : a1.values()) {
                result = 31 * result + deepRelaxedHashcode(spline, childrenA);
            }
            result = 31 * result + childrenA.get(a1.locationFunction()).relaxedHashCode();

            return result;
        } else {
            return a.hashCode();
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SplineAstNode that = (SplineAstNode) o;
        return deepEquals(this.spline, this.children, that.spline, that.children);
    }

    @Override
    public int hashCode() {
        return deepHashcode(this.spline, this.children);
    }

    @Override
    public boolean relaxedEquals(AstNode o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SplineAstNode that = (SplineAstNode) o;
        return deepRelaxedEquals(this.spline, this.children, that.spline, that.children);
    }

    @Override
    public int relaxedHashCode() {
        return deepRelaxedHashcode(this.spline, this.children);
    }
}
