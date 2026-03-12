package com.ishland.c2me.opts.dfc.mixin;

import com.google.common.base.Stopwatch;
import com.ishland.c2me.opts.dfc.common.ast.AstNode;
import com.ishland.c2me.opts.dfc.common.ducks.NoiseRouterExtension;
import com.ishland.c2me.opts.dfc.common.gen.jvm.BytecodeGen;
import com.ishland.c2me.opts.dfc.common.gen.opencl.GeneratedCLSource;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import it.unimi.dsi.fastutil.objects.Reference2ReferenceMap;
import it.unimi.dsi.fastutil.objects.Reference2ReferenceOpenHashMap;
import net.minecraft.registry.RegistryEntryLookup;
import net.minecraft.util.math.noise.DoublePerlinNoiseSampler;
import net.minecraft.world.biome.source.util.MultiNoiseUtil;
import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.noise.NoiseRouter;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(value = NoiseConfig.class, priority = 900)
public class MixinNoiseConfig {

    @Mutable
    @Shadow @Final private NoiseRouter noiseRouter;

    @Mutable
    @Shadow @Final private MultiNoiseUtil.MultiNoiseSampler multiNoiseSampler;

    @Inject(method = "<init>", at = @At("RETURN"))
    private void postCreate(ChunkGeneratorSettings chunkGeneratorSettings, RegistryEntryLookup<DoublePerlinNoiseSampler.NoiseParameters> noiseParametersLookup, long seed, CallbackInfo ci) {
        Stopwatch stopwatch = Stopwatch.createStarted();
        Reference2ReferenceMap<DensityFunction, AstNode> optoCache = new Reference2ReferenceOpenHashMap<>();
        Reference2ReferenceMap<DensityFunction, DensityFunction> tempCache = new Reference2ReferenceOpenHashMap<>();
        DensityFunction finalFinalDensity = DensityFunctionTypes.add(this.noiseRouter.finalDensity(), DensityFunctionTypes.Beardifier.INSTANCE);
        NoiseRouter original = this.noiseRouter;
        ((NoiseRouterExtension) (Object) original).c2me$setFinalFinalDensity(finalFinalDensity);
        this.noiseRouter = new NoiseRouter(
                BytecodeGen.compile(this.noiseRouter.barrierNoise(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.fluidLevelFloodednessNoise(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.fluidLevelSpreadNoise(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.lavaNoise(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.temperature(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.vegetation(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.continents(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.erosion(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.depth(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.ridges(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.preliminarySurfaceLevel(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.finalDensity(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.veinToggle(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.veinRidged(), optoCache, tempCache),
                BytecodeGen.compile(this.noiseRouter.veinGap(), optoCache, tempCache)
        );
        ((NoiseRouterExtension) (Object) this.noiseRouter).c2me$setFinalFinalDensity(
                BytecodeGen.compile(
                        finalFinalDensity,
                        optoCache,
                        tempCache
                )
        );
        ((NoiseRouterExtension) (Object) this.noiseRouter).c2me$setOriginalNoiseRouter(original);
        this.multiNoiseSampler = new MultiNoiseUtil.MultiNoiseSampler(
                BytecodeGen.compile(this.multiNoiseSampler.temperature(), optoCache, tempCache),
                BytecodeGen.compile(this.multiNoiseSampler.humidity(), optoCache, tempCache),
                BytecodeGen.compile(this.multiNoiseSampler.continentalness(), optoCache, tempCache),
                BytecodeGen.compile(this.multiNoiseSampler.erosion(), optoCache, tempCache),
                BytecodeGen.compile(this.multiNoiseSampler.depth(), optoCache, tempCache),
                BytecodeGen.compile(this.multiNoiseSampler.weirdness(), optoCache, tempCache),
                this.multiNoiseSampler.spawnTarget()
        );
        stopwatch.stop();
        System.out.println(String.format("Density function compilation finished in %s", stopwatch));
    }

}
