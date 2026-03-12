package com.ishland.c2me.opts.dfc.mixin;

import com.ishland.c2me.opts.dfc.common.ducks.NoiseRouterExtension;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.noise.NoiseRouter;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(NoiseRouter.class)
public class MixinNoiseRouter implements NoiseRouterExtension {

    @Unique
    private DensityFunction c2me$finalFinalDensity;

    @Unique
    private NoiseRouter c2me$originalNoiseRouter;

    @Override
    public DensityFunction c2me$getFinalFinalDensity() {
        return this.c2me$finalFinalDensity;
    }

    @Override
    public void c2me$setFinalFinalDensity(DensityFunction densityFunction) {
        this.c2me$finalFinalDensity = densityFunction;
    }

    @Unique
    public void c2me$setOriginalNoiseRouter(NoiseRouter originalNoiseRouter) {
        this.c2me$originalNoiseRouter = originalNoiseRouter;
    }

    @Unique
    public NoiseRouter c2me$getOriginalNoiseRouter() {
        return this.c2me$originalNoiseRouter;
    }

    @Inject(method = "apply", at = @At("RETURN"))
    private void postApply(DensityFunction.DensityFunctionVisitor visitor, CallbackInfoReturnable<NoiseRouter> cir) {
        if (this.c2me$finalFinalDensity != null) {
            ((MixinNoiseRouter) (Object) cir.getReturnValue()).c2me$finalFinalDensity = this.c2me$finalFinalDensity.apply(visitor);
        }
        ((MixinNoiseRouter) (Object) cir.getReturnValue()).c2me$originalNoiseRouter = this.c2me$originalNoiseRouter;
    }

}
