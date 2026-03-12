package com.ishland.c2me.base.mixin.exp_warning;

import net.minecraft.server.Main;
import org.slf4j.Logger;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Main.class)
public class MixinDedicatedServerMain {

    @Shadow
    @Final
    private static Logger LOGGER;

    @Inject(method = "main", at = @At(value = "INVOKE", target = "Lnet/minecraft/server/dedicated/EulaReader;<init>(Ljava/nio/file/Path;)V"))
    private static void earlyInit(String[] args, CallbackInfo ci) {
        LOGGER.warn("Be careful! You are running a experimental version of C2ME");
        LOGGER.warn("This version include features that are still under development. Your world might crash, break. Use at your own risk!");
        if (!Boolean.getBoolean("com.ishland.c2me.base.skipWarningWait")) {
            while (true) {
                LOGGER.warn("The server will start in 10 seconds...");
                try {
                    Thread.sleep(10000);
                    break;
                } catch (InterruptedException e) {
                    Thread.interrupted();
                }
            }
        }
    }

}
