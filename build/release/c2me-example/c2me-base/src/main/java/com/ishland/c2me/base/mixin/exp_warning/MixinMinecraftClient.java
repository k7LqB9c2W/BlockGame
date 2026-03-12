package com.ishland.c2me.base.mixin.exp_warning;

import com.ishland.c2me.base.common.client_warning.ModExperimentalWarningScreen;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.Screen;
import org.slf4j.Logger;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.List;
import java.util.function.Function;

@Mixin(MinecraftClient.class)
public class MixinMinecraftClient {

    @Shadow
    @Final
    private static Logger LOGGER;

    @Inject(method = "createInitScreens", at = @At("RETURN"))
    private void appendInitScreen(List<Function<Runnable, Screen>> list, CallbackInfoReturnable<Boolean> cir) {
        LOGGER.warn("Be careful! You are running a experimental version of C2ME");
        LOGGER.warn("This version include features that are still under development. Your world might crash, break. Use at your own risk!");
        list.add(ModExperimentalWarningScreen::new);
    }

}
