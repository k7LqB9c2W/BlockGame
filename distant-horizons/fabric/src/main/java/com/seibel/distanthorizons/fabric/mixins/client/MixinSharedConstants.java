package com.seibel.distanthorizons.fabric.mixins.client;

import com.seibel.distanthorizons.common.commonMixins.DhUpdateScreenBase;
import com.seibel.distanthorizons.common.wrappers.world.ClientLevelWrapper;
import com.seibel.distanthorizons.core.api.internal.ClientApi;
import com.seibel.distanthorizons.core.jar.updater.SelfUpdater;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import net.minecraft.SharedConstants;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import org.spongepowered.asm.mixin.*;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * At the moment this is only used for the auto updater
 *
 * @author coolGi
 */
@Mixin(SharedConstants.class)
public abstract class MixinSharedConstants
{
	@Mutable
	@Shadow @Final public static boolean IS_RUNNING_IN_IDE;
	
	@Inject(method = "<clinit>", at = @At("TAIL"))
	private static void setIsRunningInIde(CallbackInfo ci) 
	{
		IS_RUNNING_IN_IDE = true;
	}
	
}
