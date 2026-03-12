package com.ishland.c2me.opts.accel.opencl.mixin;

import com.ishland.c2me.opts.accel.opencl.common.Config;
import com.ishland.c2me.opts.accel.opencl.common.ducks.MinecraftServerExtension;
import com.ishland.c2me.opts.accel.opencl.common.enumeration.OpenCLDeviceLocator;
import com.ishland.c2me.opts.accel.opencl.common.enumeration.OpenCLDeviceMetadata;
import com.ishland.c2me.opts.accel.opencl.common.gen.CLServerGlobalContext;
import net.minecraft.server.MinecraftServer;
import org.slf4j.Logger;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.List;

@Mixin(MinecraftServer.class)
public class MixinMinecraftServer implements MinecraftServerExtension {

    @Shadow @Final private static Logger LOGGER;

    @Unique
    private CLServerGlobalContext c2me$clContext;

    @Inject(method = "runServer", at = @At("HEAD"))
    private void preRunServer(CallbackInfo ci) {
        try {
            if (this.c2me$clContext != null) {
                throw new IllegalStateException("Context already exists?");
            }
            this.c2me$clContext = new CLServerGlobalContext();
            List<OpenCLDeviceMetadata> metadataList = OpenCLDeviceLocator.enumerateAll();
            if (metadataList.isEmpty()) {
                LOGGER.warn("No OpenCL devices found");
                if (!Config.allowIncompatibilityFallback) {
                    throw new IllegalStateException("No OpenCL devices found");
                }
                return;
            }
            for (OpenCLDeviceMetadata openCLDeviceMetadata : metadataList) {
                if (!Config.deviceUUIDWhitelist.isEmpty() && !Config.deviceUUIDWhitelist.contains(openCLDeviceMetadata.deviceUUID)) {
                    LOGGER.info("Skipping OpenCL device {} since it's not in the whitelist", openCLDeviceMetadata.deviceUUID);
                    continue;
                }
                if (Config.deviceUUIDBlacklist.contains(openCLDeviceMetadata.deviceUUID)) {
                    LOGGER.info("Skipping OpenCL device {} since it's in the blacklist", openCLDeviceMetadata.deviceUUID);
                    continue;
                }
                this.c2me$clContext.openDevice(openCLDeviceMetadata);
            }
        } catch (Throwable t) {
            LOGGER.error("Failed to initialize OpenCL context", t);
            this.c2me$clContext = null;
            if (!Config.allowIncompatibilityFallback) {
                throw t;
            }
        }
    }

    @Inject(method = "shutdown", at = @At("RETURN"))
    private void postStopServer(CallbackInfo ci) {
        try {
            this.c2me$clContext.closeAllDevices();
            this.c2me$clContext = null;
        } catch (Throwable t) {
            LOGGER.error("Failed to release OpenCL context", t);
        }
    }

    @Override
    public CLServerGlobalContext c2me$getCLContext() {
        return this.c2me$clContext;
    }
}
