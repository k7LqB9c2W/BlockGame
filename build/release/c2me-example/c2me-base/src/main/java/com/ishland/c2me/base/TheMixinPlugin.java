package com.ishland.c2me.base;

import com.ishland.c2me.base.common.C2MEConstants;
import com.ishland.c2me.base.common.ModuleMixinPlugin;

/**
 * Used internally for c2me-base, do not subclass.
 */
public final class TheMixinPlugin extends ModuleMixinPlugin {

    @Override
    public boolean shouldApplyMixin(String targetClassName, String mixinClassName) {
        if (!super.shouldApplyMixin(targetClassName, mixinClassName)) {
            return false;
        }

        if (mixinClassName.startsWith("com.ishland.c2me.base.mixin.util.log4j2shutdownhookisnomore."))
            return ModuleEntryPoint.disableLoggingShutdownHook;

        if (mixinClassName.startsWith("com.ishland.c2me.base.mixin.exp_warning."))
            return C2MEConstants.ENABLE_EXPERIMENTAL_WARNINGS;

        return true;
    }
}
