package com.seibel.distanthorizons.common.render.blaze.util;

#if MC_VER <= MC_1_21_10
public class BlazeDhVertexFormatUtil {}

#else
	
import com.mojang.blaze3d.vertex.VertexFormatElement;
import com.seibel.distanthorizons.api.enums.config.EDhApiRenderApi;
import com.seibel.distanthorizons.core.config.Config;
import com.seibel.distanthorizons.core.dataObjects.render.bufferBuilding.LodQuadBuilder;
import com.seibel.distanthorizons.core.dependencyInjection.SingletonInjector;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.wrapperInterfaces.IVersionConstants;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IMinecraftClientWrapper;

/**
 * @see LodQuadBuilder
 */
public class BlazeDhVertexFormatUtil
{
	private static final DhLogger LOGGER = new DhLoggerBuilder().build();
	
	
	
	public static final VertexFormatElement SCREEN_POS;
	public static final VertexFormatElement RGBA_FLOAT_COLOR;
	
	public static final VertexFormatElement SHORT_XYZ_POS;
	public static final VertexFormatElement BYTE_PAD;
	/** contains light and micro-offset */
	public static final VertexFormatElement META;
	public static final VertexFormatElement RGBA_UBYTE_COLOR;
	public static final VertexFormatElement IRIS_MATERIAL;
	public static final VertexFormatElement IRIS_NORMAL;
	
	public static final VertexFormatElement FLOAT_XYZ_POS;
	
	
	
	
	static
	{
		EDhApiRenderApi renderingApi = Config.Client.Advanced.Graphics.Experimental.renderingApi.get();
		if (renderingApi == EDhApiRenderApi.AUTO)
		{
			IVersionConstants versionConstants = SingletonInjector.INSTANCE.get(IVersionConstants.class);
			renderingApi = versionConstants.getDefaultRenderingApi();
		}
		
		boolean register = (renderingApi == EDhApiRenderApi.BLAZE_3D);
		if (register)
		{
			LOGGER.debug("Attempting to register ["+VertexFormatElement.class.getSimpleName()+"]...");
			
			try
			{
				SCREEN_POS = VertexFormatElement.register(/*id*/22, /*index*/0, VertexFormatElement.Type.FLOAT, VertexFormatElement.Usage.POSITION, /*count*/ 2);
				RGBA_FLOAT_COLOR = VertexFormatElement.register(/*id*/23, /*index*/0, VertexFormatElement.Type.FLOAT, VertexFormatElement.Usage.COLOR, /*count*/ 4);
				
				SHORT_XYZ_POS = VertexFormatElement.register(/*id*/24, /*index*/0, VertexFormatElement.Type.USHORT, VertexFormatElement.Usage.POSITION, /*count*/ 3);
				BYTE_PAD = VertexFormatElement.register(/*id*/25, /*index*/0, VertexFormatElement.Type.BYTE, VertexFormatElement.Usage.GENERIC, /*count*/ 1);
				
				META = VertexFormatElement.register(/*id*/26, /*index*/0, VertexFormatElement.Type.USHORT, VertexFormatElement.Usage.GENERIC, /*count*/ 1);
				RGBA_UBYTE_COLOR = VertexFormatElement.register(/*id*/27, /*index*/0, VertexFormatElement.Type.UBYTE, VertexFormatElement.Usage.COLOR, /*count*/ 4);
				IRIS_MATERIAL = VertexFormatElement.register(/*id*/28, /*index*/0, VertexFormatElement.Type.BYTE, VertexFormatElement.Usage.GENERIC, /*count*/ 1);
				IRIS_NORMAL = VertexFormatElement.register(/*id*/29, /*index*/0, VertexFormatElement.Type.BYTE, VertexFormatElement.Usage.GENERIC, /*count*/ 1);
				
				FLOAT_XYZ_POS = VertexFormatElement.register(/*id*/30, /*index*/0, VertexFormatElement.Type.FLOAT, VertexFormatElement.Usage.POSITION, /*count*/ 3);
			}
			catch (Exception e)
			{
				String message = "Unable to register one or more ["+VertexFormatElement.class.getSimpleName()+"] this is likely caused by another mod registering their own custom ["+VertexFormatElement.class.getSimpleName()+"]'s. This should be fixed in the next major Minecraft version.";
				
				IMinecraftClientWrapper mc = SingletonInjector.INSTANCE.get(IMinecraftClientWrapper.class);
				mc.crashMinecraft(message, new Exception(message, e));
				
				// here to make the compiler happy, the process should shut down before this
				throw new RuntimeException(e);
			}
			
			LOGGER.debug("Successfully registered ["+VertexFormatElement.class.getSimpleName()+"].");
		}
		else
		{
			SCREEN_POS = null;
			RGBA_FLOAT_COLOR = null;
			
			SHORT_XYZ_POS = null;
			BYTE_PAD = null;
			
			META = null;
			RGBA_UBYTE_COLOR = null;
			IRIS_MATERIAL = null;
			IRIS_NORMAL = null;
			
			FLOAT_XYZ_POS = null;
		}
	}
	
	
	
}
#endif