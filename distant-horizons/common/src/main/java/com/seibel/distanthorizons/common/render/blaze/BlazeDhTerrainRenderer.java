package com.seibel.distanthorizons.common.render.blaze;

#if MC_VER <= MC_1_21_10
public class BlazeDhTerrainRenderer {}

#else

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.buffers.Std140Builder;
import com.mojang.blaze3d.buffers.Std140SizeCalculator;
import com.mojang.blaze3d.pipeline.BlendFunction;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.platform.DepthTestFunction;
import com.mojang.blaze3d.platform.PolygonMode;
import com.mojang.blaze3d.shaders.UniformType;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.VertexFormat;
import com.seibel.distanthorizons.api.methods.events.abstractEvents.DhApiBeforeBufferRenderEvent;
import com.seibel.distanthorizons.common.render.blaze.util.BlazeDhVertexFormatUtil;
import com.seibel.distanthorizons.common.render.blaze.util.BlazeUniformUtil;
import com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureViewWrapper;
import com.seibel.distanthorizons.common.render.blaze.wrappers.uniform.BlazeLodUniformBufferWrapper;
import com.seibel.distanthorizons.common.render.blaze.wrappers.buffer.BlazeVertexBufferWrapper;
import com.seibel.distanthorizons.common.wrappers.misc.LightMapWrapper;
import com.seibel.distanthorizons.core.config.Config;
import com.seibel.distanthorizons.core.dataObjects.render.bufferBuilding.LodBufferContainer;
import com.seibel.distanthorizons.core.dataObjects.render.bufferBuilding.LodQuadBuilder;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.pos.DhSectionPos;
import com.seibel.distanthorizons.common.render.openGl.glObject.enums.GLEnums;
import com.seibel.distanthorizons.common.render.openGl.glObject.buffer.GlQuadElementBuffer;
import com.seibel.distanthorizons.core.render.RenderParams;
import com.seibel.distanthorizons.core.util.RenderUtil;
import com.seibel.distanthorizons.core.util.math.Mat4f;
import com.seibel.distanthorizons.core.util.math.Vec3d;
import com.seibel.distanthorizons.core.util.math.Vec3f;
import com.seibel.distanthorizons.core.util.objects.SortedArraySet;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IProfilerWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.IDhTerrainRenderer;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.IVertexBufferWrapper;
import com.seibel.distanthorizons.coreapi.DependencyInjection.ApiEventInjector;
import net.minecraft.resources.Identifier;
import org.lwjgl.opengl.GL32;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/** Renders rendering DH's LOD terrain. */
public class BlazeDhTerrainRenderer implements IDhTerrainRenderer
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build();
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	public static final BlazeDhTerrainRenderer INSTANCE = new BlazeDhTerrainRenderer();
	
	
	private RenderPipeline opaquePipeline;
	private RenderPipeline transparentPipeline;
	private boolean init = false;
	
	private GpuBuffer indexBuffer;
	
	private GpuBuffer fragUniformBuffer;
	private GpuBuffer vertSharedUniformBuffer;
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	private BlazeDhTerrainRenderer() { }
	
	private void tryInit()
	{
		if (this.init)
		{
			return;
		}
		this.init = true; // todo only set when succeeded (in case of exception)
		
		
		VertexFormat vertexFormat = VertexFormat.builder()
			.add("vPosition", BlazeDhVertexFormatUtil.SHORT_XYZ_POS)
			.add("meta", BlazeDhVertexFormatUtil.META)
			.add("vColor", BlazeDhVertexFormatUtil.RGBA_UBYTE_COLOR)
			.add("irisMaterial", BlazeDhVertexFormatUtil.IRIS_MATERIAL)
			.add("irisNormal", BlazeDhVertexFormatUtil.IRIS_NORMAL)
			.add("paddingTwo", BlazeDhVertexFormatUtil.BYTE_PAD)
			.add("paddingThree", BlazeDhVertexFormatUtil.BYTE_PAD) // padding is to make sure the format is a multiple of 4
			.build();
		
		RenderPipeline.Builder pipelineBuilder = RenderPipeline.builder();
		{
			pipelineBuilder.withCull(true);
			pipelineBuilder.withDepthWrite(true);
			pipelineBuilder.withDepthTestFunction(DepthTestFunction.LESS_DEPTH_TEST);
			pipelineBuilder.withColorWrite(true);
			pipelineBuilder.withPolygonMode(PolygonMode.FILL);
			pipelineBuilder.withLocation(Identifier.parse("distanthorizons:lod_render"));
			
			pipelineBuilder.withVertexShader(Identifier.fromNamespaceAndPath("distanthorizons", "lod/blaze/vert"));
			pipelineBuilder.withFragmentShader(Identifier.fromNamespaceAndPath("distanthorizons", "lod/blaze/frag"));
			
			pipelineBuilder.withSampler("uLightMap");
			
			pipelineBuilder.withUniform("vertUniqueUniformBlock", UniformType.UNIFORM_BUFFER);
			pipelineBuilder.withUniform("vertSharedUniformBlock", UniformType.UNIFORM_BUFFER);
			pipelineBuilder.withUniform("fragUniformBlock", UniformType.UNIFORM_BUFFER);
			
			pipelineBuilder.withVertexFormat(vertexFormat, VertexFormat.Mode.TRIANGLES);
		}
		
		// opaque
		{
			pipelineBuilder.withoutBlend();
			this.opaquePipeline = pipelineBuilder.build();
		}
		
		// transparent
		{
			pipelineBuilder.withBlend(BlendFunction.TRANSLUCENT);
			this.transparentPipeline = pipelineBuilder.build();
		}
	}
	
	//endregion
	
	
	
	//========//
	// render //
	//========//
	//region
	
	@Override
	public void render(
		RenderParams renderEventParam, 
		boolean opaquePass,
		SortedArraySet<LodBufferContainer> bufferContainers,
		IProfilerWrapper profiler)
	{
		this.tryInit();
		
		
		profiler.push("vert unique uniforms");
		{
			// create data //
			
			for (int lodIndex = 0; lodIndex < bufferContainers.size(); lodIndex++)
			{
				LodBufferContainer bufferContainer = bufferContainers.get(lodIndex);
				bufferContainer.uniformContainer.tryUpload();
			}
		}
		
		profiler.popPush("vert share uniforms");
		{
			Mat4f combinedMatrix = new Mat4f(renderEventParam.dhProjectionMatrix);
			combinedMatrix.multiply(renderEventParam.dhModelViewMatrix);
			
			float earthCurveRatio = Config.Client.Advanced.Graphics.Experimental.earthCurveRatio.get();
			if (earthCurveRatio < -1.0f || earthCurveRatio > 1.0f)
			{
				earthCurveRatio = /*6371KM*/ 6371000.0f / earthCurveRatio;
			}
			else
			{
				// disable curvature if the config value is between -1 and 1
				earthCurveRatio = 0.0f;
			}
			
			
			// upload data //
			
			int uniformBufferSize = new Std140SizeCalculator()
				.putInt() // uIsWhiteWorld
				
				.putFloat() // uWorldYOffset
				.putFloat() // uMircoOffset
				.putFloat() // uEarthRadius
				
				.putVec3() // uCameraPos
				.putMat4f() // uCombinedMatrix
				.get();
			
			ByteBuffer buffer = ByteBuffer.allocateDirect(uniformBufferSize);
			buffer.order(ByteOrder.nativeOrder());
			Std140Builder.intoBuffer(buffer)
				.putInt(0) // uIsWhiteWorld
				
				.putFloat((float) renderEventParam.worldYOffset) // uWorldYOffset
				.putFloat(0.01f) // uMircoOffset // 0.01 block offset
				.putFloat(earthCurveRatio) // uEarthRadius
				
				.putVec3(
					(float)renderEventParam.exactCameraPosition.x,
					(float)renderEventParam.exactCameraPosition.y,
					(float)renderEventParam.exactCameraPosition.z) // uCameraPos
				.putMat4f(combinedMatrix.createJomlMatrix()) // uCombinedMatrix
				.get();
			
			this.vertSharedUniformBuffer = BlazeUniformUtil.createBuffer("vertSharedUniformBlock", uniformBufferSize, this.vertSharedUniformBuffer);
			GpuBufferSlice bufferSlice = new GpuBufferSlice(this.vertSharedUniformBuffer, 0, uniformBufferSize);
			
			COMMAND_ENCODER.writeToBuffer(bufferSlice, buffer);
		}
		
		profiler.popPush("set frag uniforms");
		{
			int uniformBufferSize = new Std140SizeCalculator()
				.putFloat() // uClipDistance
				.putFloat() // uNoiseIntensity
				.putInt() // uNoiseSteps
				.putInt() // uNoiseDropoff
				.putInt() // uDitherDhRendering
				.putInt() // uNoiseEnabled
				.get();
			
			
			// create data //
			
			float dhNearClipDistance = RenderUtil.getNearClipPlaneInBlocks();
			if (!Config.Client.Advanced.Debugging.lodOnlyMode.get())
			{
				// this added value prevents the near clip plane and discard circle from touching, which looks bad
				dhNearClipDistance += 16f;
			}
			
			
			// upload data //
			
			ByteBuffer buffer = ByteBuffer.allocateDirect(uniformBufferSize);
			buffer.order(ByteOrder.nativeOrder());
			buffer = Std140Builder.intoBuffer(buffer)
				.putFloat(dhNearClipDistance) // uClipDistance
				.putFloat(Config.Client.Advanced.Graphics.NoiseTexture.noiseIntensity.get()) // uNoiseIntensity
				.putInt(Config.Client.Advanced.Graphics.NoiseTexture.noiseSteps.get()) // uNoiseSteps
				.putInt(Config.Client.Advanced.Graphics.NoiseTexture.noiseDropoff.get()) // uNoiseDropoff
				.putInt(Config.Client.Advanced.Graphics.Quality.ditherDhFade.get() ? 1 : 0) // uDitherDhRendering
				.putInt(Config.Client.Advanced.Graphics.NoiseTexture.enableNoiseTexture.get() ? 1 : 0) // uNoiseEnabled
				.get()
			;
			
			this.fragUniformBuffer = BlazeUniformUtil.createBuffer("fragUniformBlock", uniformBufferSize, this.fragUniformBuffer);
			GpuBufferSlice bufferSlice = new GpuBufferSlice(this.fragUniformBuffer, 0, uniformBufferSize);
			
			COMMAND_ENCODER.writeToBuffer(bufferSlice, buffer);
		}
		
		// create index buffer
		{
			if (this.indexBuffer == null)
			{
				ByteBuffer buffer = MemoryUtil.memAlloc(LodQuadBuilder.getMaxBufferByteSize() * GLEnums.getTypeSize(GL32.GL_UNSIGNED_INT) * 6);
				GlQuadElementBuffer.buildBuffer(LodQuadBuilder.getMaxBufferByteSize(), buffer, GL32.GL_UNSIGNED_INT);
				
				
				// create buffer if needed
				if (this.indexBuffer == null
					|| this.indexBuffer.size() < buffer.capacity())
				{
					int usage = GpuBuffer.USAGE_COPY_DST 
						| GpuBuffer.USAGE_VERTEX 
						| GpuBuffer.USAGE_INDEX 
						| GpuBuffer.USAGE_UNIFORM;
					this.indexBuffer = GPU_DEVICE.createBuffer(this::getIndexBufferName, usage, buffer.capacity());
				}
				
				GpuBufferSlice bufferSlice = new GpuBufferSlice(this.indexBuffer, /*offset*/ 0, buffer.capacity());
				COMMAND_ENCODER.writeToBuffer(bufferSlice, buffer);
			}
		}
		
		
		
		// render pass setup
		{
			profiler.popPush("setup");
			
			// create a render pass
			OptionalInt optionalClearColorAsInt = OptionalInt.empty();
			OptionalDouble optionalDepthValueAsDouble = OptionalDouble.empty();
			
			try(RenderPass renderPass = COMMAND_ENCODER.createRenderPass(
				this::getRenderPassName,
				BlazeDhMetaRenderer.INSTANCE.dhColorTextureWrapper.textureView,
				optionalClearColorAsInt,
				BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.textureView, optionalDepthValueAsDouble)
				)
			{
				// bind MC Lightmap
				//renderPass.bindTexture("uLightMap", this.mcLightTextureViewWrapper.textureView, this.mcLightTextureViewWrapper.textureSampler);
				LightMapWrapper lightMapWrapper = (LightMapWrapper) renderEventParam.lightmap;
				BlazeTextureViewWrapper lightmapTextureViewWrapper = lightMapWrapper.getTextureViewWrapper();
				renderPass.bindTexture("uLightMap", lightmapTextureViewWrapper.textureView, lightmapTextureViewWrapper.textureSampler);
				
				// set pipeline
				renderPass.setPipeline(opaquePass ? this.opaquePipeline : this.transparentPipeline);
				renderPass.setIndexBuffer(this.indexBuffer, VertexFormat.IndexType.INT);
				
				// shared uniforms
				renderPass.setUniform("fragUniformBlock", this.fragUniformBuffer);
				renderPass.setUniform("vertSharedUniformBlock", this.vertSharedUniformBuffer);
				
				
				
				for (int lodIndex = 0; lodIndex < bufferContainers.size(); lodIndex++)
				{
					profiler.popPush("binding");
					
					LodBufferContainer bufferContainer = bufferContainers.get(lodIndex);
					BlazeLodUniformBufferWrapper uniformWrapper = (BlazeLodUniformBufferWrapper)bufferContainer.uniformContainer;
					
					boolean columnBuilderDebugEnabled = Config.Client.Advanced.Debugging.ColumnBuilderDebugging.columnBuilderDebugEnable.get();
					if (columnBuilderDebugEnabled)
					{
						if (DhSectionPos.getDetailLevel(bufferContainer.pos) == Config.Client.Advanced.Debugging.ColumnBuilderDebugging.columnBuilderDebugDetailLevel.get()
							&& DhSectionPos.getX(bufferContainer.pos) == Config.Client.Advanced.Debugging.ColumnBuilderDebugging.columnBuilderDebugXPos.get()
							&& DhSectionPos.getZ(bufferContainer.pos) == Config.Client.Advanced.Debugging.ColumnBuilderDebugging.columnBuilderDebugZPos.get())
						{
							int breakpoint = 0;
						}
						else
						{
							continue;
						}
					}
					
					renderPass.setUniform("vertUniqueUniformBlock", uniformWrapper.gpuBuffer);
					
					
					
					profiler.popPush("rendering");
					
					// render each buffer
					IVertexBufferWrapper[] bufferWrapperList = opaquePass ? bufferContainer.vbos : bufferContainer.vbosTransparent;
					for (int i = 0; i < bufferWrapperList.length; i++)
					{
						BlazeVertexBufferWrapper bufferWrapper = (BlazeVertexBufferWrapper) bufferWrapperList[i];
						if (!bufferWrapper.uploaded
							|| bufferWrapper.vertexCount == 0)
						{
							continue;
						}
						
						// fire render event
						{
							Vec3d camPos = renderEventParam.exactCameraPosition;
							Vec3f modelPos = new Vec3f(
								(float) (bufferContainer.minCornerBlockPos.getX() - camPos.x),
								(float) (bufferContainer.minCornerBlockPos.getY() - camPos.y),
								(float) (bufferContainer.minCornerBlockPos.getZ() - camPos.z));
							ApiEventInjector.INSTANCE.fireAllEvents(DhApiBeforeBufferRenderEvent.class, new DhApiBeforeBufferRenderEvent.EventParam(renderEventParam, modelPos));
						}
						
						renderPass.setVertexBuffer(0, bufferWrapper.vboGpuBuffer); // vertex buffer can only be "0" lol
						
						if (!bufferWrapper.vboGpuBuffer.isClosed())
						{
							renderPass.drawIndexed(
								/*indexStart*/ 0,
								/*firstIndex*/0,
								/*indexCount*/bufferWrapper.indexCount,
								/*instanceCount*/1);
						}
					}
				}
				
			}
		}
		
		profiler.pop();
	}
	private String getIndexBufferName() { return "distantHorizons:LodIndexBuffer"; }
	private String getRenderPassName() { return "distantHorizons:McLodRenderer"; }
	
	//endregion
	
	
	
}
#endif