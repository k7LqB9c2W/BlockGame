/*
 *    This file is part of the Distant Horizons mod
 *    licensed under the GNU LGPL v3 License.
 *
 *    Copyright (C) 2020 James Seibel
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, version 3.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package com.seibel.distanthorizons.common.render.blaze.test;

#if MC_VER <= MC_1_21_10
public class BlazeDhTestTriangleRenderer {}

#else
	
import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.platform.DepthTestFunction;
import com.mojang.blaze3d.platform.PolygonMode;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.*;
import com.mojang.blaze3d.vertex.VertexFormat;
import com.seibel.distanthorizons.common.render.blaze.util.BlazeDhVertexFormatUtil;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.render.RenderParams;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.IDhTestTriangleRenderer;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.Identifier;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Renders the OpenGL/Vulkan triangle
 * to the center of the screen to confirm DH's
 * apply shader is running correctly
 */
public class BlazeDhTestTriangleRenderer implements IDhTestTriangleRenderer
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build();
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	public static final BlazeDhTestTriangleRenderer INSTANCE = new BlazeDhTestTriangleRenderer();
	
	private RenderPipeline pipeline;
	private boolean init = false;
	
	private GpuTextureView mcColorTextureView;
	
	private GpuBuffer vboGpuBuffer;
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	private BlazeDhTestTriangleRenderer() { }
	
	private void tryInit()
	{
		if (this.init)
		{
			return;
		}
		this.init = true;
		
		
		
		VertexFormat vertexFormat = VertexFormat.builder()
			.add("vPosition", BlazeDhVertexFormatUtil.SCREEN_POS)
			.add("vColor", BlazeDhVertexFormatUtil.RGBA_FLOAT_COLOR)
			.build();
		
		//int breakpointOne = 0;
		// needs to manually be set if the VertexFormatElement isn't registered
		//this.vertexFormat.getOffsetsByElement()[this.posForm.id()] = 0;
		//this.vertexFormat.getOffsetsByElement()[this.colForm.id()] = Float.BYTES * 2;
		//
		//int breakpointTwo = 0;
		
		
		RenderPipeline.Builder pipelineBuilder = RenderPipeline.builder();
		{
			pipelineBuilder.withCull(false);
			pipelineBuilder.withDepthWrite(false);
			pipelineBuilder.withDepthTestFunction(DepthTestFunction.NO_DEPTH_TEST);
			pipelineBuilder.withColorWrite(true);
			pipelineBuilder.withoutBlend();
			pipelineBuilder.withPolygonMode(PolygonMode.FILL);
			pipelineBuilder.withLocation(Identifier.parse("distanthorizons:test_render"));
			
			pipelineBuilder.withVertexShader(Identifier.fromNamespaceAndPath("distanthorizons", "test/blaze/vert"));
			pipelineBuilder.withFragmentShader(Identifier.fromNamespaceAndPath("distanthorizons", "test/blaze/frag"));
			
			pipelineBuilder.withVertexFormat(vertexFormat, VertexFormat.Mode.TRIANGLES);
		}
		this.pipeline = pipelineBuilder.build();
		
		
		this.mcColorTextureView = GPU_DEVICE.createTextureView(Minecraft.getInstance().getMainRenderTarget().getColorTexture());
		
		
		this.uploadVertexData();
	}
	private void uploadVertexData()
	{
		// vertices for the OpenGL/Vulkan Triangle
		float[] vertices = new float[]
			{
				// PosX,Y,    ColorR,G,B,A
				-0.5f, -0.5f,   1.0f, 0.0f, 0.0f, 1.0f,
				0.5f, -0.5f,   0.0f, 1.0f, 0.0f, 1.0f,
				0.0f,  0.5f,   0.0f, 0.0f, 1.0f, 1.0f,
			};
		
		
		int usage = GpuBuffer.USAGE_COPY_DST
			| GpuBuffer.USAGE_VERTEX;
		int size = vertices.length * Float.BYTES;
		this.vboGpuBuffer = GPU_DEVICE.createBuffer(this::getRenderPassName, usage, size);
		
		{
			int offset = 0;
			int length = vertices.length * Float.BYTES;
			GpuBufferSlice bufferSlice = new GpuBufferSlice(this.vboGpuBuffer, offset, length);
			
			ByteBuffer byteBuffer = ByteBuffer.allocateDirect(vertices.length * Float.BYTES);
			// Fill buffer with vertices.
			byteBuffer.order(ByteOrder.nativeOrder());
			byteBuffer.asFloatBuffer().put(vertices);
			byteBuffer.rewind();
			
			COMMAND_ENCODER.writeToBuffer(bufferSlice, byteBuffer);
		}
	}
	
	//endregion
	
	
	
	//========//
	// render //
	//========//
	//region
	
	@Override
	public void render(RenderParams renderParams)
	{
		this.tryInit();
		
		try (RenderPass renderPass = COMMAND_ENCODER.createRenderPass(
			this::getRenderPassName,
			this.mcColorTextureView,
			/*optionalClearColorAsInt*/ OptionalInt.empty(),
			/*mcDepthTextureView*/ null, 
			/*optionalDepthValueAsDouble*/ OptionalDouble.empty()))
		{
			renderPass.setVertexBuffer(0, this.vboGpuBuffer);
			renderPass.setPipeline(this.pipeline);
			renderPass.draw(/*indexStart*/ 0, /*indexCount*/ 3);
		}
	}
	private String getRenderPassName() { return "distantHorizons:DhTestRenderer"; }
	
	//endregion
	
	
	
}
#endif