package com.seibel.distanthorizons.common.render.blaze.wrappers.uniform;

#if MC_VER <= MC_1_21_10
public class BlazeUniformBufferWrapper {}

#else

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderSystem;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.IUniformBufferWrapper;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class BlazeUniformBufferWrapper implements IUniformBufferWrapper
{
	private static final DhLogger LOGGER = new DhLoggerBuilder().build();
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	
	private final String name;
	
	private int cpuBufferSize = 0;
	private int gpuBufferSize = 0;
	
	private ByteBuffer cpuBuffer = null;
	public GpuBuffer gpuBuffer = null;
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	public BlazeUniformBufferWrapper(String name) { this.name = name; }
	
	//endregion
	
	
	
	//========//
	// render //
	//========//
	//region
	
	protected ByteBuffer getOrCreateBuffer(int size)
	{
		if (this.cpuBuffer == null 
			|| this.cpuBufferSize != size)
		{
			this.cpuBuffer = ByteBuffer.allocateDirect(size);
			this.cpuBuffer.order(ByteOrder.nativeOrder());
			
			this.cpuBufferSize = size;
		}
		
		return this.cpuBuffer;
	}
	
	@Override
	public void upload() throws IllegalStateException
	{
		if (this.cpuBuffer == null)
		{
			throw new IllegalStateException("Upload called before buffer was created");
		}
		
		if (this.gpuBuffer == null
			|| this.gpuBufferSize != this.cpuBufferSize)
		{
			if (this.gpuBuffer != null)
			{
				this.gpuBuffer.close();
			}
			
			int usage = GpuBuffer.USAGE_COPY_DST
				| GpuBuffer.USAGE_VERTEX
				| GpuBuffer.USAGE_UNIFORM;
			this.gpuBuffer = GPU_DEVICE.createBuffer(this::getBufferName, usage, this.cpuBufferSize);
			
			this.gpuBufferSize = this.cpuBufferSize;
		}
		
		
		
		int byteSize = (this.cpuBuffer.limit() - this.cpuBuffer.position());
		GpuBufferSlice bufferSlice = new GpuBufferSlice(this.gpuBuffer, /*offset*/0, byteSize);
		if (!bufferSlice.buffer().isClosed())
		{
			COMMAND_ENCODER.writeToBuffer(bufferSlice, this.cpuBuffer);
		}
		else
		{
			LOGGER.warn("Uploading to buffer ["+this.name+"] failed due to already being closed");
		}
	}
	private String getBufferName() { return this.name; }
	
	//endregion
	
	
	
	//================//
	// base overrides //
	//================//
	//region
	
	@Override
	public void close()
	{
		if (this.gpuBuffer != null)
		{
			this.gpuBuffer.close();
		}
	}
	
	//endregion
	
	
	
}
#endif