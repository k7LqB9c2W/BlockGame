cbuffer BuildParams : register(b0)
{
    uint gSrcMip;
    uint gSrcWidth;
    uint gSrcHeight;
    uint gDstWidth;
    uint gDstHeight;
};

Texture2D<float> gSource : register(t0);
RWTexture2D<float> gDestination : register(u0);

[numthreads(8, 8, 1)]
void DepthPyramidMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 dstCoord = dispatchThreadId.xy;
    if (dstCoord.x >= gDstWidth || dstCoord.y >= gDstHeight)
    {
        return;
    }

    if (gSrcWidth == gDstWidth && gSrcHeight == gDstHeight)
    {
        gDestination[dstCoord] = gSource.Load(int3(dstCoord, gSrcMip));
        return;
    }

    const uint2 srcBase = dstCoord * 2u;
    const uint maxSrcX = (gSrcWidth > 0u) ? (gSrcWidth - 1u) : 0u;
    const uint maxSrcY = (gSrcHeight > 0u) ? (gSrcHeight - 1u) : 0u;

    const float depth0 = gSource.Load(int3(min(srcBase.x, maxSrcX), min(srcBase.y, maxSrcY), gSrcMip));
    const float depth1 = gSource.Load(int3(min(srcBase.x + 1u, maxSrcX), min(srcBase.y, maxSrcY), gSrcMip));
    const float depth2 = gSource.Load(int3(min(srcBase.x, maxSrcX), min(srcBase.y + 1u, maxSrcY), gSrcMip));
    const float depth3 = gSource.Load(int3(min(srcBase.x + 1u, maxSrcX), min(srcBase.y + 1u, maxSrcY), gSrcMip));

    gDestination[dstCoord] = max(max(depth0, depth1), max(depth2, depth3));
}
