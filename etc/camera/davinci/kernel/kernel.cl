#pragma OPENCL EXTENSION cl_arm_printf : enable

/*******************************************************************
 * HAAR & Inverse HAAR
 *******************************************************************/
__kernel void ApplyHaar(
    __global unsigned char *pSrc, 
    __global unsigned char *pDstLL,
    __global unsigned char *pDstLH,
    __global unsigned char *pDstHL,
    __global unsigned char *pDstHH,
    int width, 
    int height, 
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    __global unsigned char* pSrcL  = pSrc  + y * ( stride << 1 );
    __global unsigned char* pSrcL2 = pSrcL + stride;
   
	uint src_offset = x << 4;
    uchar16 a = vload16(0, pSrcL  + src_offset);
    uchar16 b = vload16(0, pSrcL2 + src_offset);

    short8 A = convert_short8( a.even );
    short8 B = convert_short8( a.odd  );
    short8 C = convert_short8( b.even );
    short8 D = convert_short8( b.odd  );

	uint dst_offset = y * ( stride >> 1 ) + ( x << 3 );
    uchar8 d;
    short8 T;
	
    // pDstLL
    d = convert_uchar8( ( A + B + C + D + (short8)(2) ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstLL + dst_offset);
    }
    else
    {
        vstore4(d.lo, 0, pDstLL + dst_offset);
    }
    
    // pDstLH
    // If T > 0 then you need to add 2 to round to the nearest integer
    // If T < 0 you only need to add 1 to round to the nearest integer
    // The "<" operator returns 0 or 1 for scalars but 0 or -1 for vector types
    T = ( B + D ) - ( A + C );
    d = convert_uchar8_sat( ( (short8)514 + ( T < (short8)0 ) + T ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstLH + dst_offset);
    }
    else
    {
        vstore4(d.lo, 0, pDstLH + dst_offset);
    }
    
    // pDstHL
    T = ( C + D ) - ( A + B );
    d = convert_uchar8_sat( ( (short8)514 + ( T < (short8)0 ) + T ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstHL + dst_offset);
    }
    else
    {
        vstore4(d.lo, 0, pDstHL + dst_offset);
    }
    
    // pDstHH
    T = ( A + D ) - ( C + B );
    d = convert_uchar8_sat( ( (short8)514 + ( T < (short8)0 ) + T ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstHH + dst_offset);
    }
    else
    {
        vstore4(d.lo, 0, pDstHH + dst_offset);
    }
}

__kernel void ApplyHaar_UV(
    __global unsigned char *pSrc, 
    __global unsigned char *pDstLL,
    __global unsigned char *pDstLH,
    __global unsigned char *pDstHL,
    int width, 
    int height, 
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    __global unsigned char* pSrcL  = pSrc + y * ( stride << 1 );
    __global unsigned char* pSrcL2 = pSrcL + stride;
	
    uint src_offset = x << 4;
    uchar16 a = vload16(0, pSrcL + src_offset);
    uchar16 b = vload16(0, pSrcL2 + src_offset);
    
    short8 A = convert_short8( a.even );
    short8 B = convert_short8( a.odd  );
    short8 C = convert_short8( b.even );
    short8 D = convert_short8( b.odd  );
	
	uint dst_offset = y * ( stride >> 1 ) + ( x << 3 );
    uchar8 d;
    short8 T;
    
    // pDstLL
    d = convert_uchar8( ( A + B + C + D + (short8)(2) ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstLL + dst_offset);
    }
    else if( (src_offset + 12) <= width )
    {
        vstore4(d.s0123, 0, pDstLL + dst_offset);
        vstore2(d.s45, 0, pDstLL + dst_offset + 4);
    }
    else if( (src_offset + 8) <= width )
    {
        vstore4(d.s0123, 0, pDstLL + dst_offset);
    }
    else
    {
        vstore2(d.s01, 0, pDstLL + dst_offset);
    }
    
    // pDstLH
    T = ( B + D ) - ( A + C );
    d = convert_uchar8_sat( ( (short8)514 + ( T < (short8)0 ) + T ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstLH + dst_offset);
    }
    else if( (src_offset + 12) <= width )
    {
        vstore4(d.s0123, 0, pDstLH + dst_offset);
        vstore2(d.s45, 0, pDstLH + dst_offset + 4);
    }
    else if( (src_offset + 8) <= width )
    {
        vstore4(d.s0123, 0, pDstLH + dst_offset);
    }
    else
    {
        vstore2(d.s01, 0, pDstLH + dst_offset);
    }
    
    // pDstHL
    T = ( C + D ) - ( A + B );
    d = convert_uchar8_sat( ( (short8)514 + ( T < (short8)0 ) + T ) >> 2 );
    if( (src_offset + 16) <= width )
    {
        vstore8(d, 0, pDstHL + dst_offset);
    }
    else if( (src_offset + 12) <= width )
    {
        vstore4(d.s0123, 0, pDstHL + dst_offset);
        vstore2(d.s45, 0, pDstHL + dst_offset + 4);
    }
    else if( (src_offset + 8) <= width )
    {
        vstore4(d.s0123, 0, pDstHL + dst_offset);
    }
    else
    {
        vstore2(d.s01, 0, pDstHL + dst_offset);
    }
}

__kernel void ApplyInverseHaar_new_v(
    __global unsigned char *pSrcLL,
    __global unsigned char *pSrcLH,
    __global unsigned char *pSrcHL,
    __global unsigned char *pSrcHH,
    __global unsigned char *pDst,
    int width,
    int height,
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
	uint src_offset = y * ( stride >> 1 ) + ( x << 3 );
    short8 temp0 = convert_short8( vload8(0, pSrcLL + src_offset) );
    short8 temp1 = convert_short8( vload8(0, pSrcLH + src_offset) );
    short8 temp2 = convert_short8( vload8(0, pSrcHL + src_offset) );
    short8 temp3 = convert_short8( vload8(0, pSrcHH + src_offset) );
    
    short8 pix0 = temp0 - temp1 - temp2 + temp3 + (short8)(128);
    short8 pix1 = temp0 + temp1 - temp2 - temp3 + (short8)(128);
    short8 pix2 = temp0 - temp1 + temp2 - temp3 + (short8)(128);
    short8 pix3 = temp0 + temp1 + temp2 + temp3 - (short8)(384);

    __global unsigned char *pDstLine0 = pDst + y * ( stride << 1 );
    __global unsigned char *pDstLine1 = pDstLine0 + stride;
    uint dst_offset = x << 4;
	
    if( (dst_offset + 16) <= width )
    {
        ushort16 mask16 = (ushort16)(0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15);
        uchar16 tmp0 = convert_uchar16( shuffle2( clamp(pix0, (short8)(0), (short8)(255)), clamp(pix1, (short8)(0), (short8)(255)), mask16 ) );
        uchar16 tmp1 = convert_uchar16( shuffle2( clamp(pix2, (short8)(0), (short8)(255)), clamp(pix3, (short8)(0), (short8)(255)), mask16 ) );
        vstore16(tmp0, 0, pDstLine0 + dst_offset);
        vstore16(tmp1, 0, pDstLine1 + dst_offset);
    }
    else
    {
        ushort8 mask8 = (ushort8)(0,4,1,5,2,6,3,7);
        uchar8 tmp0 = convert_uchar8( shuffle2( clamp(pix0.s0123, (short4)(0), (short4)(255)), clamp(pix1.s0123, (short4)(0), (short4)(255)), mask8 ) );
        uchar8 tmp1 = convert_uchar8( shuffle2( clamp(pix2.s0123, (short4)(0), (short4)(255)), clamp(pix3.s0123, (short4)(0), (short4)(255)), mask8 ) );
        vstore8(tmp0.s01234567, 0, pDstLine0 + dst_offset);
        vstore8(tmp1.s01234567, 0, pDstLine1 + dst_offset);
    }
}

__kernel void ApplyInverseHaar_new_v_UV(
    __global unsigned char *pSrcLL,
    __global unsigned char *pSrcLH,
    __global unsigned char *pSrcHL,
    __global unsigned char *pDst,
    int width,
    int height,
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    uint src_offset = y * (stride >> 1) + (x << 3);
    short8 temp0 = convert_short8( vload8(0, pSrcLL + src_offset) );
    short8 temp1 = convert_short8( vload8(0, pSrcLH + src_offset) );
    short8 temp2 = convert_short8( vload8(0, pSrcHL + src_offset) );
    short8 temp3 = (short8)128;
    
    short8 pix0 = temp0 - temp1 - temp2 + temp3 + (short8)(128);
    short8 pix1 = temp0 + temp1 - temp2 - temp3 + (short8)(128);
    short8 pix2 = temp0 - temp1 + temp2 - temp3 + (short8)(128);
    short8 pix3 = temp0 + temp1 + temp2 + temp3 - (short8)(384);
    
    ushort16 mask = (ushort16)(0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15);    
    uchar16 result0 = convert_uchar16( shuffle2( clamp( pix0, (short8)(0), (short8)(255) ), clamp( pix1, (short8)(0), (short8)(255) ), mask ) );
    uchar16 result1 = convert_uchar16( shuffle2( clamp( pix2, (short8)(0), (short8)(255) ), clamp( pix3, (short8)(0), (short8)(255) ), mask ) );

    __global unsigned char *pDstLine0 = pDst + y * (stride << 1);
    __global unsigned char *pDstLine1 = pDstLine0 + stride;
    uint dst_offset = x << 4;

    if( (dst_offset + 16) <= width )
    {
        vstore16(result0, 0, pDstLine0 + dst_offset);
        vstore16(result1, 0, pDstLine1 + dst_offset);
    }
    else if( (dst_offset + 12) <= width )
    {
        vstore8(result0.s01234567, 0, pDstLine0 + dst_offset);
        vstore4(result0.s89ab, 0, pDstLine0 + dst_offset + 8);

        vstore8(result1.s01234567, 0, pDstLine1 + dst_offset);
        vstore4(result1.s89ab, 0, pDstLine1 + dst_offset + 8);
    }
    else if( (dst_offset + 8) <= width )
    {
        vstore8(result0.s01234567, 0, pDstLine0 + dst_offset);
        vstore8(result1.s01234567, 0, pDstLine1 + dst_offset);
    }
    else
    {
        vstore4(result0.s0123, 0, pDstLine0 + dst_offset);
        vstore4(result1.s0123, 0, pDstLine1 + dst_offset);
    }
}

/*******************************************************************
 * UpScale & DownScale
 *******************************************************************/
__kernel void UpScale(
	__global unsigned char *pImageSrc,
	__global unsigned char *pImageDst,
	int nWidth,
	int nHeight,
	int dstStride)
{
	int x = get_global_id(0);
 	int y = get_global_id(1);
 
 	__global unsigned char *pDst1 = pImageDst + (y + 1) * (dstStride << 1) + (x << 4) + 1;
 	__global unsigned char *pDst2 = pImageDst + dstStride + y * (dstStride << 1) + (x << 4) + 1;
 	__global unsigned char  *pSrc = pImageSrc + y * nWidth + (x << 3);
 
  	ushort8 p00, p01, p10, p11;
  	ushort8 r0, r1;
	uchar16 mask = (uchar16)(0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15);

  	uchar16 tt;
  	tt = vload16(0, pSrc);
  	p00 = convert_ushort8(tt.s01234567);
  	p01 = convert_ushort8(tt.s12345678);
  	tt = vload16(0, pSrc + nWidth);
  	p10 = convert_ushort8(tt.s01234567);
  	p11 = convert_ushort8(tt.s12345678);

 	uchar8 c0, c1;
 	r0 = p00 * ((ushort8)(9)) + p01 * ((ushort8)(3)) + p10 * ((ushort8)(3)) + p11 + (ushort8)(8);
 	r1 = p00 * ((ushort8)(3)) + p01 * ((ushort8)(9)) + p10 + p11 * ((ushort8)(3)) + (ushort8)(8);

 	c0 = convert_uchar8(r0 >> 4);
 	c1 = convert_uchar8(r1 >> 4);
 	vstore16(shuffle2(c0, c1, mask), 0, pDst2);

    r1 = r1 - (p01 << 3) + (p10 << 3);
    r0 = r0 - (p00 << 3) + (p11 << 3);

    c0 = convert_uchar8(r1 >> 4);
 	c1 = convert_uchar8(r0 >> 4);
 	vstore16(shuffle2(c0, c1, mask), 0, pDst1);
}

__kernel void UpScale_FirstRow(
    __global unsigned char *pImageSrc,
    __global unsigned char *pImageDst,
    int nWidth,
    int nHeight,
    int dstStride)
{
    int x = get_global_id(0);
    int offset = ( x << 1 ) + 1;
    
    if( ( x != 0 ) && ( x != (nWidth - 2) ) )
    {
        pImageDst[offset] = (pImageSrc[x]*3 + pImageSrc[x+1] + 2) >> 2;
        pImageDst[offset+1] = (pImageSrc[x] + pImageSrc[x+1]*3 + 2) >> 2;
    }
    else if( x != 0 )
    {
        pImageDst[offset] = (pImageSrc[x]*3 + pImageSrc[x+1] + 2) >> 2;
        pImageDst[offset+1] = (pImageSrc[x] + pImageSrc[x+1]*3 + 2) >> 2;
        pImageDst[offset+2] = pImageSrc[x+1];
    }
    else
    {
        pImageDst[0] = pImageSrc[0];
        pImageDst[1] = (pImageSrc[0]*3 + pImageSrc[1] + 2) >> 2;
        pImageDst[2] = (pImageSrc[0] + pImageSrc[1]*3 + 2) >> 2;
    }
}

__kernel void UpScale_FirstLine(
    __global unsigned char *pImageSrc,
    __global unsigned char *pImageDst,
    int nWidth,
    int nHeight,
    int dstStride)
{
    int x = get_global_id(0);
    if ( x >= nHeight - 1 )
    {
        return;
    }
    
    pImageSrc += x * nWidth;
    pImageDst += ( x << 1) * dstStride + dstStride;
    
    pImageDst[0] = ( pImageSrc[0] * 3 + pImageSrc[nWidth] + 2 ) >> 2;
    pImageDst[dstStride] = ( pImageSrc[0] + pImageSrc[nWidth] * 3 + 2 ) >> 2;
}

__kernel void UpScale_RemainLine(
    __global unsigned char *pImageSrc,
    __global unsigned char *pImageDst,
    int nWidth,
    int nHeight,
    int dstStride,
    int vOffset)
{
    int x = get_global_id(0) + vOffset;
    int y = get_global_id(1);

    if( x >= nWidth || y >= nHeight - 1 )
    {
    	return;
    }
    
    int offset = ( x << 1 ) * dstStride;
    pImageDst += ( ( y << 1 ) + 1 ) * dstStride + ( x << 1 );
    pImageSrc += y * nWidth + x;
    
    if( x != nWidth - 1 )
    {
        pImageDst[1] = (pImageSrc[0]*9 + pImageSrc[1]*3 + pImageSrc[nWidth]*3 + pImageSrc[nWidth+1] + 8) >> 4;
        pImageDst[2] = (pImageSrc[0]*3 + pImageSrc[1]*9 + pImageSrc[nWidth] + pImageSrc[nWidth+1]*3 + 8) >> 4;
        pImageDst[dstStride+1] = (pImageSrc[0]*3 + pImageSrc[1] + pImageSrc[nWidth]*9 + pImageSrc[nWidth+1]*3 + 8) >> 4;
        pImageDst[dstStride+2] = (pImageSrc[0] + pImageSrc[1]*3 + pImageSrc[nWidth]*3 + pImageSrc[nWidth+1]*9 + 8) >> 4;
    }
    else
    {
        pImageDst[1] = (pImageSrc[0]*3 + pImageSrc[nWidth] + 2) >> 2;
        pImageDst[dstStride+1] = (pImageSrc[0] + pImageSrc[nWidth]*3 + 2) >> 2;
    }
}

__kernel void UpScale_LastRow(
    __global unsigned char *pImageSrc,
    __global unsigned char *pImageDst,
    int nWidth,
    int nHeight,
    int dstStride)
{
    int x = get_global_id(0);
    
    int offset = ( x << 1 ) + 1;
    pImageSrc += ( nHeight - 1 ) * nWidth;
    pImageDst += ( (nHeight << 1 ) - 1 ) * dstStride;
    
    if( x != 0 && x != nWidth-2 )
    {
        pImageDst[offset] = (pImageSrc[x]*3 + pImageSrc[x+1] + 2) >> 2;
        pImageDst[offset+1] = (pImageSrc[x] + pImageSrc[x+1]*3 + 2) >> 2;
    }
    else if( x != 0 )
    {
        pImageDst[offset] = (pImageSrc[x]*3 + pImageSrc[x+1] + 2) >> 2;
        pImageDst[offset+1] = (pImageSrc[x] + pImageSrc[x+1]*3 + 2) >> 2;
        pImageDst[offset+2] = pImageSrc[x+1];
    }
    else
    {
        pImageDst[0] = pImageSrc[0];
        pImageDst[1] = (pImageSrc[0]*3 + pImageSrc[1] + 2) >> 2;
        pImageDst[2] = (pImageSrc[0] + pImageSrc[1]*3 + 2) >> 2;
    }
}

__kernel void DownScale(
    __global unsigned char *pImageSrc,
    __global unsigned char *pImageDst,
    int nWidth,
    int nHeight,
    int srcStride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    pImageSrc += ( y << 1 ) * nWidth + ( x << 4 );
    pImageDst += y * ( nWidth >> 1 ) + ( x << 3 );
    short16 add1 = convert_short16( vload16(0, pImageSrc) ) + convert_short16( vload16(0, pImageSrc + nWidth) );
    uchar8 result = convert_uchar8( ( (short8)(add1.s02468ace) + (short8)(add1.s13579bdf) + (short8)(2) ) >> 2 );

    if( x < (nWidth >> 4) )
    {         
        vstore8(result, 0, pImageDst);
    }
    else // process rest pixels in each line which is not devided by 16
    {
        int num = ( nWidth - (x << 4) ) >> 1;   
        uchar res[8] = {result.s0, result.s1, result.s2, result.s3, result.s4, result.s5, result.s6, result.s7};
        for(int i = 0; i < num; i++)
        {
            pImageDst[i] = res[i];
        }
    }	
}

/*******************************************************************
 * Gaussian blur
 *******************************************************************/
__kernel void opt_GaussianBlur(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    int width,
    int height,
    int srcStride,
    int bufStride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    __global unsigned char *pSrcL = pSrc + y * srcStride + ( x << 4 );
    __global unsigned char *pDstL = pDst + (y + 1) * bufStride + ( x << 4 ) + 1;
    
    short16 m1_src16 = convert_short16( vload16(0, pSrcL    ) );
    short16      r16 = convert_short16( vload16(0, pSrcL + 2) );
    short16 m2_src16 = (short16)(m1_src16.s12345678, r16.s789abcde);
    short16 m3_src16 = (short16)(m1_src16.s23456789, r16.s89abcdef);
    short16     dst0 = (m1_src16 + (m2_src16 << 1) + m3_src16) >> 2;
    
    pSrcL += srcStride;
    short16 m4_src16 = convert_short16( vload16(0, pSrcL    ) );
                 r16 = convert_short16( vload16(0, pSrcL + 2) );
    short16 m5_src16 = (short16)(m4_src16.s12345678, r16.s789abcde);
    short16 m6_src16 = (short16)(m4_src16.s23456789, r16.s89abcdef);
    short16     dst1 = (m4_src16 + (m5_src16 << 1) + m6_src16) >> 2;
    
    pSrcL += srcStride;
    short16 m7_src16 = convert_short16( vload16(0, pSrcL    ) );
                 r16 = convert_short16( vload16(0, pSrcL + 2) );
    short16 m8_src16 = (short16)(m7_src16.s12345678, r16.s789abcde);
    short16 m9_src16 = (short16)(m7_src16.s23456789, r16.s89abcdef);
    short16     dst2 = (m7_src16 + (m8_src16 << 1) + m9_src16) >> 2;
    
    short16 dst = (dst0 + (dst1 << 1) + dst2) >> 2;
    vstore16(convert_uchar16(dst), 0, pDstL);
}

__kernel void opt_RS_GaussianBlur(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    int width,
    int height,
    int srcStride,
    int bufStride)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	__global unsigned char *pSrcL = pSrc + (y - 1) * srcStride + x - 1;
	__global unsigned char *pDstL = pDst + y * bufStride + x;

	short dst, dst0, dst1, dst2;

	short4 m1_src4 = convert_short4(vload4(0, pSrcL));
    dst0 = (m1_src4.s0 + 2*m1_src4.s1 + m1_src4.s2) >> 2;

	pSrcL += srcStride;	
	short4 m2_src4 = convert_short4(vload4(0, pSrcL));
    dst1 = (m2_src4.s0 + 2*m2_src4.s1 + m2_src4.s2) >> 2; 

	pSrcL += srcStride;
	short4 m3_src4 = convert_short4(vload4(0, pSrcL));
	dst2 = (m3_src4.s0 + 2*m3_src4.s1 + m3_src4.s2) >> 2; 

    dst = (dst0 + 2*dst1 + dst2) >> 2;
    *pDstL = (uchar)(dst);
}

__kernel void opt_LR_GaussianBlur(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    int width,
    int height,
    int srcStride,
    int bufStride)
{
    int y = get_global_id(0);

    __global unsigned char *pSrcL = pSrc + y * srcStride + srcStride;
    __global unsigned char *pDstL = pDst + y * bufStride + bufStride;

    short2 line0, line1, line2;
    short dst, dst0, dst1, dst2;

    line0 = convert_short2( vload2(0, pSrcL - srcStride) );
    dst0 = (line0.s0 * 3 + line0.s1) >> 2;
    line1 = convert_short2( vload2(0, pSrcL) );
    dst1 = (line1.s0 * 3 + line1.s1) >> 2;
    line2 = convert_short2( vload2(0, pSrcL + srcStride) );
    dst2 = (line2.s0 * 3 + line2.s1) >> 2;
    dst = (dst0 + dst1 * 2 + dst2) >> 2;
    *pDstL = (uchar)(dst);
    
    line0 = convert_short2( vload2(0, pSrcL - 2) );
    dst0 = (line0.s0 + line0.s1 * 3) >> 2;
    line1 = convert_short2( vload2(0, pSrcL + srcStride - 2) );
    dst1 = (line1.s0 + line1.s1 * 3) >> 2;
    line2 = convert_short2( vload2(0, pSrcL + srcStride * 2 - 2) );
    dst2 = (line2.s0 + line2.s1 * 3) >> 2;
    dst = (dst0 + dst1 * 2 + dst2) >> 2;
    pDstL += bufStride - 1;
    *pDstL = (uchar)(dst);
}

__kernel void opt_TB_GaussianBlur(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    int width,
    int height,
    int srcStride,
    int bufStride)
{
    int x = get_global_id(0);
    
    __global unsigned char *pSrcT = pSrc + x;
    __global unsigned char *pDstT = pDst + x;
    
    __global unsigned char *pSrcB = pSrc + srcStride * (height - 1) + x;
    __global unsigned char *pDstB = pDst + bufStride * (height - 1) + x;
    
    short4 line0, line1;
    short dst, dst0, dst1;
    
    if( x == 0 )
    {
        line0 = convert_short4( vload4(0,pSrcT) );
        dst0 = (line0.s0 * 3 + line0.s1) >> 2;
        line1 = convert_short4( vload4(0,pSrcT + srcStride) );
        dst1 = (line1.s0 * 3 + line1.s1) >> 2;
        dst = (dst0 * 3 + dst1) >> 2;
        *pDstT = (uchar)(dst);

        line0 = convert_short4( vload4(0,pSrcB) );
        dst0 = (line0.s0 * 3 + line0.s1) >> 2;
        line1 = convert_short4( vload4(0,pSrcB - srcStride) );
        dst1 = (line1.s0 * 3 + line1.s1) >> 2;
        dst = (dst0 * 3 + dst1) >> 2;
        *pDstB = (uchar)(dst);
    }
    else if( x == (width - 1) )
    {
        line0 = convert_short4( vload4(0, pSrcT - 1) );
        dst0 = (line0.s0 + 3 * line0.s1) >> 2;
        line1 = convert_short4( vload4(0, pSrcT + srcStride - 1) );
        dst1 = (line1.s0 + 3 * line1.s1) >> 2;
        dst = (dst0 * 3 + dst1) >> 2;
        *pDstT = (uchar)(dst);

        line0 = convert_short4( vload4(0, pSrcB - 1) );
        dst0 = (line0.s0 + 3 * line0.s1) >> 2;
        line1 = convert_short4( vload4(0, pSrcB - srcStride - 1) ); 
        dst1 = (line1.s0 + 3 * line1.s1) >> 2;	
        dst = (dst0 * 3 + dst1) >> 2;
        *pDstB = (uchar)(dst);
    }
    else
    {
        line0 = convert_short4( vload4(0, pSrcT - 1) );
        dst0 = (line0.s0 + 2 * line0.s1 + line0.s2) >> 2;
        line1 = convert_short4( vload4(0, pSrcT + srcStride - 1) ); 	
        dst1 = (line1.s0 + 2 * line1.s1 + line1.s2) >> 2;
        dst = (dst0 * 3 + dst1) >> 2;
        *pDstT = (uchar)(dst);

        line0 = convert_short4( vload4(0, pSrcB - 1) );
        dst0 = (line0.s0 + 2 * line0.s1 + line0.s2) >> 2;
        line1 = convert_short4( vload4(0, pSrcB - srcStride - 1) ); 
        dst1 = (line1.s0 + 2 * line1.s1 + line1.s2) >> 2;	
        dst = (dst0 * 3 + dst1) >> 2;
        *pDstB = (uchar)(dst);
    }
}

/********************************************************************************
  Function Description:     

    sobel 算子:
    水平         垂直
    -1  0  1     1  2  1
    -2  0  2     0  0  0
    -1  0  1    -1 -2 -1

  Parameters:          
      > pSrc			- [in]  输入图像
      > pEdge			- [out] 输出边缘信息
      > width			- [in]  输入图像pSrc与输入图像边缘信息的宽度一样
      > height			- [in]  输入图像pSrc与输入图像边缘信息的高度一样
      > stride			- [in]  输入图像pSrc的步长
      > edgeStride		- [in]  输入图像边缘信息pEdge的步长
 ********************************************************************************/
__kernel void CalcSobel(
    __global unsigned char *pSrc,
    __global unsigned char *pEdge,
    int width,
    int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    __global unsigned char *pEdgeL = pEdge + ( y + 1 ) * width + ( x << 3 ) + 1;
    __global unsigned char *pSrcL0 = pSrc + y * width + ( x << 3 );
    __global unsigned char *pSrcL1 = pSrcL0 + width;
    __global unsigned char *pSrcL2 = pSrcL1 + width;
    
    short8 a0 = convert_short8( vload8( 0, pSrcL0     ) );
    short8 a1 = convert_short8( vload8( 0, pSrcL0 + 1 ) );
    short8 a2 = convert_short8( vload8( 0, pSrcL0 + 2 ) );
    short8 b0 = convert_short8( vload8( 0, pSrcL1     ) );
    short8 b2 = convert_short8( vload8( 0, pSrcL1 + 2 ) );
    short8 c0 = convert_short8( vload8( 0, pSrcL2     ) );
    short8 c1 = convert_short8( vload8( 0, pSrcL2 + 1 ) );
    short8 c2 = convert_short8( vload8( 0, pSrcL2 + 2 ) );
    
    short8 gx, gy;
    ushort8 edgeV;
    gx = (a2 + (b2 << 1) + c2) - (a0 + (b0 << 1) + c0);
    gy = (a0 + (a1 << 1) + a2) - (c0 + (c1 << 1) + c2);
    edgeV = min(abs(gx) + abs(gy), (ushort8)(255));
    vstore8(convert_uchar8(edgeV), 0, pEdgeL);
}

// require input width should be a even number
__kernel void CalcSobel_BT(
    __global unsigned char *pEdge,
    int offset)
{
    int x = get_global_id(0) << 1;
    vstore2((uchar2)0, 0, pEdge + offset + x);
}

__kernel void CalcSobel_LR(
    __global unsigned char *pSrc,
    __global unsigned char *pEdge,
    int width,
    int height,
    int vOffset)
{
    int x = get_global_id(0);
    int y = get_global_id(1) + 1;

    int offset = y * width;
    pSrc += offset;
    pEdge += offset;

    if (x == 0)
    {
        pEdge[0] = 0;
        pEdge[width-1] = 0;
    }
    else
    {
        int gx, gy;
        int edgeV;
        __global unsigned char *pSrcL0 = pSrc - width + vOffset + x - 1;
        __global unsigned char *pSrcL1 = pSrcL0 + width;
        __global unsigned char *pSrcL2 = pSrcL1 + width;

        gx = (pSrcL0[2] + ((int)pSrcL1[2] << 1) + pSrcL2[2]) - (pSrcL0[0] + ((int)pSrcL1[0] << 1) + pSrcL2[0]);
        gy = (pSrcL0[0] + ((int)pSrcL0[1] << 1) + pSrcL0[2]) - (pSrcL2[0] + ((int)pSrcL2[1] << 1) + pSrcL2[2]);

        edgeV = (abs(gx) + abs(gy));
        pEdge[vOffset+x] = min(edgeV, 255);
    }
}

/********************************************************************************
  Function Description:     

    利用小波的两个高频分量进行边缘分析

    sobel 算子:
     水平       垂直
    -1  0 1     1  2  1
    -2  0 2     0  0  0
    -1  0 1    -1 -2 -1

  Parameters:          
      > pSrcV               - [in]  输入harr 小波的垂直方向高频图像
      > pSrcH               - [in]  输入harr 小波的水平方向高频图像
      > pEdge				- [out] 输出边缘信息
      > width				- [in]  输入图像pSrcV、pSrcH与输入图像边缘信息的宽度一样
      > height				- [in]  输入图像pSrcV、pSrcH与输入图像边缘信息的高度一样
      > stride				- [in]  输入图像pSrcV、pSrcH的步长
      > edgeStride			- [in]  输入图像边缘信息pEdge的步长      
 ********************************************************************************/
__kernel void CalcSobel_HF(
    __global unsigned char *pSrcV,
    __global unsigned char *pSrcH,
    __global unsigned char *pEdge,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    short8 gx, gy;
    ushort8 edgeV;
    
    __global unsigned char *pEdgeL = pEdge + ( y + 1 ) * width + ( x << 3 ) + 1;
    __global unsigned char *pSrcL0 = pSrcV + y * width + ( x << 3 );
    __global unsigned char *pSrcL1 = pSrcL0 + width;
    __global unsigned char *pSrcL2 = pSrcL1 + width;
    
    short8 a0 = convert_short8( vload8(0, pSrcL0    ) );
    short8 a1 = convert_short8( vload8(0, pSrcL0 + 1) );
    short8 a2 = convert_short8( vload8(0, pSrcL0 + 2) );
    short8 c0 = convert_short8( vload8(0, pSrcL2    ) );
    short8 c1 = convert_short8( vload8(0, pSrcL2 + 1) );
    short8 c2 = convert_short8( vload8(0, pSrcL2 + 2) );
    gy = (a0 + (a1 << 1) + a2) - (c0 + (c1 << 1) + c2);
    
    pSrcL0 = pSrcH + y * width + ( x << 3 );
    pSrcL1 = pSrcL0 + width;
    pSrcL2 = pSrcL1 + width;
    a0 = convert_short8( vload8(0, pSrcL0    ) );
    a2 = convert_short8( vload8(0, pSrcL0 + 2) );
    a1 = convert_short8( vload8(0, pSrcL1    ) );
    c1 = convert_short8( vload8(0, pSrcL1 + 2) );
    c0 = convert_short8( vload8(0, pSrcL2    ) );
    c2 = convert_short8( vload8(0, pSrcL2 + 2) );
    gx = (a2 + (c1 << 1) + c2) - (a0 + (a1 << 1) + c0);
    
    edgeV = min(abs(gx) + abs(gy), (ushort8)(255));
    vstore8(convert_uchar8(edgeV), 0, pEdgeL);
}

__kernel void CalcSobel_HF_BT(
    __global unsigned char *pEdge,
    int offset)
{
    int x = get_global_id(0) << 2;
    vstore4( (uchar4)0, 0, pEdge + offset + x );
}

__kernel void CalcSobel_HF_LR(
    __global unsigned char *pSrcV,
    __global unsigned char *pSrcH,
    __global unsigned char *pEdge,
    int width,
    int height,
    int vOffset)
{
    int x = get_global_id(0);
    int y = get_global_id(1) + 1;
    
    int offset = y * width;
    pSrcV += offset;
    pSrcH += offset;
    pEdge += offset;
    
    if( x == 0 )
    {
        pEdge[0] = 0;
        pEdge[width-1] = 0;
    }
    else
    {
        int gx, gy;
        int edgeV;
        __global unsigned char *pSrcVL0 = pSrcV - width + vOffset + x - 1;
        __global unsigned char *pSrcVL1 = pSrcVL0 + width;
        __global unsigned char *pSrcVL2 = pSrcVL1 + width;
        
        __global unsigned char *pSrcHL0 = pSrcH - width + vOffset + x - 1;
        __global unsigned char *pSrcHL1 = pSrcHL0 + width;
        __global unsigned char *pSrcHL2 = pSrcHL1 + width;
        
        gx = (pSrcHL0[2] + ((int)pSrcHL1[2] << 1) + pSrcHL2[2]) - (pSrcHL0[0] + ((int)pSrcHL1[0] << 1) + pSrcHL2[0]);
        gy = (pSrcVL0[0] + ((int)pSrcVL0[1] << 1) + pSrcVL0[2]) - (pSrcVL2[0] + ((int)pSrcVL2[1] << 1) + pSrcVL2[2]);
            
        edgeV = (abs(gx) + abs(gy));
        pEdge[vOffset+x] = min(edgeV, 255);
    }
}

/*******************************************************************
 * Recover details
 *******************************************************************/
__kernel void RecoverDetail(
    __global unsigned char *pEdge,
    __global unsigned char *pFiltered,
    __global unsigned char *pDst,
    int detailRecover,
    int width,
    int height,
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    pFiltered += y * stride + ( x << 3 );
    pDst      += y * stride + ( x << 3 );
    pEdge     += y * stride + ( x << 3 );
    
    short8 edgeWeight = (short8)(32767 / detailRecover);
    short8       edge = convert_short8( vload8(0,     pEdge) );
    short8        dst = convert_short8( vload8(0,      pDst) );
    short8     filter = convert_short8( vload8(0, pFiltered) );
    
    int8  weight = max(convert_int8(edge)*convert_int8(edgeWeight), (int8)(6553));
    short8 value = convert_short8((convert_int8(filter)*((int8)(32767)-weight) + convert_int8(dst)*weight + (int8)(16384)) >> 15);
    char8  judge = convert_char8(edge < (short8)detailRecover);

    value = min(value, (short8)(255));	
    value = dst * convert_short8(judge + (char8)1) + value * convert_short8(abs(judge));

    vstore8(convert_uchar8(value), 0, pDst);
}

/*******************************************************************
 * HFNR
 *******************************************************************/
__kernel void HFNR_Edge_luma(
    __global unsigned char *pSrc,
    __global unsigned char *pEdge,
    int width,
    int height,
    int stride,
    int nrTH,
    int edgeTH,
    int edgeWeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if( ( x >= ( width >> 3 ) ) || ( y >= height ) )
    {
        return;
    }
    
    pSrc += y * stride + ( x << 3 );
    pEdge += y * width + ( x << 3 );
    
    short8 src = convert_short8( vload8(0, pSrc) );
    short8 edge = convert_short8( vload8(0, pEdge) );
    
    short8 diff = src - (short8)(128);
    ushort8 absDiff = abs(diff);
    char8 judge = convert_char8(absDiff > (ushort8)1) & convert_char8(absDiff < (ushort8)nrTH) & convert_char8(edge < (short8)edgeTH);
    int8 weight = mul24(convert_int8(edge), (int8)(edgeWeight));
    diff = convert_short8((convert_int8(diff) * weight) >> 15);
    short8 value = clamp(diff+(short8)(128), (short8)(0), (short8)(255));
    
    value = src * convert_short8(judge+(char8)1) + value * convert_short8(abs(judge));
    vstore8(convert_uchar8(value), 0, pSrc);
}

/********************************************************************************
  Function Description:       
      1.  小波高频部分滤波，单点去噪
  Parameters:          
      > pSrc       - [in]  小波系数
      > pDnCoef    - [in]  去噪偏移查询表
      > pEdge      - [out] 输入图像亮度边缘信息
      > width      - [in]  图像宽度(pSrc、pEdge)
      > height     - [in]  图像高度(pSrc、pEdge)
      > Stride     - [in]  图像步长(pSrc、pEdge)          
      > nrTH       - [in]  高频信息降噪强度
      > edgeTH     - [in]  边缘信息亮度噪声判断强度
 ********************************************************************************/
__kernel void HFNR_LUT_Edge_luma(
    __global unsigned char *pSrc,
    __global short *pDnCoef,
    __global unsigned char *pEdge,
    int width,
    int height,
    int stride,
    int edgeTH)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    pSrc += y * stride + ( x << 2 );
    pEdge += y * width + ( x << 2 );
    
    uchar4 src = vload4(0, pSrc);
    uchar4 edge = vload4(0, pEdge);
    int4 currPix, weight;
    uint4 calDiff, dstPix;
    int4 edgeWeight = (int4)(32767/edgeTH);
    
    char4 judge = convert_char4(convert_int4(edge) < (int4)edgeTH) & convert_char4(abs(convert_short4(src) - (short4)128) > (ushort4)1);//true -1; false 0
    weight = (int4)32767 - mul24(convert_int4(edge), edgeWeight);
    currPix = (convert_int4(src) << 7);
    calDiff = convert_uint4(((int4)49167 - currPix)>>5);
    dstPix = convert_uint4(currPix + (weight * (int4)(pDnCoef[calDiff.s0], pDnCoef[calDiff.s1], pDnCoef[calDiff.s2], pDnCoef[calDiff.s3]) >> 15));
    dstPix = (dstPix + (uint4)63) >> 7;
    
    src = convert_uchar4(judge + (char4)1) * src + convert_uchar4(dstPix)*(abs(judge));
    vstore4(src, 0, pSrc);
}

/********************************************************************************
  bilateral horizontal & vertical
 ********************************************************************************/
__inline unsigned char bilateral( short8 ucTemp, __global short* lutR, ushort8 lutS)
{
    ushort8 lutrPos = abs_diff((short8)(ucTemp.s3), ucTemp);

    //lutR is limited to a 32 x sizeof(short) table. Clamp lutrPos to [0-31];
    lutrPos = clamp(lutrPos, (ushort8)(0), (ushort8)(31));
    
    // Only use pixels from [-3,3] 
    ushort8 weight = (ushort8)( lutR[lutrPos.s0], lutR[lutrPos.s1], lutR[lutrPos.s2], lutR[lutrPos.s3],
                                lutR[lutrPos.s4], lutR[lutrPos.s5], lutR[lutrPos.s6], 0 ) * lutS;
    
    //max pixel = 4096 * 255 = 1 044 480 --> int
    int8 pixel = convert_int8(ucTemp) * convert_int8( weight );
    
    //max weightsum = 4096 * 8 = 32768 --> ushort
    weight.s0123 += weight.s4567;
    weight.s01 += weight.s23;
    weight.s0 += weight.s1;
    
    pixel.s0123 += pixel.s4567;
    pixel.s01 += pixel.s23;
    pixel.s0 += pixel.s1;
    
    // Change the division operation to the same with C code. Otherwise, there is deviation.
    //return round( native_divide(convert_float(pixel.s0), convert_float(weight.s0)));
    return ( ( pixel.s0 + (int)(weight.s0 >> 1) ) / weight.s0 );
}

__kernel void BilateralFilter_Horizontal_r3(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    __global short *lutR,
    ushort8 lutS,
    int width,
    int height,
    int win)
{
    int x = get_global_id(0) << 1;
    int y = get_global_id(1);

    int minLimit = win;
    int maxLimit = width - win -1;

    pSrc += y * width + x;
    pDst += y * width + x;

    if( x < minLimit || x >= maxLimit )
    {
        if (x == 2)
        {
            uchar8 in = vload8(0, pSrc - 2);
            uchar2 out;
            
            out.s0 = in.s2;
            out.s1 = bilateral( convert_short8( in.s01234567), lutR, lutS);
            
            vstore2(out, 0, pDst);
        }
        else if (x == maxLimit)
        {
            uchar8 in = vload8(0, pSrc - 4);
            uchar2 out;
            
            out.s0 = bilateral( convert_short8( in.s12345677), lutR, lutS);
            out.s1 = in.s5;
            
            vstore2(out, 0, pDst);
        }
        else
        {
            vstore2( vload2(0,pSrc), 0, pDst);
        }
    }
    else
    {
        uchar8 in = vload8(0, pSrc-3);
        uchar2 out;
        
        //Only 7 elements are used by the function, so the last element doesn't matter
        out.s0 = bilateral( convert_short8( in.s01234567), lutR, lutS);
        out.s1 = bilateral( convert_short8( in.s12345677), lutR, lutS);

        vstore2(out, 0, pDst);
    }
}

__kernel void BilateralFilter_Vertical_r3(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    __global short *lutR,
    ushort8 lutS,
    int width,
    int height,
    int win)
{
    int x = get_global_id(0) << 1;
    int y = get_global_id(1);

    int minLimit = win;
    int maxLimit = height - win -1;

    pSrc += y * width + x;
    pDst += y * width + x;

    if( y < minLimit || y > maxLimit )
    {
        vstore2( vload2(0, pSrc), 0, pDst );
    }
    else
    {
        int pixelsum = 0;
        int weightsum = 0;
        
        uchar2 p0,p1,p2,p3,p4,p5,p6;
        uchar2 out;
        p0 = vload2(0, pSrc - 3 * width);
        p1 = vload2(0, pSrc - 2 * width);
        p2 = vload2(0, pSrc -     width);
        p3 = vload2(0, pSrc            );
        p4 = vload2(0, pSrc +     width);
        p5 = vload2(0, pSrc + 2 * width);
        p6 = vload2(0, pSrc + 3 * width);
		
        //Only 7 elements are used by the function, so the last element doesn't matter
        out.s0 = bilateral( (short8)( p0.s0,p1.s0,p2.s0,p3.s0,p4.s0,p5.s0,p6.s0, 0), lutR, lutS);
        out.s1 = bilateral( (short8)( p0.s1,p1.s1,p2.s1,p3.s1,p4.s1,p5.s1,p6.s1, 0), lutR, lutS);

        vstore2(out,0,pDst);
    }
}

//win = 4
__inline unsigned char bilateral_w4(short16 ucTemp, __global short* lutR, ushort16 lutS)
{
    ushort16 lutrPos = abs_diff((short16)(ucTemp.s4), ucTemp);
    
    // Only use pixels from [-3,3] 
    ushort16 weight = (ushort16)( lutR[lutrPos.s0], lutR[lutrPos.s1], lutR[lutrPos.s2], lutR[lutrPos.s3],
                                  lutR[lutrPos.s4], lutR[lutrPos.s5], lutR[lutrPos.s6], lutR[lutrPos.s7],
                                  lutR[lutrPos.s8], 0, 0, 0, 0, 0, 0, 0) * lutS;
    
    //max pixel = 4096 * 255 = 1 044 480 --> int
    //int8 pixel = convert_int8(ucTemp) * convert_int8( weight );
    int4 pixel0 = convert_int4(ucTemp.s0123) * convert_int4(weight.s0123);
    int4 pixel1 = convert_int4(ucTemp.s4567) * convert_int4(weight.s4567);
    int4 pixel2 = convert_int4(ucTemp.s89ab) * convert_int4(weight.s89ab);
    
    //max weightsum = 4096 * 8 = 32768 --> ushort
    weight.s0123 += weight.s4567;
    weight.s01 += weight.s23;
    weight.s0 += weight.s1;
    weight.s0 += weight.s8;
    
    //pixel.s0123 += pixel.s4567;
    //pixel.s01 += pixel.s23;
    //pixel.s0 += pixel.s1;
    pixel0 += pixel1;
    pixel0.s01 += pixel0.s23;
    pixel0.s0 += pixel0.s1;
    pixel0.s0 += pixel2.s0;
    
    return round( native_divide(convert_float(pixel0.s0), convert_float(weight.s0)));
}

__kernel void BilateralFilter_Horizontal_r4(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    __global short *lutR,
    ushort16 lutS,
    int width,
    int height,
    int win)
{
    int x = get_global_id(0) << 1;
    int y = get_global_id(1);

    int minLimit = win;
    int maxLimit = width - win -1;

    pSrc += y * width + x;
    pDst += y * width + x;

    if( x < minLimit || x >= maxLimit )
    {
        vstore2( vload2(0,pSrc), 0, pDst);
    }
    else
    {
        uchar16 in = vload16(0, pSrc-4);
        uchar2 out;
        
        //Only 7 elements are used by the function, so the last element doesn't matter
        out.s0 = bilateral_w4( convert_short16( in.s0123456789abcdef), lutR, lutS);
        out.s1 = bilateral_w4( convert_short16( in.s123456789abcedff), lutR, lutS);

        vstore2(out, 0, pDst);
    }
}

__kernel void BilateralFilter_Vertical_r4(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    __global short *lutR,
    ushort16 lutS,
    int width,
    int height,
    int win)
{
    int x = get_global_id(0) << 1;
    int y = get_global_id(1);

    int minLimit = win;
    int maxLimit = height - win -1;

    pSrc += y * width + x;
    pDst += y * width + x;

    if( y < minLimit || y > maxLimit )
    {
        vstore2( vload2(0, pSrc), 0, pDst );
    }
    else
    {
        int pixelsum = 0;
        int weightsum = 0;
        
        uchar2 p0, p1, p2, p3, p4, p5, p6, p7, p8;
        uchar2 out;
        p0 = vload2(0, pSrc - 4 * width);
        p1 = vload2(0, pSrc - 3 * width);
        p2 = vload2(0, pSrc - 2 * width);
        p3 = vload2(0, pSrc -     width);
        p4 = vload2(0, pSrc            );
        p5 = vload2(0, pSrc +     width);
        p6 = vload2(0, pSrc + 2 * width);
        p7 = vload2(0, pSrc + 3 * width);
        p8 = vload2(0, pSrc + 4 * width);
		
        //Only 7 elements are used by the function, so the last element doesn't matter
        out.s0 = bilateral_w4( (short16)( p0.s0,p1.s0,p2.s0,p3.s0,p4.s0,p5.s0,p6.s0,p7.s0,p8.s0,0,0,0,0,0,0,0), lutR, lutS);
        out.s1 = bilateral_w4( (short16)( p0.s1,p1.s1,p2.s1,p3.s1,p4.s1,p5.s1,p6.s1,p7.s1,p8.s1,0,0,0,0,0,0,0), lutR, lutS);

        vstore2(out,0,pDst);
    }
}
/*******************************************************************
 * HFNR night
 *******************************************************************/
__kernel void HFNR_LUT_Edge_luma_night(
    __global unsigned char *pSrc,
    __global short *pDnCoef,
    __global unsigned char *pEdge,
    int width,
    int height,
    int stride,
    int nrEdgeTH)
{
    uint x = get_global_id(0) << 2;
    uint y = get_global_id(1);
    
    pSrc += y * width + x;
    pEdge += y * width + x;
    
    short4 orig = convert_short4( vload4(0, pSrc ) );
    short4 edge = convert_short4( vload4(0, pEdge) ); 
    short4 coef = (short4)( pDnCoef[orig.s0], pDnCoef[orig.s1], pDnCoef[orig.s2], pDnCoef[orig.s3] );
    
    short4 tmp0 = orig - coef;
    edge = (short4) (nrEdgeTH) - edge;
    short4 tmp1 = ( tmp0 + orig + (short4)(1) ) >> 1;
    uchar4 result = convert_uchar4( select(tmp0, tmp1, edge) );
    vstore4(result, 0, pSrc);
}

__kernel void HFNR_Edge_luma_night(
    __global unsigned char *pSrc,
    __global unsigned char *pEdge,
    int width, 
    int height, 
    int stride, 
    int nrTH)
{
	uint x = get_global_id(0) << 3;
  	uint y = get_global_id(1);

  	pSrc += y * width + x;
  	pEdge += y * width + x;

 	ushort8 edgeWeight = (ushort8)(32767 / nrTH);
  	short8 src = convert_short8( vload8(0, pSrc) );

  	short8 diff = src - (short8)(128);
  	ushort8 absDiff = abs(diff);

	// 得到 if(absDiff > 3 && absDiff < nrTH) 判断结果，满足这个条件，值为-1，不满足这个条件值为0 [z00204322]
  	char8 difJudge = convert_char8( ( convert_short8(absDiff) > (short8)3   ) &
 									( convert_short8(absDiff) < (short8)nrTH) );	// 0|0 0|1 1|0

 	//得到if(pEdgeL[x] > 60)判断结果，不满足这个条件，值为-1，满足这个条件值为0 [z00204322]
  	short8 edgeL = convert_short8( vload8(0, pEdge) ); 
  	edgeL = convert_short8( edgeL > (short8)60 );
 
  	uint8 weight = convert_uint8(absDiff) * convert_uint8(edgeWeight);
  	diff = convert_short8((convert_int8(diff) * convert_int8(weight))>>15);
  	short8 value = clamp(diff + (short8)(128), (short8)(0), (short8)(255));	//得到第一步value的值 [z00204322]
 
  	//计算pEdgeL[x] > 60判断后的value值
  	value = value * ((short8)(1) + edgeL) - ((src + value + (short8)(1))>>1) * edgeL;
  
  	//得到最终结果
  	value = src * convert_short8((char8)1 + difJudge) - value * convert_short8(difJudge);
  	vstore8(convert_uchar8(value), 0, pSrc);
}

__kernel void HFNR_LUT_Edge( 
    __global unsigned char *pSrc, 
    __global short int *pDnCoef, 
    __global unsigned char *pEdge, 
    __global unsigned char *pEdgeUV, 
    int width, 
    int height, 
    int stride, 
    int edgeTH, 
    int edgeUVTH, 
    int protectLevel)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    uint   offset = y * width + ( x << 3 );
    uchar8    src = vload8(0, pSrc + offset);
    uchar8   edge = vload8(0, pEdge + offset);
    uchar8 edgeUV = vload8(0, pEdgeUV + offset);
    
    ushort8 currPix = convert_ushort8(src);
    short8   weight = (short8)32767 - convert_short8(edge) * ((short8)32767 / (short8)edgeTH);
    ushort8  dstPix = currPix - convert_ushort8( ( convert_int8(weight) * (int8)(pDnCoef[src.s0], pDnCoef[src.s1], pDnCoef[src.s2], 
    				                              pDnCoef[src.s3], pDnCoef[src.s4], pDnCoef[src.s5], pDnCoef[src.s6], pDnCoef[src.s7] ) ) >> 15 );
    char8     judge = convert_char8(convert_short8(edge) < (short8)edgeTH) & convert_char8(convert_short8(edgeUV) < (short8)edgeUVTH);
    uchar8    value = select(src, convert_uchar8(dstPix), judge);
    
	if( ( ( x << 3 ) + 8 ) <= width )
    {
        vstore8(value, 0, pSrc + offset);
    }
    else if( ( ( x << 3 ) + 6 ) <= width )
    {
        vstore4(value.lo, 0, pSrc + offset);
        vstore2(value.hi.lo, 0, pSrc + offset + 4);
    }
    else if( ( ( x << 3 ) + 4 ) <= width )
    {
        vstore4(value.s0123, 0, pSrc + offset);
    }
    else
    {
        vstore2(value.s01, 0, pSrc + offset);
    }
}

// !!! Have potential issue !!!
// Some picture size might need process rest pixel number 6. Doesn't get such case so far.
__kernel void HFNR_Edge_night( 
    __global unsigned char *pSrc,
    int width,
    int height,
    int stride,
    int nrTH)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    pSrc += y * width + ( x << 3 );
    
    ushort8 edgeWeight = (ushort8)(32767 / nrTH);
    short8 src = convert_short8( vload8(0, pSrc) );
    
    short8 diff = src - (short8)(128);
    ushort8 absDiff = abs(diff);
    uint8 weight = convert_uint8(absDiff) * convert_uint8(edgeWeight);
    diff = convert_short8((convert_int8(diff) * convert_int8(weight)) >> 15);
    diff = clamp( diff + (short8)(128), (short8)(0), (short8)(255) );
    
    //得到 if(absDiff < nrTH) 判断结果，不满足这个条件，值为0，满足这个条件值为-1 [z00204322]
    char8 judge = convert_char8( convert_short8(absDiff) < (short8)nrTH );
    diff = src * convert_short8( judge + (char8)1 ) - diff * convert_short8(judge);
    
    // How many pixels left?
    // if( ( ( x + 1 ) << 3 ) > width ) vstore4? vstore2? vstore 4 + 2?
    
    if( ( (x << 3) + 8 ) <= width )
    {
        vstore8(convert_uchar8(diff), 0, pSrc);
    }
    else if( ( (x << 3) + 4 ) <= width )
    {
        vstore4(convert_uchar4(diff.lo), 0, pSrc);
    }
    else
    {
        vstore2(convert_uchar2(diff.lo.lo), 0, pSrc);
    }
}

// All image width and height are integer times of 16, so the rest pixels number is 8
// Need modify this kernel if this kernel need process any even number of width and height.
__kernel void AlphaBlend_V( 
    __global unsigned char *pSrcDst, 
    __global unsigned char *pBlendBy, 
    int width, 
    int height, 
    int stride, 
    unsigned char alpha)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    uint offset = y * width + ( x << 4 );
    short16 src   = convert_short16( vload16(0, pSrcDst  + offset) );
    short16 blend = convert_short16( vload16(0, pBlendBy + offset) );
    short16 dst = ( ( src - blend ) * alpha + ( blend << 7 ) + (short16)(64) ) >> 7;
    
    // The last x direction kernel is used to process rest pixels of each line.
    if( ((x << 4) + 16) <= width )
    {
        vstore16(convert_uchar16(dst), 0, pSrcDst + offset);
    }
    else
    {
        vstore8(convert_uchar8(dst.lo), 0, pSrcDst + offset);
    }	
}

__kernel void EdgeCoef(
    __global unsigned char *pSrcV,
    __global unsigned char *pSrcH,
    int width,
    int height)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    uint offset = y * width + ( x << 4 );
    
    uchar16 srcV = vload16(0, pSrcV + offset);
    uchar16 srcH = vload16(0, pSrcH + offset);
    uchar16 dst = max(srcH, srcV);
	
    if( ( (x << 4) + 16 ) <= width )
    {
        vstore16(dst, 0, pSrcH + offset);
    }
    else if( ( (x << 4) + 12 ) <= width )
    {
        vstore8(dst.lo, 0, pSrcH + offset);
        vstore4(dst.hi.lo, 0, pSrcH + offset + 8);
    }
    else if( ( (x << 4) + 8 ) <= width )
    {
        vstore8(dst.lo, 0, pSrcH + offset);
    }
    else if( ( (x << 4) + 6 ) <= width )
    {
        vstore4(dst.lo.lo, 0, pSrcH + offset);
        vstore2(dst.lo.hi.lo, 0, pSrcH + offset + 4);
    }
    else if( ( (x << 4) + 4 ) <= width )
    {
        vstore4(dst.lo.lo, 0, pSrcH + offset);
    }
    else
    {
        vstore2(dst.lo.lo.lo, 0, pSrcH + offset);
    }
}

//传导水平方向
__kernel void FastNr2DVector_h(
    __global unsigned char *pSrc,
    __global unsigned short *pPrePix,
    __global short *horizontal,
    int width,
    int height)
{
    uint y = get_global_id(0);
    
    pSrc += ( y << 1 ) * width;
    pPrePix += ( y << 1 ) * width;
    
    ushort4 sL0 = convert_ushort4( vload4(0, pSrc        ) ) << 7;
    ushort4 sL1 = convert_ushort4( vload4(0, pSrc + width) ) << 7;
    
    ushort2 dR, dR0, dR1, dR2, dR3;
    int2 dMul;
    ushort2 curPix, d;
    
    // R0
    dR0 = (ushort2)(sL0.s0, sL1.s0);
    
    // R1
    curPix = (ushort2)(sL0.s1, sL1.s1);
    dMul = convert_int2(dR0) - convert_int2(curPix);
    d = convert_ushort2( (dMul + (int2)32783) >> 5 );
    dR1 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
    
    // R2
    curPix = (ushort2)(sL0.s2, sL1.s2);
    dMul = convert_int2(dR1) - convert_int2(curPix);
    d = convert_ushort2( (dMul + (int2)32783) >> 5 );
    dR2 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
    
    // R3
    curPix = (ushort2)(sL0.s3, sL1.s3);
    dMul = convert_int2(dR2) - convert_int2(curPix);
    d = convert_ushort2( (dMul + (int2)32783) >> 5 );
    dR3 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
    
    vstore4((ushort4)(dR0.s0, dR1.s0, dR2.s0, dR3.s0), 0, pPrePix);
    vstore4((ushort4)(dR0.s1, dR1.s1, dR2.s1, dR3.s1), 0, pPrePix + width);
	
    // loop number should be increase 1 to process rest pixels.
    for(uint x = 1; x < ( (width + 3) >> 2 ); x++)
    {
        pSrc += 4;
        pPrePix += 4;
		
        sL0 = convert_ushort4( vload4(0, pSrc        ) ) << 7;
        sL1 = convert_ushort4( vload4(0, pSrc + width) ) << 7;
        dR = dR3;
        
        // R0
        curPix = (ushort2)(sL0.s0, sL1.s0);
        dMul = convert_int2(dR) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR0 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        // R1
        curPix = (ushort2)(sL0.s1, sL1.s1);
        dMul = convert_int2(dR0) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR1 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        // R2
        curPix = (ushort2)(sL0.s2, sL1.s2);
        dMul = convert_int2(dR1) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR2 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        // R3
        curPix = (ushort2)(sL0.s3, sL1.s3);
        dMul = convert_int2(dR2) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR3 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        if( ( (x << 2) + 4 ) <= width )
        {
            vstore4((ushort4)(dR0.s0, dR1.s0, dR2.s0, dR3.s0), 0, pPrePix);
            vstore4((ushort4)(dR0.s1, dR1.s1, dR2.s1, dR3.s1), 0, pPrePix + width);
        }
        else
        {   
            vstore2((ushort2)(dR0.s0, dR1.s0), 0, pPrePix);
            vstore2((ushort2)(dR0.s1, dR1.s1), 0, pPrePix + width);
        }
    }
}

//传导垂直方向
__kernel void FastNr2DVector_v(
    __global unsigned short *pPrePix,
    __global unsigned short *pPreLine,
    __global short *vertical,
    int width,
    int height)
{
    uint x = get_global_id(0);
    
    pPrePix += ( x << 3 );
    pPreLine += ( x << 3 );
    
    ushort8 preL = vload8(0, pPrePix);
    ushort8 preP;
    int8 dMul;
    ushort8 d;
    
    // first line pPreLine[x] = pPrePix[x];
    if( ( (x << 3) + 8 ) <= width )
    {
        vstore8(preL, 0, pPreLine);
    }
    else if( ( (x << 3) + 6 ) <= width )
    {
        vstore4(preL.lo, 0, pPreLine);
        vstore2(preL.hi.lo, 0, pPreLine + 4);
    }
    else if( ( (x << 3) + 4 ) <= width )
    {
        vstore4(preL.lo, 0, pPreLine);
    }
    else
    {
        vstore2(preL.lo.lo, 0, pPreLine);
    }
		
    for(uint y = 1; y < height; y++)
    {
        pPrePix += width;
        pPreLine += width;
        
        preP = vload8(0, pPrePix);
        dMul = convert_int8(preL) - convert_int8(preP);
        d = convert_ushort8( (dMul + (int8)32783) >> 5 );
        preL = preP + (ushort8)( vertical[d.s0], vertical[d.s1], vertical[d.s2], vertical[d.s3], 
        						 vertical[d.s4], vertical[d.s5], vertical[d.s6], vertical[d.s7] );
	
        if( ( (x << 3) + 8 ) <= width )
        {
            vstore8(preL, 0, pPreLine);
        }
        else if( ( (x << 3) + 6 ) <= width )
        {
            vstore4(preL.lo, 0, pPreLine);
            vstore2(preL.hi.lo, 0, pPreLine + 4);
        }
        else if( ( (x << 3) + 4 ) <= width )
        {
            vstore4(preL.lo, 0, pPreLine);
        }
        else
        {
            vstore2(preL.lo.lo, 0, pPreLine);
        }
    }
}

//传导综合
__kernel void FastNr2DVector(
    __global unsigned char *pSrc,
    __global unsigned char *pDst,
    __global unsigned short *pPreLine,
    __global short *detail,
    int width,
    int height)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    uint offset = y * width + (x << 3);
    ushort8 curPix = convert_ushort8( vload8(0, pSrc + offset) ) << 7;
    ushort8   preL = vload8(0, pPreLine + offset);

    int8      dMul = convert_int8(curPix) - convert_int8(preL);
    ushort8      d = convert_ushort8( ( dMul + (int8)32783 ) >> 5 );
    
    ushort8 DstPix = preL + (ushort8)( detail[d.s0], detail[d.s1], detail[d.s2], detail[d.s3],
    								   detail[d.s4], detail[d.s5], detail[d.s6], detail[d.s7] );
    uchar8    tmp0 = convert_uchar8( ( DstPix + (ushort8)63 ) >> 7 );

    if( ( (x << 3) + 8 ) <= width )
    {
        vstore8(tmp0, 0, pDst + offset);
    }
    else if( ( (x << 3) + 6 ) <= width )
    {
        vstore4(tmp0.lo, 0, pDst + offset);
        vstore2(tmp0.hi.lo, 0, pDst + offset + 4);		
    }
    else if( ( (x << 3) + 4 ) <= width )
    {
        vstore4(tmp0.lo, 0, pDst + offset);
    }
    else
    {
        vstore2(tmp0.lo.lo, 0, pDst + offset);
    }
}

//反向传导水平方向
__kernel void FastNr2DVectorDU_h(
    __global unsigned char *pSrc,
    __global unsigned short *pPrePix,
    __global short *horizontal,
    int width,
    int height)
{
    uint y = get_global_id(0);
    
    pSrc += ( y << 1 ) * width + width - 4;
    pPrePix += ( y << 1 ) * width + width - 4;
    
    ushort4 sL0 = convert_ushort4( vload4(0, pSrc        ) ) << 7;
    ushort4 sL1 = convert_ushort4( vload4(0, pSrc + width) ) << 7;
    
    ushort2 dR, dR0, dR1, dR2, dR3;
    int2 dMul;
    ushort2 curPix, d;
    
    // R0
    dR0 = (ushort2)(sL0.s3, sL1.s3);
    
    // R1
    curPix = (ushort2)(sL0.s2, sL1.s2);
    dMul = convert_int2(dR0) - convert_int2(curPix);
    d = convert_ushort2( (dMul + (int2)32783) >> 5 );
    dR1 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
    
    // R2
    curPix = (ushort2)(sL0.s1, sL1.s1);
    dMul = convert_int2(dR1) - convert_int2(curPix);
    d = convert_ushort2( (dMul + (int2)32783) >> 5 );
    dR2 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
    
    // R3
    curPix = (ushort2)(sL0.s0, sL1.s0);
    dMul = convert_int2(dR2) - convert_int2(curPix);
    d = convert_ushort2( (dMul + (int2)32783) >> 5 );
    dR3 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
    
    vstore4((ushort4)(dR3.s0, dR2.s0, dR1.s0, dR0.s0), 0, pPrePix);
    vstore4((ushort4)(dR3.s1, dR2.s1, dR1.s1, dR0.s1), 0, pPrePix + width);
    
    // loop number should be increase 1 to process rest pixels.
    for(uint x = 1; x < ( (width + 3) >> 2 ); x++)
    {
        pSrc -= 4;
        pPrePix -= 4;
        
        sL0 = convert_ushort4( vload4(0, pSrc        ) ) << 7;
        sL1 = convert_ushort4( vload4(0, pSrc + width) ) << 7;
        dR = dR3;
        
        // R0
        curPix = (ushort2)(sL0.s3, sL1.s3);
        dMul = convert_int2(dR) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR0 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        // R1
        curPix = (ushort2)(sL0.s2, sL1.s2);
        dMul = convert_int2(dR0) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR1 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        // R2
        curPix = (ushort2)(sL0.s1, sL1.s1);
        dMul = convert_int2(dR1) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR2 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        // R3
        curPix = (ushort2)(sL0.s0, sL1.s0);
        dMul = convert_int2(dR2) - convert_int2(curPix);
        d = convert_ushort2( (dMul + (int2)32783) >> 5 );
        dR3 = curPix + (ushort2)(horizontal[d.s0], horizontal[d.s1]);
        
        if( ( (x << 2) + 4 ) <= width )
        {
            vstore4((ushort4)(dR3.s0, dR2.s0, dR1.s0, dR0.s0), 0, pPrePix);
            vstore4((ushort4)(dR3.s1, dR2.s1, dR1.s1, dR0.s1), 0, pPrePix + width);
        }
        else
        {
            /*
            vstore2((ushort2)(dR3.s0, dR2.s0), 0, pPrePix);
            vstore2((ushort2)(dR3.s1, dR2.s1), 0, pPrePix + width);
            */
            // Back to head of line. need store last 2 pixels instead of first 2
            vstore2((ushort2)(dR1.s0, dR0.s0), 0, pPrePix + 2);
            vstore2((ushort2)(dR1.s1, dR0.s1), 0, pPrePix + width + 2);
        }
    }
}

//反向传导垂直方向
__kernel void FastNr2DVectorDU_v(
    __global unsigned short *pPrePix,
    __global unsigned short *pPreLine,
    __global short *vertical,
    int width,
    int height)
{
    uint x = get_global_id(0);
    
    pPrePix += ( height - 1 ) * width + ( x << 3 );
    pPreLine += ( height - 1 ) * width + ( x << 3 );
    
    ushort8 preL = vload8(0, pPrePix);
    ushort8 preP;
    int8 dMul;
    ushort8 d;
    
    if( ( (x << 3) + 8 ) <= width )
    {
        vstore8(preL, 0, pPreLine);
    }
    else if( ( (x << 3) + 6 ) <= width )
    {
        vstore4(preL.lo, 0, pPreLine);
    	vstore2(preL.hi.lo, 0, pPreLine + 4);
    }
    else if( ( (x << 3) + 4 ) <= width )
    {
        vstore4(preL.lo, 0, pPreLine);
    }
    else
    {
    	vstore2(preL.lo.lo, 0, pPreLine);        
    }
    
    for(uint y = 1; y < height; y ++)
    {
        pPrePix -= width;
        pPreLine -= width;
        
        preP = vload8(0, pPrePix);
        dMul = convert_int8(preL) - convert_int8(preP);
        d = convert_ushort8((dMul+(int8)32783)>>5);
        preL = preP + (ushort8)( vertical[d.s0], vertical[d.s1], vertical[d.s2], vertical[d.s3], 
        						 vertical[d.s4], vertical[d.s5], vertical[d.s6], vertical[d.s7] );
        
        if( ( (x << 3) + 8 ) <= width )
        {
            vstore8(preL, 0, pPreLine);
        }
        else if( ( (x << 3) + 6 ) <= width )
        {
            vstore4(preL.lo, 0, pPreLine);
        	vstore2(preL.hi.lo, 0, pPreLine + 4);
        }
        else if( ( (x << 3) + 4 ) <= width )
        {
            vstore4(preL.lo, 0, pPreLine);
        }
        else
        {
        	vstore2(preL.lo.lo, 0, pPreLine);        
        }
    }
}

/**************************************************************************************************
  UV correction & decouple/couple
 **************************************************************************************************/
// UV correction
// Input 2 lines of Y and 1 line of UV, output the corrected value back to the input vector.
// Algorithm conforms to UVCorrection under $(LOCAL_PATH)/libCore/core/source/colorenhance/colormanagement.cpp
void UV_Correction(uchar8* y1, uchar8* y2, uchar8* uv, __global char* pLUT, int step_index)
{
    int lutSize_index    = 8 - step_index;
    int lutSize_index_x2 = lutSize_index << 1;
	
    uint4 index_UVY = ( ( convert_uint4( (*uv).even ) >> step_index ) << lutSize_index_x2 ) + 
                      ( ( convert_uint4( (*uv).odd  ) >> step_index ) << lutSize_index    ) + 
                      ( ( convert_uint4( (*y1).even ) >> step_index )                     );

    index_UVY = ( index_UVY << 1 ) + index_UVY;

    // Read table to get diff value of y/u/v and store them into a vector3.
    char3 diff_value[8];
    diff_value[0] = vload3(0, pLUT + index_UVY.s0);
    diff_value[1] = vload3(0, pLUT + index_UVY.s1);
    diff_value[2] = vload3(0, pLUT + index_UVY.s2);
    diff_value[3] = vload3(0, pLUT + index_UVY.s3);

    // Rearrange diff value to vectors which is convenient for vector calculation.
    short8  diff_y = (short8)( diff_value[0].s0, diff_value[0].s0, diff_value[1].s0, diff_value[1].s0, 
                               diff_value[2].s0, diff_value[2].s0, diff_value[3].s0, diff_value[3].s0 );
    short8 diff_uv = (short8)( diff_value[0].s1, diff_value[0].s2, diff_value[1].s1, diff_value[1].s2, 
                               diff_value[2].s1, diff_value[2].s2, diff_value[3].s1, diff_value[3].s2 );

    *y1 = convert_uchar8_sat( convert_short8(*y1) + diff_y  );
    *y2 = convert_uchar8_sat( convert_short8(*y2) + diff_y  );
    *uv = convert_uchar8_sat( convert_short8(*uv) + diff_uv );
}

// src is a NV12 picture with stride.
// dstx are Y/U/V without stride
__kernel void UVCorrection_nv12_image(
    __global uchar* src_nv12,
    __global uchar* dst_y,
    __global uchar* dst_u,
    __global uchar* dst_v,
    __global char *pLUT,
    uint width,
    uint height,
    uint stride,
    int step_index)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    __global uchar* src_y = src_nv12 + stride * (y << 1);
    __global uchar* src_uv = src_nv12 + stride * height + stride * y;

    // Load src YUV data
    uchar8 y1 = vload8(x, src_y);
    uchar8 y2 = vload8(x, src_y + stride);
    uchar8 uv = vload8(x, src_uv);
/*	
    if( x == 1 && y == 0 )
    {
	    printf("width = %d, height = %d, stride = %d\n", width, height, stride);
        printf("Y1: %#v8hhx \n", y1);
        printf("Y2: %#v8hhx \n", y2);
        printf("UV: %#v8hhx \n", uv);
    }
*/

    // Do UV correction:
    UV_Correction(&y1, &y2, &uv, pLUT, step_index);

    dst_y += y * (width << 1);
    dst_u += y * (width >> 1);
    dst_v += y * (width >> 1);
    
	// store Y/U/V into 3 separate buffers.
    // stride is removed after process.
    vstore8(y1, x, dst_y);
    vstore8(y2, x, dst_y + width);
    
    vstore4(uv.even, x, dst_u);
    vstore4(uv.odd,  x, dst_v);
}

// src is a NV12 picture with stride.
// dst_x are Y/U/V without stride
#define C0 14
#define C1 2
#define F1(d0, d1) ((d0 * (ushort4)C0 + d1 * (ushort4)C1 + (ushort4)8) >> 4)
#define F2(d0, d1) ((d0 * (ushort4)C1 + d1 * (ushort4)C0 + (ushort4)8) >> 4)
__kernel void UVCorrection_nv12_new(
    __global uchar* src_nv12,
    __global uchar* dst_y,
    __global uchar* dst_u,
    __global uchar* dst_v,
    uint width,
    uint height,
    uint stride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
	uint offset_uv;
	
    // clamp to (0, width - 16)
    if( x == 0 )
    {
        offset_uv = 0;
    }
    else if( x == get_global_size(0) - 1 )
    {
        offset_uv = width - 16;
    }
	else
	{
        offset_uv = 8 * x - 4;	
	}
	
    __global uchar* src_y = src_nv12 + stride * (y << 1);
    __global uchar* src_uv = src_nv12 + stride * height + stride * y;

    // Load src YUV data. input buffer stride might not equal to width.
    uchar8  y1 = vload8(x, src_y);
    uchar8  y2 = vload8(x, src_y + stride);
    uchar16 uv = vload16(0, src_uv + offset_uv); // load Un Vn ... Un+7 Vn+7
/*	
    if( x == 519 && y == 0 )
    {
	    printf("width = %d, height = %d, stride = %d, offset_uv = %d\n", width, height, stride, offset_uv);
		printf("Y1: %#v8hhx \n", y1);
        printf("Y2: %#v8hhx \n", y2);
        printf("UV: %#v16hhx\n", uv);
    }
*/
    // Do UV correction:
    uchar8 uv_out;
	ushort4 tmp0;
	ushort4 tmp1;
	ushort4 tmp2;	
	
	if( x == 0 ) // first work item of each line
	{
/*	
	    if( y == 0 )
		{
		    printf("Hi, I'm the first one\n");
			printf("x = %d, offset_uv = %d\n", x, offset_uv);
		}
*/
        // u'0 = f2(u0, u0) = u0
        // u'1 = f1(u0, u2)
        // u'2 = f2(u0, u2)
        // u'3 = f1(u2, u4)
        
        // v'0 = f2(v0, v0) = v0
        // v'1 = f1(v0, v2)
        // v'2 = f2(v0, v2)
        // v'3 = f1(v2, v4)
        tmp0 = convert_ushort4( (uchar4)(uv.s0, uv.s0, uv.s1, uv.s1) );
	    tmp1 = convert_ushort4( (uchar4)(uv.s0, uv.s4, uv.s1, uv.s5) );
	    tmp2 = convert_ushort4( (uchar4)(uv.s4, uv.s8, uv.s5, uv.s9) );
    }
	else if( x == get_global_size(0) - 1 ) // last work item of each line
	{
/*
	    if( y == 0 )
		{
		    printf("Hi, I'm the last one\n");
			printf("x = %d, offset_uv = %d\n", x, offset_uv);
		}
*/
        tmp0 = convert_ushort4( (uchar4)(uv.s4, uv.s8, uv.s5, uv.s9) );
	    tmp1 = convert_ushort4( (uchar4)(uv.s8, uv.sc, uv.s9, uv.sd) );
	    tmp2 = convert_ushort4( (uchar4)(uv.sc, uv.sc, uv.sd, uv.sd) );
	}
	else // middle work items of each line
	{
	    // for example: 2nd work item like below:
        // u'4 = f2(u2, u4)
		// u'5 = f1(u4, u6)
		// u'6 = f2(u4, u6)
		// u'7 = f1(u6, u8)
		
		// v'4 = f2(u2, u4)
		// v'5 = f1(u4, u6)
		// v'6 = f2(u4, u6)
		// v'7 = f1(u6, u8)
	    tmp0 = convert_ushort4( (uchar4)(uv.s0, uv.s4, uv.s1, uv.s5) );
	    tmp1 = convert_ushort4( (uchar4)(uv.s4, uv.s8, uv.s5, uv.s9) );
	    tmp2 = convert_ushort4( (uchar4)(uv.s8, uv.sc, uv.s9, uv.sd) );
	}
	
    uv_out.even = convert_uchar4( F2(tmp0, tmp1) );
	uv_out.odd  = convert_uchar4( F1(tmp1, tmp2) );
/*
    if( x == 519 && y == 0 )
    {
        printf("U_out: %#v4hhx V_out: %#v4hhx \n", uv_out.lo, uv_out.hi);
	}
*/	
    dst_y += y * (width << 1);
    dst_u += y * (width >> 1);
    dst_v += y * (width >> 1);
    
	// store Y/U/V into 3 separate buffers. stride is removed after process.
    vstore8(y1, x, dst_y);
    vstore8(y2, x, dst_y + width);
    
    vstore4(uv_out.lo, x, dst_u);
    vstore4(uv_out.hi, x, dst_v);
}

// src is a NV12 picture with stride.
// dst_x are Y/U/V without stride
__kernel void NV12_decouple_image(
    __global uchar* src_nv12,
    __global uchar* dst_y,
    __global uchar* dst_u,
    __global uchar* dst_v,
    uint width, 
    uint height,
    uint stride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    __global uchar* src_y = src_nv12 + stride * (y << 1);
    __global uchar* src_uv = src_nv12 + stride * height + stride * y;

    // Load src YUV data
    uchar8 y1 = vload8(x, src_y);
    uchar8 y2 = vload8(x, src_y + stride);
    uchar8 uv = vload8(x, src_uv);
/*
    if( x == 1 && y == 0 )
    {
	    printf("width = %d, height = %d, stride = %d\n", width, height, stride);
        printf("Y1: %#v8hhx \n", y1);
        printf("Y2: %#v8hhx \n", y2);
        printf("UV: %#v8hhx \n", uv);
    }
*/
    dst_y += y * (width << 1);
    dst_u += y * (width >> 1);
    dst_v += y * (width >> 1);
    
	// store Y/U/V into 3 separate buffers.
    // stride is removed after process.
    vstore8(y1, x, dst_y);
    vstore8(y2, x, dst_y + width);
    
    vstore4(uv.even, x, dst_u);
    vstore4(uv.odd,  x, dst_v);
}

__kernel void YUV420_couple_image(
    __global uchar* src_y,
    __global uchar* src_u,
    __global uchar* src_v,
    __global uchar* dst_yuv420,
    uint width, 
    uint height,
    uint stride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
	
    src_y += y * (width << 1);
    src_u += y * (width >> 1);
    src_v += y * (width >> 1);
	
    // Read 2 lines of Y, 16 bytes each line and add correction data.
    uchar16 y1 = vload16(x, src_y);
    uchar16 y2 = vload16(x, src_y  + width);
   
    // Read 1 line of UV,  8 bytes each line.
    uchar8 u = vload8(x, src_u);
    uchar8 v = vload8(x, src_v);

    // Write back
	// !!! Becareful, the UV stride should not be stride/2, here actually assume stride == width. !!!
    __global uchar* dst_y = dst_yuv420 + stride * (y << 1);
    __global uchar* dst_u = dst_yuv420 + stride * height + (stride >> 1) * y;
    __global uchar* dst_v = dst_yuv420 + ( (stride * height * 5) >> 2 ) + (stride >> 1) * y;
    vstore16(y1, x, dst_y);
    vstore16(y2, x, dst_y + stride);

    vstore8(u, x, dst_u);
    vstore8(v, x, dst_v);
}

__kernel void NV12_couple_image(
    __global uchar* src_y,
    __global uchar* src_u,
    __global uchar* src_v,
    __global uchar* dst_nv12,
    uint width, 
    uint height,
    uint stride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    src_y += y * (width << 1);
    src_u += y * (width >> 1);
    src_v += y * (width >> 1);

    // Read 2 lines of Y, 16 bytes each line and add correction data.
	uchar16 y1 = vload16(x, src_y);
	uchar16 y2 = vload16(x, src_y + width);
    
    // Read 1 line of UV,  8 bytes each line.
    uchar8 u = vload8(x, src_u);
    uchar8 v = vload8(x, src_v);
	uchar16 uv = (uchar16)(u.s0, v.s0, u.s1, v.s1, u.s2, v.s2, u.s3, v.s3,
                           u.s4, v.s4, u.s5, v.s5, u.s6, v.s6, u.s7, v.s7);

    // Write back
    __global uchar* dst_y = dst_nv12 + stride * (y << 1);
    __global uchar* dst_uv = dst_nv12 + stride * height + stride * y;
    vstore16(y1, x, dst_y);
    vstore16(y2, x, dst_y + stride);
    vstore16(uv, x, dst_uv);
}

// for haar correction
__kernel void YUV420_couple_image_haar_correction(
    __global uchar* src_y,
    __global uchar* src_u,
    __global uchar* src_v,
    __global char* y_err,
    __global uchar* dst_yuv420,
    uint width, 
    uint height,
    uint stride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
	
    src_y += y * (width << 1);
    src_u += y * (width >> 1);
    src_v += y * (width >> 1);
    y_err += y * (width << 1);
/*
    if( x == 0 && y == 0 )
    {
	    printf("width = %d, height = %d, stride = %d\n", width, height, stride);
        printf("Y_src: %#v16hhx \n", vload16(x, src_y));
        printf("Y_err: %#v16hhx \n", vload16(x, y_err));
    }
*/	
    // Read 2 lines of Y, 16 bytes each line and add correction data.
    uchar16 y1 = convert_uchar16_sat( convert_short16( vload16(x, y_err) ) + convert_short16( vload16(x, src_y) ) );
    uchar16 y2 = convert_uchar16_sat( convert_short16( vload16(x, y_err + width) ) + convert_short16( vload16(x, src_y  + width) ) );
   
    // Read 1 line of UV,  8 bytes each line.
    uchar8 u = vload8(x, src_u);
    uchar8 v = vload8(x, src_v);

    // Write back
	// !!! Becareful, the UV stride should not be stride/2, here actually assume stride == width. !!!
    __global uchar* dst_y = dst_yuv420 + stride * (y << 1);
    __global uchar* dst_u = dst_yuv420 + stride * height + (stride >> 1) * y;
    __global uchar* dst_v = dst_yuv420 + ( (stride * height * 5) >> 2 ) + (stride >> 1) * y;
    vstore16(y1, x, dst_y);
    vstore16(y2, x, dst_y + stride);

    vstore8(u, x, dst_u);
    vstore8(v, x, dst_v);
}

// for haar correction
__kernel void NV12_couple_image_haar_correction(
    __global uchar* src_y,
    __global uchar* src_u,
    __global uchar* src_v,
    __global char* y_err,
    __global uchar* dst_nv12,
    uint width, 
    uint height,
    uint stride)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    src_y += y * (width << 1);
    src_u += y * (width >> 1);
    src_v += y * (width >> 1);
    y_err += y * (width << 1);

	uchar16 y1_data = vload16(x, src_y);
	char16 y1_err_data = vload16(x, y_err);
	uchar16 y2_data = vload16(x, src_y + width);
	char16 y2_err_data = vload16(x, y_err + width);

    // Read 2 lines of Y, 16 bytes each line and add correction data.
    uchar16 y1 = convert_uchar16_sat( convert_short16(y1_data) + convert_short16(y1_err_data) );
    uchar16 y2 = convert_uchar16_sat( convert_short16(y2_data) + convert_short16(y2_err_data) );

/*
	short16 y1_16 = convert_short16(y1_data);
	short16 y1_err_16 = convert_short16(y1_err_data);
	short16 sum = y1_16 + y1_err_16;
	
    if( x == 0 && y == 0 )
    {
	    printf("width = %d, height = %d, stride = %d\n", width, height, stride);
        printf("y1_data: %#v16hhx\n", y1_data);
        printf("y1_err_data: %#v16hhx\n", y1_err_data);
        printf("y1_data_16: %#v16hx\n", y1_16);
        printf("y1_err_data_16: %#v16hx\n", y1_err_16);
        printf("sum: %#v16hx\n", sum);
    }
*/
    
    // Read 1 line of UV,  8 bytes each line.
    uchar8 u = vload8(x, src_u);
    uchar8 v = vload8(x, src_v);
	uchar16 uv = (uchar16)(u.s0, v.s0, u.s1, v.s1, u.s2, v.s2, u.s3, v.s3,
                           u.s4, v.s4, u.s5, v.s5, u.s6, v.s6, u.s7, v.s7);

    // Write back
    __global uchar* dst_y = dst_nv12 + stride * (y << 1);
    __global uchar* dst_uv = dst_nv12 + stride * height + stride * y;
    vstore16(y1, x, dst_y);
    vstore16(y2, x, dst_y + stride);
    vstore16(uv, x, dst_uv);
}

__kernel void ApplyInverse_haar_correction(
    __global unsigned char *pSrcLL,
    __global unsigned char *pSrcLH,
    __global unsigned char *pSrcHL,
    __global unsigned char *pSrcHH,
    __global unsigned char *pSrc,
    __global char *pErr,
    int width,
    int height,
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    uint src_offset = y * ( stride >> 1 ) + ( x << 3 );
    x = x << 4;
    pSrc += y * (stride << 1) + x;
    pErr += y * (stride << 1) + x;
    
    short8 temp0 = convert_short8( vload8(0, pSrcLL + src_offset) );
    short8 temp1 = convert_short8( vload8(0, pSrcLH + src_offset) );
    short8 temp2 = convert_short8( vload8(0, pSrcHL + src_offset) );
    short8 temp3 = convert_short8( vload8(0, pSrcHH + src_offset) );
    
    uchar16 src0 = vload16(0, pSrc);
    uchar16 src1 = vload16(0, pSrc + stride);
    
    short8 pix0 = temp0 - temp1 - temp2 + temp3 + (short8)(128);
    short8 pix1 = temp0 + temp1 - temp2 - temp3 + (short8)(128);
    short8 pix2 = temp0 - temp1 + temp2 - temp3 + (short8)(128);
    short8 pix3 = temp0 + temp1 + temp2 + temp3 - (short8)(384);
    
    ushort16 mask16 = (ushort16)(0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15);
    uchar16 tmp0 = convert_uchar16_sat(shuffle2(pix0, pix1, mask16));
    uchar16 tmp1 = convert_uchar16_sat(shuffle2(pix2, pix3, mask16));
    short16 tmp2 = convert_short16(src0) - convert_short16(tmp0);
    short16 tmp3 = convert_short16(src1) - convert_short16(tmp1);
    vstore16(convert_char16(tmp2), 0, pErr);
    vstore16(convert_char16(tmp3), 0, pErr + stride);

    // No rest pixel issue so far.
    /*    
    if( (x + 16) <= width )
    {
        ushort16 mask16 = (ushort16)(0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15);
        //uchar16 tmp0 = convert_uchar16( shuffle2( clamp(pix0, (short8)(0), (short8)(255)), clamp(pix1, (short8)(0), (short8)(255)), mask16 ) );
        //uchar16 tmp1 = convert_uchar16( shuffle2( clamp(pix2, (short8)(0), (short8)(255)), clamp(pix3, (short8)(0), (short8)(255)), mask16 ) );
        //vstore16(tmp0, 0, pErrLine0 + dst_offset);
        //vstore16(tmp1, 0, pErrLine1 + dst_offset);
        
        //short16 tmp0 = ( shuffle2( clamp(pix0, (short8)(0), (short8)(255)), clamp(pix1, (short8)(0), (short8)(255)), mask16 ) );
        //short16 tmp1 = ( shuffle2( clamp(pix2, (short8)(0), (short8)(255)), clamp(pix3, (short8)(0), (short8)(255)), mask16 ) );
        short16 tmp0 = convert_short16(convert_uchar16_sat(shuffle2(pix0, pix1, mask16)));
        short16 tmp1 = convert_short16(convert_uchar16_sat(shuffle2(pix2, pix3, mask16)));
        tmp0 = convert_short16(src0) - tmp0;
        tmp1 = convert_short16(src1) - tmp1;
        vstore16(convert_char16(tmp0), 0, pErr);
        vstore16(convert_char16(tmp1), 0, pErr + stride);
    }
    else
    {
        ushort8 mask8 = (ushort8)(0,4,1,5,2,6,3,7);
        //char8 tmp0 = convert_char8( shuffle2( clamp(pix0.s0123, (short4)(0), (short4)(255)), clamp(pix1.s0123, (short4)(0), (short4)(255)), mask8 ) );
        //char8 tmp1 = convert_char8( shuffle2( clamp(pix2.s0123, (short4)(0), (short4)(255)), clamp(pix3.s0123, (short4)(0), (short4)(255)), mask8 ) );
        //vstore8(tmp0.s01234567, 0, pErrLine0 + dst_offset);
        //vstore8(tmp1.s01234567, 0, pErrLine1 + dst_offset);
        short8 tmp0 = convert_short8(convert_uchar8_sat(shuffle2(pix0.s0123, pix1.s0123, mask8)));
        short8 tmp1 = convert_short8(convert_uchar8_sat(shuffle2(pix2.s0123, pix3.s0123, mask8)));
        char8 tmp2 = convert_char8(convert_short8(src0.lo) - tmp0);
        char8 tmp3 = convert_char8(convert_short8(src1.lo) - tmp0);
        vstore8(tmp2,0,pErr);
        vstore8(tmp3,0,pErr + stride);
    }
    */
}

// host program ensures the data size is multiple of 16.
__kernel void copy_buffer(
    __global unsigned char *dst,
    __global unsigned char *src,
    int dst_offset,
	int src_offset)
	
{
    uint x = get_global_id(0);

    vstore16( vload16(x, src + src_offset), x, dst + dst_offset );
}

/**************************************************************************************************
  colorenhancement global analysis
 **************************************************************************************************/
__kernel void Analyze_saturation_horizontal(
    __global unsigned char *src_u,
    __global unsigned char *src_v,
    __global unsigned short *dst_sat,
    __local unsigned short *sat_buf,
    short sT,
    int width_u,
    int stride_u)
{
    int y = get_global_id(0);
    src_u += y * stride_u;
    src_v += y * stride_u;
    int x;
    uchar16 u_in, v_in;
    short8 tmp0, tmp1;
    uchar16 tmp2;
    ushort sat = 0;
    
    for(x=0; ((x<<4) <= (width_u - 16)); x++)
    {
        u_in = vload16(0,src_u + (x<<4));
        v_in = vload16(0,src_v + (x<<4));
        tmp0 = convert_short8(u_in.lo) - (short8)128;
        tmp0 = tmp0 * tmp0;
        tmp1 = convert_short8(v_in.lo) - (short8)128;
        tmp1 = tmp1 * tmp1;
        tmp0 += tmp1;
        tmp2.lo = convert_uchar8(tmp0 > (short8)sT);
        tmp0 = convert_short8(u_in.hi) - (short8)128;
        tmp0 = tmp0 * tmp0;
        tmp1 = convert_short8(v_in.hi) - (short8)128;
        tmp1 = tmp1 * tmp1;
        tmp0 += tmp1;        
        tmp2.hi = convert_uchar8(tmp0 > (short8)sT);
        tmp2 = select((uchar16)0, (uchar16)1, tmp2);
        tmp2.lo += tmp2.hi;   //8
        tmp2.lo.lo += tmp2.lo.hi; //4
        tmp2.lo.lo.lo += tmp2.lo.lo.hi; //2
        tmp2.s0 += tmp2.s1;
        sat += (ushort) tmp2.s0;
    }
    if(width_u & 0x8)
    {
        u_in = vload16(0,src_u + (x<<4));
        v_in = vload16(0,src_v + (x<<4));
        tmp0 = convert_short8(u_in.lo) - (short8)128;
        tmp0 = tmp0 * tmp0;
        tmp1 = convert_short8(v_in.lo) - (short8)128;
        tmp1 = tmp1 * tmp1;
        tmp0 += tmp1;
        tmp2.lo = convert_uchar8(tmp0 > (short8)sT);
        tmp2.lo = select((uchar8)0, (uchar8)1, tmp2.lo);
        tmp2.lo.lo += tmp2.lo.hi; //4
        tmp2.lo.lo.lo += tmp2.lo.lo.hi; //2
        tmp2.s0 += tmp2.s1;
        sat += (ushort) tmp2.s0;
    }
    sat_buf[y] = sat;
    barrier(CLK_LOCAL_MEM_FENCE);
    ushort8 tmp3;
    if((y & 0x7) == 0)
    {   
        tmp3 = vload8(0,sat_buf + y);
        tmp3.lo += tmp3.hi; //4
        tmp3.lo.lo += tmp3.lo.hi; //2
        tmp3.s0 += tmp3.s1;
        dst_sat[y >> 3] = tmp3.s0;
    }
}

__kernel void Analyze_saturation_vertical()
{}

__kernel void Analyze_smoothness_horizontal()
{}

__kernel void Analyze_smoothness_vertical()
{}

#pragma OPENCL EXTENSION cl_arm_printf : disable