#include <metal_stdlib>
using namespace metal;

#define vec2 float2
#define vec3 float3
#define vec4 float4
#define ivec2 int2
#define ivec3 int3
#define ivec4 int4
#define mat2 float2x2
#define mat3 float3x3
#define mat4 float4x4

kernel void processimage(
    texture2d<float,access::read> src1[[texture(0)]],
    texture2d<float,access::read> src2[[texture(1)]],
    texture2d<float,access::write> dst[[texture(2)]],
    uint2 gid[[thread_position_in_grid]]) {
        
    if(gid.x<1920&&gid.y<1088) {
        
        float d = src2.read((gid>>1)).b;
        float m = (src2.read(uint2(960,0)+(gid>>1)).b);
        
        if(d>128) {
            
            float3 residual = float3(
                src2.read(gid).r-src2.read(gid).g,
                src2.read(uint2(0,1088)+(gid>>1)).r-src2.read(uint2(0,1088)+(gid>>1)).g,
                src2.read(uint2(960,1088)+(gid>>1)).r-src2.read(uint2(960,1088)+(gid>>1)).g
            ); 
            
            if(m>128) {   
                          
                int x = src2.read(uint2(  0,544)+(gid>>1)).b-128;
                int y = src2.read(uint2(960,544)+(gid>>1)).b-128;
                
                int vx = x>>1;
                int vy = y>>1;
                
                int ox = x&1;
                int oy = y&1;
                
                if(ox&&oy) {
                    dst.write(float4(residual+(src1.read(gid+uint2(vx,vy)).rgb+src1.read(gid+uint2(vx+ox,(vy+oy))).rgb)*.5,255.),gid);
                }
                else {
                    dst.write(float4(residual+src1.read(gid+uint2(vx,vy)).rgb,255.),gid);
                }
            }
            else {
                dst.write(float4(residual+128.,255.),gid);
            }
        }
        else {
            dst.write(float4(src1.read(gid).rgb,255.),gid);
        }
    }
    else {
        dst.write(float4(0.,0.,0.,255.),gid);
    }
}
