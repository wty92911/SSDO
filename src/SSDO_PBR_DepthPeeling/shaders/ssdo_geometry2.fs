#version 330 core
#extension GL_ARB_conservative_depth : enable
#extension GL_EXT_conservative_depth : enable
layout (depth_any) out float gl_FragDepth;
layout (location = 0) out vec3 secondZBuffer;
uniform sampler2D gPosition;
uniform mat4 projection;

in vec3 FragPos;
const float EPS=1e-2;

float Abs(float x)
{
	return x<0?-x:x;
}

void main()
{
	vec4 offset=vec4(FragPos, 1.0f);
	offset=projection * offset;
	offset.xyz/=offset.w;
	offset.xyz=offset.xyz*0.5+0.5;
	float original=texture(gPosition, vec2(gl_FragCoord.x/1600.0f, gl_FragCoord.y/1200.0f)).z;
	if(Abs(FragPos.z-original)<EPS)
		gl_FragDepth = 0.99;
	else
		gl_FragDepth = gl_FragCoord.z;
	secondZBuffer.xyz=FragPos;
}