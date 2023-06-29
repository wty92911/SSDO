#version 330 core
out vec3 FragColor;

in vec2 TexCoords;

uniform sampler2D ssdoInput;
uniform int gs;
uniform float offset_factor;
uniform float gauss[17 * 17];
uniform sampler2D texture_diffuse1;

void main() 
{
    float gausssum=0.0f;
    vec2 texelSize = 1.0 / vec2(textureSize(ssdoInput, 0)) * offset_factor;
    vec3 result = vec3(0.0f);
    for (int x = -gs; x <= gs; ++x) 
    {
        for (int y = -gs; y <= gs; ++y) 
        {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result +=gauss[(x + gs)* (gs * 2 + 1) + y + gs] * texture(ssdoInput, TexCoords + offset).rgb;
            gausssum += gauss[(x + gs)* (gs * 2 + 1) + y + gs];
        }
    }
   
    result=result/gausssum;
    FragColor = result;

}  