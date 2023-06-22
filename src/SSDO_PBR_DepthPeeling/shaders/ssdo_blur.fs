#version 330 core
out vec3 FragColor;

in vec2 TexCoords;

uniform sampler2D ssdoInput;
uniform float gauss[25];
uniform sampler2D texture_diffuse1;
float gausssum=0.0f;
void main() 
{
    vec2 texelSize = 1.0 / vec2(textureSize(ssdoInput, 0));
    vec3 result = vec3(0.0f);
    for (int x = -2; x <=2; ++x) 
    {
        for (int y = -2; y <= 2; ++y) 
        {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result +=gauss[(x+1)*5+y+1]*texture(ssdoInput, TexCoords + offset).rgb;
        }
    }
    for(int i=0;i<25;i++)
        gausssum+=gauss[i];
    result=result/gausssum;
    FragColor = result;
}  