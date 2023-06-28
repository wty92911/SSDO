#include <config.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <random>
#include <cmath>
#include <map>

#include "utils.hpp"


float PI = 3.1415926;
float sigma = 1.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path, bool gammaCorrection);
void renderQuad();


int SCR_WIDTH = 800;
int SCR_HEIGHT = 600;


Camera camera(glm::vec3(5.0f, 5.0f, 0.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;


float deltaTime = 0.0f;
float lastFrame = 0.0f;

std::vector<glm::vec3> ssdoKernel = get_ssdo_kernel();
std::vector<glm::vec3> ssdoNoise = get_ssdo_noise();

class Buffers{
    public:
    unsigned int gBuffer;
    unsigned int gPosition, gNormal, gAlbedo;
    unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    unsigned int rboDepth;
    unsigned int ssdoFBO;
    unsigned int ssdoColorBuffer;
    unsigned int ssdoFBO2, ssdoBlurFBO;
    unsigned int ssdoColorBuffer2, ssdoColorBufferBlur;
    unsigned int noiseTexture;
}buf;

class Config{
    public:
    map<string, string> cfgfile;
    bool config_modified = true;
    bool use_ssdo = true;
    float sphere_radius = 0.5;
    float direct_strength = 0.5;
    float indirect_strength = 0.5;
    float scale = 1.0;

    void load_initial(){
        scale = (float)atof(cfgfile["scale"].c_str());
    }
}config;

void remake(){
    
    glGenFramebuffers(1, &buf.gBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, buf.gBuffer);
    
    glGenTextures(1, &buf.gPosition);
    glBindTexture(GL_TEXTURE_2D, buf.gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.gPosition, 0);
    // normal 
    glGenTextures(1, &buf.gNormal);
    glBindTexture(GL_TEXTURE_2D, buf.gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buf.gNormal, 0);
    // color
    glGenTextures(1, &buf.gAlbedo);
    glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buf.gAlbedo, 0);

    
    glDrawBuffers(3, buf.attachments);

    
    glGenRenderbuffers(1, &buf.rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, buf.rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buf.rboDepth);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glGenFramebuffers(1, &buf.ssdoFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO);
    
    // SSDO color buffer
    glGenTextures(1, &buf.ssdoColorBuffer);
    glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBuffer, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "SSDO Framebuffer not complete!" << std::endl;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    
    glGenFramebuffers(1, &buf.ssdoFBO2);glGenFramebuffers(1, &buf.ssdoBlurFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO2);
    
    glGenTextures(1, &buf.ssdoColorBuffer2);
    glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBuffer2, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "SSDO Framebuffer not complete!" << std::endl;
    // make some blur
    glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoBlurFBO);
    glGenTextures(1, &buf.ssdoColorBufferBlur);
    glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBufferBlur);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBufferBlur, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "SSAO Blur Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenTextures(1, &buf.noiseTexture);
    glBindTexture(GL_TEXTURE_2D, buf.noiseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssdoNoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

class ImGui{
    Config* cfg;
public:

    void init(Config* _cfg){
        cfg = _cfg;
    }

    void pre_tick(){

    }

    void tick(){

    }

    void after_tick(){

    }

}imgui;

class Shaders{
public:
    const std::string shader_dir = RES_DIR + "/shaders_ssdo";
    Shader shaderGeometryPass = Shader((shader_dir + "/geometry.vs").c_str(), (shader_dir + "/geometry.fs").c_str());
    Shader shaderdirectlight = Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/direct_light.fs").c_str());
    Shader shaderindirectlight = Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/indirect_light.fs").c_str());
    Shader shaderssdoblur = Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/blur.fs").c_str());
    Shader shaderLightingPass = Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/lighting.fs").c_str());   
}shaders;

void set_uniform_shader_params(){
    shaders.shaderdirectlight.use();
    shaders.shaderdirectlight.setInt("gPosition", 0);
    shaders.shaderdirectlight.setInt("gNormal", 1);
    shaders.shaderdirectlight.setInt("gAlbedo", 2);
    shaders.shaderdirectlight.setInt("texNoise", 3);
    shaders.shaderindirectlight.use();
    shaders.shaderindirectlight.setInt("gPosition", 0);
    shaders.shaderindirectlight.setInt("gNormal", 1);
    shaders.shaderindirectlight.setInt("gAlbedo", 2);
    shaders.shaderindirectlight.setInt("texNoise", 3);
    shaders.shaderindirectlight.setInt("directLight", 4);
    shaders.shaderssdoblur.use();
    shaders.shaderssdoblur.setInt("ssdoInput", 0);
    shaders.shaderLightingPass.use();
    shaders.shaderLightingPass.setInt("gPosition", 0);
    shaders.shaderLightingPass.setInt("gNormal", 1);
    shaders.shaderLightingPass.setInt("gAlbedo", 2);
    shaders.shaderLightingPass.setInt("ssdo", 3);

    shaders.shaderssdoblur.use();
    float gauss[3][3];
    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            gauss[i][j] = exp(-float((i - 1) * (i - 1) + (j - 1) * (j - 1)) / (2.0f * sigma * sigma)) / (2.0f * PI * sigma * sigma);
        }
    }
    for (unsigned int i = 0;i < 9;i++) {
        shaders.shaderssdoblur.setFloat("gauss[" + std::to_string(i) + "]", gauss[i / 3][i % 3]);
    }
}


int main(){

    
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif


    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "SSDO", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    config.cfgfile = load_config("config.txt");
    config.load_initial();

    

    Model model_loaded((RES_DIR + "/" + config.cfgfile["model"]).c_str());

    std::cout << model_loaded.textures_loaded.size() << std::endl;
    
    glm::vec3 lightPos = glm::vec3(2.0, 4.0, -2.0);
    glm::vec3 lightColor = glm::vec3(0.5, 0.5, 0.5);


    int last_width = 0, last_height = 0;


    while (!glfwWindowShouldClose(window))
    {

        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);

        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        SCR_WIDTH = width;
        SCR_HEIGHT = height;
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glViewport(0, 0, width, height);


        if(width != last_width || height != last_height){
            last_width = width;
            last_height = height;
                
            remake();
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, buf.gBuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 50.0f);
        glm::mat4 view = camera.GetViewMatrix();
        shaders.shaderGeometryPass.use();
        shaders.shaderGeometryPass.setMat4("projection", projection);
        shaders.shaderGeometryPass.setMat4("view", view);
        shaders.shaderGeometryPass.setInt("invertedNormals", 0);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -0.5f, 0.0));
        model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0.0, 1.0, 0.0));
        model = glm::scale(model, glm::vec3(config.scale));
        shaders.shaderGeometryPass.setMat4("model", model);
        model_loaded.Draw(shaders.shaderGeometryPass);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //directlight
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        shaders.shaderdirectlight.use();
        for (unsigned int i = 0; i < 64; ++i)
            shaders.shaderdirectlight.setVec3("samples[" + std::to_string(i) + "]", ssdoKernel[i]);
        shaders.shaderdirectlight.setMat4("projection", projection);
        // glm::vec3 lightPosView = glm::vec3(view * glm::vec4(lightPos, 1.0));
        shaders.shaderdirectlight.setVec3("lightcolor", lightColor);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.noiseTexture);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //indireclight
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO2);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.shaderindirectlight.use();
        for (unsigned int i = 0; i < 64; ++i)
            shaders.shaderindirectlight.setVec3("samples[" + std::to_string(i) + "]", ssdoKernel[i]);
        shaders.shaderindirectlight.setMat4("projection", projection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.noiseTexture);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        //blur stage
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoBlurFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.shaderssdoblur.use();
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer2);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);



        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.shaderLightingPass.use();
        glm::vec3 lightPosView = glm::vec3(view * glm::vec4(lightPos, 1.0));
        shaders.shaderLightingPass.setVec3("light.Position", lightPosView);
        shaders.shaderLightingPass.setVec3("light.Color", lightColor);
        const float linear = 0.09f;
        const float quadratic = 0.032f;
        shaders.shaderLightingPass.setFloat("light.Linear", linear);
        shaders.shaderLightingPass.setFloat("light.Quadratic", quadratic);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBufferBlur);
        renderQuad();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}


unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;

unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    //glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; 
    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
