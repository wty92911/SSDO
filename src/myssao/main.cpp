#include <config.h>
//#include <glad/glad.h>
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>


#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iostream>
#include <random>
#include <cmath>
#include <map>

#include "utils.hpp"

const int MAX_SAMPLE = 256;
const int MAX_GK_SIZE = 8; // array size is (size * 2 + 1)

float PI = 3.1415926;
float sigma = 1.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path, bool gammaCorrection);
void renderQuad();



float lastX = 0.0;
float lastY = 0.0;
bool firstMouse = true;
bool rightMouseButtonPressed = false;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

Camera camera(glm::vec3(0.0F));


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

struct ModelInfo{
    string name;
    Model model;
    float scale;
};

class Config{
    public:
    map<string, string> cfgfile;

    vector<ModelInfo> models;
    int selected_model = 0;

    bool config_modified = true;
    bool use_ssdo = true;
    float sphere_radius = 0.5;
    int sample_count = 64;

    int gaussian_size = 1;
    float gaussian_sigma = 1.0;
    float gaussian_offset_factor = 1.0;

    float direct_strength = 1.0;
    float indirect_strength = 1.0;
    float scale = 1.0;

    float light_x = 0.0, light_y = 0.0, light_z = 0.0;
    float light_r = 0.5, light_g = 0.5, light_b = 0.5;

    bool light_follow = false;
    
    bool requires_remake_kernel = true;

    glm::vec3 get_light_pos(){
        return glm::vec3(light_x, light_y, light_z);
    }

    glm::vec3 get_light_color(){
        return glm::vec3(light_r, light_g, light_b);
    }

    void load_initial(){
        models.clear();
        vector<string> mm = split(cfgfile["models"], ",");
        for(string s : mm){
            string path = cfgfile["model_" + s + "_path"];
            string scale = cfgfile["model_" + s + "_scale"];
            models.push_back(ModelInfo{s, Model(path.c_str()), (float)atof(scale.c_str())});
        }
    }
}config;


std::vector<glm::vec3> ssdoKernel = get_ssdo_kernel(MAX_SAMPLE);
std::vector<glm::vec3> ssdoNoise = get_ssdo_noise(16);
float gaussian[MAX_GK_SIZE * 2 + 1][MAX_GK_SIZE * 2 + 1];

void remake_kernels(){
    ssdoKernel = get_ssdo_kernel(config.sample_count);

    int gs = config.gaussian_size;
    float sigma = config.gaussian_sigma;
    for (int i = 0;i < gs * 2 + 1;i++) {
        for (int j = 0;j < gs * 2 + 1;j++) {
            gaussian[i][j] = exp(-float((i - gs) * (i - gs) + (j - gs) * (j - gs)) / (2.0f * sigma * sigma)) / (2.0f * PI * sigma * sigma);
        }
    }
}

void remake(int width, int height){
    
    glGenFramebuffers(1, &buf.gBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, buf.gBuffer);
    
    glGenTextures(1, &buf.gPosition);
    glBindTexture(GL_TEXTURE_2D, buf.gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.gPosition, 0);
    // normal 
    glGenTextures(1, &buf.gNormal);
    glBindTexture(GL_TEXTURE_2D, buf.gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buf.gNormal, 0);
    // color
    glGenTextures(1, &buf.gAlbedo);
    glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buf.gAlbedo, 0);

    
    glDrawBuffers(3, buf.attachments);

    
    glGenRenderbuffers(1, &buf.rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, buf.rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buf.rboDepth);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glGenFramebuffers(1, &buf.ssdoFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO);
    
    // SSDO color buffer
    glGenTextures(1, &buf.ssdoColorBuffer);
    glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBuffer2, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "SSDO Framebuffer not complete!" << std::endl;
    // make some blur
    glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoBlurFBO);
    glGenTextures(1, &buf.ssdoColorBufferBlur);
    glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBufferBlur);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
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

class MyImGui{
    Config* cfg;
    char modelnames[1024];
public:

    void init(Config* _cfg, GLFWwindow *window){
        cfg = _cfg;
        
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330 core");

        memset(modelnames, 0, sizeof(modelnames));
        int p = 0;
        for(ModelInfo mi : config.models){
            strcpy(modelnames + p, mi.name.c_str());
            p += strlen(mi.name.c_str()) + 1;
        }
    }

    void pre_tick(){
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void tick(){
        draw_ui();
    }

    void draw_ui(){
        ImGui::Combo("Model", &config.selected_model, modelnames);

        ImGui::SliderFloat("Light X", &config.light_x, -10.0F, 10.0F, "%.02f", 1.0F);
        ImGui::SliderFloat("Y", &config.light_y, -10.0F, 10.0F, "%.02f", 1.0F);
        ImGui::SliderFloat("Z", &config.light_z, -10.0F, 10.0F, "%.02f", 1.0F);

        if(ImGui::Button(config.light_follow ? "Light Follows Camera: On" : "Light Follows Camera: Off")){
            config.light_follow = !config.light_follow;
        }

        ImGui::SliderFloat("Light R", &config.light_r, 0.0F, 1.0F, "%.02f", 1.0F);
        ImGui::SliderFloat("G", &config.light_g, 0.0F, 1.0F, "%.02f", 1.0F);
        ImGui::SliderFloat("B", &config.light_b, 0.0F, 1.0F, "%.02f", 1.0F);

        config.requires_remake_kernel |= ImGui::SliderFloat("Sample Radius", &config.sphere_radius, 0.0F, 5.0F, "%.02f", 1.0F);
        config.requires_remake_kernel |= ImGui::SliderInt("Sample Count", &config.sample_count, 1, 256);
        
        config.requires_remake_kernel |= ImGui::SliderInt("Blur Kernel Size", &config.gaussian_size, 1, 8);
        config.requires_remake_kernel |= ImGui::SliderFloat("Blur Kernel Sigma", &config.gaussian_sigma, 0.0, 5.0, "%.02f", 1.0F);
        config.requires_remake_kernel |= ImGui::SliderFloat("Blur Kernel Offset Factor", &config.gaussian_offset_factor, 0.0, 5.0, "%.02f", 1.0F);

        ImGui::SliderFloat("Direct Light Strength", &config.direct_strength, 0.0F, 4.0F, "%.02f", 1.0F);
        ImGui::SliderFloat("Inirect Light Strength", &config.indirect_strength, 0.0F, 4.0F, "%.02f", 1.0F);
    }

    void after_tick(){
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

}imgui;

class Shaders{
public:
    Shader * shaderGeometryPass;
    Shader * shaderdirectlight;
    Shader * shaderindirectlight;
    Shader * shaderssdoblur;
    Shader * shaderLightingPass;   
}shaders;

void set_uniform_shader_params(){
    shaders.shaderdirectlight->use();
    shaders.shaderdirectlight->setInt("gPosition", 0);
    shaders.shaderdirectlight->setInt("gNormal", 1);
    shaders.shaderdirectlight->setInt("gAlbedo", 2);
    shaders.shaderdirectlight->setInt("texNoise", 3);
    shaders.shaderindirectlight->use();
    shaders.shaderindirectlight->setInt("gPosition", 0);
    shaders.shaderindirectlight->setInt("gNormal", 1);
    shaders.shaderindirectlight->setInt("gAlbedo", 2);
    shaders.shaderindirectlight->setInt("texNoise", 3);
    shaders.shaderindirectlight->setInt("directLight", 4);
    shaders.shaderssdoblur->use();
    shaders.shaderssdoblur->setInt("ssdoInput", 0);
    shaders.shaderLightingPass->use();
    shaders.shaderLightingPass->setInt("gPosition", 0);
    shaders.shaderLightingPass->setInt("gNormal", 1);
    shaders.shaderLightingPass->setInt("gAlbedo", 2);
    shaders.shaderLightingPass->setInt("ssdo", 3);

    
}


int main(){

    
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif


    GLFWwindow* window = glfwCreateWindow(800, 600, "SSDO", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
    //     std::cout << "Failed to initialize GLAD" << std::endl;
    //     return -1;
    // }

    if (glewInit() != GLEW_OK) {
         std::cout << "Failed to initialize GLEW" << std::endl;
         return -1;
    }

    config.cfgfile = load_config("config.txt");
    config.load_initial();

    imgui.init(&config, window);
    
    glEnable(GL_DEPTH_TEST);

    const std::string shader_dir = RES_DIR + "/shaders_ssdo";
    shaders.shaderGeometryPass = new Shader((shader_dir + "/geometry.vs").c_str(), (shader_dir + "/geometry.fs").c_str());
    shaders.shaderdirectlight = new Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/direct_light.fs").c_str());
    shaders.shaderindirectlight = new Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/indirect_light.fs").c_str());
    shaders.shaderssdoblur = new Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/blur.fs").c_str());
    shaders.shaderLightingPass = new Shader((shader_dir + "/ssdo.vs").c_str(), (shader_dir + "/lighting.fs").c_str()); 

    // std::cout << model_loaded.textures_loaded.size() << std::endl;

    set_uniform_shader_params();

    int last_width = -1, last_height = -1;

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);

        imgui.pre_tick();

        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glViewport(0, 0, width, height);


        if(width != last_width || height != last_height){
            last_width = width;
            last_height = height;
                
            remake(width, height);
        }

        if(config.requires_remake_kernel){
            remake_kernels();
            config.requires_remake_kernel = false;
        }

        if(config.light_follow){
            config.light_x = camera.Position.x;
            config.light_y = camera.Position.y;
            config.light_z = camera.Position.z;
        }
        
        glm::vec3 lightPos = config.get_light_pos();
        glm::vec3 lightColor = config.get_light_color();
            
        glBindFramebuffer(GL_FRAMEBUFFER, buf.gBuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)width / (float)height, 0.1f, 50.0f);
        glm::mat4 view = camera.GetViewMatrix();
        shaders.shaderGeometryPass->use();
        shaders.shaderGeometryPass->setMat4("projection", projection);
        shaders.shaderGeometryPass->setMat4("view", view);
        shaders.shaderGeometryPass->setInt("invertedNormals", 0);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -0.5f, 0.0));
        model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0.0, 1.0, 0.0));
        model = glm::scale(model, glm::vec3(config.models[config.selected_model].scale));
        shaders.shaderGeometryPass->setMat4("model", model);
        config.models[config.selected_model].model.Draw(*shaders.shaderGeometryPass);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //directlight
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        shaders.shaderdirectlight->use();
        for (unsigned int i = 0; i < config.sample_count; ++i)
            shaders.shaderdirectlight->setVec3("samples[" + std::to_string(i) + "]", ssdoKernel[i]);
        shaders.shaderdirectlight->setMat4("projection", projection);
        shaders.shaderdirectlight->setInt("samplecount", config.sample_count);
        // glm::vec3 lightPosView = glm::vec3(view * glm::vec4(lightPos, 1.0));
        shaders.shaderdirectlight->setVec3("lightcolor", lightColor);
        shaders.shaderdirectlight->setFloat("radius", config.sphere_radius);
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
        shaders.shaderindirectlight->use();
        for (unsigned int i = 0; i < config.sample_count; ++i)
            shaders.shaderindirectlight->setVec3("samples[" + std::to_string(i) + "]", ssdoKernel[i]);
        shaders.shaderindirectlight->setInt("samplecount", config.sample_count);
        shaders.shaderindirectlight->setMat4("projection", projection);
        shaders.shaderindirectlight->setFloat("direct_strength", config.direct_strength);
        shaders.shaderindirectlight->setFloat("indirect_strength", config.indirect_strength);
        shaders.shaderindirectlight->setFloat("radius", config.sphere_radius);
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
        shaders.shaderssdoblur->use();
        int gs = config.gaussian_size;
        for (unsigned int i = 0;i < gs * 2 + 1;i++) {
            for (unsigned int j = 0;j < gs * 2 + 1;j++) {
                shaders.shaderssdoblur->setFloat("gauss[" + std::to_string(i * (gs * 2 + 1) + j) + "]", gaussian[i][j]);
            }
        }
        shaders.shaderssdoblur->setInt("gs", gs);
        shaders.shaderssdoblur->setFloat("offset_factor", config.gaussian_offset_factor);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer2);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);



        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.shaderLightingPass->use();
        glm::vec3 lightPosView = glm::vec3(view * glm::vec4(lightPos, 1.0));
        shaders.shaderLightingPass->setVec3("light.Position", lightPosView);
        shaders.shaderLightingPass->setVec3("light.Color", lightColor);
        const float linear = 0.09f;
        const float quadratic = 0.032f;
        shaders.shaderLightingPass->setFloat("light.Linear", linear);
        shaders.shaderLightingPass->setFloat("light.Quadratic", quadratic);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBufferBlur);
        renderQuad();

        imgui.draw_ui();
        imgui.after_tick();
        
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
void renderQuad() {
    if (quadVAO == 0) {
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
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    //glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn){
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    if (rightMouseButtonPressed)
    {
        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS)
        {
            rightMouseButtonPressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            rightMouseButtonPressed = false;
        }
    }
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
