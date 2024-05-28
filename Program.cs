using System.Runtime.InteropServices;
using AutoGen.Core;
using FluentAssertions;
using Phi;
using Phi.Agent;
using Phi.Pipeline;
using TorchSharp;
using static TorchSharp.torch;

var phi2Folder = @"C:\Users\xiaoyuz\source\repos\Phi-2";
var device = "cuda";

if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

var defaultType = ScalarType.BFloat16;
torch.manual_seed(1);

Console.WriteLine("Loading Phi3 from huggingface model weight folder");
var timer = System.Diagnostics.Stopwatch.StartNew();
var model = Phi2ForCasualLM.FromPretrained(phi2Folder, device: device, torchDtype: defaultType, checkPointName: "model.safetensors.index.json");
var tokenizer = Phi2Tokenizer.FromPretrained(phi2Folder);
var pipeline = new CasualLMPipeline(tokenizer, model, device);


timer.Stop();
Console.WriteLine($"Phi3 loaded in {timer.ElapsedMilliseconds / 1000} s");

// agent
var agent = new Phi3Agent(pipeline, "assistant")
    .RegisterPrintMessage();
var question = @"count to 3";
var systemMessage = new TextMessage(Role.System, "You are a helpful AI assistant that always respond in JSON format");
var userMessage = new TextMessage(Role.User, question);
for (int i = 0; i!= 100; ++i)
{
    var reply = await agent.SendAsync(chatHistory: [systemMessage, userMessage]);
}
//// QA Format
//int maxLen = 1024;
//float temperature = 0.0f;
//Console.WriteLine($"QA Format: maxLen: {maxLen} temperature: {temperature}");
//var prompt = "Can you provide ways to eat combinations of bananas and dragonfruits?";
//// wait for user to press enter
//Console.WriteLine($"Prompt: {prompt}");
//Console.WriteLine("Press enter to continue inferencing QA format");

//Console.WriteLine(prompt);
//pipeline.Generate(prompt, maxLen: maxLen, stopSequences: ["<|end|>"], temperature: temperature, device: device);
