using System.Runtime.InteropServices;
using System.Text.Json;
using AutoGen.Core;
using FluentAssertions;
using Phi;
using Phi.Agent;
using Phi.Pipeline;
using TorchSharp;
using static TorchSharp.torch;

var phiFolder = @"C:\Users\xiaoyuz\source\repos\Phi-3-mini-4k-instruct";
var device = "cpu";

if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

var defaultType = ScalarType.Float16;
torch.manual_seed(1);

Console.WriteLine("Loading Phi3 from huggingface model weight folder");
var timer = System.Diagnostics.Stopwatch.StartNew();
var model = Phi3ForCasualLM.FromPretrained(phiFolder, device: device, torchDtype: defaultType, checkPointName: "model.safetensors.index.json");
var tokenizer = LLama2Tokenizer.FromPretrained(phiFolder);

var deviceSizeMap = new Dictionary<string, long>
{
    ["cuda:0"] = 0L * 1024 * 1024 * 1024,
    ["cpu"] = 64L * 1024 * 1024 * 1024,
    ["disk"] = 2L * 1024 * 1024 * 1024 * 1024,
};
//model.ToQuantizedModule();
var deviceMap = model.InferDeviceMapForEachLayer(
    devices: [ "cuda:0", "cpu", "disk" ],
    deviceSizeMapInByte: deviceSizeMap);

var json = JsonSerializer.Serialize(deviceMap, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(json);
model = model.ToDynamicLoadingModel(deviceMap, "cuda:0");

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
