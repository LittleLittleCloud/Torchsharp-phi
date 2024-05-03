using System.Runtime.InteropServices;
using FluentAssertions;
using Phi;
using Phi.Pipeline;
using TorchSharp;
using static TorchSharp.torch;

// Dynamic loading libtorch because Cuda 12 only support GPU driver >= 520
// And I can't upgrade GPU driver because it's a cloud machine.

var phi2Folder = @"C:\Users\xiaoyuz\source\repos\Phi-3-mini-128k-instruct";
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
var model = Phi3ForCasualLM.FromPretrained(phi2Folder, device: device, torchDtype: defaultType, checkPointName: "model.safetensors.index.json");
var tokenizer = LLama2Tokenizer.FromPretrained(phi2Folder);
var pipeline = new CasualLMPipeline(tokenizer, model, device);


timer.Stop();
Console.WriteLine($"Phi3 loaded in {timer.ElapsedMilliseconds / 1000} s");

// QA Format
int maxLen = 1024;
float temperature = 0.0f;
Console.WriteLine($"QA Format: maxLen: {maxLen} temperature: {temperature}");
var prompt = "Can you provide ways to eat combinations of bananas and dragonfruits?";
// wait for user to press enter
Console.WriteLine($"Prompt: {prompt}");
Console.WriteLine("Press enter to continue inferencing QA format");

Console.WriteLine(prompt);
pipeline.Generate(prompt, maxLen: maxLen, stopSequences: ["<|end|>"], temperature: temperature, device: device);
