using System.Runtime.InteropServices;
using FluentAssertions;
using TorchSharp;
using static TorchSharp.torch;

// Dynamic loading libtorch because Cuda 12 only support GPU driver >= 520
// And I can't upgrade GPU driver because it's a cloud machine.

var phi2Folder = @"C:\Users\xiaoyuz\source\repos\phi-2";
var device = "cuda";

if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

var defaultType = ScalarType.Float16;
torch.set_default_dtype(defaultType);
torch.manual_seed(1);

Console.WriteLine("Loading Phi2 from huggingface model weight folder");
var timer = System.Diagnostics.Stopwatch.StartNew();
var phi2 = PhiForCasualLM.FromPretrained(phi2Folder, device: device, defaultDType: defaultType, checkPointName: "model.safetensors.index.json");

timer.Stop();
Console.WriteLine($"Phi2 loaded in {timer.ElapsedMilliseconds / 1000} s");

// QA Format
int maxLen = 512;
float temperature = 0.0f;
Console.WriteLine($"QA Format: maxLen: {maxLen} temperature: {temperature}");
var prompt = "Instruct: A skier slides down a frictionless slope of height 40m and length 80m, what's the skier's speed at the bottom, think step by step.\nOutput:";
// wait for user to press enter
Console.WriteLine($"Prompt: {prompt}");
Console.WriteLine("Press enter to continue inferencing QA format");
Console.ReadLine();

Console.WriteLine(prompt);
phi2.Generate(prompt, maxLen: maxLen, temperature: temperature);