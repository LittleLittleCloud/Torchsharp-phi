using System.Runtime.InteropServices;
using FluentAssertions;
using TorchSharp;
using static TorchSharp.torch;

var libTorch = "/home/xiaoyuz/llama/venv/lib/python3.8/site-packages/torch/lib/libtorch.so";
NativeLibrary.Load(libTorch);

var phi2Folder = "/home/xiaoyuz/phi-2";
var device = "cuda";

if (device == "cuda")
{
    torch.InitializeDeviceType(DeviceType.CUDA);
    torch.cuda.is_available().Should().BeTrue();
}

torch.manual_seed(100);

// var phi2 = PhiForCasualLM.FromPretrained(phi2Folder, device: "cuda");

var tokenizer = BPETokenizer.FromPretrained(phi2Folder);

var inputIds = tokenizer.Encode("Instruct: A skier slides down a frictionless slope of height 40m and length 80m, what's the skier's speed at the bottom?\nOutput:");

var inputTensor = torch.tensor(inputIds.ToArray(), dtype: ScalarType.Int64, device: device).unsqueeze(0);
var attentionMask = torch.ones_like(inputTensor);

var phi2 = PhiForCasualLM.FromPretrained(phi2Folder, device: device);
(var token, var logits) = phi2.Generate(inputTensor, attentionMask, temperature: 0f, maxLen: 30);

var tokenIds = token[0].to_type(ScalarType.Int32).data<int>().ToArray();
var output = tokenizer.Decode(tokenIds);
Console.WriteLine(output);

