using System;
using System.Collections;
using System.IO;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using Hjg.Pngcs;

/// <summary>
/// ROS 2 on-demand camera capture over rosbridge.
/// Service type is std_srvs/srv/Trigger.
/// Each trigger publishes one RGB image (and optional depth image) to topics.
/// </summary>
[RequireComponent(typeof(Camera))]
public class RosCameraService : MonoBehaviour
{
    [Header("Camera References")]
    [SerializeField] private Camera rgbCamera;
    [SerializeField] private Camera depthCamera;
    [SerializeField] private Shader replacementShader;

    [Header("ROS Bridge")]
    [SerializeField] private ConnectRosBridge connectRos;

    [Header("Service Settings")]
    [SerializeField] private string serviceName = "/capture_image";
    [SerializeField] private string frameIdOverride;

    [Header("Publish Topics")]
    [SerializeField] private string rgbTopic = "/capture_image/rgb/compressed";
    [SerializeField] private bool publishDepth = false;
    [SerializeField] private string depthTopic = "/capture_image/depth/compressed";

    [Header("Image Resolution")]
    [SerializeField] private int imageWidth = 1280;
    [SerializeField] private int imageHeight = 720;

    [Range(0, 6)]
    [SerializeField] private int depthShaderOutputMode = 5;

    private string CameraName => gameObject.name;

    private RenderTexture _rgbRT;
    private RenderTexture _depthRT;
    private Texture2D _rgbTex;
    private ushort[] _depthBuffer16;

    private bool _rgbReadbackInFlight;
    private bool _depthReadbackInFlight;

    private bool _pendingRequest;
    private string _pendingId;

    private byte[] _rgbBytes;
    private byte[] _depthBytes;

    private bool _rosAdvertised;

    private void Awake()
    {
        if (!rgbCamera)
            rgbCamera = GetComponent<Camera>();
        if (!depthCamera)
            depthCamera = rgbCamera;
    }

    private void OnEnable()
    {
        if (connectRos != null)
            connectRos.OnRosMessageEvent += OnRosMessage;
    }

    private void OnDisable()
    {
        if (connectRos != null)
            connectRos.OnRosMessageEvent -= OnRosMessage;
    }

    private void Start()
    {
        if (!rgbCamera)
        {
            Debug.LogError($"[RosCameraService] RGB camera missing on {CameraName}, disabling.");
            enabled = false;
            return;
        }

        if (!replacementShader)
            replacementShader = Shader.Find("Hidden/UberReplacementPhysical");

        _rgbRT = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        _rgbRT.Create();
        _rgbTex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);

        _depthRT = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.RFloat);
        _depthRT.Create();
        _depthBuffer16 = new ushort[imageWidth * imageHeight];

        rgbCamera.usePhysicalProperties = false;
        rgbCamera.rect = new Rect(0, 0, 1, 1);

        ApplyCameraRouteNames();
        AdvertiseRosEndpoints();
    }

    private void ApplyCameraRouteNames()
    {
        string cameraKey = BuildCameraRouteKey(CameraName);
        serviceName = $"/capture_image/{cameraKey}";
        rgbTopic = $"/capture_image/{cameraKey}/rgb/compressed";
        depthTopic = $"/capture_image/{cameraKey}/depth/compressed";

        Debug.Log($"[RosCameraService] '{CameraName}' routes: service={serviceName}, rgb={rgbTopic}, depth={depthTopic}");
    }

    private static string BuildCameraRouteKey(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return "camera";

        string trimmed = name.Trim();
        var buffer = new System.Text.StringBuilder(trimmed.Length);
        foreach (char ch in trimmed)
        {
            if (char.IsLetterOrDigit(ch) || ch == '_')
                buffer.Append(ch);
            else
                buffer.Append('_');
        }

        string key = buffer.ToString();
        if (key.Length > 0 && char.IsDigit(key[0]))
            key = $"cam_{key}";

        return key.Length == 0 ? "camera" : key;
    }

    private void OnDestroy()
    {
        if (_rgbRT)
        {
            _rgbRT.Release();
            Destroy(_rgbRT);
        }

        if (_depthRT)
        {
            _depthRT.Release();
            Destroy(_depthRT);
        }

        if (_rgbTex)
            Destroy(_rgbTex);
    }

    private void OnRosMessage(string rawJson)
    {
        if (!rawJson.Contains("\"call_service\""))
            return;

        string targetService = ExtractJsonString(rawJson, "service");
        if (targetService != serviceName)
            return;

        if (_pendingRequest)
        {
            Debug.LogWarning($"[RosCameraService] '{CameraName}' request in progress, ignoring.");
            return;
        }

        string callId = ExtractJsonString(rawJson, "id");
        if (string.IsNullOrEmpty(callId))
            callId = Guid.NewGuid().ToString("N");

        _pendingId = callId;
        _pendingRequest = true;

        Debug.Log($"[RosCameraService] Trigger request received (id={callId}).");
        StartCoroutine(CaptureAndRespond());
    }

    private IEnumerator CaptureAndRespond()
    {
        _rgbBytes = null;
        _depthBytes = null;

        _rgbReadbackInFlight = true;
        rgbCamera.targetTexture = _rgbRT;
        rgbCamera.Render();
        rgbCamera.targetTexture = null;
        AsyncGPUReadback.Request(_rgbRT, 0, TextureFormat.RGBA32, OnRgbReadback);

        while (_rgbReadbackInFlight)
            yield return null;

        if (_rgbBytes == null)
        {
            SendErrorResponse(_pendingId, "RGB GPU readback failed.");
            _pendingRequest = false;
            yield break;
        }

        if (publishDepth)
        {
            if (!replacementShader)
            {
                SendErrorResponse(_pendingId, "Depth shader missing.");
                _pendingRequest = false;
                yield break;
            }

            _depthReadbackInFlight = true;
            Shader.SetGlobalFloat("_OutputMode", depthShaderOutputMode);
            depthCamera.targetTexture = _depthRT;
            depthCamera.RenderWithShader(replacementShader, "");
            depthCamera.targetTexture = null;
            AsyncGPUReadback.Request(_depthRT, 0, OnDepthReadback);

            while (_depthReadbackInFlight)
                yield return null;

            if (_depthBytes == null)
            {
                SendErrorResponse(_pendingId, "Depth GPU readback failed.");
                _pendingRequest = false;
                yield break;
            }
        }

        if (!_rosAdvertised)
            AdvertiseRosEndpoints();

        string frameId = string.IsNullOrEmpty(frameIdOverride) ? gameObject.name : frameIdOverride;
        (long sec, long nanosec) = GetRosTime();

        PublishCompressedImage(rgbTopic, "jpeg", _rgbBytes, frameId, sec, nanosec);
        if (publishDepth && _depthBytes != null)
            PublishCompressedImage(depthTopic, "png", _depthBytes, frameId, sec, nanosec);

        string responseMessage = publishDepth
            ? $"Captured {CameraName} and published RGB+depth topics."
            : $"Captured {CameraName} and published RGB topic.";

        SendSuccessResponse(_pendingId, responseMessage);
        _pendingRequest = false;
    }

    private void OnRgbReadback(AsyncGPUReadbackRequest request)
    {
        _rgbReadbackInFlight = false;

        if (request.hasError)
        {
            Debug.LogError("[RosCameraService] RGB readback error.");
            return;
        }

        NativeArray<byte> raw = request.GetData<byte>();
        _rgbTex.LoadRawTextureData(raw);
        _rgbTex.Apply();
        _rgbBytes = _rgbTex.EncodeToJPG(100);
    }

    private void OnDepthReadback(AsyncGPUReadbackRequest request)
    {
        _depthReadbackInFlight = false;

        if (request.hasError)
        {
            Debug.LogError("[RosCameraService] Depth readback error.");
            return;
        }

        NativeArray<float> data = request.GetData<float>();

        if (_depthBuffer16 == null || _depthBuffer16.Length != data.Length)
            _depthBuffer16 = new ushort[data.Length];

        for (int i = 0; i < data.Length; i++)
            _depthBuffer16[i] = (ushort)Mathf.Clamp(data[i] * 1000f, 0f, 65535f);

        _depthBytes = EncodeTo16BitPng(_depthBuffer16, imageWidth, imageHeight);
    }

    private void AdvertiseRosEndpoints()
    {
        if (!IsRosReady())
        {
            Invoke(nameof(AdvertiseRosEndpoints), 1f);
            return;
        }

        connectRos.ws.Send($@"{{
            ""op"": ""advertise_service"",
            ""service"": ""{EscapeJson(serviceName)}"",
            ""type"": ""std_srvs/srv/Trigger""
        }}");

        AdvertiseTopic(rgbTopic, "sensor_msgs/CompressedImage");
        if (publishDepth)
            AdvertiseTopic(depthTopic, "sensor_msgs/CompressedImage");

        _rosAdvertised = true;
        Debug.Log($"[RosCameraService] '{CameraName}' advertised Trigger service {serviceName}.");
    }

    private void AdvertiseTopic(string topicName, string topicType)
    {
        connectRos.ws.Send($@"{{
            ""op"": ""advertise"",
            ""topic"": ""{EscapeJson(topicName)}"",
            ""type"": ""{EscapeJson(topicType)}""
        }}");
    }

    private void PublishCompressedImage(
        string topicName,
        string format,
        byte[] payload,
        string frameId,
        long sec,
        long nanosec)
    {
        if (!IsRosReady())
        {
            Debug.LogError("[RosCameraService] rosbridge not ready while publishing image.");
            return;
        }

        string base64 = Convert.ToBase64String(payload);
        connectRos.ws.Send($@"{{
            ""op"": ""publish"",
            ""topic"": ""{EscapeJson(topicName)}"",
            ""msg"": {{
                ""header"": {{
                    ""stamp"": {{""sec"": {sec}, ""nanosec"": {nanosec}}},
                    ""frame_id"": ""{EscapeJson(frameId)}""
                }},
                ""format"": ""{EscapeJson(format)}"",
                ""data"": ""{base64}""
            }}
        }}");
    }

    private void SendSuccessResponse(string callId, string message)
    {
        if (!IsRosReady())
            return;

        connectRos.ws.Send($@"{{
            ""op"": ""service_response"",
            ""service"": ""{EscapeJson(serviceName)}"",
            ""id"": ""{EscapeJson(callId)}"",
            ""result"": true,
            ""values"": {{
                ""success"": true,
                ""message"": ""{EscapeJson(message)}""
            }}
        }}");
    }

    private void SendErrorResponse(string callId, string errorMessage)
    {
        Debug.LogError($"[RosCameraService] Error: {errorMessage}");

        if (!IsRosReady())
            return;

        connectRos.ws.Send($@"{{
            ""op"": ""service_response"",
            ""service"": ""{EscapeJson(serviceName)}"",
            ""id"": ""{EscapeJson(callId)}"",
            ""result"": false,
            ""values"": {{
                ""success"": false,
                ""message"": ""{EscapeJson(errorMessage)}""
            }}
        }}");
    }

    private bool IsRosReady()
    {
        return connectRos != null
            && connectRos.ws != null
            && connectRos.ws.ReadyState == WebSocketSharp.WebSocketState.Open;
    }

    private static (long sec, long nanosec) GetRosTime()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        long sec = now.ToUnixTimeSeconds();
        long nanosec = (now.ToUnixTimeMilliseconds() % 1000) * 1000000;
        return (sec, nanosec);
    }

    private static string ExtractJsonString(string json, string key)
    {
        string search = $"\"{key}\"";
        int keyIdx = json.IndexOf(search, StringComparison.Ordinal);
        if (keyIdx < 0)
            return string.Empty;

        int colonIdx = json.IndexOf(':', keyIdx + search.Length);
        if (colonIdx < 0)
            return string.Empty;

        int q1 = json.IndexOf('"', colonIdx + 1);
        if (q1 < 0)
            return string.Empty;

        int q2 = json.IndexOf('"', q1 + 1);
        if (q2 < 0)
            return string.Empty;

        return json.Substring(q1 + 1, q2 - q1 - 1);
    }

    private static string EscapeJson(string text)
    {
        return text?.Replace("\\", "\\\\").Replace("\"", "\\\"") ?? string.Empty;
    }

    private static byte[] EncodeTo16BitPng(ushort[] rawValues, int width, int height)
    {
        var info = new ImageInfo(width, height, 16, false, true, false);
        using var ms = new MemoryStream();
        var writer = new PngWriter(ms, info, string.Empty);
        int[] row = new int[width];

        for (int r = 0; r < height; r++)
        {
            int offset = r * width;
            for (int c = 0; c < width; c++)
                row[c] = rawValues[offset + c];

            writer.WriteRowInt(row, r);
        }

        writer.End();
        return ms.ToArray();
    }
}
