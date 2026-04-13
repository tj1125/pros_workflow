using UnityEngine;
using WebSocketSharp;
using TMPro;
using System.Collections;

public class ConnectRosBridge : MonoBehaviour
{
    public TMP_InputField ipAddressInputField;
    public WebSocket ws;
    public TMP_Text statusText; // 用於顯示狀態的文字
    private bool isConnected = false; // 連線狀態旗標
    private bool isReconnecting = false; // 是否正在嘗試重連
    string errorMsg = "Rosbridge connection failed";

    // ── 事件廣播機制 ──
    public event System.Action<string> OnRosMessageEvent;
    private System.Collections.Generic.Queue<string> messageQueue = new System.Collections.Generic.Queue<string>();

    void Start()
    {
        LoadIPAddress();
        ConnectToRosBridge();
    }

    void Update()
    {
        // 確保 OnRosMessageEvent 在主執行緒 (Main Thread) 執行
        while (true)
        {
            string msg = null;
            lock (messageQueue)
            {
                if (messageQueue.Count > 0)
                    msg = messageQueue.Dequeue();
            }

            if (msg == null) break;
            
            OnRosMessageEvent?.Invoke(msg);
        }
    }

    private void OnDestroy()
    {
        if (ws != null && ws.IsAlive)
        {
            ws.Close();
        }
    }

    private void LoadIPAddress()
    {
        string savedIPAddress = PlayerPrefs.GetString("IPAddress", "");
        if (ipAddressInputField != null && !string.IsNullOrEmpty(savedIPAddress))
        {
            ipAddressInputField.text = savedIPAddress;
        }
    }

    public void SaveIPAddress()
    {
        if (ipAddressInputField != null)
        {
            PlayerPrefs.SetString("IPAddress", ipAddressInputField.text);
            PlayerPrefs.Save();
        }
    }

    public void ConnectToRosBridge()
    {

        string normalMsg = "";
        string protocol = "ws://";
        string ipAddress = ipAddressInputField.text;
        string port = "9090";
        string rosbridgeServerAddress = ipAddress + ":" + port;

        ws = new WebSocket(protocol + rosbridgeServerAddress);

        // 訂閱事件：將不同執行緒收到的訊息放入佇列
        ws.OnMessage += (sender, e) =>
        {
            lock (messageQueue)
            {
                messageQueue.Enqueue(e.Data);
            }
        };

        ws.OnOpen += (sender, e) =>
        {
            isConnected = true;
            isReconnecting = false;
        };

        ws.OnClose += (sender, e) =>
        {
            isConnected = false;
            UpdateStatusText(errorMsg);
        };

        ws.OnError += (sender, e) =>
        {
            isConnected = false;
            UpdateStatusText(errorMsg);
        };

        try
        {
            UpdateStatusText(normalMsg); // 更新狀態為“正在連接”
            ws.Connect(); // 嘗試連接
            if (!ws.IsAlive)
            {
                Debug.LogError("WebSocket is not alive after connection attempt.");
                UpdateStatusText(errorMsg); // 如果連接失敗，顯示錯誤訊息
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"WebSocket connection failed: {ex.Message}");
            isConnected = false;
            UpdateStatusText(errorMsg); // 捕捉異常時更新狀態
        }

    }

    public void Send(string message)
    {
        if (ws == null || ws.ReadyState != WebSocketState.Open)
        {
            Debug.LogError("Failed to send message: WebSocket is not connected.");
            return;
        }

        try
        {
            ws.Send(message);
            Debug.Log("Message sent: " + message);
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to send message: " + ex.Message);
        }
    }

    private void UpdateStatusText(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
    }

}
