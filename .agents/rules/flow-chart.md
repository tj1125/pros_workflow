---
trigger: manual
---

flowchart TB
 subgraph LangGraph["LangGraph 框架"]
    subgraph Docker["🐳 Docker (本機)"]
            Brain["🟦 Brain - VLM\n(Gemini / Ollama)\nLangGraph Stateful Orchestration"]
            NAV["🟦 Nav Agent\n導航節點 (A2A Client)"]
            Grasp["🟦 GraspGen Agent\n抓取生成節點 (A2A Client)"]
            Approach["🟦 Approach Agent\n靠近節點 (A2A Client)"]
            View["🟦 View Agent\n視野調整節點 (A2A Client)"]
    end
    subgraph GPU_Server["⚡ RTX 3090 推論伺服器"]
            InfNAV["🟥 Inference NAV\n(A2A Server)"]
            InfGrasp["🟥 Inference GraspGen\n(A2A Server)"]
            InfView["🟥 Inference View\nVLM + RL\n(A2A Server)"]
    end
    subgraph Unity["🤖 Unity 環境資訊"]
            UnityNode["🟩 Unity Env. Image\n(數位孿生環境)"]
            InfoBus["🟩 Unity Info Bus\n(msg or image...)\nLangGraph State"]
    end
        Brain -- 決定要執行哪個Agent --> NAV & Grasp & Approach & View
        InfoBus <-- ROS / Rosbridge --> NAV & Grasp & Approach & View
        NAV -- 回覆Agent輸出的資訊 --> Brain
        Grasp -- 回覆Agent輸出的資訊 --> Brain
        Approach -- 回覆Agent輸出的資訊 --> Brain
        View -- 回覆Agent輸出的資訊 --> Brain
        NAV -- A2A HTTPS JSON --> InfNAV
        InfNAV -- 推論結果 --> NAV
        Grasp -- A2A HTTPS JSON --> InfGrasp
        InfGrasp -- 推論結果 --> Grasp
        View -- A2A HTTPS JSON --> InfView
        InfView -- 推論結果 --> View
        UnityNode -- ROS / Rosbridge --> Brain
end