using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

[System.Serializable]
public class PhysicsMetricsJson
{
    public float occlusion_rate;
    public float centering_score;
    public float chassis_stability;
}

/// <summary>
/// Collects physics metrics (occlusion, centering, stability) and publishes them to ROS as a JSON string.
/// </summary>
public class PhysicsMetricPublisher : MonoBehaviour
{
    ROSConnection ros;
    public string metricsTopic = "/view_agent/physics_metrics";
    public float publishRateHz = 10f;
    private float timeElapsed;

    [Header("References")]
    public OcclusionRaycastSensor occlusionSensor;
    public Camera viewCamera;
    public Transform targetObject;
    public Transform robotChassis; // To calculate IMU stability

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>(metricsTopic);
    }

    void Update()
    {
        timeElapsed += Time.deltaTime;
        float publishInterval = 1f / publishRateHz;
        
        if (timeElapsed > publishInterval)
        {
            PublishMetrics();
            timeElapsed = 0f;
        }
    }

    void PublishMetrics()
    {
        PhysicsMetricsJson data = new PhysicsMetricsJson();
        
        // 1. Occlusion Rate
        if (occlusionSensor != null)
        {
            data.occlusion_rate = occlusionSensor.currentOcclusionRate;
        }

        // 2. Centering Score
        if (viewCamera != null && targetObject != null)
        {
            Vector3 viewportPos = viewCamera.WorldToViewportPoint(targetObject.position);
            // viewportPos.x/y are [0,1]. Center is (0.5, 0.5)
            // Distance from center max is ~0.707. Normalize score to [0,1] where 1 is center.
            float distFromCenter = Vector2.Distance(new Vector2(viewportPos.x, viewportPos.y), new Vector2(0.5f, 0.5f));
            data.centering_score = Mathf.Clamp01(1.0f - (distFromCenter * 2f));
            
            // If it's behind the camera, override centering to 0
            if (viewportPos.z < 0) data.centering_score = 0;
        }

        // 3. Chassis Stability
        if (robotChassis != null)
        {
            // Simple stability metric: dot product of chassis up vector with world up vector
            // 1.0 = perfectly upright, < 1.0 = tilted
            float tilt = Vector3.Dot(robotChassis.up, Vector3.up);
            data.chassis_stability = Mathf.Clamp01(tilt);
        }
        else
        {
            data.chassis_stability = 1.0f;
        }

        // Convert to JSON and publish
        string jsonStr = JsonUtility.ToJson(data);
        StringMsg msg = new StringMsg(jsonStr);
        ros.Publish(metricsTopic, msg);
    }
}
