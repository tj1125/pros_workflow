using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

[System.Serializable]
public class RandomizeRequestJson
{
    public int seed;
    public int num_obstacles;
    public int target_object_idx;
    public int lighting_mode;
    public float chassis_slope_deg;
    public float camera_noise;
}

/// <summary>
/// Listens to /env_randomize from ROS, and randomizes the Unity scene.
/// </summary>
public class EnvRandomizer : MonoBehaviour
{
    ROSConnection ros;
    public string randomizeTopic = "/env_randomize";

    [Header("Environment Elements")]
    public Transform targetObject;
    public Transform[] obstacles;
    public Transform robotChassisBase;
    public Light directionalLight;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<StringMsg>(randomizeTopic, OnRandomizeReceived);
        Debug.Log("EnvRandomizer subscribed to " + randomizeTopic);
    }

    void OnRandomizeReceived(StringMsg msg)
    {
        Debug.Log("Received randomize command: " + msg.data);
        RandomizeRequestJson req = JsonUtility.FromJson<RandomizeRequestJson>(msg.data);
        
        // Initialize Unity's random state using the provided seed
        Random.InitState(req.seed);

        ApplyRandomization(req);
    }

    void ApplyRandomization(RandomizeRequestJson req)
    {
        // 1. Target Object Position (randomizing within a local area)
        if (targetObject != null)
        {
            targetObject.localPosition = new Vector3(
                Random.Range(-0.5f, 0.5f),
                targetObject.localPosition.y, // Keep height
                Random.Range(0.2f, 0.8f)      // Randomize distance from robot
            );
        }

        // 2. Obstacles
        if (obstacles != null)
        {
            // Activate num_obstacles, deactivate others
            for (int i = 0; i < obstacles.Length; i++)
            {
                obstacles[i].gameObject.SetActive(i < req.num_obstacles);
                if (i < req.num_obstacles)
                {
                    // Randomize obstacle position
                    obstacles[i].localPosition = new Vector3(
                        Random.Range(-0.5f, 0.5f),
                        obstacles[i].localPosition.y,
                        Random.Range(0.1f, 0.9f)
                    );
                }
            }
        }

        // 3. Slope (Chassis tilt)
        if (robotChassisBase != null)
        {
            // Tilt the base around X and Z axes
            robotChassisBase.localRotation = Quaternion.Euler(
                Random.Range(-req.chassis_slope_deg, req.chassis_slope_deg),
                robotChassisBase.localRotation.eulerAngles.y,
                Random.Range(-req.chassis_slope_deg, req.chassis_slope_deg)
            );
        }

        // 4. Lighting
        if (directionalLight != null)
        {
            // 0=normal, 1=bright, 2=dim, 3=backlit
            switch (req.lighting_mode)
            {
                case 1: directionalLight.intensity = 2.0f; break;
                case 2: directionalLight.intensity = 0.5f; break;
                default: directionalLight.intensity = 1.0f; break;
            }
        }
    }
}
