using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Sensor;

/// <summary>
/// Publishes RGB, Depth, Joint States, and IMU data to Rosbridge for View Agent.
/// Requires Unity Robotics Hub ROSTCPConnector.
/// </summary>
public class ViewAgentSensorPublisher : MonoBehaviour
{
    ROSConnection ros;

    [Header("ROS Topics")]
    public string rgbTopic = "/camera/color/image_raw";
    public string depthTopic = "/camera/depth/image_raw";
    public string jointTopic = "/joint_states";
    public string imuTopic = "/imu/data";

    [Header("Sensors")]
    public Camera rgbCamera;
    public Camera depthCamera;
    public Transform imuTransform;
    
    [Header("Robot Joints (6-DOF)")]
    public List<ArticulationBody> armJoints;

    [Header("Publish Settings")]
    public float publishRateHz = 10f;
    private float timeElapsed;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(rgbTopic);
        ros.RegisterPublisher<ImageMsg>(depthTopic);
        ros.RegisterPublisher<JointStateMsg>(jointTopic);
        ros.RegisterPublisher<ImuMsg>(imuTopic);
    }

    void Update()
    {
        timeElapsed += Time.deltaTime;
        float publishInterval = 1f / publishRateHz;
        
        if (timeElapsed > publishInterval)
        {
            PublishAllSensors();
            timeElapsed = 0f;
        }
    }

    void PublishAllSensors()
    {
        PublishRGBDepth();
        PublishJointStates();
        PublishIMU();
    }

    void PublishRGBDepth()
    {
        // 1. Render RGB to ImageMsg
        // NOTE: In an actual project, use RenderTextures to efficiently extract byte arrays.
        // ImageMsg rgbMsg = new ImageMsg { ... };
        // ros.Publish(rgbTopic, rgbMsg);

        // 2. Render Depth to ImageMsg (float32 or uint16 converted)
        // ImageMsg depthMsg = new ImageMsg { ... };
        // ros.Publish(depthTopic, depthMsg);
    }

    void PublishJointStates()
    {
        if (armJoints == null || armJoints.Count == 0) return;

        JointStateMsg msg = new JointStateMsg();
        msg.name = new string[armJoints.Count];
        msg.position = new double[armJoints.Count];

        for (int i = 0; i < armJoints.Count; i++)
        {
            msg.name[i] = "joint" + (i + 1);
            // ArticulationBody.jointPosition returns an array (1 element for revolute)
            msg.position[i] = armJoints[i].jointPosition[0]; 
        }

        ros.Publish(jointTopic, msg);
    }

    void PublishIMU()
    {
        if (imuTransform == null) return;

        ImuMsg msg = new ImuMsg();
        // Set linear acceleration and angular velocity fields from Physics
        // msg.linear_acceleration = ...
        // msg.angular_velocity = ...
        
        ros.Publish(imuTopic, msg);
    }
}
