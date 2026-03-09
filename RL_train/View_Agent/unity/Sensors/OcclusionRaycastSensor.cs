using UnityEngine;

/// <summary>
/// Calculates how much of the target object is occluded from the camera's perspective.
/// </summary>
public class OcclusionRaycastSensor : MonoBehaviour
{
    [Header("Settings")]
    public Camera viewCamera;
    public Transform targetObject;
    public int raysPerDimension = 10;
    public LayerMask obstacleLayer;

    [Tooltip("The current calculated occlusion rate (0.0 = fully visible, 1.0 = fully occluded).")]
    public float currentOcclusionRate = 0.0f;

    void Update()
    {
        CalculateOcclusion();
    }

    void CalculateOcclusion()
    {
        if (viewCamera == null || targetObject == null) return;

        // Bounding box of the target object
        Collider targetCollider = targetObject.GetComponent<Collider>();
        if (targetCollider == null) return;

        Bounds bounds = targetCollider.bounds;
        
        int totalRays = 0;
        int hitObstacles = 0;

        // Grid-based raycasting towards the bounding box
        for (int i = 0; i < raysPerDimension; i++)
        {
            for (int j = 0; j < raysPerDimension; j++)
            {
                for (int k = 0; k < raysPerDimension; k++)
                {
                    float tu = i / (float)(raysPerDimension - 1);
                    float tv = j / (float)(raysPerDimension - 1);
                    float tw = k / (float)(raysPerDimension - 1);

                    Vector3 targetPoint = new Vector3(
                        Mathf.Lerp(bounds.min.x, bounds.max.x, tu),
                        Mathf.Lerp(bounds.min.y, bounds.max.y, tv),
                        Mathf.Lerp(bounds.min.z, bounds.max.z, tw)
                    );

                    Vector3 origin = viewCamera.transform.position;
                    Vector3 direction = targetPoint - origin;
                    float distance = direction.magnitude;

                    totalRays++;

                    // Raycast check: if it hits something on the obstacle layer before reaching the target
                    if (Physics.Raycast(origin, direction, out RaycastHit hit, distance, obstacleLayer))
                    {
                        hitObstacles++;
                    }
                }
            }
        }

        if (totalRays > 0)
        {
            currentOcclusionRate = (float)hitObstacles / totalRays;
        }
    }
}
