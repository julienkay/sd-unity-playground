using UnityEngine;

namespace Doji.AI.Depth.Samples {

    /// <summary>
    /// Draws instanced points from a depth map
    /// </summary>
    public class DepthPointRenderer : MonoBehaviour {

        public Texture Source {
            get {
                return _source;
            }
            set {
                _source = value;
                if (Material != null) {
                    Material.SetTexture("_MainTex", value);
                }
            }
        }
        [SerializeField]
        private Texture _source;

        public Texture Depth {
            get {
                return _depth;
            }
            set {
                _depth = value;
                if (Material != null) {
                    Material.SetTexture("_Depth", value);
                }
            }
        }
        [SerializeField]
        private Texture _depth;

        public Mesh InstancingMesh;
        public Material Material;

        [Tooltip("The nearest point that the minimum depth value is being mapped to (in meters).")]
        public float MinDepth = 2;

        [Tooltip("The furthest point that the maximum depth value is being mapped to (in meters).")]
        public float MaxDepth = 10;

        /// <summary>
        /// The minimum predicted depth value in the <see cref="Depth"/> texture.
        /// </summary>
        public float MinPred { get; set; }

        /// <summary>
        /// The maximum predicted depth value in the <see cref="Depth"/> texture.
        /// </summary>
        public float MaxPred { get; set; }

        private ComputeBuffer _argsBuffer;
        private Bounds _bounds = new Bounds(Vector3.zero, Vector3.one * int.MaxValue);

        private void Start() {
            _argsBuffer = CreateArgsBuffer(Depth.width * Depth.height);
        }

        private void Update() {
            Material.SetMatrix("_Transform", transform.localToWorldMatrix);
            Material.SetVector("_CameraWorldPos", Camera.main.transform.position);
            SetScaleShift();

            // Render points via instancing
            Graphics.DrawMeshInstancedIndirect(InstancingMesh, 0, Material, _bounds, _argsBuffer);
        }

        private ComputeBuffer CreateArgsBuffer(int instanceCount) {
            ComputeBuffer argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
            // Indirect args
            uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
            args[0] = InstancingMesh.GetIndexCount(0);
            args[1] = (uint)instanceCount;
            args[2] = InstancingMesh.GetIndexStart(0);
            args[3] = InstancingMesh.GetBaseVertex(0);
            argsBuffer.SetData(args);
            return argsBuffer;
        }

        private void SetScaleShift() {
            float invScale = (0.5f / MaxDepth - 0.5f / MinDepth) / (MinPred - MaxPred);
            float invShift = 0.5f / MinDepth - (invScale * MaxPred);
            Material.SetFloat("_Scale", invScale);
            Material.SetFloat("_Shift", invShift);
        }

        private void OnDestroy() {
            Dispose();
        }

        private void Dispose() {
            _argsBuffer?.Dispose();
        }

#if UNITY_EDITOR
        private void OnValidate() {
            
        }
#endif
    }
}